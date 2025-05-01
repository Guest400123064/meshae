from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange, repeat, pack
from torchtyping import TensorType
from torch_geometric.nn.conv import SAGEConv

from meshae.utils import quantize

MeshAEFeatureNameType = Literal["area", "norm", "angle", "vertex"]


@dataclass
class MeshAEFeatureConfig:
    r"""Configurations for face feature quantization.

    Parameters
    ----------
    name : MeshAEFeatureNameType
        Name of the features mentioned in the PivotMesh paper [1]_. Available
        options are ``"area", "norm", "angle", "vertex"``.
    embedding_dim : int, default=128
        Embedding (of the quantized features) dimension.
    num_quant_bins : int, default=128
        Quantization resolution, discretize the ``quant_high_low`` range into
        ``num_quant_bins`` buckets.
    quant_high_low : tuple, default=(0.5, -0.5)
        Range of the continuous inputs to be discretized.
    num_extracted_features: int, default=9
        The number of features extracted from each face. For instance, ``area``
        is considered 1 feature because it is a scalar; ``vertex`` is considered
        9 features for trimesh because there are 3 vertices in each face, with
        each face having 3 x-y-z dimensions.

    References
    ----------
    .. [1] `"PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance", Wang et al.
            <https://arxiv.org/html/2405.16890v1#S3>`_
    """
    name: MeshAEFeatureNameType
    embedding_dim: int = 128
    num_quant_bins: int = 128
    quant_high_low: tuple[float, float] = (0.5, -0.5)
    num_extracted_features: int = 9


class MeshAEFaceEmbedding(nn.Module):
    r"""Create embeddings from extracted and quantized face features.

    The PivotMesh paper [1]_ did not mention the details about the exact compositions of
    face embeddings. In their source code [2]_, one can see that the face embeddings are
    created from four components, following MeshGPT [3]_ [4]_:

    - 9 vertex coordinates per trimesh face.
    - 3 arccosine values (0 to pi) per trimesh face.
    - 3 face normal vector coordinates per trimesh face.
    - 1 area per trimesh face.

    All coordinates and features are quantized according to pre-defined resolution and
    then passed through their individual embedding layers. Embeddings are concatenated
    together to form the face embedding.

    Parameters
    ----------
    feature_configs : list[MeshAEFeatureConfig]
        Configurations for how to quantize and embed features extracted from faces. Please
        refer to ``MeshAEFeatureConfig`` for more details.
    num_sageconv_layers : int, default=1
        Number of ``SAGEConv`` layers to capture face topological structures before sending
        the face embeddings to VQ-VAE encoder.
    hidden_size : int, default=512
        Hidden state dimension.

    References
    ----------
    .. [1] `"PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance", Wang et al.
            <https://arxiv.org/html/2405.16890v1#S3>`_
    .. [2] `"PivotMesh GitHub Repository: 'meshAE.py'", Wang et al.
            <https://github.com/whaohan/pivotmesh/blob/ed5652d7584e631cd400fc44e8fee285289e1482/model/meshAE.py#L252>`_
    .. [3] `"MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers", Siddiqui et al.
            <https://arxiv.org/abs/2311.15475>`_
    .. [4] `"Implementation of MeshGPT", @lucidrains et al.
            <https://github.com/lucidrains/meshgpt-pytorch/blob/672d921d733ea5f05f5d0724efbbf2ab88440981/meshgpt_pytorch/meshgpt_pytorch.py#L157>`_
    """

    def __init__(
        self,
        feature_configs: list[MeshAEFeatureConfig],
        num_sageconv_layers: int = 1,
        hidden_size: int = 512,
    ):
        super().__init__()

        self.input_size = 0
        self.feature_configs = feature_configs
        for cfg in feature_configs:
            embed = nn.Embedding(
                cfg.num_quant_bins + 1,
                embedding_dim=cfg.embedding_dim,
                padding_idx=0,
            )
            self.add_module(f"embed_{cfg.name}", embed)
            self.input_size += cfg.embedding_dim * cfg.num_extracted_features

        self.hidden_size = hidden_size
        self.proj_in = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.GELU(),
        )

        self.sageconv_in = None
        self.sageconv_act = None
        self.sageconv_hidden = None
        self.sageconv_params = {"normalize": True, "project": True}
        self.num_sageconv_layers = num_sageconv_layers
        if num_sageconv_layers > 0:
            self.sageconv_in = SAGEConv(hidden_size, hidden_size, **self.sageconv_params)
            self.sageconv_act = nn.Sequential(
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
            self.sageconv_hidden = nn.ModuleList([])
            for _ in range(num_sageconv_layers - 1):
                sageconv_layer = SAGEConv(hidden_size, hidden_size, **self.sageconv_params)
                self.sageconv_hidden.append(sageconv_layer)

    @beartype
    def forward(
        self,
        *,
        vertices: TensorType["b", "n_vertex", 3, float],
        faces: TensorType["b", "n_face", 3, int],
        edges: TensorType["b", "n_edge", 2, int],
        face_masks: TensorType["b", "n_face", bool],
        edge_masks: TensorType["b", "n_edge", bool],
    ) -> TensorType["b", "n_face", "n_feat", float]:
        r"""Create face embeddings from a batch of meshes.

        Parameters
        ----------
        vertices : TensorType["b", "n_vertex", 3, float]
            Batch of mesh vertex sets with each vertex represented by x-y-z coordinates.
        faces : TensorType["b", "n_face", 3, int]
            Batch of mesh face sequences with each face represented by three vertex IDs.
        edges : TensorType["b", "n_edge", 2, int]
            Batch of **face** edges used to identify the topological structure between
            faces. This is only used in ``SAGEConv`` layers.
        face_masks : TensorType["b", "n_face", bool]
            Boolean masks used to separate actual faces from paddings.
        edge_masks : TensorType["b", "n_edge", bool]
            Boolean masks used to separate actual faces from paddings.

        Returns
        -------
        TensorType["b", "n_face", "d_feat", float]
            Batch of face embedding sequences with all feature embeddings concatenated.
            In other words, ``d_feat = self.input_size``.
        """
        # features = quantize(features, high_low=self.high_low, num_bins=self.num_bins)
        # return rearrange(
        #     self.embedding(features), "b n_face n_feat d -> b n_face (n_feat d)",
        # )

    @beartype
    def extract_features(
        self, face_coordinates: TensorType["b", "n_face", 3, 3, float],
    ) -> dict[MeshAEFeatureNameType, TensorType[..., float]]:
        r"""Extract and quantize """
        pass
