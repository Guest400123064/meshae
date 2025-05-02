from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from torch_geometric.nn.conv import SAGEConv
from vector_quantize_pytorch import ResidualVQ
from x_transformers import Encoder

from meshae.utils import quantize

MeshAEFeatNameType = Literal["area", "norm", "angle", "vertex"]


@dataclass
class MeshAEFeatEmbedConfig:
    r"""Configurations for face feature quantization.

    Parameters
    ----------
    embedding_dim : int, default=128
        Embedding (of the quantized features) dimension.
    num_bins : int, default=128
        Quantization resolution, discretize the ``high_low`` range into
        ``num_bins`` buckets.
    high_low : tuple, default=(0.5, -0.5)
        Range of the continuous inputs to be discretized.

    References
    ----------
    .. [1] `"PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance", Wang et al.
            <https://arxiv.org/html/2405.16890v1#S3>`_
    """
    embedding_dim: int = 128
    num_bins: int = 128
    high_low: tuple[float, float] = (0.5, -0.5)


class MeshAEEmbedding(nn.Module):
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

    PADDING_IDX = 0
    NUM_EXTRACTED_FEATURES = {"vertex": 9, "angle": 3, "norm": 3, "area": 1}

    def __init__(
        self,
        feature_configs: dict[MeshAEFeatNameType, MeshAEFeatEmbedConfig],
        num_sageconv_layers: int = 1,
        hidden_size: int = 512,
    ):
        super().__init__()

        self.input_size = 0
        self.embeddings = nn.ModuleDict({})
        self.feature_configs = feature_configs
        for name, cfg in feature_configs.items():
            self.input_size += cfg.embedding_dim * self.NUM_EXTRACTED_FEATURES[name]
            self.embeddings[name] = nn.Embedding(
                cfg.num_bins + 1,
                embedding_dim=cfg.embedding_dim,
                padding_idx=self.PADDING_IDX,
            )

        self.hidden_size = hidden_size
        self.proj_in = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.GELU(),
        )

        self.sageconv_in = None
        self.sageconv_activate = None
        self.sageconv_hidden = None
        self.sageconv_params = {"normalize": True, "project": True}
        self.num_sageconv_layers = num_sageconv_layers
        if num_sageconv_layers > 0:
            self.sageconv_in = SAGEConv(hidden_size, hidden_size, **self.sageconv_params)
            self.sageconv_activate = nn.Sequential(
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
            self.sageconv_hidden = nn.ModuleList([])
            for _ in range(num_sageconv_layers - 1):
                sageconv_layer = SAGEConv(hidden_size, hidden_size, **self.sageconv_params)
                self.sageconv_hidden.append(sageconv_layer)

    def forward(
        self,
        *,
        vertices: TensorType["b", "n_vertex", 3, float],
        faces: TensorType["b", "n_face", 3, int],
        edges: TensorType["b", "n_edge", 2, int],
        face_masks: TensorType["b", "n_face", bool],
        edge_masks: TensorType["b", "n_edge", bool],
    ) -> TensorType["b", "n_face", -1, float]:
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
        TensorType["b", "n_face", -1, float]
            Batch of face embedding sequences with all feature embeddings concatenated and
            passed through initial projection and, if any, ``SAGEConv`` layers.
        """
        coords = vertices[
            torch.arange(vertices.size(0), device=vertices.device)[:, None, None],
            faces.masked_fill(~face_masks.unsqueeze(-1), 0),  # Ensures no indexing error
        ]
        embeds = torch.cat(
            [
                self.embeddings[name](indices) for name, indices in
                self.extract_features(coords).items()
            ],
            dim=-1,
        )

        embeds = self.proj_in(embeds)
        if self.sageconv_in is not None:

            # In PyG implementation of the SAGEConv operator, a batch of graphs are treated
            # as a single disjoint graph. Nodes from different batches will be flattened to
            # a 2D tensor, i.e., a single list of node embeddings, where padding nodes will
            # be removed. Thus, the edge indices has to be shifted accordingly based on the
            # number of nodes (number of faces) in each graph from that batch.
            B, T, _ = embeds.size()

            offsets = F.pad(face_masks.long().sum(-1).cumsum(0), (1, -1), value=0)
            edges = (edges + offsets[:, None, None])[edge_masks].T

            embeds = self.sageconv_in(embeds[face_masks], edges)
            embeds = self.sageconv_activate(embeds)
            for conv in self.sageconv_hidden:
                embeds = conv(embeds, edges)

            embeds = (
                embeds.new_zeros((B, T, embeds.size(-1)))
                .masked_scatter(face_masks.unsqueeze(-1), embeds)
            )

        return embeds

    def extract_features(
        self, coords: TensorType["b", "n_face", 3, 3, float],
    ) -> dict[MeshAEFeatNameType, TensorType["b", "n_face", -1, int]]:
        r"""Extract and quantize features extracted from batch of faces.

        Parameters
        ----------
        coords : TensorType["b", "n_face", 3, 3, float]
            Batch of trimesh faces with each face represented as the 3 x-y-z
            coordinates of 3 vertices.

        Returns
        -------
        dict[MeshAEFeatNameType, TensorType["b", "n_face", -1, int]]
            A dictionary mapping from feature names to batches of extracted and
            quantized feature indices.
        """

        def _quantize(name, feat):
            r"""Handy short cut for quantization.

            Note that we shift the indices by 1 because index 0 is reserved for
            the padding index based on ``self.PADDING_IDX``.
            """
            cfg = self.feature_configs[name]
            ret = quantize(feat, high_low=cfg.high_low, num_bins=cfg.num_bins)

            return ret + 1

        shifts = torch.roll(coords, 1, dims=(2,))
        e1, e2, *_ = (coords - shifts).unbind(2)

        cross = torch.cross(e1, e2, dim=-1)
        feats = {
            "norm": _quantize("norm", F.normalize(cross, 2, dim=-1)),
            "area": _quantize("area", cross.norm(-1, keepdim=True) * 0.5),
            "vertex": _quantize("vertex", coords.flatten(-2)),
            "angle": _quantize(
                "angle", (
                    F.cosine_similarity(coords, shifts, dim=-1)
                    .clamp(1e-5 - 1, 1 - 1e-5)
                    .arccos()
                ),
            ),
        }
        return feats


class MeshAEEncoder(nn.Module):
    r""""""

    def __init__(
        self,
        feature_configs: dict[MeshAEFeatNameType, MeshAEFeatEmbedConfig],
        *,
        codebook_size: int = 256,
        hidden_size: int = 512,
        num_sageconv_layers: int = 1,
        num_encoder_layers: int = 12,
        num_encoder_heads: int = 8,
        num_quantizers: int = 3,
        num_codebook_codes: int = 4096,
    ):
        super().__init__()

        self.feature_configs = feature_configs
        self.num_sageconv_layers = num_sageconv_layers
        self.hidden_size = hidden_size
        self.embedding = MeshAEEmbedding(
            feature_configs, num_sageconv_layers, hidden_size,
        )

        self.num_encoder_layers = num_encoder_layers
        self.num_encoder_heads = num_encoder_heads
        self.encoder = Encoder(
            dim=hidden_size,
            depth=num_encoder_layers,
            heads=num_encoder_heads,
        )

        self.num_quantizers = num_quantizers
        self.num_codebook_codes = num_codebook_codes
        self.codebook_size = codebook_size
        self.quantizer = ResidualVQ(
            dim=hidden_size,
            num_quantizers=num_quantizers,
            codebook_size=num_codebook_codes,
            codebook_dim=codebook_size,
            shared_codebook=True,
        )

    def forward(self,):
        pass


class MeshAEDecoder(nn.Module):
    pass


class MeshAEModel(nn.Module):
    r""""""

    def __init__(
        self,
        feature_configs: dict[MeshAEFeatNameType, MeshAEFeatEmbedConfig],
        *,
        codebook_size: int = 256,
        hidden_size: int = 512,
        num_sageconv_layers: int = 1,
        num_encoder_layers: int = 12,
        num_encoder_heads: int = 8,
        num_quantizers: int = 3,
        num_codebook_codes: int = 4096,    
    ):
        super().__init__()

    def forward(self,) -> TensorType[(), float]:
        pass

    @torch.no_grad
    def encode(self,):
        pass

    @torch.no_grad
    def decode(self,):
        pass
