from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchtyping import TensorType
from torch_geometric.nn.conv import SAGEConv
from vector_quantize_pytorch import ResidualVQ
from x_transformers import Encoder

from meshae.utils import quantize

b = None
n_edge = None
n_face = None
n_vrtx = None

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

    NUM_EXTRACTED_FEATURES = {"vertex": 9, "angle": 3, "norm": 3, "area": 1}

    def __init__(
        self,
        *,
        feature_configs: dict[MeshAEFeatNameType, MeshAEFeatEmbedConfig],
        num_sageconv_layers: int = 1,
        hidden_size: int = 512,
    ) -> None:
        super().__init__()

        self.input_size = 0
        self.embeddings = nn.ModuleDict({})
        self.feature_configs = feature_configs
        for name, cfg in feature_configs.items():
            self.input_size += cfg.embedding_dim * self.NUM_EXTRACTED_FEATURES[name]
            self.embeddings[name] = nn.Embedding(
                cfg.num_bins + 1, embedding_dim=cfg.embedding_dim,
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
        coords: TensorType["b", "n_face", 3, 3, float],
        face_masks: TensorType["b", "n_face", bool],
        edges: TensorType["b", "n_edge", 2, int],
        edge_masks: TensorType["b", "n_edge", bool],
    ) -> TensorType["b", "n_face", -1, float]:
        r"""Create face embeddings from a batch of meshes.

        Parameters
        ----------
        coords : TensorType["b", "n_face", 3, 3, float]
            Batch of trimesh faces with each face represented as the 3 x-y-z
            coordinates of 3 vertices.
        face_masks : TensorType["b", "n_face", bool]
            Boolean masks used to separate actual faces from paddings. Actual faces have
            corresponding face mask values being 1.
        edges : TensorType["b", "n_edge", 2, int]
            Batch of **face** edges used to identify the topological structure between
            faces. This is only used in ``SAGEConv`` layers.
        edge_masks : TensorType["b", "n_edge", bool]
            Boolean masks used to separate actual faces from paddings.

        Returns
        -------
        embeds : TensorType["b", "n_face", -1, float]
            Batch of face embedding sequences with all feature embeddings concatenated and
            passed through initial projection and, if any, ``SAGEConv`` layers.
        """
        embeds = torch.cat(
            [
                self.embeddings[name](indices) for name, indices in
                self.extract_features(coords).items()
            ],
            dim=-1,
        )
        embeds = self.proj_in(embeds)

        # In PyG implementation of the SAGEConv operator, a batch of graphs are treated
        # as a single disjoint graph. Nodes from different batches will be flattened to
        # a 2D tensor, i.e., a single list of node embeddings, where padding nodes will
        # be removed. Thus, the edge indices has to be shifted accordingly based on the
        # number of nodes (number of faces) in each graph from that batch.
        if self.sageconv_in is not None:
            B, T, _ = embeds.size()

            offsets = F.pad(face_masks.long().sum(-1).cumsum(0), (1, -1), value=0)
            edges = (edges + offsets[:, None, None])[edge_masks].T

            embeds = self.sageconv_in(embeds[face_masks], edges)
            embeds = self.sageconv_activate(embeds)
            for conv in self.sageconv_hidden:
                embeds = conv(embeds, edges)

            embeds = (
                embeds.new_empty((B, T, embeds.size(-1)))
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
        feats : dict[MeshAEFeatNameType, TensorType["b", "n_face", -1, int]]
            A dictionary mapping from feature names to batches of extracted and
            quantized feature indices.
        """

        def _quantize(name, feat):
            cfg = self.feature_configs[name]
            return quantize(feat, high_low=cfg.high_low, num_bins=cfg.num_bins)

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
    r"""Compute latents and corresponding face codes for input mesh.

    Parameters
    ----------
    codebook_size : int, default=256
        Codebook code (latent) dimension.
    hidden_size : int, default=512
        Hidden state dimension.
    num_sageconv_layers : int, default=1
        Number of ``SAGEConv`` layers to capture face topological structures before sending
        the face embeddings to VQ-VAE encoder.
    num_quantizers : int, default=2
        The number of codebook embeddings used to approximate the input embedding. If
        set to 1, the RQ-VAE reduces to the regular VQ-VAE. 
    num_codebook_codes : int, default=4096
        Number of unique codebook codes.
    commitment_weight : float, default=1.0
        The weighting parameter for the VQ-VAE commitment loss term.
    """

    def __init__(
        self,
        *,
        codebook_size: int = 256,
        hidden_size: int = 512,
        num_encoder_layers: int = 6,
        num_encoder_heads: int = 8,
        num_quantizers: int = 2,
        num_codebook_codes: int = 4096,
        commitment_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_encoder_heads = num_encoder_heads
        self.encoder = Encoder(
            dim=hidden_size,
            depth=num_encoder_layers,
            heads=num_encoder_heads,
            pre_norm=True,
        )

        self.num_quantizers = num_quantizers
        self.num_codebook_codes = num_codebook_codes
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.proj_vertex = nn.Linear(hidden_size, codebook_size * 3)
        self.proj_latent = nn.Linear(codebook_size * 3, hidden_size)
        self.quantizer = ResidualVQ(
            dim=codebook_size,
            num_quantizers=num_quantizers,
            codebook_size=num_codebook_codes,
            codebook_dim=codebook_size,
            shared_codebook=True,
            rotation_trick=True,
            commitment_weight=commitment_weight,
        )

    def forward(
        self,
        faces: TensorType["b", "n_face", 3, int],
        face_embeds: TensorType["b", "n_face", -1, float],
        face_masks: TensorType["b", "n_face", bool],
    ) -> tuple[
        TensorType["b", "n_face", -1, float],
        TensorType["b", "n_face", -1, int],
        TensorType[(), float],
    ]:
        r"""Create face embeddings from a batch of meshes.

        To facilitate the decoding process, the PivotMesh paper [1]_ mentioned an operation
        introduced in the MeshGPT paper [2]_ to ensure consistent embeddings between shared
        vertices. Specifically, face embeddings are projected to vertex embeddings and then
        embeddings for shared vertices (from different faces) are substituted with average-
        pooling embeddings.

        Parameters
        ----------
        faces : TensorType["b", "n_face", 3, int]
            Batch of mesh face sequences with each face represented by three vertex ids.
        face_embeds : TensorType["b", "n_face", -1, float]
            Batch of mesh face embedding sequences. The embedding size should be equal
            to ``self.hidden_size``.
        face_masks : TensorType["b", "n_face", bool]
            Boolean masks used to separate actual faces from paddings. Actual faces have
            corresponding face mask values being 1.

        Returns
        -------
        face_embeds : TensorType["b", "n_face", -1, float]
            Batch of face embedding sequences created from quantized vertex latents.
        face_codes : TensorType["b", "n_face", -1, int]
            Batch of face codebook code sequences. The number of code depends on the
            number of quantizers in the RQ-VAE.
        commit_loss : TensorType[(), float]
            The commit loss used to update the VQ-VAE codebook codes.

        References
        ----------
        .. [1] `"PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance", Wang et al.
            <https://arxiv.org/html/2405.16890v1#S3>`_
        .. [2] `"MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers", Siddiqui et al.
                <https://arxiv.org/abs/2311.15475>`_
        """
        face_embeds = self.encoder(face_embeds, mask=face_masks)
        face_embeds = self.proj_vertex(face_embeds)

        B, T, D = face_embeds.size()

        # To achieve the vertex embedding aggregation operation we first create a vertex embedding
        # memory bank of size N x E, where N is number of unique, non-padding vertex ids of meshes
        # within the batch, and E is the vertex embedding dim. In this implementation, E should be
        # equal to the codebook code dim.
        #
        # Then, based on the number of unique ids if each individual mesh, we compute an offset to
        # reindex the vertices from different mesh objects to avoid overlap. In turn, the batch of
        # faces can be flattened into a single sequence and drop paddings. Lastly we use scattered
        # reduce to perform the average pooling operation.
        #
        # Essentially, it is the vertex embeddings going through the quantizer not those of faces.
        #
        # NOTE: Padding indices should be negative.
        vrtx_counts = faces.flatten(-2).amax(-1).cumsum(0).int() + 1
        vrtx_embeds = face_embeds.new_empty((vrtx_counts.max().item(), D // 3))

        offsets = F.pad(vrtx_counts, (1, -1), value=0)
        indices = (faces + offsets[:, None, None])[face_masks].flatten()

        # A few sanity checks to verify `indices` validity
        assert indices.max().item() + 1 == vrtx_embeds.size(-1)
        assert indices.size(-1) == face_masks.sum().item()
        if B > 1:
            num_face_first_mesh = face_masks[0].sum().int().item()
            num_vrtx_first_mesh = vrtx_counts[0].item()
            assert indices[num_face_first_mesh * 3:].min().item() == num_vrtx_first_mesh

        vrtx_embeds, vrtx_codes, commit_loss = self.quantizer(
            vrtx_embeds.scatter_reduce_(
                0, indices.unsqueeze(-1).expand(-1, D // 3),
                src=rearrange(face_embeds[face_masks], "t (v e) -> (t v) e", v=3),
                reduce="mean",
                include_self=False,
            ),
            return_all_codes=False,
        )
        face_embeds = self.proj_latent(
            face_embeds.new_empty((B, T, D))
            .masked_scatter(
                face_masks.unsqueeze(-1),
                rearrange(vrtx_embeds[indices], "(t v) e -> t (v e)", v=3),
            ),
        )
        face_codes = (
            faces.new_empty((B, T, self.num_quantizers * 3))
            .masked_scatter(
                face_masks.unsqueeze(-1),
                rearrange(vrtx_codes[indices], "(t v) q -> t (v q)", v=3),
            )
        )
        return face_embeds, face_codes, commit_loss.sum()


class MeshAEDecoder(nn.Module):
    r"""

    Parameters
    ----------
    """

    def __init__(
        self,
        *,
        hidden_size: int = 512,
        num_decoder_layers: int = 6,
        num_decoder_heads: int = 8,
        num_refiner_layers: int = 6,
        num_refiner_heads: int = 8,
        coord_num_bins: int = 128,
    ):
        super().__init__()

        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_heads = num_decoder_heads
        self.decoder = Encoder(
            dim=hidden_size,
            depth=num_decoder_layers,
            heads=num_decoder_heads,
            pre_norm=True,
        )

        self.num_refiner_layers = num_refiner_layers
        self.num_refiner_heads = num_refiner_heads
        self.proj_refine = nn.Sequential(
            nn.Linear(hidden_size, 3 * hidden_size),
            Rearrange("b t (v d) -> b (t v) d", v=3, d=hidden_size),
        )
        self.refiner = Encoder(
            dim=hidden_size,
            depth=num_refiner_layers,
            heads=num_refiner_heads,
            pre_norm=True,
        )

        self.coord_num_bins = coord_num_bins
        self.proj_logit = nn.Sequential(
            nn.Linear(hidden_size, 3 * coord_num_bins),
            Rearrange("b (t v) (c q) -> b t (v c) q", v=3, c=3, q=coord_num_bins),
        )

    def forward(
        self,
        face_embeds: TensorType["b", "n_face", -1, float],
        face_masks: TensorType["b", "n_face", bool],
    ) -> TensorType["b", "n_face", 9, -1, float]:
        r"""Decode from quantized face embeddings into vertex logits.

        Parameters
        ----------
        face_embeds : TensorType["b", "n_face", -1, float]
            Batch of face embedding sequences created from quantized vertex latents.
        face_masks : TensorType["b", "n_face", bool]
            Boolean masks used to separate actual faces from paddings. Actual faces have
            corresponding face mask values being 1.

        Returns
        -------
        logits : TensorType["b", "n_face", 9, -1, float]
            Predicted face vertex logits.
        """
        face_embeds = self.decoder(face_embeds, mask=face_masks)
        face_embeds = self.proj_refine(face_embeds)
        face_embeds = self.refiner(
            face_embeds,
            mask=repeat(face_masks, "b t -> b (t v)", v=3),
        )
        return self.proj_logit(face_embeds)


class MeshAEModel(nn.Module):
    r"""

    Parameters
    ----------
    """

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
        num_decoder_layers: int = 12,
        num_decoder_heads: int = 8,
        num_refiner_layers: int = 6,
        num_refiner_heads: int = 8,
        coord_num_bins: int = 128,
        commitment_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.feature_configs = feature_configs
        self.hidden_size = hidden_size
        self.num_sageconv_layers = num_sageconv_layers
        self.embedding = MeshAEEmbedding(
            feature_configs=feature_configs,
            num_sageconv_layers=num_sageconv_layers,
            hidden_size=hidden_size,
        )

        self.codebook_size = codebook_size
        self.num_encoder_layers = num_encoder_layers
        self.num_encoder_heads = num_encoder_heads
        self.num_quantizers = num_quantizers
        self.num_codebook_codes = num_codebook_codes
        self.commitment_weight = commitment_weight
        self.encoder = MeshAEEncoder(
            codebook_size=codebook_size,
            hidden_size=hidden_size,
            num_encoder_layers=num_encoder_layers,
            num_encoder_heads=num_encoder_heads,
            num_quantizers=num_quantizers,
            num_codebook_codes=num_codebook_codes,
            commitment_weight=commitment_weight,
        )

        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_heads = num_decoder_heads
        self.num_refiner_layers = num_refiner_layers
        self.num_refiner_heads = num_refiner_heads
        self.coord_num_bins = coord_num_bins
        self.decoder = MeshAEDecoder(
            hidden_size=hidden_size,
            num_decoder_layers=num_decoder_layers,
            num_decoder_heads=num_decoder_heads,
            num_refiner_layers=num_refiner_layers,
            num_refiner_heads=num_refiner_heads,
            coord_num_bins=coord_num_bins,
        )

    def forward(
        self,
        *,
        vertices: TensorType["b", "n_vrtx", 3, float],
        faces: TensorType["b", "n_face", 3, int],
        edges: TensorType["b", "n_edge", 2, int],
        face_masks: TensorType["b", "n_face", bool],
        edge_masks: TensorType["b", "n_edge", bool],
    ) -> TensorType[(), float]:
        r"""

        Parameters
        ----------
        vertices : TensorType["b", "n_vrtx", 3, float]
            Batch of mesh vertex sets with each vertex represented by x-y-z coordinates.
        faces : TensorType["b", "n_face", 3, int]
            Batch of mesh face sequences with each face represented by three vertex ids.
        edges : TensorType["b", "n_edge", 2, int]
            Batch of **face** edges used to identify the topological structure between
            faces. This is only used in ``SAGEConv`` layers.
        face_masks : TensorType["b", "n_face", bool]
            Boolean masks used to separate actual faces from paddings. Actual faces have
            corresponding face mask values being 1.
        edge_masks : TensorType["b", "n_edge", bool]
            Boolean masks used to separate actual faces from paddings.
        """
        faces = faces.masked_fill(~face_masks.unsqueeze(-1), 0)
        batches = torch.arange(vertices.size(0), device=vertices.device)

        coords = vertices[batches[:, None, None], faces]        
        face_embeds = self.embedding(coords, face_masks, edges, edge_masks)
        face_embeds, _, commit_loss = self.encoder(faces, face_embeds, face_masks)

        logits = self.decoder(face_embeds, face_masks)


        return logits, commit_loss

    @torch.no_grad
    def encode(self,):
        pass

    @torch.no_grad
    def decode(self,):
        pass
