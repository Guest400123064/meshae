from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch_geometric.nn.conv import SAGEConv
from vector_quantize_pytorch import ResidualVQ
from x_transformers import Encoder

from meshae.config import MeshAEFeatEmbedConfig
from meshae.typing import MeshAEFeatNameType
from meshae.utils import dequantize, gaussian_blur1d, quantize

if TYPE_CHECKING:
    from torchtyping import TensorType

    from meshae.typing import b, n_edge, n_face, n_vrtx


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

    NUM_EXTRACTED_FEATURES = {"vrtx": 9, "acos": 3, "norm": 3, "area": 1}

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
                cfg.num_bins,
                embedding_dim=cfg.embedding_dim,
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
            self.sageconv_in = SAGEConv(
                hidden_size, hidden_size, **self.sageconv_params
            )
            self.sageconv_activate = nn.Sequential(
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
            self.sageconv_hidden = nn.ModuleList([])
            for _ in range(num_sageconv_layers - 1):
                sageconv_layer = SAGEConv(
                    hidden_size, hidden_size, **self.sageconv_params
                )
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
            Boolean masks used to separate actual edges from paddings.

        Returns
        -------
        embeds : TensorType["b", "n_face", -1, float]
            Batch of face embedding sequences with all feature embeddings concatenated and
            passed through initial projection and, if any, ``SAGEConv`` layers.
        """
        embeds = torch.cat(
            [
                self.embeddings[name](indices).flatten(-2)
                for name, indices in self.extract_features(coords).items()
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

            embeds = embeds.new_empty((B, T, embeds.size(-1))).masked_scatter(
                face_masks.unsqueeze(-1), embeds
            )

        return embeds

    def extract_features(
        self,
        coords: TensorType["b", "n_face", 3, 3, float],
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
            "area": _quantize("area", cross.norm(dim=-1, keepdim=True) * 0.5),
            "vrtx": _quantize("vrtx", coords.flatten(-2)),
            "acos": _quantize(
                "acos",
                (
                    F.cosine_similarity(coords, shifts, dim=-1)
                    .clamp(1e-5 - 1, 1 - 1e-5)
                    .arccos()
                ),
            ),
        }
        return feats


class MeshAEEncoder(nn.Module):
    r"""Compute latents and corresponding face codes for input mesh.

    Encoder module incorporates a transformer encoder and a vector quantizer. Face
    embeddings from the embedding layer are passed through the network to get quantized
    into latents and commitment loss (including both ``z`` to ``e`` and ``e`` to ``z``)
    is returned as well.

    To facilitate the decoding process, the PivotMesh paper [1]_ mentioned an operation
    introduced in the MeshGPT paper [2]_ to ensure consistent embeddings between shared
    vertices. Specifically, face embeddings are projected to vertex embeddings and then
    embeddings for shared vertices (from different faces) are substituted with average-
    pooling embeddings. Please refer to the ``self.forward`` method for more details

    Parameters
    ----------
    codebook_size : int, default=256
        Codebook code (latent) dimension.
    hidden_size : int, default=512
        Hidden state dimension.
    num_encoder_layers : int, default=6
        Number of encoder transformer layers.
    num_encoder_heads : int, default=8
        Number of encoder transformer heads.
    num_quantizers : int, default=2
        The number of codebook embeddings used to approximate the input embedding. If
        set to 1, the RQ-VAE reduces to the regular VQ-VAE.
    num_codebook_codes : int, default=4096
        Number of unique codebook codes.
    sample_codebook_temp : float, default=0.1
        The codebook sampling temperature parameter for RQ-VAE. Setting to 0.0 is
        equivalent to deterministic code retrieval [3]_.
    commitment_weight : float, default=1.0
        The weighting parameter for the VQ-VAE commitment loss term.

    References
    ----------
    .. [1] `"PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance", Wang et al.
            <https://arxiv.org/html/2405.16890v1#S3>`_
    .. [2] `"MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers", Siddiqui et al.
            <https://arxiv.org/abs/2311.15475>`_
    .. [3] `"Autoregressive Image Generation using Residual Quantization", Lee et al.
            <https://arxiv.org/abs/2203.01941>`_
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
        sample_codebook_temp: float = 0.1,
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
        self.sample_codebook_temp = sample_codebook_temp
        self.commitment_weight = commitment_weight
        self.proj_vertex = nn.Linear(hidden_size, codebook_size * 3)
        self.proj_latent = nn.Linear(codebook_size * 3, hidden_size)
        self.quantizer = ResidualVQ(
            dim=codebook_size,
            num_quantizers=num_quantizers,
            codebook_size=num_codebook_codes,
            codebook_dim=codebook_size,
            shared_codebook=True,
            rotation_trick=False,
            stochastic_sample_codes=False,
            sample_codebook_temp=sample_codebook_temp,
            commitment_weight=commitment_weight,
            ema_update=True,
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
            number of quantizers in the RQ-VAE. Specifically, it will be ``3 * n_quant``.
        commit_loss : TensorType[(), float]
            The commit loss used to update the VQ-VAE codebook codes.
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
        # NOTE: Padding index should be NOT exceed the maximum number of vertices in each mesh. It
        # could be 0 since we will mask out the paddings anyway.
        vrtx_counts = faces.amax((1, 2)).int() + 1
        vrtx_embeds = face_embeds.new_empty((vrtx_counts.sum().item(), D // 3))

        offsets = F.pad(vrtx_counts.cumsum(0), (1, -1), value=0)
        indices = (faces + offsets[:, None, None])[face_masks].flatten()

        # A few sanity checks to verify `indices` validity
        assert indices.max().item() + 1 == vrtx_embeds.size(0)
        assert indices.size(-1) == face_masks.sum().item() * 3
        if B > 1:
            num_face_first_mesh = face_masks[0].sum().int().item()
            num_vrtx_first_mesh = vrtx_counts[0].item()
            assert (
                indices[num_face_first_mesh * 3 :].min().item() == num_vrtx_first_mesh
            )

        vrtx_embeds, vrtx_codes, commit_loss = self.quantizer(
            vrtx_embeds.scatter_reduce_(
                0,
                indices.unsqueeze(-1).expand(-1, D // 3),
                src=rearrange(face_embeds[face_masks], "t (v e) -> (t v) e", v=3),
                reduce="mean",
                include_self=False,
            ),
            return_all_codes=False,
        )
        face_embeds = self.proj_latent(
            face_embeds.new_empty((B, T, D), dtype=vrtx_embeds.dtype).masked_scatter(
                face_masks.unsqueeze(-1),
                rearrange(vrtx_embeds[indices], "(t v) e -> t (v e)", v=3),
            ),
        )
        face_codes = faces.new_empty((B, T, self.num_quantizers * 3)).masked_scatter(
            face_masks.unsqueeze(-1),
            rearrange(vrtx_codes[indices], "(t v) q -> t (v q)", v=3),
        )
        return face_embeds, face_codes, commit_loss.sum()


class MeshAEDecoder(nn.Module):
    r"""Implements the PivotMesh coarse-to-fine decoder [1]_.

    Parameters
    ----------
    hidden_size : int, default=512
        Hidden state dimension for both coarse decoder and fine decoder. In the PivotMesh
        implementation [2]_, the hidden size for two decoders can be different.
    num_decoder_layers : int, default=4
        Number of coarse decoder transformer layers.
    num_decoder_heads : int, default=8
        Number of coarse decoder transformer heads.
    num_refiner_layers : int, default=2
        Number of refiner decoder transformer layers. The refiner transforms from face latents
        to vertex latents.
    num_refiner_heads : int, default=8
        Number of refiner decoder transformer heads. The refiner transforms from face latents
        to vertex latents.
    coord_num_bins : int, default=128
        Number of quantization bins for coordinates.

    References
    ----------
    .. [1] `"PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance", Wang et al.
            <https://arxiv.org/html/2405.16890v1#S3>`_
    .. [2] `"PivotMesh GitHub Repository: 'meshAE.py'", Wang et al.
            <https://github.com/whaohan/pivotmesh/blob/ed5652d7584e631cd400fc44e8fee285289e1482/model/meshAE.py#L199>`_
    """

    def __init__(
        self,
        *,
        hidden_size: int = 512,
        num_decoder_layers: int = 4,
        num_decoder_heads: int = 8,
        num_refiner_layers: int = 2,
        num_refiner_heads: int = 8,
        coord_num_bins: int = 128,
    ) -> None:
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
            nn.GELU(),
            nn.Linear(3 * hidden_size, 3 * hidden_size),
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
    r"""PivotMesh mesh tokenizer [1]_.

    Parameters
    ----------
    feature_configs : dict[MeshAEFeatNameType, MeshAEFeatureConfig]
        Configurations for how to quantize and embed features extracted from faces. Please
        refer to ``MeshAEFeatureConfig`` for more details.
    codebook_size : int, default=256
        Codebook code (latent) dimension.
    hidden_size : int, default=512
        Hidden state dimension.
    num_encoder_layers : int, default=6
        Number of encoder transformer layers.
    num_encoder_heads : int, default=8
        Number of encoder transformer heads.
    num_quantizers : int, default=2
        The number of codebook embeddings used to approximate the input embedding. If
        set to 1, the RQ-VAE reduces to the regular VQ-VAE.
    num_codebook_codes : int, default=4096
        Number of unique codebook codes.
    num_decoder_layers : int, default=4
        Number of coarse decoder transformer layers.
    num_decoder_heads : int, default=8
        Number of coarse decoder transformer heads.
    num_refiner_layers : int, default=2
        Number of refiner decoder transformer layers. The refiner transforms from face latents
        to vertex latents.
    num_refiner_heads : int, default=8
        Number of refiner decoder transformer heads. The refiner transforms from face latents
        to vertex latents.
    sample_codebook_temp : float, default=0.1
        The codebook sampling temperature parameter for RQ-VAE. Setting to 0.0 is
        equivalent to deterministic code retrieval [3]_.
    commitment_weight : float, default=1.0
        The weighting parameter for the VQ-VAE commitment loss term.
    bin_smooth_blur_sigma : float, default=0.4
        Gaussian blur sigma parameter used to smooth the quantized, one-hot encoded coordinates
        for reconstruction loss calculation [2]_.

    References
    ----------
    .. [1] `"PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance", Wang et al.
            <https://arxiv.org/html/2405.16890v1#S3>`_
    .. [2] `"MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers", Siddiqui et al.
            <https://arxiv.org/abs/2311.15475>`_
    .. [3] `"Autoregressive Image Generation using Residual Quantization", Lee et al.
            <https://arxiv.org/abs/2203.01941>`_
    """

    @beartype
    def __init__(
        self,
        feature_configs: dict[MeshAEFeatNameType, MeshAEFeatEmbedConfig],
        *,
        codebook_size: int = 256,
        hidden_size: int = 512,
        num_sageconv_layers: int = 1,
        num_encoder_layers: int = 12,
        num_encoder_heads: int = 8,
        num_quantizers: int = 2,
        num_codebook_codes: int = 4096,
        num_decoder_layers: int = 12,
        num_decoder_heads: int = 8,
        num_refiner_layers: int = 6,
        num_refiner_heads: int = 8,
        sample_codebook_temp: float = 0.1,
        commitment_weight: float = 1.0,
        bin_smooth_blur_sigma: float = 0.0,
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
        self.sample_codebook_temp = sample_codebook_temp
        self.commitment_weight = commitment_weight
        self.encoder = MeshAEEncoder(
            codebook_size=codebook_size,
            hidden_size=hidden_size,
            num_encoder_layers=num_encoder_layers,
            num_encoder_heads=num_encoder_heads,
            num_quantizers=num_quantizers,
            num_codebook_codes=num_codebook_codes,
            sample_codebook_temp=sample_codebook_temp,
            commitment_weight=commitment_weight,
        )

        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_heads = num_decoder_heads
        self.num_refiner_layers = num_refiner_layers
        self.num_refiner_heads = num_refiner_heads
        self.decoder = MeshAEDecoder(
            hidden_size=hidden_size,
            num_decoder_layers=num_decoder_layers,
            num_decoder_heads=num_decoder_heads,
            num_refiner_layers=num_refiner_layers,
            num_refiner_heads=num_refiner_heads,
            coord_num_bins=feature_configs["vrtx"].num_bins,
        )

        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

        self.apply(self.init_parameters)

    def init_parameters(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        *,
        vertices: TensorType["b", "n_vrtx", 3, float],
        faces: TensorType["b", "n_face", 3, int],
        edges: TensorType["b", "n_edge", 2, int],
        face_masks: TensorType["b", "n_face", bool],
        edge_masks: TensorType["b", "n_edge", bool],
    ) -> tuple[
        TensorType[(), float],
        tuple[TensorType[(), float], TensorType[(), float]],
        TensorType["b", "n_face", 9, -1, float],
        TensorType["b", "n_face", 3, 3, float],
    ]:
        r"""Full encode-decode path and return VQ-VAE losses for model training.

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
            Boolean masks used to separate actual edges from paddings.

        Returns
        -------
        loss : TensorType[(), float]
            Combined loss for auto-encoder training.
        loss_breakdown : tuple[TensorType[(), float], TensorType[(), float]]
            A tuple consisting of the reconstruction loss and commitment loss.
        logits : TensorType["b", "n_face", 9, -1, float]
            Predicted face vertex logits. Vertex and coordinate dimensions are flattened
            into a single dimension. The size of the last dimension depending on the
            coordinate quantization settings.
        coords : TensorType["b", "n_face", 3, 3, float]
            Batch of trimesh faces with each face represented as the 3 x-y-z
            coordinates of 3 vertices.
        """
        faces = faces.masked_fill(~face_masks.unsqueeze(-1), 0)
        index = torch.arange(vertices.size(0), device=vertices.device)[:, None, None]

        coords = vertices[index, faces]
        embeds = self.embedding(coords, face_masks, edges, edge_masks)
        embeds, _, commit_loss = self.encoder(faces, embeds, face_masks)
        logits = self.decoder(embeds, face_masks)

        with torch.autocast(coords.device.type, enabled=False):
            recon_loss = self.compute_recon_loss(coords.flatten(-2), logits, face_masks)

        return (recon_loss + commit_loss), (recon_loss, commit_loss), logits, coords

    def compute_recon_loss(
        self,
        coords: TensorType["b", "n_face", 9, float],
        logits: TensorType["b", "n_face", 9, -1, float],
        face_masks: TensorType["b", "n_face", bool],
    ) -> TensorType[(), float]:
        r"""Compute reconstruction loss.

        Optionally, one can apply Gaussian blur to the one-hot quantized coordinates.
        This is controlled by the ``self.bin_smooth_blur_sigma`` value. Setting to 0.0
        results in regular one-hot encoding.

        Parameters
        ----------
        coords : TensorType["b", "n_face", 9, float]
            Raw face coordinates without quantization, having the vertex and coordinate
            dimensions flattened.
        logits : TensorType["b", "n_face", 9, -1, float]
            Predicted logits from decoder.
        face_masks : TensorType["b", "n_face", bool]
            Boolean masks used to separate actual faces from paddings. Actual faces have
            corresponding face mask values being 1.

        Returns
        -------
        recon_loss : TensorType[(), float]
            Reconstruction loss.
        """
        logits = rearrange(logits, "b ... q -> b q (...)").log_softmax(1)
        face_masks = repeat(face_masks, "b t -> b (t r)", r=9)

        config = self.feature_configs["vrtx"]
        coords = rearrange(
            quantize(coords, high_low=config.high_low, num_bins=config.num_bins),
            "b ... -> b 1 (...)",
        )
        coords = torch.zeros_like(logits).scatter(1, coords, 1.0)
        if self.bin_smooth_blur_sigma > 0.0:
            coords = gaussian_blur1d(coords, sigma=self.bin_smooth_blur_sigma)

        assert coords.size() == logits.size(), "Coords and logits have different shape."

        recon_loss = (-coords * logits).sum(1)[face_masks].mean()
        return recon_loss

    @torch.no_grad()
    def encode(
        self,
        vertices: TensorType["b", "n_vrtx", 3, float],
        faces: TensorType["b", "n_face", 3, int],
        edges: TensorType["b", "n_edge", 2, int],
        face_masks: TensorType["b", "n_face", bool],
        edge_masks: TensorType["b", "n_edge", bool],
    ) -> TensorType["b", "n_face", -1, int]:
        r"""Set to eval mode and produce codes for batch of faces.

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
            Boolean masks used to separate actual edges from paddings.

        Returns
        -------
        codes : TensorType["b", "n_face", -1, int]
            Batch of face codebook code sequences. The number of code depends on the
            number of quantizers in the RQ-VAE. Specifically, it will be ``3 * n_quant``.
        """
        mode = self.training
        self.eval()

        faces = faces.masked_fill(~face_masks.unsqueeze(-1), 0)
        index = torch.arange(vertices.size(0), device=vertices.device)[:, None, None]

        coords = vertices[index, faces]
        embeds = self.embedding(coords, face_masks, edges, edge_masks)
        _, codes, _ = self.encoder(faces, embeds, face_masks)

        self.train(mode)
        return codes

    @torch.no_grad()
    def decode(
        self,
        codes: TensorType["b", "n_face", -1, int],
        face_masks: TensorType["b", "n_face", bool] | None = None,
        return_quantized: bool = False,
    ) -> TensorType["b", "n_face", 3, 3, float] | TensorType["b", "n_face", 3, 3, int]:
        r"""Decode VQ-VAE codebook codes into mesh face vertex coordinates.

        Parameters
        ----------
        codes : TensorType["b", "n_face", -1, int]
            Batch of face embedding codes
        face_masks : TensorType["b", "n_face", bool] | None, default=None
            Boolean masks used to separate actual faces from paddings. Actual faces have
            corresponding face mask values being 1. If not set, all codes are treated as
            regular faces instead of padding.
        return_quantized : bool, default=False
            Set to ``True`` to return quantized coordinate indices instead of continuous values.

        Returns
        -------
        coords : TensorType["b", "n_face", 3, 3, float] | TensorType["b", "n_face", 3, 3, int]
            Predicted coordinates decoded from input codes. If ``return_quantized`` is set
            to true, the quantized coordinate indices will be returned instead of continuous
            x-y-z coordinate values.
        """
        mode = self.training
        self.eval()

        B, T, _ = codes.size()
        if face_masks is None:
            face_masks = torch.ones((B, T), device=codes.device, dtype=bool)

        embeds = self.encoder.quantizer.get_output_from_indices(codes)
        coords = rearrange(
            self.decoder(embeds, face_masks).argmax(-1),
            "b t (v r) -> b t v r",
            v=3,
            c=3,
        )

        self.train(mode)
        if return_quantized:
            return coords

        config = self.feature_configs["vrtx"]
        coords = dequantize(coords, high_low=config.high_low, num_bins=config.num_bins)

        # Mask out padding faces with NaN
        coords = coords.masked_fill_(
            ~rearrange(face_masks, "b t -> b t 1 1"), float("nan")
        )
        return coords
