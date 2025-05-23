from __future__ import annotations

from dataclasses import dataclass

from meshae.typing import MeshAEFeatNameType


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


@dataclass
class MeshAEModelConfig:
    r"""Configurations for auto-encoder model initialization.

    Parameters
    ----------
    feature_configs : list[MeshAEFeatureConfig]
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
    bin_smooth_blur_sigma : float, default=0.0
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

    feature_configs: dict[MeshAEFeatNameType, MeshAEFeatEmbedConfig]
    codebook_size: int = 256
    hidden_size: int = 512
    num_sageconv_layers: int = 1
    num_encoder_layers: int = 12
    num_encoder_heads: int = 8
    num_quantizers: int = 2
    num_codebook_codes: int = 4096
    num_decoder_layers: int = 12
    num_decoder_heads: int = 8
    num_refiner_layers: int = 6
    num_refiner_heads: int = 8
    sample_codebook_temp: float = 0.1
    commitment_weight: float = 1.0
    bin_smooth_blur_sigma: float = 0.0
