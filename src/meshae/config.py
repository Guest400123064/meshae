from __future__ import annotations

from dataclasses import dataclass


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
