from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torchtyping import TensorType


def quantize(
    tensor: TensorType[..., float],
    *,
    high_low: tuple[float, float],
    num_bins: int,
) -> TensorType[..., int]:
    r"""Discretize continuous inputs into bin IDs."""

    high, low = high_low
    if high <= low:
        msg = f"High must be strictly greater than low, got: <{high_low}>."
        raise ValueError(msg)

    tensor = ((tensor - low) / (high - low)) * num_bins - 0.5
    return tensor.round().long().clamp(min=0, max=(num_bins - 1))


def dequantize(
    tensor: TensorType[..., int],
    *,
    high_low: tuple[float, float],
    num_bins: int,
) -> TensorType[..., float]:
    r"""Reconstruct continuous inputs from discrete bin IDs."""

    high, low = high_low
    if high <= low:
        msg = f"High must be strictly greater than low, got: <{high_low}>."
        raise ValueError(msg)

    tensor = (tensor.float() + 0.5) / num_bins
    return tensor * (high - low) + low


def gaussian_blur1d(
    tensor: TensorType[..., float], *, sigma: float = 1.0,
) -> TensorType[..., float]:
    r"""Apply 1D Gaussian blur to the last dimension of input tensor."""

    width = int(math.ceil(sigma * 5))
    width += (width + 1) % 2
    half_width = width // 2

    distance = torch.arange(
        -half_width, half_width + 1, dtype=tensor.dtype, device=tensor.device,
    )
    gaussian = torch.exp(-(distance**2) / (2 * sigma**2))
    gaussian = F.normalize(gaussian, 1, dim=-1)
    channels = tensor.size(-1)

    tensor = F.conv1d(
        rearrange(tensor, "... n c -> ... c n"),
        repeat(gaussian, "n -> c 1 n", c=channels),
        padding=half_width,
        groups=channels,
    )
    return rearrange(tensor, "... c n -> ... n c")
