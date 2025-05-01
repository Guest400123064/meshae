from __future__ import annotations

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
