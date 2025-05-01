from __future__ import annotations

from torchtyping import TensorType


def quantize(
    t: TensorType[..., float],
    *,
    high_low: tuple[float, float],
    num_bins: int,
) -> TensorType[..., int]:
    r"""Discretize continuous inputs into bin IDs."""

    h, l = high_low
    if h <= l:
        msg = f"High must be strictly greater than low, got: <{high_low}>."
        raise ValueError(msg)

    t = ((t - l) / (h - l)) * num_bins - 0.5
    return t.round().long().clamp(min=0, max=(num_bins - 1))


def dequantize(
    t: TensorType[..., int],
    *,
    high_low: tuple[float, float],
    num_bins: int,
) -> TensorType[..., float]:
    r"""Reconstruct continuous inputs from discrete bin IDs."""

    h, l = high_low
    if h <= l:
        msg = f"High must be strictly greater than low, got: <{high_low}>."
        raise ValueError(msg)

    t = (t.float() + 0.5) / num_bins
    return t * (h - l) + l
