from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce, partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


@dataclass
class FaceTokenConfig:
    r""""""

    size_in: int = 9
    size_out: int = 9
    size_hidden: int = 512
    size_intermediate: int = 512
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3

    codebook_size: int = 1024
    codebook_dim: int = 256
    codebook_heads: int = 4
    rotation_trick: bool = True


class FaceTokenMLP(nn.Module):
    r"""Simple gated linear unit module with layer norm and residual connection.

    Parameters
    ----------
    config : FaceTokenConfig
        Configuration object for the VAE model. See ``FaceTokenConfig`` for more details.
    """

    def __init__(self, config: FaceTokenConfig) -> None:
        super().__init__()
        self.config = config

        self.norm = nn.LayerNorm(config.size_hidden)
        self.proj_up = nn.Linear(config.size_hidden, config.size_intermediate)
        self.proj_gate = nn.Linear(config.size_hidden, config.size_intermediate)
        self.proj_down = nn.Linear(config.size_intermediate, config.size_hidden)

    def forward(self, x_or_q: torch.Tensor) -> torch.Tensor:
        r_conn = x_or_q
        l_norm = self.norm(x_or_q)
        return r_conn + self.proj_down(self.proj_up(l_norm) * F.gelu(self.proj_gate(l_norm)))


class FaceTokenModel(nn.Module):
    r"""The Face VAE model of mesh tokenizer.

    Parameters
    ----------
    config : FaceTokenConfig
        Configuration object for the VAE model. See ``FaceTokenConfig`` for more details.
    """

    def __init__(self, config: FaceTokenConfig) -> None:
        super().__init__()
        self.config = config

        self.vq = VectorQuantize(
            dim=config.size_hidden,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            heads=config.codebook_heads,
            rotation_trick=config.rotation_trick,
        )
        self.proj_in = nn.Linear(config.size_in, config.size_hidden)
        self.proj_out = nn.Linear(config.size_hidden, config.size_out)
        self.position = nn.Parameter(torch.randn(config.size_in))
        self.encoder = nn.ModuleList(FaceTokenMLP(config) for _ in range(config.num_encoder_layers))
        self.decoder = nn.ModuleList(FaceTokenMLP(config) for _ in range(config.num_decoder_layers))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = reduce(lambda x, mlp: mlp(x), self.encoder, self.proj_in(self.position + x))
        quantized, _, loss = self.vq(embedding)
        reconstruct = self.proj_out(reduce(lambda x, mlp: mlp(x), self.decoder, quantized))
        return reconstruct, loss
