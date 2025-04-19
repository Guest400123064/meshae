from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce, partial
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from ezgatr.nn import EquiLinear, EquiRMSNorm
from ezgatr.nn.functional import (
    equi_join,
    inner_product,
    geometric_product,
    scaler_gated_gelu,
)


@dataclass
class FaceTokenConfig:
    r"""Configuration object for the face VAE model.

    Parameters
    ----------
    """

    input_size: int = 4
    output_size: int = 4
    hidden_size: int = 8
    codebook_size: int = 4
    intermediate_size: int = 16
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_codebook_codes: int = 1024
    num_codebook_heads: int = 4


class FaceTokenBilinear(nn.Module):
    r"""Implements the geometric bilinear operation in GATr.

    Parameters
    ----------
    config : FaceTokenConfig
        Configuration object for the VAE model. See ``FaceTokenConfig`` for more details.
    """

    def __init__(self, config: FaceTokenConfig):
        super().__init__()

        self.config = config
        if config.intermediate_size % 2 != 0:
            msg = f"Intermediate size must be even, got <{config.intermediate_size}>."
            raise ValueError(msg)

        self.proj_b = EquiLinear(config.hidden_size, config.intermediate_size * 2)
        self.proj_o = EquiLinear(config.intermediate_size, config.hidden_size)

    def forward(self, x, r):
        geom_l, geom_r, join_l, join_r = torch.split(
            self.proj_b(x),
            split_size_or_sections=self.config.intermediate_size // 2,
            dim=-2,
        )
        x = torch.cat(
            [geometric_product(geom_l, geom_r), equi_join(join_l, join_r, r)],
            dim=-2,
        )
        return self.proj_o(x)


class FaceTokenLayer(nn.Module):
    r"""FFN block with residual connection.

    Parameters
    ----------
    config : FaceTokenConfig
        Configuration object for the VAE model. See ``FaceTokenConfig`` for more details.
    """

    def __init__(self, config: FaceTokenConfig):
        super().__init__()

        self.config = config

        self.bili = FaceTokenBilinear(config)
        self.proj = EquiLinear(config.hidden_size, config.hidden_size)
        self.norm = EquiRMSNorm(config.hidden_size, 1e-6)

    def forward(self, x, r):
        c = x
        x = scaler_gated_gelu(self.bili(self.norm(x), r), "none")
        return self.proj(x) + c


class FaceTokenCodeBook(nn.Module):
    r""""""

    def __init__(self, config: FaceTokenConfig):
        super().__init__()

        self.config = config

        self.proj_h = EquiLinear(config.hidden_size, config.num_codebook_heads)
        self.proj_o = EquiLinear(config.num_codebook_heads, config.hidden_size)
        self.lookup = nn.Embedding(config.num_codebook_codes, config.codebook_size * 16)

    def forward(self, x):
        pass


class FaceTokenModel(nn.Module):
    r"""The Face VAE model of mesh tokenizer.

    Parameters
    ----------
    config : FaceTokenConfig
        Configuration object for the VAE model. See ``FaceTokenConfig`` for more details.
    """

    def __init__(self, config: FaceTokenConfig):
        super().__init__()

        self.config = config

        self.proj_i = EquiLinear(config.input_size, config.hidden_size)
        self.proj_o = EquiLinear(config.hidden_size, config.output_size)
        self.codebook = FaceTokenCodeBook(config)
        self.encoder = nn.ModuleList(
            FaceTokenLayer(config) for _ in range(config.num_encoder_layers)
        )
        self.decoder = nn.ModuleList(
            FaceTokenLayer(config) for _ in range(config.num_decoder_layers)
        )

        self.codebook.apply(self._init_params)
        self.encoder.apply(partial(self._init_params, n=config.num_encoder_layers))
        self.decoder.apply(partial(self._init_params, n=config.num_decoder_layers))

    def _init_params(self, m: nn.Module, n: int = 1) -> None:
        r"""Initialize layers with modified kaiming normal.

        In the GPT-2 paper [1]_, the linear layers weights are down scaled by the square
        root of number of layers to enhance stability. But this might be trivial in this
        case because the encoder and decoder models are pretty shallow.

        Parameters
        ----------
        m : nn.Module
            Module to initialize.
        n : int
            Number of layers in the sub-network.

        References
        ----------
        .. [1] `"Language Models are Unsupervised Multitask Learners", Radford et al., 2020
                <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`_
        """
        if isinstance(m, EquiLinear):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data /= math.sqrt(n)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, f: torch.Tensor, r: Optional[torch.Tensor] = None) -> tuple[torch.Tensor]:
        r"""Forward pass of mesh face VAE.

        This method pass through both the encoder and decoder components of face VAE to
        obtain the reconstructed face, encoded face, and associated codebook latent. The
        outputs should be supplied to VAE losses.

        For tokenizer inference, please checkout ``FaceTokenModel.encode`` and associated
        ``FaceTokenModel.decode`` methods.

        Parameters
        ----------
        f : torch.Tensor
            Batch of mesh faces encoded as multi-vectors. Expecting shape ``(B, C, 16)``
            where ``C`` denotes the number of input channels. This can vary depending on
            the face encoding schema.
        r : Optional[torch.Tensor]
            Batch of reference multi-vectors used in the ``equi_join`` operation. This is
            only supplied to the join operations in encoder layers. If not supplied, the
            average multi-vectors across the input channels will be used. Decoder layers
            will use the average of latent multi-vectors channels as the reference.

        Returns
        -------
        tuple[torch.Tensor]

        """
        r = r or torch.mean(f, keepdim=True, dim=(1, 2))
        e = reduce(lambda x, l: l(x, r), self.encoder, self.proj_i(f))
        z = self.codebook(e)
        r = torch.mean(z, keepdim=True, dim=(1, 2)).detach()
        f = reduce(lambda x, l: l(x, r), self.decoder, z)
        return self.proj_o(f)

    @torch.no_grad
    def encode(self, f: torch.Tensor) -> torch.LongTensor:
        r"""Encode pass at inference time.
        """
        self.eval()

    @torch.no_grad
    def decode(self, i: torch.LongTensor) -> torch.Tensor:
        f"""Decode pass at inference time.
        """
        self.eval()
