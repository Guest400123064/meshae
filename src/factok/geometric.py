from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache, partial, reduce
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ezgatr.nn import EquiLinear, EquiRMSNorm
from ezgatr.nn.functional import (
    equi_join,
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
    intermediate_size: int = 8
    codebook_size: int = 2
    frozen_codebook_size: int = 2
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_codebook_codes: int = 256
    num_codebook_heads: int = 8


class FaceTokenVQ(nn.Module):
    r"""Face encoding vector quantizer.

    """

    def __init__(self, config: FaceTokenConfig):
        super().__init__()

        self.config = config

        # The codebook initialization method follows @lucidrains's vector-quantize-pytorch
        #     by paring a Gaussian random noise frozen notebook with a codebook transformation.
        #     We set the transformation to a be simple EquiLinear operation.
        frozen_codebook = (
            torch.randn(
                config.num_codebook_codes, config.frozen_codebook_size, 16,
                dtype=torch.float32,
                requires_grad=False,
            )
            / math.sqrt(config.frozen_codebook_size)
        )
        self.register_buffer("frozen_codebook", frozen_codebook)
        self.proj_i = EquiLinear(config.frozen_codebook_size, config.codebook_size)
        self.proj_o = EquiLinear(config.codebook_size, config.codebook_size)

    @property
    def codebook(self) -> torch.Tensor:
        return self.proj_o(scaler_gated_gelu(self.proj_i(self.frozen_codebook)))

    def _rearrange_x_shape(self, t, is_restore=False):
        expr = "b (h d) k -> (b h) (d k)"
        if is_restore:
            expr = "(b h) (d k) -> b (h d) k"
        return rearrange(
            t,
            expr,
            h=self.config.num_codebook_heads,
            d=self.config.codebook_size,
            k=16,
        )

    def _rotate_x_to_e(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        r"""Rotate ``x`` to ``e`` with Householder transformation.

        Parameters
        ----------
        x : torch.Tensor
            Flattened encoded mesh faces of shape ``((B * H), (D * 16))``.
        e : torch.Tensor
            Selected batch of code with the same shape as ``x``.

        Returns
        -------
        torch.Tensor
            The transformed ``x`` to match ``e`` numerically.
        """

        def _div(n, d):
            return n / d.clamp(min=1e-6)

        def _rot(xd, ed, x):
            sd = F.normalize(xd + ed, p=2, dim=-1, eps=1e-6)
            return (
                x
                - 2 * sd * (sd * x).sum(dim=-1, keepdim=True)
                + 2 * ed * (xd * x).sum(dim=-1, keepdim=True)
            )

        xn = x.norm(dim=-1, keepdim=True)
        en = e.norm(dim=-1, keepdim=True)

        r = _rot(_div(x, xn).detach(), _div(e, en).detach(), x)
        return r * _div(en, xn).detach()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Lookup the codebook and return ``num_codebook_heads`` codes for each head.

        Parameters
        ----------
        x : torch.Tensor
            Multi-head encoded batch of faces with shape ``(B, (H * D), 16)`` where ``H`` denotes
            the number of heads and ``D`` denotes the number of channels in each code.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The codes directly selected from the codebook and the "straight-through" quantization
            estimated from the rotation trick.
        """
        x = self._rearrange_x_shape(x)

        codebook = rearrange(self.codebook, "... d k -> ... (d k)")
        with torch.no_grad():
            i = torch.cdist(x, codebook, p=2).argmin(dim=-1).squeeze()

        e = codebook[i]
        s = self._rearrange_x_shape(self._rotate_x_to_e(x, e), is_restore=True)
        return self._rearrange_x_shape(e, is_restore=True), s

    @torch.no_grad
    def lookup(self, i: torch.LongTensor) -> torch.Tensor:
        return self._rearrange_x_shape(self.codebook[i])


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

    def forward(self, f_or_z, r):
        geom_l, geom_r, join_l, join_r = torch.split(
            self.proj_b(f_or_z),
            split_size_or_sections=self.config.intermediate_size // 2,
            dim=-2,
        )
        f_or_z = torch.cat(
            [geometric_product(geom_l, geom_r), equi_join(join_l, join_r, r)],
            dim=-2,
        )
        return self.proj_o(f_or_z)


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
        self.norm = EquiRMSNorm(config.hidden_size, 1e-9)

    def forward(self, f_or_z, r):
        return self.proj(
            scaler_gated_gelu(self.bili(self.norm(f_or_z), r))
        )


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
        self.proj_x = EquiLinear(
            config.hidden_size,
            config.num_codebook_heads * config.codebook_size,
        )
        self.proj_z = EquiLinear(
            config.num_codebook_heads * config.codebook_size,
            config.hidden_size,
        )
        self.vq = FaceTokenVQ(config)
        self.encoder = nn.ModuleList(
            FaceTokenLayer(config) for _ in range(config.num_encoder_layers)
        )
        self.decoder = nn.ModuleList(
            FaceTokenLayer(config) for _ in range(config.num_decoder_layers)
        )

        self.encoder.apply(partial(self._init_params, n=config.num_encoder_layers))
        self.decoder.apply(partial(self._init_params, n=config.num_decoder_layers))

    def _init_params(self, m: nn.Module, n: int = 1) -> None:
        if isinstance(m, EquiLinear):
            nn.init.kaiming_uniform_(m.weight)
            m.weight.data *= math.sqrt(
                self.config.num_encoder_layers + self.config.num_decoder_layers
            )

    def forward(self, f: torch.Tensor, r: Optional[torch.Tensor] = None) -> tuple[torch.Tensor]:
        r"""Forward pass of mesh face VAE.

        This method pass through both the encoder and decoder components of face VAE to
        obtain the reconstructed face, encoded face, and associated codebook codes. The
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
            Reconstructed face, encoded face, codebook codes, in that order.
        """
        r = r or torch.mean(f, keepdim=True, dim=tuple(range(1, len(f.shape) - 1)))

        # Explanation of namings:
        #    - 'x': Encoded faces, i.e., the output from ``z_e(x)``.
        #    - 'e': Nearest codebook code from 'x', i.e., the ``e_k``.
        #    - 's': Straight through component for encoder gradient flow.
        #    - 'z': Latent input for decoder, i.e., the input to ``z_q(x)``.
        f = reduce(lambda f, l: l(f, r), self.encoder, self.proj_i(f))
        x = self.proj_x(f)

        # Lookup the codebook to get the latent representations.
        #     The encoded faces 'x' should be of shape (B, (H * D), 16) where H denotes the
        #     number of codebook heads and D denotes the number of channels of each codebook
        #     code (i.e., the 'e's). 'x' and 'e' will be returned for codebook update.
        # e, s = self.vq(x)
        s = x
        z = self.proj_z(s)

        # Use the average multi-vector across (projected) latent channels as reference.
        r = torch.mean(z, keepdim=True, dim=tuple(range(1, len(z.shape) - 1))).detach()
        p = reduce(lambda z, l: l(z, r), self.decoder, z)
        return self.proj_o(p), x

    @torch.no_grad
    def encode(self, f: torch.Tensor, r: Optional[torch.Tensor] = None) -> torch.LongTensor:
        r"""Encode pass at inference time.

        Parameters
        ----------
        f : torch.Tensor
            Batch of mesh faces encoded as multi-vectors. Expecting shape ``(B, N, C, 16)``
            where ``C`` denotes the number of input channels and ``N`` denotes the number of
            faces in each batch record.
        r : Optional[torch.Tensor]
            Batch of reference multi-vectors used in the ``equi_join`` operation. If not
            supplied, the average multi-vectors across the input channels will be used.

        Returns
        -------
        torch.LongTensor
            Batch of codebook IDs of the encoded faces of shape ``(B, N)``.
        """
        self.eval()

        r = r or torch.mean(f, keepdim=True, dim=tuple(range(1, len(f.shape) - 1)))
        f = reduce(lambda f, l: l(f, r), self.encoder, self.proj_i(f))
        return self.proj_x(f)

    @torch.no_grad
    def decode(self, i: torch.LongTensor) -> torch.Tensor:
        f"""Decode pass at inference time.

        Parameters
        ----------
        i : torch.LongTensor

        Returns
        -------
        torch.Tensor
            Decoded batch of mesh faces encoded as multi-vectors of shape ``(B, N, C, 16)``.
        """
        self.eval()
        raise NotImplementedError
