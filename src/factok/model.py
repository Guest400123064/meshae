from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache, partial, reduce
from typing import Optional

import torch
import torch.nn as nn
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
    codebook_size: int = 4
    intermediate_size: int = 16
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_codebook_codes: int = 1024
    num_codebook_heads: int = 4
    ema_decay: float = 0.99


@lru_cache(maxsize=None, typed=True)
def _compute_ip_selector(
    device: torch.device, keep_tri_vector: bool = True
) -> torch.Tensor:
    r"""Get blades involved in the inner product calculation."""

    idx = [0, 2, 3, 4, 8, 9, 10, 14]
    if not keep_tri_vector:
        idx.pop(-1)
    return torch.tensor(idx, device=device)


@lru_cache(maxsize=None, typed=True)
def _compute_tv_selector(device: torch.device) -> torch.Tensor:
    r"""Get blades corresponding to tri-vectors."""

    return torch.tensor([11, 12, 13, 14], device=device)


def _flatten_mv(mv: torch.Tensor) -> torch.Tensor:
    r"""Shortcut for flattening multi-vector dimensions."""

    return rearrange(mv, "... c k -> ... (c k)")


def _compute_ip_elem(x_or_e: torch.Tensor) -> torch.Tensor:
    r"""Simply select inner product blades and flatten the multi-vector dimension."""

    idx = _compute_ip_selector(x_or_e.device, keep_tri_vector=False)
    return _flatten_mv(x_or_e[..., idx])


def _linear_square_normalizer(e123: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    r"""Apply linear square normalization to the input tensor.

    Parameters
    ----------
    e123 : torch.Tensor
        Coefficients corresponds to the ``e_{123}`` blade.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Normalized multi-vector tensor.
    """
    return e123 / (e123.pow(2) + eps)


@lru_cache(maxsize=None, typed=True)
def _compute_da_qk_basis(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute basis queries and keys in the distance-aware attention.

    Parameters
    ----------
    device: torch.device
        Device for the basis.
    dtype: torch.dtype
        Data type for the basis.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Basis tensor for the queries and keys in the distance-aware attention.
        Both with shape (4, 4, 5).
    """
    bq = torch.zeros((4, 4, 5), device=device, dtype=dtype)
    bk = torch.zeros((4, 4, 5), device=device, dtype=dtype)
    r3 = torch.arange(3, device=device)

    bq[r3, r3, 0] = 1.0
    bk[3, 3, 0] = -1.0

    bq[3, 3, 1] = 1.0
    bk[r3, r3, 1] = -1.0

    bq[r3, 3, r3 + 2] = 1.0
    bk[r3, 3, r3 + 2] = 2.0

    return bq, bk


def _compute_da_elem(x_or_e: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    r"""Compute the basis for distance-aware attention.

    Please refer to the GATr paper [1]_ for more details.

    References
    ----------
    .. [2] `"Geometric Algebra Transformer", Brehmer et al., 2023
        <https://arxiv.org/abs/2305.18415>`_
    """
    idx = _compute_tv_selector(x_or_e.device)
    tri = x_or_e[..., idx]
    ret = tri * _linear_square_normalizer(tri[..., [3]], eps=1e-3)
    return _flatten_mv(
        torch.einsum("ijk, ...i, ...j -> ...k", basis, ret, ret)
    )


def compute_xe_dist(x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    r"""Compute the inner product distance between every ``x`` and ``e``.

    In the original VQ-VAE paper [1]_, the authors use the Euclidean distance
    between encoded inputs and codebook codes to determine the nearest code.
    However, since geometric algebra representations have strict geometric
    interpretations, we chose to have the distance function aligned with the
    equi-variant geometric attention formulation in GATr [2]_. Specifically,
    the distances are decomposed into an inner product component and distance
    (3D space) aware component.

    TODO: There is one thing we may need to check later, which is whether blades
    involving ``e_0`` elements are updated or not. From the current distance
    calculation, it seems that the ``e_0`` elements are never updated. Thus, this
    may creates a situation where the encoder is well-trained while the codebook
    was under-trained. Probably an easy way to fix this is to pass the codes through
    a grade-mixing layer first.

    Parameters
    ----------
    x : torch.Tensor
        Reshaped multi-head encoded faces with shape ``((B * H), D, 16)`` where
        ``D`` denotes the number of channels in each code.
    e : torch.Tensor
        Codebook of shape ``(K, D, 16)`` where ``K`` denotes the number of codebook
        codes and ``D`` denotes the number of channels in each code. This distance
        function is also used to compute the dictionary loss. In that case, the
        ``e`` input is expected to have shape ``((B * H), D, 16)`` as well.

    Returns
    -------
    torch.Tensor
        A dot product matrix of shape ``((B * H), K)`` where ``H`` denotes the
        number of codebook heads and ``K`` denotes the number of codebook codes.

    References
    ----------
    .. [1] `"Neural Discrete Representation Learning", Van Den Oord et al., 2018
            <https://arxiv.org/abs/2305.18415>`_
    .. [2] `"Geometric Algebra Transformer", Brehmer et al., 2023
            <https://arxiv.org/abs/2305.18415>`_
    """
    bq, bk = _compute_da_qk_basis(x.device, x.dtype)
    q = torch.cat([_compute_ip_elem(x), _compute_da_elem(x, bq)], dim=-1)
    k = torch.cat([_compute_ip_elem(e), _compute_da_elem(e, bk)], dim=-1)
    return q @ k.T


class FaceTokenVQ(nn.Module):
    r""""""

    def __init__(self, config: FaceTokenConfig):
        super().__init__()

        self.config = config

        # TODO: There could be better param init method...
        codebook = nn.Parameter(
            torch.randn(
                config.num_codebook_codes, config.codebook_size, 16,
                dtype=torch.float32,
            ),
            requires_grad=True,
        )
        self.register_parameter("codebook", codebook)
        self.proj_m = EquiLinear(config.codebook_size, config.codebook_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Lookup the codebook and return ``num_codebook_heads`` codes for each head.

        Parameters
        ----------
        x : torch.Tensor
            Multi-head encoded batch of faces with shape ``(B, (H * D), 16)`` where
            ``H`` denotes the number of heads and ``D`` denotes the number of channels
            in each code.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The codes directly selected from the codebook and the numerically-equivalent
            codes for the straight-through gradient computation [1]_.

        References
        ----------
        .. [1] `"Neural Discrete Representation Learning", Van Den Oord et al., 2018
            <https://arxiv.org/abs/2305.18415>`_
        """
        x_flat = rearrange(
            x, "b (h d) k -> (b h) d k",
            h=self.config.num_codebook_heads,
            d=self.config.codebook_size,
            k=16,
        )

        # Let's perform a grade-mixing first to make sure the 'e_0' blades are updated
        e = self.proj_m(self.codebook)
        assert e.shape == self.codebook.shape

        # Note that we are using the inner product "distance" here instead of L2,
        #     so we should do an 'argmax' operation. Then followed by straight-through
        #     gradient component.
        i = compute_xe_dist(x_flat, e).argmax(dim=-1).squeeze()
        e = rearrange(
            e[i],
            "(b h) d k -> b (h d) k",
            h=self.config.num_codebook_heads,
            d=self.config.codebook_size,
            k=16,
        )
        return e, (e - x).detach() + x

    @torch.no_grad
    def lookup(self, i: torch.LongTensor) -> torch.Tensor:
        return rearrange(
            self.proj_m(self.codebook[i]).detach(),
            "(b h) d k -> b (h d) k",
            h=self.config.num_codebook_heads,
            d=self.config.codebook_size,
            k=16,
        )


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
        self.norm = EquiRMSNorm(config.hidden_size, 1e-6)

    def forward(self, f_or_z, r):
        c = f_or_z
        f_or_z = scaler_gated_gelu(self.bili(self.norm(f_or_z), r), "none")
        return self.proj(f_or_z) + c


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
        r = r or torch.mean(f, keepdim=True, dim=(1,))

        # Explanation of namings:
        #    - 'x': Encoded faces, i.e., the ``Proj_x(z_e(x))``.
        #    - 'e': Nearest codebook code from 'x', i.e., the ``e_k``.
        #    - 's': Numerically the same as 'e', used for straight-through gradient.
        #    - 'z': Latent input for decoder, i.e., the ``Proj_z(z_q(x))``.
        f = reduce(lambda f, l: l(f, r), self.encoder, self.proj_i(f))
        x = self.proj_x(f)

        # Lookup the codebook to get the latent representations.
        #     The encoded faces 'x' should be of shape (B, (H * D), 16) where H denotes the
        #     number of codebook heads and D denotes the number of channels of each codebook
        #     code (i.e., the 'e's). 'x' and 'e' will be returned for codebook update.
        e, s = self.vq(x)
        z = self.proj_z(s)

        # Use the average multi-vector across (projected) latent channels as reference.
        r = torch.mean(z, keepdim=True, dim=(1,)).detach()
        p = reduce(lambda z, l: l(z, r), self.decoder, z)
        return self.proj_o(p), x, e

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
        raise NotImplementedError

    @torch.no_grad
    def decode(self, i: torch.LongTensor) -> torch.Tensor:
        f"""Decode pass at inference time.

        Parameters
        ----------
        i : torch.LongTensor
            TODO: /////////////////////////////////////////////////////////////////////////

        Returns
        -------
        torch.Tensor
            Decoded batch of mesh faces encoded as multi-vectors of shape ``(B, N, C, 16)``.
        """
        self.eval()
        raise NotImplementedError
