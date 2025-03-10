import math
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Literal

import torch
import torch.nn as nn
from einops import rearrange
from ezgatr.nn import EquiLinear, EquiRMSNorm
from ezgatr.nn.functional import (
    equi_geometric_attention,
    equi_join,
    geometric_product,
    scaler_gated_gelu,
)


@dataclass
class ModelConfig:
    r"""Configuration for image to point-cloud model.

    Parameters
    ----------
    adpt_num_blocks: int
        Number of vision encoder adaptor (FFN) blocks.
    adpt_dim_hidden: int
        Adaptor input and output size. Tying the input and output size
        together to retain the encoder output dimension. This size must
        be a multiple of 16 to be compatible with the GATr modules.
    adpt_dim_intermediate: int
        Size of the intermediate representations within FFN blocks.
    gnrt_num_blocks: int
        Number of point cloud generator blocks.
    gnrt_dim_hidden: int
        Number of channels for the hidden representations passed between
        generator blocks.
    gnrt_dim_intermediate: int
        Number of channels passed to the geometric bilinear blocks. The number
        must be divisible by 2 because the input multi-vectors will be projected
        to left and right components.
    gnrt_num_attn_heads: int
        Number of attention heads in the self-attention and cross-attention blocks.
    gnrt_lin_bias: bool
        Whether ``equi_linear`` projection contains bias term. As a reminder, bias
        terms are only added to the scalar terms of multi-vectors.
    gnrt_attn_kinds: dict[Literal["ipa", "daa"], dict[str, Any] | None]
        Kinds of similarity measures to consider in the attention calculation
        along with additional configuration/parameters sent to the corresponding
        query-key generating function. One should supply a dictionary mapping
        from the kind to parameters in addition to query and key tensors. Use
        ``None`` to denote no additional parameters supplied. Available options:
        - "ipa": Inner product attention
        - "daa": Distance-aware attention
    """

    adpt_num_blocks: int = 2
    adpt_dim_hidden: int = 768
    adpt_dim_intermediate: int = 1024

    gnrt_num_points: int = 1024
    gnrt_num_blocks: int = 2
    gnrt_dim_hidden: int = 48
    gnrt_dim_intermediate: int = 48
    gnrt_num_attn_heads: int = 8
    gnrt_lin_bias: bool = True
    gnrt_attn_kinds: dict[Literal["ipa", "daa"], dict[str, Any] | None] = field(
        default_factory=lambda: {"ipa": None, "daa": None}
    )


class AdaptorBlock(nn.Module):
    r"""Adaptor layer to transform vision embedding space to GATr space.

    The adaptor FFN is implemented as Gated Linear Units (GLU).

    Parameters
    ----------
    config: ModelConfig
        Configuration for the entire generator model. See ``ModelConfig``
        for more details.
    idx: int
        Layer index.
    """

    config: ModelConfig
    idx: int
    proj_gate: nn.Linear
    proj_next: nn.Linear
    activation: nn.GELU
    layer_norm: nn.RMSNorm

    def __init__(self, config: ModelConfig, idx: int):
        super().__init__()

        self.config = config
        self.idx = idx

        self.proj_gate = nn.Linear(
            config.adpt_dim_hidden,
            config.adpt_dim_intermediate * 2,
            bias=False,
        )
        self.proj_next = nn.Linear(
            config.adpt_dim_intermediate,
            config.adpt_dim_hidden,
            bias=False,
        )
        self.activation = nn.GELU(approximate="none")
        self.layer_norm = nn.RMSNorm(config.adpt_dim_hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        r"""Vision embedding transformation.

        Parameters
        ----------
        hidden: torch.Tensor
            Output from the vision encoder or from the previous adaptor
            block, having the shape ``(B, L, E)``.

        Returns
        -------
        torch.Tensor
            Batch of transformed (sequence of) vision embeddings.
        """
        hidden_res_conn = hidden
        hidden = self.layer_norm(hidden)

        size_inter = self.config.adpt_dim_intermediate
        hidden_gated = self.proj_gate(hidden)
        hidden_gated = (
            self.activation(hidden_gated[:, :, size_inter:])
            * hidden_gated[:, :, :size_inter]
        )

        hidden_gated = self.proj_next(hidden_gated)
        return hidden_gated + hidden_res_conn


class AdaptorModel(nn.Module):
    r"""Vision encoder adaptor model.

    It serves the purpose of transforming the vision embeddings from embedding
    space to multi-vector space.

    Parameters
    ----------
    config: ModelConfig
        Configuration for the entire generator model. See ``ModelConfig``
        for more details.
    """

    config: ModelConfig
    blocks: nn.ModuleList

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.blocks = nn.ModuleList(
            AdaptorBlock(config, i) for i in range(config.adpt_num_blocks)
        )
        self.apply(self._init_params)

    def _init_params(self, module: nn.Module):
        r"""Parameter initialization for all adaptor modules.

        Slight adjustment to Kaiming init by down-scaling the weights
        by the number of encoder layers, following the GPT-2 paper.

        Parameters
        ----------
        module : nn.Module
            Module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            module.weight.data /= math.sqrt(self.config.adpt_num_blocks)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        r"""Adaptor forward pass.

        Simply pass through all adaptor blocks and rearrange to channels of
        multi-vectors in the end.

        Parameters
        ----------
        hidden: torch.Tensor
            The output from vision encoder. The hidden dimension must be
            divisible by 16 to be compatible with GATr generator.

        Returns
        -------
        torch.Tensor
            Transformed vision embedding represented as multi-vectors.
        """
        hidden = reduce(lambda x, block: block(x), self.blocks, hidden)
        return rearrange(hidden, "... (c k) -> ... c k", k=16)


class GeneratorBilinear(nn.Module):
    r"""Implements the geometric bilinear sub-layer of the geometric MLP.

    Geometric bilinear operation consists of geometric product and equivariant
    join operations. The results of two operations are concatenated along the
    hidden channel axis and passed through a final equivariant linear projection
    before being passed to the next layer, block, or module.

    In both geometric product and equivariant join operations, the input
    multi-vectors are first projected to a hidden space with the same number of
    channels, i.e., left and right. Then, the results of each operation are
    derived from the interaction of left and right hidden representations, each
    with half number of ``gnrt_dim_intermediate``.

    Parameters
    ----------
    config: ModelConfig
        Configuration object for the model. See ``ModelConfig`` for more details.
    """

    config: ModelConfig
    proj_bili: EquiLinear
    proj_next: EquiLinear

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        if config.gnrt_dim_intermediate % 2 != 0:
            raise ValueError("Number of intermediate channels must be even.")

        self.proj_bili = EquiLinear(
            config.gnrt_dim_hidden,
            config.gnrt_dim_intermediate * 2,
            bias=config.gnrt_lin_bias,
        )
        self.proj_next = EquiLinear(
            config.gnrt_dim_intermediate,
            config.gnrt_dim_hidden,
            bias=config.gnrt_lin_bias,
        )

    def forward(
        self, hidden: torch.Tensor, reference: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Forward pass of the geometric bilinear block.

        Parameters
        ----------
        hidden: torch.Tensor
            Batch of input hidden multi-vector representation tensor.
        reference : torch.Tensor, optional
            Reference tensor for the equivariant join operation.

        Returns
        -------
        torch.Tensor
            Batch of output hidden multi-vector representation tensor of the
            same number of hidden channels.
        """
        size_inter = self.config.gnrt_dim_intermediate // 2
        lg, rg, lj, rj = torch.split(self.proj_bili(hidden), size_inter, dim=-2)

        hidden = torch.cat(
            [geometric_product(lg, rg), equi_join(lj, rj, reference)], dim=-2
        )
        return self.proj_next(hidden)


class GeneratorMLP(nn.Module):
    r"""Geometric MLP layer.

    Here we fix the structure of the MLP block to be a single equivariant linear
    projection followed by a gated GELU activation function. In addition, the
    equivariant normalization layer can be configured to be learnable.

    Parameters
    ----------
    config: ModelConfig
        Configuration object for the model. See ``ModelConfig`` for more details.
    """

    config: ModelConfig
    equi_bili: GeneratorBilinear
    proj_next: EquiLinear
    layer_norm: EquiRMSNorm

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.equi_bili = GeneratorBilinear(config)
        self.proj_next = EquiLinear(
            config.gnrt_dim_hidden,
            config.gnrt_dim_hidden,
            bias=config.gnrt_lin_bias,
        )
        self.layer_norm = EquiRMSNorm(config.gnrt_dim_hidden)

    def forward(
        self, hidden: torch.Tensor, reference: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Forward pass of the geometric MLP block.

        Parameters
        ----------
        hidden: torch.Tensor
            Batch of input hidden multi-vector representation tensor.
        reference : torch.Tensor, optional
            Reference tensor for the equivariant join operation. By default,
            the average value of all tokens and all channels will be used as
            the reference multi-vector.

        Returns
        -------
        torch.Tensor
            Batch of output hidden multi-vector representation tensor of the
            same number of hidden channels.
        """
        hidden_res_conn = hidden
        reference = reference or torch.mean(
            hidden,
            dim=tuple(range(1, len(hidden.shape) - 1)),
            keepdim=True,
        )

        hidden = self.equi_bili(self.layer_norm(hidden), reference)
        hidden = self.proj_next(scaler_gated_gelu(hidden, "none"))

        return hidden + hidden_res_conn


class GeneratorSelfAttention(nn.Module):
    r"""Geometric self-attention block without scaler channels.

    The GATr attention calculation is slightly different from the original
    transformers implementation in that each head has the sample number-of-
    channels as the input tensor, instead of dividing into smaller chunks.
    In this case, the final output linear transformation maps from
    ``gnrt_dim_hidden * gnrt_num_attn_heads`` to ``gnrt_dim_hidden``.

    Parameters
    ----------
    config : ModelConfig
        Configuration object for the model. See ``ModelConfig`` for more details.
    """

    config: ModelConfig
    proj_attn: EquiLinear
    proj_next: EquiLinear
    layer_norm: EquiRMSNorm

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.proj_attn = EquiLinear(
            config.gnrt_dim_hidden,
            config.gnrt_dim_hidden * config.gnrt_num_attn_heads * 3,
            bias=config.gnrt_lin_bias,
        )
        self.proj_next = EquiLinear(
            config.gnrt_dim_hidden * config.gnrt_num_attn_heads,
            config.gnrt_dim_hidden,
            bias=config.gnrt_lin_bias,
        )
        self.layer_norm = EquiRMSNorm(config.gnrt_dim_hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the geometric self-attention block.

        Parameters
        ----------
        hidden: torch.Tensor
            Batch of input hidden multi-vector representation tensor.

        Returns
        -------
        torch.Tensor:
            Hidden states after time-mixing.
        """
        hidden_res_conn = hidden
        hidden = self.layer_norm(hidden)

        q, k, v = rearrange(
            self.proj_attn(hidden),
            "b t (qkv h c) k -> qkv b h t c k",
            qkv=3,
            h=self.config.gnrt_num_attn_heads,
            c=self.config.gnrt_dim_hidden,
        )
        hidden, _ = equi_geometric_attention(
            q,
            k,
            v,
            kinds=self.config.attn_kinds,
            is_causal=False,
        )
        hidden = rearrange(
            hidden,
            "b h t c k -> b t (h c) k",
            h=self.config.gnrt_num_attn_heads,
        )
        hidden = self.proj_next(hidden)

        return hidden + hidden_res_conn


class GeneratorCrossAttention(nn.Module):
    r"""
    """

    config: ModelConfig
    proj_attn_gnrt: EquiLinear
    proj_attn_adpt: EquiLinear
    proj_next: EquiLinear
    layer_norm: EquiRMSNorm

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.proj_attn_adpt = EquiLinear(
            config.gnrt_dim_hidden,
            config.gnrt_dim_hidden * config.gnrt_num_attn_heads * 2,
            bias=config.gnrt_lin_bias,
        )
        self.proj_attn_gnrt = EquiLinear(
            config.gnrt_dim_hidden,
            config.gnrt_dim_hidden * config.gnrt_num_attn_heads,
            bias=config.gnrt_lin_bias,
        )
        self.proj_next = EquiLinear(
            config.gnrt_dim_hidden * config.gnrt_num_attn_heads,
            config.gnrt_dim_hidden,
            bias=config.gnrt_lin_bias,
        )
        self.layer_norm = EquiRMSNorm(config.gnrt_dim_hidden)

    def forward(
        self,
        hidden: torch.Tensor,
        vision_embedding: torch.Tensor,
    ) -> torch.Tensor:
        r""""""



class GeneratorBlock(nn.Module):
    r""""""

    config: ModelConfig
    idx: int

    def __init__(self, config: ModelConfig, idx: int):
        super().__init__()

        self.config = config
        self.idx = idx

    def forward(self,):
        pass


class GeneratorModel(nn.Module):
    r"""Glue generator blocks together.

    Parameters
    ----------
    config: ModelConfig
        Configuration object for the model. See ``ModelConfig`` for more details.
    """

    config: ModelConfig
    head: EquiLinear
    embedding: EquiLinear
    blocks: nn.ModuleList

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.head = EquiLinear(
            config.gnrt_dim_hidden,
            1,
            bias=config.gnrt_lin_bias,
        )
        self.embedding = EquiLinear(
            1,
            config.gnrt_dim_hidden,
            bias=config.gnrt_lin_bias,
        )
        self.blocks = nn.ModuleList(
            GeneratorBlock(config, i) for i in range(config.gnrt_num_blocks)
        )
        self.apply(self._init_params)

    def _init_params(self, module: nn.Module):
        r"""Parameter initialization for all generator modules.

        Slight adjustment to Kaiming init by down-scaling the weights
        by the number of encoder layers, following the GPT-2 paper.

        Parameters
        ----------
        module: nn.Module
            Module to initialize.
        """
        if isinstance(module, EquiLinear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            module.weight.data /= math.sqrt(self.config.gnrt_num_blocks)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        noise: torch.Tensor,
        vision_embedding: torch.Tensor,
        reference: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Forward pass of the point cloud generator.

        Parameters
        ----------
        noise: torch.Tensor
            Randomly generated point cloud embedded as PGA multi-vectors.
        vision_embedding: torch.Tensor
            Vision embedding transformed by the adaptor.
        reference: torch.Tensor, optional
            The reference multi-vector used in geometric bilinear operation.

        Returns
        -------
        torch.Tensor
            Generated point cloud embedded as PGA multi-vectors.
        """
        point_cloud = reduce(
            lambda x, block: block(x, vision_embedding, reference),
            self.blocks,
            self.embedding(noise),
        )
        return self.head(point_cloud)


class Img2PCModel(nn.Module):
    r"""Combine vision adaptor and point cloud generator together.

    
    """
    pass
