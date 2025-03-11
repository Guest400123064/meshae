import math
from dataclasses import dataclass, field
from functools import reduce

import torch
import torch.nn as nn
from einops import rearrange
from ezgatr.interfaces import point
from gatr import GATr, MLPConfig, SelfAttentionConfig
from gatr.layers import EquiLinear


@dataclass
class Img2PCModelConfig:

    adpt_dim_hidden: int = 768
    adpt_dim_intermediate: int = 1024
    adpt_num_blocks: int = 2
    gnrt_num_blocks: int = 4


class AdaptorBlock(nn.Module):
    r"""Adaptor layer to transform vision embedding space to GATr space.

    The adaptor FFN is implemented as Gated Linear Units (GLU).

    Parameters
    ----------
    config: Img2PCModelConfig
        Configuration for the entire generator model. See ``Img2PCModelConfig``
        for more details.
    idx: int
        Layer index.
    """

    config: Img2PCModelConfig
    idx: int
    proj_gate: nn.Linear
    proj_next: nn.Linear
    activation: nn.GELU
    layer_norm: nn.LayerNorm

    def __init__(self, config: Img2PCModelConfig, idx: int):
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
        self.layer_norm = nn.LayerNorm(config.adpt_dim_hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
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
    config: Img2PCModelConfig
        Configuration for the entire generator model. See ``Img2PCModelConfig``
        for more details.
    """

    config: Img2PCModelConfig
    blocks: nn.ModuleList

    def __init__(self, config: Img2PCModelConfig):
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


class Img2PCModel(nn.Module):
    r""""""

    config: Img2PCModelConfig
    embedding: EquiLinear
    adaptor: AdaptorModel
    generator: GATr

    def __init__(self, config: Img2PCModelConfig):
        super().__init__()

        self.config = config

        self.embedding = EquiLinear(1, config.adpt_dim_hidden // 16)
        self.adaptor = AdaptorModel(config)
        self.generator = GATr(
            in_mv_channels=config.adpt_dim_hidden // 16,
            out_mv_channels=1,
            hidden_mv_channels=config.adpt_dim_hidden // 16,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=1,  # Dummy setting, should be 0
            num_blocks=config.gnrt_num_blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )

    def forward(self, noise, vision_embedding):
        prompt = self.adaptor(vision_embedding)
        noise, _ = self.embedding(noise)

        point_cloud, _ = self.generator(
            torch.cat([prompt, noise], dim=1),
            scalars=None,
        )

        return point.decode(point_cloud[:, vision_embedding.shape[1]:])
