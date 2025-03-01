from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    r"""Configuration for image to point-cloud model.

    Parameters
    ----------
    adpt_num_layers: int
        Number of vision encoder adaptor (FFN) blocks.
    adpt_dim_hidden: int
        Adaptor input and output size. Tying the input and output size
        together to retain the encoder output dimension. This size must
        be a multiple of 16 to be compatible with the GATr modules.
    adpt_dim_intermediate: int
        Size of the intermediate representations within FFN blocks.
    gnrt_num_layers: int
        Number of point cloud generator blocks.
    gnrt_num_point_in_pc: int
        Number of point in the point could.
    gnrt_num_prompt_tokens: int
        Number of vision embedding tokens. For instance, the ViT-16x16
        model can have at most 196 + 1 tokens to be used as the prompt.

    """

    adpt_num_layers: int = 2
    adpt_dim_hidden: int = 768
    adpt_dim_intermediate: int = 1024


class AdaptorBlock(nn.Module):
    r"""Adaptor layer to transform vision embedding space to GATr space.

    The adaptor FFN is implemented as Gated Linear Units (GLU).

    Parameters
    ----------
    config: ModelConfig
        Configuration for the entire generator model. See ``ModelConfig``
        for more details.
    """

    config: ModelConfig
    proj_gate: nn.Linear
    proj_next: nn.Linear
    activation: nn.GELU
    layer_norm: nn.RMSNorm

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

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

        # Gating mechanism
        size_inter = self.config.adpt_dim_intermediate
        hidden_gated = self.proj_gate(hidden)
        hidden_gated = (
            self.activation(hidden_gated[:, :, size_inter:])
            * hidden_gated[:, :, :size_inter]
        )

        # Projection and output
        hidden_gated = self.proj_next(hidden_gated)
        return hidden_gated + hidden_res_conn
