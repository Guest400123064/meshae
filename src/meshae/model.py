from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
from x_transformers import Encoder


class MeshAEEncoder(nn.Module):
    pass


class MeshAEDecoder(nn.Module):
    pass


class MeshAEModel(nn.Module):
    r""""""

    def forward(self,) -> TensorType[(), float]:
        pass

    @torch.no_grad
    def encode(self):
        pass

    @torch.no_grad
    def decode(self):
        pass
