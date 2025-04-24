from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
import torch.nn.functional as F
from ezgatr.interfaces import point

if TYPE_CHECKING:
    from factok.geometric import FaceTokenModel


class FaceTokenLoss(nn.Module):
    def __init__(self, model: FaceTokenModel, beta: float = 1.0):
        super().__init__()

        self.model = model
        self.beta = beta

    def forward(self, f, r=None):
        p, l = self.model(f, r)
        return (
            F.mse_loss(point.decode(f), point.decode(p)) + l
        )
