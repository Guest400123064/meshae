from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from pytorch_accelerated import Trainer

if TYPE_CHECKING:
    from torchtyping import TensorType

    from meshae.typing import MeshAEDatumKeyType


def batch_to(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


class MeshAETrainer(Trainer):
    r"""
    """

    def calculate_train_batch_loss(
        self,
        batch: dict[MeshAEDatumKeyType, TensorType],
    ) -> dict[str, Any]:
        r"""
        """
        loss, *_ = self.model(**batch)
        batch_size = batch["faces"].size(0)

        return {"loss": loss, "batch_size": batch_size}

    def calculate_eval_batch_loss(
        self,
        batch: dict[MeshAEDatumKeyType, TensorType],
    ) -> dict[str, Any]:
        r"""
        """
        mode = self.model.training
        self.model.eval()

        with torch.no_grad():
            loss, *_ = self.model(**batch)
            batch_size = batch["faces"].size(0)

        self.model.train(mode)
        return {"loss": loss, "batch_size": batch_size}
