from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from pytorch_accelerated import Trainer

if TYPE_CHECKING:
    from torchtyping import TensorType

    from meshae.typing import MeshAEDatumKeyType


class MeshAETrainer(Trainer):
    r"""Trainer class for training management."""

    def calculate_train_batch_loss(
        self,
        batch: dict[MeshAEDatumKeyType, TensorType],
    ) -> dict[str, Any]:
        r"""Custom forward pass and loss calculation.

        We need to customize the forward pass because the loss calculation is embedded into the
        model definition.

        Parameters
        ----------
        batch : dict[MeshAEDatumKeyType, TensorType]
            Batch of auto-encoder inputs collated into a single dictionary.

        Returns
        -------
        loss : TensorType[(), float]
            Combined loss (reconstruction loss and VQ-VQE commit loss) for auto-encoder training.
        batch_size : int
            Batch size.
        """
        loss, *_ = self.model(**batch)
        batch_size = batch["faces"].size(0)

        return {"loss": loss, "batch_size": batch_size}

    def calculate_eval_batch_loss(
        self,
        batch: dict[MeshAEDatumKeyType, TensorType],
    ) -> dict[str, Any]:
        r"""Custom forward pass and loss calculation.

        We need to customize the forward pass because the loss calculation is embedded into the
        model definition. Further, for evaluation runs, we make sure that the model is in the
        eval mode.

        Parameters
        ----------
        batch : dict[MeshAEDatumKeyType, TensorType]
            Batch of auto-encoder inputs collated into a single dictionary.

        Returns
        -------
        loss : TensorType[(), float]
            Combined loss (reconstruction loss and VQ-VQE commit loss) for auto-encoder training.
        batch_size : int
            Batch size.
        """
        mode = self.model.training
        self.model.eval()

        with torch.no_grad():
            loss, *_ = self.model(**batch)
            batch_size = batch["faces"].size(0)

        self.model.train(mode)
        return {"loss": loss, "batch_size": batch_size}
