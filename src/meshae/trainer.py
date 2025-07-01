from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from pytorch_accelerated import Trainer
from pytorch_accelerated.callbacks import TrainerCallback, LogMetricsCallback

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


class MeshAECheckpointCallback(TrainerCallback):
    r"""Save a more comprehensive checkpoint on a more granular basis.

    The ``pytorch_accelerated`` package comes with ``SaveBestModelCallback`` callback
    that can handle simpler checkpointing behavior. Specifically, it saves a ``best.pt``
    file after the end of each training epoch based on the evaluation loss (or any other
    specified metric value). This class aim at a more flexible checkpointing behavior
    enabling:

    - Granular checkpointing frequency: when training the model with a large dataset, we
    may only train the model for one epoch. So, we may want to set a frequency based on
    number of iterations.
    - More informative checkpoint name: instead of having a fixed binary file one would
    want to have the naming reflect the number of epochs and number of completed iters.
    """

    def __init__(
        self,
        save_path: str | Path,
        watch_metric: str = "eval_loss_epoch",
        greater_is_better: bool = False,
        reset_on_train: bool = True,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        load_saved_checkpoint: bool = True,
    ) -> None:
        pass


class MeshAELoggerCallback(LogMetricsCallback):
    r""""""
