from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import numpy as np
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

    Parameters
    ----------
    save_path : str | Path
        The path to the folder to which checkpoints are saved.
    watch_metric : str, default="eval_loss_epoch"
        The metric used to compare model performance. This should be accessible from the
        trainer's run history.
    checkpoint_frequency : int, default=1000
        Checkpointing frequency in number of iterations. Note that a checkpoint will always
        be saved at the end of each epoch.
    greater_is_better : bool, default=False
        Whether an increase in the ``watch_metric`` should be interpreted as the model
        performing better.
    reset_on_train : bool, default=True
        Whether to reset the best metric on subsequent training runs. If ``True``, only
        the metrics observed during the current training run will be compared.
    save_optimizer : bool, default=True
        Whether to also save the optimizer as part of the model checkpoint.
    save_scheduler : bool, default=True
        Whether to also save the scheduler as part of the model checkpoint.
    load_saved_checkpoint : bool, default=True,
        Whether to load the saved checkpoint at the end of the training run.
    """

    def __init__(
        self,
        save_path: str | Path,
        watch_metric: str = "eval_loss_epoch",
        checkpoint_frequency : int = 1000,
        greater_is_better: bool = False,
        reset_on_train: bool = True,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        load_saved_checkpoint: bool = True,
    ) -> None:
        self.watch_metric = watch_metric
        self.checkpoint_frequency = checkpoint_frequency
        self.greater_is_better = greater_is_better
        self.operator = np.greater if self.greater_is_better else np.less
        self.best_metric = None
        self.best_metric_epoch = None
        self.save_path = save_path
        self.save_file = None
        self.reset_on_train = reset_on_train
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.load_saved_checkpoint = load_saved_checkpoint

    def on_training_run_start(self, trainer, **kwargs):
        if self.reset_on_train:
            self.best_metric = None

    def on_training_run_epoch_end(self, trainer, **kwargs):
        if self.load_saved_checkpoint:
            trainer.print(
                f"Loading checkpoint with {self.watch_metric}: {self.best_metric} "
                f"from epoch {self.best_metric_epoch}"
            )
            trainer.load_checkpoint(self.save_file)


class MeshAELoggerCallback(LogMetricsCallback):
    r""""""
