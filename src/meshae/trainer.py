from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import wandb
from pytorch_accelerated import Trainer
from pytorch_accelerated.callbacks import TrainerCallback

from meshae.utils import tensor_describe

if TYPE_CHECKING:
    from torchtyping import TensorType
    from wandb.sdk.wandb_run import Run

    from meshae.typing import MeshAEDatumKeyType


class MeshAETrainer(Trainer):
    r"""Trainer class for training management."""

    @property
    def current_step(self) -> int:
        r"""Number of times model parameters are updated **within an epoch**."""

        return self._current_step

    def train_epoch_start(self):
        r"""Initialize a iteration counter on epoch start.

        Note that the current step is different from the number of forward steps. Instead,
        it records the number of times the parameters are updated. Therefore, the counter
        should be updated only when ``optimizer_step`` is invoked. 
        """
        super().train_epoch_start()

        self._current_step = 0

    def optimizer_step(self) -> None:
        r"""Override the optimizer step to enable interim checkpointing.

        There are two reasons to override the default optimizer step:

        - Increment the iteration counter.
        - Save checkpoint accordingly.

        Note that checkpointing will be passed over to the scheduler step if the scheduler is
        available to make sure the optimizer and scheduler states are synchronized.
        """
        super().optimizer_step()

        self._current_step += 1
        if self.scheduler is None:
            self.callback_handler.call_event("save_checkpoint_on_invoke", self)

        g = self.model.decoder.refiner.layers[0][1].to_q.weight.grad.detach().norm(p=2)
        print(g)

    def scheduler_step(self):
        r"""Include a checkpointing step."""

        super().scheduler_step()

        self.callback_handler.call_event("save_checkpoint_on_invoke", self)

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
        loss, (recon_loss, commit_loss), *_ = self.model(**batch)
        batch_size = batch["faces"].size(0)

        print(f"Recon loss: {recon_loss:.2f}; commit loss: {commit_loss:.2f}")
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

    One may use this callback in conjunction with builtin ``SaveBestModelCallback`` to
    prevent training process from breaking in the middle.

    Parameters
    ----------
    save_path : str | Path
        The path to the **folder** to which checkpoints are saved. The file names are
        generated automatically based on the current epoch number and iteration number.
    checkpoint_frequency : int, default=1000
        Checkpointing frequency in number of iterations. If the frequency is lower than
        the maximum number of iterations within each epoch, no checkpoint is saved.
    save_optimizer : bool, default=True
        Whether to also save the optimizer as part of the model checkpoint.
    save_scheduler : bool, default=True
        Whether to also save the scheduler as part of the model checkpoint.
    """

    def __init__(
        self,
        save_path: str | Path,
        checkpoint_frequency : int = 1000,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
    ) -> None:
        self.checkpoint_frequency = checkpoint_frequency
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler

        self.save_path = Path(save_path)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)

    def make_checkpoint_full_path(self, trainer: MeshAETrainer) -> str:
        ce = trainer.run_history.current_epoch
        cs = trainer.current_step + 1
        fn = f"ckpt-{ce:02}-{cs:04}.pt"

        return str(self.save_path / fn)

    def save_checkpoint_on_invoke(self, trainer: MeshAETrainer) -> None:
        if (trainer.current_step + 1) % self.checkpoint_frequency == 0:
            trainer.save_checkpoint(
                save_path=self.make_checkpoint_full_path(trainer),
                save_optimizer=self.save_optimizer,
                save_scheduler=self.save_scheduler,
            )


class MeshAELoggerCallback(TrainerCallback):
    r"""This callback is used to log the metrics and key debug data to wandb.

    The ``pytorch-accelerated`` package default logging behavior is to log the metrics
    by the end of each epoch. This callback is used to log the metrics and key debug data,
    such as the gradient norm, at a specific frequency. All information is logged to wandb.

    Parameters
    ----------
    wandb_kwargs : dict[str, Any]
        Keyword arguments to pass to the ``wandb.init`` function.
    debug_parameters : list[str] | None, default=None
        The parameters to watch for key statistics as well as those of gradients'.
    debug_frequency : int, default=1000
        The frequency of parameter debugging frequencies in number of iterations.
        Note that the number of iterations is measured by the number of optimizer steps.
    """

    def __init__(
        self,
        wandb_kwargs: dict[str, Any],
        debug_parameters: list[str] | None = None,
        debug_frequency: int = 1000,
    ) -> None:
        self.wandb_kwargs = wandb_kwargs
        self.run = wandb.init(**self.wandb_kwargs)

        self.debug_parameters = debug_parameters or []
        self.debug_frequency = debug_frequency

    def on_training_run_end(self, trainer, **kwargs) -> None:
        self.run.finish()
