from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pytorch_accelerated.callbacks import SaveBestModelCallback, get_default_callbacks
from pytorch_accelerated.schedulers import CosineLrScheduler
from torch.utils.data import Subset

from meshae import MeshAEModel, MeshAEModelConfig
from meshae.dataset import MeshAECollateFn, MeshAEDataset
from meshae.trainer import MeshAECheckpointCallback, MeshAETrainer


def parse_cli_args() -> argparse.Namespace:
    r"""Build CLI interface for the training script.

    Most model specification and training run settings should be configured through
    the ``model.json`` and ``train.yml`` files. CLI accepts addition settings related
    to model checkpointing, logging, and debugging.
    """
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Training script for the mesh autoencoder.",
        epilog="Please raise an Issue in the GitHub repo for any question.",
        allow_abbrev=True,
    )
    parser.add_argument(
        "--model-config", "-m",
        type=str,
        required=True,
        help="Path to the model specifications, e.g., the hidden size.",
    )
    parser.add_argument(
        "--train-config", "-t",
        type=str,
        required=True,
        help="Path to the training specifications, e.g., the number of epochs.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the directory storing model checkpoints.",
    )
    parser.add_argument(
        "--ckpt-freq",
        type=int,
        default=-1,
        required=False,
        help="Checkpointing frequency in number of iterations.",
    )
    args = parser.parse_args()
    return args


def init_data_args(
    config: dict[Literal["dataset", "collate_fn"], Any],
) -> tuple[MeshAEDataset, MeshAEDataset | Subset]:
    r"""Initialize data arguments for the training loop.

    Data arguments (returned as dictionary) include:

    - Collate function
    - Training dataset
    - Eval dataset

    If the eval dataset is not configured, a random 8192 examples
    will be sampled from the training dataset, used as validation
    samples for checkpointing etc.

    Parameters
    ----------
    config : dict[Literal["dataset", "collate_fn"], Any]
        Model training specifications.

    Returns
    -------
    collate_fn : MeshAECollateFn
        The batch collation function configured based on training
        specifications.
    train_dataset : MeshAEDataset
        The training samples.
    eval_dataset : MeshAEDataset
        The evaluation samples.
    """
    ret = {
        "collate_fn": MeshAECollateFn(**config["collate_fn"]),
        "train_dataset": MeshAEDataset(**config["dataset"]["train"]),
    }
    if "eval" in config.keys():
        ret["eval_dataset"] = MeshAEDataset(**config["dataset"]["eval"])
        return ret

    indices = torch.randperm(len(ret["train_dataset"]))[:8192]
    ret["eval_dataset"] = Subset(ret["train_dataset"], indices)
    return ret


def init_random_model(model_config: str | Path) -> MeshAEModel:
    r"""Initialize a model with randomized weight given specifications.

    Model specification is expected to be stored in a JSON file. An error will
    be raised if the provided path does not exist.
    """
    model_config = Path(model_config)
    if not model_config.exists():
        msg = f"Model config at <{model_config}> not found; default model initialized."
        raise FileNotFoundError(msg)

    with model_config.open() as fin:
        config = MeshAEModelConfig.from_dict(json.load(fin))

    model = MeshAEModel(**config.to_dict())
    return model


def main():
    args = parse_cli_args()
    with open(args.train_config) as fin:
        train_config = yaml.safe_load(fin)

    model = init_random_model(args.model_config)
    optimizer = torch.optim.AdamW(model.parameters(), **train_config["optimizer"])
    scheduler = CosineLrScheduler.create_scheduler_fn(**train_config["scheduler"])

    ckpt_path = Path(args.ckpt_path)
    ckpt_callback_freq = MeshAECheckpointCallback(ckpt_path, args.ckpt_freq)
    ckpt_callback_best = SaveBestModelCallback(str(ckpt_path / "champion.pt"))

    trainer = MeshAETrainer(
        model,
        loss_func=None,
        optimizer=optimizer,
        callbacks=(
            ckpt_callback_freq,
            ckpt_callback_best,
            *get_default_callbacks(),
        ),
    )
    trainer.train(
        create_scheduler_fn=scheduler,
        **init_data_args(train_config),
        **train_config["train"],
    )


if __name__ == "__main__":
    main()
