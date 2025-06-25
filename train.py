from __future__ import annotations

import argparse
import json
import yaml
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Subset
from pytorch_accelerated.callbacks import get_default_callbacks

from meshae import MeshAEModel, MeshAEModelConfig
from meshae.dataset import MeshAEDataset, MeshAECollateFn
from meshae.trainer import MeshAETrainer, MeshAECheckpointCallback


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
        "--ckpt-path", "-cp",
        type=str,
        required=True,
        help="Path to the directory storing model checkpoints.",
    )
    parser.add_argument(
        "--ckpt-freq", "-cf",
        type=int,
        default=-1,
        required=False,
        help="",
    )
    args = parser.parse_args()
    return args


def init_data_args(
    config: dict[Literal["dataset", "collate_fn"], Any],
) -> tuple[MeshAEDataset, MeshAEDataset | Subset]:
    r"""Initialize datasets for training and validation.

    If validation configuration is not provided, a random subsample of 2048 training samples
    will be used as the validation set for evaluation and checkpointing.
    """
    ds_train = MeshAEDataset(**config["train"])
    if "eval" in config.keys():
        ds_eval = MeshAEDataset(**config["eval"])
        return ds_train, ds_eval

    indices = torch.randperm(len(ds_train))[:2048]
    ds_eval = Subset(ds_train, indices)
    return ds_train, ds_eval


def init_random_model(model_config: str | Path) -> MeshAEModel:
    r"""Initialize a model with randomized weight given model specifications.

    An error will be raised if the provided model config file does not exists.
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
    r""""""

    args = parse_cli_args()
    with open(args.train_config) as fin:
        train_config = yaml.safe_load(fin)

    model = init_random_model(args.model_config)
    optimizer = torch.optim.AdamW(model.parameters(), **train_config["optimizer"])

    trainer = MeshAETrainer(
        model,
        loss_func=None,
        optimizer=optimizer,
        callbacks=(
            MeshAECheckpointCallback(
                args
            ),
            *get_default_callbacks(),
        ),
    )
    trainer.train(
        create_scheduler_fn=None,
        **init_data_args(train_config),
        **train_config["train"],
    )


if __name__ == "__main__":
    main()
