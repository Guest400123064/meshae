from __future__ import annotations

import argparse
import yaml
from typing import Any

import torch
from easydict import EasyDict
from meshae import MeshAEModel, MeshAEModelConfig
from meshae.dataset import MeshAEDataset, MeshAECollateFn
from meshae.trainer import MeshAETrainer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train mesh VQ-VAE.",
        epilog="Please raise an Issue for questions.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model and trainer config.",
    )
    parser.add_argument(
        "--data-train",
        type=str,
        required=True,
        help="Path to the directory containing training mesh objects.",
    )
    parser.add_argument(
        "--data-valid",
        type=str,
        required=False,
        default=None,
        help="Path to the directory containing validation mesh objects.",
    )
    return parser.parse_args()


def create_trainer(config: EasyDict[str, Any]) -> MeshAETrainer:

    model = MeshAEModel(**MeshAEModelConfig(**config.model).asdict())
    optimizer = torch.optim.AdamW(model.parameters(), **config.optimizer)

    trainer = MeshAETrainer(
        model,
        loss_func=None,
        optimizer=optimizer,
    )
    return trainer


def main() -> None:
    args = get_args()
    with open(args.config) as f:
        config = EasyDict(**yaml.safe_load(f))

    training_dataset = MeshAEDataset(args.data_train, sort_by=config.dataset.sort_by)
    eval_dataset = args.data_valid
    if isinstance(eval_dataset, str):
        eval_dataset = MeshAEDataset(eval_dataset, sort_by=config.dataset.sort_by)

    trainer = create_trainer(config)
    trainer.train(
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        collate_fn=MeshAECollateFn(**config.collate_fn),
        **config.train,
    )


if __name__ == "__main__":
    main()
