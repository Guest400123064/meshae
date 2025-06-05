from __future__ import annotations

import argparse
import yaml
from typing import Any

import torch
from easydict import EasyDict
from pytorch_accelerated.callbacks import get_default_callbacks, SaveBestModelCallback
from meshae import MeshAEModel, MeshAEModelConfig
from meshae.dataset import MeshAEDataset, MeshAECollateFn
from meshae.trainer import MeshAETrainer


def create_trainer(config: EasyDict[str, Any]) -> MeshAETrainer:

    model = MeshAEModel(**MeshAEModelConfig.from_dict(config.model).to_dict())
    optimizer = torch.optim.AdamW(model.parameters(), **config.optimizer)
    callbacks = [SaveBestModelCallback(**config.checkpoint)]
    callbacks.extend(get_default_callbacks())

    trainer = MeshAETrainer(
        model,
        loss_func=None,
        optimizer=optimizer,
        callbacks=callbacks,
    )
    return trainer


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the model and trainer config.")

    args = parser.parse_args()
    with open(args.config) as f:
        config = EasyDict(**yaml.safe_load(f))

    trainer = create_trainer(config)
    trainer.train(
        train_dataset=MeshAEDataset(**config.train_dataset),
        eval_dataset=MeshAEDataset(**config.eval_dataset),
        collate_fn=MeshAECollateFn(**config.collate_fn),
        **config.train,
    )


if __name__ == "__main__":
    main()
