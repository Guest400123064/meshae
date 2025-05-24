from __future__ import annotations

import argparse

import numpy as np
import torch

from meshae import MeshAEFeatEmbedConfig, MeshAEModel
from meshae.dataset import MeshAEDataset, MeshAECollateFn
from meshae.trainer import MeshAETrainer


def get_args() -> argparse.Namespace:
    pass


def main(args):
    collate_fn = MeshAECollateFn()

    ds_train = MeshAEDataset("data/objaverse/train/")
    ds_valid = MeshAEDataset("data/objaverse/valid/")

    trainer = MeshAETrainer(
        model=model, loss_func=None, optimizer=optimizer,
    )
    trainer.train(
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        num_epochs=2,
        per_device_batch_size=8,
        collate_fn=collate_fn,
    )


feat_configs = {
    "vrtx": MeshAEFeatEmbedConfig(high_low=(0.5, -0.5)),
    "acos": MeshAEFeatEmbedConfig(high_low=(np.pi, 0.0)),
    "norm": MeshAEFeatEmbedConfig(high_low=(1.0, -1.0), num_bins=512),
    "area": MeshAEFeatEmbedConfig(high_low=(1.0, 0.0)),
}
model = MeshAEModel(
    feat_configs,
    num_sageconv_layers=1,
    num_quantizers=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
    num_refiner_layers=2,
    bin_smooth_blur_sigma=0.0,
).to("cuda:0")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)


if __name__ == "__main__":
    main()
