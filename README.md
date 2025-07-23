# Annotated Mesh Auto-Encoder

This project implements, refactors, and annotates the mesh auto-encoder architecture from PivotMesh.

## Installation

```bash
poetry install
```

## Data

The data is stored in the `data` directory. The data is organized as follows:

```
data/
├── objaverse/
│   ├── train/
│   └── eval/
```

## Quick Start

Before staring the training, you need to:

1. Download the data from the Objaverse dataset and place it in the `data` directory.
2. Login to Weights and Biases through CLI via `wandb login`.

After that, you can start the training with the following command as an example:

```bash
accelerate launch --config-file configs/default/accelerate.yml train.py \
    --model-config configs/default/config.json \
    --train-config configs/default/train.yml \
    --checkpoint-path checkpoints/default \
    --checkpoint-frequency 100 \
    --debug-parameters encoder.proj_latent.weight decoder.proj_refine.0.weight \
    --debug-frequency 5
```
