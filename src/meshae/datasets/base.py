from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class BaseDatasetConfig:
    path: Path 


class BaseDataset(Dataset):
    def __init__(self, config: BaseDatasetConfig) -> None:
        super().__init__()

        self.config = config

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def load_one(self, path: Path):


