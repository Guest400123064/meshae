from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import trimesh
import torch
import numpy as np
from einops import rearrange
from scipy.spatial.transform import Rotation
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torchtyping import TensorType


class BaseDataset(Dataset):
    def __init__(
        self,
        path: Path,

    ) -> None:
        super().__init__()

        self.path = path

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def load_one(self, path: Path):
        pass
