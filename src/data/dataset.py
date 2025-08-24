
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

class StockSequenceDataset(Dataset):
    """Turn 2D feature matrix into (C=1, T=window, F) sequences and targets."""
    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int, stride: int = 1):
        assert X.ndim == 2, "X must be 2D: (N, F)"
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.window_size = window_size
        self.stride = stride
        self.indices = self._build_indices(len(X), window_size, stride)

    def _build_indices(self, n: int, w: int, s: int):
        idx = []
        i = w - 1
        while i < n:
            idx.append(i)
            i += s
        return idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = self.indices[i]
        start = end - self.window_size + 1
        seq = self.X[start:end+1]  # (T, F)
        seq = seq[None, :, :]      # (C=1, T, F)
        target = self.y[end]
        return torch.from_numpy(seq), torch.tensor(target, dtype=torch.long)
