#!/usr/bin/env python3
"""Dataset utilities for DynaPath LLM experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DynaPathNPYDataset(Dataset):
    """Load processed path-token arrays produced by prepare scripts."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        target_scale: float = 1.0,
    ):
        data_dir = Path(data_dir)
        self.road_ids = np.load(data_dir / "data_road.npy").astype(np.int64)
        self.dynamic_x = np.load(data_dir / "dynamic_path.npy").astype(np.float32)
        self.row_num = np.load(data_dir / "row_num.npy").astype(np.int64)
        self.targets = np.load(data_dir / "trip_time.npy").astype(np.float32) / target_scale
        self.departure_time = np.load(data_dir / "departure_time.npy").astype(np.int64)
        self.pad_id = int(np.load(data_dir / "pad_id.npy")[0])
        self.road_size = int(np.load(data_dir / "road_size.npy")[0]) + 1

        order = np.argsort(self.departure_time)
        n = len(order)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        if split == "train":
            self.indices = order[:n_train]
        elif split == "val":
            self.indices = order[n_train : n_train + n_val]
        elif split == "test":
            self.indices = order[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = self.indices[item]
        return {
            "road_ids": torch.from_numpy(self.road_ids[idx]),
            "dynamic_x": torch.from_numpy(self.dynamic_x[idx]),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "valid_len": torch.tensor(self.row_num[idx], dtype=torch.long),
        }


def dynapath_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Crop each batch to its longest valid path and build an attention mask."""

    max_len = int(max(item["valid_len"].item() for item in batch))
    road_ids = torch.stack([item["road_ids"][:max_len] for item in batch])
    dynamic_x = torch.stack([item["dynamic_x"][:max_len] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    valid_len = torch.stack([item["valid_len"] for item in batch])
    arange = torch.arange(max_len).unsqueeze(0)
    attention_mask = arange < valid_len.unsqueeze(1)
    return {
        "road_ids": road_ids,
        "dynamic_x": dynamic_x,
        "targets": targets,
        "valid_len": valid_len,
        "attention_mask": attention_mask,
    }
