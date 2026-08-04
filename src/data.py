"""Memory-mapped token dataset + random-offset batch sampling.

Uses `np.memmap` rather than loading the .bin shards into RAM: the full train
shard can run into the hundreds of MB to low GB, and this machine has as
little as ~4 GB free at times. memmap means each `get_batch` call only pages
in the `block_size` windows it actually reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class BinDataset:
    """Random-offset sampler over data/tokenized/{split}.bin."""

    def __init__(self, tokenized_dir: str | Path, seed: int | None = None):
        self._dir = Path(tokenized_dir)
        with (self._dir / "meta.json").open("r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self._arrays: dict[str, np.memmap] = {}
        self._rng = np.random.default_rng(seed)

    def _array(self, split: str) -> np.memmap:
        if split not in self._arrays:
            path = self._dir / f"{split}.bin"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing — run `python -m data.tokenize` first")
            self._arrays[split] = np.memmap(path, dtype=np.uint16, mode="r")
        return self._arrays[split]

    def __len__(self) -> int:
        return sum(len(self._array(s)) for s in ("train", "val") if (self._dir / f"{s}.bin").exists())

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x, y) of shape (batch_size, block_size), y = x shifted by one."""
        data = self._array(split)
        if len(data) <= block_size:
            raise ValueError(
                f"{split} split has only {len(data)} tokens, need > block_size={block_size}"
            )

        ix = self._rng.integers(0, len(data) - block_size - 1, size=batch_size)
        # Cast up-front: uint16 -> int64. Embedding lookups need signed/wide
        # indices, and torch has no uint16 tensor dtype to hand off directly.
        x = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
        y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        if device.type == "cuda":
            x_t = x_t.pin_memory().to(device, non_blocking=True)
            y_t = y_t.pin_memory().to(device, non_blocking=True)
        else:
            x_t = x_t.to(device)
            y_t = y_t.to(device)
        return x_t, y_t
