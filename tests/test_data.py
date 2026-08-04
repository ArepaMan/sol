"""BinDataset / get_batch shape, shift, bounds, and determinism."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.data import BinDataset

VOCAB_SIZE = 50
N_TOKENS = 10_000


@pytest.fixture
def tokenized_dir(tmp_path):
    # Sequential-mod-vocab data: makes the shift assertion (y == x shifted by
    # one) trivial to check exactly, and everything stays < VOCAB_SIZE.
    arr = (np.arange(N_TOKENS, dtype=np.int64) % VOCAB_SIZE).astype(np.uint16)
    (tmp_path / "train.bin").write_bytes(arr.tobytes())
    (tmp_path / "val.bin").write_bytes(arr.tobytes())
    meta = {"vocab_size": VOCAB_SIZE, "dtype": "uint16", "eot_id": 0}
    with (tmp_path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)
    return tmp_path


def test_get_batch_shapes(tokenized_dir):
    ds = BinDataset(tokenized_dir, seed=0)
    x, y = ds.get_batch("train", batch_size=8, block_size=32, device=torch.device("cpu"))
    assert x.shape == (8, 32)
    assert y.shape == (8, 32)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_y_is_x_shifted_by_one(tokenized_dir):
    ds = BinDataset(tokenized_dir, seed=0)
    x, y = ds.get_batch("train", batch_size=4, block_size=16, device=torch.device("cpu"))
    # x[:, 1:] should equal y[:, :-1] since y is x's window shifted forward by one token.
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_no_index_exceeds_vocab_size(tokenized_dir):
    ds = BinDataset(tokenized_dir, seed=0)
    x, y = ds.get_batch("train", batch_size=8, block_size=32, device=torch.device("cpu"))
    assert x.max().item() < VOCAB_SIZE
    assert y.max().item() < VOCAB_SIZE


def test_rng_state_roundtrip_resumes_batch_sequence(tokenized_dir):
    """The property M4's checkpoint/resume gate depends on: saving and
    restoring get_rng_state()/set_rng_state() must continue the *same*
    pseudo-random batch sequence, not restart it."""
    ds = BinDataset(tokenized_dir, seed=0)
    ds.get_batch("train", batch_size=4, block_size=16, device=torch.device("cpu"))  # advance state
    saved_state = ds.get_rng_state()
    x_next_a, y_next_a = ds.get_batch("train", batch_size=4, block_size=16, device=torch.device("cpu"))

    ds2 = BinDataset(tokenized_dir, seed=999)  # different seed entirely
    ds2.set_rng_state(saved_state)
    x_next_b, y_next_b = ds2.get_batch("train", batch_size=4, block_size=16, device=torch.device("cpu"))

    assert torch.equal(x_next_a, x_next_b)
    assert torch.equal(y_next_a, y_next_b)


def test_determinism_under_fixed_seed(tokenized_dir):
    ds1 = BinDataset(tokenized_dir, seed=123)
    ds2 = BinDataset(tokenized_dir, seed=123)
    x1, y1 = ds1.get_batch("train", batch_size=8, block_size=32, device=torch.device("cpu"))
    x2, y2 = ds2.get_batch("train", batch_size=8, block_size=32, device=torch.device("cpu"))
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_different_seeds_give_different_batches(tokenized_dir):
    ds1 = BinDataset(tokenized_dir, seed=1)
    ds2 = BinDataset(tokenized_dir, seed=2)
    x1, _ = ds1.get_batch("train", batch_size=8, block_size=32, device=torch.device("cpu"))
    x2, _ = ds2.get_batch("train", batch_size=8, block_size=32, device=torch.device("cpu"))
    assert not torch.equal(x1, x2)


def test_raises_when_split_shorter_than_block_size(tokenized_dir):
    ds = BinDataset(tokenized_dir, seed=0)
    with pytest.raises(ValueError, match="block_size"):
        ds.get_batch("train", batch_size=2, block_size=N_TOKENS * 2, device=torch.device("cpu"))


def test_missing_split_file_raises(tokenized_dir):
    ds = BinDataset(tokenized_dir, seed=0)
    with pytest.raises(FileNotFoundError):
        ds.get_batch("test", batch_size=2, block_size=8, device=torch.device("cpu"))
