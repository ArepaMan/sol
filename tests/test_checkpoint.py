"""Checkpoint save/load round-trip tests.

The GPU test here is a real regression test: `torch.load(..., map_location=device)`
moves *every* tensor in the checkpoint onto that device, including the RNG
state byte tensor — which `torch.set_rng_state` rejects unless it's on CPU.
This was caught for real during Gate 3 (M4), not hypothesized in advance.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.config import ModelConfig, SolConfig
from src.data import BinDataset
from src.model import GPT
from src.train import load_checkpoint, save_checkpoint

TINY_MODEL_CONFIG = ModelConfig(
    n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=97, dropout=0.0, bias=False
)


@pytest.fixture
def tokenized_dir(tmp_path):
    arr = (np.arange(2000, dtype=np.int64) % 97).astype(np.uint16)
    (tmp_path / "train.bin").write_bytes(arr.tobytes())
    (tmp_path / "val.bin").write_bytes(arr.tobytes())
    meta = {"vocab_size": 97, "dtype": "uint16", "eot_id": 0}
    with (tmp_path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)
    return tmp_path


def _build(device, tokenized_dir):
    dataset = BinDataset(tokenized_dir, seed=0)
    model = GPT(TINY_MODEL_CONFIG).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95), device_type=device.type
    )
    return dataset, model, optimizer


def _round_trip(device, tokenized_dir, tmp_path):
    cfg = SolConfig(model=TINY_MODEL_CONFIG)
    dataset, model, optimizer = _build(device, tokenized_dir)

    # Advance state a bit so there's something to actually verify continuity of.
    x, y = dataset.get_batch("train", 2, 16, device)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model, optimizer, iter_num=5, best_val_loss=1.23, cfg=cfg, dataset=dataset)

    dataset2, model2, optimizer2 = _build(device, tokenized_dir)
    iter_num, best_val_loss = load_checkpoint(ckpt_path, model2, optimizer2, dataset2, device)

    assert iter_num == 5
    assert best_val_loss == pytest.approx(1.23)
    # RNG state round-tripped correctly means the next batch drawn from the
    # restored dataset matches what the original would have drawn next.
    x_orig, _ = dataset.get_batch("train", 2, 16, device)
    x_restored, _ = dataset2.get_batch("train", 2, 16, device)
    assert torch.equal(x_orig, x_restored)


def test_checkpoint_round_trip_cpu(tokenized_dir, tmp_path):
    _round_trip(torch.device("cpu"), tokenized_dir, tmp_path)


@pytest.mark.gpu
def test_checkpoint_round_trip_cuda(tokenized_dir, tmp_path):
    # This is the case that broke: map_location=<cuda device> in torch.load
    # moves the RNG state tensor onto CUDA, and torch.set_rng_state rejects
    # anything that isn't a CPU ByteTensor.
    _round_trip(torch.device("cuda"), tokenized_dir, tmp_path)
