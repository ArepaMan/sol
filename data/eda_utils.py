"""Pure functions behind data/eda.ipynb — kept separate and unit-tested so the
notebook itself stays thin (load, call, plot) rather than the usual home for
untested one-off analysis code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def char_lengths_from_jsonl(path: str | Path) -> np.ndarray:
    """One pass over a prepare.py output file; returns len(text) per doc, in order."""
    lengths = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lengths.append(len(json.loads(line)["text"]))
    return np.array(lengths, dtype=np.int64)


def token_lengths_from_bin(bin_path: str | Path, eot_id: int) -> np.ndarray:
    """Per-document token counts recovered from a tokenize.py .bin shard.

    Each document's ids are followed by exactly one eot_id (see
    data/tokenize.py), so document boundaries are exactly the eot positions —
    no need to re-tokenize to get the full, exact distribution.
    """
    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    eot_positions = np.flatnonzero(arr == eot_id)
    boundaries = np.concatenate(([-1], eot_positions))
    lengths = np.diff(boundaries) - 1
    return lengths


def chunked_token_frequency(
    bin_path: str | Path, vocab_size: int, chunk_size: int = 20_000_000
) -> np.ndarray:
    """Frequency of every token id across the full shard, without materialising
    the whole array as int64 at once (357M tokens x 8 bytes would be ~2.7GB —
    more than this machine reliably has free)."""
    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    counts = np.zeros(vocab_size, dtype=np.int64)
    for start in range(0, len(arr), chunk_size):
        block = np.asarray(arr[start : start + chunk_size], dtype=np.int64)
        counts += np.bincount(block, minlength=vocab_size)
    return counts


def percentiles(arr: np.ndarray, ps=(50, 90, 99)) -> dict[int, float]:
    return {p: float(np.percentile(arr, p)) for p in ps}


def coverage_at(arr: np.ndarray, threshold: int) -> float:
    """Fraction of values <= threshold — e.g. "what % of stories fit in block_size=512"."""
    return float(np.mean(arr <= threshold))
