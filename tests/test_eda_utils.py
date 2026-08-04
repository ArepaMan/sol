"""Correctness of the EDA notebook's data-crunching helpers on small synthetic
fixtures — the notebook itself only calls these and plots, so this is where
the actual logic gets checked."""

from __future__ import annotations

import json

import numpy as np
import pytest

from data.eda_utils import (
    char_lengths_from_jsonl,
    chunked_token_frequency,
    coverage_at,
    percentiles,
    token_lengths_from_bin,
)


def test_char_lengths_from_jsonl(tmp_path):
    path = tmp_path / "docs.jsonl"
    texts = ["short", "a bit longer than that", "x"]
    with path.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")
    lengths = char_lengths_from_jsonl(path)
    assert lengths.tolist() == [len(t) for t in texts]


def test_token_lengths_from_bin_recovers_doc_boundaries(tmp_path):
    eot = 0
    # Three "documents" of token lengths 3, 1, 5, each EOT-terminated.
    docs = [[7, 8, 9], [4], [1, 2, 3, 4, 5]]
    flat = []
    for d in docs:
        flat.extend(d)
        flat.append(eot)
    arr = np.array(flat, dtype=np.uint16)
    path = tmp_path / "shard.bin"
    path.write_bytes(arr.tobytes())

    lengths = token_lengths_from_bin(path, eot_id=eot)
    assert lengths.tolist() == [3, 1, 5]


def test_token_lengths_handles_single_document(tmp_path):
    arr = np.array([5, 6, 7, 0], dtype=np.uint16)  # one doc, then EOT
    path = tmp_path / "shard.bin"
    path.write_bytes(arr.tobytes())
    assert token_lengths_from_bin(path, eot_id=0).tolist() == [3]


def test_chunked_token_frequency_matches_full_bincount(tmp_path):
    rng = np.random.default_rng(0)
    vocab_size = 50
    data = rng.integers(0, vocab_size, size=10_000).astype(np.uint16)
    path = tmp_path / "shard.bin"
    path.write_bytes(data.tobytes())

    expected = np.bincount(data.astype(np.int64), minlength=vocab_size)
    # Deliberately small chunk_size to force multiple chunks in this test.
    actual = chunked_token_frequency(path, vocab_size, chunk_size=777)

    assert np.array_equal(actual, expected)
    assert actual.sum() == 10_000


def test_percentiles():
    arr = np.arange(1, 101)  # 1..100
    p = percentiles(arr, ps=(50, 90, 99))
    assert p[50] == pytest.approx(50.5)
    assert p[90] == pytest.approx(90.1)
    assert p[99] == pytest.approx(99.01)


def test_coverage_at():
    arr = np.array([100, 200, 300, 400, 500])
    assert coverage_at(arr, 300) == 0.6
    assert coverage_at(arr, 1000) == 1.0
    assert coverage_at(arr, 0) == 0.0
