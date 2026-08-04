"""Integration checks against the real M1 pipeline output.

Skipped automatically on a fresh clone or CI, since data/processed and
data/tokenized are gitignored (multi-GB derived artifacts, not source). Run
the pipeline first (`docs/COMMANDS.md`) to exercise these.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from data.clean import content_hash
from src.config import load_config

PROCESSED_DIR = Path("data/processed")
TOKENIZED_DIR = Path("data/tokenized")
CONFIG_PATH = "configs/micro_50m_8gb.yaml"

pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / "stats.json").exists() or not (TOKENIZED_DIR / "meta.json").exists(),
    reason="pipeline artifacts not present — run data/prepare.py, train_tokenizer.py, tokenize.py",
)


@pytest.fixture(scope="module")
def stats():
    with (PROCESSED_DIR / "stats.json").open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def meta():
    with (TOKENIZED_DIR / "meta.json").open(encoding="utf-8") as f:
        return json.load(f)


def test_train_token_count_matches_config_cap(meta):
    # The M1 exit criterion: if the real corpus doesn't clear the original
    # 400M target, the config cap is lowered to the measured value rather
    # than padded. This pins train.bin's real size to that measured cap.
    cfg = load_config(CONFIG_PATH)
    assert meta["splits"]["train"]["token_count"] == cfg.data.max_train_tokens


def test_no_cross_split_leakage(stats):
    # Non-zero here just means leakage was found *and removed* from val — the
    # real disjointness proof is test_val_disjoint_from_train_by_hash below.
    # This only checks prepare.py's own count is sane (not negative, exists).
    assert stats["validation"]["dropped_cross_split_leakage"] >= 0


@pytest.mark.slow
def test_val_disjoint_from_train_by_hash():
    """Recomputes content hashes from the actual written files — the real
    proof of disjointness, not just trusting the counter prepare.py logged."""
    train_hashes: set[str] = set()
    with (PROCESSED_DIR / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            train_hashes.add(content_hash(json.loads(line)["text"]))

    with (PROCESSED_DIR / "val.jsonl").open(encoding="utf-8") as f:
        for line in f:
            h = content_hash(json.loads(line)["text"])
            assert h not in train_hashes


def test_no_duplicate_hashes_within_train():
    seen: set[str] = set()
    with (PROCESSED_DIR / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            h = content_hash(json.loads(line)["text"])
            assert h not in seen, "exact-dup slipped past prepare.py's in-pass dedup"
            seen.add(h)


def test_token_ids_never_exceed_vocab_size(meta):
    vocab_size = meta["vocab_size"]
    for split in ("train", "val"):
        arr = np.memmap(TOKENIZED_DIR / f"{split}.bin", dtype=np.uint16, mode="r")
        # Sample rather than scan 357M tokens in a test: a strided sample
        # across the whole file still catches a systematic overflow, and
        # tokenize.py itself asserts on every batch at write time.
        sample = arr[::997]
        assert sample.max() < vocab_size


def test_first_document_round_trips_through_the_real_tokenizer(meta):
    tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer.json")
    eot_id = meta["eot_id"]

    with (PROCESSED_DIR / "train.jsonl").open(encoding="utf-8") as f:
        first_text = json.loads(f.readline())["text"]

    arr = np.memmap(TOKENIZED_DIR / "train.bin", dtype=np.uint16, mode="r")
    first_eot = int(np.argmax(arr == eot_id))  # index of the first EOT token
    first_doc_ids = arr[:first_eot].tolist()

    assert tokenizer.decode(first_doc_ids) == first_text


def test_char_count_kept_matches_actual_written_files(stats):
    """Regression test: char_count_kept was originally accumulated inside
    _clean_split, *before* cross-split leakage removal dropped ~6,300 val
    docs — so the stat kept counting characters from documents that were
    never actually written to val.jsonl. Verify the stat matches reality."""
    for split_name, filename in (("train", "train.jsonl"), ("validation", "val.jsonl")):
        total_chars = 0
        with (PROCESSED_DIR / filename).open(encoding="utf-8") as f:
            for line in f:
                total_chars += len(json.loads(line)["text"])
        assert stats[split_name]["char_count_kept"] == total_chars


def test_dedup_rate_within_sane_bounds(stats):
    # TinyStories is synthetic and genuinely repetitive (see docs/DATA_CARD.md)
    # — a high dup rate is expected, not a bug. This just catches the
    # pathological case (e.g. a cleaning bug collapsing everything to one hash).
    assert 0.0 < stats["train"]["exact_dup_rate"] < 0.5
