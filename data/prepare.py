"""Download TinyStories, clean, dedup, and write data/processed/{train,val}.jsonl + stats.json.

Usage:
    python -m data.prepare --config configs/micro_50m_8gb.yaml
    python -m data.prepare --config configs/micro_50m_8gb.yaml --near-dedup

Uses TinyStories' own train/validation split (2,119,719 / 21,990 stories) rather
than carving a fresh split from one pool — it was already held out by the
dataset's original authors. The dedup pass still checks for cross-split
leakage and removes any validation story whose exact-dedup hash also appears
in train.

Streams the HF `datasets` Arrow table row by row rather than materializing it
as a Python list — this machine can have as little as ~4 GB free RAM during a
session, and 2.1M fully-materialized dicts would not be a good use of it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from data.clean import MAX_CHARS, MIN_CHARS, clean_text, content_hash
from data.near_dedup import find_near_duplicates
from src.config import load_config

_DATASET_IDS = {"tinystories": "roneneldan/TinyStories"}


@dataclass
class SplitStats:
    raw_count: int = 0
    dropped_too_short: int = 0
    dropped_too_long: int = 0
    dropped_exact_dup: int = 0
    dropped_near_dup: int = 0
    dropped_cross_split_leakage: int = 0
    kept: int = 0
    char_count_kept: int = 0

    def as_dict(self) -> dict:
        return {**self.__dict__, "exact_dup_rate": self._rate(self.dropped_exact_dup)}

    def _rate(self, n: int) -> float:
        return round(n / self.raw_count, 4) if self.raw_count else 0.0


def _clean_split(
    rows,
    split_name: str,
    min_chars: int,
    max_chars: int,
    seen_hashes: set[str],
) -> tuple[list[str], list[str], SplitStats]:
    """Returns (kept_texts, kept_hashes, stats). One exact-dedup pass, in order."""
    stats = SplitStats()
    texts: list[str] = []
    hashes: list[str] = []

    for row in tqdm(rows, desc=f"clean[{split_name}]", unit="doc"):
        stats.raw_count += 1
        text = clean_text(row["text"])

        if len(text) < min_chars:
            stats.dropped_too_short += 1
            continue
        if len(text) > max_chars:
            stats.dropped_too_long += 1
            continue

        h = content_hash(text)
        if h in seen_hashes:
            stats.dropped_exact_dup += 1
            continue

        seen_hashes.add(h)
        texts.append(text)
        hashes.append(h)
        stats.kept += 1
        stats.char_count_kept += len(text)

    return texts, hashes, stats


def _write_jsonl(path: Path, split_name: str, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            f.write(json.dumps({"id": f"{split_name}-{i}", "text": text}, ensure_ascii=False))
            f.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/micro_50m_8gb.yaml")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument(
        "--near-dedup",
        action="store_true",
        help="Also run MinHash near-dedup. O(n) with a real constant factor — "
        "expect this to dominate runtime on the full train split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap rows read per split, for a fast dry run.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    dataset_id = _DATASET_IDS.get(cfg.data.dataset)
    if dataset_id is None:
        sys.exit(f"Unknown dataset '{cfg.data.dataset}' — known: {list(_DATASET_IDS)}")

    from datasets import load_dataset  # deferred: slow import, only needed here

    print(f"Loading {dataset_id} ...")
    ds = load_dataset(dataset_id)

    train_rows = ds["train"] if args.limit is None else ds["train"].select(range(args.limit))
    val_rows = (
        ds["validation"] if args.limit is None else ds["validation"].select(range(min(args.limit, len(ds["validation"]))))
    )

    seen_hashes: set[str] = set()
    train_texts, train_hashes, train_stats = _clean_split(
        train_rows, "train", args.min_chars, args.max_chars, seen_hashes
    )
    val_texts, val_hashes, val_stats = _clean_split(
        val_rows, "validation", args.min_chars, args.max_chars, seen_hashes
    )

    # Cross-split leakage: a val story whose hash also landed in train (added to
    # seen_hashes first, since train was processed first) was already dropped as
    # an "exact dup" above. Report that count under its real name instead of
    # burying it in the generic exact-dup bucket.
    train_hash_set = set(train_hashes)
    leaked = 0
    keep_val_texts: list[str] = []
    for text, h in zip(val_texts, val_hashes):
        if h in train_hash_set:
            leaked += 1
            continue
        keep_val_texts.append(text)
    val_texts = keep_val_texts
    val_stats.dropped_cross_split_leakage = leaked
    val_stats.kept -= leaked

    train_dup_rate = train_stats._rate(train_stats.dropped_exact_dup)
    if train_dup_rate > 0.15:
        print(
            f"WARNING: train exact-dup rate {train_dup_rate:.1%} > 15% — "
            "consider whether near-dedup threshold needs raising. See docs/DATA_CARD.md."
        )

    if args.near_dedup:
        print("Running near-dedup on validation (small, cheap) ...")
        t0 = time.time()
        drop_idx = find_near_duplicates(val_texts)
        val_texts = [t for i, t in enumerate(val_texts) if i not in drop_idx]
        val_stats.dropped_near_dup = len(drop_idx)
        val_stats.kept -= len(drop_idx)
        print(f"  dropped {len(drop_idx)} in {time.time() - t0:.1f}s")

        print("Running near-dedup on train (this is the expensive part) ...")
        t0 = time.time()
        drop_idx = find_near_duplicates(train_texts)
        train_texts = [t for i, t in enumerate(train_texts) if i not in drop_idx]
        train_stats.dropped_near_dup = len(drop_idx)
        train_stats.kept -= len(drop_idx)
        print(f"  dropped {len(drop_idx)} in {time.time() - t0:.1f}s")

    out_dir = Path(args.out_dir)
    _write_jsonl(out_dir / "train.jsonl", "train", train_texts)
    _write_jsonl(out_dir / "val.jsonl", "validation", val_texts)

    stats = {
        "dataset": dataset_id,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "near_dedup_enabled": args.near_dedup,
        "seed": cfg.seed,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train": train_stats.as_dict(),
        "validation": val_stats.as_dict(),
        "notes": (
            "Uses TinyStories' native train/validation split rather than a "
            "custom val_split carve-out (see docs/PROJECT.md 'Resolved decisions'). "
            "config.data.val_split is informational only for this dataset."
        ),
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\ntrain: {train_stats.kept:,} kept / {train_stats.raw_count:,} raw")
    print(f"val:   {val_stats.kept:,} kept / {val_stats.raw_count:,} raw "
          f"({leaked:,} dropped for cross-split leakage)")
    print(f"wrote {out_dir}/{{train,val}}.jsonl + stats.json")


if __name__ == "__main__":
    main()
