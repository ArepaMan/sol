"""Encode data/processed/{train,val}.jsonl to uint16 binary token shards.

Usage:
    python -m data.tokenize --workers 8

Writes data/tokenized/{train,val}.bin (raw uint16, EOT appended after every
document) and data/tokenized/meta.json (token counts, dtype, tokenizer hash).

vocab_size (32000) fits comfortably under 65535, so uint16 is safe — but every
batch is asserted against vocab_size before it is written, because a silent
overflow here would corrupt training data in a way nothing downstream would
catch until loss curves looked wrong for no visible reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Must happen before `from tokenizers import Tokenizer` — the library reads
# this env var once, at import time, to size its internal (rayon) thread pool.
if "--workers" in sys.argv:
    _idx = sys.argv.index("--workers")
    os.environ.setdefault("RAYON_NUM_THREADS", sys.argv[_idx + 1])

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

BATCH_DOCS = 10_000


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_batches(jsonl_path: Path, batch_size: int):
    batch = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            batch.append(json.loads(line)["text"])
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def encode_split(
    jsonl_path: Path,
    tokenizer: Tokenizer,
    eot_id: int,
    vocab_size: int,
    out_path: Path,
    split_name: str,
) -> dict:
    doc_count = 0
    token_count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as out_f:
        for batch in tqdm(_iter_batches(jsonl_path, BATCH_DOCS), desc=f"tokenize[{split_name}]"):
            encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
            ids_list = []
            for enc in encodings:
                ids_list.append(np.array(enc.ids, dtype=np.uint32))
                ids_list.append(np.array([eot_id], dtype=np.uint32))
            arr = np.concatenate(ids_list)

            if arr.max(initial=0) >= vocab_size:
                raise ValueError(
                    f"token id {arr.max()} >= vocab_size {vocab_size} in {split_name} — "
                    "tokenizer/config vocab_size mismatch, refusing to write uint16"
                )

            arr = arr.astype(np.uint16)
            out_f.write(arr.tobytes())
            doc_count += len(batch)
            token_count += len(arr)

    return {"doc_count": doc_count, "token_count": token_count, "path": str(out_path)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--out-dir", default="data/tokenized")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args(argv)

    tokenizer_path = Path(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    if eot_id is None:
        raise SystemExit("tokenizer has no <|endoftext|> special token")
    vocab_size = tokenizer.get_vocab_size()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)

    meta = {
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "eot_id": eot_id,
        "pad_id": tokenizer.token_to_id("<|pad|>"),
        "unk_id": tokenizer.token_to_id("<|unk|>"),
        "tokenizer_sha256": _sha256_file(tokenizer_path),
        "splits": {},
    }

    for split_name, filename in (("train", "train.jsonl"), ("val", "val.jsonl")):
        jsonl_path = processed_dir / filename
        if not jsonl_path.exists():
            raise SystemExit(f"missing {jsonl_path} — run data/prepare.py first")
        result = encode_split(
            jsonl_path, tokenizer, eot_id, vocab_size, out_dir / f"{split_name}.bin", split_name
        )
        meta["splits"][split_name] = result
        print(f"{split_name}: {result['doc_count']:,} docs -> {result['token_count']:,} tokens")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_dir}/meta.json")


if __name__ == "__main__":
    main()
