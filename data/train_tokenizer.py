"""Train a byte-level BPE tokenizer on data/processed/train.jsonl only.

Never trains on validation — training a tokenizer on data you will later
evaluate on is a real (if subtle) form of leakage: the tokenizer's merge
table would then reflect the eval set's exact word forms, and its
vocabulary/compression on val would look better than it should.

Usage:
    python -m data.train_tokenizer --vocab-size 32000 --input data/processed/train.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|unk|>"]


def _iter_texts(jsonl_path: Path, sample_size: int | None):
    with jsonl_path.open("r", encoding="utf-8") as f:
        lines = f if sample_size is None else itertools.islice(f, sample_size)
        for line in lines:
            yield json.loads(line)["text"]


def train_tokenizer(
    input_path: Path,
    vocab_size: int,
    sample_size: int | None = None,
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        # Without this, a byte that never appears in the (sub)corpus — e.g. a
        # digit, an uppercase letter, an emoji's constituent bytes — has no
        # single-byte token to fall back to and gets silently swallowed by
        # <|unk|> at decode time. Byte-level BPE is only lossless if every one
        # of the 256 byte-level symbols is guaranteed a vocab slot up front.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(_iter_texts(input_path, sample_size), trainer=trainer)
    return tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/train.jsonl")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Cap number of docs used for training (default: all of train.jsonl).",
    )
    parser.add_argument("--out-dir", default="data/tokenizer")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if "val" in input_path.name:
        raise SystemExit(f"refusing to train the tokenizer on {input_path} — use train.jsonl")

    tokenizer = train_tokenizer(input_path, args.vocab_size, args.sample_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / "tokenizer.json"))

    actual_vocab = tokenizer.get_vocab_size()
    print(f"vocab size: {actual_vocab} (requested {args.vocab_size})")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok!r} -> id {tokenizer.token_to_id(tok)}")
    print(f"saved to {out_dir / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
