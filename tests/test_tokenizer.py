"""Tokenizer training + round-trip correctness.

Byte-level BPE is lossless by construction (every byte maps to a printable
unicode surrogate before merging), so decode(encode(s)) == s should hold
exactly — including unicode, emoji, and irregular whitespace. That is the
property this file exists to pin down; a broken pre-tokenizer/decoder pairing
is the kind of bug that silently corrupts every downstream token.
"""

from __future__ import annotations

import json

import pytest

from data.train_tokenizer import SPECIAL_TOKENS, train_tokenizer

ROUND_TRIP_FIXTURES = [
    "Once upon a time, there was a little fox.",
    "Multiple   spaces   and\ttabs\tand\nnewlines.",
    "Emoji test: a fox ran 🦊 through the forest 🌲🌲🌲!",
    "Ünïcödé áccénts and çedillas.",
    "ALL CAPS SHOUTING and then whispering.",
    "Numbers 123, punctuation!! ...and, quotes \"like this\".",
    "",  # empty string is a legal edge case
    "   leading and trailing whitespace   ",
]


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    train_dir = tmp_path_factory.mktemp("tok_train")
    train_path = train_dir / "train.jsonl"
    # A larger, more repetitive corpus than the round-trip fixtures so BPE has
    # something to merge; vocab_size 300 covers the 256 base byte tokens + specials.
    corpus = [
        "Once upon a time there was a little fox who lived in the forest.",
        "The fox loved to play with his friends every single day.",
        "One day the fox found a shiny red ball near the old oak tree.",
        "Lily and Tom played together in the sunny green park all afternoon.",
    ] * 50
    with train_path.open("w", encoding="utf-8") as f:
        for text in corpus:
            f.write(json.dumps({"text": text}) + "\n")
    return train_tokenizer(train_path, vocab_size=300)


@pytest.mark.parametrize("text", ROUND_TRIP_FIXTURES)
def test_round_trip(tiny_tokenizer, text):
    encoded = tiny_tokenizer.encode(text)
    decoded = tiny_tokenizer.decode(encoded.ids)
    assert decoded == text


def test_special_tokens_present_with_stable_ids(tiny_tokenizer):
    ids = [tiny_tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS]
    assert all(i is not None for i in ids)
    assert len(set(ids)) == len(ids)  # all distinct


def test_vocab_size_respects_ceiling(tiny_tokenizer):
    # BPE trainer treats vocab_size as an upper bound, not a guarantee — a
    # tiny/repetitive corpus can plateau below it. Never exceed it.
    assert tiny_tokenizer.get_vocab_size() <= 300
