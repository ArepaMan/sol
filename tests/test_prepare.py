"""Unit tests for the cleaning and near-dedup building blocks in data/prepare.py."""

from __future__ import annotations

from data.clean import clean_text, content_hash, is_acceptable_length
from data.near_dedup import find_near_duplicates
from data.prepare import _clean_split, _remove_cross_split_leakage


def test_clean_text_strips_control_chars_keeps_newline():
    dirty = "Once\x00 upon\x07 a time\nthere was a fox."
    cleaned = clean_text(dirty)
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert "\n" in cleaned


def test_clean_text_normalizes_tab_to_space():
    # \t is exempted from the *control-char* strip (it's not deleted outright)
    # but is still prose whitespace, so the horizontal-whitespace collapse
    # step folds it into a single space like any other run of spaces/tabs.
    assert clean_text("a\tb") == "a b"


def test_clean_text_nfkc_normalizes():
    # U+FB01 LATIN SMALL LIGATURE FI -> "fi" under NFKC.
    assert clean_text("ﬁsh") == "fish"


def test_clean_text_collapses_horizontal_whitespace_and_excess_blank_lines():
    dirty = "A   fox    ran.\n\n\n\nThe end."
    cleaned = clean_text(dirty)
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_text_strips_leading_trailing_whitespace():
    assert clean_text("  a story  ") == "a story"


def test_is_acceptable_length():
    assert not is_acceptable_length("short", min_chars=20)
    assert is_acceptable_length("x" * 100, min_chars=20, max_chars=2000)
    assert not is_acceptable_length("x" * 2001, min_chars=20, max_chars=2000)


def test_content_hash_deterministic_and_sensitive_to_content():
    a = content_hash("Once upon a time")
    b = content_hash("Once upon a time")
    c = content_hash("Once upon a time.")
    assert a == b
    assert a != c


def test_near_dedup_flags_near_identical_text():
    base = "the quick brown fox jumps over the lazy dog in the sunny field today"
    near_dup = base.replace("sunny", "cloudy")  # one word changed out of ~13
    distinct = "a completely different sentence about spaceships and robots exploring mars"

    texts = [base, near_dup, distinct]
    dropped = find_near_duplicates(texts, k=3, threshold=0.7)

    assert 1 in dropped  # near_dup dropped as a duplicate of base
    assert 2 not in dropped  # distinct text survives
    assert 0 not in dropped  # first occurrence is always kept


def test_cross_split_leakage_detected_with_independent_hash_sets():
    """Regression test for a real bug: prepare.py originally passed one
    shared `seen_hashes` set to both _clean_split calls (train first), so any
    val doc matching a train hash was silently absorbed into val's own
    exact-dup counter before the leakage check ever ran — making leakage
    structurally always report 0, regardless of the true rate.
    """
    shared_text = "Lily found a shiny red ball in the park and played all afternoon."
    train_rows = [{"text": shared_text}, {"text": "A distinct train-only story about a fox."}]
    val_rows = [
        {"text": shared_text},  # true cross-split duplicate
        {"text": "A distinct val-only story about a dragon."},
    ]

    # Independent sets — the fixed behavior.
    train_texts, train_hashes, train_stats = _clean_split(train_rows, "train", 10, 2000, set())
    val_texts, val_hashes, val_stats = _clean_split(val_rows, "validation", 10, 2000, set())

    # Before the leakage pass, the shared_text duplicate is still present in
    # val (independent sets don't cross-filter) — this is what makes the
    # leakage check downstream actually able to find something.
    assert shared_text in val_texts
    assert val_stats.dropped_exact_dup == 0

    kept, leaked = _remove_cross_split_leakage(val_texts, val_hashes, train_hashes)
    assert leaked == 1
    assert shared_text not in kept
    assert "A distinct val-only story about a dragon." in kept


def test_cross_split_leakage_is_undetectable_with_a_shared_hash_set():
    """Documents the bug pattern itself: reproduces the original (buggy)
    shared-set call and shows the leakage check is structurally blind to it —
    guarding against this regressing back in.
    """
    shared_text = "Lily found a shiny red ball in the park and played all afternoon."
    train_rows = [{"text": shared_text}]
    val_rows = [{"text": shared_text}]

    shared_seen_hashes: set[str] = set()
    _, train_hashes, _ = _clean_split(train_rows, "train", 10, 2000, shared_seen_hashes)
    val_texts, val_hashes, val_stats = _clean_split(
        val_rows, "validation", 10, 2000, shared_seen_hashes
    )

    # The duplicate was already absorbed into val's own exact-dup counter —
    # it never reaches the leakage check at all.
    assert val_texts == []
    assert val_stats.dropped_exact_dup == 1

    kept, leaked = _remove_cross_split_leakage(val_texts, val_hashes, train_hashes)
    assert leaked == 0  # structurally blind, by construction — this is the bug


def test_near_dedup_keeps_all_distinct_short_stories():
    texts = [
        "Lily saw a red ball in the park and played all day.",
        "Tom found a blue kite stuck in a tall tree near his house.",
        "Sara baked cookies with her mom on a rainy Sunday afternoon.",
    ]
    dropped = find_near_duplicates(texts, k=3, threshold=0.8)
    assert dropped == set()
