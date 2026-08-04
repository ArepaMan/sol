"""Unit tests for the cleaning and near-dedup building blocks in data/prepare.py."""

from __future__ import annotations

from data.clean import clean_text, content_hash, is_acceptable_length
from data.near_dedup import find_near_duplicates


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


def test_near_dedup_keeps_all_distinct_short_stories():
    texts = [
        "Lily saw a red ball in the park and played all day.",
        "Tom found a blue kite stuck in a tall tree near his house.",
        "Sara baked cookies with her mom on a rainy Sunday afternoon.",
    ]
    dropped = find_near_duplicates(texts, k=3, threshold=0.8)
    assert dropped == set()
