"""Text cleaning shared by prepare.py and its tests.

Kept separate from prepare.py so it can be unit-tested without touching the
network or the HF datasets cache.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Control characters (Unicode category Cc) except \n and \t, which we keep.
_CONTROL_CHARS = re.compile(
    "[" + "".join(chr(i) for i in range(0x20) if chr(i) not in "\n\t") + chr(0x7F) + "]"
)
_HORIZONTAL_WS_RUN = re.compile(r"[ \t]+")
_BLANK_LINE_RUN = re.compile(r"\n{3,}")

MIN_CHARS = 20
MAX_CHARS = 2000


def clean_text(text: str) -> str:
    """NFKC-normalise, strip control chars, collapse excess whitespace.

    Deliberately conservative: it does not touch casing, punctuation, or
    paragraph structure beyond collapsing obviously-excessive whitespace, so
    cleaning cannot itself introduce near-duplicates that near-dedup would
    then need to catch.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _HORIZONTAL_WS_RUN.sub(" ", text)
    text = _BLANK_LINE_RUN.sub("\n\n", text)
    return text.strip()


def is_acceptable_length(text: str, min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS) -> bool:
    return min_chars <= len(text) <= max_chars


def content_hash(text: str) -> str:
    """Stable hash used for both exact-dedup and the train/val leakage check."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
