"""Unit tests for the per-session generation cap (`app/limits.py`).

Scoped to `resolve_cap` and `remaining` — the policy — rather than to the
`render_*` functions, which are Streamlit drawing calls. What must not regress
is the arithmetic a visitor is judged by: what the cap is, what disables it,
and that a bad environment value fails toward *capping* rather than toward an
open tap or a crash.
"""

from __future__ import annotations

import pytest

from app.limits import DEFAULT_CAP, WARN_AT, remaining, resolve_cap


def test_default_cap_is_generous_enough_for_a_real_visit() -> None:
    # Sized in the module docstring from what a thorough visit actually does
    # (~10 generations). The guard is against someone quietly shrinking it to a
    # number that cuts a visitor off mid-look.
    assert DEFAULT_CAP >= 15
    assert WARN_AT < DEFAULT_CAP


def test_unset_env_uses_the_default() -> None:
    assert resolve_cap(None) == DEFAULT_CAP


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_env_uses_the_default(blank: str) -> None:
    assert resolve_cap(blank) == DEFAULT_CAP


def test_env_overrides_the_cap() -> None:
    assert resolve_cap("3") == 3
    assert resolve_cap("  7  ") == 7


@pytest.mark.parametrize("off", ["0", "-1", "-100"])
def test_zero_or_negative_disables_the_cap(off: str) -> None:
    # Uncapped is how a local run and the test suite want to behave.
    assert resolve_cap(off) is None


@pytest.mark.parametrize("junk", ["twenty", "3.5", "20x", "--"])
def test_unparseable_env_falls_back_to_the_default_rather_than_raising(junk: str) -> None:
    # A typo'd env var on a deploy must not take the demo down, and the safe
    # direction for a *cap* is to keep capping rather than to open the tap.
    assert resolve_cap(junk) == DEFAULT_CAP


def test_remaining_counts_down() -> None:
    assert remaining(0, 20) == 20
    assert remaining(1, 20) == 19
    assert remaining(20, 20) == 0


def test_remaining_never_goes_negative() -> None:
    # The UI compares against 0 to decide "exhausted"; a negative would read as
    # non-zero to a naive check and hand back an unlimited budget.
    assert remaining(21, 20) == 0
    assert remaining(9999, 20) == 0


def test_remaining_is_none_when_uncapped() -> None:
    assert remaining(0, None) is None
    assert remaining(10_000, None) is None
