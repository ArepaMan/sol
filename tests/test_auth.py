"""Unit tests for the demo's password gate (`app/auth.py`).

Scoped to the two pure functions — `resolve_password` and `verify` — rather
than to `gate()`, which is a rendering function whose behaviour is Streamlit's
form/rerun machinery, not ours. What must not regress is the *policy*: where
the password comes from, that a blank one means "disabled" rather than "matches
the empty string", and that the comparison is constant-time and
whitespace-tolerant.
"""

from __future__ import annotations

import pytest

from app.auth import ENV_VAR, resolve_password, verify


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No password configured unless a test says so. Without this, a developer
    who exports SOL_APP_PASSWORD in their shell gets a different test run than
    CI does."""
    monkeypatch.delenv(ENV_VAR, raising=False)


def test_gate_is_disabled_when_nothing_is_configured() -> None:
    # The failure mode of a missing secret is an open demo, not a locked-out
    # one — forks and local runs must work with no setup.
    assert resolve_password() is None


def test_env_var_configures_the_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "correct horse")
    assert resolve_password() == "correct horse"


def test_surrounding_whitespace_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "  spaced  ")
    assert resolve_password() == "spaced"


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_password_means_disabled_not_empty_match(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    # The dangerous reading of a blank secret is "the password is '' and an
    # empty submit box unlocks it". It must read as "not configured" instead.
    monkeypatch.setenv(ENV_VAR, blank)
    assert resolve_password() is None


def test_verify_accepts_the_password() -> None:
    assert verify("correct horse", "correct horse")


@pytest.mark.parametrize("wrong", ["", "correct", "Correct Horse", "correct  horse", "x"])
def test_verify_rejects_everything_else(wrong: str) -> None:
    assert not verify(wrong, "correct horse")


def test_verify_tolerates_whitespace_from_copy_paste() -> None:
    # The password is shared by hand, so a trailing newline off a clipboard is
    # a support ticket, not an attack.
    assert verify("  correct horse\n", "correct horse")


def test_verify_is_not_fooled_by_a_prefix() -> None:
    assert not verify("correct hors", "correct horse")
    assert not verify("correct horses", "correct horse")
