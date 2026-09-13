"""Per-session generation cap for the deployed demo.

**Why a cap when there is already a password.** The gate decides *who* gets in;
this decides how much of the single free vCPU any one visitor can hold. The two
are not redundant — the password is handed out by invitation, so everyone past
the gate is welcome, and the thing worth preventing is one enthusiastic tab
holding the queue while someone else is trying to look at the project.

**Why 20.** Sized from what the audience actually does: click two or three of
the four example prompts, write one or two of their own, then re-run a
favourite once or twice at a different temperature to see the sampling
controls do something. That is ~10. Twenty is double the honest ceiling of a
thorough visit, which is the right side to err on — a recruiter who hits a wall
mid-look is a much worse outcome than a few extra minutes of CPU.

**What it is worth, precisely.** `st.session_state` lives for one websocket
connection, so reloading the page starts a fresh session and a fresh count.
That is a real limit on what this can enforce and it is not worked around here:
a server-side identity to pin a budget to is exactly the machinery the password
gate deliberately does without. So this is a courtesy speed bump against a tab
left looping, not a quota. The message shown at the cap says the reload works,
because a visitor who thinks the demo is broken costs more than a visitor who
runs twenty-one stories.

`SOL_SESSION_CAP` overrides the number; `0` disables the cap entirely, which is
what a local run wants.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

ENV_VAR = "SOL_SESSION_CAP"
STATE_KEY = "sol_generations_used"

#: Generations per session. See the module docstring for where 20 comes from.
DEFAULT_CAP = 20

#: Remaining-count threshold below which the sidebar caption turns into a
#: visible warning. Five is roughly "one more round of experimenting", which is
#: enough notice to finish a thought rather than be cut off mid-way.
WARN_AT = 5


def resolve_cap(raw: str | None = None) -> int | None:
    """Return the cap in force, or ``None`` when generation is uncapped.

    `raw` is the environment value, taken from `SOL_SESSION_CAP` when not
    passed — a parameter so the parsing rules can be tested without touching
    the process environment. Anything unparseable falls back to `DEFAULT_CAP`
    rather than raising: a typo'd env var on a deploy should not take the demo
    down, and the safe direction for a *cap* is to keep capping.
    """
    if raw is None:
        raw = os.environ.get(ENV_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_CAP
    try:
        cap = int(raw.strip())
    except ValueError:
        return DEFAULT_CAP
    if cap <= 0:
        return None
    return cap


def remaining(used: int, cap: int | None) -> int | None:
    """Generations left, or ``None`` when uncapped. Never negative."""
    if cap is None:
        return None
    return max(cap - used, 0)


def used_this_session() -> int:
    return int(st.session_state.get(STATE_KEY, 0))


def record_generation() -> None:
    """Count one generation against this session's budget."""
    st.session_state[STATE_KEY] = used_this_session() + 1


def session_remaining() -> int | None:
    return remaining(used_this_session(), resolve_cap())


def render_budget_caption(slot: Any = st) -> None:
    """Draw the remaining-count line, into `slot` if one is given.

    Shown from the first visit rather than sprung at the end: stating the
    budget up front reads as a deliberate constraint, which it is, while a
    counter that only appears once it is nearly spent reads as a trap.

    `slot` exists because Streamlit renders top to bottom and the sidebar is
    drawn *before* the generation that spends a story, which would leave the
    number one behind all the way until the next interaction. The caller passes
    an `st.empty()` reserved in the sidebar and fills it at the end of the
    script instead. Rerunning to resync would also work and is the obvious
    reach, but it discards the story that was just streamed — the one thing on
    screen the visitor actually came for.
    """
    left = session_remaining()
    if left is None:
        return
    cap = resolve_cap()
    line = f"{left} of {cap} stories left in this session."
    if left <= WARN_AT:
        slot.warning(line)
    else:
        slot.caption(line)


def render_exhausted_notice(contact_url: str) -> None:
    """Explain the wall, and how to get past it, without a dead end."""
    cap = resolve_cap()
    st.info(
        f"**That's this session's {cap} stories.** The cap keeps one tab from "
        "monopolising the single shared CPU this demo runs on — "
        "**reload the page to start a fresh session** and carry on. "
        f"If you'd like to talk about how any of it works, [get in touch]({contact_url}); "
        "the About tab has the evals and the honest limitations in the meantime."
    )
