"""Password gate for the deployed demo.

**What this actually protects.** Sol runs its own weights on the Streamlit
Community Cloud container — there is no paid inference API behind it, so the
budget at risk is not tokens billed to anyone, it is the *one* free shared
vCPU and the 1 GB RAM ceiling that `app/streamlit_app.py` is built around.
Generation is ~29 tok/s single-threaded; a handful of people looping 400-token
requests is enough to make the app unusable for everyone else, because
`@st.cache_resource` means they are all queued against the same process. The
gate exists to keep the demo shareable-on-request rather than open to drive-by
load.

**Where it sits.** `gate()` is called *before* `load_generator()`, so an
unauthenticated visitor never triggers the 103 MiB `hf_hub_download` or the
model load. That ordering is the whole point — a gate drawn after the spinner
would still pay the expensive part for every visitor.

**What it is not.** One shared password, no accounts, no server-side session
store. It stops casual abuse; it is not an authorization system, and anyone
with the password can hand it on. Deliberate: the threat model is
"budget-conscious portfolio demo", not "multi-tenant service".

Configure it on Streamlit Community Cloud under *Settings -> Secrets*:

    app_password = "the-password-you-share"

Locally, `SOL_APP_PASSWORD` does the same job. **If neither is set the gate
opens** — a fork, a local `streamlit run`, and the test suite all work
unchanged, and the deployed app is the only place that needs the secret. See
`docs/DEPLOY.md`.
"""

from __future__ import annotations

import hmac
import os
import time

import streamlit as st

from app.about import CONTACT

SECRET_KEY = "app_password"
ENV_VAR = "SOL_APP_PASSWORD"
STATE_KEY = "sol_authenticated"
ATTEMPTS_KEY = "sol_auth_attempts"

#: Sleep added per failed attempt, in seconds. Streamlit reruns the script on
#: submit, so this throttles a scripted guesser inside one session without
#: blocking anyone who simply mistyped once. It is a speed bump, not a lockout:
#: a determined attacker just opens a new session. Against a password long
#: enough to be worth sharing by hand, that is enough.
BACKOFF_SECONDS = 1.0


def resolve_password() -> str | None:
    """Return the configured password, or ``None`` when the gate is disabled.

    `st.secrets` is the deployed path and `SOL_APP_PASSWORD` the local one.

    The `load_if_toml_exists()` call is not belt-and-braces: a plain
    `st.secrets.get(...)` with no secrets.toml on disk — the normal state on a
    dev machine and in CI — makes Streamlit *render a red "No secrets found"
    box into the app* before it raises, so catching the exception is not enough
    to keep that off the page. `load_if_toml_exists()` is the same lookup
    without the reporting. The `except` underneath it covers a malformed file.
    """
    secret = None
    try:
        if st.secrets.load_if_toml_exists():
            secret = st.secrets.get(SECRET_KEY)
    except Exception:  # unreadable or malformed secrets.toml; fall through to env
        secret = None
    raw = secret if secret is not None else os.environ.get(ENV_VAR)
    if raw is None:
        return None
    password = str(raw).strip()
    # An empty or whitespace-only value is treated as "not configured" rather
    # than as a password that matches "": a blank secret is a misconfiguration,
    # and the failure mode should be an open demo, not one nobody can enter.
    return password or None


def verify(candidate: str, expected: str) -> bool:
    """Constant-time comparison of a submitted password against the real one.

    `hmac.compare_digest` rather than `==` so the check does not leak the
    length of the shared prefix through timing. Both sides are stripped because
    the password travels by hand — through a chat message, an email, a
    clipboard — and a trailing space is a support ticket, not an attack.
    """
    return hmac.compare_digest(candidate.strip().encode(), expected.strip().encode())


def gate(*, tagline: str = "") -> bool:
    """Render the login form and report whether this session may continue.

    Returns ``True`` when the gate is disabled or already satisfied. Otherwise
    it draws the form and returns ``False``; the caller is expected to
    `st.stop()`.
    """
    expected = resolve_password()
    if expected is None:
        return True
    if st.session_state.get(STATE_KEY):
        return True

    st.title("Sol ☀️")
    if tagline:
        st.markdown(tagline)
    st.info(
        "This demo is password-protected. It runs a real model on one free shared "
        "CPU, so access is handed out by request — **[reach out to me]"
        f"({CONTACT}) and I'll send you the password**. The code, the weights, the "
        "evals, and the writeups are all public in the meantime."
    )

    with st.form("sol_password"):
        candidate = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter", type="primary")

    if submitted:
        if verify(candidate, expected):
            st.session_state[STATE_KEY] = True
            st.session_state.pop(ATTEMPTS_KEY, None)
            st.rerun()
        attempts = st.session_state.get(ATTEMPTS_KEY, 0) + 1
        st.session_state[ATTEMPTS_KEY] = attempts
        time.sleep(BACKOFF_SECONDS)
        st.error(f"Incorrect password ({attempts} attempt{'s' if attempts > 1 else ''}).")

    return False
