"""Streamlit demo for Sol — the deployed app (Streamlit Community Cloud).

This exists instead of the Gradio app (`demo.py`) for one reason, recorded
here so it doesn't look arbitrary: Hugging Face moved Gradio Spaces on
free `cpu-basic` behind a PRO subscription. The Gradio app was written, tested,
and works; it just has nowhere free to live. Community Cloud gives a real
Python backend for free, so the UI moved and nothing else did.

Both UIs are thin wrappers over `SolGenerator` (`src/infer.py`) — that's the
payoff for putting generation in one place rather than in the app. The sampling
behaviour is identical because it is literally the same code.

**Deployment constraint that shapes this file: 1 GB RAM.** `@st.cache_resource`
means one model instance is shared across every visitor and every rerun —
Streamlit reruns the whole script on each widget interaction, so without it,
each click would try to load a 212 MB fp32 model and the app would OOM almost
immediately.

Run locally:
    SOL_MODEL_DIR=export/sol-001 streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.about import ABOUT_MD, EXAMPLES, FACTS, GITHUB, TAGLINE  # noqa: E402
from src.infer import SolGenerator  # noqa: E402

REPO_ID = os.environ.get("SOL_REPO_ID", "SpicyGuac/sol-001")
LOCAL_DIR = os.environ.get("SOL_MODEL_DIR")
BUNDLE_FILES = ("model.pt", "config.yaml", "tokenizer.json", "meta.json")

st.set_page_config(page_title="Sol — a 52M-parameter story model", page_icon="☀️", layout="centered")


@st.cache_resource(show_spinner=False)
def load_generator() -> tuple[SolGenerator, float]:
    """Load once per process, shared across sessions. See the module docstring
    on why this is a memory requirement here, not an optimisation."""
    t0 = time.perf_counter()
    if LOCAL_DIR:
        model_dir = LOCAL_DIR
    else:
        from huggingface_hub import hf_hub_download

        paths = [hf_hub_download(repo_id=REPO_ID, filename=name) for name in BUNDLE_FILES]
        model_dir = str(Path(paths[0]).parent)
    return SolGenerator.from_pretrained(model_dir), time.perf_counter() - t0


st.title("Sol ☀️")
st.markdown(TAGLINE)

with st.spinner("Loading the model — first visit after the app sleeps takes ~30-60s…"):
    generator, load_seconds = load_generator()

story_tab, about_tab = st.tabs(["Write a story", "About / Limitations"])

with st.sidebar:
    st.header("Sampling")
    max_new_tokens = st.slider("Max new tokens", 20, 400, 200, step=10)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, step=0.05)
    top_k = st.slider("Top-k (0 = off)", 0, 500, 200, step=10)
    seed = st.number_input("Seed", value=42, step=1)
    st.caption(
        f"Model loaded in {load_seconds:.1f}s. Generation runs on a free shared CPU at "
        "roughly 26-30 tokens/second, so a 200-token story takes ~7s."
    )
    st.divider()
    st.markdown(f"[Source on GitHub]({GITHUB})")

with story_tab:
    prompt = st.text_area(
        "Story opening",
        value=EXAMPLES[0],
        height=100,
        help="Sol continues text; it does not answer questions. Start a story.",
    )

    st.caption("Or try one of these:")
    example_cols = st.columns(len(EXAMPLES))
    for col, example in zip(example_cols, EXAMPLES, strict=True):
        if col.button(example[:28] + "…", use_container_width=True):
            st.session_state["prompt_override"] = example
            st.rerun()

    if "prompt_override" in st.session_state:
        prompt = st.session_state.pop("prompt_override")
        st.info(f"Using example: *{prompt}*")

    if st.button("Write the story", type="primary"):
        cleaned = (prompt or "").strip()
        if not cleaned:
            st.warning("Type a story opening first — Sol continues text, it doesn't start from nothing.")
        else:
            t0 = time.perf_counter()
            st.write(f"**{cleaned}**")
            # stream() yields decoded deltas, which is exactly what
            # st.write_stream consumes — no adapter needed.
            completion = st.write_stream(
                generator.stream(
                    cleaned,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_k=int(top_k) if int(top_k) > 0 else None,
                    seed=int(seed),
                )
            )
            dt = time.perf_counter() - t0
            n_tokens = len(generator.tokenizer.encode(str(completion)).ids)
            st.caption(f"{n_tokens} tokens in {dt:.1f}s = {n_tokens / dt:.1f} tok/s")

with about_tab:
    st.markdown(ABOUT_MD)
    st.table({"": [k for k, _ in FACTS], " ": [v for _, v in FACTS]})
