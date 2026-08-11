"""Gradio demo for Sol — the HF Space build. **Not the deployed app.**

Hugging Face moved Gradio Spaces on free `cpu-basic` behind a PRO subscription,
which this project isn't paying for. This file is kept, working and tested,
because it is exactly what you'd push to a Space with PRO (`app/README.md`
carries the Space YAML header) — and because deleting it would erase a real
deployment constraint that measurement, not planning, surfaced. The deployed
app is `app/streamlit_app.py`; both are thin wrappers over the same
`SolGenerator`, which is the point of putting generation in `src/infer.py`.

Its dependencies are in `app/requirements-gradio.txt`, not
`app/requirements.txt` — the latter is what Streamlit Community Cloud installs.

Two things about this file are deployment decisions, not style:

**The model loads at import time**, not inside the request handler. A Space
that loads 100 MB of weights per click feels broken; loading once at import
moves that cost into the cold start, where a spinner already explains it.

**Weights come from a HF model repo, not this repo.** `SOL_REPO_ID` is fetched
with `hf_hub_download`, which caches to disk — so a Space that is merely idle,
not rebuilt, skips the download entirely. Set `SOL_MODEL_DIR` instead to run
against a local export (`python -m scripts.export_weights`), which is what
`docs/DEPLOY.md`'s local-smoke-test step does.

Run locally:
    SOL_MODEL_DIR=export/sol-001 python app/demo.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import gradio as gr

# Allow `python app/demo.py` from the repo root and `python demo.py` from
# inside a Space, where the repo layout is flattened.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infer import SolGenerator  # noqa: E402

# Imported, not duplicated: the two UIs must never disagree about what the
# model claims it can't do. See app/about.py.
try:
    from app.about import ABOUT_MD, EXAMPLES
except ImportError:  # flattened layout inside a Space
    from about import ABOUT_MD, EXAMPLES

REPO_ID = os.environ.get("SOL_REPO_ID", "SpicyGuac/sol-001")
LOCAL_DIR = os.environ.get("SOL_MODEL_DIR")

BUNDLE_FILES = ("model.pt", "config.yaml", "tokenizer.json", "meta.json")


def _resolve_model_dir() -> str:
    if LOCAL_DIR:
        return LOCAL_DIR
    from huggingface_hub import hf_hub_download

    paths = [hf_hub_download(repo_id=REPO_ID, filename=name) for name in BUNDLE_FILES]
    # hf_hub_download returns per-file cache paths that all live in the same
    # snapshot directory, so taking the parent of any one of them is safe.
    return str(Path(paths[0]).parent)


_t0 = time.perf_counter()
GENERATOR = SolGenerator.from_pretrained(_resolve_model_dir())
LOAD_SECONDS = time.perf_counter() - _t0
print(f"[sol] model ready in {LOAD_SECONDS:.1f}s", flush=True)


def stream_story(prompt: str, max_new_tokens: int, temperature: float, top_k: int, seed: int):
    """Gradio streaming handler: yields the growing text on every new token."""
    prompt = (prompt or "").strip()
    if not prompt:
        yield "Type a story opening first — for example, *Once upon a time there was a small dragon who*"
        return

    text = prompt
    yield text
    for chunk in GENERATOR.stream(
        prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k) if int(top_k) > 0 else None,
        seed=int(seed),
    ):
        text += chunk
        yield text


CSS = """
.sol-note { font-size: 0.9em; opacity: 0.75; }
"""

with gr.Blocks(title="Sol — a 52M-parameter story model", css=CSS) as demo:
    gr.Markdown(
        "# Sol\n"
        "A 52M-parameter transformer trained from scratch on TinyStories, on one 8 GB laptop GPU. "
        "Give it the opening of a children's story and it will finish it."
    )

    with gr.Tabs():
        with gr.Tab("Write a story"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(
                        label="Story opening",
                        value="Once upon a time, there was a little girl named Lily who",
                        lines=3,
                    )
                    go = gr.Button("Write the story", variant="primary")
                    output = gr.Textbox(label="Sol's story", lines=14, show_copy_button=True)
                    gr.Markdown(
                        "This Space runs on a free CPU and sleeps when idle — **the first "
                        "request after a nap can take ~60 s** while the model loads. "
                        "After that, generation streams at roughly 15–25 tokens/second.",
                        elem_classes="sol-note",
                    )
                with gr.Column(scale=1):
                    max_new_tokens = gr.Slider(20, 400, value=200, step=10, label="Max new tokens")
                    temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                    top_k = gr.Slider(0, 500, value=200, step=10, label="Top-k (0 = off)")
                    seed = gr.Number(value=42, precision=0, label="Seed")

            gr.Examples(
                examples=[
                    [text, 200, 0.8, 200, 42 + i * 7] for i, text in enumerate(EXAMPLES)
                ],
                inputs=[prompt, max_new_tokens, temperature, top_k, seed],
            )

        with gr.Tab("About / Limitations"):
            gr.Markdown(ABOUT_MD)

    inputs = [prompt, max_new_tokens, temperature, top_k, seed]
    go.click(stream_story, inputs=inputs, outputs=output)
    prompt.submit(stream_story, inputs=inputs, outputs=output)


if __name__ == "__main__":
    demo.launch()
