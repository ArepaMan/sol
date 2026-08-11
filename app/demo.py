"""Gradio demo for Sol — deployed to a Hugging Face Space.

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

REPO_ID = os.environ.get("SOL_REPO_ID", "ArepaMan/sol-001")
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


ABOUT = """
## What Sol is

A **52,901,712-parameter** decoder-only transformer, trained from scratch on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) on a single
8 GB laptop GPU (RTX 4070 Laptop, BF16). No pretrained weights, no fine-tuning of
anything — the tokenizer, the architecture, and the training loop are all in the
repo. It is a portfolio project about doing the fundamentals properly, not a
general-purpose assistant.

**Code:** [github.com/ArepaMan/sol](https://github.com/ArepaMan/sol)

| | |
|---|---|
| Parameters | 52.9M (8 layers, 8 heads, 592 embd) |
| Context | 512 tokens |
| Vocab | 32k byte-level BPE, trained in-domain |
| Training | 40,000 iters, 1.31B tokens processed (~3.66 epochs) |
| Val perplexity | **3.719**, 95% CI [3.693, 3.745] |
| Baselines beaten | trigram 23.4 · unigram 379.0 · uniform 32000 |

## What it cannot do

- **It only writes children's stories.** TinyStories is simple narrative prose
  aimed at 3–4-year-olds. That is the entire world this model has seen.
- **It will not answer your question.** There is no instruction tuning and no
  chat format. Ask it "What is the capital of France?" and it will continue that
  sentence as if it were the first line of a story, because to Sol it is.
- **Coherence is the binding limitation, not grammar.** Hand-scored on 60
  prompts: grammar 4.00/5, coherence **3.15/5**. Named characters drift and
  swap roles over ~150 tokens. Grammar is largely fine; the plot is not.
- **Out-of-domain prompts fall apart** — 5.00/5 on-topic in-domain vs **1.47/5**
  out-of-domain, on the same rubric.
- **512-token context.** Longer prompts are truncated to their tail.
- **Occasional mojibake** (stray `Â` characters, odd fragments). Traced, not
  guessed: 6.20% of TinyStories' *own* training documents contain it. Inherited
  from upstream, documented rather than quietly patched.

Full writeup: [`docs/LIMITATIONS.md`](https://github.com/ArepaMan/sol/blob/master/docs/LIMITATIONS.md)
· [`eval/results.md`](https://github.com/ArepaMan/sol/blob/master/eval/results.md)

## Sampling controls

**Temperature** — below ~0.6 the model repeats itself; above ~1.1 it loses the
thread. 0.8 is the default used for every number quoted above.
**Top-k** — restricts each step to the k most likely tokens. **Seed** — fixed
seed plus fixed settings reproduces a story exactly.
"""

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
                    ["Once upon a time, there was a little girl named Lily who", 200, 0.8, 200, 42],
                    ["Tom and his sister found a shiny box under the old tree. Inside was", 200, 0.8, 200, 7],
                    ['"Where did my hat go?" asked the small bear.', 200, 0.9, 200, 13],
                    ["The robot was sad because nobody would play with it. Then one day", 200, 0.8, 200, 21],
                ],
                inputs=[prompt, max_new_tokens, temperature, top_k, seed],
            )

        with gr.Tab("About / Limitations"):
            gr.Markdown(ABOUT)

    inputs = [prompt, max_new_tokens, temperature, top_k, seed]
    go.click(stream_story, inputs=inputs, outputs=output)
    prompt.submit(stream_story, inputs=inputs, outputs=output)


if __name__ == "__main__":
    demo.launch()
