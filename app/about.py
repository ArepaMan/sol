"""Shared demo copy — the honest-limitations text, in one place.

There are two UIs (`streamlit_app.py`, deployed; `demo.py`, Gradio, local),
and the one thing that must not drift between them is what the model claims
about itself. Every number here traces to a file in the repo: the rubric
scores to `eval/rubric_scores.csv`, the perplexity to `eval/results.md`, the
mojibake rate to `docs/DATA_CARD.md`.
"""

from __future__ import annotations

GITHUB = "https://github.com/ArepaMan/sol"
HF_MODEL = "https://huggingface.co/SpicyGuac/sol-001"

TAGLINE = (
    "A 52M-parameter transformer trained from scratch on TinyStories, on one 8 GB "
    "laptop GPU. Give it the opening of a children's story and it will finish it."
)

FACTS = [
    ("Parameters", "52,901,712 (8 layers, 8 heads, 592 embd)"),
    ("Context", "512 tokens"),
    ("Vocab", "32k byte-level BPE, trained in-domain"),
    ("Training", "40,000 iters, 1.31B tokens processed (~3.66 epochs)"),
    ("Val perplexity", "3.719, 95% CI [3.693, 3.745]"),
    ("Baselines beaten", "trigram 23.4 · unigram 379.0 · uniform 32000"),
]

ABOUT_MD = f"""
## What Sol is

A **52,901,712-parameter** decoder-only transformer, trained from scratch on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) on a single
8 GB laptop GPU (RTX 4070 Laptop, BF16). No pretrained weights, no fine-tuning of
anything — the tokenizer, the architecture, and the training loop are all in the
repo. It is a portfolio project about doing the fundamentals properly, not a
general-purpose assistant.

**Code:** [github.com/ArepaMan/sol]({GITHUB}) · **Weights:** [SpicyGuac/sol-001]({HF_MODEL})

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
- **Occasional mojibake** (stray characters, odd fragments). Traced, not
  guessed: 6.20% of TinyStories' *own* training documents contain it. Inherited
  from upstream, documented rather than quietly patched.

Full writeup: [`docs/LIMITATIONS.md`]({GITHUB}/blob/master/docs/LIMITATIONS.md)
· [`eval/results.md`]({GITHUB}/blob/master/eval/results.md)

## Sampling controls

**Temperature** — below ~0.6 the model repeats itself; above ~1.1 it loses the
thread. 0.8 is the default used for every number quoted above.
**Top-k** — restricts each step to the k most likely tokens. **Seed** — a fixed
seed plus fixed settings reproduces a story exactly, on the same hardware.
"""

EXAMPLES = [
    "Once upon a time, there was a little girl named Lily who",
    "Tom and his sister found a shiny box under the old tree. Inside was",
    '"Where did my hat go?" asked the small bear.',
    "The robot was sad because nobody would play with it. Then one day",
]
