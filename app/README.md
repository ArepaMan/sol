---
title: Sol — 52M-Parameter Story Model
emoji: ☀️
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: demo.py
pinned: true
license: mit
short_description: A 52M transformer trained from scratch on TinyStories
models:
  - ArepaMan/sol-001
---

# Sol

A **52,901,712-parameter** decoder-only transformer, trained from scratch on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) on a single
8 GB laptop GPU (RTX 4070 Laptop, BF16). No pretrained weights. The tokenizer,
architecture, training loop, evaluation harness, and ablations are all in the
source repo: **[github.com/ArepaMan/sol](https://github.com/ArepaMan/sol)**.

Give it the opening of a children's story and it finishes it.

| | |
|---|---|
| Parameters | 52.9M (8 layers, 8 heads, 592 embd, 512 context) |
| Val perplexity | 3.719, 95% CI [3.693, 3.745] |
| vs trigram / unigram baseline | 23.4 / 379.0 |
| Rubric (n=60, 1–5) | grammar 4.00 · coherence 3.15 · on-topic 4.12 |

## Limitations

Children's stories only. No instruction tuning — it will not answer a question,
it will continue it as prose. Coherence, not grammar, is the binding limitation:
named characters drift over ~150 tokens. See the **About / Limitations** tab in
the app, or [`docs/LIMITATIONS.md`](https://github.com/ArepaMan/sol/blob/master/docs/LIMITATIONS.md).

## Notes on this Space

- Free CPU hardware; sleeps when idle. First request after a nap costs a cold
  start (measured — see `docs/DEPLOY.md` in the source repo). `pinned: true`,
  a CPU-only torch wheel, and import-time model loading are all there to keep
  that number down.
- Weights are **not** in this Space. They live in the model repo
  [`ArepaMan/sol-001`](https://huggingface.co/ArepaMan/sol-001) and are fetched
  once via `hf_hub_download`, which caches to disk.
