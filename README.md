# Sol

[![CI](https://github.com/ArepaMan/sol/actions/workflows/ci.yml/badge.svg)](https://github.com/ArepaMan/sol/actions/workflows/ci.yml)

A ~52M-parameter decoder-only transformer trained from scratch on TinyStories — data
pipeline, hand-written architecture, eval harness, ablations, and a deployed demo, all on
a single 8 GB laptop GPU.

**Hardware target:** NVIDIA GeForce RTX 4070 Laptop GPU (Ada Lovelace, sm_89, 8 GB) · BF16

## Status

🚀 **Live demo: <https://sol-52m.streamlit.app>** · Weights:
[`SpicyGuac/sol-001`](https://huggingface.co/SpicyGuac/sol-001)

✅ **M0–M9 complete — the roadmap is done.** Environment, data pipeline, EDA, model,
training loop, baseline run, evaluation harness, ablations, a deployed demo, and the
docs/spec-drift work are all finished. **Sol-001: val perplexity 3.719, 95% CI [3.693, 3.745]** on
the full 15,141-document val set — vs a trigram baseline at 23.4 and a unigram baseline
at 379.0 (`eval/results.md`). `src/model.py` measures at **52,901,712 params**.
Ablations (M7): learning rate has by far the largest effect (20–200× seed-to-seed
noise); data scale (100M vs full corpus) matters, but only narrowly
(`experiments/002_lr_sweep/`, `experiments/003_data_scale/`,
`experiments/004_seed_variance/`). The demo runs at **29.0 tok/s** on a free shared
vCPU (range 26.2–30.0, n=5), loading in 6.7 s — up from 16.0 before the post-M9 KV
cache — see [`docs/DEPLOY.md`](docs/DEPLOY.md).

> **A note on hosting, because it changed the plan.** M8 targeted a free Hugging Face
> Space. Hugging Face has since moved Gradio Spaces behind a PRO subscription
> (`402 Payment Required` at `repo create`), so the demo moved to Streamlit Community
> Cloud — free, with a **1 GB RAM ceiling** that became the milestone's real engineering
> constraint. The Gradio app is kept and still works for anyone with PRO. Both UIs are
> thin wrappers over the same `SolGenerator`, which is what putting generation in
> `src/infer.py` rather than in the app bought.

See [`docs/DATA_CARD.md`](docs/DATA_CARD.md) for the full dataset writeup, including a
real bug caught and fixed mid-pipeline: validation's "28.67% duplicate rate" turned out
to be 0% internal duplication and 28.67% exact overlap with train (cross-split leakage,
now filtered) — a data-quality finding, not just a pipeline stat.

Full plan: [`docs/ROADMAP.md`](docs/ROADMAP.md) · QA: [`docs/QA_CHECKLIST.md`](docs/QA_CHECKLIST.md) · Design rationale: [`docs/PROJECT.md`](docs/PROJECT.md) · **Interview notes:** [`docs/INTERVIEW_NOTES.md`](docs/INTERVIEW_NOTES.md) · Agent context: [`AGENTS.md`](AGENTS.md)

Every number quoted in this README is generated into [`docs/spec.json`](docs/spec.json) by
`scripts/export_spec.py` and guarded by `tests/test_spec_drift.py` — edit a config without
regenerating and the suite goes red.

## Problem

Applied ML interviews reward candidates who understand tokenization, training dynamics,
and hardware trade-offs — not just API calls to foundation models. Sol is scoped to prove
those fundamentals end to end: a small model trained properly and measured honestly,
rather than a large one fine-tuned and demoed on vibes.

## Approach

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Architecture | 8 layers, 8 heads, 592 embd, 512 context | **52.9M params, measured**; fits 8 GB comfortably (2344 MiB peak) without checkpointing |
| Vocab | 32k byte-level BPE, trained in-domain | Standard for small LMs; train-split only |
| Precision | **BF16** | Ada supports it — no GradScaler, no loss-scale debugging |
| Implementation | Hand-written `src/model.py` | The architecture code *is* the deliverable |
| Data | TinyStories, deduped, 357.85M-token cap (measured) | ~3.66 epochs at 40k iters |

**What gets measured:** validation perplexity with a bootstrap CI against three baselines
(uniform, unigram, trigram), an anchored 1–5 generation rubric scored blind, automatic
repetition metrics (distinct-2/3), and a seed-variance run so the ablations can be read
against real run-to-run noise.

## Results

Baseline run 001 (`experiments/001_baseline/`), evaluated on the full val set (M6):

| Metric | Original target | **Measured** |
|--------|--------|--------|
| Val perplexity | 15–25 | **3.719** [3.693, 3.745] (95% CI) |
| Peak VRAM | < 7400 MiB | **2344 MiB** |
| vs trigram baseline | beat it | **3.719 vs 23.4** |
| Rubric: grammar / coherence / repetition | — | **4.00 / 3.15 / 3.98** out of 5 (n=60) |
| Generation | Coherent short stories, some repetition | Grammatical; entity/character drift over ~150 tokens is the main flaw — see `docs/LIMITATIONS.md` |

Full breakdown, baselines table, per-length-bucket perplexity, and the qualitative
rubric: [`eval/results.md`](eval/results.md).

![Baseline loss curve](experiments/001_baseline/loss_curve.png)

**Ablations (M7)**, all at a reduced 8000-iter budget with seed variance as the yardstick
(4.490 ± 0.0045 ppl across 3 seeds — n=3, so this range substitutes for a t-test rather
than pretending one is meaningful):

| Ablation | Result | vs seed-noise floor |
|---|---|---|
| Learning rate: 1e-4 / 3e-4 / 1e-3 | 5.380 / 4.489 / **4.162** ppl | 20–200× — real, large effect |
| Data scale: 100M vs full corpus | 4.572 / **4.489** ppl | 18× — real, but small (the milestone's honest negative result) |

Full writeups: [`experiments/002_lr_sweep/results.md`](experiments/002_lr_sweep/results.md),
[`experiments/003_data_scale/results.md`](experiments/003_data_scale/results.md),
[`experiments/004_seed_variance/results.md`](experiments/004_seed_variance/results.md).

## Try it

```powershell
python -m src.infer --prompt "Once upon a time, there was a little girl named Lily who" --seed 42
```

```powershell
python -m scripts.export_weights --out export/sol-001; $env:SOL_MODEL_DIR = "export/sol-001"; streamlit run app/streamlit_app.py
```

Generation stops at the model's own end-of-story token, streams, and is
reproducible per device: `--seed 42` twice gives byte-identical stdout.
Generation uses a **KV cache**, added after the roadmap closed and measured
against the published baseline it replaced: **23.4 → 82.8 tok/s on CPU (3.5×)**,
**124.4 → 132.4 on the 4070 (1.06×)**, n=7 per cell, cached and uncached output
byte-identical at fp32. The GPU number being the small one is the interesting
part — at batch 1 a 52M model is launch-overhead-bound, not FLOP-bound, so
deleting redundant arithmetic barely helps; the CPU, which is what the demo
runs on, is genuinely compute-bound. Deployment procedure, the cold-start
analysis, and why the cache buys nothing past `block_size`:
[`docs/DEPLOY.md`](docs/DEPLOY.md).

## How to run

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip.exe install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip.exe install -r requirements.txt
pytest
```

Full command reference: [`docs/COMMANDS.md`](docs/COMMANDS.md).

> The system Python here is 3.14, which has no PyTorch wheel — the 3.12 venv is required,
> and torch must come from the CUDA index or pip silently installs the CPU build.

## Repo layout

```text
sol/
├── configs/     # YAML training configs — single source of truth for every number
├── data/        # prepare.py, train_tokenizer.py, tokenize.py, eda.ipynb
├── src/         # config, model, train, eval, baselines, generate_samples, infer
├── eval/        # prompts.jsonl, generations.jsonl, rubric_scores.csv, results.md
├── experiments/ # benchmark + ablation runs and results
├── app/         # streamlit_app.py (deployed) + demo.py (Gradio, local/HF-PRO)
├── scripts/     # export_weights, export_spec, plot_ablations, ablation runners
├── tests/       # env, config, tokenizer, data, model, training, eval, spec drift
└── docs/        # ROADMAP, PROJECT, DATA_CARD, RUBRIC, LIMITATIONS, DEPLOY, QA_CHECKLIST
```

## What I'd do next

Tracked in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) as results land. Top three right
now: (1) coherence — named entities drift across ~150-token generations, the binding
constraint per M6's rubric (3.15/5 vs 4.00/5 grammar); (2) a mojibake artifact inherited
from 6.2% of TinyStories' own training documents, confirmed by grepping the raw corpus,
not yet cleaned; (3) learned positional embeddings rather than RoPE, LayerNorm + GELU
rather than RMSNorm + SwiGLU — both real architecture trade-offs, not yet ablated.
