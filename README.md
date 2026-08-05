# Sol

A ~52M-parameter decoder-only transformer trained from scratch on TinyStories — data
pipeline, hand-written architecture, eval harness, ablations, and a deployed demo, all on
a single 8 GB laptop GPU.

**Hardware target:** NVIDIA GeForce RTX 4070 Laptop GPU (Ada Lovelace, sm_89, 8 GB) · BF16

## Status

✅ **M0–M5 complete** — environment, data pipeline, EDA, model, training loop, and the
first real training run are all done. **Baseline run 001: val loss 1.3569, perplexity
3.88** — clears the ≤3.2 target by a wide margin (the original target was a
pre-measurement guess; TinyStories' narrow, templated vocabulary makes much lower loss
achievable than general-domain text would — see `experiments/001_baseline/run.md`).
`src/model.py` measures at **52,901,712 params**. M6 (evaluation harness — perplexity
with a proper CI, baselines, qualitative rubric) is next.

See [`docs/DATA_CARD.md`](docs/DATA_CARD.md) for the full dataset writeup, including a
real bug caught and fixed mid-pipeline: validation's "28.67% duplicate rate" turned out
to be 0% internal duplication and 28.67% exact overlap with train (cross-split leakage,
now filtered) — a data-quality finding, not just a pipeline stat.

Full plan: [`docs/ROADMAP.md`](docs/ROADMAP.md) · Design rationale: [`docs/PROJECT.md`](docs/PROJECT.md) · Agent context: [`AGENTS.md`](AGENTS.md)

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

Baseline run 001 (`experiments/001_baseline/`), measured on the final checkpoint:

| Metric | Original target | **Measured** |
|--------|--------|--------|
| Val loss | 2.8–3.2 | **1.3569** |
| Perplexity | 15–25 | **3.88** |
| Peak VRAM | < 7400 MiB | **2344 MiB** |
| Generation | Coherent short stories, some repetition | Not yet qualitatively evaluated — M6 |

Baselines to compare against (uniform/unigram/trigram) and a qualitative rubric land in M6.

![Baseline loss curve](experiments/001_baseline/loss_curve.png)

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
├── src/         # config, model, train, eval, infer
├── experiments/ # benchmark + ablation runs and results
├── app/         # Gradio demo (deployed to a HF Space)
├── tests/       # env, config, tokenizer, data, model, training
└── docs/        # ROADMAP, PROJECT, DATA_CARD, RUBRIC, LIMITATIONS
```

## What I'd do next

Tracked in `docs/LIMITATIONS.md` as results land. Current known trade-offs: learned
positional embeddings rather than RoPE, 512-token context, LayerNorm + GELU rather than
RMSNorm + SwiGLU, and a single-rater qualitative rubric.
