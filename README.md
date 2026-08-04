# Sol

A ~52M-parameter decoder-only transformer trained from scratch on TinyStories — data
pipeline, hand-written architecture, eval harness, ablations, and a deployed demo, all on
a single 8 GB laptop GPU.

**Hardware target:** NVIDIA GeForce RTX 4070 Laptop GPU (Ada Lovelace, sm_89, 8 GB) · BF16

## Status

🚧 **M0–M4 complete** — environment bootstrapped, data pipeline done, EDA + data card
written, model built and tested, training loop built and all three smoke gates passed.
TinyStories: 1,748,358 train docs / 357,852,786 tokens, 15,141 val docs / 2,956,183
tokens. `src/model.py` measures at **52,901,712 params**. Gate 1 (overfit one batch)
reached loss 0.0224; Gate 2 (VRAM/throughput sweep) settled `gradient_checkpointing:
false` and `compile: false` by measurement, not assumption — chosen config peaks at
2137 MiB, tokens/s scales to **~16.9h for the real 40,000-iter baseline run**; Gate 3
(resume) verified a checkpoint round-trip, including RNG state. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) M4 for full results. The actual baseline run (M5)
is next — nothing has been trained end-to-end yet, and every number below marked
*target* is a target, not a result.

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
| Architecture | 8 layers, 8 heads, 592 embd, 512 context | **52.9M params, measured**; fits 8 GB comfortably (2137 MiB peak) without checkpointing |
| Vocab | 32k byte-level BPE, trained in-domain | Standard for small LMs; train-split only |
| Precision | **BF16** | Ada supports it — no GradScaler, no loss-scale debugging |
| Implementation | Hand-written `src/model.py` | The architecture code *is* the deliverable |
| Data | TinyStories, deduped, 400M-token cap | ~3.3 epochs at 40k iters |

**What gets measured:** validation perplexity with a bootstrap CI against three baselines
(uniform, unigram, trigram), an anchored 1–5 generation rubric scored blind, automatic
repetition metrics (distinct-2/3), and a seed-variance run so the ablations can be read
against real run-to-run noise.

## Results

Not yet trained. Targets from [`docs/PROJECT.md`](docs/PROJECT.md):

| Metric | Target |
|--------|--------|
| Val loss | 2.8–3.2 |
| Perplexity | 15–25 |
| Peak VRAM | < 7400 MiB |
| Generation | Coherent short stories, some repetition |

This table is replaced with measured numbers — and baselines to compare them against — in M9.

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
