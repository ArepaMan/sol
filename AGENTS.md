# Sol — Agent Context

Read this file at the start of any session working on this repo.

## What this project is

**Sol** is a portfolio project: train a **~52M-parameter decoder-only transformer from scratch** on TinyStories, with a full data pipeline, eval harness, ablations, and a demo.

**Owner goals:** Applied ML, AI Engineer, or Data Scientist roles. The project must be interview-ready — measurable results, documented trade-offs, reproducible commands.

## Hardware constraint (do not exceed without discussion)

| Spec | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Architecture | Ada Lovelace, sm_89 |
| VRAM | 8 GB (8188 MiB) |
| Precision | **BF16** — Ada supports it; no GradScaler, no loss-scale debugging |
| Max context | **512** tokens |
| Gradient checkpointing | Required |

> An earlier draft targeted an RTX 2070 Super and concluded "FP16, not BF16 — Turing".
> That machine is not this machine. 8 GB is still the binding constraint, so the
> hardware-aware sizing story is unchanged; only the precision rationale moved.

## Locked-in model config (~52M params)

```yaml
n_layer: 8
n_head: 8
n_embd: 512
block_size: 512
vocab_size: 32000
dropout: 0.0
bias: false
```

**Training defaults:** micro-batch 4, grad accum 16, LR 3e-4, warmup 1000, max_iters ~40000, max_train_tokens 400M.

**Scale, stated precisely** — these are two different numbers and the original spec conflated them:

| Quantity | Value |
|----------|-------|
| Tokens per step | 4 × 16 × 512 = **32,768** |
| Tokens processed over the run | 32,768 × 40,000 = **1.31B** |
| Corpus size (`max_train_tokens`) | **400M** |
| Passes over the corpus | **~3.3 epochs** |

Wall-clock is measured in M4, not guessed. See `experiments/000_benchmark/results.md`.

Full config lives in `configs/micro_50m_8gb.yaml`. It is the **single source of truth** —
read values via `src/config.py`, never hardcode a hyperparameter.

## Stack (planned)

- Python 3.11+, PyTorch 2.x
- Training base: **litgpt** or **nanoGPT** (adapt, don't reinvent matmul)
- `tokenizers` (BPE), `datasets`, `wandb`, `gradio`, `pytest`

## Repo layout

```text
configs/       # YAML training configs
data/          # prepare.py, tokenize.py, stats notebook
src/           # model.py, train.py, eval.py, infer.py
experiments/   # ablation runs (001_baseline, 002_lr_sweep, etc.)
app/           # Gradio / FastAPI demo
tests/         # unit tests (tokenizer, data)
docs/          # PROJECT.md (full spec), interview talking points
```

## Roadmap

Ten dependency-ordered milestones, each with a verifiable exit criterion.
Full detail — files, commands, risks, exit gates — in [`docs/ROADMAP.md`](docs/ROADMAP.md).

| M | Name | Exit criterion (short) |
|---|------|------------------------|
| 0 | Env bootstrap, spec correction, public repo | `bf16_supported() == True`, pytest green, repo pushed |
| 1 | Data pipeline + BPE tokenizer | `meta.json` token count; round-trip test green |
| 2 | EDA notebook + data card | Notebook runs clean; every data-card number traceable |
| 3 | `src/model.py` hand-written | 6 architecture tests green, incl. causality |
| 4 | Train loop + **smoke gates** + benchmark | Overfit-one-batch < 0.1; peak VRAM < 7400 MiB |
| 5 | Baseline run 001 | val loss ≤ 3.2 |
| 6 | Eval harness | ppl + CI vs 3 baselines; 60 prompts scored |
| 7 | Ablations + **seed variance** | sd across seeds reported as the yardstick |
| 8 | Infer CLI, Gradio, HF Space | Public Space generates in incognito |
| 9 | Docs, spec de-drift, portfolio wiring | `npm run build` green; `/projects/sol` live |

**Never skip:** M2 (the DS signal) and M4's overfit gate — both cheap, both carry
disproportionate interview weight.

## Interview skills to demonstrate (build these into the project)

1. **Problem framing** — scoped task, not "ChatGPT clone"
2. **Data pipeline** — clean, dedupe, tokenize, EDA, data card
3. **Transformer fundamentals** — causal attention, next-token loss, can whiteboard architecture
4. **Training** — LR schedule, grad clip, checkpointing, OOM debugging
5. **Evaluation** — perplexity + qualitative generation rubric; compare baselines
6. **Ablations** — LR (1e-4 vs 3e-4), data scale (100M vs 400M tokens)
7. **Reproducibility** — configs, seeds, `make` or documented CLI commands
8. **Deployment** — Gradio demo; optional FastAPI inference
9. **Honest limitations** — repetition, weak reasoning; "what I'd do next" in README

## README must include (hiring-manager pattern)

1. Problem (one paragraph)
2. Approach (architecture + what you measure)
3. Results table (numbers vs baseline)
4. How to run (3 commands or Docker one-liner)
5. What I'd do next (3 limitations + next experiments)

## Current status

**Milestone: M0 (environment bootstrap).**

- [x] Folder scaffold, `.gitignore`
- [x] `AGENTS.md`, `docs/PROJECT.md`, `docs/ROADMAP.md`, Cursor rule
- [x] `configs/micro_50m_8gb.yaml` (bfloat16, 4070-targeted)
- [x] Python 3.12 venv + `requirements.txt` + `pyproject.toml`
- [x] `src/config.py`, `src/utils.py`, `tests/test_env.py`, `tests/test_config.py`
- [ ] First commit + public GitHub remote
- [ ] M1 data pipeline
- [ ] M3 model / M4 training / M6 eval / M8 demo

## Environment

Python 3.14 is the system default and has **no PyTorch wheel**. Always use the venv
interpreter by absolute path:

```
C:\Users\manol\Projects\sol\.venv\Scripts\python.exe
```

Torch must come from the CUDA index (`--index-url https://download.pytorch.org/whl/cu124`)
or pip silently resolves the CPU-only wheel. `tests/test_env.py` is the gate.

## Conventions

- Config-driven training (YAML in `configs/`)
- No secrets in git (`.env` gitignored)
- Prefer minimal diffs; match existing style
- Do not commit unless user asks
