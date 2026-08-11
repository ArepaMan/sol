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
| Gradient checkpointing | **Off** at `batch_size=4` — measured, not assumed (see below) |

> An earlier draft targeted an RTX 2070 Super and concluded "FP16, not BF16 — Turing".
> That machine is not this machine. 8 GB is still the binding constraint, so the
> hardware-aware sizing story is unchanged; only the precision rationale moved.

> A second earlier draft said gradient checkpointing was **required**. M4's benchmark
> measured `batch_size=4` at 2137 MiB without it (~21,500 tok/s) vs 1460 MiB with it
> (~17,000 tok/s, ~26% slower) — both far under the 8 GB budget. The "required"
> assumption predated knowing this model's true (much smaller than originally guessed,
> see the M3 `n_embd` correction) memory footprint. It becomes genuinely necessary only
> around `batch_size` 24–32 — and above that, a real footgun: exceeding physical VRAM on
> Windows doesn't cleanly OOM, it silently spills into shared/system memory at ~15% of
> normal throughput. See `experiments/000_benchmark/results.md`.

## Locked-in model config (~52M params, measured)

```yaml
n_layer: 8
n_head: 8
n_embd: 592
block_size: 512
vocab_size: 32000
dropout: 0.0
bias: false
```

**Measured (not estimated) param count: 52,901,712** with weight tying
(`tests/test_model.py::test_param_count_near_52m`). `n_embd=592` rather than
the earlier-guessed 512 — see `configs/micro_50m_8gb.yaml`'s header comment
for why: 512 measured at 41.8M (a ~20% gap from the project's own "~52M"
branding), and 592 (=74×8, keeps `head_dim` an integer) lands within 1.73%
while leaving every other locked value — 8 layers, 512 context, 32k vocab,
weight tying — untouched.

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

**M0–M7 complete. M8 code complete; weights published, live demo deploy pending.**
Weights are at <https://huggingface.co/SpicyGuac/sol-001>. **The free HF Space plan
died mid-milestone** — HF moved Gradio Spaces behind PRO (`402 Payment Required`),
so the demo was ported to Streamlit Community Cloud (`app/streamlit_app.py`),
whose **1 GB RAM ceiling** is now the binding deployment constraint. The final
deploy is a GitHub-OAuth web flow at share.streamlit.io that only the owner can
complete. Next: M9 (docs, spec de-drift, portfolio wiring). See `docs/DEPLOY.md`.

- [x] Folder scaffold, `.gitignore`
- [x] `AGENTS.md`, `docs/PROJECT.md`, `docs/ROADMAP.md`, Cursor rule
- [x] `configs/micro_50m_8gb.yaml` (bfloat16, 4070-targeted)
- [x] Python 3.12 venv + `requirements.txt` + `pyproject.toml`
- [x] `src/config.py`, `src/utils.py`, `tests/test_env.py`, `tests/test_config.py`
- [x] First commit + public remote — <https://github.com/ArepaMan/sol>
- [x] M1 data pipeline: `data/{prepare,train_tokenizer,tokenize}.py`, `src/data.py`
- [x] M2 EDA notebook + data card: `data/eda.ipynb`, `docs/DATA_CARD.md`,
      `docs/figures/*.png`, `src/plot_style.py`
- [x] M3 model: `src/model.py` (52,901,712 params, measured), `tests/test_model.py`
- [x] M4 training: `src/train.py`, `src/benchmark.py` — all 3 gates passed, see
      `docs/ROADMAP.md` M4 for full measured results
- [x] M5 baseline run: `experiments/001_baseline/` — **val loss 1.3569, ppl 3.88**
      (periodic 200-batch re-eval), clears the ≤3.2 target with a large margin. Use
      `checkpoints/001_baseline/latest.pt` (not `best.pt` — see run.md for why)
- [x] M6 eval harness: `src/eval.py`, `src/baselines.py`, `src/generate_samples.py`,
      `eval/prompts.jsonl` (60), `eval/rubric_scores.csv`, `docs/RUBRIC.md`,
      `docs/LIMITATIONS.md`, `tests/test_eval.py` (19 tests). **Full-val-set Sol-001
      perplexity: 3.719, 95% CI [3.693, 3.745]** — supersedes M5's periodic-eval number
      (both tell the same story). Beats trigram (23.4), unigram (379.0), uniform (32000)
      baselines by a wide margin. Rubric: grammar 4.00/5, coherence 3.15/5 (the real
      limiting factor — entity drift over long generations), on-topic 5.00/5 in-domain vs
      1.47/5 out-of-domain (expected). Found and traced (not assumed) a mojibake artifact
      in 6.20% of TinyStories' own train documents — inherited from upstream, not a bug
      in this repo. Full writeup: `eval/results.md`, `docs/ROADMAP.md` M6 section.
- [x] M7 ablations: `configs/ablations/*.yaml` (6 configs covering 8 logical arms via a
      shared run — see `002_lr_3e-4.yaml`'s header), `scripts/eval_ablation_checkpoints.py`,
      `experiments/{002_lr_sweep,003_data_scale,004_seed_variance}/results.md`.
      **Seed variance (yardstick): 4.490 ± 0.0045 ppl** across seeds 42/43/44. **LR sweep**:
      1e-3 best (4.162 ppl), 20–200× the seed-noise floor — large, real effect. **Data
      scale**: full corpus beats 100M tokens but only by 18× seed noise (4.489 vs 4.572
      ppl) — small, the milestone's stated honest negative result. `src/data.py`'s
      `max_train_tokens` cap on `BinDataset` is now actually enforced (was
      descriptive-only through M6; confirmed a no-op at the full-corpus value). Full
      writeup: `docs/ROADMAP.md` M7 section.
- [x] M8 inference + demo (code): `src/infer.py` (`SolGenerator` + CLI),
      `scripts/export_weights.py`, `app/{demo.py,requirements.txt,README.md}`,
      `docs/DEPLOY.md`, `tests/test_infer.py` (12 tests). Closes M6's EOT-stop gap —
      generation now ends itself (191 of a 300-token budget on the test prompt) instead
      of running the budget into a second story. **Reproducibility gate passes**:
      `--seed 42` twice is byte-identical (timing was moved to stderr to make that
      true). **CPU 21.6 tok/s** vs ~90 tok/s on the 4070. Export bundle **103.1 MiB**
      — first cut was 137 MiB until the tied `wte`/`lm_head` tensor's shared storage
      was preserved through the bf16 cast. `app/api.py` deliberately skipped: Gradio
      already serves `POST /gradio_api/call/stream_story`.
- [x] M8 weights published: <https://huggingface.co/SpicyGuac/sol-001> (103.1 MiB).
      Cold-cache `hf_hub_download` path verified end-to-end — 8.7 s to model-ready,
      and the generated story was byte-identical to the local-bundle run, confirming
      the uploaded weights are the tested weights.
- [x] M8 hosting pivot: HF now requires PRO for Gradio Spaces, so `app/streamlit_app.py`
      (Streamlit Community Cloud, free) is the deployed UI. Gradio kept for local/PRO.
      **1 GB RAM ceiling** is the constraint — measured 786 MB with the CUDA torch
      wheel, ~600 MB estimated with the CPU-only wheel the deploy installs.
      `@st.cache_resource` is load-bearing for memory, not just speed.
- [ ] M8 deploy + incognito check — share.streamlit.io, owner-only OAuth flow. **Not done.**
- [ ] M9 docs, spec de-drift, portfolio wiring

Verified on this machine: torch `2.6.0+cu124`, CUDA available, **bf16 supported**,
RTX 4070 Laptop (sm_89), 123 tests passing. `gradient_checkpointing` flipped to
**false** after measurement (M4) — chosen config peaks at 2344 MiB, far under the
7400 MiB target. M5's baseline run took 30.3h wall-clock (16.9h training + 12.17h the
laptop was asleep, twice — see `experiments/001_baseline/run.md`); awake-time
throughput (~20,067 tok/s) matched M4's benchmark almost exactly.

**M1 measured numbers** (see `data/processed/stats.json`, `data/tokenized/meta.json`,
`docs/DATA_CARD.md`): train 1,748,358 docs / 357,852,786 tokens (14.58% within-train
exact-dup rate — TinyStories is genuinely repetitive, see `docs/PROJECT.md`), val
15,141 docs / 2,956,183 tokens. Val's own internal dup rate is 0% — but **28.67% of
raw validation documents are exact duplicates of a train document** (cross-split
leakage, filtered out; see `docs/DATA_CARD.md` for how this was caught and fixed).
`max_train_tokens` in the config was lowered from the original 400M target to the
measured 357,852,786 — see the comment in `configs/micro_50m_8gb.yaml`.

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
