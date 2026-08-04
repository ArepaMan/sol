# Sol — Milestone Roadmap

Ten dependency-ordered milestones. Every one ends in something demonstrable and has a
verifiable exit criterion — no milestone leaves the repo broken.

**Effort:** ~55–75 h hands-on + ~35–65 h unattended GPU, over 4–6 calendar weeks.
Legend: **[P]** = also touches the portfolio repo.

| M | Name | Hands-on | Unattended | [P] |
|---|------|----------|------------|-----|
| 0 | Env bootstrap, spec correction, public repo | 2–3 h | — | ✅ |
| 1 | Data pipeline + BPE tokenizer | 8–12 h | 1–2 h | |
| 2 | EDA notebook + data card | 5–7 h | — | |
| 3 | `src/model.py` hand-written + tests | 8–12 h | — | |
| 4 | Training loop + smoke gates + benchmark | 8–10 h | — | |
| 5 | Baseline run 001 | 1 h | 12–24 h | |
| 6 | Eval harness | 6–8 h | 1 h | |
| 7 | Ablations + seed variance | 3 h | 20–40 h | |
| 8 | Infer CLI, Gradio, HF Space | 6–8 h | — | |
| 9 | Docs, spec de-drift, portfolio wiring | 6–8 h | — | ✅ |

**Minimum viable cut** if time runs short: M0→M1→M2→M3→M4→M5→M6 (perplexity + baselines
only)→M8→M9. Drop M7's ablations and M6's manual rubric. **Never drop M2 or M4's gates** —
both are cheap and carry disproportionate interview weight.

**Stop and reassess after:** M4 (is the time budget real?), M5 (did val loss land in
2.8–3.2?), M7 (do the ablations survive seed noise?).

---

## M0 — Environment bootstrap, spec correction, public repo [P]

Python 3.14 is the system default and has no PyTorch wheel — a 3.12 venv is a hard
prerequisite for everything downstream. This milestone also corrects four spec errors
before they propagate further (see "Spec corrections" below).

**Files:** `requirements.txt`, `pyproject.toml`, `src/config.py`, `src/utils.py`,
`tests/test_env.py`, `tests/test_config.py`, `docs/COMMANDS.md`, `docs/ROADMAP.md`.

**Exit criteria**
- `torch.cuda.is_bf16_supported()` → `True`, device name contains `4070`
- `pytest` exits 0
- ≥1 commit; public GitHub remote resolves
- No **live** spec claim mentions `2070` or `float16`. Remaining occurrences are
  deliberate "this was wrong, here's why" notes — those stay, since the correction is
  itself a talking point.

**Skills:** reproducibility (7), problem framing (1).

### Spec corrections made in M0

| # | Was | Is |
|---|-----|-----|
| 1 | RTX 2070 Super, "FP16 not BF16 — Turing" | RTX 4070 Laptop (Ada, sm_89), **BF16**, no GradScaler |
| 2 | `AGENTS.md` claimed `[x] Git repo` | Repo had **zero commits**, everything untracked |
| 3 | "400M tokens ≈ 30–40 h" | 40k iters × 32,768 = **1.31B tokens ≈ 3.3 epochs**; wall-clock measured in M4 |
| 4 | litgpt vs nanoGPT unresolved | Hand-written `model.py`, nanoGPT as reference |

---

## M1 — Data pipeline: download, clean, dedupe, tokenize, shard

**Files:** `data/prepare.py` (NFKC normalise, length filter, SHA1 exact-dedup, optional
MinHash near-dedup, writes `stats.json` with every counter), `data/train_tokenizer.py`
(32k byte-level BPE, **train split only — never val**), `data/tokenize.py` (→ `uint16`
`.bin` shards + `meta.json`), `src/data.py` (`np.memmap` + `get_batch`),
`tests/test_tokenizer.py`, `tests/test_data.py`.

**Exit criteria**
- `meta.json` reports the real train-token count. TinyStories is ~470M GPT-2 tokens; an
  in-domain 32k BPE yields fewer. **If you don't clear 400M, lower `max_train_tokens`
  honestly** rather than padding.
- Val split is document-level and provably disjoint (assert on the dedup hash sets)
- `pytest tests/ -q` green

**Risks.** Tokenizer bugs are silent and catastrophic — the round-trip test is the gate,
plus an `arr.max() < vocab_size` assert before writing. Dedup can be over-aggressive on a
synthetic corpus that is *legitimately* repetitive: log the dup rate before deleting, and
if exact-dup > 15% raise the threshold and record the decision in the data card.

**Skills:** data pipeline (2), reproducibility (7).

---

## M2 — EDA notebook + data card ← primary Data Scientist signal

Kept standalone, not folded into M1. This is what turns a training project into a
DS-credible one, and it produces the figures the portfolio gallery uses.

**Files:** `data/eda.ipynb`, `docs/DATA_CARD.md`, `docs/figures/{length_dist,zipf,dedup_funnel}.png`.

Notebook covers: length distribution in chars **and** tokens with p50/p90/p99
(empirically justifying `block_size=512` — "X% of stories fit"); tokens-vs-chars
compression scatter; top-50 frequency + Zipf log-log; UNK rate; dedup funnel waterfall;
train/val length comparison with a KS test showing the split isn't skewed.

**Exit criteria**
- `jupyter nbconvert --execute --to notebook --inplace data/eda.ipynb` exits 0 on a clean kernel
- **Every number in `DATA_CARD.md` traces to `stats.json`/`meta.json`** — nothing hand-typed
- `block_size=512` justified by a measured percentile, replacing the spec's hand-wave

**Skills:** data pipeline (2), problem framing (1), honest limitations (9).

---

## M3 — `src/model.py` hand-written + architecture tests

~300 lines: `CausalSelfAttention` via `F.scaled_dot_product_attention(..., is_causal=True)`;
4× GELU `MLP`; pre-LN `Block`; `GPT` with weight tying, learned positional embeddings,
GPT-2 scaled init (`0.02/sqrt(2*n_layer)` on residual projections); `configure_optimizers`
splitting decay/no-decay groups with fused AdamW; `estimate_mfu()`; `generate()` with
temperature/top-k; per-block checkpointing via `checkpoint(..., use_reentrant=False)`.

**`tests/test_model.py` — six tests:**

| # | Test | Catches |
|---|------|---------|
| a | param count within 5% of 52M | config drift |
| b | forward shape `(B,T,vocab)` | plumbing |
| c | **causality** — perturbing `x[:, t+1:]` must not change `logits[:, t]` | the bug that silently invalidates the whole project |
| d | loss at init ≈ `ln(32000)` ≈ 10.37 ± 0.15 | broken init or weight tying |
| e | `generate()` respects `max_new_tokens`, never emits ids ≥ vocab | sampling bugs |
| f | checkpointing on/off give identical logits (atol 1e-4) | re-entrant checkpointing breaking SDPA |

Test (c) is the one to name in interviews.

**Skills:** transformer fundamentals (3).

---

## M4 — Training loop + smoke gates + VRAM/throughput benchmark

**The most important milestone for interview credibility.** Nothing long-running starts
until all three gates are green.

**Files:** `src/train.py`, `src/benchmark.py`, `tests/test_train.py`,
`experiments/000_benchmark/results.md`.

`src/train.py` carries: cosine LR + linear warmup as a **pure `get_lr(it)` function** (so
it is unit-testable without a GPU); grad accumulation; `autocast(dtype=torch.bfloat16)`
**with no GradScaler** — BF16 needs none; `clip_grad_norm_(1.0)` before the step; W&B
logging of loss/lr/grad_norm/tokens-per-s/MFU/peak-VRAM and **`data_time` vs `step_time`
separately**; checkpoint on interval and on best-val; full resume including RNG state;
`--overfit-batch` mode.

### Gate 1 — overfit a single batch
```
python -m src.train --config configs/micro_50m_8gb.yaml --overfit-batch --max-iters 200 --no-wandb
```
Loss must go **10.37 → < 0.1 within 200 iters**. A plateau above ~1.0 means a bug in the
target shift, the mask, or the optimizer param groups — debug *here*, not ten hours in.

> *"I never start a long run without proving the model can memorise one batch."*

### Gate 2 — VRAM + throughput benchmark
```
python -m src.benchmark --config configs/micro_50m_8gb.yaml --sweep
```
Measured table for at least `(b4,ckpt on)`, `(b4,ckpt off)`, `(b8,ckpt on)`,
`(b4,ckpt on,compile)`. Chosen config peak VRAM **< 7400 MiB** (~800 MiB headroom for
Windows WDDM + display). **Settle `torch.compile` on evidence:** flip it on if it buys
≥15% tokens/s and compiles in <5 min; if Triton fails on Windows, record *that* as the
reason. Either outcome is a good answer.

### Gate 3 — short real run + resume
```
python -m src.train --config configs/micro_50m_8gb.yaml --max-iters 500
```
Val loss strictly decreasing; checkpoint written; `--resume` continues within ±0.02 —
proving the resume path *before* you need it at hour 9.

### Recalibrate the time estimate here, by measurement
`hours = 40000 × 32768 / tokens_per_sec / 3600`. Expect 15k–30k tok/s → **12–24 h**.
Replace the spec's number with the measured one. **If the projection exceeds ~20 h, cut
`max_iters` to ~24000 (≈2 epochs)** rather than quietly overrunning.

**Risks.** OOM → the fallback ladder becomes a measurement, not a surprise; set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Dataloader bottleneck → the
`data_time`/`step_time` split diagnoses it (should be <3% with memmap). WDDM reserving
VRAM → benchmark with the browser closed and note the delta.

**Skills:** training (4), OOM debugging (4), reproducibility (7).

---

## M5 — Baseline run 001

```
python -m src.train --config configs/micro_50m_8gb.yaml --run-name 001_baseline
```

**Files:** `experiments/001_baseline/{config.yaml,run.md,loss_curve.png}`. `run.md`
records start/end, git sha, GPU, wall-clock, tokens/s, final val loss, W&B URL.

**Exit criteria**
- **val loss ≤ 3.2.** If higher, that's a finding — but understand why before ablating.
- Laptop-specific: run on AC power, check `nvidia-smi -q -d TEMPERATURE,PERFORMANCE`
  mid-run, record the sustained clock. Honest hardware reporting differentiates.

**Checkpoint distribution, decided here.** Full ckpt with Adam state ≈ 630 MB;
weights-only fp32 ≈ 208 MB; bf16 ≈ 104 MB. **None go in git** — `.gitignore` already
blocks `*.pt`, `checkpoints/`, `*.bin`; verify with `git status --ignored`. Weights-only
export is published to a HF model repo in M8.

---

## M6 — Evaluation harness: perplexity + baselines + rubric

**Files:** `src/eval.py` (full-val-set perplexity + bootstrap 95% CI over documents +
per-length-bucket slice), `src/baselines.py` (uniform / unigram / trigram-with-backoff),
`eval/prompts.jsonl` (60 categorised prompts), `src/generate_samples.py`,
`docs/RUBRIC.md` (anchored 1–5 descriptions + 2 worked examples),
`eval/rubric_scores.csv`, `tests/test_eval.py`.

**Exit criteria**
- Results table: uniform / unigram / trigram / Sol-001, each with ppl + CI
- ≥60 prompts scored blind, mean ± sd per dimension
- **Repetition measured automatically too** — distinct-2/3 ratios, max repeated substring
  — so the rubric isn't the only evidence

**Risk:** rubric theatre. Cap at 60 prompts, score in one blind sitting, keep the
automatic metrics as the objective backstop.

**Skills:** evaluation (5), honest limitations (9).

---

## M7 — Ablations with statistically honest reporting

| ID | Variable | Runs |
|----|----------|------|
| 002 | LR 1e-4 / 3e-4 / 1e-3 | 3 @ ~8000 iters |
| 003 | Data scale 100M vs 400M | 2 @ **equal iteration count** (data-diversity, not compute) |
| 004 | **Seed variance** 42/43/44 | 3 @ baseline config |

**004 is the point.** Report `val_loss = mean ± sd over 3 seeds`, then state plainly
whether each ablation's gap exceeds that sd. Explicitly refuse a t-test on n=3 and say
why (underpowered); report effect size against seed noise instead.

**Exit criteria**
- 3 `results.md` files, each with config diff + metric + one-line conclusion
- Seed sd reported and used as the yardstick
- **At least one negative or null result stated as such** — interviewers value this more
  than a clean win

**Skills:** ablations (6), evaluation (5), honest limitations (9).

---

## M8 — Inference CLI, Gradio app, HF Space + Hub weights

**Files:** `src/infer.py`, `app/demo.py`, `app/requirements.txt` (**CPU torch only** —
image size is the difference between a 40 s and a 4 min cold start), `app/README.md`
(Space YAML header, `pinned: true`), `docs/DEPLOY.md`, optionally `app/api.py`.

Gradio app needs an **"About / Limitations" tab**: children's stories only, won't answer
questions, no instruction tuning, 512-token context, 52M params. Model loads at module
import, not per-request.

**Weights:** bf16 weights-only (~104 MB) + `tokenizer.json` + `config.yaml` to a HF
**model** repo; the Space fetches via `hf_hub_download`. Not in git, not in the Space repo.

**Exit criteria**
- Public Space generates in a fresh incognito window
- **Cold start measured** (free CPU Spaces sleep after 48 h; expect 30–90 s). Mitigate
  with `pinned: true`, minimal requirements, import-time load, and an in-UI "first
  request may take ~60 s" note.
- CPU latency recorded (expect 15–40 tok/s at 52M)
- `src/infer.py --seed 42` twice → byte-identical output

**Skills:** deployment (8), honest limitations (9).

---

## M9 — Docs, spec de-drift, portfolio wiring [P]

### 9a — Sol docs
`README.md` to the hiring-manager pattern: Problem / Approach / **Results table** / How
to run (3 commands) / What I'd do next. Plus `docs/LIMITATIONS.md` and
`docs/INTERVIEW_NOTES.md` (the 9-skill map with the concrete artifact and number for each).

### 9b — Kill the three-way spec drift
Sol's numbers currently live in three places that will diverge: this repo's YAML, the
portfolio's `sol.mdx`, and the portfolio's `sol-spec-explorer.tsx` (three hardcoded
arrays). Fix by generating:

1. `scripts/export_spec.py` reads the YAML + `experiments/*/results.md` → emits
   `docs/spec.json` and a typed `docs/sol-spec.ts`
2. `tests/test_spec_drift.py` regenerates in-memory and asserts equality with the
   committed `spec.json` — fails if someone edits the YAML without regenerating
3. Portfolio imports the generated `.ts` instead of its local arrays
4. The sync command is documented in `docs/DEPLOY.md`

> One number in three repos → one number in one repo. That is the interview story.

### 9c — Portfolio content
`status` planned → in-qa → finished; add `liveUrl` + `repoUrl`; flip `featured` to true;
`demo.type` → `live-link`; populate `demo.gallery` with the loss curve, ablation plots,
and a Space screenshot (each caption carrying a real number); replace the placeholder SVG
thumbnail; replace every "planned/target" string with a measured one.

**The "Honest status" paragraph in `sol.mdx` is currently the honest thing to say and
will become the dishonest thing to say.** Rewriting it is part of the milestone.

**Exit criteria**
- `pytest` green including `test_spec_drift.py`
- `npm run build` green; `/projects/sol` renders live link, repo link, gallery, correct numbers
- No `2070` / `float16` / `coming-soon` / `not started` for Sol in either repo

---

## Interview-skill coverage

| # | Skill | Milestone | Artifact |
|---|-------|-----------|----------|
| 1 | Problem framing | M0, M2 | Scoped spec; `block_size` justified by measured percentile |
| 2 | Data pipeline | M1, M2 | `stats.json`, dedup funnel, `DATA_CARD.md` |
| 3 | Transformer fundamentals | M3 | Hand-written `model.py`; causality + init-loss tests |
| 4 | Training | M4, M5 | Overfit gate, VRAM benchmark, resume test |
| 5 | Evaluation | M6 | Perplexity + CI vs 3 baselines; anchored rubric; distinct-n |
| 6 | Ablations | M7 | LR, data scale, seed variance as the yardstick |
| 7 | Reproducibility | M0, M1, M9 | YAML configs, seeds, `COMMANDS.md`, spec-drift test |
| 8 | Deployment | M8 | Gradio HF Space, HF Hub weights |
| 9 | Honest limitations | M2, M6, M7, M9 | `LIMITATIONS.md`, a stated null result |
