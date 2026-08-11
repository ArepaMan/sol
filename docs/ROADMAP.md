# Sol — Milestone Roadmap

Ten dependency-ordered milestones. Every one ends in something demonstrable and has a
verifiable exit criterion — no milestone leaves the repo broken.

**Effort:** ~55–75 h hands-on + ~35–65 h unattended GPU, over 4–6 calendar weeks.
Legend: **[P]** = also touches the portfolio repo.

| M | Name | Hands-on | Unattended | [P] |
|---|------|----------|------------|-----|
| 0 | ✅ Env bootstrap, spec correction, public repo | 2–3 h | — | ✅ |
| 1 | ✅ Data pipeline + BPE tokenizer | 8–12 h | 1–2 h | |
| 2 | ✅ EDA notebook + data card | 5–7 h | — | |
| 3 | ✅ `src/model.py` hand-written + tests | 8–12 h | — | |
| 4 | ✅ Training loop + smoke gates + benchmark | 8–10 h | — | |
| 5 | ✅ Baseline run 001 | 1 h | 12–24 h | |
| 6 | ✅ Eval harness | 6–8 h | 1 h | |
| 7 | ✅ Ablations + seed variance | 3 h | 20–40 h | |
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

## M1 — Data pipeline: download, clean, dedupe, tokenize, shard ✅ done

**Files:** `data/prepare.py` (NFKC normalise, length filter, SHA1 exact-dedup, optional
MinHash near-dedup, writes `stats.json` with every counter), `data/near_dedup.py`
(MinHash + LSH banding, salted-hash implementation — see below), `data/train_tokenizer.py`
(32k byte-level BPE, **train split only — never val**), `data/tokenize.py` (→ `uint16`
`.bin` shards + `meta.json`), `src/data.py` (`np.memmap` + `get_batch`),
`tests/test_tokenizer.py`, `tests/test_data.py`, `tests/test_prepare.py`,
`tests/test_pipeline_artifacts.py` (integration tests against the real generated files).

**Decision made here:** TinyStories ships its own train/validation split (native, held
out by the dataset authors) rather than one pool to carve 5% from. Used as-is;
`config.data.val_split` is now informational only. See "Resolved decisions" above.

**Measured results**

| | train | validation |
|---|---|---|
| raw | 2,119,719 | 21,990 |
| dropped (length/dup) | 371,361 | 6,849 |
| kept | 1,748,358 | 15,141 |
| tokens (32k BPE) | 357,852,786 | 2,956,183 |
| exact-dup rate (within own split) | 14.58% | 0.00% |
| cross-split leakage (val docs also in train) | — | 28.67% (6,304 docs, filtered) |

`max_train_tokens` in the config was lowered from the 400M target to the measured
357,852,786 — the exit criterion below, applied. Near-dedup was implemented and unit
tested but not run on the full corpus: the train dup rate sits just under the 15%
threshold that would have triggered it (see `docs/PROJECT.md`).

**Exit criteria**
- ✅ `meta.json` reports the real train-token count (357,852,786). Didn't clear the
  400M target, so `max_train_tokens` was **lowered honestly** rather than padded.
- ✅ Val split is document-level and provably disjoint — `tests/test_pipeline_artifacts.py`
  recomputes hashes from the actual written files and checks the intersection is empty,
  rather than trusting prepare.py's self-reported counter alone.
- ✅ `pytest -q` green (54 tests, incl. real-artifact integration tests)

**Risks, and what actually happened.** Tokenizer bugs are silent and catastrophic — hit
one for real: the first `BpeTrainer` had no `initial_alphabet`, so bytes absent from a
tiny test corpus (digits, uppercase, punctuation) silently fell back to `<|unk|>` instead
of the lossless byte-level guarantee. The round-trip test caught it immediately; fix was
one line (`initial_alphabet=pre_tokenizers.ByteLevel.alphabet()`). Also hit an int64
overflow in the first MinHash implementation (`a * hash` with both operands near 2^61 —
classic modular-hash mistake); replaced with salted BLAKE2B hashing per permutation,
which uses Python's arbitrary-precision ints and can't overflow.

A third bug surfaced while writing the M2 data card, not by a test failing: train and
validation were originally deduped against one *shared* hash set (train first), which
silently absorbed every cross-split duplicate into validation's own exact-dup counter
before the separate leakage-check pass ran — making that pass structurally report 0
regardless of the true rate. Fixed with independent per-split hash sets plus a real
leakage pass; `tests/test_prepare.py` now pins both the fix and the original bug pattern
as regression tests. The corrected accounting revealed validation's dup rate is
genuinely 0% *within itself* — the entire 28.67% figure was cross-split leakage (val
documents also present in the 2.1M-story train set), a more interesting finding than
"validation is repetitive" and one with a real implication for interpreting val loss
(see `docs/DATA_CARD.md`).

**Skills:** data pipeline (2), reproducibility (7).

---

## M2 — EDA notebook + data card ← primary Data Scientist signal ✅ done

Kept standalone, not folded into M1. This is what turns a training project into a
DS-credible one, and it produces the figures the portfolio gallery uses.

**Files:** `data/eda_utils.py` (unit-tested pure functions — length/frequency
extraction, kept out of the notebook so the logic is testable), `data/eda.ipynb`
(generated by `data/_build_eda_notebook.py`, kept as reviewable Python rather than
hand-edited JSON), `src/plot_style.py` (shared validated palette + figure export,
reused by M7's ablation plots later), `docs/DATA_CARD.md`,
`docs/figures/{length_dist,zipf,dedup_funnel}.png`.

Notebook covers: length distribution in chars **and** tokens with p50/p90/p99
(empirically justifying `block_size=512`); tokens-vs-chars compression scatter;
top-50 frequency + Zipf log-log; UNK rate; dedup funnel waterfall; train/val length
comparison with a KS test.

**Measured results**

- **`block_size=512` justified by measurement:** train token-length p50=184,
  p90=298, **p99=479** — 99.74% of documents fit whole within 512 tokens.
- **UNK rate: 0.0000%** across all 357.85M train tokens — confirms the byte-level
  BPE lossless guarantee held at full scale, not just in the M1 unit tests.
- Train/val length KS statistic: 0.043 (chars) / 0.047 (tokens) — close distributions.
- Mean compression: 4.19 chars/token.
- All 3 figures exported at ≥1200px (1503–1950px actual).

**Exit criteria**
- ✅ `jupyter nbconvert --execute --to notebook --inplace data/eda.ipynb` exits 0
- ✅ Every number in `DATA_CARD.md` traces to `stats.json`/`meta.json`/the notebook's
  printed summary cell — nothing hand-typed
- ✅ `block_size=512` justified by a measured percentile
- ✅ 3 figures ≥1200px wide (pinned by `tests/test_plot_style.py`)

**What actually happened.** Two real footguns, both now documented:
`_build_eda_notebook.py` must be run as `python -m data._build_eda_notebook`, not by
path — `data/tokenize.py` shadows the stdlib `tokenize` module the moment `data/`
lands on `sys.path[0]`, which produces a confusing circular-import error in `inspect`
(pulled in transitively by `tqdm`). And the first length-distribution figure overlaid
train (n=1.75M) against val (n=15k) as raw counts, rendering val as an invisible flat
line — switched to `density=True` so the comparison shows what it's meant to (shape,
not count).

Writing this milestone's data card is also what surfaced the cross-split-leakage
accounting bug described in M1 above — a case of documentation work catching a real
bug, not just recording results.

**Exit criteria**
- `jupyter nbconvert --execute --to notebook --inplace data/eda.ipynb` exits 0 on a clean kernel
- **Every number in `DATA_CARD.md` traces to `stats.json`/`meta.json`** — nothing hand-typed
- `block_size=512` justified by a measured percentile, replacing the spec's hand-wave

**Skills:** data pipeline (2), problem framing (1), honest limitations (9).

---

## M3 — `src/model.py` hand-written + architecture tests ✅ done

~230 lines: `CausalSelfAttention` via `F.scaled_dot_product_attention(..., is_causal=True)`;
4× GELU `MLP`; pre-LN `Block`; `GPT` with weight tying, learned positional embeddings,
GPT-2 scaled init (`0.02/sqrt(2*n_layer)` on residual projections); `configure_optimizers`
splitting decay/no-decay groups with fused AdamW; `estimate_mfu()`; `generate()` with
temperature/top-k; per-block checkpointing via `checkpoint(..., use_reentrant=False)`.

**Decision made here:** the config's `n_embd` changed from the originally-guessed 512 to
**592**. With weight tying and every other value locked (8 layers, 512 context, 32k vocab),
512 measures at 41.8M params — a ~20% gap from the project's "~52M" branding. 592 (=74×8,
keeps `head_dim` an integer) measures at **52,901,712** — 1.73% off — without touching
anything else already documented. See `configs/micro_50m_8gb.yaml`'s header comment and
`docs/PROJECT.md` for the full reasoning (untying weights instead would hit 58.2M but
reverses an already-deliberate design decision for no good reason).

**`tests/test_model.py` — 12 tests, not the originally-planned 6:**

| Test | Catches |
|------|---------|
| param count == 52,901,712, within 5% of 52M | config drift |
| forward shape `(B,T,vocab)` | plumbing |
| **causality** — perturbing `x[:, t+1:]` must not change `logits[:, t]` | the bug that silently invalidates the whole project |
| loss at init ≈ `ln(32000)` ≈ 10.37 ± 0.15 | broken init or weight tying |
| `generate()` respects `max_new_tokens`, never emits ids ≥ vocab, preserves prompt prefix | sampling bugs |
| `generate()` with `top_k=1` is seed-independent (argmax every step) | sampling non-determinism |
| checkpointing on/off give identical logits (atol 1e-4) | re-entrant checkpointing breaking SDPA |
| `wte.weight is lm_head.weight` + param count isn't double-counted | weight-tying bugs |
| sequence longer than `block_size` raises | silent truncation/index errors |
| `configure_optimizers` decay/no-decay split is dim-based and correct | wrong params getting weight-decayed |
| **(gpu)** full ~53M-param model, real config, bf16 autocast, forward+backward on the actual RTX 4070 | architecture bugs that only appear at real scale/precision, not the tiny CPU test config |

Test (c), causality, is the one to name in interviews — and it was verified to actually
catch a broken implementation, not just pass vacuously: monkeypatching
`F.scaled_dot_product_attention` to force `is_causal=False` flips the assertion to fail,
confirming the test has teeth.

**Quick VRAM preview** (not the formal M4 benchmark): one forward+backward at the real
config's `batch_size=4`, bf16, gradient checkpointing on — **1043 MiB** peak. Comfortably
under the 7400 MiB target, though optimizer state and the full training loop (M4) will
add more.

**Skills:** transformer fundamentals (3).

---

## M4 — Training loop + smoke gates + VRAM/throughput benchmark ✅ done

**The most important milestone for interview credibility.** Nothing long-running starts
until all three gates are green.

**Files:** `src/train.py`, `src/benchmark.py`, `tests/test_train.py`,
`tests/test_benchmark.py`, `tests/test_checkpoint.py`, `experiments/000_benchmark/results.md`.

`src/train.py` carries: cosine LR + linear warmup as a **pure `get_lr(it)` function** (so
it is unit-testable without a GPU); grad accumulation; `autocast(dtype=torch.bfloat16)`
**with no GradScaler** — BF16 needs none; `clip_grad_norm_(1.0)` before the step; console
+ optional W&B logging of loss/lr/grad_norm/tokens-per-s/peak-VRAM and **`data_time` vs
`step_time` separately**; checkpoint on interval and on best-val; full resume including
RNG state (python/numpy/torch/cuda **and** `BinDataset`'s own per-instance `Generator` —
see `src/data.py`); `--overfit-batch` mode.

### Gate 1 — overfit a single batch ✅
```
python -m src.train --config configs/micro_50m_8gb.yaml --overfit-batch --max-iters 260 --no-wandb
```
**Result: loss 10.4330 → 0.0224 by iter 240** (200 iters technically cleared the <0.1
bar at 0.0996, but that's a photo finish for a gate whose whole point is a *confident*
signal — extended to 260 for real headroom). Val/checkpoint mechanics untouched.

> *"I never start a long run without proving the model can memorise one batch."*

### Gate 2 — VRAM + throughput benchmark ✅
```
python -m src.benchmark --config configs/micro_50m_8gb.yaml --sweep
```
**Result** (`experiments/000_benchmark/results.md`, RTX 4070 Laptop, bf16):

| batch_size | ckpt | compile | peak VRAM | tokens/s | status |
|---|---|---|---|---|---|
| 4 | True | False | 1460 MiB | 22,484 | ok |
| **4** | **False** | **False** | **2137 MiB** | **28,722** | **ok — chosen** |
| 8 | True | False | 2247 MiB | 23,110 | ok |
| 4 | True | True | — | — | Triton not installed |
| 16 | False | False | 6337 MiB | 30,262 | ok |
| 32 | False | False | 11965 MiB | 4,934 | ⚠️ shared-memory spill |
| 16 | True | False | 3836 MiB | 22,901 | ok |
| 32 | True | False | 7016 MiB | 22,139 | ok |
| 64 | False | False | — | — | OOM |

**`compile`: settled false, by measurement.** Triton isn't installed on this Windows
setup — a real `BackendCompilerFailed`, not a guess.

**`gradient_checkpointing`: flipped from the originally-"required" true to false** —
raised to the user rather than decided silently, since it reversed a documented hard
constraint. At the chosen `batch_size=4`, checkpointing cost ~26% throughput (17k vs
21.5–28.7k tok/s, depending on measurement) for VRAM headroom that isn't needed (2137
MiB vs the 7400 MiB target). See `configs/micro_50m_8gb.yaml`'s header comment and
`docs/PROJECT.md` for the full reasoning.

**A real footgun found while probing headroom, not guessed:** `batch_size=32` without
checkpointing doesn't cleanly OOM — `peak_vram` reports **11,965 MiB, more than the
card's 8,188 MiB physical total**, while `ok=True` and throughput silently collapses to
~15% of normal (4,934 vs ~30k tok/s at batch 16). CUDA on Windows falls back to slow
shared/system memory instead of raising. `BenchmarkResult.suspected_shared_memory_spill`
exists specifically to catch this instead of reading a large-but-"fine" VRAM number and
missing what actually happened. A clean OOM only appears at `batch_size=64`.

### Gate 3 — short real run + resume ✅
```
python -m src.train --config configs/micro_50m_8gb.yaml --run-name gate3 --max-iters 500 --eval-interval 100 --eval-iters 20 --no-wandb
python -m src.train --config configs/micro_50m_8gb.yaml --run-name gate3 --max-iters 1000 --eval-interval 100 --eval-iters 20 --resume --no-wandb
```
**Result:** val loss strictly decreasing across both runs — 6.90 → 4.76 → 4.02 → 3.49
(part 1) → 2.98 → 2.81 → 2.68 → 2.49 (part 2, post-resume). No discontinuity at the
resume boundary (iter 480 train loss 3.2602 → iter 500 3.2703, noise-level). Data
loading overhead stayed at 0.5–4.2% throughout — confirms the memmap loader isn't the
bottleneck.

**A real bug caught here, not hypothesized:** the first resume attempt crashed —
`TypeError: RNG state must be a torch.ByteTensor`. `torch.load(..., map_location=device)`
moves *every* tensor in the checkpoint onto that device, including the RNG state byte
tensor, which `torch.set_rng_state` specifically rejects unless it's on CPU. Fixed by
explicitly `.cpu()`-ing the RNG tensors in `_restore_rng_state` before restoring them.
`tests/test_checkpoint.py::test_checkpoint_round_trip_cuda` pins this — verified it
actually catches the regression by reverting the fix and confirming the test fails with
the exact original error, then restoring it.

**A latent caveat documented, not fixed:** `--max-iters` (without an explicit
`--lr-decay-iters`) also resets `lr_decay_iters` to match. A resumed run passing a
*different* `--max-iters` than the original changes the LR decay horizon mid-training.
Harmless here (both gate runs' `max_iters` stayed under `warmup_iters=1000`, so both
remained pure linear warmup with no discontinuity) but a real risk once either value
exceeds `warmup_iters`. Documented in `docs/COMMANDS.md`; recommendation is to keep
`--max-iters` (or an explicit `--lr-decay-iters`) consistent across a run and its resumes.

### Time estimate, recalibrated by measurement, not guessed
Isolated benchmark (b4, no-ckpt): 28,722 tok/s → 12.7 h for 40,000 iters. **Real
production loop** (Gate 3, steady-state, includes eval + checkpoint overhead): **21,500
tok/s → 16.9 h.** The gap between the two is itself a finding — an isolated
micro-benchmark optimistically undercounts real-loop overhead. **16.9 h is under the ~20
h cutoff, so `max_iters` stays at 40,000** rather than being cut to ~24,000.

**Skills:** training (4), OOM debugging (4), reproducibility (7).

---

## M5 — Baseline run 001 ✅ done

```
python -m src.train --config configs/micro_50m_8gb.yaml --run-name 001_baseline --no-wandb
```

**Files:** `experiments/001_baseline/{config.yaml,run.md,loss_curve.png,train.log}`.
`src/plot_curves.py` (new, shared with M7) parses `train.log` and plots the curve.

**Result: val loss 1.3569, perplexity 3.88 — clears the ≤3.2 exit criterion with a
large margin.** (`--no-wandb`: W&B isn't authenticated on this machine, and logging in
isn't something to do without the user's own credentials — console log to `train.log`
served the same purpose.)

**Exit criteria**
- ✅ **val loss ≤ 3.2** — 1.3569 achieved. The original spec's 2.8–3.2 "realistic"
  target undersold TinyStories: a narrow-vocabulary, templated corpus (`docs/DATA_CARD.md`)
  lets a 52.9M-param model reach much lower loss than the same size would on
  general-domain text. A spec correction, not luck — same pattern as the M1/M3 corrections.
- ✅ Laptop-specific: ran on AC power; `nvidia-smi -q -d PERFORMANCE` showed no active
  throttle reasons throughout, clock stayed near max boost whenever computing. **No
  thermal throttling** — see the wall-clock story below, which was a different problem.

**A finding worth being honest about: `best.pt` wasn't actually best.** Training-time
"best" tracking uses the periodic 100-batch eval, which is noisy enough that iter
34,000 (val 1.3747 in that noisy estimate) got flagged best, while the true final
checkpoint (iter 40,000) re-evaluated at 1.3569 under a more thorough 200-batch pass —
actually better. **Use `latest.pt`, not `best.pt`, going forward** (M6, M8). A real
limitation of small-sample best-checkpoint selection, left visible rather than quietly
fixed — M6's full-val-set eval is the proper fix.

**Wall-clock overrun, and why it doesn't count against the model.** The run took 30.3h
wall-clock against an M4-measured 16.9h estimate — but 12.17h of that was the laptop
asleep (once automatic, once the user manually sleeping it overnight; the user declined
to permanently disable AC sleep for one run). System sleep suspends the whole process in
memory, unaffected by (and not needing) the checkpoint/resume mechanism built for actual
process death. The implied awake-time throughput (≈20,067 tok/s from total tokens ÷
18.14h awake) matches M4's Gate 3 measurement almost exactly — confirming the benchmark
generalized correctly to the full run; **the overrun was a laptop-power-management
story, not a training or hardware problem.** Full breakdown in `experiments/001_baseline/run.md`.

**Checkpoint distribution, decided here.** Full ckpt with Adam state ≈ 605 MB (measured:
`best.pt`/`latest.pt` both ~605 MB). **None go in git** — `.gitignore` already blocks
`*.pt`, `checkpoints/`, `*.bin`; verified with `git status --ignored`. Weights-only
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

### M6 — measured results

Full writeup: [`eval/results.md`](../eval/results.md), scores:
[`eval/rubric_scores.csv`](../eval/rubric_scores.csv), repetition:
[`eval/repetition_summary.md`](../eval/repetition_summary.md), limitations rollup:
[`docs/LIMITATIONS.md`](LIMITATIONS.md).

| Model | Perplexity | 95% CI (10k-replicate document bootstrap) |
|---|---|---|
| uniform | 32000.000 | [32000.000, 32000.000] |
| unigram | 379.010 | [377.499, 380.497] |
| trigram (stupid backoff) | 23.425 | [23.246, 23.600] |
| **Sol-001** | **3.719** | **[3.693, 3.745]** |

Evaluated on the **full val set** (15,141 docs, 2,955,739 tokens, 28 truncated to
`block_size`), not a periodic training-time sample — this supersedes the 200-batch
re-eval number from M5's `run.md` (ppl 3.88) with a more thorough, full-coverage,
per-document measurement (ppl 3.719). Both numbers tell the same story (Sol-001 clears
its target by a wide margin); M6's is the one to cite going forward.

**Deviation from the original scope, stated plainly:** the trigram baseline is fit on a
10M-token prefix of train, not the full 357.85M-token corpus — a documented scope
decision (`src/baselines.py`), not a shortcut hidden from the results. **Model eval
batch size is 4, not a larger value** — 16 and 64 both measured OOM on the 8 GB card
(the 32k-vocab logits tensor, promoted to float32 inside `F.cross_entropy`, dominates
memory at `block_size=512`); 4 matches the training config's own micro-batch size, which
is not a coincidence.

**60/60 prompts scored** (15 each: story-start, dialogue, continuation, out-of-domain),
single-rater per `docs/RUBRIC.md`. Rubric mean: grammar 4.00/5, coherence **3.15/5**
(the real limiting factor — entity/character drift over ~150-token generations, not
grammar), on-topic 5.00/5 in-domain vs **1.47/5 out-of-domain** (the correct, expected
failure mode for a model with zero non-narrative training exposure). Automatic backstop:
distinct-2 0.933, distinct-3 0.984 overall — the rubric's qualitative repetition score
(3.98/5) and the automatic metrics broadly agree, per the M6 risk note.

**A real, traced-not-assumed finding:** 6.20% of TinyStories train documents
(108,464/1,748,358) contain a mojibake artifact inherited from the dataset's own
upstream pipeline (curly quotes double-encoded via CP1252-as-UTF-8) — confirmed by
grepping the raw corpus, not guessed from the symptom. Sol-001 reproduces it faithfully
in a handful of the 60 generations. Not a bug in this repo (tokenizer round-trips it
exactly); see `docs/DATA_CARD.md` and `docs/LIMITATIONS.md`.

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

### M7 — measured results

Full writeups: [`experiments/002_lr_sweep/results.md`](../experiments/002_lr_sweep/results.md),
[`experiments/003_data_scale/results.md`](../experiments/003_data_scale/results.md),
[`experiments/004_seed_variance/results.md`](../experiments/004_seed_variance/results.md).
All six checkpoints scored with the same document-level, full-val-set,
bootstrap-CI method as M6 (`scripts/eval_ablation_checkpoints.py`), so every
number below is directly comparable.

**Seed variance (the yardstick): 4.490 ± 0.0045 ppl across seeds 42/43/44** —
n=3 is too underpowered for a t-test, so this range (max−min = 0.0089 ppl) is
reported directly and used to judge the other two ablations, per this
milestone's own instruction rather than a workaround for lacking one.

| Ablation | Result | vs seed-noise floor |
|---|---|---|
| **002 LR sweep** | 1e-4: 5.380 · **3e-4: 4.489** · 1e-3: **4.162** (best) | gaps are 20–200× seed sd — clear real signal |
| **003 data scale** | 100M: 4.572 · full corpus: 4.489 | gap is 18× seed sd (real, but small — see below) |

**002 is the clear, large effect.** At this 8000-iter budget, higher LR (1e-3)
beats the project's own baseline LR (3e-4), which beats a too-low LR (1e-4) —
expected direction, real magnitude (up to ~0.9 ppl, vs ~0.005 ppl of seed
noise). Explicitly scoped: this does **not** claim 1e-3 would beat 3e-4 over
the full 40k-iter baseline run — a short-schedule "covers more ground faster"
result is a different question from long-run stability.

**003 is this milestone's honest negative/null result, stated as such per the
exit criteria.** More data does help (100M loses to the full corpus by 18× the
seed-noise floor — real, not noise) but the effect is small: a ~1.8% relative
perplexity difference, dwarfed by the LR sweep's ~20% swing. Data scale is the
variable that sounds like it should matter most and mattered the least of
anything tested — most likely because TinyStories' own internal repetitiveness
(`docs/DATA_CARD.md`, 14.58% within-train exact-dup rate) makes a 100M-token
slice a less punishing cut than it would be on a less templated corpus.

**Real infrastructure findings from this milestone, not just training results:**
`data.max_train_tokens` was descriptive-only through M6 — `src/data.py`'s
`BinDataset` now actually enforces it for `003`'s data-scale arm (confirmed a
no-op at the full corpus value, so earlier baselines are unaffected). Also: a
runner-script bug (`python | tee` masking a killed process's exit code, no
`set -o pipefail`) caused one real incident — killing what looked like a
stalled run silently cascaded the pipeline into the next ablation arm before
the first had reached its full iteration count. Caught, fixed, and the
affected run (`003_data_100m`) was re-resumed to completion from its last
checkpoint before these numbers were measured — not silently patched over.

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
