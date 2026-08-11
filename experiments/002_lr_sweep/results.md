# 002 — Learning rate sweep

**Status: complete.**

## Setup

Three runs, identical config except `learning_rate` / `min_lr` (10:1 peak:min
ratio held constant across arms): `configs/ablations/002_lr_{1e-4,3e-4,1e-3}.yaml`.
8000 iters each, seed 42, full train corpus, `warmup_iters=1000` held at the
same absolute value across all three arms (not rescaled) so a warmup-fraction
difference can't confound the learning-rate comparison. Full detail and
rationale in each config's header comment.

Perplexity via `scripts/eval_ablation_checkpoints.py` (document-level,
full-val-set, bootstrap-CI — same method as M6, same val set).

## Results

| Learning rate | min_lr | Val perplexity | 95% CI | Δ from 3e-4 |
|---|---|---|---|---|
| 1e-4 | 1e-5 | 5.380 | [5.342, 5.418] | +0.891 |
| **3e-4** (baseline) | 3e-5 | 4.489 | [4.458, 4.520] | — |
| 1e-3 | 1e-4 | **4.162** | [4.133, 4.190] | −0.327 |

## Reading this against the seed-variance yardstick

Seed-to-seed noise at this config is ±0.0045 ppl (`experiments/004_seed_variance/results.md`).
Every gap here is **20–200× that noise** — this is real signal, not which-seed-you-drew.

- **1e-4 is clearly worse** (+0.891 ppl, ~198× the seed sd): at only 8000 iters,
  the lower peak LR simply hasn't covered enough ground yet. This is the
  expected, unsurprising direction — flagged so the sweep isn't read as "we
  tried three arbitrary values," it's a real trade-off with a real loser.
- **1e-3 is best at this budget** (−0.327 ppl, ~73× the seed sd) — a real,
  fairly large win over the project's own baseline LR (3e-4, used for the full
  40k-iter run in `experiments/001_baseline/`) when the schedule is this short.

**Scope caveat, stated plainly:** this ablation ran at 8000 iters, not the
baseline's 40000. "1e-3 wins here" is not a claim that 1e-3 would beat 3e-4 over
a full 40k-iter run — a higher LR covering more ground *faster* in a short
schedule is a different question from which LR reaches the lowest loss over a
much longer one (where a too-high LR more often destabilizes or plateaus worse).
This ablation answers the short-schedule question it was actually run at; it
does not retroactively second-guess the baseline's 3e-4 choice for the full run.

**One-line conclusion:** learning rate has the largest, clearest effect of any
ablation in M7 — tens of times larger than seed noise — and within this
8000-iter budget, higher (1e-3) beats the project's baseline LR (3e-4), which
beats a too-low LR (1e-4), in the expected direction.
