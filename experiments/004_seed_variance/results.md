# 004 — Seed variance

**Status: complete.** This is the ablation that makes the other two interpretable —
without it there is no way to tell a real learning-rate or data-scale effect from
run-to-run noise (`docs/ROADMAP.md` M7).

## Setup

Three runs, identical config (`configs/ablations/002_lr_3e-4.yaml`), seed only
differs: 42, 43, 44. 8000 iters each (reduced from the 40k baseline schedule —
disclosed ablation budget, `docs/PROJECT.md`), `lr=3e-4`, full train corpus
(357,852,786 tokens), `batch_size=4 × grad_accum=16 × block_size=512`.

**Seed 42 is not a separate run.** It's `configs/ablations/002_lr_3e-4.yaml`,
already trained as 002's middle-LR arm and 003's full-data arm — same config,
same seed, one checkpoint, reused rather than reproduced under a third name.
See that config's header comment.

Perplexity measured the same way as M6 (`src/eval.py`'s document-level,
full-val-set, bootstrap-CI method), via `scripts/eval_ablation_checkpoints.py`
so all six ablation checkpoints are scored identically and are directly
comparable to each other.

## Results

| Seed | Run | Val perplexity | 95% CI |
|---|---|---|---|
| 42 | `ablation_002_lr_3e-4` (shared) | 4.489 | [4.458, 4.520] |
| 43 | `ablation_004_seed_43` | 4.495 | [4.463, 4.526] |
| 44 | `ablation_004_seed_44` | 4.486 | [4.455, 4.517] |

**Mean ± sd (n=3, sample std): 4.490 ± 0.0045.** Range across all three seeds:
0.0089 ppl.

## The yardstick

**Any gap smaller than ~0.005–0.01 ppl at this training budget (8000 iters, this
config) is indistinguishable from seed noise.** A gap has to clear roughly this
bar before it's read as a real effect rather than which-seed-you-happened-to-draw.
n=3 is too underpowered for a formal significance test (a t-test on 3 samples per
arm would be close to meaningless), so this range/sd is reported directly as the
yardstick instead, per `docs/ROADMAP.md` M7's explicit instruction — an honest
substitute for statistical power we don't have, not a workaround to avoid stating
one.

**One-line conclusion:** three identically-configured runs differing only by seed
land within 0.009 ppl of each other — real effects in 002 and 003 need to clear
that bar by a wide margin to be believed, and (see those files) both do.
