# 003 — Data scale (100M vs full corpus, equal iteration count)

**Status: complete.**

## Setup

Two runs, identical config except `data.max_train_tokens`:
`configs/ablations/003_data_100m.yaml` (100,000,000 tokens) vs
`configs/ablations/002_lr_3e-4.yaml` (357,852,786 tokens — the full measured
corpus; reused as this arm rather than re-run, since it's already the exact
same config). Both trained for exactly 8000 iters / 262,144,000 tokens
**processed** — this is the point of "equal iteration count": the 100M-token
arm repeats its slice ~2.62× over the run, the full-data arm sees ~0.73 epochs
without repetition. The comparison is about data *diversity*, not compute
(`docs/ROADMAP.md` M7).

`max_train_tokens` now actually restricts `BinDataset`'s sampling range
(`src/data.py`) — through M5/M6 it was descriptive only. See that module's
docstring.

Perplexity via `scripts/eval_ablation_checkpoints.py` (document-level,
full-val-set, bootstrap-CI — same method as M6, same val set).

## Results

| Train data | Val perplexity | 95% CI | Δ from full |
|---|---|---|---|
| 100M tokens (2.62 passes) | 4.572 | [4.540, 4.605] | +0.083 |
| **Full corpus, 357.85M tokens** (0.73 passes) | 4.489 | [4.458, 4.520] | — |

## Reading this against the seed-variance yardstick

Seed-to-seed noise at this config is ±0.0045 ppl (`experiments/004_seed_variance/results.md`).
The 0.083 ppl gap is **~18× that noise** — technically real signal, not draw luck.

**But stated plainly, this is a small, near-null effect in practical terms.**
18× seed noise clears the statistical bar, but 0.083 ppl is a ~1.8% relative
difference — a fraction of the ~0.9 ppl swing the LR sweep produced from a
hyperparameter change with zero extra data cost. At this compute budget
(8000 iters), **more data helps, but only marginally** — repeating a smaller,
still-diverse 100M-token slice 2.6 times costs surprisingly little compared to
seeing 2.6× less repetition from the full corpus. This is the honest negative
result this milestone's exit criteria ask for: data scale, the variable that
sounds like it should matter most, mattered the least of anything tested here.

**One-line conclusion:** more training data helps (100M loses to the full
corpus by 18× the seed-noise floor), but the effect is small relative to the
LR sweep's — TinyStories' internal repetitiveness (`docs/DATA_CARD.md`,
14.58% within-train exact-dup rate) likely makes a 100M-token slice a less
punishing cut than it would be on a less templated corpus.
