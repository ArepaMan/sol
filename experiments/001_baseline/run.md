# 001 — Baseline run

**Status: complete.** Exit criterion (val loss ≤ 3.2) cleared with a large margin.

## Setup

| | |
|---|---|
| Command | `python -m src.train --config configs/micro_50m_8gb.yaml --run-name 001_baseline --no-wandb` |
| Config | frozen copy at `experiments/001_baseline/config.yaml` |
| Git commit | `4cbf955ae3fad49178e8ff0d81fcd30f23790f59` |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU (Ada, sm_89, 8188 MiB) |
| Precision | bfloat16 |
| gradient_checkpointing | false (measured decision, M4) |
| Params | 52,901,712 (measured, M3) |
| Log | `experiments/001_baseline/train.log` |

## Why `--no-wandb`

W&B is not authenticated on this machine (`wandb.Api()` raises `UsageError: api_key not
configured`), and logging in isn't something to do without the user directly providing
credentials. Console logging to `train.log` (every `log_interval=25` iters, plus
periodic eval) serves the same purpose here — `loss_curve.png` is built directly from
parsing that log via `src/plot_curves.py`.

## Timeline

| | |
|---|---|
| Started (UTC) | 2026-08-04T17:22:56Z |
| Finished (UTC) | 2026-08-05T23:41:31Z (checkpoint mtime) |
| Wall-clock elapsed | **30.31 hours** |
| System sleep during run | 12.17h, across 2 events (see below) |
| **Actual training (awake) time** | **≈18.14 hours** |
| Implied awake throughput | ≈20,067 tok/s (1,310,720,000 tokens ÷ 18.14h) — consistent with M4's Gate 3 measurement (~21,500 tok/s) |
| max_iters | 40,000 (reached exactly — loop completes `range(0, 40000)`) |

**Wall-clock ≠ training time, and that gap is itself worth recording.** The laptop went
to sleep twice during the run — once automatically (Windows' default AC power plan
sleeps on idle unless explicitly disabled — a setting change the user declined to make,
reasonably, since a one-off portfolio training run doesn't justify a permanent system
change) and once because the user manually slept it overnight:

| Sleep event | Duration | Cause |
|---|---|---|
| ~00:51–05:51 UTC (Aug 5) | 5.00h | Automatic — AC idle timeout not disabled |
| ~05:51 UTC–13:01 UTC (Aug 5) | 7.17h | User manually slept the laptop overnight |

No further sleep occurred after 13:01 UTC — the remaining ~10.7h ran uninterrupted to
completion. Neither event affected correctness: system sleep suspends the whole process
in memory (unlike the checkpoint/resume mechanism, which exists for actual process
death, not sleep), so training picked up exactly where it left off each time with zero
lost progress or need to `--resume`. The **implied awake-time throughput (≈20,067
tok/s) is consistent with M4's Gate 3 measurement** — confirms that benchmark
generalized correctly to the full 40k-iteration run; the wall-clock overrun was
entirely a laptop-power-management story, not a training or hardware performance
problem. If replicating this without interruptions, budget the **M4-measured ~16.9h**,
not this run's ~30.3h wall-clock.

## Results

**Evaluated on the actual final checkpoint** (`latest.pt`, iter 40,000) with a larger,
less noisy 200-batch eval — not just the last periodic 100-batch training-time eval —
for the authoritative numbers:

| Metric | `latest.pt` (iter 40,000) | `best.pt` (iter 34,000) |
|---|---|---|
| Train loss | 1.2909 | 1.3147 |
| **Val loss** | **1.3569** | 1.3747 |
| Val perplexity | **3.88** | 3.95 |

**Exit criterion: val loss ≤ 3.2 — cleared at 1.3569, well under target.** The
original spec's "realistic" target (val loss 2.8–3.2, perplexity 15–25) undersold what's
achievable on TinyStories — a corpus that is, by design, narrow-vocabulary and
templated (see `docs/DATA_CARD.md`), so a 52.9M-parameter model reaches much lower loss
here than the same size would on general-domain text. This is a spec correction, not a
lucky result: the original number was a guess made before any data existed to measure
against, same as the token-count and param-count corrections in M1/M3.

**A finding worth being honest about: `best.pt` isn't actually best.** The training
loop's `best_val_loss` tracking uses the periodic 100-batch eval (`eval_iters: 100` in
the config) — small enough to be noisy. Iter 34,000 happened to land a lucky low sample
in that 100-batch estimate (1.3747 in training-time logging), but under a more thorough
200-batch re-evaluation *after* the fact, the true final checkpoint (iter 40,000, never
flagged as "best" during training because none of its periodic samples happened to beat
34,000's) is actually the better model (1.3569). **`latest.pt`, not `best.pt`, is the
checkpoint to use going forward** (M6 eval, M8 demo). Noted as a real limitation of the
current best-checkpoint selection, not silently corrected away — a larger/full-val-set
eval (M6) would resolve this properly.

**Loss curve:** `loss_curve.png` — sharp drop from init (10.4335, matching `ln(32000)
≈10.37`) to under the 3.2 target within ~1,500 iters, then a long, well-behaved plateau
from ~iter 2,000 onward. Train and val track closely throughout with no visible
overfitting — consistent with the corpus (357.85M tokens) being large relative to the
model (52.9M params) and ~3.66 epochs of exposure.

## Monitoring notes

- GPU stayed thermally healthy throughout every checked window: 70–75°C, no throttle
  signature (`nvidia-smi -q -d PERFORMANCE` showed no active throttle reasons; clock
  stayed near max boost ~2200–2265 MHz whenever the process was actively computing).
  **No thermal throttling occurred** — the wall-clock overrun was entirely sleep, not
  heat.
- `data_time_frac` stayed low throughout (2.6–4.6%), confirming the memmap dataloader
  (`src/data.py`) was never the bottleneck, consistent with M4's finding.
- Peak VRAM steady at 2344 MiB — matches Gate 2's `batch_size=4, ckpt=false` measurement
  almost exactly, confirming that benchmark's peak-memory number holds under the real
  40k-iteration loop, not just a short synthetic sweep.
- 1,600 per-step log lines, 79 periodic evals recorded in `train.log`.
