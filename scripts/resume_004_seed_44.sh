#!/usr/bin/env bash
# One-off resume for the final ablation run, 004_seed_44, after killing it
# per user instruction (6h25m with checkpoints frozen at iter 6500/6000,
# CPU/GPU activity present but severely degraded — no cooling hardware
# available). latest.pt was at iter 6000; --resume replays 6000->wherever it
# was killed deterministically before continuing fresh.
set -eo pipefail
cd /c/Users/manol/Projects/sol

LOCK=".ablation_run.lock"
if [ -e "$LOCK" ]; then
  echo "=== LOCK exists (pid $(cat "$LOCK" 2>/dev/null)) — refusing to start at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

PY=".venv/Scripts/python.exe"
echo "=== resuming 004_seed_44 at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"$PY" -m src.train --config configs/ablations/004_seed_44.yaml --resume --no-wandb 2>&1 | tee -a experiments/ablation_004_seed_44/train.log
echo "=== finished 004_seed_44 at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== ALL M7 ABLATION TRAINING RUNS COMPLETE ==="
