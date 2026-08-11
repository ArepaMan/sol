#!/usr/bin/env bash
# Second resume attempt. The first resume (run_ablations_remaining.sh) also
# died when the harness session that launched it was torn down — the
# Bash tool's run_in_background is apparently a child of the Claude Code
# session process, not a true OS-detached process, so it does not survive
# the session dying. This time launched via PowerShell Start-Process
# specifically to get a real detached process.
#
# A lock file guards against a second concurrent invocation of this script
# (observed once: 5 separate sets of python processes ended up running
# simultaneously against the same experiment between two check-ins, cause
# unconfirmed — possibly a retry artifact from the very first crashed
# run_ablations.sh invocation's "fork: resource temporarily unavailable"
# loop. Checkpoints were unaffected that time — torch.save only happens at
# defined boundaries, and both files still loaded cleanly — but a lock
# makes any repeat fail loudly instead of silently colliding again).
#
# `set -o pipefail` is load-bearing, not decoration: without it, `python |
# tee` reports the exit status of `tee` (always 0), not python's. A killed
# or crashed python process was silently treated as "finished successfully"
# by `set -e`, and the script cascaded straight into the next ablation arm
# — this actually happened (killing what looked like a stalled
# 003_data_100m mid-run cascaded through 004_seed_43 and into 004_seed_44
# within seconds, both abandoned with zero real training). pipefail makes
# any future kill or crash stop the script here instead of silently
# skipping ahead.
set -eo pipefail
cd /c/Users/manol/Projects/sol

LOCK=".ablation_run.lock"
if [ -e "$LOCK" ]; then
  echo "=== LOCK $LOCK exists (pid $(cat "$LOCK" 2>/dev/null)) — refusing to start a second run at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

PY=".venv/Scripts/python.exe"

if [ -f "checkpoints/ablation_003_data_100m/latest.pt" ]; then
  echo "=== resuming 003_data_100m at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  "$PY" -m src.train --config configs/ablations/003_data_100m.yaml --resume --no-wandb 2>&1 | tee -a experiments/ablation_003_data_100m/train.log
  echo "=== finished 003_data_100m at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
fi

for cfg in configs/ablations/004_seed_43.yaml configs/ablations/004_seed_44.yaml; do
  name=$(basename "$cfg" .yaml)
  exp_dir="experiments/ablation_${name}"
  mkdir -p "$exp_dir"
  cp "$cfg" "$exp_dir/config.yaml"
  echo "=== starting $name at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  "$PY" -m src.train --config "$cfg" --no-wandb 2>&1 | tee "$exp_dir/train.log"
  echo "=== finished $name at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
done

echo "=== all remaining ablation runs complete ==="
