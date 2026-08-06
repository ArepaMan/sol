#!/usr/bin/env bash
# M7 ablation runner: trains the 5 distinct configs sequentially (ablation_002_lr_3e-4
# does triple duty as 002's middle-LR arm, 003's full-data arm, and 004's seed=42
# arm — see configs/ablations/002_lr_3e-4.yaml's header — so only 5 runs are needed
# to cover 002 (3 arms) + 003 (2 arms) + 004 (3 arms) = 8 logical arms).
#
# Usage: bash scripts/run_ablations.sh   (run from repo root; takes ~18-22h GPU time)
set -e

PY=".venv/Scripts/python.exe"
RUNS=(
  "configs/ablations/002_lr_1e-4.yaml"
  "configs/ablations/002_lr_3e-4.yaml"
  "configs/ablations/002_lr_1e-3.yaml"
  "configs/ablations/003_data_100m.yaml"
  "configs/ablations/004_seed_43.yaml"
  "configs/ablations/004_seed_44.yaml"
)

for cfg in "${RUNS[@]}"; do
  name=$(basename "$cfg" .yaml)
  exp_dir="experiments/ablation_${name}"
  mkdir -p "$exp_dir"
  cp "$cfg" "$exp_dir/config.yaml"
  echo "=== starting $name at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  "$PY" -m src.train --config "$cfg" --no-wandb 2>&1 | tee "$exp_dir/train.log"
  echo "=== finished $name at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
done

echo "=== all ablation runs complete ==="
