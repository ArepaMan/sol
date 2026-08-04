# Commands

Canonical CLI for Sol. No Makefile — `make` is not reliably present on Windows, so the
documented `python -m` form is the contract.

## Interpreter

The system Python is 3.14, which has **no PyTorch wheel**. Always use the venv:

```
C:\Users\manol\Projects\sol\.venv\Scripts\python.exe
```

Activate it for an interactive session:

```powershell
.\.venv\Scripts\Activate.ps1
```

Everything below assumes an activated venv (or substitute the absolute path for `python`).

## Setup (M0)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip.exe install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip.exe install -r requirements.txt
```

The `--index-url` on the first install is **not optional**. Without it pip resolves the
CPU-only wheel and every GPU path fails much later with a confusing error.

> **Always invoke scripts under `data/` as modules** (`python -m data.tokenize`), never
> by path (`python data/tokenize.py`). Running by path puts `data/` on `sys.path[0]`,
> and `data/tokenize.py` then shadows the stdlib `tokenize` module — anything that
> imports it transitively (e.g. `inspect`, pulled in by `tqdm`) fails with a confusing
> circular-import error. Every command below already uses the `-m` form.

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

Expected: `2.6.0+cu124 NVIDIA GeForce RTX 4070 Laptop GPU True`

## Tests

```powershell
pytest                      # everything
pytest -m "not gpu"         # CPU-only subset (CI, or no GPU attached)
pytest tests/test_model.py  # M3 architecture tests
```

## Data pipeline (M1)

```powershell
python -m data.prepare --config configs/micro_50m_8gb.yaml --near-dedup
python -m data.train_tokenizer --vocab-size 32000 --input data/processed/train.jsonl
python -m data.tokenize --workers 8
```

## EDA (M2)

```powershell
jupyter nbconvert --execute --to notebook --inplace data/eda.ipynb
```

## Training (M4–M5)

```powershell
# Gate 1 — overfit one batch. Loss must reach < 0.1 within 200 iters.
python -m src.train --config configs/micro_50m_8gb.yaml --overfit-batch --max-iters 200 --no-wandb

# Gate 2 — VRAM / throughput sweep. Target peak < 7400 MiB.
python -m src.benchmark --config configs/micro_50m_8gb.yaml --sweep

# Gate 3 — short run, then verify resume.
python -m src.train --config configs/micro_50m_8gb.yaml --max-iters 500
python -m src.train --config configs/micro_50m_8gb.yaml --max-iters 1000 --resume

# Baseline run.
python -m src.train --config configs/micro_50m_8gb.yaml --run-name 001_baseline
```

If VRAM is tight:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
```

## Evaluation (M6)

```powershell
python -m src.eval --checkpoint checkpoints/001_baseline/best.pt --split val
python -m src.baselines --split val
python -m src.generate_samples --checkpoint checkpoints/001_baseline/best.pt --prompts eval/prompts.jsonl
```

## Inference & demo (M8)

```powershell
python -m src.infer --checkpoint checkpoints/001_baseline/best.pt --prompt "Once upon a time" --max-new-tokens 200 --temperature 0.8 --top-k 40 --seed 42
python app/demo.py
```

## Spec sync (M9)

Regenerate the spec artifacts after **any** config change, then copy to the portfolio:

```powershell
python scripts/export_spec.py
copy docs\sol-spec.ts ..\portfolio\src\data\sol-spec.ts
```

`tests/test_spec_drift.py` fails if you edit the YAML without regenerating.
