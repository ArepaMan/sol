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

## Continuous integration

`.github/workflows/ci.yml` runs on every push and PR to `master`. It installs
`requirements-ci.txt` (CPU torch, not the training environment) on Python 3.12
and runs three checks, in this order:

```powershell
ruff check app/ src/infer.py scripts/export_spec.py scripts/export_weights.py scripts/plot_ablations.py tests/test_infer.py tests/test_spec_drift.py
python -m scripts.export_spec --check
pytest -m "not gpu" -q
```

Expected: **124 passed, 5 skipped, 5 deselected**. The 5 deselected are GPU
tests; the 5 skipped are `tests/test_pipeline_artifacts.py`'s corpus-dependent
checks, which have no `.jsonl`/`.bin` to read on a runner. That combination —
no GPU, no dataset — is the whole point: CI is a fresh clone, which is what
found the five red tests in M9's final QA (commit 11e16ff).

`ruff` is scoped to the modules that are already clean; ~36 pre-existing
findings in the M1–M6 files and `data/eda.ipynb` are a separate cleanup.

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

# Gate 3 — short run, then verify resume (same --run-name both times so the
# second command finds the first's checkpoint).
python -m src.train --config configs/micro_50m_8gb.yaml --run-name gate3 --max-iters 500 --no-wandb
python -m src.train --config configs/micro_50m_8gb.yaml --run-name gate3 --max-iters 1000 --resume --no-wandb
```

> **`--resume` caveat:** `--max-iters` (without an explicit `--lr-decay-iters`) also
> resets `lr_decay_iters` to match. If a resumed run passes a *different*
> `--max-iters` than the original, the LR schedule's decay horizon changes
> mid-training — harmless as long as both values stay under `warmup_iters` (still
> pure linear warmup either way, as in the example above), but a real
> discontinuity risk once either run's `max_iters` exceeds `warmup_iters`. Prefer
> the same `--max-iters` (or an explicit matching `--lr-decay-iters`) across a
> run and its resumes for a real (non-gate) training run.

```powershell
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

Note the checkpoint default is `latest.pt`, not `best.pt` — M6 re-evaluated
both on the full val set and `latest.pt` (iter 40000) wins. See
`docs/LIMITATIONS.md`.

```powershell
python -m src.infer --prompt "Once upon a time" --max-new-tokens 200 --temperature 0.8 --top-k 200 --seed 42
```

```powershell
python -m src.infer --stream --n 3 --seed 42
```

Build the deployable bundle (bf16 weights + tokenizer + config, 103 MiB), then
run the CLI and the demo against it — this is the exact path the HF Space takes:

```powershell
python -m scripts.export_weights --out export/sol-001
```

```powershell
python -m src.infer --model-dir export/sol-001 --device cpu --seed 42
```

```powershell
$env:SOL_MODEL_DIR = "export/sol-001"; python app/demo.py
```

Reproducibility check (stdout only — timing goes to stderr on purpose, so this
diff is clean):

```powershell
python -m src.infer --seed 42 > a.txt; python -m src.infer --seed 42 > b.txt; fc.exe a.txt b.txt
```

Full deployment procedure, including the HF model repo and the Space:
[`docs/DEPLOY.md`](DEPLOY.md).

## Spec sync (M9)

Regenerate the spec artifacts after **any** config change, then copy to the portfolio:

```powershell
python -m scripts.export_spec
copy docs\sol-spec.ts ..\portfolio\src\lib\sol-spec.ts
```

`tests/test_spec_drift.py` fails if you edit the YAML — or any eval artifact —
without regenerating. Verified: changing `learning_rate` in the config turns two
tests red until `export_spec` is re-run.

To check freshness without writing (what CI would run):

```powershell
python -m scripts.export_spec --check
```
