# Deploying Sol

Sol ships as two Hugging Face repos plus one local export step:

| Where | What | Why separate |
|---|---|---|
| `export/sol-001/` (local) | bf16 weights + tokenizer + config | Built from a 635 MB training checkpoint; git-ignored |
| HF **model** repo `SpicyGuac/sol-001` | the same bundle, 103 MiB | Weights don't belong in a Space repo — Spaces rebuild on every push, and a 100 MB blob in the app repo makes every rebuild re-upload it |
| HF **Space** `SpicyGuac/sol` | `app/` + `src/` only | Fetches the weights at startup with `hf_hub_download`, which caches to disk |

The GitHub repo is the source of truth for the code. The Space is a deployment
target, not a fork.

> **The two namespaces don't match**: GitHub is `ArepaMan`, Hugging Face is
> `SpicyGuac`. Every `huggingface.co` path below uses `SpicyGuac`; every
> `github.com` link in `app/demo.py` and `app/README.md` uses `ArepaMan`. Worth
> stating because a copy-paste between them fails with a 404 that reads like a
> permissions problem.

---

## 1. Export the weights bundle

```bash
python -m scripts.export_weights --out export/sol-001
```

Measured output (from `checkpoints/001_baseline/latest.pt`, iter 40000):

```
config.yaml      3.3 KiB
manifest.json    295.0 B
meta.json        460.0 B
model.pt         100.9 MiB
tokenizer.json   2.2 MiB
bundle: 103.1 MiB
```

**Why 100.9 MiB and not 137 MiB.** 52,901,712 params × 2 bytes (bf16) is
100.9 MiB — but a naive `{k: v.to(bfloat16)}` over `state_dict()` writes the
32000×592 embedding table *twice*, because `transformer.wte.weight` and
`lm_head.weight` are the same tied tensor and casting each entry independently
breaks the shared storage `torch.save` would otherwise dedupe. The export
script casts once per storage. That's 36 MiB off a cold-start download.

## 2. Smoke-test locally against the exported bundle

Do this before uploading anything. It exercises the exact code path the Space
runs, minus the network:

```bash
python -m src.infer --model-dir export/sol-001 --device cpu --seed 42
```

```bash
SOL_MODEL_DIR=export/sol-001 python app/demo.py
```

Then open <http://127.0.0.1:7860> and generate once from each tab.

## 3. Upload the weights to a model repo

Requires a Hugging Face account and a **write** token from
<https://huggingface.co/settings/tokens>. Log in from your own shell — the
token is a credential and does not belong in this repo, in a script, or in a
config file:

```bash
huggingface-cli login
```

```bash
huggingface-cli upload SpicyGuac/sol-001 export/sol-001 . --repo-type model
```

## 4. Create and push the Space

```bash
huggingface-cli repo create sol --type space --space_sdk gradio
```

The Space needs `app/` at its root plus the `src/` package it imports:

```bash
git clone https://huggingface.co/spaces/SpicyGuac/sol /tmp/sol-space
```

```bash
cp app/demo.py app/requirements.txt app/README.md /tmp/sol-space/ && mkdir -p /tmp/sol-space/src && cp src/__init__.py src/config.py src/model.py src/infer.py src/utils.py /tmp/sol-space/src/
```

`app/README.md` becomes the Space's `README.md` — its YAML header
(`sdk_version`, `app_file: demo.py`, `pinned: true`) is what configures the
Space, so it must land at the root, not in a subdirectory.

Only those five `src/` modules are needed. `train.py`, `eval.py`, `data.py`,
`baselines.py` and friends pull in `datasets`, `wandb`, `matplotlib` — none of
which are in `app/requirements.txt`, and all of which would inflate the image.

```bash
cd /tmp/sol-space && git add -A && git commit -m "Deploy Sol demo" && git push
```

---

## Cold start: what was measured, and what the mitigations are

Free CPU Spaces sleep after 48 hours idle. The first request after that pays
for container start + `pip install` (cached image, usually skipped) + Python
imports + weight download (cached to disk unless the image was rebuilt) +
model load.

**Measured locally** (RTX 4070 laptop, warm page cache, `SOL_MODEL_DIR` set so
no download happens):

| Stage | Time |
|---|---|
| `import gradio` + `import torch` + `src` | 19.6 s |
| `SolGenerator.from_pretrained` (load + to(device)) | 1.2 s |
| **Total to "model ready"** | **~21 s** |

A free Space's 2 vCPU is slower than this machine and adds a one-time ~103 MiB
download on a fresh image, so **30–90 s is the honest expected range** for a
true cold start. Replace this paragraph with the measured Space number once the
Space has been up long enough to have slept and woken.

Four mitigations, all already in the code:

1. **`pinned: true`** in `app/README.md` — pinned Spaces are less aggressively
   evicted.
2. **CPU-only torch wheel** (`--extra-index-url .../whl/cpu` in
   `app/requirements.txt`). The default PyPI wheel bundles ~2.5 GB of CUDA
   libraries a CPU Space can never use.
3. **Import-time model load** (`app/demo.py` builds `GENERATOR` at module
   scope). The cost lands in the container start, where the Space UI already
   shows a "Building"/"Starting" state, rather than on the first click, where
   it looks like a hang.
4. **An in-UI note** telling the user the first request may take ~60 s. The
   honest fix for a limitation you can't remove is to say it out loud.

## Inference latency

Measured with `python -m src.infer --model-dir export/sol-001 --device cpu
--seed 42 --max-new-tokens 200`:

| Device | Rate |
|---|---|
| RTX 4070 Laptop (bf16) | ~90 tok/s |
| This machine's CPU (fp32) | **21.6 tok/s** |

The Space's 2 vCPU will land below the local CPU number — expect roughly
15–20 tok/s there, i.e. a 200-token story in 10–13 s once the model is warm.

`src/infer.py` uses fp32 on CPU deliberately (`resolve_dtype`): CPU bf16 runs
through reference kernels on most x86 parts and is slower than fp32, and since
the weights were bf16 already, fp32 inference is a strict widening — no quality
change.

## Reproducibility

```bash
python -m src.infer --prompt "Once upon a time" --seed 42 > a.txt
python -m src.infer --prompt "Once upon a time" --seed 42 > b.txt
diff a.txt b.txt   # empty
```

Timing goes to **stderr**, not stdout, precisely so this diff is clean.

Determinism is per-device: a CUDA seed-42 sample and a CPU seed-42 sample are
both individually reproducible, but they are not the same story. Different
kernels produce different floating-point rounding, which changes the sampled
token, which changes everything after it.
