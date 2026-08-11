# Deploying Sol

Sol ships as three pieces:

| Where | What | Why separate |
|---|---|---|
| `export/sol-001/` (local) | bf16 weights + tokenizer + config | Built from a 635 MB training checkpoint; git-ignored |
| HF **model** repo [`SpicyGuac/sol-001`](https://huggingface.co/SpicyGuac/sol-001) | the same bundle, 103 MiB | Weights don't belong in an app repo. Hosting them on the Hub means the app repo stays small and the weights are independently versioned and citable |
| **Streamlit Community Cloud**, from GitHub `ArepaMan/sol` | `app/streamlit_app.py` + the `src/` package | Fetches the weights at startup with `hf_hub_download`, which caches to disk |

> **The two namespaces don't match**: GitHub is `ArepaMan`, Hugging Face is
> `SpicyGuac`. Every `huggingface.co` path below uses `SpicyGuac`; every
> `github.com` link uses `ArepaMan`. Worth stating because a copy-paste between
> them fails with a 404 that reads like a permissions problem.

## Why not a Hugging Face Space

The original M8 plan was a Gradio app on a free HF Space. That plan died at the
`repo create` call:

```
402 Client Error: Payment Required
Static Spaces are free for everyone, but hosting Gradio and Docker Spaces
on free cpu-basic requires a PRO subscription.
```

HF moved Gradio Spaces behind PRO after this roadmap was written. The Gradio app
(`app/demo.py`) was already built, tested, and working — it simply had nowhere
free to live. It is **kept, not deleted**: it still runs locally, `app/README.md`
still carries a valid Space YAML header, and `app/requirements-gradio.txt` still
pins its deps, so anyone with PRO can deploy it unchanged.

Streamlit Community Cloud gives a real Python backend for free, so the UI layer
moved and nothing else did. Both apps are thin wrappers over `SolGenerator`
(`src/infer.py`) — that is the payoff for having put generation in one module
instead of in the app. Sampling behaviour is identical because it is the same
code.

**What Community Cloud costs instead of money:**

| | Free HF Space (as planned) | Streamlit Community Cloud (actual) |
|---|---|---|
| Sleeps after | 48 h | **12 h** |
| RAM | 16 GB | **1 GB** ← the binding constraint |
| Deploy source | push to a Space git repo | connect a **public GitHub repo** |
| Cost | now PRO-only | free |

The 1 GB ceiling is the real engineering constraint, and it's why
`app/streamlit_app.py` uses `@st.cache_resource` — see "Memory" below.

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

Do this before uploading anything. It exercises the exact code path the deployed
app runs, minus the network:

```bash
python -m src.infer --model-dir export/sol-001 --device cpu --seed 42
```

```bash
SOL_MODEL_DIR=export/sol-001 streamlit run app/streamlit_app.py
```

Then open <http://localhost:8501> and generate once from each tab.

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

Then verify the download path with a **clean cache**, which is the one thing
that cannot be tested any other way — a wrong repo id or a missing file fails
here with a readable traceback instead of inside a deploy log:

```bash
HF_HOME=/tmp/hf_clean SOL_MODEL_DIR= python -c "import sys; sys.path.insert(0,'app'); import streamlit_app"
```

Measured on a cold cache: **8.7 s** to model-ready (~7.5 s downloading 103 MiB,
1.2 s loading), against 1.2 s when the bundle is already local.

## 4. Deploy to Streamlit Community Cloud

Community Cloud deploys from a **public GitHub repo**, not from a git push to
the host — so step one is making sure `master` is current on GitHub. There is no
separate deploy repo to sync, which is one real advantage over the Space route.

1. Push to GitHub. The app reads `app/streamlit_app.py` from whatever `master`
   holds.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **New app** → **Deploy a public app from GitHub**, then:

   | Field | Value |
   |---|---|
   | Repository | `ArepaMan/sol` |
   | Branch | `master` |
   | Main file path | `app/streamlit_app.py` |

4. **Open "Advanced settings" and set Python to 3.12.** Not optional — see below.
5. Pick the subdomain — minimum 6 characters, so `sol` is rejected; `sol-52m`
   is what this deploys as.

### Two build failures, both hit on the first real attempt

**Community Cloud defaults to Python 3.14.** torch 2.6.0 publishes wheels for
3.9–3.13 only, so the resolve dies with `no wheels with a matching Python ABI
tag`. The error text points at the package index rather than at the Python
version, which sends you looking in the wrong place. There is no `runtime.txt`
equivalent here: the Python version is chosen in **Advanced settings at deploy
time**, and changing it afterwards means deleting and recreating the app.

**A bare `torch==2.6.0` installs the CUDA build even with the CPU index listed.**
`--extra-index-url` is additive, not preferential — pip finds 2.6.0 on PyPI
first and stops looking. That is an ~800 MB download and ~2.5 GB installed, on
an app with a 1 GB memory budget. The fix is pinning **`torch==2.6.0+cpu`**: the
`+cpu` local-version tag exists only on the PyTorch index, so requesting it is
what actually forces the small wheel. Easy to miss locally, because the machine
you develop on is usually one that *wants* the CUDA wheel.

**Why `app/requirements.txt` is the file that gets installed:** Community Cloud
looks for a dependency file in the repo root *or* in the entrypoint's directory.
The root `requirements.txt` here is the **training** environment — `datasets`,
`wandb`, `matplotlib`, `jupyter`, CUDA torch — and installing it would blow both
the build time and the memory budget. Keeping the app's deps next to the app is
what stops that, and it is not optional.

Only five `src/` modules are actually imported: `config`, `model`, `infer`,
`utils`, `__init__`. The rest of `src/` is never touched at runtime, so none of
the training dependencies are needed.

## Memory: the binding constraint

Community Cloud's free tier caps an app at **1 GB RAM**. Measured on this
machine, CPU-only, loading from the exported bundle:

| Stage | RSS |
|---|---|
| bare Python | 20.0 MB |
| + `import torch` | 386.2 MB |
| + `import streamlit` | 415.1 MB |
| + model loaded (fp32) | **785.9 MB** |
| after a 200-token generation | 778.2 MB |

**786 MB is with the CUDA-enabled torch wheel**, which this machine has. The
deployed app installs the CPU-only wheel (`--extra-index-url .../whl/cpu`),
which drops most of that 386 MB torch import — the honest estimate on Community
Cloud is **~600 MB, i.e. roughly 60% of the ceiling**. Comfortable enough to
deploy, tight enough that it is worth writing down.

Two consequences already in the code:

1. **`@st.cache_resource` on the loader is a memory requirement, not an
   optimisation.** Streamlit reruns the entire script on every widget
   interaction. Without the cache, each click would allocate another 212 MB
   fp32 model and the app would OOM within a few clicks.
2. **fp32, not bf16, on CPU** (`resolve_dtype` in `src/infer.py`). bf16 would
   halve the model to 106 MB, but CPU bf16 runs through reference kernels and
   is slower than fp32. If the app ever does hit the ceiling, switching to bf16
   is the first lever — it buys ~106 MB at a throughput cost.

## Cold start: what was measured, and what the mitigations are

Community Cloud apps sleep after **12 hours** without traffic (not 48 h — that
was the HF Space number the roadmap assumed). Waking one pays for container
start + Python imports + weight download (cached to disk unless the container
was rebuilt) + model load.

**Measured locally**, which bounds the parts that aren't hosting-dependent:

| Stage | Time |
|---|---|
| `import torch` + `import streamlit` + `src` | ~16 s |
| `SolGenerator.from_pretrained`, bundle already local | 1.2 s |
| `SolGenerator.from_pretrained`, **cold HF cache** (downloads 103 MiB) | 8.7 s |
| **Total to "model ready", cold cache** | **~25 s** |

Community Cloud's CPU is slower than this machine, so **30–60 s is the honest
expected range** for a wake-from-sleep. Replace this paragraph with the measured
number once the app has actually slept and been woken — do not quote the local
figure as if it were the deployed one.

Mitigations, all already in the code:

1. **CPU-only torch wheel** (`--extra-index-url .../whl/cpu` in
   `app/requirements.txt`). The default PyPI wheel bundles ~2.5 GB of CUDA
   libraries a CPU host can never use.
2. **`@st.cache_resource` on the loader**, so the model loads once per container
   rather than once per interaction. This is doing double duty as the memory
   mitigation above.
3. **An in-UI spinner that names the number** — "first visit after the app
   sleeps takes ~30-60s". The honest fix for a limitation you can't remove is to
   say it out loud rather than let it read as a hang.

`pinned: true` in `app/README.md` no longer does anything for the deployed app;
it applies to HF Spaces and is kept only for the Gradio/PRO path.

## Inference latency

Measured with `python -m src.infer --model-dir export/sol-001 --device cpu
--seed 42 --max-new-tokens 200`:

| Device | Rate |
|---|---|
| RTX 4070 Laptop (bf16) | ~90 tok/s |
| This machine's CPU (fp32) | **21.6–24.6 tok/s** |

Community Cloud's shared CPU will land at or below the local CPU number — expect
roughly 15–25 tok/s, i.e. a 200-token story in 8–13 s once the model is warm.
That is why the app streams token-by-token rather than returning one block.

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
