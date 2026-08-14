# Deploying Sol

**Live: <https://sol-52m.streamlit.app>** · Weights:
<https://huggingface.co/SpicyGuac/sol-001>

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

| Stage | Local | **Deployed (Community Cloud)** |
|---|---|---|
| `SolGenerator.from_pretrained`, bundle already local | 1.2 s | — |
| `SolGenerator.from_pretrained`, cold HF cache (103 MiB download) | 8.7 s | — |
| **Model load, first visit after a fresh deploy** | ~25 s (incl. imports) | **6.7 s** |

The deployed number came in **well under** the 30–60 s predicted. The container's
network is much faster than a home connection for the 103 MiB pull, which more
than offsets its slower CPU — the estimate was wrong in the useful direction,
and it was wrong because it extrapolated from local bandwidth.

**Still unmeasured: a true wake-from-sleep.** 6.7 s is a *fresh deploy* with a
warm container. Community Cloud sleeps an app after 12 h idle, and waking one
additionally pays for container start. That number requires waiting 12 h without
traffic; it is not something a deploy can measure on the day. The in-UI copy
says "~30-60s" — deliberately conservative until there is a real figure to
replace it with.

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

Measured with `python -m src.infer --model-dir export/sol-001 --device {cpu,cuda}
--prompt "Once upon a time" --seed 42 --max-new-tokens 200`, adding
`--no-kv-cache` for the "before" column. n=7 per cell, the two arms
**interleaved** so any thermal or clock drift lands on both.

| Device | Before (no KV cache) | After (KV cache) | |
|---|---|---|---|
| This machine's CPU (fp32) | 23.4 tok/s (22.7–24.6) | **82.8 tok/s (74.9–88.0)** | **3.5×** |
| RTX 4070 Laptop (bf16) | 124.4 tok/s (123.2–125.7) | **132.4 tok/s (130.5–133.2)** | **1.06×** |
| **Community Cloud, deployed (fp32)** | mean 16.0, range 12.0–20.0 (n=7) | **mean 29.0, range 26.2–30.0** (n=5) | **1.8×** |

### The deployed gain is 1.8×, not the local 3.5×

Measured through the app's own per-generation caption, seeds 42–46 at the default
sampling settings, after discarding one warm-up generation:

| Seed | Tokens | Seconds | tok/s |
|---|---|---|---|
| 42 | 199 | 7.6 | 26.2 |
| 43 | 200 | 6.8 | 29.4 |
| 44 | 156 | 5.3 | 29.6 |
| 45 | 152 | 5.1 | 30.0 |
| 46 | 200 | 6.7 | 29.8 |

**This comparison is weaker than the local one, and the reason matters.** Locally
the two arms were interleaved in a single session, so drift hit both equally. The
deployed app exposes no `--no-kv-cache` switch, so its "before" column is the
*historical* 16.0 from M8/M9 QA — a different session, on a shared host, with
different neighbours. The 1.8× therefore carries the noise of two measurement
sessions rather than one controlled experiment. It is the honest number
available, not a clean speedup, and it should not be quoted with the same
confidence as the local 3.5×.

Two things survive that caveat. The gain is large enough that the ranges do not
overlap at all — the new low (26.2) sits above the old high (20.0). And the new
spread is **much tighter**: 3.8 tok/s wide against the old 8.0. Four of the five
samples land in 29.4–30.0; only seed 42, the first recorded generation, sits low
at 26.2, which looks like container warm-up surviving the discarded run. A faster
generation simply gives a noisy shared vCPU less wall-clock in which to interfere.

Why 1.8× and not 3.5×: the deployed host is a shared vCPU, not this machine's
CPU. Fewer cores and less memory bandwidth for the same arithmetic means less to
win by removing redundant work. The deployed figure is what users actually get,
so it is the one the app's caption quotes.

**The GPU barely moved, and that is the informative half.** At batch 1 with a
52M-parameter model, generation on a 4070 is bound by kernel-launch and
Python-loop overhead, not by arithmetic — 132 tok/s is ~7.5 ms per token across
8 layers, which is nowhere near what that GPU can compute in 7.5 ms. Deleting
redundant FLOPs from something that was never FLOP-bound buys 6%. The CPU is
genuinely compute-bound, so it gets 3.5× locally and 1.8× deployed — and CPU is
what the deployed app runs on, so that is the number that reaches users.

**A first pass measured the GPU at 1.12×.** Alternating the two arms instead of
running all of one then all of the other dropped it to 1.06× and tightened both
ranges by an order of magnitude. The extra 6% was cold-clock drift on the arm
that ran first. Written down because the discarded number was the flattering one.

**The old "~90 tok/s" GPU figure was under-measured.** Re-running the *unchanged*
code path today (`--no-kv-cache`, which is byte-for-byte the pre-cache loop)
gives 124.4 tok/s, n=7, range 123.2–125.7. The ~90 appears to have come from a
single sample on the `--stream` path, which pays for per-token console flushing:
measured now, streaming on GPU gives mean 102.2 with a range of **77.4–119.2**.
That is the same error as the 15–25 band below — one sample quoted as if it
described a distribution — and it is corrected here rather than left standing.

### Past `block_size`, the cache buys nothing

Sol uses **learned absolute** positional embeddings. Once generation passes the
512-token window, the window slides, every surviving token is re-embedded one
position lower, and every cached key/value is stale at once — so the cache is
dropped and re-prefilled each step. Measured on a 661-token prompt (truncated to
511, so every step runs in that regime): **6.2 tok/s cached vs 5.9 uncached** —
i.e. nothing, within noise. RoPE would not rescue this either; the shift is in
the embedding, not just in the attention score. Sol's default 200 new tokens
from a short prompt never reaches this path, which is why the cache still pays.

### Memory: the cache is free in practice

19 MB was the estimate; 18.5 MB is the exact figure (2 × 8 layers × 512 ctx ×
592 channels × 4 bytes). Peak working set, measured in separate processes:

| | Before | After |
|---|---|---|
| 200 tokens from a short prompt | 775.6 MB | 775.3 MB |
| 200 tokens at full 512-token context | 796.4 MB | 797.6 MB |

Unchanged. The cache's 18.5 MB is absorbed because the uncached path was
allocating comparable transient activations for the whole prefix on every step
anyway — it just threw them away afterwards. Nothing here moves against the
1 GB ceiling.

### The deployed baseline this replaces

**The originally predicted 15–25 tok/s band was wrong, and QA caught it.** The
single measurement taken at deploy time was 16.0 tok/s, which sat inside that
band and looked like confirmation. Six further generations during manual QA came
in at 12.0, 14.2, 17.8, 20.0, 15.1 and 16.9 — two below the predicted floor,
none within 5 tok/s of the predicted ceiling. The mean across all seven is
exactly 16.0, so the point estimate was fine and the *band* was fiction: a
single sample cannot tell you a range, and quoting one as if it could is the
error worth naming.

A shared vCPU also has no throughput guarantee — that spread is partly other
tenants, not just variance in story length.

`src/infer.py` uses fp32 on CPU deliberately (`resolve_dtype`): CPU bf16 runs
through reference kernels on most x86 parts and is slower than fp32, and since
the weights were bf16 already, fp32 inference is a strict widening — no quality
change.

### Reproducing the before/after

The pre-cache path is kept reachable rather than deleted, so the comparison can
be re-run at any time instead of taken on trust:

```bash
python -m src.infer --model-dir export/sol-001 --device cpu --seed 42 --no-kv-cache
```

Its output is byte-identical to the cached run at fp32 — that equivalence is the
safety argument for the whole change, and it is enforced by `tests/test_infer.py`
rather than checked once by hand.

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

**It is also per-precision, which the KV cache made visible.** Cached and
uncached generation are byte-identical in fp32 — on CPU and on CUDA, verified
over 1,500 sampled tokens. In bf16 they are not: a cached decode step reduces
over the key dimension in a different order than a full-width forward, and
bf16's 8 mantissa bits accumulated over 8 layers put that difference at ~1% of
logit scale, which is eventually enough to move a multinomial draw across a CDF
boundary. Both stories are valid samples from the same distribution; neither is
more correct. The deployed app is CPU fp32, so this does not touch it, but it
is the reason `--seed 42` on the 4070 no longer reproduces a story generated
before the cache landed.
