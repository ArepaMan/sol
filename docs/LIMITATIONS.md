# Sol — Known limitations

Tracked as results land, not written once at the end. Each entry states what
was measured, not just asserted, and points at the artifact that backs it.

## Architecture (known trade-offs, not yet measured against alternatives)

- **Learned positional embeddings, not RoPE.** Simpler to implement and
  sufficient at 512-token context; would need to change if context length
  grew substantially.
- **512-token context.** Justified by measurement, not guesswork — 99.74% of
  train documents fit whole within 512 tokens (`docs/DATA_CARD.md`). Longer
  documents are truncated (M6 measured 28 of 15,141 val docs, 0.18%).
- **LayerNorm + GELU, not RMSNorm + SwiGLU.** The GPT-2-era standard, not the
  current one; a real ablation candidate for M7 if time allows, not attempted
  here.

## Evaluation (M6 findings — `eval/`, `docs/RUBRIC.md`)

- **Coherence is the binding constraint, not grammar.** Rubric mean: grammar
  4.00/5, coherence 3.15/5 (n=60, `eval/rubric_scores.csv`). The dominant
  failure mode is local, not global: named characters swap mid-generation
  (`continuation-03`, `continuation-13`), objects appear with no
  introduction (`story-start-05`), or a stated action contradicts an earlier
  line (`dialogue-11`). A 52M-parameter, 8-layer model does not reliably
  carry entity identity across a 150-token generation. See `eval/results.md`
  for the full breakdown.
- **Out-of-domain probes correctly fail, and fail informatively.** On-topic
  rubric score drops to 1.47/5 for finance/legal/technical prompts (vs a
  ceiling 5.00/5 for in-domain prompts) — the model was trained on zero
  non-narrative text, so it either grabs one topical word as a pun
  (`ood-08`: "train" → train station) or reverts to a fresh children's story
  within the first clause. This is the expected behavior for an
  un-instruction-tuned narrow-domain model, not a bug — but it is a hard
  limit on what this model can be used for.
- **6.20% of the TinyStories training corpus contains a mojibake artifact**
  (curly quotes double-encoded as CP1252-through-UTF-8, e.g. `â€œ`, `â€TM`)
  inherited from the upstream dataset's own pipeline — confirmed by grepping
  `data/processed/train.jsonl` directly, not assumed. Sol-001 faithfully
  reproduces this pattern in ~a few percent of generations
  (`continuation-08`, `ood-06` in `eval/generations.jsonl`). Not a bug in
  this repo's tokenizer or decode path (round-trips exactly in
  `tests/test_tokenizer.py`); a data-cleaning pass to normalize or strip it
  is a concrete next step, not attempted in M6 since that milestone is eval,
  not a second pass over `data/clean.py`.
- **Generation doesn't stop at the model's own end-of-story token.**
  **Fixed in M8** — `src/infer.py`'s sampling loop breaks on `eot_id` (read
  from `data/tokenized/meta.json`, not hardcoded), and both the CLI and the
  Gradio demo route through it. Measured on the same prompt: the model stops
  itself at 191 of a 300-token budget, on a complete closing sentence. The
  fix landed in `src/infer.py` rather than `src/model.py` so the M3
  architecture file stays an architecture file, and `src/generate_samples.py`
  was deliberately left alone so M6's published numbers still describe the
  code that produced them. Original M6 finding below, unedited.
  `src/generate_samples.py` always spends its full `--max-new-tokens`
  budget, so most 150-token samples run past a natural ending into a second,
  unrelated story with no separating punctuation. The model itself emits
  `<|endoftext|>` correctly (that's *why* a fresh "Once upon a time" starts
  right after) — this is a generation-script gap, to fix by stopping on EOT
  in M8's `src/infer.py`, not a training or model defect.
- **Rubric scoring is single-rater.** `docs/RUBRIC.md` states this plainly:
  one contributor scored all 60 generations, not an independently replicated
  or crowd-sourced study. The automatic repetition metrics (distinct-2/3,
  max repeated substring — `eval/repetition_summary.md`) exist specifically
  as an objective backstop to this, per the M6 risk note in
  `docs/ROADMAP.md` ("rubric theatre").
- **Baseline n-gram models are fit on a 10M-token prefix of train**, not the
  full 357.85M-token corpus (`src/baselines.py`) — a deliberate scope
  decision (a full-corpus trigram table needs an on-disk sparse structure
  out of scope for what's meant to be a weak reference floor), documented
  in-code rather than silently capped.

## Training / checkpointing (M5 finding — `experiments/001_baseline/run.md`)

- **`best.pt`'s training-time selection was noisy and, on re-evaluation,
  wrong.** The periodic 100-batch eval used during training flagged iter
  34,000 as best (val 1.3747); M6's full-val-set, non-overlapping,
  per-document evaluation shows the true final checkpoint (iter 40,000) is
  actually better (val loss → **perplexity 3.719, 95% CI [3.693, 3.745]**,
  `eval/results.md`) — resolving the ambiguity `run.md` flagged at the time.
  `latest.pt` is confirmed the correct checkpoint to use going forward.

## Deployment (M8 findings — `docs/DEPLOY.md`, `app/`)

- **CPU inference is slower than the GPU, and that is the demo's real speed.**
  Measured on the deployed app (<https://sol-52m.streamlit.app>) over 7
  generations: **mean 16.0 tok/s, range 12.0–20.0** — a 10–17 second wait for a
  200-token story, which is why the demo streams rather than returning one
  block. **Superseded post-M9 by the KV cache: mean 29.0, range 26.2–30.0
  (n=5), so ~7 s.** The original finding stays because the streaming decision
  was made against it and still holds — 7 s of nothing would still read as
  broken. See the inference-optimisation section below for why the deployed
  gain (1.8×) is only half the local one (3.5×).
- **A predicted throughput band was published on the strength of one sample.**
  The deploy-time measurement was 16.0 tok/s against a predicted 15–25, which
  looked like confirmation. Manual QA's six further runs landed 12.0–20.0 — two
  below the predicted floor, none near the ceiling. The mean was still exactly
  16.0, so the point estimate held and the *range* was invented. Fixed in the
  docs and in the app's own caption. Noted here because it is the same class of
  error the project criticises elsewhere: quoting a spread you did not measure.
- **Sampling is reproducible per device, not across devices.** `--seed 42`
  twice on the same device is byte-identical (an M8 exit criterion; timing
  output goes to stderr so the diff is clean). CUDA seed 42 and CPU seed 42
  produce *different* stories — different kernels, different floating-point
  rounding, a different sampled token, and everything after it diverges.
  Worth stating because "seeded" is often read as stronger than it is.
  **And per precision, too** — see the KV cache section below.
- **Cold start is a real cost, mitigated but not eliminated.** Free Spaces
  sleep after 48h idle. Four mitigations are in place (`pinned: true`, a
  CPU-only torch wheel, import-time model loading, and an in-UI warning) —
  see `docs/DEPLOY.md`. The honest fix for a limitation you can't remove is
  to tell the user about it in the UI, which the demo does.
- **The free deployment tier has a 1 GB RAM ceiling, and the app sits at
  roughly 60% of it.** Measured CPU-only: bare Python 20 MB → +torch 386 MB →
  +streamlit 415 MB → **+model 786 MB** (with the CUDA-enabled torch wheel;
  the CPU-only wheel the deployed app installs should land near 600 MB).
  Headroom exists but is not generous. If it is ever exceeded, the first lever
  is bf16 weights on CPU — ~106 MB saved at a throughput cost, since CPU bf16
  runs through reference kernels.
- **The original deployment target stopped being free mid-milestone.** HF moved
  Gradio Spaces behind PRO (`402 Payment Required`), so the demo runs on
  Streamlit Community Cloud instead. The Gradio app is kept and works, but is
  undeployed — meaning it is tested locally and by `tests/test_infer.py`'s
  coverage of the shared generator, not by a live service. Two UIs is a real
  maintenance cost, mitigated by keeping all generation in `src/infer.py` and
  all limitations copy in `app/about.py`, so neither can drift.
- **The Space runs a five-module subset of `src/`.** `demo.py` needs only
  `config`, `model`, `infer`, `utils`, `__init__`; shipping `train.py` or
  `eval.py` would drag in `datasets`, `wandb`, and `matplotlib` for no
  runtime benefit. The cost is that the Space is a *copy*, so it can drift
  from GitHub — `docs/DEPLOY.md` documents the sync step rather than
  pretending it's automatic.

## Inference optimisation (post-M9 — KV cache, `src/model.py`, `docs/DEPLOY.md`)

- **The KV cache does nothing past `block_size`, and cannot be made to.** Sol
  uses learned *absolute* positional embeddings. When generation passes 512
  tokens the window slides, every surviving token is re-embedded one position
  lower, and every cached key/value goes stale simultaneously — so the cache
  resets and re-prefills the whole window each step. Measured in exactly that
  regime (661-token prompt, truncated to 511, every step sliding): **6.2 tok/s
  cached vs 5.9 uncached**, which is nothing. RoPE, the usual answer, would not
  fix it either: the shift is in the embedding, not just in the attention
  score. A cache that survived the slide would need position-independent
  keys — a different architecture, not a different cache. Sol's default 200
  tokens from a short prompt never enters this regime, which is why the cache
  is still worth having; but the speedup is conditional and the condition is
  worth saying out loud.
- **The deployed before/after is not a controlled comparison, unlike the local
  one.** Locally the cached and uncached arms were interleaved in one session,
  so drift hit both. The deployed app exposes no `--no-kv-cache` switch, so its
  "before" is the *historical* 16.0 from M8/M9 QA — a different session, on a
  shared host, with different neighbours. The measured **1.8× (16.0 → 29.0)**
  therefore carries two sessions' worth of noise. What survives the caveat is
  that the ranges do not overlap at all (new low 26.2 > old high 20.0), and
  that the new spread is much tighter (3.8 tok/s wide vs 8.0). Recorded in
  `docs/spec.json` as `deployed_before_is_same_session_control: false`, so the
  caveat travels with the number instead of living only in this document.
- **The deployed gain is half the local one, and the local number is the one
  that flatters.** 1.8× deployed against 3.5× on this machine — a free shared
  vCPU has fewer cores and less memory bandwidth, so there is less to win by
  removing redundant arithmetic. Quoting the 3.5× as "the speedup" would be
  quoting the number measured on hardware no user touches.
- **The GPU speedup is 1.06×, not the ~4× the FLOP argument predicts.** At
  batch 1 with a 52M-parameter model, generation on the 4070 is bound by
  kernel-launch and Python-loop overhead, not arithmetic: 132 tok/s is ~7.5 ms
  per token across 8 layers, far more time than that GPU needs to do the maths.
  Removing redundant FLOPs from a workload that was never FLOP-bound buys ~6%.
  The CPU, which is genuinely compute-bound and is what the deployed app runs
  on, gets **3.5×** (23.4 → 82.8 tok/s, n=7 interleaved). The honest reading is
  that this optimisation was aimed at the right target by accident of where the
  app is deployed, not by the reasoning that motivated it.
- **Byte-identical cached-vs-uncached generation holds at fp32, not at bf16.**
  A cached decode step reduces over the key dimension in a different order than
  a full-width forward does. In fp32 that difference is ~2e-6 of logit scale
  and sampling never notices (verified byte-identical over 1,500 sampled tokens
  on CUDA and across three seeds on CPU). In bf16 — 8 mantissa bits accumulated
  over 8 layers — it reaches ~1e-2 of logit scale, enough to move a multinomial
  draw across a CDF boundary and diverge the story from there. Both outputs are
  valid samples from the same distribution and every argmax agrees on trained
  weights, so this is rounding rather than a logic error; it is pinned as a
  gpu-marked test parametrised over both dtypes, with the tolerances as the
  point. The practical cost: a bf16 CUDA story generated before the cache
  cannot be reproduced after it, at the same seed.
- **A first GPU measurement said 1.12× and was wrong.** Running all the
  uncached samples and then all the cached ones let the GPU's clocks drift
  between the two arms. Interleaving them dropped the figure to 1.06× and
  tightened both ranges by an order of magnitude. Recorded because the
  discarded number was the one that flattered the change — the same failure
  mode as the 15–25 tok/s band above, just with a subtler cause.
- **The previously published "~90 tok/s" GPU figure was under-measured.**
  Re-running the *unchanged* path today (`--no-kv-cache` is byte-for-byte the
  pre-cache loop) gives 124.4 tok/s, n=7, range 123.2–125.7. The old number
  looks like a single sample taken on the `--stream` path, which pays for
  per-token console flushing — measured now, that path gives mean 102.2 with a
  range of 77.4–119.2. Corrected in `docs/DEPLOY.md` rather than left standing.
- **`src/generate_samples.py` is deliberately pinned to the uncached path.**
  It produced M6's published artifacts and `docs/RUBRIC.md` cites generations
  by id, so it has to keep reproducing them from committed code. Measured 10/10
  byte-identical with the cache at its CUDA fp32 — but "identical in practice"
  is the wrong guarantee for a script whose output other documents index into,
  and the cache buys only ~6% on GPU. Same call M8 made when it left this file
  out of the EOT-stop fix.

## What's next

M9 closed the loop: spec de-drift (`scripts/export_spec.py`) and portfolio
wiring. The three standing technical gaps, unchanged and unfixed, are
coherence/entity drift, the inherited mojibake in 6.20% of the training
corpus, and the un-ablated architecture choices (learned PE, LayerNorm+GELU)
at the top of this document — the first of which the KV cache work has now
given a second, independent reason to revisit, since learned absolute
positions are also what caps the cache at `block_size`.
