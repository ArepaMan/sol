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

- **CPU inference is ~5.6× slower than the GPU, and that is the demo's real
  speed.** Measured on the deployed app (<https://sol-52m.streamlit.app>) over
  7 generations: **mean 16.0 tok/s, range 12.0–20.0**. Locally: 21.6–24.6 tok/s
  on this machine's CPU (fp32) and ~90 tok/s on the RTX 4070 (bf16). A
  200-token story is a 10–17 second wait, which is why the demo streams rather
  than returning one block.
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

## What's next

M9 closes the loop: spec de-drift (`scripts/export_spec.py`) and portfolio
wiring. The three standing technical gaps, unchanged and unfixed, are
coherence/entity drift, the inherited mojibake in 6.20% of the training
corpus, and the un-ablated architecture choices (learned PE, LayerNorm+GELU)
at the top of this document.
