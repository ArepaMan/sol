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

## What's next

Ablations (M7) will add seed-variance as the honest yardstick for reading
any of the above as signal vs noise. Deployment (M8) inherits the EOT-stop
fix and should carry an explicit "About / Limitations" tab reflecting this
document, not a cleaned-up subset of it.
