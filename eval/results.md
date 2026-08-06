# M6 — Evaluation results

Val set: 15,141 documents, 2,955,739 tokens (28 truncated to block_size). Baselines fit on the first 10,000,000 train tokens (see `src/baselines.py` for why that's a prefix, not the full corpus). 95% CI from a 10,000-replicate document-level bootstrap.

| Model | Perplexity | 95% CI |
|---|---|---|
| uniform | 32000.000 | [32000.000, 32000.000] |
| unigram | 379.010 | [377.499, 380.497] |
| trigram | 23.425 | [23.246, 23.600] |
| sol-001 | 3.719 | [3.693, 3.745] |

## Per-length-bucket perplexity (Sol-001)

| Bucket (tokens) | n docs | Perplexity |
|---|---|---|
| 1-64 | 5 | 3.823 |
| 65-128 | 918 | 3.817 |
| 129-256 | 12571 | 3.916 |
| 257-512 | 1647 | 3.008 |
| 513+ | 0 | — |

## Qualitative rubric (60 prompts, single-rater — see `docs/RUBRIC.md`)

Mean ± population sd per dimension, 1–5 scale. Full scores in `eval/rubric_scores.csv`.

| | n | Grammar | Coherence | On-topic | Repetition (qualitative) |
|---|---|---|---|---|---|
| **overall** | 60 | 4.00 ± 0.68 | 3.15 ± 0.79 | 4.12 ± 1.56 | 3.98 ± 0.65 |
| story-start | 15 | 4.40 ± 0.61 | 3.40 ± 0.71 | 5.00 ± 0.00 | 4.13 ± 0.62 |
| dialogue | 15 | 4.13 ± 0.62 | 3.33 ± 0.70 | 5.00 ± 0.00 | 3.80 ± 0.83 |
| continuation | 15 | 3.93 ± 0.68 | 3.27 ± 0.93 | 5.00 ± 0.00 | 4.20 ± 0.40 |
| out-of-domain | 15 | 3.53 ± 0.50 | 2.60 ± 0.49 | 1.47 ± 0.62 | 3.80 ± 0.54 |

**Reading the on-topic column:** for the three in-domain categories it's a near-ceiling
5.00 (the model never leaves story register — it has never seen anything else). For
out-of-domain probes it's the opposite and expected result, **1.47** — per
`docs/RUBRIC.md`, this dimension there scores register-imitation, not quality, and the
honest finding is that Sol-001 does not imitate finance/legal/technical register at all;
it either grabs one topical word (`ood-08`: "train" → "train station" pun) or reverts to
a fresh children's story within the first clause. This is the expected, correct failure
mode for a model trained on zero non-narrative text, not a defect.

**Coherence (3.15/5 overall) is the real limiting factor**, not grammar (4.00/5) or the
automatic repetition metrics (distinct-2 0.933, distinct-3 0.984 overall — see
`eval/repetition_summary.md`; the two disagree with the qualitative repetition score only
at the extremes, e.g. `ood-05`'s literal duplicated ingredient clause). The dominant
coherence failure across all categories is **local, not global**: sentence-to-sentence
grammar holds up, but named characters swap (`continuation-03`, `continuation-13`),
objects/animals appear with no introduction (`story-start-05`, `continuation-04`), or a
character's stated action contradicts an earlier line (`dialogue-11`'s ball
"disappeared" then "appeared in the bush"). A 52M-parameter model doesn't carry enough
context state to track entity identity across an 8-layer, 150-token generation reliably —
consistent with `docs/LIMITATIONS.md`'s existing "weak reasoning" caveat, now with a
concrete failure pattern attached to it rather than a generic hedge.

**A data bug, not a decode bug — traced upstream, not assumed:** `continuation-08` and
`ood-06` contain mojibake curly quotes (`â€œ`, `â€TM`) instead of `"`/`'`. The first
hypothesis was a decode-side encoding bug in `src/generate_samples.py` or
`Tokenizer.decode()` — but `tokenizer.encode(x)` → `tokenizer.decode(...)` round-trips a
real curly-quote string exactly, and grepping the source corpus settles it directly:
**108,464 of 1,748,358 train documents (6.20%) already contain this exact mojibake
pattern in `data/processed/train.jsonl`**, i.e. before tokenization ever touches the
data. TinyStories itself carries a CP1252-as-UTF-8 mis-decode from its own upstream
pipeline (a classic "smart quotes" double-encoding), and Sol-001 faithfully learned to
reproduce it because it's genuinely present ~6% of the time in training. This is an
inherited data-quality issue, not a bug anywhere in this repo — flagged in
`docs/LIMITATIONS.md` and `docs/DATA_CARD.md` rather than silently absorbed, the same
treatment given to the cross-split-leakage finding in M2. A cleaning pass
(`data/clean.py`) to normalize or strip this pattern is a concrete, well-scoped "what
I'd do next" item, not attempted here since M6 is eval, not a second data pipeline pass.

**Also visible, not scored by the rubric:** most generations run past the model's own
natural end-of-story boundary and into a second, unrelated story with no separating
punctuation (e.g. `story-start-06`, `continuation-01`) — because
`src/generate_samples.py` always generates the full `--max-new-tokens` budget rather
than stopping at `<|endoftext|>`. The model itself is emitting EOT correctly (that's
*why* a fresh "Once upon a time" starts right after); this is a generation-script
behavior to fix in M8's `src/infer.py` (stop at EOT), not a training or model defect.
