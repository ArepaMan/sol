# Sol — Project Specification

Design decisions and context for **Sol**, captured from project planning (July 2026).

## Objective

Build a from-scratch small language model as a **portfolio piece** for Applied ML / AI Engineer / Data Scientist interviews. Success = reproducible training, measured evaluation, deployed demo, and clear documentation of trade-offs.

## Why from scratch (not fine-tune)

Shows understanding of:

- Tokenization and data engineering
- Transformer architecture and training dynamics
- Evaluation beyond "it generates text"
- Hardware-aware model sizing

Fine-tuning is faster but weaker signal for fundamentals. Sol prioritizes depth.

## Dataset

**Primary:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

**Corpus cap:** originally targeted at 400M tokens; **measured at 357,852,786** after
cleaning, dedup, and encoding with the in-domain 32k BPE tokenizer (~4.13 chars/token —
see `data/processed/stats.json`, `data/tokenized/meta.json`). Lowered honestly to the
measured value rather than padded. Note this is the *corpus size*, not the number of
tokens processed — at 40,000 iters × 32,768 tokens/step the model sees **1.31B tokens ≈
3.66 epochs** over that corpus. Wall-clock is measured in M4, not estimated here.

**Why not full corpus:** Diminishing returns for portfolio; enables data-scale ablation (100M vs 400M).

### Data pipeline requirements

1. Download and subset
2. Clean: dedupe, length filter, basic quality heuristics
3. Train BPE tokenizer (32k vocab)
4. Encode to binary shards for training
5. EDA notebook: token counts, length distribution, data card in README

**M1 status:** done. Exact-dedup rate: 14.58% train (within-split — a real property of
this synthetic, templated corpus, not a pipeline bug). Validation's own internal dup
rate is 0% — but **28.67% of raw validation documents turned out to be exact
duplicates of a train document** (cross-split leakage). That distinction was wrong in
an earlier version of `data/prepare.py`: train and validation were deduped against a
single shared hash set, which silently folded every cross-split duplicate into
validation's "exact_dup" counter before a separate leakage-check pass ever ran — so
that pass structurally reported 0 leakage regardless of the true rate. Fixed by
deduping each split against its own hash set, then running a real cross-split check
afterward; `tests/test_prepare.py` now pins both the fixed behavior and the original
bug pattern as regression tests. The end *result* (no document text shared between
`train.jsonl` and `val.jsonl`) was already correct and independently
hash-verified throughout (`tests/test_pipeline_artifacts.py`) — only the accounting
of *why* documents were dropped was wrong. See `docs/DATA_CARD.md` for the full
writeup and the practical implication for val-loss interpretation.

Near-dedup (`data/near_dedup.py`, MinHash + LSH banding) is implemented and
unit-tested but was not run on the full corpus — the 14.58% train rate sits just
under the 15% threshold that would have triggered it, and MinHash on 1.75M documents
would materially extend the runtime for a marginal expected gain. Revisit if M5's val
loss looks worse than the 2.8–3.2 target and templated repeats look like a plausible
cause.

## Model architecture

Decoder-only GPT (causal LM), **52,901,712 parameters, measured** (M3;
`tests/test_model.py::test_param_count_near_52m`):

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Layers | 8 | Comfortably fits 8GB even without checkpointing — see M4 below |
| Heads | 8 | head_dim = 74 |
| Embedding | 592 | Not the originally-guessed 512 — see below |
| Context | 512 | Validated in M2: covers 99.74% of train docs whole (measured p99=479) |
| Vocab | 32k BPE | Standard for small LMs |
| Norm | LayerNorm, pre-LN | Hand-written; simpler to reason about than RMSNorm |
| Activation | GELU | 4× MLP, GPT-2 style |
| Positional | Learned | RoPE is listed as a "what I'd do next" in `docs/LIMITATIONS.md` |

**Why `n_embd=592`, not 512:** the original 512 was a guess that predates ever
building the model. With weight tying (below) and the rest of this table fixed,
512 measures at 41.8M params — a ~20% gap from the project's own "~52M" branding
(README, portfolio tagline). The fix was picking the cheapest lever: keep 8
layers, 512 context, 32k vocab, and weight tying exactly as already documented,
and widen the embedding dimension until the count lands close. 592 (=74×8, so
`head_dim` stays an integer) measures at 52.9M — 1.73% off — versus untying
weights instead, which would hit 58.2M (~12% off) while reversing an already
deliberate design decision for no good reason. See `configs/micro_50m_8gb.yaml`'s
header comment for the same reasoning in the config itself.

**Implementation:** `src/model.py` is written by hand (~300 lines) using nanoGPT as a
reference, not forked from it. `F.scaled_dot_product_attention(..., is_causal=True)`
provides the Flash kernel without hand-rolling the matmul. The point of the project is
to demonstrate the fundamentals, so the architecture code is the deliverable.

## Training hyperparameters (RTX 4070 Laptop, 8 GB)

```yaml
batch_size: 4
gradient_accumulation_steps: 16
learning_rate: 3.0e-4
min_lr: 3.0e-5
warmup_iters: 1000
max_iters: 40000
weight_decay: 0.1
grad_clip: 1.0
precision: bfloat16
gradient_checkpointing: false  # measured in M4 — see below, not the originally-assumed true
compile: false  # Triton not installed on Windows — measured, not assumed
```

**Why BF16 rather than FP16:** the RTX 4070 Laptop is Ada Lovelace (sm_89), which
supports BF16 natively. BF16 keeps FP32's exponent range, so it needs **no GradScaler**
and none of the loss-scale tuning FP16 demands — one fewer source of silent divergence
in a 20-hour run. (An earlier draft of this spec targeted a Turing-generation RTX 2070
Super, where FP16 would have been forced.)

**Why `gradient_checkpointing: false`, not the originally-"required" true:** M4's
benchmark (`src/benchmark.py`, `experiments/000_benchmark/results.md`) measured
`batch_size=4` at **2137 MiB without checkpointing** (~21,500 tok/s) vs **1460 MiB with
it** (~17,000 tok/s — ~26% slower). Both are far under the 8 GB budget; the "required"
assumption predated knowing this model's true, much smaller than originally guessed,
memory footprint (see the `n_embd` correction above). Checkpointing only becomes
genuinely necessary around `batch_size` 24–32.

**A real footgun found while measuring this, not guessed:** exceeding physical VRAM on
this Windows/CUDA setup does **not** cleanly OOM. At `batch_size=32` without
checkpointing, `torch.cuda.max_memory_allocated()` reported ~11,965 MiB — more than the
card's 8,188 MiB physical total — while `ok=True` and throughput silently collapsed to
~15% of normal (4,468 vs ~21,500 tok/s). CUDA on Windows falls back to slow shared/system
memory rather than raising. `src/benchmark.py`'s `suspected_shared_memory_spill` check
exists specifically to catch this instead of reading a large-but-"fine" VRAM number and
missing what actually happened. A clean OOM only appears at `batch_size=64`.

**OOM fallback order** (if `batch_size` is ever increased and this stops fitting):

1. `gradient_checkpointing: true` first — free at up to ~26% throughput cost, no
   architecture change, and covers up to `batch_size` ~16–24 with headroom to spare
2. batch 2, grad accum 32 (if still short)
3. n_layer 6 (~39M params) or block_size 384 (last resort — changes the architecture)

Target: peak VRAM < 7400 MiB, leaving ~800 MiB headroom for Windows WDDM. **Never trust
a peak-VRAM number alone above ~8000 MiB on this setup without checking for the
shared-memory-spill signature above.**

## Results (M5 baseline + M6 full evaluation, measured — the original "Expected results" table below is kept for record)

| Metric | Original target | **Measured** |
|--------|--------|--------|
| Val loss (M5, periodic 200-batch eval) | ~2.8–3.2 | **1.3569** |
| Val perplexity (M6, full val set, document-level) | ~15–25 | **3.719, 95% CI [3.693, 3.745]** |
| vs trigram / unigram / uniform baselines | beat them | **3.719 vs 23.4 / 379.0 / 32000** |
| Rubric: grammar / coherence / on-topic / repetition (n=60) | — | **4.00 / 3.15 / 4.12 / 3.98** out of 5 |
| Generation | Coherent short stories; some repetition | Grammatical; entity/character drift over ~150 tokens is the binding constraint, not grammar or n-gram repetition |

The original targets were a guess made before any data existed to measure against, same
as the token-count and param-count corrections in M1/M3. TinyStories' narrow, templated
vocabulary (`docs/DATA_CARD.md`) lets a model this size reach much lower loss than the
same size would on general-domain text — the gap is a property of the dataset, not a
signal that something is wrong. Full M5 run details, including an honest note that
`best.pt`'s training-time selection was noisy, in `experiments/001_baseline/run.md`;
**M6's full-val-set eval resolves that ambiguity** (`eval/results.md`) — `latest.pt`
(iter 40,000) really is the better checkpoint. M6 also traced (not assumed) a
data-quality finding: 6.20% of TinyStories' own train documents carry a mojibake
artifact inherited from the dataset's upstream pipeline, reproduced faithfully by
Sol-001 in a handful of generations — see `docs/DATA_CARD.md` and `docs/LIMITATIONS.md`.

<details>
<summary>Original "Expected results (realistic)" table, pre-measurement</summary>

| Metric | Target |
|--------|--------|
| Val loss | ~2.8–3.2 |
| Perplexity | ~15–25 |
| Generation | Coherent short stories; some repetition |

</details>

## Ablations — measured, M7

| ID | Variable | Values | **Result** |
|----|----------|--------|--------|
| 000 | VRAM / throughput benchmark | batch × checkpointing × compile sweep | `experiments/000_benchmark/results.md` |
| 001 | Baseline | Default config | `experiments/001_baseline/run.md` — ppl 3.719 |
| 002 | Learning rate | 1e-4 vs 3e-4 vs 1e-3 | **1e-3 best (4.162 ppl)**, real effect, 20–200× seed noise |
| 003 | Data scale | 100M vs full (357.85M) tokens, **equal iteration count** | full corpus wins but by a small margin (4.489 vs 4.572 ppl) — the milestone's stated negative/null result |
| 004 | **Seed variance** | seeds 42 / 43 / 44, baseline config | **4.490 ± 0.0045 ppl** — the yardstick for 002/003 above |

Configs: `configs/ablations/`. Results: `experiments/002_lr_sweep/results.md`,
`experiments/003_data_scale/results.md`, `experiments/004_seed_variance/results.md`.

**004 is what makes 002 and 003 interpretable.** With a single run per arm there is no
way to tell a real learning-rate effect from run-to-run noise. Each ablation's gap is
reported against the seed-to-seed standard deviation, stating plainly whether it clears
that bar (both do — 002 by a wide margin, 003 narrowly). n=3 is too few for a t-test,
so it isn't computed — the range/sd is reported directly instead, per the milestone's
own instruction.

Ablations ran at reduced `max_iters=8000` (not the 40k baseline) to fit the compute
budget — disclosed in every ablation config's header and in
`docs/ROADMAP.md`'s M7 section, not papered over. (`docs/ABLATIONS.md` was the
originally-planned filename for this writeup; the actual results ended up living in
`experiments/*/results.md` instead, alongside every other milestone's — no separate
file was created, so this correction points at where the real numbers are.)

## Evaluation plan

### Automatic

- Validation perplexity over the **full** val set, with a bootstrap 95% CI over documents
- Per-length-bucket perplexity (short vs long stories)
- Training/val loss curves (W&B)
- Repetition measured objectively: distinct-2 / distinct-3 n-gram ratios, max repeated
  substring length — so the manual rubric is not the only evidence

### Baselines (not optional — a perplexity number means nothing without a floor) — measured, M6

| Baseline | Expected ppl | **Measured ppl** |
|----------|--------------|-------------------|
| Uniform over vocab | 32,000 | **32,000.000** |
| Unigram frequency | — | **379.010** |
| Trigram + stupid backoff (Brants et al. 2007) | — | **23.425** |
| Sol-001 | target 15–25 | **3.719** |

Trigram uses stupid backoff, not add-k backoff as originally planned — simpler, standard
for n-gram baselines at this scale, and doesn't need a full normalized distribution since
only the probability of the *actual* next val token is ever queried
(`src/baselines.py`). Fit on a 10M-token prefix of train, not the full 357.85M-token
corpus — a documented scope decision, not a shortcut: a full-corpus trigram table needs
an on-disk sparse structure out of scope for a reference floor.

### Qualitative — measured, M6

- 60 fixed prompts, categorised (15 each: story-start, dialogue, continuation,
  out-of-domain probe) — `eval/prompts.jsonl`
- Rubric with **anchored descriptions per score**: grammar, coherence, on-topic,
  repetition (1–5) — `docs/RUBRIC.md`
- **Single rater, stated as such** (not literally blind across models — there is only one
  model). `docs/RUBRIC.md` explains exactly what "blind" means here and what it doesn't;
  this is the main validity limitation of the qualitative scores and is not hidden.
- Sample outputs saved per checkpoint: `eval/generations.jsonl`, scores in
  `eval/rubric_scores.csv`, aggregate in `eval/results.md`

## Deployment

- **Phase 1:** `src/infer.py` CLI generation
- **Phase 2:** `app/demo.py` Gradio UI, deployed to a free **Hugging Face Space** (CPU is
  ample for 52M params) — gives the portfolio a real `liveUrl`
- **Weights:** published to a HF **model** repo, fetched by the Space at startup.
  Never committed to git (`.gitignore` blocks `*.pt`).
- **Stretch:** FastAPI endpoint with p50/p95 latency notes

## Interview narrative (one-liner)

> I trained a 52M-parameter decoder-only transformer from scratch on TinyStories using a custom data pipeline and eval harness on a single 8 GB laptop GPU, with documented ablations on learning rate and training data scale — and a seed-variance run so I could say which of those differences were real.

## Skills mapping

| Skill area | Where it shows up in Sol |
|------------|--------------------------|
| Data science | EDA, cleaning, data card |
| ML fundamentals | Architecture, loss, training loop |
| Experimentation | W&B, ablations, configs |
| ML engineering | Reproducible pipeline, checkpointing |
| AI engineering | Demo deployment, inference CLI |
| Communication | README results table, limitations |

## Resolved decisions

- [x] **Training base:** hand-written `src/model.py`, nanoGPT as reference. Not litgpt,
      not a fork — the architecture code *is* the portfolio artifact.
- [x] **Entrypoints:** documented `python -m src.x` commands in `docs/COMMANDS.md`.
      No Makefile; `make` is not reliably present on Windows.
- [x] **Precision:** BF16 (Ada Lovelace).
- [x] **Demo hosting:** Hugging Face Space, linked from the portfolio via `liveUrl`.
- [x] **Repo:** public on GitHub, linked via `repoUrl`.

## Open decisions (resolve during implementation)

- [ ] Exact TinyStories subset download strategy (M1)
- [ ] `torch.compile` on/off — decided by measurement in M4, not by preference
- [ ] Near-dedup aggressiveness — depends on the exact-dup rate measured in M1
- [ ] Whether `max_iters` stays at 40000 — cut to ~24000 if M4 projects > 20 h

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [litgpt](https://github.com/Lightning-AI/litgpt)
- [TinyStories paper/dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
