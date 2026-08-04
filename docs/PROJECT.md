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

**Corpus cap:** 400M tokens. Note this is the *corpus size*, not the number of tokens
processed — at 40,000 iters × 32,768 tokens/step the model sees **1.31B tokens ≈ 3.3
epochs** over that corpus. Wall-clock is measured in M4, not estimated here.

**Why not full corpus:** Diminishing returns for portfolio; enables data-scale ablation (100M vs 400M).

### Data pipeline requirements

1. Download and subset
2. Clean: dedupe, length filter, basic quality heuristics
3. Train BPE tokenizer (32k vocab)
4. Encode to binary shards for training
5. EDA notebook: token counts, length distribution, data card in README

## Model architecture

Decoder-only GPT (causal LM), ~52M parameters:

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Layers | 8 | Fits 8GB with checkpointing |
| Heads | 8 | head_dim = 64 |
| Embedding | 512 | Standard small-GPT scale |
| Context | 512 | 1024 likely OOM on 8GB; validate against the measured token-length p90 in M2 |
| Vocab | 32k BPE | Standard for small LMs |
| Norm | LayerNorm, pre-LN | Hand-written; simpler to reason about than RMSNorm |
| Activation | GELU | 4× MLP, GPT-2 style |
| Positional | Learned | RoPE is listed as a "what I'd do next" in `docs/LIMITATIONS.md` |

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
gradient_checkpointing: true
compile: false  # revisit in M4 benchmark
```

**Why BF16 rather than FP16:** the RTX 4070 Laptop is Ada Lovelace (sm_89), which
supports BF16 natively. BF16 keeps FP32's exponent range, so it needs **no GradScaler**
and none of the loss-scale tuning FP16 demands — one fewer source of silent divergence
in a 20-hour run. (An earlier draft of this spec targeted a Turing-generation RTX 2070
Super, where FP16 would have been forced.)

**OOM fallback order:**

1. batch 2, grad accum 32
2. n_layer 6 (~39M params)
3. block_size 384

This ladder is exercised as a *measurement* in M4 (`src/benchmark.py`), not discovered
mid-run. Target: peak VRAM < 7400 MiB, leaving ~800 MiB headroom for Windows WDDM.

## Expected results (realistic)

| Metric | Target |
|--------|--------|
| Val loss | ~2.8–3.2 |
| Perplexity | ~15–25 |
| Generation | Coherent short stories; some repetition |

## Planned ablations

| ID | Variable | Values |
|----|----------|--------|
| 000 | VRAM / throughput benchmark | batch × checkpointing × compile sweep |
| 001 | Baseline | Default config |
| 002 | Learning rate | 1e-4 vs 3e-4 vs 1e-3 |
| 003 | Data scale | 100M vs 400M tokens, **equal iteration count** |
| 004 | **Seed variance** | seeds 42 / 43 / 44, baseline config |

Store configs and results under `experiments/`.

**004 is what makes 002 and 003 interpretable.** With a single run per arm there is no
way to tell a real learning-rate effect from run-to-run noise. Report each ablation's
gap against the seed-to-seed standard deviation and state plainly whether it clears
that bar. n=3 is too few for a t-test — say so rather than computing one.

Ablations run at reduced `max_iters` (~8000) to fit the GPU budget; that limitation is
disclosed in `docs/ABLATIONS.md` rather than papered over.

## Evaluation plan

### Automatic

- Validation perplexity over the **full** val set, with a bootstrap 95% CI over documents
- Per-length-bucket perplexity (short vs long stories)
- Training/val loss curves (W&B)
- Repetition measured objectively: distinct-2 / distinct-3 n-gram ratios, max repeated
  substring length — so the manual rubric is not the only evidence

### Baselines (not optional — a perplexity number means nothing without a floor)

| Baseline | Expected ppl |
|----------|--------------|
| Uniform over vocab | 32,000 |
| Unigram frequency | — |
| Trigram + add-k backoff | — |
| Sol | target 15–25 |

### Qualitative

- 60 fixed prompts, categorised (story-start, dialogue, continuation, out-of-domain probe)
- Rubric with **anchored descriptions per score**: fluency, coherence, repetition,
  prompt-adherence (1–5) — see `docs/RUBRIC.md`
- Scored **blind** (runs shuffled, ids hidden). Single rater is the main validity
  limitation and is stated as such.
- Save sample outputs per checkpoint in `experiments/`

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
