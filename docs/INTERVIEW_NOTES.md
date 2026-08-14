# Sol — interview notes

Ten skills, each with the artifact that proves it and the number to quote. If a
row has no number, it isn't evidence yet. (Nine were the roadmap's; the tenth,
inference optimisation, was added after it — the roadmap being finished is not
the same as the project being finished.)

Every figure here is generated into [`docs/spec.json`](spec.json) by
`scripts/export_spec.py` and guarded by `tests/test_spec_drift.py`, so this page
cannot quietly go stale — which is itself skill #7.

---

## 1. Problem framing

**Artifact:** `docs/PROJECT.md`, `configs/micro_50m_8gb.yaml`
**Number:** 512-token context, justified by **99.74% of train documents fitting
whole** — not by guesswork.

Scoped to "small model trained properly and measured honestly", not "ChatGPT
clone". Every locked hyperparameter has a written reason. The context length is
the cleanest example: it's a measured percentile of the actual corpus, so the
answer to "why 512?" is a number rather than a vibe.

**The thing worth telling:** the original spec targeted an RTX 2070 Super with
float16. That was the wrong machine. The hardware-aware sizing story survived;
the precision rationale changed to bf16 on Ada. Correcting a spec to measured
reality, in writing, is the habit — not getting it right first time.

## 2. Data pipeline

**Artifact:** `data/{prepare,train_tokenizer,tokenize}.py`, `docs/DATA_CARD.md`, `data/eda.ipynb`
**Number:** 2,119,719 raw → **1,748,358 train docs / 357,852,786 tokens**;
**14.58%** within-train exact-duplicate rate.

**The thing worth telling:** validation looked like it had a 28.67% duplicate
rate. It didn't — internal duplication was **0%**, and 28.67% of val documents
were exact duplicates of *train* documents. Cross-split leakage, not
redundancy. 6,304 documents dropped. Reporting "28.67% dupes" would have been
true and useless; finding out which kind of duplication it was is the actual
work, and it changes what you do about it.

## 3. Transformer fundamentals

**Artifact:** `src/model.py` (hand-written), `tests/test_model.py`
**Number:** **52,901,712 parameters, counted** — not estimated.

Pre-LN causal decoder, weight-tied embeddings, GPT-2-style scaled residual init.
The causality test is the one that matters: it asserts token *t*'s logits are
unchanged when tokens after *t* are perturbed. That's the property the whole
architecture exists to guarantee, and it's cheap to test and easy to break.

**The thing worth telling:** `n_embd=512` measured 41.8M against a project
branded "~52M" — a 20% gap. 592 (= 74×8, keeping `head_dim` integral) lands
within 1.73% while leaving every other locked value untouched. The parameter
count is *measured in a test* precisely so a claim like that can't drift.

## 4. Training

**Artifact:** `src/train.py`, `src/benchmark.py`, `experiments/000_benchmark/results.md`
**Number:** peak **2,344 MiB** against a 7,400 MiB budget; **~20,067 tok/s**
sustained; 30.3 h wall-clock (16.9 h training, 12.17 h with the laptop asleep).

Cosine LR with warmup, grad clipping, gradient accumulation, checkpoint/resume
with RNG state — verified by resuming and reproducing the identical loss curve.

**The thing worth telling:** the spec said gradient checkpointing was
*required*. Measurement said otherwise: 2,137 MiB without it at ~21,500 tok/s
vs 1,460 MiB with it at ~17,000 tok/s — 26% slower to solve a problem that
didn't exist at this batch size. It was turned **off**. Also learned: exceeding
VRAM on Windows doesn't cleanly OOM, it silently spills to system memory at
~15% throughput, which is a far nastier failure than a crash.

## 5. Evaluation

**Artifact:** `src/eval.py`, `src/baselines.py`, `eval/results.md`, `docs/RUBRIC.md`
**Number:** **perplexity 3.719, 95% CI [3.693, 3.745]** on the full
15,141-document val set, vs trigram **23.4**, unigram **379.0**, uniform
**32,000**.

Document-level, full-val-set, bootstrap CI over 10,000 resamples. Three
baselines, because "perplexity 3.7" means nothing without knowing what trivial
models score. Plus an anchored 1–5 rubric over 60 prompts, with automatic
repetition metrics (distinct-2/3) as the objective backstop against rubric
theatre.

**The thing worth telling:** the original target was perplexity 15–25. Measured
3.719. That is not a triumph — it means the target was wrong about how hard
TinyStories is. The interesting result is elsewhere: **grammar 4.00/5 but
coherence 3.15/5**. The model writes clean sentences and loses track of who's
in the story. Perplexity alone would never have surfaced that.

## 6. Ablations

**Artifact:** `experiments/{002_lr_sweep,003_data_scale,004_seed_variance}/results.md`
**Number:** seed variance **4.490 ± 0.0045 ppl** (n=3) — the yardstick
everything else is read against.

- **Learning rate:** 1e-4 → 5.380, 3e-4 → 4.489, 1e-3 → **4.162**. Gaps of
  20–200× the seed-noise floor. Real, large.
- **Data scale:** 100M → 4.572 vs full corpus → 4.489. Only **18×** seed noise,
  a ~1.8% relative difference.

**The thing worth telling:** the seed-variance run is the point. With n=3 a
t-test would be theatre, so the range is reported directly and used as the
yardstick. And the honest result is the null one: **data scale, the variable
that sounds like it should matter most, mattered least** — plausibly because
TinyStories is repetitive enough (14.58% exact dupes) that a 100M slice isn't
a punishing cut. Anyone can report the ablation that worked.

## 7. Reproducibility

**Artifact:** `scripts/export_spec.py`, `tests/test_spec_drift.py`, `docs/COMMANDS.md`
**Number:** **151 tests**; `python -m src.infer --seed 42` twice → byte-identical.

YAML is the single source of truth; nothing downstream hardcodes a
hyperparameter. Determinism is per-device and stated as such — CUDA seed 42 and
CPU seed 42 are each reproducible but produce *different* stories, because
different kernels round differently and one changed token diverges everything
after it. "Seeded" is weaker than people assume.

**The thing worth telling:** the numbers used to live in three places — this
repo's YAML, the portfolio's page, and a hardcoded TS component — and they
drifted. The portfolio claimed a 2070 Super, float16, n_embd 512, and "target
perplexity 15–25" long after all four were false. The fix isn't discipline,
it's a generator plus a test that fails when they disagree.

## 8. Deployment

**Artifact:** `app/streamlit_app.py`, `docs/DEPLOY.md`, <https://sol-52m.streamlit.app>
**Number:** **16.0 tok/s** deployed (199 tokens in 12.4 s), **6.7 s** model load,
**103.1 MiB** weight bundle.

Weights on the HF Hub, app on Streamlit Community Cloud, fetched via
`hf_hub_download`. Streams token-by-token and stops at the model's own
end-of-story token.

**The thing worth telling:** the plan was a free Hugging Face Space. Mid-milestone
HF moved Gradio Spaces behind PRO — `402 Payment Required`. The port to Streamlit
cost about an hour and touched **zero inference code**, because generation lives
in `src/infer.py` and the UIs are thin wrappers. Then the free tier's real cost
turned out to be memory, not money: a **1 GB RAM ceiling** vs a Space's 16 GB,
which is why `@st.cache_resource` is load-bearing (Streamlit reruns the whole
script per interaction; an uncached loader would allocate a fresh 212 MB model
per click). Two build failures worth knowing: Community Cloud defaults to Python
3.14 (torch 2.6.0 stops at 3.13), and a bare `torch==2.6.0` pulls the **CUDA**
wheel even with the CPU index listed, because `--extra-index-url` is additive
rather than preferential.

## 9. Honest limitations

**Artifact:** `docs/LIMITATIONS.md`
**Number:** on-topic **5.00/5 in-domain vs 1.47/5 out-of-domain**; mojibake in
**6.20%** of TinyStories' own training documents.

**The thing worth telling:** a mojibake artifact showed up in generations and
the obvious hypothesis was a tokenizer decode bug. A round-trip test said the
tokenizer was fine. Grepping the raw corpus found the pattern in 108,464 of
1,748,358 training documents — inherited from upstream, not introduced here.
The first hypothesis was wrong, and both the wrong hypothesis and the trace to
the real cause are in the git history rather than tidied away.

`docs/LIMITATIONS.md` is written as a rolling document per milestone, not
assembled at the end. Entries include ones that make the project look worse —
the 2-UI maintenance cost, the single-rater rubric, `best.pt` being selected
wrongly by a noisy periodic eval and superseded by `latest.pt` on
re-measurement.

## 10. Inference optimisation

**Artifact:** `src/model.py` (`KVCache`), `tests/test_infer.py`, `docs/DEPLOY.md`
**Number:** CPU **23.4 → 82.8 tok/s (3.5×)**, n=7 interleaved; GPU **124.4 →
132.4 (1.06×)**; peak RSS unchanged.

Generation re-ran the whole growing prefix through all 8 layers for every
token — O(n²) to produce n tokens, and every recomputation provably redundant
because causality means a past token's keys and values cannot depend on
anything after it. The cache is ~19 MB against ~400 MB of headroom.

The safety argument is one property, not a code review: cached and uncached
generation must be **byte-identical** from the same seed. That is a total
property — one flipped token diverges everything after it — so it is very hard
to pass by accident. The pre-cache path is kept reachable (`--no-kv-cache`)
specifically so the comparison stays runnable; "the old code still exists and
still agrees" beats "the new code looks right".

**The thing worth telling:** the GPU sped up by **6%**, not the ~4× the FLOP
count predicts, and that is the more interesting number. At batch 1 with a 52M
model, generation on a 4070 is bound by kernel-launch and Python-loop overhead
— 132 tok/s is ~7.5 ms per token across 8 layers, nowhere near that GPU's
arithmetic limit. Deleting redundant FLOPs from a workload that was never
FLOP-bound buys almost nothing. The CPU is genuinely compute-bound, and the CPU
is what the deployed app runs on, so users got the 3.5×. The optimisation
landed on the right target because of where the app is deployed, not because of
the reasoning that motivated it.

**The second thing worth telling:** the byte-identity test passed on CPU and
then failed on CUDA bf16, on 4 of 5 seeds. Chasing it on *logits* rather than
on stories settled it in one measurement — max |Δlogit| is 2e-6 of logit scale
in fp32 and 1e-2 in bf16, which is exactly 8 mantissa bits accumulated over 8
layers, while every argmax on trained weights still agrees. Rounding, not a
masking bug; a masking bug moves logits by order 1, not 1%. It is now a
gpu-marked test parametrised over both dtypes with the tolerances as the
payload. And the first GPU speedup measurement, 1.12×, was thrown out: running
all of one arm then all of the other let clocks drift: interleaving gave 1.06×
with ranges ten times tighter.

**And the limitation:** past `block_size` the cache buys nothing — 6.2 vs 5.9
tok/s, measured. Learned absolute positional embeddings mean a sliding window
re-embeds every surviving token one position lower and invalidates the entire
cache at once. RoPE would not fix it; the shift is in the embedding, not the
score. Which is a second, independent argument for the RoPE change that
`docs/LIMITATIONS.md` already wanted for coherence.

---

## Three questions to have an answer for

**"Why 52M and not something useful?"** Because the deliverable is the
*process*, and 52M is the largest model that trains to convergence in ~17 hours
on one 8 GB laptop GPU. A bigger model would have meant fewer experiments and no
ablations. The constraint drove the scope, and that trade is stated up front.

**"Your perplexity is 3.7, GPT-2 is ~20 on WebText — is yours better?"** No, and
the comparison is meaningless. TinyStories is a deliberately simple corpus with a
32k in-domain vocabulary; the numbers aren't commensurable. The defensible claim
is the one made in the repo: **3.719 vs a trigram's 23.4 on the same val set with
the same tokenizer**.

**"What would you do next?"** Coherence is the binding constraint (3.15/5 vs
4.00/5 grammar), so: RoPE instead of learned positional embeddings, and the M7
sweep's own finding that 1e-3 beat 3e-4 at the 8k-iter budget — which the
baseline run never got, since it predates the sweep. Then clean the inherited
mojibake. In that order, because the first is most likely to move the metric
that's actually limiting.
