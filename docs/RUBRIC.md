# Sol — Generation rubric (M6)

Anchored 1–5 scoring on four dimensions, applied to the 60 generations in
`eval/generations.jsonl` (prompts in `eval/prompts.jsonl`: 15 story-start, 15
dialogue, 15 continuation, 15 out-of-domain probes). Scores land in
`eval/rubric_scores.csv`.

## Rater and blinding — stated plainly

This project has one contributor. The generations were scored by the project
author (working with the AI assistant that also built the harness) reading
each completion **blind to its automatic-metric scores and without looking
back at which model produced it** — there is only one model (Sol-001) being
scored here, so "blind" mainly guards against being swayed by the distinct-n
numbers while judging coherence, not against knowing which checkpoint is
which. This is a **single-rater score, not an independently replicated
human-evaluation study** — see `docs/LIMITATIONS.md`. The automatic
repetition metrics (distinct-2/3, max repeated substring, in
`eval/repetition_summary.md`) exist specifically as an objective backstop to
this rubric, per the M6 risk note in `docs/ROADMAP.md` ("rubric theatre").

## Dimensions

1. **Grammar** — is the sentence-level English well-formed?
2. **Coherence** — does the text hang together as a sequence of events (even
   if simple), rather than reading as disconnected sentences?
3. **On-topic** — for story-start/dialogue/continuation prompts, does the
   completion stay inside a children's-story register (concrete, simple
   objects/emotions)? For out-of-domain prompts, this dimension instead
   scores whether the model visibly tries to imitate the prompt's register
   (finance/legal/technical) vs immediately reverting to story-mode — **both
   are informative, not a pass/fail**: TinyStories has zero non-narrative
   text, so reverting to story-mode on an out-of-domain prompt is the
   *expected*, honest failure mode, not a bug.
4. **Repetition (qualitative)** — independent human impression of looping /
   repeated phrases, to sanity-check the automatic distinct-n metric agrees
   with what a reader actually notices.

## Anchors (1–5, same scale for every dimension)

| Score | Meaning |
|---|---|
| 1 | Fails badly — incoherent, ungrammatical, or degenerates into a repetition loop within the first sentence or two |
| 2 | Mostly broken — occasional coherent phrase but no sustained sense |
| 3 | Understandable but flawed — coherent short stretch, then drifts, repeats, or contradicts itself |
| 4 | Good — a full simple story beat that holds together, minor slips only |
| 5 | Excellent — reads like an actual (simple) TinyStories entry end to end |

## Worked examples

**Example A — story-start prompt, scored 4/5 coherence, 5/5 grammar, 5/5 on-topic, 5/5 repetition.**

> Prompt: *"Once upon a time, there was a little girl named Lily."*
> Generation: *"...She loved to play in the garden with her dog. One day, she
> found a small box under a tree. She opened it and saw a shiny key inside.
> Lily was very happy and ran home to show her mom."*

Grammar is clean throughout. Coherence is high — a simple find-object →
react → resolve arc, standard TinyStories shape — docked one point only
because the box/key thread is dropped rather than paid off. On-topic: fully
in the register (garden, dog, mom, simple emotion words). No repeated
phrases.

**Example B — out-of-domain prompt, scored 2/5 coherence, 4/5 grammar, on-topic n/a (reverted, expected), 3/5 repetition.**

> Prompt: *"The quarterly earnings report showed a 12% increase in revenue driven by"*
> Generation: *"...a little girl named Sara. She was very happy because she
> had a new toy. She played with her toy every day and it made her feel
> good. She played with her toy and it made her feel good and she was
> happy."*

Grammar per-sentence is fine (4/5). Coherence is low (2/5): the completion
has no relationship at all to the prompt's topic or register — it does not
attempt financial language even briefly, it just starts a new, unrelated
TinyStories-style sentence. This is the expected out-of-domain failure mode
described above, not scored as a defect on its own; what *is* scored down is
the tail-end near-repetition ("she played with her toy... it made her feel
good" appearing twice with only "and she was happy" added), which drags the
qualitative repetition score to 3/5 even though it falls short of a hard
loop.

## How scores are aggregated

`eval/rubric_scores.csv` has one row per generation with columns
`id, category, grammar, coherence, on_topic, repetition, notes`. Results are
reported in `eval/results.md` as **mean ± sd per dimension, overall and
broken out per category** — the sd is reported alongside the mean because
n=60 single-rater scores are noisy enough that the point estimate alone
would overstate precision.
