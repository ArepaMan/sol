# Sol — final QA checklist

Two halves: what a machine can check (automated, run it in one go) and what
only a human can judge (does the output read like a story, does the page read
like a portfolio piece). Skipping the second half is how projects ship with
green tests and an embarrassing demo.

Last full pass: M9, all green except the noted open item.

---

## A. Automated — run these first

```powershell
python -m pytest -q; python -m scripts.export_spec --check; python -m ruff check src tests scripts
```

Expect **134 passed**, `spec is current`. `ruff` reports pre-existing findings
in M1–M6 modules and the EDA notebook; nothing in `app/`, `src/infer.py`,
`scripts/export_spec.py`, `scripts/plot_ablations.py`.

### The clone test — the one people skip

Everything above passes on a machine that already has the data. It says nothing
about what an interviewer gets. Do this instead:

```powershell
git clone https://github.com/ArepaMan/sol.git $env:TEMP\sol-qa; cd $env:TEMP\sol-qa; python -m pytest -q
```

Expect **129 passed, 5 skipped** — the 5 skips are the pipeline integration
tests that need the gitignored corpus, and they must say **skipped**, not
failed. (M9's QA found them *failing*: the skip guard keyed on `stats.json`,
which is committed, instead of the `.jsonl`/`.bin` files, which aren't. Five red
tests on a stranger's first `pytest` reads as a broken project.)

### Determinism

```powershell
python -m src.infer --seed 42 --max-new-tokens 120 > a.txt 2> $null; python -m src.infer --seed 42 --max-new-tokens 120 > b.txt 2> $null; fc.exe a.txt b.txt
```

Must report no differences. Timing goes to stderr precisely so this is clean.
Note determinism is **per device** — a CUDA seed-42 story and a CPU seed-42
story differ, and that's expected, not a bug.

### Public endpoints

All four must resolve anonymously (no token, logged out):

| URL | Expect |
|---|---|
| <https://sol-52m.streamlit.app> | 200/303 → app loads |
| <https://huggingface.co/SpicyGuac/sol-001> | 200 |
| `…/sol-001/resolve/main/config.yaml` | 200 (weights readable without auth — the Space depends on it) |
| <https://github.com/ArepaMan/sol> | 200 |

---

## B. Manual — 15 minutes, and only you can do it

### 1. The live demo, in a real incognito window

Not a logged-in tab. <https://sol-52m.streamlit.app>

- [ ] Loads without an error screen
- [ ] Note **how long** until the story box appears — if the app had slept
      >12 h, this is the wake-from-sleep cold start that is still unmeasured.
      Write the number down; `docs/DEPLOY.md` has a conservative placeholder
      waiting to be replaced with it.
- [ ] Click **Write the story**. Text should **stream** in, not appear at once.
- [ ] The story **ends by itself** before the token budget, on a complete
      sentence. (If it runs the full 200 and stops mid-word, the EOT-stop
      regressed.)
- [ ] Throughput caption reads roughly **15–25 tok/s**.

### 2. Read three stories like a skeptic

Generate three with different seeds and actually read them.

- [ ] Grammar is clean — this should be the model's strong suit (4.00/5).
- [ ] You can see the documented weakness: characters drift, a name changes,
      an object appears from nowhere (coherence 3.15/5). **You should be able
      to find this.** If the stories look flawless you're reading too kindly;
      the claim in the docs is that this flaw is real.
- [ ] No crash, no empty output, no wall of repeated text.

### 3. Break it on purpose

- [ ] Ask it a question: *"What is the capital of France?"* → it should
      continue the sentence as story prose, **not** answer. That's the
      documented no-instruction-tuning limitation, working as described.
- [ ] Empty prompt → a friendly warning, not a traceback.
- [ ] Paste ~600 words → should still generate (prompt truncated to its tail).
- [ ] Temperature to 1.5, then to 0.1 → visibly wilder, then visibly repetitive.

### 4. The About / Limitations tab

- [ ] Every claim on it matches what you just saw. This tab is the project's
      honesty surface; a wrong number here costs more than a bug.

### 5. The portfolio page

Local: `npm run build; npx next start -p 3100` → <http://localhost:3100/projects/sol>

- [ ] Status badge reads **Finished**, not Planned
- [ ] **Live demo** and **Source** buttons both resolve
- [ ] The Results table renders as a **table**, not literal `|` pipes
- [ ] Four figures visible, each with a caption containing a real number
- [ ] Spec explorer: all four tabs (Results / Architecture / Training /
      Ablations) show real values
- [ ] Nothing anywhere says 2070, float16, coming-soon, or not started

### 6. The five-minute skim

Open the GitHub README as if you'd never seen it and give it five minutes.

- [ ] Can you tell what the project is in one paragraph?
- [ ] Is there a number next to every claim?
- [ ] Do the three "what I'd do next" items sound honest rather than defensive?
- [ ] Would you click the demo link?

### 7. Phone

- [ ] Open the live demo on your phone. Interviewers do this and nobody tests
      it. Streamlit is responsive by default, but the sidebar collapses —
      confirm the controls are still reachable.

---

## Known open item

**The wake-from-sleep cold start is unmeasured.** It needs 12 h of no traffic
on the app, then a first visit. `docs/DEPLOY.md` carries a deliberately
conservative "~30-60s" in the UI copy until there is a real number. This is
tracked, not forgotten — measure it opportunistically and update both.
