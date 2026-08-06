# M6 — Repetition metrics (automatic backstop for the manual rubric)

60 generations, 150 max new tokens, temperature=0.8, top_k=200.

| Category | n | distinct-2 | distinct-3 | max repeated substring (mean / max) |
|---|---|---|---|---|
| continuation | 15 | 0.938 | 0.991 | 2.9 / 4 |
| dialogue | 15 | 0.927 | 0.980 | 3.5 / 7 |
| out-of-domain | 15 | 0.949 | 0.989 | 3.1 / 4 |
| story-start | 15 | 0.916 | 0.976 | 3.6 / 6 |
| **all** | 60 | 0.933 | 0.984 | 3.2 / 7 |
