# Phase 10 — Interface

**State:** ⬜ not started · **Effort:** ~1 week

## Question

Can an analyst get from a signal to the specific complaints that produced it,
without trusting the system?

## Context

**This phase is deliberately last and the most cuttable.** Nothing in the
research question depends on it: the result — `B1 > B2 = B3 > HarmScope > B0` —
is complete without a single line of UI, and a console built before the numbers
existed would have been effort spent making a null result look impressive.

The one requirement that is not cosmetic is **evidence linkage**. Every signal
must resolve to the complaint IDs behind it. A growth number a user cannot audit
is a number they have to take on faith, and this project's entire posture is that
nobody should have to.

That constraint already shaped the schema: `signals` carries both `n_supporting`
(raw complaints) and `n_supporting_groups` (distinct dup-groups) precisely so a
reader can see that "400 complaints in 3 groups" is weak.

## What runs (planned)

1. FastAPI service over `signals`, `clusters`, and the evidence join.
2. Analyst console: signal list → cluster detail → exemplar narratives →
   complaint IDs.
3. Streamlit fallback if the React app is cut.

## Tech (planned)

| Choice | Why this one |
|---|---|
| **FastAPI** | Reads the same DuckDB file; no new store, no sync problem. |
| **React + Vite + TS** | Analyst-facing console. |
| **Streamlit fallback** | Explicitly named in ROADMAP so that cutting the React app is a planned outcome rather than a failure. |

## What it must not do

- **Never present a signal without its denominator.** Standing rule 5.
- **Never imply wrongdoing.** Complaints are allegations. The README's "what it
  does not claim" section is a UI requirement, not just prose.
- **Never surface cluster identity as stable.** ARI is 0.505 across refits; a UI
  that shows "cluster #412" as a persistent object would be asserting something
  [Phase 4](phase-04-clustering.md) measured to be false.

## Honest note

Given the Phase 7 result, the most valuable version of this interface is probably
one that helps a reader interrogate *why the systems tie* — the granularity
table, the threshold sweep, the adjudication sample — rather than one that
presents HarmScope's alerts as a product.
