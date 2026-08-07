# Phase 6 — Ground truth & backtest **[GATE]**

**State:** ✅ gate passed · **Effort:** ~1.5 weeks

## Question

Does a signal appear *before* the corresponding public enforcement action — and
is that measurable without the model having seen the future?

## Context

This is where the project can be wrong in ways that are invisible from inside it.
Two exposures, handled differently:

**Leakage** is preventable, so it is prevented structurally. Nothing here filters
an existing artifact by date; each date-dependent stage is *recomputed* from
inputs restricted to `date_received < cutoff`. What is refit and what is not
follows from whether a stage's output can depend on when a complaint arrived:

| Stage | | Why |
|---|---|---|
| embeddings, `dup_pairs` | reused | A vector and a pairwise Jaccard do not change because another complaint exists |
| grouping, campaigns | **refit** | Seed selection and time-windowed features are both date-dependent |
| clustering, novelty | **refit** | Cluster definitions are *the* leakage vector |
| panel, signals | **refit** | Obviously |

**Curation order** is not preventable, so it is bounded and stated.

## What runs

1. Scrape 212 enforcement actions; **freeze before any backtest**, git SHA
   recorded.
2. Mechanically curate to 112 usable.
3. Refit every date-dependent stage at each of 8 annual cutoffs.
4. Resolve each action against *its own cutoff's* refit.
5. Blind adjudication: `worklist.py` emits blinded candidates, `verdicts` reads
   them back.

## Tech

| Choice | Why this one |
|---|---|
| **Annual cutoffs, 2017–2024** | Deliberately conservative: an action filed in December is evaluated against a model up to twelve months stale, so every lead time understates the truth. Understating your own result is the right direction to err. |
| **`run_for_cutoff` resolves by recorded cutoff, never recency** | A missing refit is an error rather than a silent fallback to the run that saw everything. |
| **Blinding is structural** | The leakage test forbids `adjudicate.py` from naming any signal column, which forces the split: `worklist.py` alone reads signal strength and strips every trace of why a candidate was chosen. |

## Acceptance

| Criterion | Result | Verdict |
|---|---:|---|
| All 7 anti-leakage tests in CI | 7 covered | 6 pass, 1 `xfail(strict)` |
| One cutoff refits end-to-end, documented wall time | **3.2 min** | pass |
| Adjudication hides signal status | structural | pass |

## Findings

**The refit found two real leaks before a single test was written.** `as_of` and
the signal date were read from the whole corpus rather than the data in scope, so
a 2017 refit stamped its 367 clusters `as_of 2026-08-03` and dated all 2,953
disproportionality signals *after their own cutoff*.

This one is worth dwelling on: `EVALUATION §5` item 2 asserts the leakage
guarantee **on** `as_of`. That field being wrong silently voids the check meant
to catch it.

**Item 4 is written to fail**, marked `xfail(strict=True)` so it would also fail
if it started passing for the wrong reason. The curated ground truth cannot have
a SHA predating a detection run that already happened, and the test asserts the
real requirement rather than being pointed at the candidates file to go green.

**The methodological exposure, stated plainly.** The candidate pool was frozen
pre-detection and that is verifiable from git. But curation happened *after*
Phases 4–5 ran, so the curator knew which companies alert. That cannot be undone,
only bounded:

- Both curation rules were pre-registered and committed *before* the code
  applying them existed.
- The `usable` rule has **no free parameter** — no minimum-volume threshold, so a
  three-complaint company stays in and produces an honest miss.
- Bias audit: resolved companies' alert share **0.695 observed vs 0.675
  expected**, a 1.03× excess. Not proof of absence, but the strongest cheap check
  available.

**Adjudication roughly halves both systems** — see [Phase 7](phase-07-baselines.md)
for the numbers. Only ~57% and ~50% of company-level detections survive being
asked whether the cluster describes the harm the order describes. **A
company-level fire is not a harm-level match**, and that correction is far larger
and far better supported than any difference between systems.

**Three actions neither system can reach, for structural reasons.** Bank of
America's order concerns its own processing of garnishment notices; Citibank's is
ECOA discrimination, which applicants are never told is the reason for denial;
Santander's is a GAP add-on disclosure. Complaint narratives cannot contain what
consumers do not know happened to them. A ceiling on the corpus, not on either
method.
