# Phase 9 — Evaluation & write-up

**State:** 🔄 two items pulled forward; the rest not started · **Effort:** ~1 week

## Question

Can a skeptical reader find, in the repo, the exact reason for every claim and
every failure?

## Context

Two Phase 9 items were run early because they changed how earlier phases read,
and leaving them until the end would have meant reporting a result whose
foundation was untested. Both are recorded here and in
`ENGINEERING_NOTES.md`.

## Item pulled forward 1 — threshold sensitivity

ROADMAP requires a sensitivity analysis on every threshold in `config.py`. The
one that gates an alert — `min_supporting_groups`, frozen at 15 — was swept
because the Phase 7 headline turned on it.

| Floor | HarmScope | B1 | Gap |
|---:|---:|---:|---|
| 1–5 | 67.9% | 66.1% | HarmScope +1.8 |
| 10 | 55.4% | 58.0% | B1 +2.6 |
| **15 (frozen)** | 50.9% | 56.2% | **B1 +5.3** |
| 25 | 46.4% | 50.0% | B1 +3.6 |
| 50 | 44.6% | 44.6% | tie |
| 100 | 40.2% | 38.4% | HarmScope +1.8 |

**The ordering changes sign three times, and the frozen value sits in the band
that favours B1.** The mechanism is granularity, measured not asserted:
HarmScope splits the same complaints into 2,010 units against B1's 634, so its
support-per-unit distribution sits lower by arithmetic — median 15 against 18,
clearing the floor 51.6% against 57.3%.

**The threshold was not moved.** It was frozen before the backtest, and changing
it after seeing which system it favours is exactly trap T4. The defect is that an
*absolute* support floor is not comparable across systems with different
granularity; **the fix is a floor expressed as a quantile of each system's own
distribution**, recorded as future work rather than applied retroactively.

## Item pulled forward 2 — does the encoder choice matter?

The README carried a "provisional, dev model" flag since Phase 3, with a full
`bge-base` re-encode as the implied fix: 16.8 h plus a re-run of Phases 4–7.
Instead the question was scoped to what one family can answer.

150,000 credit_reporting representatives at the 2024 cutoff, encoded on both
models, clustered with identical config. Paired by construction.

| Metric | MiniLM | bge-base |
|---|---:|---:|
| clusters | 336 | 350 |
| fit noise | 65.4% | 63.7% |
| assigned at the fixed 0.65 floor | 94.5% | 100.0% |
| nearest-centroid cosine p10 / p50 | 0.684 / 0.794 | 0.831 / 0.890 |
| ARI vs itself, disjoint halves | 0.469 | 0.454 |
| **ARI across the two encoders** | **0.232** | |

Two guards made this readable, and both mattered:

**The cross-encoder ARI is meaningless without its within-encoder baseline.**
0.232 alone looks like proof of anything you like. Against a within-encoder
disjoint-halves ARI of 0.469 on the same sample — reproducing the 0.505 already
recorded — the honest statement is that the encoder disagrees about twice as much
as resampling does.

**`assign_max_distance = 0.35` is calibrated to MiniLM's geometry.** bge sits on
a visibly higher cosine scale, so its 100% assignment at a fixed 0.65 floor is
the threshold moving, not coverage improving. Reporting that as coverage would
have reproduced the support-floor artifact one entry after diagnosing it.

**Verdict.** Cluster *identity* is encoder-sensitive, so the provisional flag was
justified. Cluster *granularity* is not, at 4% apart — and granularity is what
drives the Phase 7 table. The swap is unlikely to move the ordering. The full
encode stays outstanding as a whole-table item, since a cross-system comparison
needs every system on one encoder.

Reproduce with `notebooks/export_bge_probe.py` → `encode_bge.py` →
`compare_encoders.py`. The encode step must run first; `bge_vectors.npy` is a
scratch artifact and is not committed.

## Still to do

- Failure analysis table for every missed action and the top 20 false alerts.
- Sensitivity analysis on the *remaining* thresholds in `config.py`.
- Quantile-floor rerun, as declared future work above.
- Calibration and subgroup breakdown.
- Limitations section from `METHODOLOGY §7`.
- Full `bge-base` encode and re-run, for the whole table at once.

## Acceptance

> A reader who is skeptical of the result can find, in the repo, the exact reason
> for every claim and every failure. **No number appears without its
> denominator.**
