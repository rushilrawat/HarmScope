# HarmScope

Emergent consumer-harm discovery over the CFPB Consumer Complaint Database.

Finds harm mechanisms that the CFPB's own `Product / Issue / Sub-issue` taxonomy does not
represent, tracks their growth, and flags company–harm pairs that are growing abnormally —
validated by whether the signal appears *before* the corresponding public enforcement action.

**This is not a complaint classifier.** Classification over CFPB data is a solved, crowded
problem. The contribution is open-world discovery + temporal validation against enforcement
lead time.

---

## Status

**Phases 0–7 complete.** All four baselines are in `baseline_results` and the
Results table at the bottom carries real numbers.

The headline, which four independent examinations now agree on: **the pipeline
beats naive volume by about 10 points and is beaten by, or tied with, every
other way of defining a unit that was tried.** The ordering is
`B1 (56.2%) > B2 = B3 (51.8%) > HarmScope (50.9%) > B0 (41.1%)`. Three separate
apparent effects have dissolved on examination — B1's unadjudicated +5.3 was a
threshold artifact, HarmScope's n=8 adjudication reversal was a sampling
artifact, and adjudication at n=14 put the two at 29.1% vs 28.1% with p=1.00.

The sharpest single result is B2, because it is the only comparison where
granularity was controlled by design rather than left to chance: **a TF-IDF +
LDA topic model at exactly HarmScope's unit count detects 58 actions to
HarmScope's 57.**

**Every system in the comparison runs on `all-MiniLM-L6-v2`, not the
`bge-base-en-v1.5` default**, and a scoped experiment now says what that costs.
150,000 credit_reporting representatives at the 2024 cutoff were encoded on both
models and clustered with identical config:

| | MiniLM | bge-base |
|---|---|---|
| clusters | 336 | 350 |
| nearest-centroid cosine, p50 | 0.794 | 0.890 |
| ARI vs itself, disjoint halves | 0.469 | 0.454 |
| ARI across the two encoders | **0.232** | **0.232** |

**The encoder changes which cluster a complaint lands in about twice as much as
resampling does** (0.232 against ~0.46), so the "provisional" flag on the Phase 4
gate numbers was justified. But it barely changes *how many* clusters there are —
336 against 350, a 4% difference — and granularity, not cluster identity, is the
mechanism behind the threshold artifact that drives the table above. Note also
that `assign_max_distance = 0.35` is calibrated to MiniLM's geometry: bge sits on
a visibly higher cosine scale, so its 100% assignment rate at that fixed floor is
an artifact of the threshold, not better coverage. Same defect class as the
support floor.

A full `bge-base` re-encode is therefore still outstanding, and it is a Phase 9
item for the whole table at once rather than a patch to one row: a cross-system
comparison needs all systems on one encoder, so re-running HarmScope means
re-running B3 too, and B0/B1/B2 use no embeddings at all.

Phase 5's mandatory negative control — shuffle cluster labels, re-run detection,
and see how much fires — **failed first at a 10.7% false-alert rate and found
three bugs**, one of them in the control itself. After the fixes it runs at
0.45% against an α of 0.05. The sequence is in `ENGINEERING_NOTES.md`; the
short version is that all three bugs computed a "distinct groups" quantity as a
sum over a partition, and all three produced plausible numbers that only a null
could expose.

`METHODOLOGY §4.3` requires the stability ARI to appear here whatever it says.
**Disjoint-halves ARI is 0.505** (credit_reporting) and **0.541** (mortgage).
The partition's granularity and coverage are highly reproducible — 688 vs 693
clusters, assignment 96.1% vs 96.1% — but which cluster a given complaint lands
in agrees about half the time. A cluster here is a region of a dense
neighbourhood, not a canonical object, and nothing downstream may treat cluster
identity as stable across refits. Cluster count was still climbing at the 500k
fit sample, so the partition has not converged.

Phases 3 and 4 both ran on `all-MiniLM-L6-v2`, the model `METHODOLOGY §3` names
for iteration, not the `bge-base-en-v1.5` default — 2.1 h versus 16.8 h measured
for the full corpus. **The Phase 4 gate numbers are therefore provisional.** They
were not reproduced on bge-base before Phase 6, as this section originally
required; the encode is still outstanding and is recorded as such rather than
quietly dropped.

The Phase 2 gate passed on a **changed** criterion: ROADMAP asked for 300
hand-labelled pairs, and no human was available, so the disagreements were
adjudicated by the model, blind, against a rule written down beforehand
(`METHODOLOGY §2.4.1`). Recorded in `ENGINEERING_NOTES.md` under Reversed
decisions rather than presented as satisfying the original bar.

| Phase | State |
|---|---|
| 0 — Scaffold | ✅ schema, config, run registry, checks, normalization, CI |
| 1 — Ingestion & normalization | ✅ 16.5M complaints, 3.83M narratives, crosswalk covers both schema eras |
| 2 — Dedup & campaign detection **[GATE]** | ✅ gate passed — precision 1.000 on blind-adjudicated labels, recall 0.765 |
| 3 — Embedding & index | ✅ 2.48M vectors, 10/10 neighbour checks — on the dev model, see note |
| 4 — Clustering & novelty **[GATE]** | ✅ gate passed — ablation AUC 0.791; **disjoint-halves ARI 0.505** |
| 5 — Signal detection | ✅ negative control passes at 0.0045 vs α 0.05 |
| 6 — Ground truth & backtest **[GATE]** | ✅ gate passed — refit at one cutoff in 3.2 min, 7 anti-leakage checks, blind adjudication on 14 actions |
| 7 — Baselines | ✅ all four in `baseline_results` — B1 56.2% > B2 = B3 51.8% > HarmScope 50.9% > B0 41.1% |
| 8 — LLM layer | ⬜ |
| 9 — Evaluation & write-up | ⬜ |
| 10 — Interface | ⬜ |

---

## Research question

> Can unsupervised harm-mechanism discovery over complaint narratives surface company–harm
> signals earlier than monitoring the existing CFPB taxonomy, without a higher false-alert rate?

Falsifiable. Has a baseline that can beat us (see `docs/EVALUATION.md`). If existing
`(Product, Issue, Sub-issue)` tuples give the same lead time, the project has no contribution
and the README must say so.

---

## What it does

1. Ingests the full CFPB complaint corpus (bulk CSV, not API pagination).
2. Detects and collapses template / mass-filed / near-duplicate narratives.
3. Embeds narratives; clusters into candidate harm mechanisms.
4. Scores each cluster for **novelty** against the existing CFPB label taxonomy.
5. Detects **abnormal growth** per cluster and per company × cluster (disproportionality +
   changepoint), with FDR correction.
6. Labels surviving clusters with an LLM (harm mechanism, actors, preconditions) — labeling
   only, never detection.
7. Serves evidence: every signal links back to the specific complaint IDs that produced it.
8. Backtests against a hand-curated set of pre-2025 CFPB enforcement actions.

---

## What it does not claim

- Does not assert that any company broke the law. Complaints are **allegations**.
- Does not estimate true harm rates or prevalence — no denominator (customer counts).
- Does not predict individual complaint outcomes.
- Does not replace CFPB supervisory or enforcement judgment.
- Complaint volume is not harm volume. Reporting propensity varies by product, company size,
  demographics, and third-party filing activity.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Store | DuckDB (single file) | Analytical, no server, handles 10M+ rows on a laptop |
| Embeddings | `sentence-transformers` (`bge-base-en-v1.5`; `all-MiniLM-L6-v2` for dev) | CPU-viable, strong retrieval quality |
| ANN index | FAISS (on-disk IVF-Flat) | Nearest-neighbour + assignment at scale |
| Dedup | `datasketch` MinHash + LSH | Template detection is the core preprocessing problem |
| Dim. reduction | UMAP | Required before density clustering |
| Clustering | HDBSCAN | No fixed k; native noise class; matches "not everything is a harm mechanism" |
| Stats | `statsmodels`, `scipy`, `ruptures` | Disproportionality, negative binomial, changepoint |
| LLM | Anthropic API (`claude-sonnet-5`) | Cluster labeling + evidence synthesis, cached |
| API | FastAPI | Serves signals + evidence |
| UI | React + Vite + TS | Analyst-facing console (Streamlit fallback, see ROADMAP Phase 8) |

---

## Docs

| File | Purpose |
|---|---|
| `docs/PROJECT_SPEC.md` | Scope, success criteria, non-goals, ethics |
| `docs/DATA.md` | Field reference, quirks, biases, template problem, ground truth |
| `docs/ARCHITECTURE.md` | Module layout, DB schema, data flow, scale strategy |
| `docs/METHODOLOGY.md` | Dedup, embedding, clustering, novelty, anomaly detection |
| `docs/EVALUATION.md` | Backtest protocol, baselines, metrics, failure analysis |
| `docs/LLM_LAYER.md` | Labeling + RAG contract, prompts, caching, guardrails |
| `docs/ROADMAP.md` | Phased build with per-phase acceptance criteria |
| `docs/ENGINEERING_NOTES.md` | Running log of decisions, failures, gotchas |
| `data/ground_truth/README.md` | Curation and freeze protocol for the hand-built files |

---

## Standing rules

1. **Docs must match reality.** After each phase, update the relevant doc *before* starting
   the next phase. Stale docs are a defect.
2. **No silent success.** Every pipeline stage asserts on its own output (row counts, null
   rates, distribution shape) and fails loudly. A stage that "completed" without producing
   valid output is a bug, not a pass.
3. **Determinism.** Fix all seeds. Record `run_id`, git SHA, params, and input row counts in
   the `runs` table for every pipeline execution.
4. **No leakage.** Nothing after the backtest cutoff date may touch a model that is evaluated
   before that date. See `docs/EVALUATION.md`.
5. **Honest metrics only.** If lead time is 11 days, the README says 11 days.

---

## Quickstart

```bash
make init                                    # venv, deps, empty schema-valid DuckDB
make test                                    # test suite
make lint                                    # ruff
make download                                # CFPB bulk CSV snapshot (~1.4 GB compressed)
python -m src.pipeline runs                  # the run registry
```

Everything the pipeline runs is registered: `python -m src.pipeline runs` shows
phase, status, output rows, git SHA (with `-dirty` when the tree was not clean),
and config fingerprint for every execution.

Phases that are not built raise and name the roadmap phase that would build
them, rather than quietly doing nothing:

```
$ python -m src.pipeline run --phase cluster
phase 'cluster' is not built.
  ROADMAP Phase 4 [GATE] — UMAP + HDBSCAN + novelty scoring
```

---

## Results

**Provisional, unadjudicated, dev model, and incomplete — B2 and B3 are still
running.** Not the headline: the headline needs blind human adjudication
(`EVALUATION.md` §1.3), all four baselines, and a `bge-base` encode.

| System | Detected | Rate | Median lead* | Units at 2024 |
|---|---|---|---|---|
| B1 — CFPB taxonomy | 63 / 112 | **56.2%** | 1518 d | 634 |
| B2 — TF-IDF + LDA | 58 / 112 | 51.8% | 1810 d | 2,010 |
| B3 — BERTopic default, no dedup | 58 / 112 | 51.8% | 1679 d | 11,336 |
| HarmScope | 57 / 112 | 50.9% | 1747 d | 2,010 |
| B0 — volume only | 46 / 112 | 41.1% | 1954 d | 12 |

\* Lead times are company-level and inflated; see `ENGINEERING_NOTES.md` Phase 6.

**HarmScope finishes last of the four non-trivial systems.** Every alternative
way of defining a unit — the published taxonomy, a bag-of-words topic model, and
off-the-shelf BERTopic without any dedup — matches or beats the pipeline. The
whole spread from B1 to HarmScope is 5.3 points, which the threshold sweep below
shows is inside the range an arbitrary threshold choice can manufacture.

**B2 is the comparison that carries the most weight, because it is the only one
where granularity is not a confound.** B2's topic count was fixed, per family and
per cutoff, to HarmScope's own discovered cluster count — decided before B2 had a
detection rate, precisely so the artifact that produced B1's lead could not
operate here. It lands on 2,010 units at 2024 against HarmScope's 2,010, median
support 16 against 15, clearing the floor 54.4% against 51.6%. Matched on every
dimension that mattered, **a TF-IDF topic model detects 58 enforcement actions to
the embedding pipeline's 57.** `EVALUATION §2` says B2 "tests whether the
embeddings buy anything". On this evidence they do not.

B3 ties B2 at 58 while running with no dedup, no campaign detection, no novelty
scoring, and default hyperparameters — the `pip install bertopic` comparison.
Its numbers are not strictly commensurable: without dedup its groups are
singletons, so `min_supporting_groups` gates it on fifteen *complaints* where
every other system is gated on fifteen *dup-groups*. That was written down before
B3 ran, not discovered after.

**Adjudication closes the gap to nothing.** On 14 actions both systems detected,
judged blind against the orders' own descriptions:

| System | Strong match | Corrected rate |
|---|---|---|
| HarmScope | 8 / 14 | 29.1% |
| B1 — CFPB taxonomy | 7 / 14 | 28.1% |

Discordant pairs 2–1, exact McNemar *p* = 1.00. **The two systems are
indistinguishable on this evidence.**

Worth recording how that number moved. At n = 8 the split was 5–3 with 2–0
discordant, which looked like a reversal in HarmScope's favour and would have
been a tempting place to stop. Six more actions took it to 8–7 with 2–1. The
apparent effect did not survive its own sample growing.

**Adjudication also cuts both systems roughly in half**, because a company-level
fire is not a harm-level match. That correction is much larger and much better
supported than any difference between the systems.

**The unadjudicated gap is a threshold artifact.** ROADMAP Phase 9 requires a
sensitivity analysis on every threshold in `config.py`. Sweeping the one that
gates an alert — `min_supporting_groups`, frozen at 15 — the ordering **changes
sign three times**:

| floor | HarmScope | B1 | gap |
|---:|---:|---:|---:|
| 1–5 | 67.9% | 66.1% | HarmScope +1.8 |
| 10 | 55.4% | 58.0% | B1 +2.6 |
| **15 (frozen)** | 50.9% | 56.2% | **B1 +5.3** |
| 25 | 46.4% | 50.0% | B1 +3.6 |
| 50 | 44.6% | 44.6% | 0.0 |
| 100 | 40.2% | 38.4% | HarmScope +1.8 |

The frozen value sits in the band that favours B1, and the mechanism is
granularity rather than quality: HarmScope splits the same complaints into 2,010
units against B1's 634, so its support-per-unit distribution sits lower (median
15 vs 18) purely by arithmetic. **An absolute support floor is not comparable
across systems whose unit counts differ threefold** — in the mid-band it
measures granularity, not whether a unit describes a harm.

The threshold was frozen before the backtest and **is not being changed after
seeing this**. The finding is that no stable ordering exists, which agrees with
the adjudicated result rather than contradicting it.

**Unadjudicated at the frozen threshold, the taxonomy baseline leads.** The ordering is
`B1 > HarmScope > B0`: the pipeline is worth something over counting complaints
per company (+9.8 points on B0), and is not worth anything over the taxonomy CFPB
already publishes (−5.3 points on B1).

HarmScope produces 2,010 units to B1's 634, so each carries less evidence per
company — 51.6% of its company-level signals clear the support floor against
B1's 57.3% — and the finer partition loses more to that floor than it gains in
specificity. That is the bill for `cluster_selection_method='leaf'` arriving.

If this survives adjudication, the honest conclusion is that the existing
taxonomy is sufficient for this task, and this table stays exactly as it is.

| Metric | Value |
|---|---|
| Enforcement actions backtested | 112 (of 212 scraped) |
| Cutoffs | annual, 2017–2024, full refit each |
| Adjudication | **not yet performed** |
| False alerts per 1,000 company-months | — |
