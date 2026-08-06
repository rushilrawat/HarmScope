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

**Phase 5 complete.** Nothing downstream has run, so there are no results yet —
see the empty table at the bottom, which stays empty until Phase 9.

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
for the full corpus. **The Phase 4 gate numbers above are therefore provisional**
and must be reproduced on bge-base before Phase 6 freezes anything.

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
| 6 — Ground truth & backtest **[GATE]** | ⬜ |
| 7 — Baselines | ⬜ |
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

_Populate after Phase 9. Do not write numbers here until the backtest has run._

**Provisional, unadjudicated, dev model.** Not the headline — the headline needs
blind human adjudication (`EVALUATION.md` §1.3) and a `bge-base` encode.

| System | Detected | Rate | Median lead* |
|---|---|---|---|
| B1 — CFPB taxonomy | 63 / 112 | **56.2%** | 1518 d |
| HarmScope | 57 / 112 | 50.9% | 1747 d |
| B0 — volume only | 46 / 112 | 41.1% | 1954 d |

\* Lead times are company-level and inflated; see `ENGINEERING_NOTES.md` Phase 6.

**As it stands, the taxonomy baseline wins.** The ordering is
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
