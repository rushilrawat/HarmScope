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

**Phase 0 complete.** Nothing downstream has run, so there are no results yet —
see the empty table at the bottom, which stays empty until Phase 9.

| Phase | State |
|---|---|
| 0 — Scaffold | ✅ schema, config, run registry, checks, normalization, CI |
| 1 — Ingestion & normalization | ✅ 16.5M complaints, 3.83M narratives, crosswalk covers both schema eras |
| 2 — Dedup & campaign detection **[GATE]** | ⬜ |
| 3 — Embedding & index | ⬜ |
| 4 — Clustering & novelty **[GATE]** | ⬜ |
| 5 — Signal detection | ⬜ |
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

| Metric | Value |
|---|---|
| Enforcement actions backtested | — |
| Actions with a matching signal before action date | — |
| Median lead time | — |
| Lead time vs. taxonomy baseline | — |
| False alerts per 1,000 company-months | — |
