# ROADMAP

10 phases. Each has a hard acceptance criterion. **A phase is not complete until its criterion
is demonstrated and the relevant doc is updated.**

Phases 2, 4, and 6 are **gates** — failing them means stopping and fixing, not proceeding with a
known-broken foundation.

Effort estimates assume part-time work alongside coursework and an internship.

---

## Phase 0 — Scaffold
**~2 days**

- Repo structure per `ARCHITECTURE.md §2`, `pyproject.toml`, pinned `requirements.txt`.
- `src/config.py` frozen dataclass. `src/db.py` with the `runs` registry.
- `db/schema.sql` applied; empty DB builds from scratch with one command.
- Logging, seeds, `.env.example`, pre-commit (ruff + black).
- CI: lint + tests on push (GitHub Actions).

**Accept:** `make init && make test` on a clean clone produces an empty, schema-valid DuckDB.

---

## Phase 1 — Ingestion & normalization
**~4 days**

- Download bulk CFPB CSV; snapshot to `data/raw/` with a manifest (URL, date, sha256, row count).
- Load → `complaints_raw` via DuckDB `read_csv_auto`. No pandas.
- PII sweep → `narratives.text_redacted`.
- Company canonicalization: normalize → fuzzy block → **manual review of top 300 by volume**.
- Taxonomy crosswalk for the schema revision; `product_family` assignment.
- Record actual counts in `DATA.md §5` (replace the order-of-magnitude placeholders).

**Accept:**
- Row counts reconcile: raw == loaded, and are logged in `runs`.
- Narrative coverage fraction computed and written into `DATA.md`.
- Top-300 company mapping committed as CSV, spot-check of 20 entries passes.
- Monthly complaint volume plotted; the 2017 taxonomy discontinuity is visible pre-crosswalk
  and gone post-crosswalk.

---

## Phase 2 — Dedup & campaign detection **[GATE]**
**~1 week**

- Exact hash grouping; MinHash + LSH near-dup; star-clustered grouping (union-find chained —
  see `METHODOLOGY §2.2`).
- Campaign features and flagging per `METHODOLOGY §2.2`.
- 300 pairs → `data/ground_truth/dedup_eval_pairs.csv`, exact-Jaccard reference labels.
- Adjudicate the strata where the detector and the reference disagree, blind, per the
  pre-registered rule in `METHODOLOGY §2.4.1` → `dedup_eval_adjudicated.csv`.

**Accept (gate):**
- Precision ≥ 0.95, recall reported, on the labeled pair set.
- **Merge audit**: report what fraction of same-group pairs the eval set can actually sample.
  Star clustering merges through a seed, so most members of a group share no verified edge and
  the eval set — drawn from `dup_pairs` — cannot see them. A precision figure without this
  number is a figure about 8% of the merges. Read a sample of the seed-mediated merges by hand.
- Campaign-flagged fraction reported per product family, and credit reporting is clearly the
  highest.
- Manually read 20 flagged campaigns and 20 unflagged high-volume groups. Write findings into
  `ENGINEERING_NOTES.md`. If the flagged set is not obviously templated on inspection, **stop
  and fix**.

> Do not begin Phase 3 until this passes. Every downstream result is invalid otherwise.

---

## Phase 3 — Embedding & index
**~3 days**

- Encode representatives; memmap + checkpointing; `embedding_map`.
- FAISS index build; nearest-neighbour sanity queries.

**Accept:**
- Pick 10 narratives by hand; their 5 nearest neighbours are topically correct on inspection.
- Encode throughput and total wall time logged.
- Re-running encode is a no-op (idempotence test passes).

---

## Phase 4 — Clustering & novelty **[GATE]**
**~1.5 weeks**

- UMAP → HDBSCAN per product family, sample-then-assign.
- Stability testing across sample sizes and disjoint halves.
- Novelty scoring + label-ablation validation.

**Accept (gate):**
- ARI between disjoint halves reported. State the number regardless of value.
- Noise fraction reported per family.
- **Label-ablation AUC reported.** If the novelty score cannot recover 10 deliberately hidden
  `Issue` categories above AUC 0.7, the scoring is not working — fix before proceeding.
- Read 15 random clusters. Can you name each one? If not, `min_cluster_size` is wrong.

---

## Phase 5 — Signal detection
**~1 week**

- `cluster_timeseries` panels with correct exposure denominators.
- **Growth statistics count distinct `dup_groups`, not raw complaints.** Phase 2's campaign flag
  misses large templates that cite no statute — a 24,507-member group scored 2 of 5 signals and
  went unflagged (`ENGINEERING_NOTES.md` Phase 2). Counting groups makes that miss harmless:
  the template contributes 1 regardless of the flag. `signals.n_supporting_groups` exists for
  this; report it alongside `n_supporting` so a signal backed by 400 complaints in 3 groups is
  visibly weak.
- PRR / ROR + shrinkage; BH FDR within family.
- EWMA + PELT changepoint on share series with NB overdispersion handling.
- Alert construction with the joint criteria.

**Accept:**
- Negative-control test: shuffle cluster assignments randomly, re-run detection. False-alert
  rate should approximate the FDR α. **If a random assignment produces many alerts, the
  statistics are wrong.** This test is mandatory.
- Signals table populated with `as_of` correctly set on every row.
- Top 20 signals eyeballed; write first impressions in `ENGINEERING_NOTES.md`.

---

## Phase 6 — Ground truth & backtest harness **[GATE]**
**~1.5 weeks**

- Scrape + hand-curate ≥ 20 (target 30–40) enforcement actions, 2016–2024.
- **Commit and freeze before running any backtest.** Record git SHA in `EVALUATION.md`.
- Rolling annual cutoff refit machinery.
- Blind adjudication tooling (randomized, decoy-injected, signal-status hidden).
- Anti-leakage test suite (`EVALUATION §5`).

**Accept (gate):**
- All 7 anti-leakage tests pass in CI.
- Full refit at one cutoff runs end-to-end and completes in a documented wall time.
- Adjudication UI hides signal status — verified by a second person or by code review.

---

## Phase 7 — Baselines
**~1 week**

- B0 volume, B1 taxonomy, B2 TF-IDF+LDA, B3 BERTopic-default (no dedup).
- All run through the identical harness.

**Accept:** all four produce `baseline_results` rows under the same `run_id` scheme. (This
criterion said `backtest_results` until 2026-08-06. `db/schema.sql` defines both tables and
`src/evaluation/backtest.py` writes only `baseline_results`, which is the one carrying
`cutoff` and `match_quality`; `backtest_results` is vestigial and nothing writes it.) B1 is
implemented with the *same* statistical machinery as HarmScope — the only difference is the
unit being tracked. Anything else makes the comparison unfair in your favour.

---

## Phase 8 — LLM layer
**~1 week**

- Cluster labeling with caching, guardrails, structured output.
- Hybrid retrieval + grounded answers.
- Human verification of ≥ 50 labels.

**Accept:**
- **Determinism test passes:** removing `src/llm/` leaves `signals` and `backtest_results`
  byte-identical.
- Label agreement rate reported.
- RAG Recall@10 and groundedness reported on the 30-question set.

---

## Phase 9 — Evaluation & write-up
**~1 week**

- Run full backtest across all systems and cutoffs.
- Metrics, calibration, subgroup breakdown.
- **Failure analysis table** for every missed action and the top 20 false alerts.
- Sensitivity analysis on every threshold in `config.py`.
- README results table populated with real numbers, whatever they are.
- Limitations section written from `METHODOLOGY §7`.

**Accept:** a reader who is skeptical of the result can find, in the repo, the exact reason for
every claim and every failure. No number appears without its denominator.

---

## Phase 10 — Interface
**~1.5 weeks**

FastAPI + React console. **Deliberately last** — it is the most cuttable phase and the least
differentiating. If time runs short, ship Streamlit and spend the time on Phase 9 instead.

Views:
1. **Signal feed** — ranked alerts, filterable by family/company/period. Backtested vs.
   unvalidated-recent visually separated.
2. **Cluster detail** — LLM label, exemplar narratives, time series with changepoint marked,
   PRR forest plot across companies.
3. **Evidence panel** — retrieved complaints with IDs and dates; company public responses
   shown alongside; standing allegations disclaimer.
4. **Backtest view** — the enforcement actions, which were caught, lead-time distribution,
   and the misses with their failure categories. **Showing your own misses in the product is
   the most credible thing in it.**

**Accept:** every number on screen is one click from its supporting complaint IDs.

---

## Total: ~10–11 weeks part-time

## Cut order under time pressure

Cut from the bottom up. Never cut a gate.

1. Phase 10 → Streamlit
2. Phase 8 RAG → labeling only
3. Phase 7 → keep B0 and B1 only (B1 is non-negotiable)
4. Phase 9 sensitivity analysis → single-threshold reporting with the limitation stated

Everything in Phases 2, 4, 6 stays. Those are what make it a research system rather than a
dashboard.

---

## Anti-scope-creep list

Do not build these in v1, no matter how interesting they get:

- Shared-vendor inference across companies
- Fine-tuned domain embeddings
- Multi-agent anything
- Real-time streaming ingestion
- Non-CFPB data sources
- A public write API or user accounts
