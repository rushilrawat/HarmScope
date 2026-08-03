# ENGINEERING_NOTES

Running log. Append per phase. Decisions, failures, and things that were harder than expected.
This file is read by future-you and by anyone evaluating whether the work was real.

---

## Standing rules

1. Update the relevant doc **before** starting the next phase.
2. Every stage asserts on its own output. A stage that "finished" without valid output is a
   defect, not a pass. (See §Known traps — this failure mode has bitten before.)
3. Fix seeds. Record `run_id`, git SHA, params, input/output row counts for every run.
4. Honest numbers only.

---

## Known traps (pre-registered — check for these actively)

### T1 — Silent success
A pipeline stage completes, writes 0 rows or all-null rows, and reports success. Downstream
stages read the empty table and also "succeed." The pipeline is green and produces nothing.

**Countermeasure:** every stage ends with explicit assertions on row count, null rate, and
distribution shape. Compare output row count against a config-declared expected range. Fail
loudly. Add a `--strict` mode that is on by default in CI.

### T2 — Campaign contamination
The top signal is a credit-repair template. Because it is huge, coherent, fast-growing, and
novel-scoring, it passes every automated check.

**Countermeasure:** read the top 10 signals by hand after every detection run. Every time. The
statistics cannot tell you this and no automated test will.

### T3 — Leakage via cluster definitions
Clustering once on the full corpus and filtering signals by date. Invisible in metrics, fatal to
the result.

**Countermeasure:** `clusters.as_of` + the anti-leakage test suite. Never bypass, even for a
"quick check."

### T4 — Threshold tuning on the backtest set
Adjusting `min_cluster_size` or the novelty threshold until the backtest looks good.

**Countermeasure:** freeze `config.py` before Phase 6, or hold out 1/3 of actions. Record which
was done.

### T5 — Ground-truth expansion after seeing results
Adding enforcement actions that the system happened to catch.

**Countermeasure:** git SHA of `enforcement_actions.csv` recorded in `EVALUATION.md` and
verified to predate the first detection run, by CI.

### T6 — Company merge errors
An over-aggressive fuzzy match merges a subsidiary into a parent, or two unrelated companies
with similar names. Silently corrupts every company-level statistic.

**Countermeasure:** manual verification of the top 300. Precision over recall on merges.

---

## Log

### Phase 0 — Scaffold
_Date:_ 2026-08-03

_What was built:_ repo skeleton, `db/schema.sql` (22 tables, verified to apply),
`src/config.py` (frozen + fingerprinted), `src/db.py` (run registry),
`src/checks.py` (stage assertions), `src/ids.py`, three pure normalization
modules (`pii`, `text`, `company`), `src/pipeline.py`, `src/ingestion/download.py`,
90 tests, Makefile, ruff + GitHub Actions CI.

_Decisions:_

- **Silent success is structural, not disciplinary.** `db.run()` is a context
  manager that raises `SilentSuccess` if the stage body exits without calling
  `finish(output_rows=...)`. You cannot accidentally record a stage that
  produced nothing as a success.
- **The registry writes on its own DuckDB cursor.** Verified empirically that
  `con.cursor()` has an independent transaction context: a stage that opens a
  transaction and fails rolls back its own output but not its failure record.
  Had the runs row shared the stage transaction, the provenance table would
  only ever have logged successes — T1 wearing a disguise.
- **`config_hash` on every run.** Trap T4's countermeasure was "freeze
  `config.py` before Phase 6". A sha256 over the frozen config, recorded per
  run, makes the freeze checkable rather than promised.
- **Card detection uses a Luhn checksum.** A 13-digit account number and a
  13-digit Visa are indistinguishable by shape; the first realistic test
  narrative caught the card rule swallowing an account number. Both are
  redacted either way — Luhn decides which counter moves, which is the point of
  keeping per-pattern counts at all.
- **Requirements are split three ways** (base / dev / ml). Phase 0–1 do not need
  torch, and CI should not spend six minutes building it.

_Surprises:_

- DuckDB enforces NOT NULL on primary-key columns, so `cluster_timeseries`'s
  specified NULL "all companies" marker row could never have been inserted.
  This would have surfaced at Phase 5 — after clustering, after hours of
  embedding. Found by reading the schema against the docs before writing any
  loader. Now a regression test.
- The CFPB bulk CSV is live and healthy: 1,408,530,128 bytes compressed,
  `Last-Modified: Sun, 02 Aug 2026`, i.e. updating daily as documented. The
  `.csv.gz` variant 404s; only `.csv.zip` exists. `PROJECT_SPEC.md` §5.4
  availability check: **passed 2026-08-03**.

_Gate:_ n/a (Phase 0 is not a gate). Acceptance `make init && make test`
verified from a clean clone.

### Phase 1 — Ingestion & normalization
_Actual corpus size:_
_Narrative coverage fraction:_
_Companies requiring manual merge:_
_Taxonomy crosswalk edge cases:_

### Phase 2 — Dedup & campaign detection [GATE]
_MinHash threshold chosen and why:_
_Precision / recall:_
_Campaign-flagged fraction by family:_
_What the flagged campaigns actually looked like on reading:_
_Gate passed:_ Y / N

### Phase 3 — Embedding & index
_Model, throughput, wall time:_
_Nearest-neighbour spot checks:_

### Phase 4 — Clustering & novelty [GATE]
_Params per family:_
_ARI disjoint halves:_
_Noise fraction:_
_Label-ablation AUC:_
_Could you name 15 random clusters?_
_Gate passed:_ Y / N

### Phase 5 — Signal detection
_Negative-control (shuffled labels) false-alert rate:_
_Expected under FDR α:_
_Top 20 signals, first impressions:_

### Phase 6 — Ground truth & backtest [GATE]
_Actions curated / usable / excluded:_
_enforcement_actions.csv frozen at SHA:_
_Wall time for one full-refit cutoff:_
_Anti-leakage tests passing:_
_Gate passed:_ Y / N

### Phase 7 — Baselines
_B1 implementation notes (must use identical statistical machinery):_

### Phase 8 — LLM layer
_Determinism test result:_
_Label agreement rate on 50 verified:_
_Observed LLM failure modes:_
_RAG Recall@10:_
_Cost incurred:_

### Phase 9 — Evaluation
_Headline result:_
_Did HarmScope beat B1:_
_Missed actions by failure category:_
_Top false alerts, what they actually were:_

### Phase 10 — Interface
_Shipped React or Streamlit:_
_What got cut:_

---

## Open questions

- Does clustering within `product_family` fragment cross-product harms too aggressively? The
  `related_clusters` linking may not be sufficient.
- Is annual-cutoff refit too coarse for actions filed early in a year? Consider quarterly
  cutoffs for a subset if wall time allows.
- Company public responses are optional and sparse — is the coverage high enough to display
  meaningfully in the evidence panel?

## Reversed decisions

_Record anything decided in the docs and later changed, with the reason. A doc that was never
wrong was never load-bearing._

### 2026-08-03 — Ground-truth window 2016–2024 → 2017–2024

`EVALUATION.md` §1.2 evaluates each action against the most recent annual cutoff
*strictly before* its `filed_date`, with cutoffs starting 2017-01-01. A 2016
action has no such cutoff, so the stated 2016–2024 window contained a year of
unevaluable rows. Narrowed the window rather than adding a 2016-01-01 cutoff,
because a model trained on 2015 alone would contribute near-certain misses for
reasons unrelated to method quality. Costs a year of candidates against the
≥ 20-usable requirement; leaves the gap to B1 — the actual contribution —
unchanged, since every system runs the identical harness.

### 2026-08-03 — Determinism test scoped to `signals`

`LLM_LAYER.md` §1 claimed deleting `src/llm/` leaves `signals` **and**
`backtest_results` byte-identical. But `EVALUATION.md` §1.3 shows the adjudicator
LLM labels, so LLM output reaches `backtest_results` through a human by design.
Scoped the claim to `signals` and stated adjudication as a deliberate
human-in-the-loop step. Overclaiming here would have undermined the one
architectural guarantee the project actually has.

### 2026-08-03 — No `--strict` flag

T1's countermeasure specified "a `--strict` mode that is on by default in CI".
`src/checks.py` always raises instead. A flag that is always on is a config for
a value that never changes; the behaviour is identical with less code. Add the
flag if a real need to run non-strict ever appears.

### 2026-08-03 — `run_id` is not a ULID

`ARCHITECTURE.md` §4 says ulid. `src/db.py` uses
`{epoch_ms:013d}-{8 hex chars}`, which has the two properties that are actually
load-bearing (lexicographic time-sortability, collision-freedom) with no
dependency. Swap in `python-ulid` if the canonical 26-character format is ever
needed by something external.

### 2026-08-03 — `dup_groups` and `campaigns` are run-scoped; `dup_pairs` split out

Documenting the per-cutoff refit split (above) exposed that two dedup tables
still had cutoff-independent keys: `dup_groups` was `PRIMARY KEY (complaint_id)`
while carrying a cutoff-dependent `is_representative`, and `campaigns.campaign_id`
was a bare VARCHAR while `campaigns.as_of` said it is regenerated per cutoff.
Eight refits, one slot each — the same collision `cluster_id` had.

Split along the line the refit table already draws: `dup_pairs` holds pairwise
similarity (computed once, date-independent), `dup_groups` holds connected
components and representative selection keyed `(run_id, complaint_id)`.
`campaign_id` now goes through `src/ids.py` like `cluster_id`.

Caught before Phase 2 wrote a single row. After that it would have been a
migration plus a full re-run.

### 2026-08-03 — `signals.company_id` uses the `'__ALL__'` sentinel too

`cluster_timeseries` got the sentinel; `signals` was left nullable. A join
between them on `company_id` — the natural Phase 5/6 query — would have returned
nothing for exactly the cluster-level rows. Not an error, just missing alerts.
Regression test: `test_cluster_level_signal_joins_its_timeseries_total`.

### 2026-08-03 — Embeddings and MinHash are not refit per cutoff

`EVALUATION.md` §1.1 said "rebuild the entire pipeline" per cutoff. The encoder
is a fixed pretrained checkpoint (§5 item 3) and MinHash similarity is pairwise,
so neither can leak. Both are now computed once and date-filtered; representative
selection, campaign detection, clustering, novelty, and signals are refit.
Roughly 8× off the most expensive stage with the leakage guarantee intact — and
the anti-leakage suite tests the guarantee directly, so the saving does not
depend on this reasoning being right.
