# ARCHITECTURE

## 1. Design principles

1. **Statistics detect. LLMs describe.** The alerting path is fully deterministic and
   reproducible. Removing the LLM layer entirely must not change which signals fire.
2. **Every artifact is addressable.** A signal → its cluster → its member complaints →
   their raw text. One join each way, no reconstruction.
3. **Batch, idempotent, resumable.** Every stage reads from a table and writes to a table.
   Re-running a stage with the same `run_id` params is a no-op or a clean overwrite.
4. **Point-in-time correctness is a schema property, not a discipline.** Every derived row
   carries the `as_of` date of the data that produced it. Backtests filter on it.

## 2. Repository layout

```
harmscope/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example                 # ANTHROPIC_API_KEY, paths
│
├── data/
│   ├── raw/                     # immutable CFPB snapshot (gitignored)
│   ├── ground_truth/
│   │   ├── enforcement_actions.csv        # hand-curated, COMMITTED
│   │   ├── company_canonical_manual.csv   # top-300 hand review, COMMITTED
│   │   ├── taxonomy_crosswalk.csv         # old->new labels, COMMITTED
│   │   └── dedup_eval_pairs.csv           # 300 pairs, exact-Jaccard labels, COMMITTED
│   ├── interim/                 # gitignored
│   └── artifacts/               # embeddings .npy, faiss index (gitignored)
│
├── db/
│   ├── schema.sql
│   └── migrations/
│
├── src/
│   ├── config.py                # single source of params; no magic numbers elsewhere
│   ├── db.py                    # DuckDB connection, run registry
│   ├── ingestion/
│   │   ├── download.py
│   │   ├── load.py              # CSV -> complaints_raw
│   │   ├── build.py             # complaints, narratives, PII sweep
│   │   └── enforcement.py       # scrape + load ground truth
│   ├── normalization/
│   │   ├── company.py           # canonical company resolution
│   │   ├── taxonomy.py          # crosswalk application
│   │   ├── pii.py               # secondary redaction sweep
│   │   └── text.py              # normalize + hash for exact dedup
│   ├── dedup/
│   │   ├── minhash.py           # MinHash, LSH banding, star clustering
│   │   ├── detect.py            # phase driver: exact -> near-dup -> campaigns
│   │   ├── campaign.py          # the five campaign-scoring signals
│   │   └── evalset.py           # the gate's 300-pair scoring
│   ├── embed/
│   │   ├── encode.py            # sentence-transformers -> .npy memmap
│   │   └── index.py             # FAISS build / query
│   ├── cluster/
│   │   ├── reduce.py            # UMAP
│   │   ├── fit.py               # HDBSCAN on sample
│   │   ├── assign.py            # full-corpus assignment
│   │   └── novelty.py           # novelty vs existing taxonomy
│   ├── signals/
│   │   ├── timeseries.py        # cluster x period x company panels
│   │   ├── disproportionality.py# PRR / ROR / shrinkage
│   │   ├── changepoint.py       # EWMA + PELT
│   │   └── correct.py           # Benjamini-Hochberg FDR
│   ├── llm/
│   │   ├── label.py             # cluster -> harm mechanism label
│   │   ├── retrieve.py          # hybrid BM25 + dense evidence retrieval
│   │   └── cache.py             # content-hash keyed, on disk
│   ├── evaluation/
│   │   ├── backtest.py          # point-in-time harness
│   │   ├── baselines.py         # B0..B3
│   │   └── metrics.py
│   ├── api/
│   │   ├── main.py
│   │   └── routes/
│   └── pipeline.py              # orchestrator, phase runner
│
├── ui/                          # React + Vite + TS
├── notebooks/
├── tests/
└── docs/
```

## 3. Data flow

```
CFPB bulk CSV
     │
     ▼
[ingest] ──────────────► complaints_raw  (immutable, as-downloaded)
     │
     ▼
[normalize] ───────────► complaints      (typed, canonical company, crosswalked taxonomy)
     │                   narratives      (text + hashes + length stats + redaction flags)
     ▼
[dedup] ───────────────► dup_groups      (exact + near-dup + campaign membership)
     │                                    ── canonical representative per group
     ▼
[embed] ───────────────► embeddings.npy  (memmap) + faiss.index + embedding_map table
     │
     ▼
[cluster] ─────────────► clusters, cluster_members
     │                   cluster_novelty
     ▼
[label] ───────────────► cluster_labels  (LLM, cached — DESCRIPTIVE ONLY)
     │
     ▼
[signals] ─────────────► cluster_timeseries, signals   ◄── detection happens here
     │
     ▼
[evaluate] ────────────► backtest_results, baseline_results
     │
     ▼
[serve] ───────────────► FastAPI ──► React console
```

The `[label]` box sits **beside** the detection path, not inside it. Deleting it breaks the UI's
readability, not its correctness.

## 4. Database schema (DuckDB)

> `db/schema.sql` is authoritative and is verified to apply by
> `tests/test_schema.py`. The listing below is the reference; if the two ever
> disagree, the file is right and this section is a defect (README standing
> rule 1).

```sql
-- ============ provenance ============
CREATE TABLE runs (
  run_id        VARCHAR PRIMARY KEY,   -- lexicographically time-sortable, see src/db.py
  phase         VARCHAR NOT NULL,
  git_sha       VARCHAR NOT NULL,      -- '-dirty' suffix if the tree had uncommitted changes
  config_hash   VARCHAR NOT NULL,      -- sha256 of the frozen Config; proves thresholds were not retuned (trap T4)
  params_json   JSON NOT NULL,
  input_rows    BIGINT,
  output_rows   BIGINT,
  started_at    TIMESTAMP NOT NULL,
  finished_at   TIMESTAMP,
  status        VARCHAR NOT NULL CHECK (status IN ('running', 'ok', 'failed')),
  error         VARCHAR
);

-- ============ core ============
CREATE TABLE complaints_raw (           -- as-downloaded, never mutated
  complaint_id           BIGINT PRIMARY KEY,
  date_received          DATE,
  date_sent_to_company   DATE,
  product                VARCHAR,
  sub_product            VARCHAR,
  issue                  VARCHAR,
  sub_issue              VARCHAR,
  company_raw            VARCHAR,
  company_public_response VARCHAR,
  company_response       VARCHAR,
  timely_response        BOOLEAN,
  state                  VARCHAR,
  zip_code               VARCHAR,
  tags                   VARCHAR,
  submitted_via          VARCHAR,
  has_narrative          BOOLEAN
);

CREATE TABLE company_canonical (
  company_id     VARCHAR PRIMARY KEY,
  canonical_name VARCHAR NOT NULL,
  parent_id      VARCHAR,              -- self-FK for subsidiaries
  verified_by    VARCHAR,              -- 'manual' | 'fuzzy'
  n_complaints   BIGINT
);
CREATE TABLE company_alias (
  alias_raw   VARCHAR PRIMARY KEY,
  company_id  VARCHAR NOT NULL REFERENCES company_canonical(company_id),
  score       DOUBLE,
  method      VARCHAR
);

-- Keyed on (product, sub_product), not product alone: two of the 2017/2023
-- restructurings are splits rather than renames (Consumer Loan -> vehicle/payday;
-- Credit card or prepaid card -> credit card/prepaid), so the sub-product decides
-- the target. Migration 001. Issue/sub-issue are not crosswalked -- the novelty
-- score is measured against the raw labels, so remapping them would destroy the
-- thing being measured.
CREATE TABLE taxonomy_crosswalk (
  product_raw     VARCHAR NOT NULL,
  sub_product_raw VARCHAR NOT NULL,    -- '*' = applies to every sub-product
  product_std     VARCHAR NOT NULL,    -- current-era name for this family
  product_family  VARCHAR NOT NULL,    -- coarse grouping used for stratification
  era             VARCHAR,             -- which schema era the raw label belongs to
  PRIMARY KEY (product_raw, sub_product_raw)
);

CREATE TABLE complaints (               -- analysis-ready
  complaint_id   BIGINT PRIMARY KEY,
  date_received  DATE NOT NULL,
  period_month   DATE NOT NULL,         -- date_trunc('month', date_received)
  company_id     VARCHAR REFERENCES company_canonical(company_id),
  product_family VARCHAR NOT NULL,
  product_std    VARCHAR, issue_std VARCHAR, sub_issue_std VARCHAR,
  state          VARCHAR,
  tags           VARCHAR,
  company_response VARCHAR,
  has_narrative  BOOLEAN NOT NULL
);

CREATE TABLE narratives (
  complaint_id   BIGINT PRIMARY KEY REFERENCES complaints(complaint_id),
  text_redacted  VARCHAR NOT NULL,      -- post-PII-sweep; the ONLY text used downstream
  text_hash      VARCHAR NOT NULL,      -- sha256 of normalized text
  char_len       INTEGER, token_len INTEGER,
  redaction_count INTEGER,
  lang           VARCHAR
);
CREATE INDEX idx_narr_hash ON narratives(text_hash);

-- ============ dedup ============
CREATE TABLE dup_pairs (                -- computed once; date-independent
  complaint_id_a BIGINT NOT NULL,
  complaint_id_b BIGINT NOT NULL,       -- always > complaint_id_a
  similarity     DOUBLE NOT NULL,
  method         VARCHAR NOT NULL,      -- exact | minhash
  PRIMARY KEY (complaint_id_a, complaint_id_b)
);

CREATE TABLE dup_groups (               -- refit per cutoff
  run_id        VARCHAR NOT NULL REFERENCES runs(run_id),
  complaint_id  BIGINT NOT NULL,
  group_id      VARCHAR NOT NULL,
  is_representative BOOLEAN NOT NULL,
  group_size    INTEGER NOT NULL,
  as_of         DATE NOT NULL,
  PRIMARY KEY (run_id, complaint_id)
);

CREATE TABLE campaigns (                -- suspected mass filings; refit per cutoff
  campaign_id   VARCHAR PRIMARY KEY,    -- '{run_id}:campaign:{local_id}'
  run_id        VARCHAR NOT NULL REFERENCES runs(run_id),
  n_complaints  BIGINT NOT NULL,
  n_groups      BIGINT NOT NULL,
  first_seen    DATE, last_seen DATE,
  top_company_id VARCHAR,
  product_family VARCHAR NOT NULL,
  -- Five scoring signals (migration 003 dropped submitted_via_concentration:
  -- every narrative-bearing complaint is 'Web', so it was a constant).
  burstiness    DOUBLE,
  state_concentration DOUBLE,
  company_concentration DOUBLE,
  length_cv     DOUBLE,
  boilerplate_score DOUBLE,
  n_signals     INTEGER NOT NULL,       -- how many fired, so a flag is auditable
  flagged       BOOLEAN NOT NULL,
  as_of         DATE NOT NULL
);
CREATE TABLE campaign_members (         -- a complaint can be in one campaign per run
  complaint_id BIGINT NOT NULL,
  campaign_id  VARCHAR NOT NULL REFERENCES campaigns(campaign_id),
  PRIMARY KEY (complaint_id, campaign_id)
);

CREATE TABLE redaction_stats (          -- per-pattern PII sweep counts, per run
  run_id      VARCHAR NOT NULL REFERENCES runs(run_id),
  pattern     VARCHAR NOT NULL,
  n_hits      BIGINT NOT NULL,
  n_documents BIGINT NOT NULL,          -- documents touched, not total hits
  PRIMARY KEY (run_id, pattern)
);

-- ============ embeddings & clusters ============
CREATE TABLE embedding_map (            -- complaint_id <-> row index in .npy / faiss
  complaint_id BIGINT PRIMARY KEY,
  row_idx      BIGINT NOT NULL,
  model        VARCHAR NOT NULL,
  dim          INTEGER NOT NULL
);

CREATE TABLE clusters (
  cluster_id     VARCHAR PRIMARY KEY,   -- '{run_id}:{product_family}:{local_id}'
  run_id         VARCHAR NOT NULL REFERENCES runs(run_id),
  product_family VARCHAR NOT NULL,      -- clustering is stratified
  n_members      BIGINT NOT NULL,
  persistence    DOUBLE,                -- HDBSCAN cluster persistence
  coherence      DOUBLE,                -- mean intra-cluster cosine similarity
  centroid_idx   BIGINT,                -- medoid complaint row_idx
  as_of          DATE NOT NULL          -- point-in-time guard
);
CREATE TABLE cluster_members (
  cluster_id   VARCHAR NOT NULL,
  complaint_id BIGINT NOT NULL,
  membership_prob DOUBLE,
  is_exemplar  BOOLEAN,
  PRIMARY KEY (cluster_id, complaint_id)
);

CREATE TABLE cluster_novelty (
  cluster_id            VARCHAR PRIMARY KEY,
  dominant_label        VARCHAR,        -- most common (issue, sub_issue) tuple
  dominant_label_share  DOUBLE,
  label_entropy         DOUBLE,
  normalized_mutual_info DOUBLE,
  novelty_score         DOUBLE NOT NULL,
  is_novel              BOOLEAN NOT NULL
);

CREATE TABLE cluster_labels (           -- LLM output; descriptive only
  cluster_id        VARCHAR PRIMARY KEY,
  harm_mechanism    VARCHAR,
  actors            VARCHAR,
  preconditions     VARCHAR,
  consumer_impact   VARCHAR,
  distinct_from_taxonomy BOOLEAN,
  rationale         VARCHAR,
  confidence        VARCHAR,
  is_likely_template BOOLEAN,           -- the LLM's own campaign suspicion, never a detector input
  model             VARCHAR, prompt_version VARCHAR,
  input_hash        VARCHAR,            -- cache key
  human_verified    BOOLEAN DEFAULT FALSE,
  human_agrees      BOOLEAN,
  generated_at      TIMESTAMP
);

-- ============ signals ============
CREATE TABLE cluster_timeseries (
  cluster_id   VARCHAR NOT NULL,
  company_id   VARCHAR NOT NULL,        -- '__ALL__' = cluster total across companies
  period_month DATE NOT NULL,
  n            BIGINT NOT NULL,
  denom        BIGINT NOT NULL,         -- exposure: complaints in same family/period/company
  share        DOUBLE NOT NULL,
  as_of        DATE NOT NULL,           -- point-in-time guard
  PRIMARY KEY (cluster_id, company_id, period_month)
);

CREATE TABLE signals (
  signal_id     VARCHAR PRIMARY KEY,
  run_id        VARCHAR NOT NULL,
  cluster_id    VARCHAR NOT NULL,
  company_id    VARCHAR NOT NULL,       -- '__ALL__' = cluster-level, matching cluster_timeseries
  period_month  DATE NOT NULL,          -- period at which the signal fires
  method        VARCHAR NOT NULL,       -- prr | ebgm | ewma | pelt
  statistic     DOUBLE NOT NULL,
  ci_low        DOUBLE, ci_high DOUBLE,
  p_value       DOUBLE,
  q_value       DOUBLE,                 -- BH-corrected
  n_supporting  BIGINT NOT NULL,        -- raw complaint count
  n_supporting_groups BIGINT,           -- distinct dup_groups; 400 complaints in 3 groups is weak
  as_of         DATE NOT NULL           -- point-in-time guard
);

-- ============ ground truth & evaluation ============
CREATE TABLE enforcement_actions (
  action_id VARCHAR PRIMARY KEY,
  filed_date DATE NOT NULL,
  company_raw VARCHAR, company_id VARCHAR,
  product_family VARCHAR,
  harm_summary VARCHAR, harm_keywords VARCHAR,
  conduct_start DATE, source_url VARCHAR,
  usable BOOLEAN NOT NULL, exclusion_reason VARCHAR
);

CREATE TABLE backtest_links (           -- human-adjudicated cluster <-> action match
  action_id  VARCHAR NOT NULL,
  cluster_id VARCHAR NOT NULL,
  match_quality VARCHAR NOT NULL,       -- strong | partial | none
  adjudicated_by VARCHAR, notes VARCHAR, adjudicated_at TIMESTAMP,
  PRIMARY KEY (action_id, cluster_id)
);

CREATE TABLE backtest_results (
  run_id VARCHAR NOT NULL, system VARCHAR NOT NULL,   -- 'harmscope' | 'B0' | 'B1' ...
  action_id VARCHAR NOT NULL,
  detected BOOLEAN NOT NULL,
  first_signal_date DATE,
  lead_time_days INTEGER,
  PRIMARY KEY (run_id, system, action_id)
);
```

### Schema notes

- **Every per-cutoff artifact is run-scoped.** `cluster_id` and `campaign_id` are
  globally unique by construction (`src/ids.py`); `dup_groups` is keyed
  `(run_id, complaint_id)`. `dup_pairs` is the one dedup artifact that is not,
  because pairwise similarity is date-independent and computed once
  (`EVALUATION.md` §1.1.1). The rule: if a stage is refit per cutoff, its output
  key carries the run.
- **`signals.company_id` and `cluster_timeseries.company_id` use the same
  `'__ALL__'` sentinel.** If one used NULL and the other the sentinel, a join on
  `company_id` would silently drop exactly the cluster-level rows — a missing
  alert rather than a visible error.
- **`cluster_id` is globally unique by construction:** `{run_id}:{product_family}:{local_id}`
  (`src/ids.py`). The backtest refits clustering once per annual cutoff, so a
  locally unique id such as `mortgage-3` would collide across cutoffs in every
  table keyed on it — `cluster_members`, `cluster_novelty`, `cluster_labels`,
  `cluster_timeseries`, `signals`, `backtest_links`. Fixing it at the point of
  creation is cheaper than making six downstream tables carry a run column.
- **`cluster_timeseries.company_id` is NOT NULL, with `'__ALL__'` for the
  cross-company total.** DuckDB enforces NOT NULL on primary-key columns, so
  the NULL marker row this table originally specified could not be inserted at
  all — every cluster-total row would have failed at Phase 5, after clustering
  and several hours of embedding. `tests/test_schema.py` is the regression test.
- `as_of` on `clusters`, `campaigns`, `cluster_timeseries`, and `signals` is the leakage
  guard. The backtest harness filters `WHERE as_of <= :cutoff` and nothing else is
  permitted to reach evaluation. `campaigns` carries it because campaign features are
  time-windowed aggregates (`EVALUATION.md` §1.1.1).
- `cluster_timeseries.denom` is the exposure/offset term. Without it, growth in a cluster is
  indistinguishable from growth in overall complaint volume. This column is load-bearing.
- Embeddings live outside DuckDB (numpy memmap + FAISS). `embedding_map` is the bridge. Storing
  3M × 768 floats in-table wastes memory for no query benefit.

## 5. Scale strategy

| Stage | Constraint | Strategy |
|---|---|---|
| Ingest | ~10⁷ rows CSV | DuckDB `read_csv_auto` directly, no pandas |
| Dedup | O(n²) naive | MinHash + LSH banding; blocking by product_family |
| Embed | 3M docs | Batched, `fp16`, memmap-append, checkpointed every 50k |
| UMAP | 3M × 768 | Fit on 500k stratified sample, `transform` the rest |
| HDBSCAN | 3M points | **Fit on sample only** (see below) |
| Assign | 3M points | Nearest-exemplar via FAISS + `approximate_predict` |

**Sample-then-assign** is the critical decision. HDBSCAN on millions of points is infeasible on
a laptop. Fit on a stratified sample (by product_family × year), then assign the full corpus by
nearest cluster exemplar with a distance threshold — points beyond threshold become noise.
Document the sample size sensitivity: refit at 100k / 250k / 500k and report cluster stability
(ARI between runs). If clusters are unstable across sample sizes, the clustering is not real.

## 6. Configuration

All tunables live in `src/config.py` as a single frozen dataclass. No magic numbers in module
code. Every parameter that appears in a paper-style result must be reachable from this file and
serialized into `runs.params_json`.

Key params: embedding model name, UMAP `n_neighbors` / `n_components`, HDBSCAN
`min_cluster_size` / `min_samples`, MinHash threshold, novelty threshold, FDR alpha, minimum
cluster size for signaling, minimum supporting-complaint count for an alert.
