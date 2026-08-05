-- HarmScope schema (DuckDB).
--
-- Two invariants are enforced structurally here rather than by discipline:
--
--   1. Point-in-time correctness. Every derived table that feeds evaluation
--      carries `as_of`, the date of the newest input row that produced it.
--      The backtest harness filters `WHERE as_of <= :cutoff` and nothing else
--      is permitted to reach evaluation. See docs/EVALUATION.md §5.
--
--   2. Globally unique cluster_id. Clusters are refit once per annual cutoff,
--      so a per-run local id (`mortgage-3`) would collide across cutoffs in
--      every table keyed on cluster_id. cluster_id is therefore
--      `{run_id}:{product_family}:{local_id}` — unique by construction.
--      See src/ids.py.

-- ============ provenance ============
CREATE TABLE IF NOT EXISTS runs (
  run_id        VARCHAR PRIMARY KEY,   -- lexicographically time-sortable, see src/db.py
  phase         VARCHAR NOT NULL,
  git_sha       VARCHAR NOT NULL,      -- '-dirty' suffix if the tree had uncommitted changes
  config_hash   VARCHAR NOT NULL,      -- sha256 of the frozen Config; proves thresholds were not retuned
  params_json   JSON NOT NULL,
  input_rows    BIGINT,
  output_rows   BIGINT,
  started_at    TIMESTAMP NOT NULL,
  finished_at   TIMESTAMP,
  status        VARCHAR NOT NULL CHECK (status IN ('running', 'ok', 'failed')),
  error         VARCHAR
);

-- ============ core ============
CREATE TABLE IF NOT EXISTS complaints_raw (   -- as-downloaded, never mutated
  complaint_id            BIGINT PRIMARY KEY,
  date_received           DATE,
  date_sent_to_company    DATE,
  product                 VARCHAR,
  sub_product             VARCHAR,
  issue                   VARCHAR,
  sub_issue               VARCHAR,
  company_raw             VARCHAR,
  company_public_response VARCHAR,
  company_response        VARCHAR,
  timely_response         BOOLEAN,
  state                   VARCHAR,
  zip_code                VARCHAR,       -- privacy-suppressed; never use as a feature (docs/DATA.md §2)
  tags                    VARCHAR,
  submitted_via           VARCHAR,
  has_narrative           BOOLEAN
);

CREATE TABLE IF NOT EXISTS company_canonical (
  company_id     VARCHAR PRIMARY KEY,
  canonical_name VARCHAR NOT NULL,
  parent_id      VARCHAR,               -- self-reference; not declared FK, see note below
  verified_by    VARCHAR NOT NULL CHECK (verified_by IN ('manual', 'fuzzy')),
  n_complaints   BIGINT
);

CREATE TABLE IF NOT EXISTS company_alias (
  alias_raw   VARCHAR PRIMARY KEY,
  company_id  VARCHAR NOT NULL REFERENCES company_canonical(company_id),
  score       DOUBLE,
  method      VARCHAR NOT NULL
);

-- `sub_product_raw` is required, not decorative: two of CFPB's taxonomy changes
-- are splits rather than renames (`Consumer Loan` -> vehicle/personal;
-- `Credit card or prepaid card` -> credit card/prepaid), so Product alone
-- cannot route those rows. '*' is the wildcard — DuckDB rejects NULL in a PK.
-- See docs/DATA.md §3.4.
CREATE TABLE IF NOT EXISTS taxonomy_crosswalk (
  product_raw     VARCHAR NOT NULL,
  sub_product_raw VARCHAR NOT NULL,     -- '*' = applies to every sub-product
  product_std     VARCHAR NOT NULL,     -- current-era name for this family
  product_family  VARCHAR NOT NULL,     -- coarse grouping used for stratification
  era             VARCHAR,              -- which schema era the raw label belongs to
  PRIMARY KEY (product_raw, sub_product_raw)
);

CREATE TABLE IF NOT EXISTS complaints (       -- analysis-ready
  complaint_id     BIGINT PRIMARY KEY,
  date_received    DATE NOT NULL,
  period_month     DATE NOT NULL,       -- date_trunc('month', date_received)
  company_id       VARCHAR REFERENCES company_canonical(company_id),
  product_family   VARCHAR NOT NULL,
  product_std      VARCHAR, issue_std VARCHAR, sub_issue_std VARCHAR,
  state            VARCHAR,
  tags             VARCHAR,
  company_response VARCHAR,
  has_narrative    BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS narratives (
  complaint_id    BIGINT PRIMARY KEY REFERENCES complaints(complaint_id),
  text_redacted   VARCHAR NOT NULL,     -- post-PII-sweep; the ONLY text used downstream
  text_hash       VARCHAR NOT NULL,     -- sha256 of normalized text (src/normalization/text.py)
  char_len        INTEGER, token_len INTEGER,
  redaction_count INTEGER NOT NULL,
  lang            VARCHAR
);

-- Per-pattern redaction counts. A single runaway regex is invisible in an
-- aggregate; docs/DATA.md §6 item 3 wants rate drift visible per run.
CREATE TABLE IF NOT EXISTS redaction_stats (
  run_id      VARCHAR NOT NULL REFERENCES runs(run_id),
  pattern     VARCHAR NOT NULL,
  n_hits      BIGINT NOT NULL,
  n_documents BIGINT NOT NULL,
  PRIMARY KEY (run_id, pattern)
);

-- ============ dedup ============
-- Split along the same line as docs/EVALUATION.md §1.1.1: pairwise similarity
-- is date-independent and computed once; grouping and representative selection
-- depend on which complaints exist before the cutoff and are refit per run.
CREATE TABLE IF NOT EXISTS dup_pairs (        -- computed once, reused by every cutoff
  complaint_id_a BIGINT NOT NULL,
  complaint_id_b BIGINT NOT NULL,             -- always > complaint_id_a
  similarity     DOUBLE NOT NULL,
  method         VARCHAR NOT NULL CHECK (method IN ('exact', 'minhash')),
  PRIMARY KEY (complaint_id_a, complaint_id_b)
);

-- Connected components over the date-filtered subgraph. Transitive closure is
-- date-dependent: if A~B and B~C but B is post-cutoff, then A and C are
-- separate groups at that cutoff. Hence one row per (run, complaint).
CREATE TABLE IF NOT EXISTS dup_groups (
  run_id            VARCHAR NOT NULL REFERENCES runs(run_id),
  complaint_id      BIGINT NOT NULL,
  group_id          VARCHAR NOT NULL,
  is_representative BOOLEAN NOT NULL,         -- earliest date_received in group, ties on lower id
  group_size        INTEGER NOT NULL,
  as_of             DATE NOT NULL,            -- point-in-time guard
  PRIMARY KEY (run_id, complaint_id)
);

CREATE TABLE IF NOT EXISTS campaigns (        -- suspected mass filings; refit per cutoff
  campaign_id           VARCHAR PRIMARY KEY,  -- '{run_id}:campaign:{local_id}', see src/ids.py
  run_id                VARCHAR NOT NULL REFERENCES runs(run_id),
  n_complaints          BIGINT NOT NULL,
  n_groups              BIGINT NOT NULL,
  first_seen            DATE, last_seen DATE,
  top_company_id        VARCHAR,
  product_family        VARCHAR NOT NULL,
  -- Five signals, not the six docs/METHODOLOGY.md §2.2 specifies: submission
  -- channel is a constant over narratives ('Web', all 3,830,002 — DATA.md §5),
  -- so it was dropped in migration 003. The flag is a count of how many fire,
  -- not a trained model: too few labels, and the features are readable enough
  -- that a human can audit a flag.
  burstiness            DOUBLE,   -- Fano factor of daily counts
  state_concentration   DOUBLE,   -- HHI over state
  company_concentration DOUBLE,   -- HHI over company_id
  length_cv             DOUBLE,   -- coefficient of variation of char_len
  boilerplate_score     DOUBLE,   -- share of members citing statute
  n_signals             INTEGER NOT NULL,
  flagged               BOOLEAN NOT NULL,
  as_of                 DATE NOT NULL     -- features are time-windowed; docs/EVALUATION.md §1.1.1
);

CREATE TABLE IF NOT EXISTS campaign_members (
  complaint_id BIGINT NOT NULL,
  campaign_id  VARCHAR NOT NULL REFERENCES campaigns(campaign_id),
  PRIMARY KEY (complaint_id, campaign_id)
);

-- ============ embeddings & clusters ============
-- complaint_id <-> row index in .npy / faiss. Keyed per model: METHODOLOGY §3
-- names two (bge-base default, MiniLM for dev iteration) and with complaint_id
-- alone as the key they could not coexist. Identical narratives share a row —
-- the memmap is keyed on text_hash, since an embedding is a pure function of
-- the text, and 3,830,002 narratives are 2,477,937 distinct texts.
CREATE TABLE IF NOT EXISTS embedding_map (
  complaint_id BIGINT NOT NULL,
  row_idx      BIGINT NOT NULL,
  model        VARCHAR NOT NULL,
  dim          INTEGER NOT NULL,
  PRIMARY KEY (complaint_id, model)
);
CREATE INDEX IF NOT EXISTS idx_embedding_row ON embedding_map(model, row_idx);

CREATE TABLE IF NOT EXISTS clusters (
  cluster_id     VARCHAR PRIMARY KEY,   -- '{run_id}:{product_family}:{local_id}'
  run_id         VARCHAR NOT NULL REFERENCES runs(run_id),
  product_family VARCHAR NOT NULL,      -- clustering is stratified
  n_members      BIGINT NOT NULL,
  persistence    DOUBLE,                -- HDBSCAN cluster persistence
  coherence      DOUBLE,                -- mean intra-cluster cosine similarity
  centroid_idx   BIGINT,                -- medoid complaint row_idx
  as_of          DATE NOT NULL          -- point-in-time guard
);

CREATE TABLE IF NOT EXISTS cluster_members (
  cluster_id      VARCHAR NOT NULL REFERENCES clusters(cluster_id),
  complaint_id    BIGINT NOT NULL,
  membership_prob DOUBLE,
  is_exemplar     BOOLEAN,
  PRIMARY KEY (cluster_id, complaint_id)
);

-- METHODOLOGY §4.2: clusters in different families whose centroids are close
-- are the same harm surfacing under two products. Ordered pair (a < b) so a
-- link is stored once. Added by migration 005 — specified since Phase 0, absent
-- from this file until Phase 4 needed it.
CREATE TABLE IF NOT EXISTS related_clusters (
  cluster_id_a VARCHAR NOT NULL REFERENCES clusters(cluster_id),
  cluster_id_b VARCHAR NOT NULL REFERENCES clusters(cluster_id),
  similarity   DOUBLE NOT NULL,   -- cosine between the two cluster centroids
  PRIMARY KEY (cluster_id_a, cluster_id_b)
);

CREATE TABLE IF NOT EXISTS cluster_novelty (
  cluster_id             VARCHAR PRIMARY KEY REFERENCES clusters(cluster_id),
  dominant_label         VARCHAR,       -- most common (issue_std, sub_issue_std) tuple
  dominant_label_share   DOUBLE,
  label_entropy          DOUBLE,
  normalized_mutual_info DOUBLE,
  novelty_score          DOUBLE NOT NULL,
  is_novel               BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_labels (   -- LLM output; descriptive only
  cluster_id             VARCHAR PRIMARY KEY REFERENCES clusters(cluster_id),
  harm_mechanism         VARCHAR,
  actors                 VARCHAR,
  preconditions          VARCHAR,
  consumer_impact        VARCHAR,
  distinct_from_taxonomy BOOLEAN,
  rationale              VARCHAR,
  confidence             VARCHAR,
  is_likely_template     BOOLEAN,
  model                  VARCHAR, prompt_version VARCHAR,
  input_hash             VARCHAR,       -- cache key
  human_verified         BOOLEAN DEFAULT FALSE,
  human_agrees           BOOLEAN,
  generated_at           TIMESTAMP
);

-- ============ signals ============
-- company_id uses the sentinel '__ALL__' for the cluster total across companies.
-- DuckDB enforces NOT NULL on primary-key columns, so a NULL marker row cannot
-- be inserted at all. See tests/test_schema.py for the regression test.
-- Keyed on the SIGNALS run, not on cluster_id alone. cluster_id is scoped to
-- the CLUSTER run, so two signals runs over one clustering shared a slot —
-- which is how the negative control silently overwrote the real panel. Same
-- collision dup_groups and campaigns were fixed for on 2026-08-03. Migration 006.
CREATE TABLE IF NOT EXISTS cluster_timeseries (
  run_id       VARCHAR NOT NULL REFERENCES runs(run_id),
  cluster_id   VARCHAR NOT NULL REFERENCES clusters(cluster_id),
  company_id   VARCHAR NOT NULL,        -- '__ALL__' = cluster total across companies
  period_month DATE NOT NULL,
  n            BIGINT NOT NULL,         -- distinct dup-groups, not complaints
  denom        BIGINT NOT NULL,         -- exposure: complaints in same family/period/company
  share        DOUBLE NOT NULL,
  as_of        DATE NOT NULL,           -- point-in-time guard
  PRIMARY KEY (run_id, cluster_id, company_id, period_month)
);

CREATE TABLE IF NOT EXISTS signals (
  signal_id            VARCHAR PRIMARY KEY,
  run_id               VARCHAR NOT NULL REFERENCES runs(run_id),
  cluster_id           VARCHAR NOT NULL REFERENCES clusters(cluster_id),
  -- Same sentinel as cluster_timeseries. If one table used NULL and the other
  -- '__ALL__', a join on company_id would silently drop exactly the
  -- cluster-level rows — a missing alert, not a visible error.
  company_id           VARCHAR NOT NULL,-- '__ALL__' = cluster-level, not company-specific
  period_month         DATE NOT NULL,   -- period at which the signal fires
  method               VARCHAR NOT NULL CHECK (method IN ('prr', 'ror', 'ebgm', 'ewma', 'pelt')),
  statistic            DOUBLE NOT NULL,
  ci_low               DOUBLE, ci_high DOUBLE,
  p_value              DOUBLE,
  q_value              DOUBLE,          -- BH-corrected
  n_supporting         BIGINT NOT NULL, -- raw complaint count
  n_supporting_groups  BIGINT NOT NULL, -- distinct dup-groups; the one that matters
  as_of                DATE NOT NULL    -- point-in-time guard
);

-- ============ ground truth & evaluation ============
CREATE TABLE IF NOT EXISTS enforcement_actions (
  action_id        VARCHAR PRIMARY KEY,
  filed_date       DATE NOT NULL,
  company_raw      VARCHAR, company_id VARCHAR,
  product_family   VARCHAR,
  harm_summary     VARCHAR,
  harm_keywords    VARCHAR,             -- curator notes only; MUST NOT reach the detection path
  conduct_start    DATE, source_url VARCHAR,
  usable           BOOLEAN NOT NULL, exclusion_reason VARCHAR
);

CREATE TABLE IF NOT EXISTS backtest_links (   -- human-adjudicated cluster <-> action match
  action_id      VARCHAR NOT NULL REFERENCES enforcement_actions(action_id),
  cluster_id     VARCHAR NOT NULL,
  match_quality  VARCHAR NOT NULL CHECK (match_quality IN ('strong', 'partial', 'none')),
  adjudicated_by VARCHAR, notes VARCHAR, adjudicated_at TIMESTAMP,
  PRIMARY KEY (action_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS backtest_results (
  run_id            VARCHAR NOT NULL,
  system            VARCHAR NOT NULL,   -- 'harmscope' | 'B0' | 'B1' | 'B2' | 'B3'
  action_id         VARCHAR NOT NULL REFERENCES enforcement_actions(action_id),
  detected          BOOLEAN NOT NULL,
  first_signal_date DATE,
  lead_time_days    INTEGER,
  PRIMARY KEY (run_id, system, action_id)
);

-- Per-system backtest outcome, one row per (run, system, action). EVALUATION §2
-- puts all five systems through the identical harness; this is where that
-- comparison lands. Referenced by ARCHITECTURE §3's data flow since Phase 0 and
-- defined in no schema until migration 006 — the same gap related_clusters had,
-- and invisible to the doc/schema drift test because the reference is in a
-- diagram rather than in the SQL block that test compares.
CREATE TABLE IF NOT EXISTS baseline_results (
  run_id         VARCHAR NOT NULL REFERENCES runs(run_id),
  system         VARCHAR NOT NULL CHECK (system IN ('B0', 'B1', 'B2', 'B3', 'harmscope')),
  action_id      VARCHAR NOT NULL REFERENCES enforcement_actions(action_id),
  cutoff         DATE NOT NULL,
  detected       BOOLEAN NOT NULL,
  first_signal   DATE,                  -- null when not detected
  lead_time_days INTEGER,               -- filed_date - first_signal; null when not detected
  match_quality  VARCHAR CHECK (match_quality IN ('strong', 'partial', 'none')),
  PRIMARY KEY (run_id, system, action_id)
);

CREATE INDEX IF NOT EXISTS idx_narr_hash        ON narratives(text_hash);
CREATE INDEX IF NOT EXISTS idx_complaints_month ON complaints(period_month);
CREATE INDEX IF NOT EXISTS idx_complaints_co    ON complaints(company_id);
CREATE INDEX IF NOT EXISTS idx_dup_group        ON dup_groups(run_id, group_id);
CREATE INDEX IF NOT EXISTS idx_campaign_run     ON campaigns(run_id);
CREATE INDEX IF NOT EXISTS idx_cluster_run      ON clusters(run_id);
CREATE INDEX IF NOT EXISTS idx_signals_asof     ON signals(as_of);

-- Note on company_canonical.parent_id: declared as a plain column, not a
-- self-referencing FOREIGN KEY. DuckDB rejects self-references in CREATE TABLE.
-- Integrity is checked in src/normalization/company.py instead.
