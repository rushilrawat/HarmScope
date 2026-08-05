-- 2026-08-05. Run-scope cluster_timeseries, and add the missing baseline_results.
--
-- Found by the Phase 6 audit, and the first one is the more serious.
--
-- `cluster_timeseries` was keyed (cluster_id, company_id, period_month) with no
-- run_id. cluster_id is scoped to the CLUSTER run, but the panel is an artifact
-- of the SIGNALS run — and two signals runs over one cluster run therefore
-- shared a slot. The negative control is exactly that: same clusters, permuted
-- labels. So running it silently overwrote the real panel, and the table has
-- been holding permuted data (1,671,777 rows) rather than the real panel
-- (1,189,216) since the control last ran. Nothing errored.
--
-- This is the identical mistake the 2026-08-03 reversed decision fixed for
-- dup_groups and campaigns: "Eight refits, one slot each — the same collision
-- cluster_id had." It was missed here because the panel's key looked
-- run-scoped by way of cluster_id, and it is, but to the wrong run. Phase 6
-- would have hit it eight times over.
--
-- `baseline_results` is referenced by ARCHITECTURE §3's data flow and required
-- by EVALUATION §2 for the four baselines, and existed in no schema at all —
-- the same gap related_clusters had, and invisible to the doc/schema drift test
-- because that test compares the SQL block and this reference is in a diagram.

DROP TABLE IF EXISTS cluster_timeseries;

CREATE TABLE cluster_timeseries (
  run_id       VARCHAR NOT NULL REFERENCES runs(run_id),  -- the SIGNALS run
  cluster_id   VARCHAR NOT NULL REFERENCES clusters(cluster_id),
  company_id   VARCHAR NOT NULL,        -- '__ALL__' = cluster total across companies
  period_month DATE NOT NULL,
  n            BIGINT NOT NULL,         -- distinct dup-groups, not complaints
  denom        BIGINT NOT NULL,         -- exposure: same family/period/company
  share        DOUBLE NOT NULL,
  as_of        DATE NOT NULL,           -- point-in-time guard
  PRIMARY KEY (run_id, cluster_id, company_id, period_month)
);

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
