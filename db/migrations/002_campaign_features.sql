-- 2026-08-03. Add the three campaign features METHODOLOGY §2.2 specifies but
-- ARCHITECTURE §4 omitted: company_concentration, length_cv,
-- submitted_via_concentration. Without them the flag rests on three signals
-- instead of six, and the two that discriminate templates best (unnaturally
-- low length variance, single submission channel) are the missing ones.
--
-- Safe to drop and recreate: campaigns are regenerated per run.

DROP TABLE IF EXISTS campaign_members;
DROP TABLE IF EXISTS campaigns;

CREATE TABLE campaigns (
  campaign_id                 VARCHAR PRIMARY KEY,
  run_id                      VARCHAR NOT NULL REFERENCES runs(run_id),
  n_complaints                BIGINT NOT NULL,
  n_groups                    BIGINT NOT NULL,
  first_seen                  DATE, last_seen DATE,
  top_company_id              VARCHAR,
  product_family              VARCHAR NOT NULL,
  burstiness                  DOUBLE,
  state_concentration         DOUBLE,
  company_concentration       DOUBLE,
  submitted_via_concentration DOUBLE,
  length_cv                   DOUBLE,
  boilerplate_score           DOUBLE,
  n_signals                   INTEGER NOT NULL,
  flagged                     BOOLEAN NOT NULL,
  as_of                       DATE NOT NULL
);

CREATE TABLE campaign_members (
  complaint_id BIGINT NOT NULL,
  campaign_id  VARCHAR NOT NULL REFERENCES campaigns(campaign_id),
  PRIMARY KEY (complaint_id, campaign_id)
);

CREATE INDEX IF NOT EXISTS idx_campaign_run ON campaigns(run_id);
