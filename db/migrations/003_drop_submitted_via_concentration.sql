-- 2026-08-04. Drop campaigns.submitted_via_concentration.
--
-- Migration 002 added it because METHODOLOGY §2.2 specifies six campaign
-- features and ARCHITECTURE §4 had only three. It called this one of "the two
-- that discriminate templates best". It discriminates nothing: every
-- narrative-bearing complaint in the corpus has submitted_via = 'Web' — all
-- 3,830,002, in all 12 product families (DATA.md §5). CFPB collects narrative
-- consent on the web form only, so conditioning on "has a narrative" conditions
-- on "arrived by web".
--
-- The column therefore held the constant 1.0, its family baseline was 1.0, and
-- the rule "exceed 1.5x the family baseline" asked for an HHI above 1.0. It
-- fired 0 times in 2,841 unflagged candidates on the 2026-08-04 run and could
-- never have fired. campaign_min_signals = 3 has always been 3-of-5.
--
-- Dropped rather than left in place: a stored constant reads as a measurement.
-- Safe to drop and recreate: campaigns are regenerated per run.

DROP TABLE IF EXISTS campaign_members;
DROP TABLE IF EXISTS campaigns;

CREATE TABLE campaigns (
  campaign_id           VARCHAR PRIMARY KEY,
  run_id                VARCHAR NOT NULL REFERENCES runs(run_id),
  n_complaints          BIGINT NOT NULL,
  n_groups              BIGINT NOT NULL,
  first_seen            DATE, last_seen DATE,
  top_company_id        VARCHAR,
  product_family        VARCHAR NOT NULL,
  burstiness            DOUBLE,
  state_concentration   DOUBLE,
  company_concentration DOUBLE,
  length_cv             DOUBLE,
  boilerplate_score     DOUBLE,
  n_signals             INTEGER NOT NULL,
  flagged               BOOLEAN NOT NULL,
  as_of                 DATE NOT NULL
);

CREATE TABLE campaign_members (
  complaint_id BIGINT NOT NULL,
  campaign_id  VARCHAR NOT NULL REFERENCES campaigns(campaign_id),
  PRIMARY KEY (complaint_id, campaign_id)
);

CREATE INDEX IF NOT EXISTS idx_campaign_run ON campaigns(run_id);
