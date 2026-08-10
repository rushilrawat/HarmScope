-- 2026-08-09. Provenance for human label verification.
--
-- Migration 007 originally created `label_verifications` without reviewer
-- provenance.  `CREATE TABLE IF NOT EXISTS` cannot retrofit a new column, so
-- a forward migration must rebuild the table.  Existing rows cannot establish
-- that a human performed the review; conservatively mark all of them `model`
-- so they remain auditable but cannot count toward the human-verification gate.
--
-- Rebuilding is intentionally repeat-safe: on a current table it copies the
-- explicit origin unchanged; on a legacy table the nullable added column is
-- backfilled to model before the constrained replacement is created.

ALTER TABLE label_verifications ADD COLUMN IF NOT EXISTS reviewer_origin VARCHAR;
UPDATE label_verifications SET reviewer_origin = 'model' WHERE reviewer_origin IS NULL;

DROP TABLE IF EXISTS label_verifications_v008;

CREATE TABLE label_verifications_v008 (
  cluster_id                  VARCHAR NOT NULL REFERENCES cluster_labels(cluster_id),
  reviewer_id                 VARCHAR NOT NULL,
  reviewer_origin             VARCHAR NOT NULL CHECK (reviewer_origin IN ('human', 'model')),
  worklist_version            VARCHAR NOT NULL,
  signals_run                 VARCHAR NOT NULL,
  is_fired                    BOOLEAN NOT NULL,
  mechanism_accuracy          VARCHAR NOT NULL CHECK (
    mechanism_accuracy IN ('agree', 'partial', 'disagree')
  ),
  taxonomy_distinctness_accuracy VARCHAR NOT NULL CHECK (
    taxonomy_distinctness_accuracy IN ('agree', 'disagree')
  ),
  template_accuracy           VARCHAR NOT NULL CHECK (
    template_accuracy IN ('agree', 'disagree')
  ),
  should_have_abstained       BOOLEAN NOT NULL,
  failure_category            VARCHAR NOT NULL CHECK (
    failure_category IN (
      'none', 'incoherent_cluster', 'overgeneralized', 'overspecific',
      'missed_submechanism', 'taxonomy_error', 'template_error',
      'unsupported_claim', 'other'
    )
  ),
  notes                       VARCHAR,
  reviewed_at                 TIMESTAMP NOT NULL,
  PRIMARY KEY (cluster_id, reviewer_id, worklist_version)
);

INSERT INTO label_verifications_v008
SELECT cluster_id, reviewer_id, reviewer_origin, worklist_version, signals_run, is_fired,
       mechanism_accuracy, taxonomy_distinctness_accuracy, template_accuracy,
       should_have_abstained, failure_category, notes, reviewed_at
FROM label_verifications;

DROP TABLE label_verifications;
ALTER TABLE label_verifications_v008 RENAME TO label_verifications;
