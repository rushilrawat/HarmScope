CREATE TABLE IF NOT EXISTS llm_usage (
  usage_id                   VARCHAR PRIMARY KEY,
  run_id                     VARCHAR,
  operation                  VARCHAR NOT NULL CHECK (operation IN ('label', 'answer')),
  cluster_id                 VARCHAR,
  question_hash              VARCHAR,
  model                      VARCHAR NOT NULL,
  prompt_version             VARCHAR NOT NULL,
  input_hash                 VARCHAR NOT NULL,
  cache_status               VARCHAR NOT NULL CHECK (cache_status IN ('hit', 'miss', 'bypass')),
  attempts                   INTEGER NOT NULL,
  input_tokens               BIGINT NOT NULL,
  output_tokens              BIGINT NOT NULL,
  cache_read_input_tokens    BIGINT NOT NULL,
  cache_creation_input_tokens BIGINT NOT NULL,
  latency_seconds            DOUBLE NOT NULL,
  estimated_cost_usd         DOUBLE NOT NULL,
  outcome                    VARCHAR NOT NULL CHECK (
    outcome IN ('ok', 'refused', 'failed', 'skipped')
  ),
  error_category             VARCHAR,
  created_at                 TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS label_verifications (
  cluster_id                  VARCHAR NOT NULL REFERENCES cluster_labels(cluster_id),
  reviewer_id                 VARCHAR NOT NULL,
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

CREATE TABLE IF NOT EXISTS rag_answers (
  question_hash        VARCHAR NOT NULL,
  evidence_hash        VARCHAR NOT NULL,
  cluster_id           VARCHAR NOT NULL REFERENCES clusters(cluster_id),
  company_id           VARCHAR NOT NULL,
  model                VARCHAR NOT NULL,
  prompt_version       VARCHAR NOT NULL,
  evidence_ids_json    JSON NOT NULL,
  answer_json          JSON NOT NULL,
  citation_valid       BOOLEAN NOT NULL,
  generated_at         TIMESTAMP NOT NULL,
  PRIMARY KEY (
    question_hash, evidence_hash, cluster_id, company_id, model, prompt_version
  )
);

CREATE TABLE IF NOT EXISTS rag_eval_results (
  eval_run_id              VARCHAR NOT NULL,
  question_id              VARCHAR NOT NULL,
  retrieval_method         VARCHAR NOT NULL CHECK (
    retrieval_method IN ('dense', 'bm25', 'fused')
  ),
  rank_first_relevant      INTEGER,
  relevant_retrieved_count INTEGER NOT NULL,
  recall_at_10             DOUBLE NOT NULL,
  reciprocal_rank          DOUBLE NOT NULL,
  latency_seconds          DOUBLE NOT NULL,
  citation_valid           BOOLEAN,
  citation_coverage        DOUBLE,
  abstention_correct       BOOLEAN,
  grounded_claims          INTEGER,
  reviewed_claims          INTEGER,
  created_at               TIMESTAMP NOT NULL,
  PRIMARY KEY (eval_run_id, question_id, retrieval_method)
);
