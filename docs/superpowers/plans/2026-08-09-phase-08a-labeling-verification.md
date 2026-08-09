# Phase 8A Labeling Reliability and Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox ( - [ ] ) syntax for tracking.

**Goal:** Make HarmScope's existing cluster-labeling job safe to resume, observable, testable without network access, and verifiable by a blinded human review.

**Architecture:** A transport-only client returns typed payload, usage, latency, and retry metadata. The label runner owns deterministic selection, validated atomic caching, one-cluster database transactions, and usage persistence. A separate verification module exports a signal-blinded worklist, validates reviewer decisions, and reports Wilson intervals.

**Tech Stack:** Python 3.11+, DuckDB 1.5.5, Anthropic Python SDK 0.75.0, JSON structured outputs, NumPy, pytest, Ruff

## Global Constraints

- The LLM is descriptive only and must never change signal detection, cluster membership, statistics, rankings, or backtest outcomes.
- Detection packages must not import src/llm or anthropic; Phase 8 imports in src/pipeline.py remain function-local.
- Only narratives.text_redacted may reach the model or a review worklist.
- The configured default model remains claude-opus-5 until an explicit evidence-backed or operator-instructed change.
- Prompt version and model remain part of every cache key.
- Live API failures for authentication, billing, permission, or invalid requests are terminal for the batch.
- A refusal or schema-invalid response is recorded for that cluster without discarding prior successful clusters.
- Estimated costs are labeled estimates and use configuration-owned per-million-token prices.
- Human verification requires at least 50 reviewed labels; an LLM review does not satisfy that gate.

## File map

- Create db/migrations/007_phase_08_llm_layer.sql — upgrades an existing database with the four Phase 8 downstream tables.
- Modify db/schema.sql — defines the same four tables for fresh databases.
- Modify docs/ARCHITECTURE.md — keeps the schema contract synchronized.
- Modify src/config.py — adds retry, pricing, and verification parameters.
- Create src/llm/client.py — Anthropic transport, retry classification, typed usage, and cost calculation.
- Modify src/llm/label.py — validates labels and performs atomic, quarantine-on-corruption caching.
- Modify src/llm/run.py — dependency-injected resumable label runner and usage persistence.
- Create src/llm/verify.py — blinded worklist export/import and Wilson-interval reporting.
- Modify src/pipeline.py — label summary plus label-verify export, record, and report commands.
- Create tests/test_llm_client.py — isolated transport and accounting tests.
- Create tests/test_llm_run.py — fake-client label-job integration tests.
- Create tests/test_llm_verify.py — sampling, validation, persistence, and reporting tests.
- Modify tests/test_llm.py — validated atomic-cache and import-boundary regressions.
- Modify tests/test_schema.py — fresh-schema and migration-upgrade coverage.
- Modify tests/test_config.py — asserts every measured Phase 8 parameter is fingerprinted.

---

### Task 1: Add the Phase 8 data contract and configuration

**Files:**
- Create: db/migrations/007_phase_08_llm_layer.sql
- Modify: db/schema.sql after cluster_labels
- Modify: docs/ARCHITECTURE.md in the schema SQL block
- Modify: src/config.py in LLMConfig
- Modify: tests/test_schema.py
- Modify: tests/test_config.py

**Interfaces:**
- Consumes: src.db.apply_schema(con) and CONFIG.to_dict()
- Produces: tables llm_usage, label_verifications, rag_answers, rag_eval_results and the new LLMConfig fields used by every later Phase 8 plan

- [ ] **Step 1: Write failing schema and configuration tests**

~~~python
def test_phase8_tables_exist_on_a_fresh_database(con):
    required = {
        "llm_usage", "label_verifications", "rag_answers", "rag_eval_results"
    }
    assert required <= set(db.table_names(con))


def test_phase8_migration_upgrades_a_pre_phase8_database(con):
    for table in ("rag_eval_results", "rag_answers", "label_verifications", "llm_usage"):
        con.execute("DROP TABLE IF EXISTS " + table)
    migration = Path("db/migrations/007_phase_08_llm_layer.sql").read_text()
    con.execute(migration)
    con.execute(migration)
    assert {
        "llm_usage", "label_verifications", "rag_answers", "rag_eval_results"
    } <= set(db.table_names(con))


def test_llm_operational_parameters_are_fingerprinted():
    payload = CONFIG.to_dict()["llm"]
    assert payload["max_retries"] == 3
    assert payload["retry_base_seconds"] == 1.0
    assert payload["input_usd_per_million"] > 0
    assert payload["output_usd_per_million"] > 0
~~~

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_schema.py::test_phase8_tables_exist_on_a_fresh_database tests/test_schema.py::test_phase8_migration_upgrades_a_pre_phase8_database tests/test_config.py::test_llm_operational_parameters_are_fingerprinted -v

Expected: FAIL because the tables and configuration fields do not exist.

- [ ] **Step 3: Add the four tables to the schema and migration**

Use identical CREATE TABLE IF NOT EXISTS statements in db/schema.sql and migration 007:

~~~sql
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
~~~

- [ ] **Step 4: Add exact operational fields to LLMConfig**

~~~python
max_retries: int = 3
retry_base_seconds: float = 1.0
retry_max_seconds: float = 30.0
input_usd_per_million: float = 5.0
output_usd_per_million: float = 25.0
cache_write_usd_per_million: float = 6.25
cache_read_usd_per_million: float = 0.50
verification_seed: int = 20260809
~~~

The prices are the current standard-global Opus-class list prices recorded in the approved design review. Re-check official pricing immediately before a paid run; a price change requires a config change and therefore a new recorded fingerprint.

- [ ] **Step 5: Mirror the table definitions in docs/ARCHITECTURE.md**

Add all columns and constraints, not abbreviated prose. Keep the existing doc/schema parity test bidirectional.

- [ ] **Step 6: Run schema, configuration, and doc-parity tests**

Run: .venv/bin/python -m pytest tests/test_schema.py tests/test_config.py -v

Expected: PASS.

- [ ] **Step 7: Commit the data contract**

~~~bash
git add db/schema.sql db/migrations/007_phase_08_llm_layer.sql docs/ARCHITECTURE.md src/config.py tests/test_schema.py tests/test_config.py
git commit -m "Add Phase 8 LLM data contract"
~~~

---

### Task 2: Build the reliable Anthropic client

**Files:**
- Create: src/llm/client.py
- Create: tests/test_llm_client.py

**Interfaces:**
- Consumes: LLMConfig pricing and retry fields
- Produces:
  - TokenUsage(input_tokens: int, output_tokens: int, cache_read_input_tokens: int, cache_creation_input_tokens: int)
  - ModelCallResult(payload: dict, model: str, stop_reason: str, usage: TokenUsage, attempts: int, latency_seconds: float, estimated_cost_usd: float)
  - ModelCallError(category: str, attempts: int, retryable: bool)
  - AnthropicModelClient.call_json(model: str, system: str, prompt: str, schema: dict, max_tokens: int) -> ModelCallResult
  - AnthropicModelClient.preflight(model: str) -> None

- [ ] **Step 1: Write failing retry, terminal-error, usage, and preflight tests**

~~~python
def test_retries_rate_limit_then_returns_usage():
    transport = SequenceTransport([
        FakeRateLimit(),
        response(payload={"ok": True}, input_tokens=100, output_tokens=20),
    ])
    sleeps = []
    client = AnthropicModelClient(
        transport, pricing(), max_retries=3, sleeper=sleeps.append,
        clock=SequenceClock([0.0, 0.4]), jitter=lambda _lo, _hi: 0.0,
    )
    got = client.call_json(
        model="m", system="s", prompt="p", schema={"type": "object"}, max_tokens=50
    )
    assert got.attempts == 2
    assert got.usage == TokenUsage(100, 20, 0, 0)
    assert sleeps == [1.0]


def test_billing_error_is_terminal_without_retry():
    client = AnthropicModelClient(
        SequenceTransport([FakeBillingError()]), pricing(), max_retries=3,
        sleeper=lambda _: None,
    )
    with pytest.raises(ModelCallError) as caught:
        client.call_json(
            model="m", system="s", prompt="p",
            schema={"type": "object"}, max_tokens=50,
        )
    assert caught.value.category == "billing"
    assert caught.value.attempts == 1
    assert caught.value.retryable is False


def test_cost_uses_all_anthropic_token_categories():
    usage = TokenUsage(1_000_000, 1_000_000, 1_000_000, 1_000_000)
    assert estimate_cost(usage, pricing()) == pytest.approx(36.75)


def test_preflight_retrieves_the_configured_model():
    transport = RecordingTransport()
    AnthropicModelClient(transport, pricing()).preflight("claude-opus-5")
    assert transport.retrieved == ["claude-opus-5"]
~~~

- [ ] **Step 2: Run the client tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_client.py -v

Expected: FAIL because src.llm.client does not exist.

- [ ] **Step 3: Define immutable result and pricing types**

~~~python
@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class ModelPricing:
    input_usd_per_million: float
    output_usd_per_million: float
    cache_write_usd_per_million: float
    cache_read_usd_per_million: float


@dataclass(frozen=True)
class ModelCallResult:
    payload: dict[str, Any]
    model: str
    stop_reason: str
    usage: TokenUsage
    attempts: int
    latency_seconds: float
    estimated_cost_usd: float


class ModelCallError(RuntimeError):
    def __init__(self, category: str, attempts: int, retryable: bool):
        super().__init__("%s after %d attempt(s)" % (category, attempts))
        self.category = category
        self.attempts = attempts
        self.retryable = retryable
~~~

- [ ] **Step 4: Implement cost calculation and explicit exception classification**

~~~python
def estimate_cost(usage: TokenUsage, price: ModelPricing) -> float:
    return (
        usage.input_tokens * price.input_usd_per_million
        + usage.output_tokens * price.output_usd_per_million
        + usage.cache_creation_input_tokens * price.cache_write_usd_per_million
        + usage.cache_read_input_tokens * price.cache_read_usd_per_million
    ) / 1_000_000


def classify_error(exc: Exception) -> tuple[str, bool]:
    import anthropic

    message = str(exc).lower()
    if "credit balance" in message or "billing" in message:
        return "billing", False
    if isinstance(exc, anthropic.RateLimitError):
        return "rate_limit", True
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.InternalServerError)):
        return "transient", True
    if isinstance(exc, anthropic.AuthenticationError):
        return "authentication", False
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "permission", False
    if isinstance(exc, anthropic.BadRequestError):
        return "invalid_request", False
    return "unexpected", False
~~~

- [ ] **Step 5: Implement preflight and bounded retry without internal SDK retries**

Construct the default SDK with max_retries=0 so the recorded attempts equal real requests. On attempt n, compute base = min(retry_base_seconds * 2 ** (n - 1), retry_max_seconds), then sleep base + jitter(0, min(base * 0.25, 1.0)). Inject sleeper, clock, and jitter; default jitter uses random.uniform and tests inject zero. Parse usage from response.usage, including cache_read_input_tokens and cache_creation_input_tokens with zero defaults. A stop_reason of refusal returns a ModelCallResult with payload {"refused": True, "stop_reason": "refusal"}.

~~~python
def preflight(self, model: str) -> None:
    try:
        self.transport.models.retrieve(model_id=model)
    except Exception as exc:
        category, retryable = classify_error(exc)
        raise ModelCallError(category, 1, retryable) from exc
~~~

- [ ] **Step 6: Run tests and lint**

Run: .venv/bin/python -m pytest tests/test_llm_client.py -v

Expected: PASS.

Run: .venv/bin/ruff check src/llm/client.py tests/test_llm_client.py

Expected: PASS.

- [ ] **Step 7: Commit the client**

~~~bash
git add src/llm/client.py tests/test_llm_client.py
git commit -m "Add reliable LLM client"
~~~

---

### Task 3: Harden caching and make labeling resumable

**Files:**
- Modify: src/llm/label.py
- Modify: src/llm/run.py
- Create: tests/test_llm_run.py
- Modify: tests/test_llm.py

**Interfaces:**
- Consumes: AnthropicModelClient.call_json and ModelCallResult from Task 2
- Produces:
  - validate_label(payload: dict) -> dict
  - cached(cache_dir: Path, key: str) -> dict | None
  - write_cache(cache_dir: Path, key: str, label: dict) -> None
  - run(..., client=None, vectors=None, cache_dir=None) -> LabelRunStats
  - record_usage(con, usage: UsageRecord) -> None

- [ ] **Step 1: Write failing cache-corruption and label-validation tests**

~~~python
def test_cache_write_is_atomic_and_validated(tmp_path):
    key = label_mod.input_hash("v1", "m", [1])
    label_mod.write_cache(tmp_path, key, complete_label())
    assert label_mod.cached(tmp_path, key) == complete_label()
    assert list(tmp_path.glob("*.tmp")) == []


def test_corrupt_cache_is_quarantined(tmp_path):
    key = "broken"
    path = tmp_path / "broken.json"
    path.write_text("{")
    assert label_mod.cached(tmp_path, key) is None
    assert not path.exists()
    assert len(list(tmp_path.glob("broken.json.corrupt-*"))) == 1


def test_label_validation_rejects_missing_or_extra_fields():
    payload = complete_label()
    payload.pop("confidence")
    with pytest.raises(LabelSchemaError, match="confidence"):
        label_mod.validate_label(payload)
    payload = complete_label() | {"extra": True}
    with pytest.raises(LabelSchemaError, match="extra"):
        label_mod.validate_label(payload)
~~~

- [ ] **Step 2: Write failing fake-client integration tests**

Seed two clusters, members, narratives, embedding_map rows, one firing signal, and a small normalized vector array. Inject a fake client that returns ModelCallResult values.

~~~python
def test_label_job_resumes_from_cache_without_second_call(label_fixture, tmp_path):
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([result(complete_label())])
    first = llm_run.run(
        con, cluster_run, signals_run, control_n=0, limit=1,
        embed_model="m", client=client, vectors=vectors, cache_dir=tmp_path,
    )
    second = llm_run.run(
        con, cluster_run, signals_run, control_n=0, limit=1,
        embed_model="m", client=client, vectors=vectors, cache_dir=tmp_path,
    )
    assert first.labelled == 1
    assert second.cached == 1
    assert client.calls == 1
    outcomes = con.execute(
        "SELECT cache_status, outcome FROM llm_usage ORDER BY created_at"
    ).fetchall()
    assert outcomes == [("miss", "ok"), ("hit", "ok")]


def test_terminal_billing_failure_stops_remaining_population(label_fixture, tmp_path):
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([ModelCallError("billing", 1, False)])
    with pytest.raises(ModelCallError, match="billing"):
        llm_run.run(
            con, cluster_run, signals_run, control_n=1, limit=None,
            embed_model="m", client=client, vectors=vectors, cache_dir=tmp_path,
        )
    assert client.calls == 1
    assert con.execute(
        "SELECT count(*) FROM llm_usage WHERE error_category = 'billing'"
    ).fetchone()[0] == 1
~~~

- [ ] **Step 3: Run focused tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm.py tests/test_llm_run.py -v

Expected: FAIL on missing validation, quarantine, injected dependencies, and usage records.

- [ ] **Step 4: Implement strict local label validation**

Require exactly the LABEL_SCHEMA property set; validate strings, actors as list[str], both booleans as bool, and confidence in {"high", "medium", "low"}. Return a shallow copy so callers never mutate cached state.

~~~python
class LabelSchemaError(ValueError):
    pass


def validate_label(payload: dict) -> dict:
    expected = set(LABEL_SCHEMA["properties"])
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise LabelSchemaError("missing=%s extra=%s" % (missing, extra))
    if payload["confidence"] not in {"high", "medium", "low"}:
        raise LabelSchemaError("invalid confidence")
    if not all(isinstance(item, str) for item in payload["actors"]):
        raise LabelSchemaError("actors must be strings")
    return dict(payload)
~~~

- [ ] **Step 5: Implement atomic cache writes and corruption quarantine**

Serialize validated JSON to a NamedTemporaryFile in cache_dir, flush and os.fsync it, then os.replace(temp_path, final_path). On JSON or schema failure while reading, rename the file to key.json.corrupt-YYYYMMDDTHHMMSS and return None. Refusal payloads use a separate two-key schema and remain cacheable.

- [ ] **Step 6: Refactor the runner around typed statistics and injected dependencies**

~~~python
@dataclass
class LabelRunStats:
    labelled: int = 0
    cached: int = 0
    refused: int = 0
    failed: int = 0
    skipped: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_seconds: float = 0.0
~~~

The default client is created only inside run(). Preflight once before the first cache miss, not before cache-only replays. For every target:

1. Select narratives and compute input_hash.
2. Load and validate cache.
3. On a miss, call the client and atomically cache a successful payload or refusal.
4. BEGIN; write cluster_labels when appropriate; insert exactly one llm_usage row; COMMIT.
5. ROLLBACK the cluster transaction on persistence errors.
6. Re-raise authentication, billing, permission, and invalid_request failures after recording them.
7. Count schema failures and refusals, then continue.

Generate usage_id with src.db.new_run_id(). Pass the surrounding label run_id from phase_label into llm_run.run so llm_usage.run_id is populated.

- [ ] **Step 7: Run labeling tests and the existing boundary suite**

Run: .venv/bin/python -m pytest tests/test_llm.py tests/test_llm_client.py tests/test_llm_run.py -v

Expected: PASS.

Run: .venv/bin/python -m pytest tests/test_signals.py tests/test_leakage.py -v

Expected: PASS, proving the detection path still cannot reach Phase 8.

- [ ] **Step 8: Commit resumable labeling**

~~~bash
git add src/llm/label.py src/llm/run.py tests/test_llm.py tests/test_llm_run.py
git commit -m "Make LLM labeling resumable and observable"
~~~

---

### Task 4: Add blinded human label verification

**Files:**
- Create: src/llm/verify.py
- Create: tests/test_llm_verify.py

**Interfaces:**
- Consumes: cluster_labels, clusters, cluster_novelty, cluster_members, narratives, and a signals_run
- Produces:
  - export_worklist(con, signals_run: str, n: int, seed: int, path: Path) -> Path
  - parse_worklist(path: Path, reviewer_id: str) -> list[Verification]
  - record(con, rows: list[Verification], signals_run: str, worklist_version: str) -> int
  - report(con, worklist_version: str | None = None) -> VerificationReport
  - wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]

- [ ] **Step 1: Write failing sampling and blinding tests**

~~~python
def test_export_is_seeded_stratified_and_signal_blinded(verification_fixture, tmp_path):
    con, signals_run = verification_fixture
    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 50
    assert "did_fire" not in rows[0]
    assert "q_value" not in rows[0]
    assert "statistic" not in rows[0]
    assert "confidence" not in rows[0]
    assert rows == list(csv.DictReader(
        verify.export_worklist(
            con, signals_run, 50, 7, tmp_path / "again.csv"
        ).open()
    ))
    assert all(row["narrative_10"] for row in rows)
~~~

Sampling uses a deterministic greedy coverage pass over fired/control, product_family, confidence, template suspicion, and taxonomy-distinctness. At each step choose the remaining row that covers the most currently underrepresented stratum values; break ties with sha256(seed + cluster_id). This avoids pretending that a full Cartesian stratification is possible in 50 rows.

- [ ] **Step 2: Write failing parse, persistence, and Wilson-report tests**

~~~python
def test_parse_rejects_unknown_decisions(tmp_path):
    path = filled_worklist(tmp_path, mechanism_accuracy="mostly")
    with pytest.raises(ValueError, match="agree.*partial.*disagree"):
        verify.parse_worklist(path, "reviewer-1")


def test_report_has_denominators_and_wilson_intervals(verification_fixture):
    con, signals_run = verification_fixture
    verify.record(con, reviewed_rows(agree=39, total=50), signals_run, "wl-v1")
    report = verify.report(con, "wl-v1")
    assert report.mechanism_agree == 39
    assert report.mechanism_total == 50
    assert report.mechanism_rate == pytest.approx(0.78)
    assert report.mechanism_ci_low < 0.78 < report.mechanism_ci_high
    assert report.by_fired_status.keys() == {"fired", "control"}
~~~

- [ ] **Step 3: Run verification tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_verify.py -v

Expected: FAIL because src.llm.verify does not exist.

- [ ] **Step 4: Implement the review CSV contract**

Export columns:

~~~python
HEADER = [
    "cluster_id", "product_family", "harm_mechanism", "actors",
    "preconditions", "consumer_impact", "dominant_taxonomy",
    "model_distinct_from_taxonomy", "model_is_likely_template",
    "narrative_1", "narrative_2", "narrative_3", "narrative_4",
    "narrative_5", "narrative_6", "narrative_7", "narrative_8",
    "narrative_9", "narrative_10", "mechanism_accuracy",
    "taxonomy_distinctness_accuracy", "template_accuracy",
    "should_have_abstained", "failure_category", "notes",
]
~~~

Normalize whitespace and cap each narrative at 1,200 characters. Do not export model confidence, fired/control status, signal identifiers, rank, statistic, p-value, or q-value.

- [ ] **Step 5: Implement strict decisions and report calculations**

mechanism_accuracy accepts agree, partial, disagree. The headline mechanism agreement is agree / reviewed; also report (agree + partial) / reviewed as lenient agreement. Taxonomy and template agreement are agree / reviewed. For every rate return numerator, denominator, point estimate, and 95% Wilson interval. Group by fired/control and confidence by joining source tables after review rather than exposing those strata in the worklist.

failure_category must be none when all three accuracy fields agree and should_have_abstained is false. Otherwise require one of the eight approved failure categories.

- [ ] **Step 6: Run tests and lint**

Run: .venv/bin/python -m pytest tests/test_llm_verify.py -v

Expected: PASS.

Run: .venv/bin/ruff check src/llm/verify.py tests/test_llm_verify.py

Expected: PASS.

- [ ] **Step 7: Commit human verification**

~~~bash
git add src/llm/verify.py tests/test_llm_verify.py
git commit -m "Add blinded LLM label verification"
~~~

---

### Task 5: Wire the labeling and verification CLI

**Files:**
- Modify: src/pipeline.py
- Create: tests/test_llm_cli.py

**Interfaces:**
- Consumes: LabelRunStats and src.llm.verify public functions
- Produces:
  - python -m src.pipeline run --phase label --limit 20
  - python -m src.pipeline label-verify export --n 50 --output PATH
  - python -m src.pipeline label-verify record --input PATH --reviewer ID
  - python -m src.pipeline label-verify report

- [ ] **Step 1: Write failing parser and summary tests**

~~~python
def test_label_verify_subcommands_parse():
    parser = pipeline.build_parser()
    args = parser.parse_args([
        "label-verify", "export", "--n", "50", "--output", "review.csv"
    ])
    assert args.verify_action == "export"
    assert args.n == 50


def test_label_summary_reports_failures_tokens_latency_and_cost(monkeypatch, capsys):
    monkeypatch.setattr(pipeline.db, "bootstrap", fake_bootstrap)
    monkeypatch.setattr(fake_llm_run, "run", lambda *a, **k: LabelRunStats(
        labelled=17, cached=2, refused=1, failed=1, skipped=1,
        input_tokens=1000, output_tokens=100, latency_seconds=4.2,
        estimated_cost_usd=0.01,
    ))
    pipeline.phase_label(label_args(limit=20))
    out = capsys.readouterr().out
    for word in ("failed", "tokens", "latency", "estimated cost"):
        assert word in out.lower()
~~~

- [ ] **Step 2: Run CLI tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_cli.py -v

Expected: FAIL because build_parser and label-verify do not exist.

- [ ] **Step 3: Extract parser construction without changing main behavior**

Move parser creation from main() into build_parser() -> argparse.ArgumentParser. main() calls build_parser(), parses argv, and invokes args.func(args). Keep all existing command names and defaults byte-for-byte.

- [ ] **Step 4: Add label-verify handlers with lazy imports**

~~~python
def cmd_label_verify(args: argparse.Namespace) -> int:
    from src.llm import verify

    con = db.bootstrap()
    signals_run = args.signals_run or latest_run(con, "signals")
    if args.verify_action == "export":
        path = verify.export_worklist(
            con, signals_run, args.n, CONFIG.llm.verification_seed, Path(args.output)
        )
        print("worklist  : %s" % path)
        return 0
    if args.verify_action == "record":
        rows = verify.parse_worklist(Path(args.input), args.reviewer)
        print("recorded  : %d" % verify.record(
            con, rows, signals_run, args.worklist_version
        ))
        return 0
    report = verify.report(con, args.worklist_version)
    print(report.render())
    return 0
~~~

Add nested action choices export, record, report. Require --output for export; require --input and --reviewer for record. Default --n to CONFIG.llm.human_verify_n. Default worklist version to sha256 of the exported cluster IDs and model/prompt version.

- [ ] **Step 5: Pass the run registry ID into the label runner and print the complete summary**

Inside db.run(... ) as r, call llm_run.run(..., run_id=r.run_id). Print every LabelRunStats field. Print a clear “estimated, not invoice” suffix after cost.

- [ ] **Step 6: Run CLI, LLM, schema, and detection tests**

Run: .venv/bin/python -m pytest tests/test_llm_cli.py tests/test_llm.py tests/test_llm_client.py tests/test_llm_run.py tests/test_llm_verify.py tests/test_schema.py -v

Expected: PASS.

Run: .venv/bin/python -m pytest tests/test_signals.py tests/test_leakage.py -v

Expected: PASS.

- [ ] **Step 7: Run Ruff and commit the completed 8A subsystem**

Run: .venv/bin/ruff check src tests

Expected: PASS.

~~~bash
git add src/pipeline.py tests/test_llm_cli.py
git commit -m "Expose Phase 8 label verification workflows"
~~~

## Phase 8A completion gate

- A fake-client run labels, caches, resumes, records usage, and performs no duplicate provider call.
- Authentication, billing, permission, and invalid-request failures stop after one failing request.
- Cache corruption is quarantined and recomputed.
- The exported review is seeded, stratified, and signal-blinded.
- The report includes human numerators, denominators, Wilson intervals, fired/control breakdown, confidence breakdown, and failure categories.
- Existing signal and leakage tests pass.
- No live API call is required for this subsystem's automated completion.
