# Phase 8C Grounded Answers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox ( - [ ] ) syntax for tracking.

**Goal:** Turn scoped hybrid-retrieval results into a cautious analyst answer whose every substantive claim cites complaint IDs from the returned evidence.

**Architecture:** Retrieval remains deterministic and separate. The answer module builds a structured prompt, calls the reliable client from Phase 8A, validates every output field and citation locally, and persists answers under a key that changes with question, model, prompt, scope, or ordered evidence IDs. The CLI renders complaint allegations, company responses, and optional enforcement context as distinct sections.

**Tech Stack:** Python 3.11+, DuckDB 1.5.5, Anthropic structured outputs, SHA-256 cache keys, pytest

## Global Constraints

- The answer generator never affects signal detection or retrieval ranking.
- Every substantive generated claim has at least one complaint_id.
- Every cited complaint_id must occur in the fused evidence passed to the model.
- An answer with no relevant evidence sets insufficient_evidence=true and contains no substantive claims.
- Answers use allegation language and do not state that conduct occurred or violated law.
- Company public responses are labeled as company statements, never complaint evidence.
- Enforcement records are optional context and cannot satisfy a complaint citation.
- The standing complaint-allegation disclaimer is deterministic UI/CLI copy, not model output.
- Raw narratives and model responses containing consumer text are never logged or committed.
- Cache identity includes normalized question, cluster, company, model, prompt version, and ordered evidence complaint IDs.

## File map

- Create src/llm/answer.py — prompt, output schema, validation, hashes, persistence, and orchestration.
- Create tests/test_llm_answer.py — schema, grounding, caching, failure, and end-to-end fake-client tests.
- Modify src/pipeline.py — ask command with lazy imports and deterministic rendering.
- Modify tests/test_llm_cli.py — ask parser and output tests.
- Modify tests/test_llm.py — answer tables remain unreachable from detection.

---

### Task 1: Define and enforce the grounded answer contract

**Files:**
- Create: src/llm/answer.py
- Create: tests/test_llm_answer.py

**Interfaces:**
- Consumes: RetrievedEvidence from src.llm.retrieve
- Produces:
  - Claim(text: str, complaint_ids: tuple[int, ...])
  - GroundedAnswer(answer: str, claims: tuple[Claim, ...], insufficient_evidence: bool, limitations: tuple[str, ...])
  - validate_answer(payload: dict, allowed_ids: set[int]) -> GroundedAnswer
  - build_prompt(question: str, evidence: list[RetrievedEvidence], enforcement_context: list[EnforcementContext]) -> str

- [ ] **Step 1: Write failing schema and grounding tests**

~~~python
def test_answer_schema_is_closed_and_complete():
    assert answer.ANSWER_SCHEMA["additionalProperties"] is False
    assert set(answer.ANSWER_SCHEMA["required"]) == set(
        answer.ANSWER_SCHEMA["properties"]
    )


def test_validator_rejects_unretrieved_citations():
    payload = answer_payload(
        claims=[{"text": "Consumers allege delayed refunds.", "complaint_ids": [999]}]
    )
    with pytest.raises(answer.CitationError, match="999"):
        answer.validate_answer(payload, {10, 20})


def test_validator_requires_a_citation_on_every_claim():
    payload = answer_payload(
        claims=[{"text": "Consumers allege delayed refunds.", "complaint_ids": []}]
    )
    with pytest.raises(answer.CitationError, match="at least one"):
        answer.validate_answer(payload, {10, 20})


def test_insufficient_evidence_cannot_smuggle_a_substantive_answer():
    payload = answer_payload(
        answer="The company withheld refunds.",
        claims=[{"text": "Refunds were withheld.", "complaint_ids": [10]}],
        insufficient_evidence=True,
    )
    with pytest.raises(answer.AnswerSchemaError, match="insufficient"):
        answer.validate_answer(payload, {10})
~~~

- [ ] **Step 2: Write failing allegation-framing and prompt-separation tests**

~~~python
def test_prompt_labels_evidence_and_company_response_separately():
    prompt = answer.build_prompt(
        "Why were refunds delayed?",
        [evidence(
            complaint_id=10,
            text_redacted="My refund did not arrive.",
            company_public_response="Company states the matter was resolved.",
        )],
        [enforcement_context(action_id="a1", harm_summary="Public action summary.")],
    )
    assert "COMPLAINT EVIDENCE [10]" in prompt
    assert "COMPANY PUBLIC RESPONSE" in prompt
    assert "ENFORCEMENT CONTEXT [a1]" in prompt
    assert prompt.index("COMPLAINT EVIDENCE") < prompt.index("COMPANY PUBLIC RESPONSE")
~~~

- [ ] **Step 3: Run focused tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_answer.py -k "schema or validator or prompt" -v

Expected: FAIL because src.llm.answer does not exist.

- [ ] **Step 4: Add the strict JSON schema**

~~~python
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "complaint_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                    },
                },
                "required": ["text", "complaint_ids"],
                "additionalProperties": False,
            },
        },
        "insufficient_evidence": {"type": "boolean"},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "claims", "insufficient_evidence", "limitations"],
    "additionalProperties": False,
}
~~~

- [ ] **Step 5: Define typed output and local validation**

~~~python
@dataclass(frozen=True)
class Claim:
    text: str
    complaint_ids: tuple[int, ...]


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    claims: tuple[Claim, ...]
    insufficient_evidence: bool
    limitations: tuple[str, ...]


class AnswerSchemaError(ValueError):
    pass


class CitationError(AnswerSchemaError):
    pass
~~~

Require the exact top-level keys. Require exact claim keys, nonblank claim text, nonempty unique integer IDs, nonblank limitation strings, and no cited ID outside allowed_ids. If insufficient_evidence is true, require answer.strip() == "" and claims == []. If false, require a nonblank answer and at least one claim.

The validator does not infer whether prose is legally cautious; the system prompt requires “complaints allege” framing and the human groundedness review measures failures. Deterministic guarantees remain schema and citation scope.

- [ ] **Step 6: Implement prompt construction**

The stable SYSTEM string says:

1. Use only supplied complaint evidence.
2. Frame all conduct as allegations.
3. Never name individuals or declare a legal violation.
4. Put each independently checkable sentence in claims with supporting IDs.
5. Company responses and enforcement records are context, not complaint evidence.
6. If evidence cannot answer the question, return an empty answer, no claims, insufficient_evidence=true, and a limitation.

build_prompt normalizes question whitespace but does not alter evidence text. Render each complaint with ID, date, company, product family, and redacted narrative. Render duplicate company public responses once in their own section. Render enforcement context last and explicitly prohibit using its action_id as a complaint citation.

- [ ] **Step 7: Run contract tests and commit**

Run: .venv/bin/python -m pytest tests/test_llm_answer.py -k "schema or validator or prompt" -v

Expected: PASS.

~~~bash
git add src/llm/answer.py tests/test_llm_answer.py
git commit -m "Define grounded RAG answer contract"
~~~

---

### Task 2: Add cache identity and database persistence

**Files:**
- Modify: src/llm/answer.py
- Modify: tests/test_llm_answer.py

**Interfaces:**
- Consumes: rag_answers table from Phase 8A
- Produces:
  - normalize_question(question: str) -> str
  - question_hash(question: str) -> str
  - evidence_hash(evidence: list[RetrievedEvidence]) -> str
  - load_cached_answer(con, question: str, cluster_id: str, company_id: str, model: str, prompt_version: str, evidence: list[RetrievedEvidence]) -> GroundedAnswer | None
  - write_cached_answer(con, ...) -> None

- [ ] **Step 1: Write failing cache-key and persistence tests**

~~~python
def test_answer_identity_changes_with_ordered_evidence():
    a = [evidence(complaint_id=10), evidence(complaint_id=20)]
    b = [evidence(complaint_id=20), evidence(complaint_id=10)]
    assert answer.evidence_hash(a) != answer.evidence_hash(b)
    assert answer.question_hash("  Why   delayed? ") == answer.question_hash(
        "why delayed?"
    )


def test_cached_answer_requires_exact_scope_model_prompt_and_evidence(answer_fixture):
    con, evidence_rows = answer_fixture
    expected = grounded_answer()
    answer.write_cached_answer(
        con, "Why delayed?", "cluster-1", "company-1", "model-a", "v1",
        evidence_rows, expected,
    )
    assert answer.load_cached_answer(
        con, "why delayed?", "cluster-1", "company-1", "model-a", "v1",
        evidence_rows,
    ) == expected
    assert answer.load_cached_answer(
        con, "why delayed?", "cluster-1", "company-1", "model-a", "v2",
        evidence_rows,
    ) is None
    assert answer.load_cached_answer(
        con, "why delayed?", "cluster-1", "company-1", "model-a", "v1",
        list(reversed(evidence_rows)),
    ) is None
~~~

- [ ] **Step 2: Run cache tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_answer.py -k "identity or cached_answer" -v

Expected: FAIL because hashes and persistence do not exist.

- [ ] **Step 3: Implement normalized hashes**

normalize_question collapses whitespace, strips, and casefolds. Reject an empty question. question_hash is SHA-256 of normalized question. evidence_hash is SHA-256 of a JSON list of complaint IDs in retrieval order. Do not sort the evidence IDs: retrieval-order changes are cache-significant.

- [ ] **Step 4: Implement parameterized cache reads and writes**

Select on question_hash, evidence_hash, cluster_id, company_id, model, and prompt_version. Parse answer_json then pass it through validate_answer using evidence_ids_json as the allowed set. A corrupt row is deleted and treated as a cache miss; it is never returned.

Use INSERT ... ON CONFLICT DO UPDATE so a repeated exact input replaces a damaged or manually repaired row without creating duplicates. Serialize with sort_keys=True.

- [ ] **Step 5: Run cache tests and commit**

Run: .venv/bin/python -m pytest tests/test_llm_answer.py -k "identity or cached_answer" -v

Expected: PASS.

~~~bash
git add src/llm/answer.py tests/test_llm_answer.py
git commit -m "Cache grounded answers by evidence identity"
~~~

---

### Task 3: Orchestrate retrieval, generation, validation, and usage

**Files:**
- Modify: src/llm/answer.py
- Modify: tests/test_llm_answer.py

**Interfaces:**
- Consumes:
  - retrieve_evidence from Phase 8B
  - AnthropicModelClient.call_json and ModelCallResult from Phase 8A
  - llm_usage table and rag_answers table
- Produces:
  - EnforcementContext(action_id: str, filed_date: date, company_id: str | None, product_family: str | None, harm_summary: str, source_url: str | None)
  - load_enforcement_context(con, company_id: str, product_family: str, limit: int = 5) -> list[EnforcementContext]
  - AnswerResult(answer: GroundedAnswer, evidence: tuple[RetrievedEvidence, ...], enforcement_context: tuple[EnforcementContext, ...], cached: bool, usage: TokenUsage, latency_seconds: float, estimated_cost_usd: float)
  - answer_question(con, cluster_id: str, company_id: str, question: str, embed_model: str, include_enforcement_context: bool = False, run_id: str | None = None, model_client=None, retriever=retrieve_evidence) -> AnswerResult

- [ ] **Step 1: Write failing fake-client orchestration tests**

~~~python
def test_answer_question_uses_fused_evidence_and_persists_once(answer_fixture):
    con, evidence_rows = answer_fixture
    client = FakeModelClient(result(answer_payload(
        claims=[{"text": "Consumers allege delayed refunds.", "complaint_ids": [10]}]
    )))
    retriever = FakeRetriever(evidence_rows)
    first = answer.answer_question(
        con, "cluster-1", "company-1", "Why delayed?", "embed-m",
        model_client=client, retriever=retriever,
    )
    second = answer.answer_question(
        con, "cluster-1", "company-1", "why delayed?", "embed-m",
        model_client=client, retriever=retriever,
    )
    assert first.cached is False
    assert second.cached is True
    assert client.calls == 1
    assert con.execute("SELECT count(*) FROM rag_answers").fetchone()[0] == 1
    assert con.execute(
        "SELECT cache_status FROM llm_usage ORDER BY created_at"
    ).fetchall() == [("miss",), ("hit",)]


def test_invalid_model_citation_is_not_cached(answer_fixture):
    con, evidence_rows = answer_fixture
    client = FakeModelClient(result(answer_payload(
        claims=[{"text": "Unsupported.", "complaint_ids": [999]}]
    )))
    with pytest.raises(answer.CitationError):
        answer.answer_question(
            con, "cluster-1", "company-1", "Question", "embed-m",
            model_client=client, retriever=FakeRetriever(evidence_rows),
        )
    assert con.execute("SELECT count(*) FROM rag_answers").fetchone()[0] == 0
    assert con.execute(
        "SELECT outcome, error_category FROM llm_usage"
    ).fetchone() == ("failed", "citation")
~~~

- [ ] **Step 2: Write failing no-evidence and enforcement-context tests**

~~~python
def test_empty_retrieval_abstains_without_model_call(answer_fixture):
    con, _ = answer_fixture
    client = FakeModelClient([])
    got = answer.answer_question(
        con, "cluster-1", "company-1", "Question", "embed-m",
        model_client=client, retriever=FakeRetriever([]),
    )
    assert got.answer.insufficient_evidence is True
    assert got.answer.claims == ()
    assert client.calls == 0


def test_enforcement_context_is_scoped_and_separate(answer_fixture):
    con, _ = answer_fixture
    rows = answer.load_enforcement_context(con, "company-1", "mortgage")
    assert all(row.company_id == "company-1" for row in rows)
    assert all(row.product_family in (None, "mortgage") for row in rows)
~~~

- [ ] **Step 3: Run orchestration tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_answer.py -k "answer_question or empty_retrieval or enforcement_context" -v

Expected: FAIL because orchestration does not exist.

- [ ] **Step 4: Implement enforcement-context loading**

Read only usable enforcement_actions where company_id matches and product_family is NULL or equals the requested family. Order by filed_date descending, action_id. Return at most limit. This query is downstream context and cannot be imported by detection code.

- [ ] **Step 5: Implement answer_question**

1. Call the injected retriever for fused top-k evidence.
2. Return a deterministic insufficient-evidence AnswerResult when evidence is empty; record llm_usage outcome=skipped, cache_status=bypass, zero tokens/cost.
3. Load an exact cached answer. On hit, record llm_usage outcome=ok, cache_status=hit, zero provider tokens/cost, and return.
4. Optionally load enforcement context.
5. Call model_client.call_json with SYSTEM, build_prompt, ANSWER_SCHEMA, and max_tokens=2000.
6. Validate payload against only retrieved complaint IDs.
7. BEGIN; write rag_answers and one llm_usage row; COMMIT.
8. On citation/schema failure, record failed usage without answer persistence and re-raise.
9. Propagate terminal ModelCallError categories after recording them.

Do not write prompt or response content into llm_usage. input_hash is SHA-256 over question hash, evidence hash, model, prompt version, cluster, and company. Persist the optional run_id on llm_usage so the evaluation harness can aggregate token, latency, and estimated-cost totals for one evaluation run.

- [ ] **Step 6: Run the full answer suite and commit**

Run: .venv/bin/python -m pytest tests/test_llm_answer.py -v

Expected: PASS.

Run: .venv/bin/ruff check src/llm/answer.py tests/test_llm_answer.py

Expected: PASS.

~~~bash
git add src/llm/answer.py tests/test_llm_answer.py
git commit -m "Generate and persist grounded RAG answers"
~~~

---

### Task 4: Add the analyst ask CLI

**Files:**
- Modify: src/pipeline.py
- Modify: tests/test_llm_cli.py
- Modify: tests/test_llm.py

**Interfaces:**
- Consumes: answer_question and AnswerResult
- Produces: python -m src.pipeline ask --cluster-id ID --company-id ID --question TEXT [--model EMBED_MODEL] [--include-enforcement-context]

- [ ] **Step 1: Write failing parser and rendering tests**

~~~python
def test_ask_command_requires_scope_and_question():
    parser = pipeline.build_parser()
    args = parser.parse_args([
        "ask", "--cluster-id", "c1", "--company-id", "co1",
        "--question", "Why were funds unavailable?",
    ])
    assert args.cluster_id == "c1"
    assert args.company_id == "co1"
    assert args.question.startswith("Why")


def test_ask_renders_sources_responses_and_disclaimer(monkeypatch, capsys):
    monkeypatch.setattr(fake_answer, "answer_question", lambda *a, **k: result())
    pipeline.cmd_ask(ask_args())
    out = capsys.readouterr().out
    assert "Complaint 10" in out
    assert "Company public response" in out
    assert "consumer allegations" in out
    assert "does not indicate" in out
~~~

- [ ] **Step 2: Run CLI tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_cli.py -k ask -v

Expected: FAIL because ask is not registered.

- [ ] **Step 3: Implement a function-local Phase 8 handler**

~~~python
DISCLAIMER = (
    "Complaints are consumer allegations. Publication does not indicate the "
    "CFPB verified the allegations or that the company acted unlawfully."
)


def cmd_ask(args: argparse.Namespace) -> int:
    from src.llm import answer

    con = db.bootstrap()
    result = answer.answer_question(
        con, args.cluster_id, args.company_id, args.question,
        args.model or CONFIG.embed.dev_model,
        include_enforcement_context=args.include_enforcement_context,
    )
    print(answer.render_cli(result, disclaimer=DISCLAIMER))
    return 0
~~~

render_cli shows:

1. The short answer.
2. Each claim followed by complaint IDs.
3. Retrieved complaints in fused order with ID, date, company, and redacted narrative.
4. Unique company public responses under a separate heading.
5. Optional enforcement context under a separate heading with source URLs.
6. Limitations, cache/cost metadata, and the fixed disclaimer.

- [ ] **Step 4: Extend the import-boundary assertions**

Assert cmd_ask imports src.llm inside the function and that pipeline has no module-level Phase 8 import. Assert detection source files contain none of rag_answers, rag_eval_results, llm_usage, or label_verifications.

- [ ] **Step 5: Run CLI, answer, retrieval, and leakage suites**

Run: .venv/bin/python -m pytest tests/test_llm_cli.py tests/test_llm_answer.py tests/test_llm_retrieve.py tests/test_llm.py -v

Expected: PASS.

Run: .venv/bin/python -m pytest tests/test_signals.py tests/test_leakage.py -v

Expected: PASS.

- [ ] **Step 6: Run Ruff and commit Phase 8C**

Run: .venv/bin/ruff check src tests

Expected: PASS.

~~~bash
git add src/pipeline.py tests/test_llm_cli.py tests/test_llm.py
git commit -m "Expose grounded analyst answers"
~~~

## Phase 8C completion gate

- A fake-client question returns structured claims and the exact supporting complaints.
- Any citation outside retrieved evidence is rejected and never cached.
- Exact repeated inputs return from the database cache without another provider call.
- Empty retrieval abstains deterministically without a provider call.
- Complaint evidence, company response, enforcement context, and disclaimer are distinct in CLI output.
- Detection and leakage suites remain green.
