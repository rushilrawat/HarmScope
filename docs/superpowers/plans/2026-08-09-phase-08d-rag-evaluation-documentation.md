# Phase 8D RAG Evaluation and Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox ( - [ ] ) syntax for tracking.

**Goal:** Measure dense, BM25, fused retrieval, grounded answers, abstention, latency, tokens, cost, and human groundedness on a reproducible thirty-question benchmark, then reconcile every Phase 8 claim with measured reality.

**Architecture:** A committed PII-safe manifest stores synthetic analyst questions and hand-marked relevant complaint IDs. One evaluation runner scores all three retrieval variants from a single query encoding, optionally generates the fused answer, and persists one row per question/method. A gitignored claim-review worklist supplies the required human groundedness denominator without committing complaint or model prose.

**Tech Stack:** Python 3.11+, DuckDB 1.5.5, CSV, NumPy, pytest, existing Phase 8 retrieval and answer modules

## Global Constraints

- The committed benchmark contains exactly 30 questions: five in each of six predeclared categories.
- Question text is sanitized and synthetic; it neither quotes nor closely paraphrases complaint narratives.
- The automated privacy guard rejects any normalized eight-token sequence shared with a complaint narrative.
- Relevant complaint IDs are hand-marked; no model-generated relevance label becomes ground truth without human review.
- Dense, BM25, and fused Recall@10 and MRR are always reported, including when fusion loses.
- Fused win/tie/loss counts are reported against each component retriever.
- Generated answers are scored for citation validity, citation coverage, and answerable/unanswerable abstention.
- Human groundedness requires at least 50 reviewed claims; an automated or LLM judge cannot satisfy it.
- Evaluation questions, answers, or reviews never enter the detector or alter its configuration.
- Documentation reports missing live/human gates explicitly rather than presenting implementation as measured completion.

## File map

- Create data/ground_truth/rag_eval_questions.csv — committed safe questions and relevant IDs.
- Modify data/ground_truth/README.md — benchmark authorship, privacy, distribution, and freeze protocol.
- Create src/llm/eval.py — manifest validation, authoring worklist, metrics, runner, claim review, and reports.
- Create tests/test_llm_eval.py — privacy, metrics, persistence, answer evaluation, and report tests.
- Modify tests/test_ground_truth.py — committed benchmark contract.
- Modify src/pipeline.py — rag-eval run/author/import/review commands with lazy imports.
- Modify tests/test_llm_cli.py — evaluation parser and output.
- Modify README.md — implemented Phase 8 capabilities and honest live status.
- Modify docs/LLM_LAYER.md — final interfaces, evaluation protocol, and model/pricing provenance.
- Modify docs/ENGINEERING_NOTES.md — commands, measured values, failures, and cost.
- Modify docs/phases/phase-08-llm.md — built/tested/live/human gate state.
- Modify docs/ROADMAP.md — acceptance wording reconciled to signals-only determinism contract.
- Modify docs/superpowers/specs/2026-08-09-phase-08-llm-layer-design.md — final implementation deviations, if any, written as explicit amendments.

---

### Task 1: Create and validate the reproducible benchmark

**Files:**
- Create: src/llm/eval.py
- Create: tests/test_llm_eval.py
- Modify: tests/test_ground_truth.py
- Create: data/ground_truth/rag_eval_questions.csv
- Modify: data/ground_truth/README.md

**Interfaces:**
- Consumes: cluster_members and narratives for privacy validation and authoring evidence
- Produces:
  - EvalQuestion(question_id: str, question: str, cluster_id: str, company_id: str | None, category: str, relevant_complaint_ids: frozenset[int], answerable: bool)
  - load_manifest(path: Path, expected_n: int = 30) -> list[EvalQuestion]
  - validate_manifest(con, questions: list[EvalQuestion]) -> None
  - export_authoring_worklist(con, seed: int, path: Path) -> Path
  - import_authoring_worklist(con, source: Path, destination: Path) -> Path

- [ ] **Step 1: Write failing manifest shape and distribution tests**

~~~python
CATEGORIES = {
    "mechanism", "actors_preconditions", "consumer_consequence",
    "time_sequence", "company_response", "unanswerable",
}


def test_manifest_has_thirty_questions_balanced_across_categories():
    rows = llm_eval.load_manifest(RAG_MANIFEST)
    assert len(rows) == 30
    counts = Counter(row.category for row in rows)
    assert counts == {category: 5 for category in CATEGORIES}
    assert len({row.question_id for row in rows}) == 30


def test_answerable_rows_have_relevance_and_unanswerable_rows_do_not():
    rows = llm_eval.load_manifest(RAG_MANIFEST)
    for row in rows:
        if row.answerable:
            assert row.relevant_complaint_ids
        else:
            assert row.category == "unanswerable"
            assert row.relevant_complaint_ids == frozenset()
~~~

- [ ] **Step 2: Write failing privacy and referential-integrity tests**

~~~python
def test_privacy_guard_rejects_eight_token_overlap(eval_fixture):
    con, _ = eval_fixture
    bad = question(
        text="the bank repeatedly failed to release the consumer refund promptly",
        relevant_ids={10},
    )
    with pytest.raises(llm_eval.ManifestError, match="eight-token"):
        llm_eval.validate_manifest(con, [bad])


def test_manifest_relevant_ids_belong_to_its_scope(eval_fixture):
    con, _ = eval_fixture
    bad = question(cluster_id="cluster-1", relevant_ids={999})
    with pytest.raises(llm_eval.ManifestError, match="outside"):
        llm_eval.validate_manifest(con, [bad])
~~~

- [ ] **Step 3: Run manifest tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_eval.py -k manifest tests/test_ground_truth.py -v

Expected: FAIL because src.llm.eval and the manifest do not exist.

- [ ] **Step 4: Implement strict CSV parsing**

Use this committed header:

~~~csv
question_id,question,cluster_id,company_id,category,answerable,relevant_complaint_ids
~~~

relevant_complaint_ids is a semicolon-separated ascending integer list. answerable accepts only true or false. Reject blank IDs/questions/scope, unknown categories, duplicate question IDs, duplicate relevant IDs, invalid booleans, answerable rows without relevant IDs, and unanswerable rows with relevant IDs.

- [ ] **Step 5: Implement database validation and direct-quote privacy guard**

For each question, verify cluster_id exists; company_id, when present, occurs among that cluster's members; and every relevant ID belongs to the exact cluster/company scope. Normalize tokens with src.llm.retrieve.tokenize. Build all eight-token shingles from the question and from the scoped text_redacted narratives; reject any intersection and report question_id plus complaint_id without printing the shared text.

Automated shingle rejection catches copying, not semantic paraphrase. import_authoring_worklist therefore requires a privacy_reviewed column equal to yes for every row; the human reviewer owns paraphrase review.

- [ ] **Step 6: Implement an authoring worklist**

export_authoring_worklist selects five clusters per category using CONFIG.llm.verification_seed, balances product families and fired/control status, and writes to data/interim. Each row includes candidate cluster/company scope, ten redacted evidence excerpts, and blank question, relevant_complaint_ids, answerable, and privacy_reviewed fields. The file is gitignored because it contains narrative text.

import_authoring_worklist:

1. Requires exactly five completed rows in each category.
2. Requires privacy_reviewed=yes.
3. Removes all narrative columns.
4. Parses and validates relevant IDs against scope.
5. Runs the eight-token guard.
6. Writes only the seven committed columns in stable question_id order.

- [ ] **Step 7: Curate the thirty-question manifest**

Run:

~~~bash
.venv/bin/python -m src.pipeline rag-eval author --output data/interim/rag_eval_authoring.csv
~~~

For each category, complete exactly five rows:

- mechanism: ask what operational failure the complaints allege.
- actors_preconditions: ask who is affected and what condition exposes them.
- consumer_consequence: ask what concrete consequence follows.
- time_sequence: ask about ordering or timing stated in the evidence.
- company_response: ask what the company publicly said about the complaints.
- unanswerable: ask for a fact absent from all ten excerpts and mark answerable=false with no relevant IDs.

Write synthetic analyst phrasing from the mechanism concept; do not copy a sentence or preserve a distinctive eight-word sequence. Mark every complaint that directly supports the answer, not merely one convenient example. A human reads the final question against the scoped narratives and sets privacy_reviewed=yes.

Import:

~~~bash
.venv/bin/python -m src.pipeline rag-eval import --input data/interim/rag_eval_authoring.csv
~~~

Expected: data/ground_truth/rag_eval_questions.csv contains 30 safe rows and no narratives.

- [ ] **Step 8: Document the freeze and run tests**

data/ground_truth/README.md records the category distribution, ID-only relevance format, direct-quote guard, human paraphrase review, author/reviewer identifiers, and the commit that first freezes the benchmark. It explicitly forbids editing relevant IDs after seeing retrieval metrics; corrections create a new manifest version and preserve the old results.

Run: .venv/bin/python -m pytest tests/test_llm_eval.py -k manifest tests/test_ground_truth.py -v

Expected: PASS.

- [ ] **Step 9: Commit the benchmark contract**

~~~bash
git add src/llm/eval.py tests/test_llm_eval.py tests/test_ground_truth.py data/ground_truth/rag_eval_questions.csv data/ground_truth/README.md
git commit -m "Add reproducible RAG evaluation set"
~~~

---

### Task 2: Measure all three retrieval variants

**Files:**
- Modify: src/llm/eval.py
- Modify: tests/test_llm_eval.py

**Interfaces:**
- Consumes: retrieve_variants from Phase 8B and rag_eval_results
- Produces:
  - RetrievalMetrics(rank_first_relevant: int | None, relevant_retrieved_count: int, recall_at_10: float, reciprocal_rank: float)
  - score_ranking(ranked_ids: list[int], relevant_ids: set[int], k: int = 10) -> RetrievalMetrics
  - evaluate_retrieval_question(question: EvalQuestion, result: RetrievalResult) -> dict[str, RetrievalMetrics]
  - run_retrieval_eval(con, questions: list[EvalQuestion], embed_model: str, eval_run_id: str, retriever=retrieve_variants) -> EvaluationSummary

- [ ] **Step 1: Write failing Recall@10 and MRR tests**

~~~python
def test_score_ranking_uses_all_relevant_ids_as_recall_denominator():
    got = llm_eval.score_ranking([9, 2, 8, 4], {2, 4, 6}, k=10)
    assert got.rank_first_relevant == 2
    assert got.relevant_retrieved_count == 2
    assert got.recall_at_10 == pytest.approx(2 / 3)
    assert got.reciprocal_rank == pytest.approx(0.5)


def test_unanswerable_question_has_zero_retrieval_metrics():
    got = llm_eval.score_ranking([1, 2, 3], set(), k=10)
    assert got == RetrievalMetrics(None, 0, 0.0, 0.0)
~~~

- [ ] **Step 2: Write failing three-variant persistence and comparison tests**

~~~python
def test_eval_persists_dense_sparse_and_fused_once(eval_fixture):
    con, questions = eval_fixture
    summary = llm_eval.run_retrieval_eval(
        con, questions, "embed-m", "eval-1", retriever=FakeVariantRetriever()
    )
    rows = con.execute(
        "SELECT question_id, retrieval_method FROM rag_eval_results "
        "ORDER BY question_id, retrieval_method"
    ).fetchall()
    assert len(rows) == len(questions) * 3
    assert {method for _, method in rows} == {"dense", "bm25", "fused"}
    assert summary.fused_vs_dense.wins + summary.fused_vs_dense.ties + (
        summary.fused_vs_dense.losses
    ) == len(questions)
~~~

- [ ] **Step 3: Run retrieval-evaluation tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_eval.py -k "score_ranking or persists_dense" -v

Expected: FAIL because metric and runner functions do not exist.

- [ ] **Step 4: Implement ranking metrics**

Filter ranked_ids to the first k. rank_first_relevant is the one-based first relevant rank or None. relevant_retrieved_count counts unique relevant IDs in top k. Recall@k divides by all hand-marked relevant IDs. MRR is 1/rank_first_relevant or 0. Unanswerable questions report zeros and are excluded from aggregate retrieval means; report their count separately.

- [ ] **Step 5: Implement one-pass three-variant evaluation**

Call retrieve_variants once per question. Score dense complaint IDs, sparse complaint IDs, and fused complaint IDs independently. Upsert three rag_eval_results rows inside one question transaction. Store RetrievalResult.dense_seconds, sparse_seconds, and fusion_seconds on the corresponding rows.

For fused-vs-component win/tie/loss, compare per-question Recall@10 first and MRR second. A win requires a lexicographically larger (Recall@10, MRR) pair; an exact pair is a tie.

- [ ] **Step 6: Aggregate without hiding losses**

EvaluationSummary.render() prints:

- number of answerable and unanswerable questions;
- macro Recall@10 and MRR for dense, BM25, and fused;
- median and p95 latency for each;
- fused win/tie/loss versus dense;
- fused win/tie/loss versus BM25;
- per-question table so a poor fused query remains visible.

- [ ] **Step 7: Run tests and commit**

Run: .venv/bin/python -m pytest tests/test_llm_eval.py -k "score_ranking or retrieval or fused" -v

Expected: PASS.

~~~bash
git add src/llm/eval.py tests/test_llm_eval.py src/llm/retrieve.py tests/test_llm_retrieve.py
git commit -m "Evaluate dense BM25 and fused retrieval"
~~~

---

### Task 3: Evaluate generated answers and abstention

**Files:**
- Modify: src/llm/eval.py
- Modify: tests/test_llm_eval.py

**Interfaces:**
- Consumes: answer_question from Phase 8C and llm_usage linked by eval_run_id
- Produces:
  - citation_validity(answer: GroundedAnswer, retrieved_ids: set[int]) -> bool
  - citation_coverage(answer: GroundedAnswer) -> float
  - run_answer_eval(con, questions: list[EvalQuestion], embed_model: str, eval_run_id: str, answerer=answer_question) -> AnswerEvaluationSummary

- [ ] **Step 1: Write failing citation and abstention tests**

~~~python
def test_citation_metrics_are_deterministic():
    grounded = GroundedAnswer(
        answer="Consumers allege two problems.",
        claims=(Claim("First.", (10,)), Claim("Second.", (20,))),
        insufficient_evidence=False,
        limitations=(),
    )
    assert llm_eval.citation_validity(grounded, {10, 20}) is True
    assert llm_eval.citation_validity(grounded, {10}) is False
    assert llm_eval.citation_coverage(grounded) == 1.0


@pytest.mark.parametrize(
    ("answerable", "insufficient", "expected"),
    [(True, False, True), (True, True, False), (False, True, True), (False, False, False)],
)
def test_abstention_accuracy(answerable, insufficient, expected):
    assert llm_eval.abstention_correct(answerable, insufficient) is expected
~~~

- [ ] **Step 2: Write failing generated-evaluation persistence test**

~~~python
def test_answer_eval_updates_fused_rows_and_aggregates_usage(eval_fixture):
    con, questions = eval_fixture
    seed_retrieval_rows(con, "eval-1", questions)
    summary = llm_eval.run_answer_eval(
        con, questions, "embed-m", "eval-1", answerer=FakeAnswerer()
    )
    got = con.execute(
        "SELECT citation_valid, citation_coverage, abstention_correct "
        "FROM rag_eval_results WHERE eval_run_id = 'eval-1' "
        "AND retrieval_method = 'fused'"
    ).fetchall()
    assert all(valid is not None for valid, _, _ in got)
    assert summary.input_tokens == 300
    assert summary.estimated_cost_usd == pytest.approx(0.03)
~~~

- [ ] **Step 3: Run answer-evaluation tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_eval.py -k "citation or abstention or answer_eval" -v

Expected: FAIL because answer metrics do not exist.

- [ ] **Step 4: Implement deterministic answer metrics**

citation_validity checks every cited ID is in retrieved_ids. citation_coverage is cited substantive claims / total claims, with 1.0 for a correct abstention containing zero claims and 0.0 for a non-abstaining zero-claim answer. abstention_correct is answerable != insufficient_evidence.

- [ ] **Step 5: Run generation with evaluation provenance**

Call answer_question with run_id=eval_run_id. Update only the fused rag_eval_results row for citation/abstention fields because answers consume fused evidence. Authentication/billing failures stop immediately and leave already completed questions persisted. Schema/citation failures are recorded as failed questions and remain visible; they are not converted to zeros.

Aggregate input/output/cache tokens, latency, and estimated cost from llm_usage WHERE run_id = eval_run_id AND operation = 'answer'. Report cache hits separately so a replay cannot appear to have zero cost without explanation.

- [ ] **Step 6: Run tests and commit**

Run: .venv/bin/python -m pytest tests/test_llm_eval.py -k "citation or abstention or answer_eval" -v

Expected: PASS.

~~~bash
git add src/llm/eval.py tests/test_llm_eval.py
git commit -m "Evaluate grounded RAG answers"
~~~

---

### Task 4: Add the fifty-claim human groundedness gate

**Files:**
- Modify: src/llm/eval.py
- Modify: tests/test_llm_eval.py

**Interfaces:**
- Consumes: cached rag_answers and retrieved complaint IDs
- Produces:
  - export_claim_review(con, eval_run_id: str, n: int, seed: int, path: Path) -> Path
  - parse_claim_review(path: Path, reviewer_id: str) -> list[ClaimReview]
  - record_claim_review(con, eval_run_id: str, reviews: list[ClaimReview]) -> GroundednessReport

- [ ] **Step 1: Write failing blinded sampling and validation tests**

~~~python
def test_claim_review_exports_fifty_claims_without_model_confidence(
    claim_fixture, tmp_path
):
    con, eval_run_id = claim_fixture
    path = llm_eval.export_claim_review(
        con, eval_run_id, 50, 7, tmp_path / "claims.csv"
    )
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 50
    assert "confidence" not in rows[0]
    assert all(row["claim_text"] and row["cited_complaint_ids"] for row in rows)


def test_claim_review_rejects_blank_or_unknown_groundedness(tmp_path):
    path = filled_claim_review(tmp_path, grounded="mostly")
    with pytest.raises(ValueError, match="yes.*no"):
        llm_eval.parse_claim_review(path, "reviewer-1")
~~~

- [ ] **Step 2: Write failing report and persistence tests**

~~~python
def test_groundedness_report_requires_fifty_and_updates_eval_rows(claim_fixture):
    con, eval_run_id = claim_fixture
    reviews = claim_reviews(grounded=44, total=50)
    report = llm_eval.record_claim_review(con, eval_run_id, reviews)
    assert report.grounded == 44
    assert report.reviewed == 50
    assert report.rate == pytest.approx(0.88)
    assert report.ci_low < 0.88 < report.ci_high
    totals = con.execute(
        "SELECT sum(grounded_claims), sum(reviewed_claims) "
        "FROM rag_eval_results WHERE eval_run_id = ? "
        "AND retrieval_method = 'fused'", [eval_run_id]
    ).fetchone()
    assert totals == (44, 50)
~~~

- [ ] **Step 3: Run claim-review tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_eval.py -k claim_review -v

Expected: FAIL because claim review does not exist.

- [ ] **Step 4: Implement deterministic claim sampling**

Read claims from rag_answers associated with the evaluation questions. Stratify across question category, answerable status, company, and claim position. Select with a fixed seed and hash tie-break; require at least 50 available claims. Export question, claim text, cited IDs, and the cited redacted evidence excerpts. Do not export model confidence, retrieval scores, or whether the expected answer is answerable.

The file lives under data/interim and remains gitignored because it contains generated and consumer text.

- [ ] **Step 5: Implement review validation and aggregated persistence**

Review columns are grounded=yes/no, failure_category, notes. failure_category is none for yes; for no require one of unsupported, contradicted, overgeneralized, citation_mismatch, or other. Store reviewer_id in the local CSV filename and report output; do not add a fifth database table. Aggregate grounded/reviewed counts by question and update the fused rag_eval_results rows. Use the same Wilson implementation as src.llm.verify.

Refuse to call the Phase 8 human gate complete when reviewed < 50.

- [ ] **Step 6: Run tests and commit**

Run: .venv/bin/python -m pytest tests/test_llm_eval.py -k claim_review -v

Expected: PASS.

~~~bash
git add src/llm/eval.py tests/test_llm_eval.py
git commit -m "Add human RAG groundedness review"
~~~

---

### Task 5: Wire evaluation CLI, run regression, and reconcile documentation

**Files:**
- Modify: src/pipeline.py
- Modify: tests/test_llm_cli.py
- Modify: README.md
- Modify: docs/LLM_LAYER.md
- Modify: docs/ENGINEERING_NOTES.md
- Modify: docs/phases/phase-08-llm.md
- Modify: docs/ROADMAP.md
- Modify: docs/superpowers/specs/2026-08-09-phase-08-llm-layer-design.md if implementation required an amendment

**Interfaces:**
- Consumes: all src.llm.eval public functions
- Produces:
  - python -m src.pipeline rag-eval
  - python -m src.pipeline rag-eval --retrieval-only
  - python -m src.pipeline rag-eval author --output PATH
  - python -m src.pipeline rag-eval import --input PATH
  - python -m src.pipeline rag-eval claims-export --run-id ID --output PATH
  - python -m src.pipeline rag-eval claims-record --run-id ID --input PATH --reviewer ID

- [ ] **Step 1: Write failing parser and report tests**

~~~python
def test_rag_eval_defaults_to_full_thirty_question_run():
    parser = pipeline.build_parser()
    args = parser.parse_args(["rag-eval"])
    assert args.eval_action == "run"
    assert args.retrieval_only is False


def test_rag_eval_output_names_every_metric(monkeypatch, capsys):
    monkeypatch.setattr(fake_eval, "run_all", lambda *a, **k: full_summary())
    pipeline.cmd_rag_eval(eval_args())
    output = capsys.readouterr().out
    for label in (
        "Recall@10", "MRR", "dense", "BM25", "fused", "win/tie/loss",
        "citation validity", "citation coverage", "abstention",
        "tokens", "estimated cost", "groundedness",
    ):
        assert label.lower() in output.lower()
~~~

- [ ] **Step 2: Run CLI tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_cli.py -k rag_eval -v

Expected: FAIL because rag-eval is not registered.

- [ ] **Step 3: Implement lazy CLI handlers and run registry**

The run action:

1. Loads and database-validates the committed 30-question manifest.
2. Creates db.run(con, "rag-eval", CONFIG, params including manifest SHA-256, embed model, retrieval_only).
3. Runs retrieval evaluation.
4. Unless --retrieval-only, runs answer evaluation.
5. Calls r.finish(output_rows=90), representing 30 questions x 3 retrieval variants.
6. Prints the complete summary and the eval run ID needed for claim review.

author, import, claims-export, and claims-record do not pretend to be completed evaluations; they print artifact paths and counts.

- [ ] **Step 4: Run automated verification**

Run: .venv/bin/python -m pytest tests/test_llm.py tests/test_llm_client.py tests/test_llm_run.py tests/test_llm_verify.py tests/test_llm_retrieve.py tests/test_llm_answer.py tests/test_llm_eval.py tests/test_llm_cli.py tests/test_schema.py tests/test_ground_truth.py -v

Expected: PASS.

Run: .venv/bin/python -m pytest

Expected: PASS.

Run: .venv/bin/ruff check src tests

Expected: PASS.

- [ ] **Step 5: Run the network-free evaluation**

Run:

~~~bash
.venv/bin/python -m src.pipeline rag-eval --retrieval-only
~~~

Expected: 90 rag_eval_results rows and printed Recall@10/MRR for all three methods. Record actual values in ENGINEERING_NOTES; do not round away losses.

- [ ] **Step 6: Run paid and human gates when external prerequisites exist**

Run:

~~~bash
.venv/bin/python -m src.pipeline run --phase label --limit 20
.venv/bin/python -m src.pipeline label-verify export --n 50 --output data/interim/label_review.csv
.venv/bin/python -m src.pipeline rag-eval
.venv/bin/python -m src.pipeline rag-eval claims-export --run-id EVAL_RUN_ID --output data/interim/rag_claim_review.csv
~~~

After human completion, record label and claim reviews with the CLI. If API credit or a human reviewer is unavailable, do not manufacture values: documentation says “implementation verified with fake clients; live provider gate blocked by billing” or “human review pending,” including date and exact blocker.

- [ ] **Step 7: Reconcile documentation with measured reality**

README.md explains the deterministic detector plus descriptive LLM boundary and lists FAISS, BM25, RRF, structured outputs, cache/resume, evaluation, and human review. It distinguishes automated completion from live/human completion.

docs/LLM_LAYER.md records exact public interfaces, retry taxonomy, cache keys, benchmark protocol, pricing fingerprint, and actual metrics when available.

docs/ENGINEERING_NOTES.md records commands, run IDs, manifest hash, per-method metrics, fusion losses, failures, tokens, estimated cost, and reviewer denominators.

docs/phases/phase-08-llm.md replaces stale “credentials expired” language with the observed billing/model/preflight state and a checklist of built, automated, live, and human gates.

docs/ROADMAP.md corrects Phase 8 acceptance to the already-approved honest contract: signals is byte-identical with src/llm removed; baseline_results is byte-identical given the same backtest_links.

The design spec receives an amendment only for real implementation deviations; never rewrite the approved decision as though it was the original plan.

- [ ] **Step 8: Commit Phase 8D**

~~~bash
git add src/llm/eval.py src/pipeline.py tests/test_llm_eval.py tests/test_llm_cli.py data/ground_truth/rag_eval_questions.csv data/ground_truth/README.md README.md docs/LLM_LAYER.md docs/ENGINEERING_NOTES.md docs/phases/phase-08-llm.md docs/ROADMAP.md docs/superpowers/specs/2026-08-09-phase-08-llm-layer-design.md
git commit -m "Complete Phase 8 RAG evaluation"
~~~

## Phase 8D completion gate

- The committed manifest contains 30 validated, balanced, PII-safe questions.
- Dense, BM25, and fused Recall@10/MRR plus fusion win/tie/loss are reported.
- Citation validity, citation coverage, and abstention accuracy are reported when generation runs.
- Token, latency, cache, and estimated-cost totals are tied to an evaluation run ID.
- At least 50 claims are human-reviewed before groundedness is called complete.
- Full pytest and Ruff suites pass.
- Every document distinguishes automated, live-provider, and human-review status.
