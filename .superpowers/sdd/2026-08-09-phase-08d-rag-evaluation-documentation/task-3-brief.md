# Phase 8D Task 3 — grounded-answer, citation, abstention, and usage evaluation

Baseline: `6a8bf4d67c7ac46369d9b93eb0748b2cc2854fcc`.

Governing plan: `docs/superpowers/plans/2026-08-09-phase-08d-rag-evaluation-documentation.md`, Task 3 and Global Constraints, reconciled to the reviewed Phase 8C `AnswerResult`/`GroundedAnswer` contracts.

## Goal

Evaluate fused grounded answers deterministically, persist answer metrics only onto their existing fused retrieval rows, classify answerability/abstention correctly, and aggregate run-linked tokens, latency, cache, and estimated cost without hiding failed questions.

## Files

- Modify `src/llm/eval.py`.
- Modify `tests/test_llm_eval.py`.
- Do not modify answer/client/schema code unless a reproduced load-bearing incompatibility requires controller approval.

## Required interfaces and semantics

- `citation_validity(answer: GroundedAnswer, retrieved_ids: set[int]) -> bool`: every cited complaint ID must be a positive ID present in retrieved evidence; malformed/duplicate claim citations fail closed without mutating the answer.
- `citation_coverage(answer: GroundedAnswer) -> float`: fraction of claims with at least one citation; correct zero-claim abstention is 1.0, non-abstaining zero-claim answer is 0.0.
- `abstention_correct(answerable: bool, insufficient_evidence: bool) -> bool` implements `answerable != insufficient_evidence` with strict boolean inputs.
- `run_answer_eval(con, questions, embed_model, eval_run_id, answerer=answer_question)` validates pure inputs before DB/provider access, requires an autocommit connection, requires an existing fused retrieval row for every question, calls the answerer with `run_id=eval_run_id`, and updates only `citation_valid`, `citation_coverage`, and `abstention_correct` on that question's fused row.
- Derive retrieved IDs from the returned `AnswerResult.evidence`; verify returned evidence scope/IDs are exact, unique, and consistent with the requested cluster/company before using them for citation validity.
- Authentication, billing, configuration, and other terminal provider/preflight failures stop immediately while previously completed questions stay committed.
- Answer schema/citation/refusal failures remain visible and do not become numeric zeros. Continue only for the explicitly documented per-question failure classes. Preserve the typed original exception/accounting behavior owned by Phase 8C.
- Because `rag_eval_results` has no failure-detail columns, represent current-run per-question failures explicitly in the typed summary/render while leaving answer fields NULL; durable failure accounting remains in `llm_usage` linked by `run_id`, operation, question hash, outcome, and error category. Do not silently add a schema or claim failure persistence that does not exist.
- Aggregate only `llm_usage` rows with exact `run_id` and `operation='answer'`: input/output/cache tokens, total latency, estimated cost, outcomes, and cache hit/miss/bypass counts. Reject corrupt/non-finite/negative aggregates rather than reporting them.
- Replays must not alter dense/BM25 rows, retrieval metrics, or human-review fields. A question update is atomic; later failure preserves prior question metrics. Summary/render must identify failed question IDs and categories without question/narrative/model prose.
- No real provider call is required for engineering verification. The live paid run remains an external gate.

## TDD and verification

Use strict RED→GREEN. Cover deterministic citation validity/coverage, malformed citation IDs, abstention matrix/types, fused-only updates, missing retrieval rows, wrong answer evidence scope/duplicates, cache hit/miss/bypass usage aggregation, paid failures, per-question continuation vs terminal stop, NULL-vs-zero failure behavior, caller transaction safety, DB failure rollback/preservation, replay, and ID-only stable rendering. Run Task 3 tests, answer/client/retrieval/schema/isolation suites, full pytest, Ruff, scoped format, and diff check. Self-review, write `task-3-report.md`, update ledger, commit, and leave a clean tree.
