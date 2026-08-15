# Phase 8D Task 2 — dense, BM25, and fused retrieval evaluation

Baseline: `461437816806b98cabfd340a6487aee2cade2a27`.

Governing plan: `docs/superpowers/plans/2026-08-09-phase-08d-rag-evaluation-documentation.md`, Task 2 and Global Constraints.

## Goal

Implement deterministic, resumable evaluation of all three Phase 8B retrieval variants from one retrieval call per question, persist one method row per question, and report losses as visibly as wins.

## Files

- Modify `src/llm/eval.py`.
- Modify `tests/test_llm_eval.py`.
- Modify another retrieval/schema file only if an independently reproduced contract gap makes it necessary; ask before expanding scope.

## Required interfaces and semantics

- `RetrievalMetrics(rank_first_relevant, relevant_retrieved_count, recall_at_10, reciprocal_rank)`.
- `score_ranking(ranked_ids, relevant_ids, k=10)` validates inputs, uses the first `k`, counts unique relevant IDs, divides Recall by all hand-marked relevant IDs, and returns zero metrics for no relevant IDs.
- `evaluate_retrieval_question(question, result)` independently scores `dense`, `bm25`, and `fused` complaint-ID orderings from the same `RetrievalResult`.
- `run_retrieval_eval(con, questions, embed_model, eval_run_id, retriever=retrieve_variants)` invokes the retriever exactly once per question and upserts exactly three `rag_eval_results` rows inside one transaction per question.
- Persist component latency on its corresponding row. Preserve completed questions when a later question fails. Reject caller-owned transactions before retrieval/persistence, and do not leave partial method rows.
- Aggregate answerable questions for macro Recall@10/MRR; report unanswerable count separately. Win/tie/loss compares `(Recall@10, MRR)` lexicographically per question and includes every question, so unanswerable questions are visible ties rather than silently omitted.
- Summary/report includes answerable/unanswerable counts, macro values for every method, median/p95 latency, fused-vs-dense and fused-vs-BM25 win/tie/loss, and a stable per-question table. Never hide a fusion loss.
- Validate exact question IDs/method keys/ranking shapes; do not print narrative text. Do not depend on a committed real manifest in unit tests; the human freeze remains pending.
- Replays with the same `(eval_run_id, question_id, method)` are deterministic/idempotent. Do not erase later answer/human-review columns merely because retrieval metrics are rerun.

## TDD and verification

Use strict RED→GREEN. Cover metric denominator/rank/duplicates/top-k/invalid inputs; no-relevance behavior; single retriever call; all three rows; latency mapping; one-question transaction rollback; prior-question survival; retry/upsert without duplicate rows or erasing answer fields; caller transaction safety; aggregation exclusion/inclusion rules; lexicographic comparison; p95; and stable loss-visible rendering. Run Task 2 tests, full eval/retrieval/schema/isolation suites, full pytest, Ruff, scoped format, and diff check. Self-review, write `task-2-report.md`, update the ledger, commit, and leave a clean tree.
