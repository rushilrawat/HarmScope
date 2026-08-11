# Phase 8D Task 1 — reproducible RAG benchmark contract and authoring workflow

Baseline: `664276400632aaab201d48d288eaa80f0e52dc23`.

Governing plan: `docs/superpowers/plans/2026-08-09-phase-08d-rag-evaluation-documentation.md`, Task 1 and Global Constraints.

## Goal

Implement the strict CSV manifest, real-database scope/privacy validation, and deterministic private authoring/import workflow for a thirty-question, six-category RAG benchmark.

## Files

- Create `src/llm/eval.py`.
- Create `tests/test_llm_eval.py`.
- Modify `tests/test_ground_truth.py`.
- Modify `data/ground_truth/README.md`.
- Create `data/ground_truth/rag_eval_questions.csv` only if its relevant IDs and paraphrase review have actually been completed by a human; otherwise prepare a gitignored draft under `data/interim/` and report the human gate explicitly.

## Non-negotiable contracts

- `EvalQuestion` and the public functions/signatures in Task 1 match the plan.
- Strict seven-column CSV parser; exactly 30 by default; exactly five per declared category; stable unique IDs; exact boolean and ascending unique semicolon-ID parsing; answerable/relevance invariants.
- Database validation proves cluster existence, optional company membership, and every relevant complaint belongs to the exact cluster/company scope.
- Privacy validation uses `src.llm.retrieve.tokenize` and rejects every normalized eight-token overlap without printing shared text.
- Export is deterministic for a fixed seed, balances product family and fired/control state as the available population permits, provides ten redacted excerpts, and writes only to an ignored interim worklist.
- Import requires exactly five completed rows per category and `privacy_reviewed=yes`, strips every narrative/helper column, revalidates scope/privacy, and writes only the seven committed columns in stable question-ID order.
- No narrative text enters Git, logs, exceptions, or committed fixtures.
- Detection isolation remains intact: no detection package imports or reads Phase 8 evaluation code/tables.

## Human-ground-truth boundary

Do not invent human provenance. Agent-authored questions/relevance IDs may be prepared only as a private draft. Do not set `privacy_reviewed=yes`, do not commit those labels as ground truth, and do not weaken tests or validation to bypass the human step. Report the exact artifact path and the minimum human action still required.

## TDD and verification

Use strict RED→GREEN. Cover malformed manifests, distribution, scope, company scope, duplicate/out-of-scope relevant IDs, eight-token overlap, no-text error messages, deterministic export, balancing/capping, import column stripping, review requirement, stable output, CSV formula neutralization where private worklist fields can be opened in a spreadsheet, and detection isolation. Run Task 1 tests, ground-truth/schema/isolation suites, full pytest, Ruff, scoped format, and diff check. Self-review, write `task-1-report.md`, commit the implementation, and leave a clean tree. If the only incomplete item is the human-authored manifest, return `DONE_WITH_CONCERNS` rather than fabricating it.
