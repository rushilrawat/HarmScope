# SDD ledger — plan: docs/superpowers/plans/2026-08-09-phase-08d-rag-evaluation-documentation.md

Baseline: 6642764; Phase 8C accepted; 492 passed, 5 expected real-database leakage skips; Ruff/format/diff clean.

Preflight data state: the feature worktree is code-isolated and has no private populated database. The real ignored database is `/Users/rushilrawat/HarmScope/data/harmscope.duckdb` with 3,830,002 narratives, 73,122 clusters, and 24,641,664 cluster-member rows. Real-data commands must use an explicit `HARMSCOPE_DATA_DIR=/Users/rushilrawat/HarmScope/data` override; committed code remains only on `codex/phase-8-llm-layer`.

Human-ground-truth constraint: agents may implement the workflow and prepare a private draft worklist, but must not mark model/agent-authored relevance IDs or paraphrase review as human ground truth. A committed benchmark cannot be represented as human-reviewed until a person completes the review fields.

Task 1 implementer: DONE_WITH_CONCERNS — strict manifest, exact DB/privacy
validation, deterministic private export/import, tests, and freeze documentation
implemented. Real ignored draft:
`/Users/rushilrawat/HarmScope/data/interim/rag_eval_authoring.csv` (30 blank
questions/reviews; 5/category; 15 fired/15 control; 30 distinct clusters; all 12
families; zero unsafe formula-prefix cells). Human author/privacy review and the
committed ID-only manifest remain intentionally pending. Verification: 58 passed
+ 1 expected focused skip; full 515 passed + 6 expected skips; Ruff, scoped
format, and diff check clean. Implementation committed with the Task 1 files.

Task 1: fix round 1/5 — four reviewer findings reproduced and addressed:
retrieval-identical scope, canonical alert status, exact CSV row arity, and
failure-atomic durable writes. Corrected real draft independently audits to 15
canonical fired / 15 canonical controls with zero mismatches. Focused
compatibility: 129 passed + 1 expected skip. Open human-curation plan concern:
`company_response` requires public-response evidence, but the mandated worklist
contains narrative excerpts only; no code change made pending plan resolution.
Final verification: focused 129 passed + 1 expected skip; full 526 passed + 6
expected skips; Ruff, scoped format, and diff check clean. The Task 1 fix commit
contains this ledger update.
Task 1 re-review: APPROVED at 4614378 — all four counterexamples fixed; 16 targeted tests and the complete suite pass; private draft independently confirms 15 canonical fired / 15 controls, 300/300 shown IDs retrievable, zero status mismatches, and blank human fields. Engineering implementation accepted. Human manifest freeze and the `company_response` plan contradiction remain open external/design gates.

Task 2 implementer: DONE — exact dense/BM25/fused scoring from one retrieval
call; per-question atomic three-row upserts; caller-autocommit rejection;
prior-question survival; replay preservation of answer/human columns and
created-at identity; answerable-only macros; all-question lexicographic
win/tie/loss; stable ID-only loss-visible rendering with linear p95. Strict TDD:
20 expected missing-interface RED failures, then 25 GREEN; 2 relevance-invariant
RED failures, then final 27 GREEN. Verification: focused 27 passed; retrieval/
schema/isolation slice 149 passed; full 548 passed + 6 expected skips; Ruff,
scoped format, and diff check clean. Implementation commit pending review gate.

Task 2 review at 25e5387: FIXES_REQUIRED — Important: independent fused-field
validation admitted impossible fused wins because membership, component
provenance, RRF score/order, and configured result length were not tied to the
component rankings. Minor: unsafe nonblank `eval_run_id` values were persisted.

Task 2: fix round 1/5 — both findings reproduced with 14 behavioral RED
counterexamples. Fused validation now requires exact output from the shared
`retrieve.reciprocal_rank_fusion` implementation using configured RRF/top-k;
`eval_run_id` now matches `_SAFE_ID` before retrieval. Fake results also use the
real fusion function, and the report demonstrates valid fusion losses. GREEN:
new regressions 19 passed; final Task 2 selection 32 passed; compatibility
slice 163 passed; full 562 passed + 6 expected skips; Ruff, scoped format, and
diff check clean. Fix commit pending scoped re-review.

Task 2 scoped re-review after bf342fc: Important fused-provenance finding
ADDRESSED; Minor run-ID shape finding functionally addressed, but residual
Minor sequencing issue remained because invalid input executed two read-only
autocommit-probe queries before validation.

Task 2: fix round 2/5 — reproduced empty-batch, unsafe-run-ID, and duplicate-
question inputs against a zero-database spy; all three RED cases touched SQL
before their validation error. Pure batch validation now precedes the
autocommit probe. Valid explicit caller transactions still reject before
retrieval/persistence without altering caller state. GREEN: targeted 5 passed;
evaluation 73 passed; compatibility slice 166 passed; full 565 passed + 6
expected skips; Ruff, scoped format, and diff check clean. Fix commit pending
scoped re-review.
Task 2 re-review: APPROVED at 6a8bf4d — zero-I/O invalid inputs, caller-transaction safety, exact fused membership/provenance/score/order/top-k validation, rollback, replay preservation, aggregates, and loss-visible rendering all reproduced with no residual finding.

Task 3: in progress.
