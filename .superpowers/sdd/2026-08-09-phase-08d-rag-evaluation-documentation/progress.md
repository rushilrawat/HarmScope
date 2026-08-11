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

Task 2: in progress.
