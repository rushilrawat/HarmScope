# SDD ledger — plan: docs/superpowers/plans/2026-08-09-phase-08d-rag-evaluation-documentation.md

Baseline: 6642764; Phase 8C accepted; 492 passed, 5 expected real-database leakage skips; Ruff/format/diff clean.

Preflight data state: the feature worktree is code-isolated and has no private populated database. The real ignored database is `/Users/rushilrawat/HarmScope/data/harmscope.duckdb` with 3,830,002 narratives, 73,122 clusters, and 24,641,664 cluster-member rows. Real-data commands must use an explicit `HARMSCOPE_DATA_DIR=/Users/rushilrawat/HarmScope/data` override; committed code remains only on `codex/phase-8-llm-layer`.

Human-ground-truth constraint: agents may implement the workflow and prepare a private draft worklist, but must not mark model/agent-authored relevance IDs or paraphrase review as human ground truth. A committed benchmark cannot be represented as human-reviewed until a person completes the review fields.

Task 1: in progress.
