# SDD ledger — plan: docs/superpowers/plans/2026-08-09-phase-08c-grounded-answers.md

Baseline: 6cb12f0; 372 passed, 5 skipped; Ruff and scoped format/diff checks clean.

Preflight: retrieval is accepted with no residual Critical/Important findings. One parked Phase 8B Minor remains: concurrent readers can race between corrupt-cache `exists()` and `read_text()`.
Task 1: fix round 1/5 (2 addressed, 0 open — unforgeable JSON prompt envelope; abstention requires a limitation; commits 104a4db..1de7b94).
Task 1: complete (commits 6cb12f0..1de7b94, re-review approved; no open findings).
Task 2: complete (commits 1de7b94..615c759, review approved clean; no findings).
Task 3: plan-level cache identity contradiction resolved before review — enforcement-enabled calls use a canonical context digest in the effective prompt-version identity.
Task 3: fix round 1/5 (3 addressed, 0 open — caller transaction safety; retriever scope/order contract; typed paid refusal accounting; commits c1affec..5b920ba).
Task 3: complete (commits 615c759..5b920ba, re-review approved; no open findings).
Task 4: fix round 1/5 (1 addressed, 0 open — preserve safe Unicode while neutralizing terminal/bidi display controls; commits acf6916..daba74b).
Task 4: complete (commits 5b920ba..daba74b, re-review approved; no open findings).
Final review: fixes required — deterministic cited synthesis/reason codes; SDK pin compatibility; cluster-derived embed model; cluster provenance; durable paid-usage fallback; display-only context boundary; cache-status/connection/nullable-summary minors.
Final fix wave: complete (commit 76cf81b) — Critical, SDK, model/provenance, context boundary, and original Minor findings addressed; 487 passed, 5 skipped; Ruff/format/diff clean.
Final scoped re-review: RESIDUAL — no Critical; one load-bearing Important remains in usage-outbox reconciliation.
Residual Important (BLOCKING): `drain_usage_outbox` can insert and unlink inside an existing caller transaction, then caller rollback loses both DB row and staged file; first-use outbox directory creation also lacks parent-directory fsync.
Residual Minor (parkable): claim duplicate normalization does not canonicalize Unicode or zero-width variants; all variants remain complaint-cited.
Phase 8C completion gate: BLOCKED at 76cf81b by durable paid-accounting residual. Per the one-final-fix-wave breaker, do not start Phase 8D without human authorization for a surgical exception.
Surgical exception: AUTHORIZED by the user on 2026-08-11, limited to transaction-safe outbox draining and first-use parent-directory fsync. Baseline remains 76cf81b.
