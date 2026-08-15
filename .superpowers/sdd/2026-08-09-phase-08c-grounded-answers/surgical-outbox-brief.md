# Phase 8C surgical exception — paid-usage outbox durability

Authorization: the user explicitly approved this surgical outbox fix on 2026-08-11.

Baseline: `76cf81b6b580474bfb72aa183ced0e2066c3cab7`.

## Scope

Fix exactly the two load-bearing durability defects left by the Phase 8C final re-review:

1. `drain_usage_outbox(con, ...)` currently writes a staged usage event inside a caller-owned DuckDB transaction and unlinks the staged file before that transaction is durably committed. A later caller rollback can therefore lose both the database row and the only durable staged copy.
2. On first use, `_stage_usage_record(...)` creates the `usage_outbox` directory but does not fsync the parent directory that contains the new directory entry.

Do not alter answer semantics, prompt/cache identity, provider behavior, schema, retrieval, CLI output, or the parked Unicode-equivalent duplicate-claim Minor.

## Required behavior

- The public outbox drain must reject a caller-owned transaction before reading, inserting, or unlinking any staged event, while leaving that transaction usable by its caller.
- In normal autocommit operation, a successfully persisted event may be unlinked only after database persistence is durable from the drain's point of view.
- Database or validation failure must retain the staged event.
- Replay remains idempotent through `usage_id`.
- Creating the outbox directory for the first time must fsync its parent directory. The existing staged-file fsync, atomic replace, outbox-directory fsync, privacy-safe payload, and removal-directory fsync contracts must remain intact.

## TDD and verification

Add behavioral regressions before implementation for at least:

- begin transaction -> call drain -> rejection before mutation -> rollback/commit still usable -> staged file still present and no committed usage row;
- first-use outbox creation causes a parent-directory fsync in addition to the file/outbox-directory durability operations;
- successful autocommit replay deletes the event and remains idempotent;
- replay database failure leaves the event.

Run the focused regressions RED then GREEN, the complete answer/client suites, the full test suite, Ruff, formatter check for changed files, and `git diff --check`. Write `surgical-outbox-report.md`, commit the scoped change, and leave the worktree clean.
