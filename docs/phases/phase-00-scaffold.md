# Phase 0 — Scaffold

**State:** ✅ complete · **Effort:** ~2 days

## Question

Can a stage of this pipeline fail without anyone noticing? Everything after this
phase is run-scoped and expensive, so a stage that "completed" while producing
nothing has to be an error rather than a quiet zero.

## Context

The failure mode this project keeps rediscovering is not a crash — it is a
plausible, well-formed, entirely wrong number. Phase 5's negative control, the
Phase 4 ablation bug, and the Phase 6 `as_of` leak all produced output that
looked fine. The scaffold exists to make that class of failure loud by default:
every stage registers itself, declares what it produced, and asserts on its own
output.

Standing rule 2 in the README — "no silent success" — is implemented here, not
aspired to.

## What runs

1. `src/config.py` — one frozen dataclass tree; every science parameter lives in
   it and nowhere else.
2. `db/schema.sql` — 22 tables, applied to an empty DuckDB file.
3. `src/db.py` — the `runs` registry as a context manager.
4. `src/checks.py` — row-count, null-rate and distribution assertions callable
   from any stage.
5. CI — ruff + pytest on push.

## Tech

| Choice | Why this one |
|---|---|
| **DuckDB 1.5.5**, single file | Analytical, no server, handles 16.9M rows on a laptop. The cost, discovered later, is single-writer locking — a long refit blocks even a read-only connect. |
| **Frozen dataclasses** for config | `config_hash` is a sha256 over every science parameter, recorded on every run. A threshold that changes mid-project is visible as a fingerprint mismatch. |
| **Context-manager run registry** | `with db.run(con, phase, CONFIG) as r:` writes a `running` row, then `ok` or `failed`. Exiting without `r.finish(output_rows=…)` raises. |
| **ruff + black**, pre-commit | Cheap, and keeps diffs about content. |

## Acceptance

> `make init && make test` on a clean clone produces an empty, schema-valid DuckDB.

**Met.**

## Findings

**A schema bug found before any data existed.** `cluster_timeseries.company_id`
was specified to carry a NULL "all companies" marker row. DuckDB enforces NOT
NULL on primary-key columns, so that row could never have been inserted. The
`'__ALL__'` sentinel in `src/ids.py` exists because of this.

Had it not been caught at scaffold time it would have surfaced at Phase 5 —
after the embedding hours were already spent.

**The registry raises rather than records a silent success.** A stage that exits
its `with` block without declaring `output_rows` is a bug, not a pass. This is
the mechanism that later made "which run produced this number" answerable at all,
which is what the Phase 6 audit depended on.
