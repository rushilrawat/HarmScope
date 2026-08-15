# Phase 8C surgical outbox implementation report

Status: DONE

Baseline: `76cf81b6b580474bfb72aa183ced0e2066c3cab7`

## Root cause

`drain_usage_outbox` called `_record_usage` without first establishing that the
DuckDB connection was in autocommit mode. In an explicit caller transaction,
the insert therefore remained provisional while `_remove_staged_usage` deleted
and fsynced the only staged copy. A later caller rollback lost the accounting
event from both stores.

`_stage_usage_record` also used `mkdir(..., exist_ok=True)` and then synced only
the staged file and outbox directory. When `usage_outbox` was new, the directory
entry in `usage_outbox.parent` had not been synced.

## TDD evidence

RED command:

```text
.venv/bin/pytest -q tests/test_llm_answer.py -k 'usage_outbox_drain_rejects_caller_transaction or paid_usage_outbox_is_fsynced_atomic_and_privacy_safe or usage_outbox_replay_database_failure_retains_event'
```

Baseline result: 4 expected failures and 1 pass. Both caller-resolution cases
failed because no `AnswerTransactionError` was raised; the malformed event was
read and raised `UsageOutboxError` before transaction rejection; and the parent
directory was absent from the observed fsync targets. The database-failure
retention characterization already passed.

GREEN command: the same focused command.

Result: `5 passed` (exit 0).

## Implementation

- `drain_usage_outbox` now calls the existing read-only `_require_autocommit`
  before resolving or reading the outbox. An explicit caller transaction is
  rejected without insert, unlink, commit, or rollback, and remains usable by
  the caller. In autocommit mode, DuckDB commits `_record_usage` when its
  statement completes, before staged-file removal.
- `_stage_usage_record` establishes that the outbox path is a directory and
  fsyncs `directory.parent` before opening an event file. Repeating the parent
  sync for an existing directory repairs a failed first sync and closes the
  concurrent-creator window, while the existing temp-file fsync, atomic replace,
  outbox-directory fsync, and removal-directory fsync remain unchanged.
- Behavioral tests cover both caller COMMIT and ROLLBACK, rejection before a
  malformed event is read, replay database-failure retention, first-use parent
  and child directory fsync attribution, and the pre-existing successful
  idempotent replay path.

## Verification

```text
.venv/bin/pytest -q tests/test_llm_answer.py tests/test_llm_client.py
```

Result: `121 passed in 1.73s` (exit 0).

```text
.venv/bin/pytest -q
```

Result: `491 passed, 5 skipped in 27.19s` (exit 0). The skips are the
pre-existing leakage checks that require a real database run.

```text
.venv/bin/ruff check src/llm/answer.py tests/test_llm_answer.py
.venv/bin/ruff format --check src/llm/answer.py tests/test_llm_answer.py
git diff --check
```

Result: Ruff reported `All checks passed!`; both changed files were already
formatted; `git diff --check` produced no findings.

## Self-review

- Scope is limited to `src/llm/answer.py`, `tests/test_llm_answer.py`, and the
  authorized SDD artifacts. Answer semantics, provider/cache identity, schema,
  retrieval, CLI output, and Unicode claim normalization are unchanged.
- The public guard runs before even the outbox existence check, so an empty or
  malformed outbox cannot bypass transaction ownership enforcement.
- Insert failure still propagates before removal, retaining the staged event.
  `usage_id` conflict handling remains unchanged, so replay stays idempotent.
- The initial self-review did not identify the failed-parent-fsync recovery
  window corrected in Fix round 1 below.

## Fix round 1 — failed parent-fsync recovery

Reviewer finding: the initial implementation synced the parent only in the
successful `mkdir` branch. If that fsync raised after creating the directory,
the empty directory remained; a retry took the `FileExistsError` branch and
staged an event without retrying the parent sync. A concurrent creator could
enter the same existing-directory path before the creator completed its sync.

RED regression:

```text
.venv/bin/pytest -q tests/test_llm_answer.py -k 'retry_repairs_failed_parent_directory_fsync'
```

Result: failed because the retry staged its event while the recorded parent
fsync count remained `1`, not `2`.

Fix: move `_fsync_directory(directory.parent)` after the directory
existence/type branch so every staging attempt performs it before opening or
writing the event.

GREEN and verification:

```text
.venv/bin/pytest -q tests/test_llm_answer.py -k 'retry_repairs_failed_parent_directory_fsync'
# 1 passed
.venv/bin/pytest tests/test_llm_answer.py -k 'usage_outbox or paid_usage_outbox or outbox_replay' -ra
# 7 passed, 105 deselected
.venv/bin/pytest tests/test_llm_answer.py tests/test_llm_client.py -ra
# 122 passed in 2.17s
.venv/bin/pytest -ra
# 492 passed, 5 skipped in 24.10s
.venv/bin/ruff check .
# All checks passed!
.venv/bin/ruff format --check src/llm/answer.py tests/test_llm_answer.py
# 2 files already formatted
git diff --check
# no findings
```

The five skips remain the pre-existing leakage checks that require a real
database. Transaction rejection, privacy, idempotence, and failure retention
remain covered and unchanged.
