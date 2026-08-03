"""DuckDB connection and the run registry.

The run registry is the countermeasure for trap T1 (silent success), and it is
built so that the failure mode is structurally hard rather than merely
discouraged:

  - A stage body that exits without calling `run.finish(output_rows=...)`
    raises `SilentSuccess`. You cannot accidentally "complete" a stage that
    produced nothing.
  - The registry writes on its own DuckDB cursor, which has an independent
    transaction context. A stage that opens a transaction and fails rolls back
    its own output but NOT its failure record. A provenance table that only
    ever logs successes is T1 wearing a disguise.

Every run records the git SHA (with a `-dirty` suffix when the working tree
had uncommitted changes) and the config fingerprint. A result produced from a
dirty tree is not reproducible and the `runs` row says so.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from src.config import CONFIG, PATHS, Config


class SilentSuccess(RuntimeError):
    """A stage exited cleanly without declaring what it produced."""


def new_run_id() -> str:
    """Lexicographically time-sortable unique id.

    ponytail: docs/ARCHITECTURE.md says "ulid". This has the two properties
    that are actually load-bearing (sortable by creation time, collision-free)
    with no dependency. Swap in `python-ulid` if the canonical 26-char format
    is ever needed externally.
    """
    return f"{int(time.time() * 1000):013d}-{secrets.token_hex(4)}"


def git_sha(root: Path | None = None) -> str:
    """Current commit, suffixed `-dirty` when the tree has uncommitted changes."""
    root = root or PATHS.root

    def _git(*args: str) -> str | None:
        try:
            # noqa S603/S607: `args` is always a literal from this module, and git
            # is resolved from PATH on purpose — pinning an absolute path would
            # break every environment that installs it somewhere else.
            out = subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607
                cwd=root, capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return "no-commit"
    dirty = _git("status", "--porcelain")
    return f"{sha}-dirty" if dirty else sha


def connect(path: Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = path or PATHS.db
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def apply_schema(con: duckdb.DuckDBPyConnection, schema_sql: Path | None = None) -> None:
    """Apply db/schema.sql. Idempotent — every statement is CREATE ... IF NOT EXISTS."""
    con.execute((schema_sql or PATHS.schema_sql).read_text())


def _columns(con: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    rows = con.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
    ).fetchall()
    out: dict[str, list[str]] = {}
    for table, column in rows:
        out.setdefault(table, []).append(column)
    return out


class SchemaDrift(RuntimeError):
    """The database on disk does not match db/schema.sql."""


def check_schema_drift(
    con: duckdb.DuckDBPyConnection, schema_sql: Path | None = None
) -> None:
    """Fail loudly when an existing database predates a schema change.

    Every statement in schema.sql is `CREATE TABLE IF NOT EXISTS`, which makes
    bootstrap idempotent but also makes it silently skip a table whose
    definition has changed. The symptom is a BinderError three stages later
    complaining about a column that schema.sql clearly declares.

    Rather than parse the DDL, apply it to a throwaway in-memory database and
    diff `information_schema` — exact, and it cannot drift from the file.
    """
    reference = duckdb.connect()
    try:
        apply_schema(reference, schema_sql)
        expected = _columns(reference)
    finally:
        reference.close()

    live = _columns(con)
    problems: list[str] = []
    for table, cols in expected.items():
        if table not in live:
            problems.append(f"  {table}: missing entirely")
            continue
        missing = [c for c in cols if c not in live[table]]
        extra = [c for c in live[table] if c not in cols]
        if missing:
            problems.append(f"  {table}: missing columns {missing}")
        if extra:
            problems.append(f"  {table}: unexpected columns {extra}")

    if problems:
        raise SchemaDrift(
            "database does not match db/schema.sql:\n"
            + "\n".join(problems)
            + "\n\nCREATE TABLE IF NOT EXISTS cannot alter an existing table. "
            "Add a migration under db/migrations/, or rebuild from scratch:\n"
            "  make clean && make init"
        )


def bootstrap(path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Create the data directories and an empty, schema-valid database."""
    PATHS.ensure()
    con = connect(path)
    apply_schema(con)
    check_schema_drift(con)
    return con


def table_names(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


@dataclass
class Run:
    """Handle for one pipeline stage execution."""

    run_id: str
    phase: str
    started_at: datetime
    _output_rows: int | None = field(default=None, repr=False)
    _input_rows: int | None = field(default=None, repr=False)

    def finish(self, output_rows: int, input_rows: int | None = None) -> None:
        """Declare what this stage produced. Required; omitting it is an error."""
        if self._output_rows is not None:
            raise RuntimeError(f"finish() already called for run {self.run_id}")
        if output_rows < 0:
            raise ValueError("output_rows must be non-negative")
        self._output_rows = output_rows
        self._input_rows = input_rows

    @property
    def declared(self) -> bool:
        return self._output_rows is not None


@contextmanager
def run(
    con: duckdb.DuckDBPyConnection,
    phase: str,
    config: Config = CONFIG,
    params: dict[str, Any] | None = None,
) -> Iterator[Run]:
    """Register a stage execution, and refuse to record it as a success unless
    the stage says what it produced.

    Usage:
        with run(con, "ingest.load") as r:
            n = do_work(con)
            r.finish(output_rows=n)
    """
    registry = con.cursor()  # independent transaction context — see module docstring
    handle = Run(run_id=new_run_id(), phase=phase, started_at=datetime.now())

    payload = {"config": config.to_dict()}
    if params:
        payload["params"] = params

    registry.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, ?, ?, ?, ?, ?, 'running')",
        [handle.run_id, phase, git_sha(), config.fingerprint,
         json.dumps(payload, sort_keys=True, default=str), handle.started_at],
    )

    def _close(status: str, error: str | None = None) -> None:
        registry.execute(
            "UPDATE runs SET status = ?, error = ?, finished_at = ?, "
            "input_rows = ?, output_rows = ? WHERE run_id = ?",
            [status, error, datetime.now(), handle._input_rows,
             handle._output_rows, handle.run_id],
        )

    try:
        yield handle
    except BaseException as exc:
        _close("failed", f"{type(exc).__name__}: {exc}")
        raise

    if not handle.declared:
        _close("failed", "stage exited without calling finish()")
        raise SilentSuccess(
            f"run {handle.run_id} ({phase}) exited without calling "
            f"finish(output_rows=...). A stage that completed without "
            f"declaring its output is a defect, not a pass."
        )
    _close("ok")
