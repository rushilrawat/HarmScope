"""Stage output assertions.

docs/ENGINEERING_NOTES.md trap T1: a stage completes, writes zero or all-null
rows, and reports success; downstream stages read the empty table and also
"succeed". The pipeline is green and produces nothing.

Every stage ends with explicit assertions on row count, null rate, and
distribution shape. These raise `CheckFailed` with the observed numbers — the
message has to be enough to debug from, so it always reports what was actually
seen, not just that something was wrong.

ponytail: there is no `--strict` flag. docs/ENGINEERING_NOTES.md specifies one,
default-on in CI. A flag that is always on is a config for a value that never
changes; these checks always raise. Recorded in docs/ENGINEERING_NOTES.md
under Reversed decisions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import duckdb

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CheckFailed(AssertionError):
    """A stage produced output that failed its own acceptance assertion."""


def _ident(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name


def expect_rows(
    con: duckdb.DuckDBPyConnection,
    table: str,
    *,
    min: int | None = None,
    max: int | None = None,
    where: str | None = None,
) -> int:
    """Assert a row count is inside an expected range. Returns the count.

    `where` is interpolated verbatim and is NOT validated — it must always be a
    literal written in this codebase, never anything derived from data or user
    input. `table` is validated by `_ident()`.
    """
    sql = f"SELECT count(*) FROM {_ident(table)}"  # noqa: S608 - see docstring
    if where:
        sql += f" WHERE {where}"
    n = con.execute(sql).fetchone()[0]
    scope = f"{table}" + (f" WHERE {where}" if where else "")
    if min is not None and n < min:
        raise CheckFailed(f"{scope}: {n:,} rows, expected at least {min:,}")
    if max is not None and n > max:
        raise CheckFailed(f"{scope}: {n:,} rows, expected at most {max:,}")
    return n


def expect_no_nulls(
    con: duckdb.DuckDBPyConnection, table: str, cols: Sequence[str]
) -> None:
    """Assert the named columns contain no NULLs."""
    table = _ident(table)
    for col in cols:
        n = con.execute(
            f"SELECT count(*) FROM {table} WHERE {_ident(col)} IS NULL"  # noqa: S608
        ).fetchone()[0]
        if n:
            total = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608
            raise CheckFailed(
                f"{table}.{col}: {n:,} NULLs out of {total:,} rows "
                f"({n / total:.1%})" if total else f"{table}.{col}: {n:,} NULLs"
            )


def expect_unique(
    con: duckdb.DuckDBPyConnection, table: str, cols: Sequence[str]
) -> None:
    """Assert the named columns are jointly unique."""
    table = _ident(table)
    key = ", ".join(_ident(c) for c in cols)
    dupes = con.execute(
        f"SELECT count(*) FROM (SELECT {key} FROM {table} "  # noqa: S608
        f"GROUP BY {key} HAVING count(*) > 1)"
    ).fetchone()[0]
    if dupes:
        raise CheckFailed(f"{table}({key}): {dupes:,} duplicated key values")


def expect_scalar(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    lo: float | None = None,
    hi: float | None = None,
    label: str = "value",
) -> float:
    """Assert a scalar query lands inside a range. Returns the value.

    Used for rates and distribution shape: narrative coverage fraction, mean
    redactions per document, campaign-flagged fraction per family.
    """
    row = con.execute(sql).fetchone()
    if row is None or row[0] is None:
        raise CheckFailed(f"{label}: query returned no value ({sql})")
    value = float(row[0])
    if lo is not None and value < lo:
        raise CheckFailed(f"{label}: {value:.6g}, expected at least {lo:.6g}")
    if hi is not None and value > hi:
        raise CheckFailed(f"{label}: {value:.6g}, expected at most {hi:.6g}")
    return value


def expect_no_leakage(
    con: duckdb.DuckDBPyConnection,
    table: str,
    cutoff,
    *,
    col: str = "as_of",
) -> None:
    """Assert no row in a point-in-time table was built from post-cutoff data.

    docs/EVALUATION.md §5 item 1. This is the check that makes the leakage
    guard a schema property rather than a discipline — it is cheap enough to
    run after every stage in a backtest refit, so run it there.
    """
    table, col = _ident(table), _ident(col)
    n = con.execute(
        f"SELECT count(*) FROM {table} WHERE {col} > ?", [cutoff]  # noqa: S608
    ).fetchone()[0]
    if n:
        worst = con.execute(f"SELECT max({col}) FROM {table}").fetchone()[0]  # noqa: S608
        raise CheckFailed(
            f"LEAKAGE: {table} has {n:,} rows with {col} > cutoff {cutoff} "
            f"(latest {worst}). Nothing after the cutoff may reach evaluation."
        )
