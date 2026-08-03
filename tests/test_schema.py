"""Schema-level regression tests.

The cluster_timeseries test is the one that matters: docs/ARCHITECTURE.md
originally declared `PRIMARY KEY (cluster_id, company_id, period_month)` with a
comment saying a NULL company_id row carries the cluster total across
companies. DuckDB enforces NOT NULL on primary-key columns, so every one of
those total rows would have failed to insert at Phase 5 — after clustering, and
after several hours of embedding.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from src import db
from src.ids import COMPANY_TOTAL


def test_schema_applies_and_is_idempotent(tmp_path):
    con = db.connect(tmp_path / "a.duckdb")
    db.apply_schema(con)
    first = db.table_names(con)
    db.apply_schema(con)  # every statement is CREATE ... IF NOT EXISTS
    assert db.table_names(con) == first
    assert {"runs", "complaints", "narratives", "clusters", "signals"} <= set(first)


def test_cluster_total_row_inserts_with_sentinel(seeded):
    con, _run_id, cid = seeded
    con.execute(
        "INSERT INTO cluster_timeseries "
        "(cluster_id, company_id, period_month, n, denom, share, as_of) "
        "VALUES (?, ?, ?, 40, 1000, 0.04, ?)",
        [cid, COMPANY_TOTAL, date(2018, 5, 1), date(2019, 1, 1)],
    )
    got = con.execute(
        "SELECT company_id, n FROM cluster_timeseries WHERE cluster_id = ?", [cid]
    ).fetchall()
    assert got == [(COMPANY_TOTAL, 40)]


def test_null_company_id_is_rejected(seeded):
    """The sentinel is load-bearing, not decorative."""
    con, _run_id, cid = seeded
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO cluster_timeseries "
            "(cluster_id, company_id, period_month, n, denom, share, as_of) "
            "VALUES (?, NULL, ?, 40, 1000, 0.04, ?)",
            [cid, date(2018, 5, 1), date(2019, 1, 1)],
        )


def test_company_total_and_per_company_rows_coexist(seeded):
    con, _run_id, cid = seeded
    rows = [
        (cid, COMPANY_TOTAL, date(2018, 5, 1), 40, 1000, 0.040),
        (cid, "co-equifax", date(2018, 5, 1), 25, 400, 0.0625),
        (cid, "co-experian", date(2018, 5, 1), 15, 600, 0.025),
    ]
    for r in rows:
        con.execute(
            "INSERT INTO cluster_timeseries "
            "(cluster_id, company_id, period_month, n, denom, share, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [*r, date(2019, 1, 1)],
        )
    total = con.execute(
        "SELECT n FROM cluster_timeseries WHERE company_id = ?", [COMPANY_TOTAL]
    ).fetchone()[0]
    per_company = con.execute(
        "SELECT sum(n) FROM cluster_timeseries WHERE company_id != ?", [COMPANY_TOTAL]
    ).fetchone()[0]
    assert total == per_company == 40


def test_run_status_is_constrained(con):
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
            "started_at, status) VALUES ('x', 'p', 's', 'c', '{}', now(), 'finished')"
        )
