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


def test_cluster_level_signal_joins_its_timeseries_total(seeded):
    """`signals` and `cluster_timeseries` must agree on how "all companies" is
    spelled. If one used NULL and the other '__ALL__', this join would return
    nothing for exactly the cluster-level rows — a missing alert, silently."""
    con, run_id, cid = seeded
    period, asof = date(2018, 5, 1), date(2019, 1, 1)
    con.execute(
        "INSERT INTO cluster_timeseries "
        "(cluster_id, company_id, period_month, n, denom, share, as_of) "
        "VALUES (?, ?, ?, 40, 1000, 0.04, ?)",
        [cid, COMPANY_TOTAL, period, asof],
    )
    con.execute(
        "INSERT INTO signals (signal_id, run_id, cluster_id, company_id, period_month, "
        "method, statistic, n_supporting, n_supporting_groups, as_of) "
        "VALUES ('s1', ?, ?, ?, ?, 'ewma', 4.2, 40, 38, ?)",
        [run_id, cid, COMPANY_TOTAL, period, asof],
    )
    joined = con.execute(
        "SELECT s.signal_id, t.n FROM signals s JOIN cluster_timeseries t "
        "USING (cluster_id, company_id, period_month)"
    ).fetchall()
    assert joined == [("s1", 40)]


def test_signals_reject_null_company_id(seeded):
    con, run_id, cid = seeded
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO signals (signal_id, run_id, cluster_id, company_id, "
            "period_month, method, statistic, n_supporting, n_supporting_groups, as_of) "
            "VALUES ('s2', ?, ?, NULL, DATE '2018-05-01', 'prr', 3.0, 10, 9, "
            "DATE '2019-01-01')",
            [run_id, cid],
        )


def test_dup_groups_hold_one_row_per_run_and_complaint(seeded):
    """Grouping is refit per cutoff (transitive closure is date-dependent), so
    the same complaint appears once per run, not once overall."""
    con, run_id, _cid = seeded
    other_run = "0000000000002-beefcafe"
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'test', 'x', 'y', '{}', now(), 'ok')",
        [other_run],
    )
    for rid, size in ((run_id, 3), (other_run, 5)):
        con.execute(
            "INSERT INTO dup_groups (run_id, complaint_id, group_id, "
            "is_representative, group_size, as_of) "
            "VALUES (?, 42, 'g1', true, ?, DATE '2019-01-01')",
            [rid, size],
        )
    sizes = con.execute(
        "SELECT group_size FROM dup_groups WHERE complaint_id = 42 ORDER BY run_id"
    ).fetchall()
    assert sizes == [(3,), (5,)]


def test_schema_drift_is_detected(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` cannot alter an existing table, so a stale
    database must be caught at bootstrap rather than three stages later as a
    BinderError about a column schema.sql clearly declares."""
    stale = tmp_path / "stale.duckdb"
    con = db.connect(stale)
    db.apply_schema(con)
    con.execute("ALTER TABLE taxonomy_crosswalk DROP COLUMN era")

    with pytest.raises(db.SchemaDrift, match="era"):
        db.check_schema_drift(con)
    con.close()


def test_fresh_database_has_no_drift(tmp_path):
    con = db.connect(tmp_path / "fresh.duckdb")
    db.apply_schema(con)
    db.check_schema_drift(con)  # must not raise
    con.close()


def test_run_status_is_constrained(con):
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
            "started_at, status) VALUES ('x', 'p', 's', 'c', '{}', now(), 'finished')"
        )
