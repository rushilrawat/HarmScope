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
from pathlib import Path

import duckdb
import pytest

from src import db
from src.ids import COMPANY_TOTAL
from src.llm import verify


def test_phase8_tables_exist_on_a_fresh_database(con):
    """Fresh installs must expose the Phase 8 persistence contract."""
    required = {
        "llm_usage", "label_verifications", "rag_answers", "rag_eval_results"
    }
    assert required <= set(db.table_names(con))
    columns = {
        row[1] for row in con.execute("PRAGMA table_info('label_verifications')").fetchall()
    }
    assert "reviewer_origin" in columns


def test_phase8_migration_upgrades_a_pre_phase8_database(con):
    """Migration 007 must be repeatable when upgrading an existing database."""
    for table in ("rag_eval_results", "rag_answers", "label_verifications", "llm_usage"):
        con.execute("DROP TABLE IF EXISTS " + table)
    migration = Path("db/migrations/007_phase_08_llm_layer.sql").read_text()
    con.execute(migration)
    con.execute(migration)
    assert {
        "llm_usage", "label_verifications", "rag_answers", "rag_eval_results"
    } <= set(db.table_names(con))
    columns = {
        row[1] for row in con.execute("PRAGMA table_info('label_verifications')").fetchall()
    }
    assert "reviewer_origin" in columns


def test_phase8_provenance_migration_backfills_legacy_reviews_as_model(tmp_path):
    """Migration 008 must preserve old reviews without letting them pass a human gate."""
    legacy = duckdb.connect(str(tmp_path / "pre-provenance.duckdb"))
    legacy.execute(
        """
        CREATE TABLE runs (
          run_id VARCHAR PRIMARY KEY,
          phase VARCHAR NOT NULL,
          git_sha VARCHAR NOT NULL,
          config_hash VARCHAR NOT NULL,
          params_json VARCHAR NOT NULL,
          started_at TIMESTAMP NOT NULL,
          status VARCHAR NOT NULL
        );
        CREATE TABLE clusters (
          cluster_id VARCHAR PRIMARY KEY,
          run_id VARCHAR NOT NULL REFERENCES runs(run_id),
          product_family VARCHAR NOT NULL,
          n_members BIGINT NOT NULL,
          as_of DATE NOT NULL
        );
        CREATE TABLE cluster_labels (
          cluster_id VARCHAR PRIMARY KEY REFERENCES clusters(cluster_id),
          confidence VARCHAR
        );
        CREATE TABLE signals (
          run_id VARCHAR NOT NULL,
          cluster_id VARCHAR NOT NULL,
          q_value DOUBLE
        );
        CREATE TABLE label_verifications (
          cluster_id VARCHAR NOT NULL REFERENCES cluster_labels(cluster_id),
          reviewer_id VARCHAR NOT NULL,
          worklist_version VARCHAR NOT NULL,
          signals_run VARCHAR NOT NULL,
          is_fired BOOLEAN NOT NULL,
          mechanism_accuracy VARCHAR NOT NULL CHECK (
            mechanism_accuracy IN ('agree', 'partial', 'disagree')
          ),
          taxonomy_distinctness_accuracy VARCHAR NOT NULL CHECK (
            taxonomy_distinctness_accuracy IN ('agree', 'disagree')
          ),
          template_accuracy VARCHAR NOT NULL CHECK (
            template_accuracy IN ('agree', 'disagree')
          ),
          should_have_abstained BOOLEAN NOT NULL,
          failure_category VARCHAR NOT NULL CHECK (
            failure_category IN (
              'none', 'incoherent_cluster', 'overgeneralized', 'overspecific',
              'missed_submechanism', 'taxonomy_error', 'template_error',
              'unsupported_claim', 'other'
            )
          ),
          notes VARCHAR,
          reviewed_at TIMESTAMP NOT NULL,
          PRIMARY KEY (cluster_id, reviewer_id, worklist_version)
        );
        """
    )
    cluster_run = "0000000000001-cluster1"
    signals_run = "0000000000002-signal01"
    legacy.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'cluster', 'test', 'test', '{}', now(), 'ok')",
        [cluster_run],
    )
    legacy.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'signals', 'test', 'test', "
        "'{\"params\": {\"cluster_run\": \"0000000000001-cluster1\"}}', now(), 'ok')",
        [signals_run],
    )
    cluster_id = f"{cluster_run}:mortgage:0"
    legacy.execute(
        "INSERT INTO clusters (cluster_id, run_id, product_family, n_members, as_of) "
        "VALUES (?, ?, 'mortgage', 30, DATE '2020-01-01')",
        [cluster_id, cluster_run],
    )
    legacy.execute(
        "INSERT INTO cluster_labels (cluster_id, confidence) VALUES (?, 'high')",
        [cluster_id],
    )

    legacy.execute(
        "INSERT INTO label_verifications "
        "(cluster_id, reviewer_id, worklist_version, signals_run, is_fired, "
        "mechanism_accuracy, taxonomy_distinctness_accuracy, template_accuracy, "
        "should_have_abstained, failure_category, notes, reviewed_at) "
        "VALUES (?, 'legacy-reviewer', 'wl-v1', ?, false, 'agree', 'agree', "
        "'agree', false, 'none', NULL, now())",
        [cluster_id, signals_run],
    )

    migration = Path("db/migrations/008_label_verification_provenance.sql").read_text()
    legacy.execute(migration)

    assert legacy.execute(
        "SELECT reviewer_origin FROM label_verifications WHERE reviewer_id = 'legacy-reviewer'"
    ).fetchone() == ("model",)
    assert verify.report(legacy, "wl-v1").mechanism_total == 0
    with pytest.raises(duckdb.ConstraintException):
        legacy.execute(
            "UPDATE label_verifications SET reviewer_origin = 'robot' "
            "WHERE reviewer_id = 'legacy-reviewer'"
        )
    with pytest.raises(duckdb.ConstraintException):
        legacy.execute(
            "UPDATE label_verifications SET reviewer_origin = NULL "
            "WHERE reviewer_id = 'legacy-reviewer'"
        )
    legacy.execute(migration)
    assert legacy.execute("SELECT count(*) FROM label_verifications").fetchone() == (1,)
    legacy.close()


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
        "(run_id, cluster_id, company_id, period_month, n, denom, share, as_of) "
        "VALUES (?, ?, ?, ?, 40, 1000, 0.04, ?)",
        [_run_id, cid, COMPANY_TOTAL, date(2018, 5, 1), date(2019, 1, 1)],
    )
    got = con.execute(
        "SELECT company_id, n FROM cluster_timeseries WHERE cluster_id = ?", [cid]
    ).fetchall()
    assert got == [(COMPANY_TOTAL, 40)]


def test_null_company_id_is_rejected(seeded):
    """The sentinel is load-bearing, not decorative."""
    con, run_id, cid = seeded
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO cluster_timeseries "
            "(run_id, cluster_id, company_id, period_month, n, denom, share, as_of) "
            "VALUES (?, ?, NULL, ?, 40, 1000, 0.04, ?)",
            [run_id, cid, date(2018, 5, 1), date(2019, 1, 1)],
        )


def test_company_total_and_per_company_rows_coexist(seeded):
    con, run_id, cid = seeded
    rows = [
        (cid, COMPANY_TOTAL, date(2018, 5, 1), 40, 1000, 0.040),
        (cid, "co-equifax", date(2018, 5, 1), 25, 400, 0.0625),
        (cid, "co-experian", date(2018, 5, 1), 15, 600, 0.025),
    ]
    for r in rows:
        con.execute(
            "INSERT INTO cluster_timeseries "
            "(run_id, cluster_id, company_id, period_month, n, denom, share, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [run_id, *r, date(2019, 1, 1)],
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
        "(run_id, cluster_id, company_id, period_month, n, denom, share, as_of) "
        "VALUES (?, ?, ?, ?, 40, 1000, 0.04, ?)",
        [run_id, cid, COMPANY_TOTAL, period, asof],
    )
    con.execute(
        "INSERT INTO signals (signal_id, run_id, cluster_id, company_id, period_month, "
        "method, statistic, n_supporting, n_supporting_groups, as_of) "
        "VALUES ('s1', ?, ?, ?, ?, 'ewma', 4.2, 40, 38, ?)",
        [run_id, cid, COMPANY_TOTAL, period, asof],
    )
    joined = con.execute(
        "SELECT s.signal_id, t.n FROM signals s JOIN cluster_timeseries t "
        "USING (run_id, cluster_id, company_id, period_month)"
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


def test_architecture_doc_matches_the_real_schema():
    """ARCHITECTURE.md §4 says it is a defect when it disagrees with schema.sql.

    Nothing checked that. Migration 002 added five columns to `campaigns` and
    changed `campaign_members`' primary key; the doc kept the old listing for
    three commits, still describing a table that had not existed since Phase 1.
    Column names only — types and constraints are schema.sql's business, and a
    check that strict would fail on whitespace.
    """
    import re

    from src.config import PATHS

    def columns(sql: str) -> dict[str, set[str]]:
        out = {}
        for table, body in re.findall(
            r"CREATE TABLE (?:IF NOT EXISTS )?(\w+)\s*\((.*?)\n\);", sql, re.S
        ):
            names = set()
            for line in body.splitlines():
                line = line.split("--")[0].strip().rstrip(",")
                token = line.split()[0] if line else ""
                if token and token.upper() not in {
                    "PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT"
                }:
                    names.add(token)
                    # `first_seen DATE, last_seen DATE` shares a line.
                    for extra in re.findall(r",\s*(\w+)\s+\w+", line):
                        names.add(extra)
            out[table] = names
        return out

    real = columns((PATHS.root / "db" / "schema.sql").read_text())
    doc_sql = re.search(
        r"```sql\n(.*?)```", (PATHS.root / "docs" / "ARCHITECTURE.md").read_text(), re.S
    )
    assert doc_sql, "ARCHITECTURE.md §4 no longer contains a sql block"
    documented = columns(doc_sql.group(1))

    assert documented, "parsed no tables out of the ARCHITECTURE.md sql block"
    # Both directions. `redaction_stats` existed since Phase 0 and appeared in
    # no version of the doc, which the one-directional check could not see.
    assert set(real) == set(documented), (
        f"undocumented in ARCHITECTURE.md: {sorted(set(real) - set(documented))}; "
        f"documented but not in schema.sql: {sorted(set(documented) - set(real))}"
    )
    for table, doc_cols in documented.items():
        assert table in real, f"ARCHITECTURE.md documents table {table!r}, schema.sql has no such table"
        assert doc_cols == real[table], (
            f"{table}: ARCHITECTURE.md and db/schema.sql disagree. "
            f"only in the doc: {sorted(doc_cols - real[table])}; "
            f"missing from the doc: {sorted(real[table] - doc_cols)}"
        )
