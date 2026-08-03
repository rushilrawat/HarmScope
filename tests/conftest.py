from __future__ import annotations

from datetime import date

import pytest

from src import db
from src.ids import cluster_id


@pytest.fixture
def con(tmp_path):
    """An empty, schema-valid database on a throwaway path."""
    connection = db.connect(tmp_path / "test.duckdb")
    db.apply_schema(connection)
    yield connection
    connection.close()


@pytest.fixture
def seeded(con):
    """A database with one run and one cluster, so FK-bearing rows can be tested."""
    run_id = "0000000000001-abcdef01"
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'test', 'deadbeef', 'cafe', '{}', now(), 'ok')",
        [run_id],
    )
    cid = cluster_id(run_id, "mortgage", 3)
    con.execute(
        "INSERT INTO clusters (cluster_id, run_id, product_family, n_members, as_of) "
        "VALUES (?, ?, 'mortgage', 120, ?)",
        [cid, run_id, date(2019, 1, 1)],
    )
    return con, run_id, cid
