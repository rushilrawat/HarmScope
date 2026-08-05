"""Run registry behaviour — the trap T1 countermeasure."""

from __future__ import annotations

import pytest

from src import db
from src.config import CONFIG


def _status(con, run_id):
    return con.execute(
        "SELECT status, error, output_rows FROM runs WHERE run_id = ?", [run_id]
    ).fetchone()


def test_successful_run_is_recorded_ok(con):
    with db.run(con, "unit", CONFIG) as r:
        r.finish(output_rows=7, input_rows=9)
    status, error, out = _status(con, r.run_id)
    assert (status, error, out) == ("ok", None, 7)


def test_run_records_git_sha_and_config_fingerprint(con):
    with db.run(con, "unit", CONFIG) as r:
        r.finish(output_rows=1)
    sha, cfg = con.execute(
        "SELECT git_sha, config_hash FROM runs WHERE run_id = ?", [r.run_id]
    ).fetchone()
    assert sha
    assert cfg == CONFIG.fingerprint


def test_stage_that_never_declares_output_fails_loudly(con):
    """A stage that 'completed' without producing valid output is a defect."""
    with pytest.raises(db.SilentSuccess), db.run(con, "unit", CONFIG) as r:
        pass  # forgot finish()
    status, error, _ = _status(con, r.run_id)
    assert status == "failed"
    assert "finish()" in error


def test_exception_is_recorded_and_reraised(con):
    with pytest.raises(ValueError, match="boom"), db.run(con, "unit", CONFIG) as r:
        raise ValueError("boom")
    status, error, _ = _status(con, r.run_id)
    assert status == "failed"
    assert error == "ValueError: boom"


def test_failure_record_survives_stage_transaction_rollback(con):
    """The registry writes on its own cursor.

    If the runs row were written inside the stage's transaction, a stage
    failure would roll back its own failure record, and the provenance table
    would only ever log successes.
    """
    with pytest.raises(RuntimeError), db.run(con, "unit", CONFIG) as r:
        con.execute("BEGIN")
        con.execute(
            "INSERT INTO enforcement_actions (action_id, filed_date, usable) "
            "VALUES ('a1', DATE '2020-01-01', true)"
        )
        raise RuntimeError("stage blew up mid-transaction")

    con.execute("ROLLBACK")
    assert con.execute("SELECT count(*) FROM enforcement_actions").fetchone()[0] == 0
    status, error, _ = _status(con, r.run_id)
    assert status == "failed"
    assert "stage blew up" in error


def test_finish_twice_is_an_error(con):
    with pytest.raises(RuntimeError, match="already called"), db.run(con, "unit", CONFIG) as r:
        r.finish(output_rows=1)
        r.finish(output_rows=2)


def test_run_ids_are_unique():
    ids = [db.new_run_id() for _ in range(1000)]
    assert len(set(ids)) == 1000


def test_run_ids_sort_by_creation_time():
    import time

    earlier = db.new_run_id()
    time.sleep(0.005)
    later = db.new_run_id()
    assert earlier < later


def test_runs_listing_marks_truncated_runs(con, capsys):
    """A --limit smoke run must not read as a full run of the phase.

    The embed smoke test wrote a runs row identical in shape to the real thing —
    same phase, same config_hash, differing only in output_rows, which nothing
    interprets as "this was a test".
    """
    import argparse

    from src import pipeline

    for run_id, params, rows in (
        ("0000000000001-aaaaaaaa", '{"model": "m", "limit": 5000}', 7822),
        ("0000000000002-bbbbbbbb", '{"model": "m", "limit": null}', 2477937),
    ):
        con.execute(
            "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
            "started_at, finished_at, status, output_rows) "
            "VALUES (?, 'embed', 'sha', 'cfg', ?, now(), now(), 'ok', ?)",
            [run_id, params, rows],
        )

    original = pipeline.db.connect
    pipeline.db.connect = lambda **kw: con
    try:
        pipeline.cmd_runs(argparse.Namespace(n=10))
    finally:
        pipeline.db.connect = original

    out = capsys.readouterr().out
    assert "embed*" in out, "the truncated run is not marked"
    assert "--limit" in out, "no legend explaining the marker"
    # The full run must NOT be marked.
    assert [ln for ln in out.splitlines() if "2477937" in ln or "2,477,937" in ln]
    assert not any(
        "embed*" in ln and "2477937" in ln for ln in out.splitlines()
    ), "the full run was marked as truncated"
