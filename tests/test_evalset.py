"""Scoring the Phase 2 gate.

The gate decides whether every downstream result is trustworthy, so the way it
can fail that matters most is not "wrong number" but "wrong number that reads
as a pass". Both tests here are for that failure mode.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from src.dedup import evalset


def _groups(con, run_id: str, assignment: dict[int, str]) -> None:
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'dedup', 'sha', 'cfg', '{}', now(), 'ok') "
        "ON CONFLICT DO NOTHING",
        [run_id],
    )
    con.executemany(
        "INSERT INTO dup_groups (run_id, complaint_id, group_id, "
        "is_representative, group_size, as_of) VALUES (?, ?, ?, true, 1, ?)",
        [(run_id, cid, gid, date(2020, 1, 1)) for cid, gid in assignment.items()],
    )


def test_score_is_scoped_to_one_run(con):
    """Two runs, opposite groupings. Unscoped, the pair looks split in both."""
    _groups(con, "run_a", {1: "g1", 2: "g1"})   # run A merged them
    _groups(con, "run_b", {1: "g1", 2: "g2"})   # run B did not
    rows = [
        {"complaint_id_a": 1, "complaint_id_b": 2, "label": "dup"},
        {"complaint_id_a": 1, "complaint_id_b": 2, "label": "not_dup"},
    ]
    assert evalset.score(con, rows, "run_a")["precision"] == 0.5
    with pytest.raises(ValueError, match="degenerate"):
        evalset.score(con, rows, "run_b")  # no predicted merges at all


def test_missing_complaint_is_not_a_merge(con):
    """One id present, one absent: the distinct-group count is trivially 1."""
    _groups(con, "run_a", {1: "g1", 3: "g3", 4: "g3"})
    rows = [
        {"complaint_id_a": 1, "complaint_id_b": 2, "label": "dup"},  # 2 absent
        {"complaint_id_a": 3, "complaint_id_b": 4, "label": "dup"},
    ]
    m = evalset.score(con, rows, "run_a")
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 1)


def test_empty_denominator_raises_rather_than_reporting_perfect(con):
    _groups(con, "run_a", {1: "g1", 2: "g2"})
    rows = [{"complaint_id_a": 1, "complaint_id_b": 2, "label": "not_dup"}]
    with pytest.raises(ValueError, match="degenerate"):
        evalset.score(con, rows, "run_a")


def test_near_misses_are_below_threshold_and_deterministic(tmp_path):
    """The rejected sample is the only recall denominator that sees LSH loss."""
    ids = np.array([10, 20, 30, 40], dtype=np.int64)
    pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    sims = np.array([0.87, 0.50, 0.70, 0.20], dtype=np.float32)

    path = evalset.write_near_misses(
        ids, pairs, sims, threshold=0.88, n=10, path=tmp_path / "nm.csv"
    )
    lines = path.read_text().strip().splitlines()
    assert lines[0] == "complaint_id_a,complaint_id_b,est_similarity"
    # Band is [threshold - 0.30, threshold) = [0.58, 0.88): 0.20 and 0.50 fall
    # outside it. Kept rows are ordered by similarity, with a < b on each.
    assert [line.split(",") for line in lines[1:]] == [
        ["30", "40", "0.7"], ["10", "20", "0.87"],
    ]
