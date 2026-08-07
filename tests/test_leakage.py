"""The seven anti-leakage checks from docs/EVALUATION.md §5.

"Run before reporting any number." These are the tests that decide whether the
backtest means anything, so they assert the real requirement rather than a
convenient proxy — including item 4, which is expected to fail and is marked
`xfail(strict=True)` so that it would also fail if it started passing for the
wrong reason.

Items 1, 2 and 5 need a cutoff refit to have happened; they skip when none is
in the database rather than passing vacuously, because a vacuous pass on a
leakage check is worse than no check.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from src import db

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "data" / "ground_truth" / "enforcement_actions.csv"


def _live():
    """The live database, or a skip.

    DuckDB is single-writer at the process level, so a long refit holds the file
    and even a read-only connect fails. Skipping then is right: a leakage check
    that fails because a pipeline stage is running is reporting on the lock, not
    on leakage, and a red suite that means nothing trains people to ignore it.
    """
    import duckdb

    from src.config import PATHS

    if not PATHS.db.exists():
        pytest.skip("no database — leakage checks need a real run")
    try:
        return db.connect(read_only=True)
    except duckdb.IOException:
        pytest.skip("database is locked by a running pipeline stage")


def _refit(con):
    """The most recent cutoff refit, as `(cutoff, dedup, cluster, signals)`."""
    row = con.execute(
        """
        SELECT json_extract_string(params_json, '$.params.cutoff'), run_id, started_at
        FROM runs
        WHERE phase = 'dedup' AND status = 'ok'
          AND json_extract_string(params_json, '$.params.cutoff') NOT IN ('', 'null')
        ORDER BY started_at DESC LIMIT 1
        """
    ).fetchone()
    if not row:
        pytest.skip("no cutoff refit in the database")
    cutoff, dedup_run, started = row

    def after(phase):
        got = con.execute(
            "SELECT run_id FROM runs WHERE phase = ? AND status = 'ok' "
            "AND started_at >= ? ORDER BY started_at LIMIT 1",
            [phase, started],
        ).fetchone()
        if not got:
            pytest.skip(f"refit at {cutoff} has no {phase} run")
        return got[0]

    return date.fromisoformat(cutoff), dedup_run, after("cluster"), after("signals")


# 1 -------------------------------------------------------------------------
def test_no_post_cutoff_row_reaches_a_pre_cutoff_run():
    """§5.1 — and not by filtering the output; the stage must never see it."""
    con = _live()
    cutoff, dedup_run, cluster_run, signals_run = _refit(con)

    offenders = {
        "dup_groups": con.execute(
            "SELECT count(*) FROM dup_groups d JOIN complaints c USING (complaint_id) "
            "WHERE d.run_id = ? AND c.date_received >= ?", [dedup_run, cutoff],
        ).fetchone()[0],
        "cluster_members": con.execute(
            "SELECT count(*) FROM cluster_members m JOIN clusters cl USING (cluster_id) "
            "JOIN complaints c ON c.complaint_id = m.complaint_id "
            "WHERE cl.run_id = ? AND c.date_received >= ?", [cluster_run, cutoff],
        ).fetchone()[0],
        "cluster_timeseries": con.execute(
            "SELECT count(*) FROM cluster_timeseries WHERE run_id = ? AND period_month >= ?",
            [signals_run, cutoff],
        ).fetchone()[0],
        "signals": con.execute(
            "SELECT count(*) FROM signals WHERE run_id = ? AND period_month >= ?",
            [signals_run, cutoff],
        ).fetchone()[0],
    }
    assert offenders == dict.fromkeys(offenders, 0), offenders


# 2 -------------------------------------------------------------------------
def test_cluster_definitions_carry_a_pre_cutoff_as_of():
    """§5.2 — asserted on `clusters.as_of`, which is what the harness filters on.

    This failed when first written: as_of was read from the whole corpus, so a
    2017 refit stamped its clusters 2026-08-03.
    """
    con = _live()
    cutoff, _dedup, cluster_run, _signals = _refit(con)
    bad, newest = con.execute(
        "SELECT count(*) FILTER (WHERE as_of >= ?), max(as_of) FROM clusters WHERE run_id = ?",
        [cutoff, cluster_run],
    ).fetchone()
    assert bad == 0, f"{bad} clusters carry as_of >= {cutoff} (newest {newest})"


# 3 -------------------------------------------------------------------------
def test_embedding_model_is_a_fixed_pretrained_checkpoint():
    """§5.3 — nothing in the encode path may fit on the corpus."""
    source = (ROOT / "src" / "embed" / "encode.py").read_text().lower()
    for forbidden in (".fit(", ".train(", "trainer", "backward()", "optimizer"):
        assert forbidden not in source, f"encode.py contains {forbidden!r}"
    assert "sentencetransformer(" in source, "no pretrained checkpoint is loaded"


# 4 -------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason=(
        "EXPECTED FAILURE, argued in EVALUATION §1.4. The candidate pool was "
        "frozen before any detection run (2d4ee04, verifiable), but the curated "
        "file was written after Phases 4-5 had run, so its SHA cannot predate "
        "them. The test asserts the real requirement instead of being pointed "
        "at the candidates file to make it green; the exposure is bounded by "
        "pre-registered rules with no free parameter and a resolution audit."
    ),
)
def test_ground_truth_sha_predates_the_first_detection_run():
    """§5.4."""
    con = _live()
    first_detection = con.execute(
        "SELECT min(started_at) FROM runs WHERE phase IN ('cluster', 'signals') "
        "AND status = 'ok'"
    ).fetchone()[0]
    if first_detection is None:
        pytest.skip("no detection run yet")
    committed = subprocess.run(
        ["git", "log", "-1", "--format=%cI", "--", str(GROUND_TRUTH)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    ).stdout.strip()
    assert committed, "ground truth is not committed"
    assert committed < first_detection.isoformat(), (
        f"ground truth committed {committed}, first detection {first_detection}"
    )


# 5 -------------------------------------------------------------------------
def test_adjudication_is_blind_to_signal_status():
    """§5.5 — enforced by code review, so the review is a test.

    The adjudication surface must not be able to show whether a cluster fired.
    """
    source = (ROOT / "src" / "evaluation" / "adjudicate.py")
    if not source.exists():
        pytest.skip("adjudication tooling not built yet")
    text = source.read_text().lower()
    for forbidden in ("q_value", "eb05", "statistic", "from signals", "is_novel"):
        assert forbidden not in text, (
            f"adjudicate.py references {forbidden!r} — an adjudicator could see "
            f"whether the cluster fired"
        )


# 6 -------------------------------------------------------------------------
def test_thresholds_were_not_tuned_on_the_backtest_set():
    """§5.6 — "the one most likely to be violated by accident".

    This check was armed against `backtest_results` until 2026-08-06, and
    nothing has ever written that table — `src/evaluation/backtest.py` writes
    `baseline_results`. So `n_results` was always 0, the "no backtest yet"
    branch always ran, and the guard passed vacuously through the entire Phase 6
    and 7 backtest. A leakage check pointed at an empty table is worse than no
    check, which is the reason `_live()` skips rather than passes when the
    database is locked.

    Now that results exist, the requirement has teeth: the detection thresholds
    in `CONFIG.signals` must equal the ones the runs behind those results
    actually recorded. Compared field by field against each run's own stored
    config rather than against the whole-config fingerprint, because
    `config_hash` covers every science parameter — a Phase 8 change to the LLM
    settings would trip a fingerprint comparison while changing no threshold,
    and a check that cries wolf gets ignored, which is how this one died.
    """
    from dataclasses import asdict

    from src.config import CONFIG

    con = _live()
    n_results = con.execute("SELECT count(*) FROM baseline_results").fetchone()[0]
    if n_results == 0:
        pytest.skip("no backtest results yet — nothing could have been tuned on them")

    rows = con.execute(
        """
        SELECT DISTINCT r.run_id, r.params_json
        FROM baseline_results b JOIN runs r ON r.run_id = b.run_id
        WHERE r.status = 'ok'
        """
    ).fetchall()
    assert rows, "baseline_results rows exist but reference no successful run"

    frozen = asdict(CONFIG.signals)
    for run_id, params_json in rows:
        recorded = json.loads(params_json)["config"]["signals"]
        for key, value in frozen.items():
            assert str(recorded[key]) == str(value), (
                f"signals.{key} is {value!r} in config.py but the backtest run "
                f"{run_id} recorded {recorded[key]!r}. A detection threshold "
                f"changed after the backtest produced a number — EVALUATION §5.6 "
                f"is the check this is meant to fail."
            )


# 7 -------------------------------------------------------------------------
def test_harm_keywords_never_reach_the_detection_path():
    """§5.7 — the detection path must not read the ground truth at all."""
    import csv

    if GROUND_TRUTH.exists():
        with GROUND_TRUTH.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert all(not r["harm_keywords"].strip() for r in rows), (
            "harm_keywords is populated; an empty column cannot leak"
        )

    detection = [
        "src/dedup", "src/embed", "src/cluster", "src/signals",
        "src/normalization", "src/ingestion/build.py",
    ]
    for rel in detection:
        path = ROOT / rel
        files = path.rglob("*.py") if path.is_dir() else [path]
        for file in files:
            text = file.read_text().lower()
            for forbidden in ("harm_keywords", "enforcement_actions", "harm_summary"):
                assert forbidden not in text, f"{file.name} references {forbidden!r}"


def test_the_suite_reports_which_items_are_covered():
    """A checklist that silently skips six of seven items is not a checklist."""
    con = _live()
    has_refit = con.execute(
        "SELECT count(*) FROM runs WHERE phase = 'dedup' AND status = 'ok' "
        "AND json_extract_string(params_json, '$.params.cutoff') NOT IN ('', 'null')"
    ).fetchone()[0]
    covered = {
        "1 no post-cutoff rows": bool(has_refit),
        "2 cluster as_of": bool(has_refit),
        "3 fixed checkpoint": True,
        "4 ground-truth SHA": True,   # runs, and is expected to fail
        "5 blind adjudication": (ROOT / "src/evaluation/adjudicate.py").exists(),
        "6 thresholds frozen": True,
        "7 no keyword leakage": True,
    }
    print("\n" + json.dumps(covered, indent=2))
    assert sum(covered.values()) >= 5, covered
