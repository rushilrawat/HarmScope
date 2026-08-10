"""Blinded human verification for descriptive cluster labels."""

from __future__ import annotations

import csv
import json
from datetime import date

import pytest

from src.llm import verify


@pytest.fixture
def verification_fixture(con):
    cluster_run = "0000000000001-cluster1"
    signals_run = "0000000000002-signal01"
    for run_id, phase in ((cluster_run, "cluster"), (signals_run, "signals")):
        params = (
            json.dumps({"params": {"cluster_run": cluster_run}})
            if phase == "signals"
            else "{}"
        )
        con.execute(
            "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
            "started_at, status) VALUES (?, ?, 'test', 'test', ?, now(), 'ok')",
            [run_id, phase, params],
        )

    for number in range(60):
        cluster_id = f"{cluster_run}:mortgage:{number}"
        family = "mortgage" if number % 2 else "credit_card"
        con.execute(
            "INSERT INTO clusters "
            "(cluster_id, run_id, product_family, n_members, centroid_idx, as_of) "
            "VALUES (?, ?, ?, 30, ?, ?)",
            [cluster_id, cluster_run, family, number * 10, date(2020, 1, 1)],
        )
        con.execute(
            "INSERT INTO cluster_novelty "
            "(cluster_id, dominant_label, novelty_score, is_novel) "
            "VALUES (?, ?, 0.5, true)",
            [cluster_id, f"taxonomy-{number % 3}"],
        )
        con.execute(
            "INSERT INTO cluster_labels "
            "(cluster_id, harm_mechanism, actors, preconditions, consumer_impact, "
            "distinct_from_taxonomy, confidence, is_likely_template) "
            "VALUES (?, ?, 'servicer', 'a payment is due', 'an unexpected fee', ?, ?, ?)",
            [
                cluster_id,
                f"A servicer applies an unexpected fee {number}.",
                bool(number % 2),
                ("high", "medium", "low")[number % 3],
                bool(number % 4),
            ],
        )
        for narrative_number in range(10):
            complaint_id = number * 10 + narrative_number + 1
            con.execute(
                "INSERT INTO complaints "
                "(complaint_id, date_received, period_month, product_family, has_narrative) "
                "VALUES (?, ?, ?, ?, true)",
                [complaint_id, date(2020, 1, 1), date(2020, 1, 1), family],
            )
            con.execute(
                "INSERT INTO narratives "
                "(complaint_id, text_redacted, text_hash, redaction_count) "
                "VALUES (?, ?, ?, 0)",
                [
                    complaint_id,
                    f" complaint {complaint_id} describes an unexpected fee ",
                    f"hash-{complaint_id}",
                ],
            )
            con.execute(
                "INSERT INTO cluster_members (cluster_id, complaint_id) VALUES (?, ?)",
                [cluster_id, complaint_id],
            )
        if number % 2:
            con.execute(
                "INSERT INTO signals "
                "(signal_id, run_id, cluster_id, company_id, period_month, method, "
                "statistic, q_value, n_supporting, n_supporting_groups, as_of) "
                "VALUES (?, ?, ?, '__ALL__', ?, 'ewma', 1.0, 0.01, 30, 30, ?)",
                [
                    f"signal-{number}", signals_run, cluster_id,
                    date(2020, 1, 1), date(2020, 1, 1),
                ],
            )
    return con, signals_run


def _add_historical_cluster(con):
    """A labelled cluster from another refit must not become a control row."""
    historical_run = "0000000000000-cluster0"
    cluster_id = f"{historical_run}:mortgage:0"
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'cluster', 'test', 'test', '{}', now(), 'ok')",
        [historical_run],
    )
    con.execute(
        "INSERT INTO clusters "
        "(cluster_id, run_id, product_family, n_members, centroid_idx, as_of) "
        "VALUES (?, ?, 'mortgage', 30, 700, ?)",
        [cluster_id, historical_run, date(2020, 1, 1)],
    )
    con.execute(
        "INSERT INTO cluster_labels "
        "(cluster_id, harm_mechanism, distinct_from_taxonomy, confidence, "
        "is_likely_template) VALUES (?, 'historical label', true, 'high', false)",
        [cluster_id],
    )
    for number in range(10):
        complaint_id = 1000 + number
        con.execute(
            "INSERT INTO complaints "
            "(complaint_id, date_received, period_month, product_family, has_narrative) "
            "VALUES (?, ?, ?, 'mortgage', true)",
            [complaint_id, date(2020, 1, 1), date(2020, 1, 1)],
        )
        con.execute(
            "INSERT INTO narratives "
            "(complaint_id, text_redacted, text_hash, redaction_count) "
            "VALUES (?, 'historical text', ?, 0)",
            [complaint_id, f"historical-{number}"],
        )
        con.execute(
            "INSERT INTO cluster_members (cluster_id, complaint_id) VALUES (?, ?)",
            [cluster_id, complaint_id],
        )
    return cluster_id


def test_export_is_seeded_stratified_and_signal_blinded(verification_fixture, tmp_path):
    """Sampling must be reproducible without leaking signal strength to reviewers."""
    con, signals_run = verification_fixture

    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    rows = list(csv.DictReader(path.open()))

    assert len(rows) == 50
    assert "did_fire" not in rows[0]
    assert "is_fired" not in rows[0]
    assert "q_value" not in rows[0]
    assert "statistic" not in rows[0]
    assert "confidence" not in rows[0]
    assert rows == list(csv.DictReader(
        verify.export_worklist(con, signals_run, 50, 7, tmp_path / "again.csv").open()
    ))
    assert all(row["narrative_10"] for row in rows)
    assert all("  " not in row["narrative_1"] for row in rows)


def test_export_requires_the_configured_human_review_minimum(
    verification_fixture, tmp_path,
):
    """A smaller worklist cannot be presented as satisfying the human-review gate."""
    con, signals_run = verification_fixture

    with pytest.raises(ValueError, match="at least 50"):
        verify.export_worklist(con, signals_run, 49, 7, tmp_path / "review.csv")


def test_export_rejects_an_eligible_population_smaller_than_requested(
    verification_fixture, tmp_path,
):
    """The exporter must not silently downgrade the required human denominator."""
    con, signals_run = verification_fixture

    with pytest.raises(ValueError, match="eligible.*60.*requested 61"):
        verify.export_worklist(con, signals_run, 61, 7, tmp_path / "review.csv")


def test_historical_cluster_run_cannot_enter_worklist_or_human_report(
    verification_fixture, tmp_path,
):
    """Controls belong to the selected signals run's cluster refit, never history."""
    con, signals_run = verification_fixture
    historical_cluster = _add_historical_cluster(con)

    rows = list(csv.DictReader(
        verify.export_worklist(con, signals_run, 60, 7, tmp_path / "review.csv").open()
    ))

    assert historical_cluster not in {row["cluster_id"] for row in rows}
    historical_review = verify.Verification(
        cluster_id=historical_cluster,
        reviewer_id="reviewer-1",
        mechanism_accuracy="agree",
        taxonomy_distinctness_accuracy="agree",
        template_accuracy="agree",
        should_have_abstained=False,
        failure_category="none",
        notes=None,
    )
    with pytest.raises(ValueError, match="does not belong"):
        verify.record(con, [historical_review], signals_run, "wl-v1")


def _filled_worklist(tmp_path, **changes):
    row = dict.fromkeys(verify.HEADER, "")
    row.update({
        "cluster_id": "cluster-1",
        "mechanism_accuracy": "agree",
        "taxonomy_distinctness_accuracy": "agree",
        "template_accuracy": "agree",
        "should_have_abstained": "false",
        "failure_category": "none",
    })
    row.update(changes)
    path = tmp_path / "filled.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=verify.HEADER)
        writer.writeheader()
        writer.writerow(row)
    return path


def test_parse_rejects_unknown_decisions(tmp_path):
    """Reviewer decision fields cannot silently accept a typo or new category."""
    path = _filled_worklist(tmp_path, mechanism_accuracy="mostly")

    with pytest.raises(ValueError, match="agree.*partial.*disagree"):
        verify.parse_worklist(path, "reviewer-1")


def test_parse_enforces_failure_category_consistency(tmp_path):
    """Fully agreeing reviews use none; a problem must have an approved category."""
    path = _filled_worklist(tmp_path, failure_category="template_error")
    with pytest.raises(ValueError, match="failure_category must be none"):
        verify.parse_worklist(path, "reviewer-1")

    path = _filled_worklist(tmp_path, mechanism_accuracy="disagree")
    with pytest.raises(ValueError, match="failure_category is required"):
        verify.parse_worklist(path, "reviewer-1")


def _reviewed_rows(agree: int, total: int):
    rows = []
    for number in range(total):
        mechanism = "agree" if number < agree else "disagree"
        rows.append(verify.Verification(
            cluster_id=f"0000000000001-cluster1:mortgage:{number}",
            reviewer_id="reviewer-1",
            mechanism_accuracy=mechanism,
            taxonomy_distinctness_accuracy="agree",
            template_accuracy="agree",
            should_have_abstained=False,
            failure_category="none" if mechanism == "agree" else "other",
            notes=None,
        ))
    return rows


def test_report_has_denominators_and_wilson_intervals(verification_fixture):
    """Reported agreement includes its human-review denominator and uncertainty."""
    con, signals_run = verification_fixture
    verify.record(con, _reviewed_rows(agree=39, total=50), signals_run, "wl-v1")

    report = verify.report(con, "wl-v1")

    assert report.mechanism_agree == 39
    assert report.mechanism_total == 50
    assert report.mechanism_rate == pytest.approx(0.78)
    assert report.mechanism_ci_low < 0.78 < report.mechanism_ci_high
    assert report.overall.mechanism.numerator == 39
    assert report.overall.mechanism.denominator == 50
    assert set(report.by_fired_status) == {"fired", "control"}
    assert report.by_fired_status["fired"].mechanism.total == 25
    assert report.by_confidence["high"].mechanism.total > 0


def test_model_origin_reviews_do_not_count_as_human_verification(verification_fixture):
    """Stored model reviews remain auditable but cannot satisfy the human gate."""
    con, signals_run = verification_fixture
    verify.record(con, _reviewed_rows(agree=39, total=50), signals_run, "wl-v1")
    model_review = verify.Verification(
        cluster_id="0000000000001-cluster1:mortgage:0",
        reviewer_id="model-judge",
        mechanism_accuracy="agree",
        taxonomy_distinctness_accuracy="agree",
        template_accuracy="agree",
        should_have_abstained=False,
        failure_category="none",
        notes=None,
        reviewer_origin="model",
    )
    assert verify.record(con, [model_review], signals_run, "wl-v1") == 1

    report = verify.report(con, "wl-v1")

    assert report.mechanism_total == 50
    assert con.execute(
        "SELECT reviewer_origin FROM label_verifications WHERE reviewer_id = 'model-judge'"
    ).fetchone() == ("model",)


def test_wilson_is_bounded_for_empty_and_complete_samples():
    """Intervals remain valid at the statistical boundaries."""
    assert verify.wilson(0, 0) == (0.0, 0.0)
    low, high = verify.wilson(50, 50)
    assert 0.0 < low < 1.0
    assert high == pytest.approx(1.0)
