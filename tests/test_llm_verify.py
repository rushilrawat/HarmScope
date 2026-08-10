"""Blinded human verification for descriptive cluster labels."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
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
            "distinct_from_taxonomy, confidence, is_likely_template, model, "
            "prompt_version, input_hash) "
            "VALUES (?, ?, 'servicer', 'a payment is due', 'an unexpected fee', ?, ?, ?, "
            "'claude-opus-5', 'v1', ?)",
            [
                cluster_id,
                f"A servicer applies an unexpected fee {number}.",
                bool(number % 2),
                ("high", "medium", "low")[number % 3],
                bool(number % 4),
                f"input-{number}",
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
        "is_likely_template, model, prompt_version, input_hash) VALUES "
        "(?, 'historical label', true, 'high', false, 'claude-opus-5', 'v1', 'old')",
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
    assert all(not row["harm_mechanism"].startswith("'") for row in rows)
    first_number = int(rows[0]["cluster_id"].rsplit(":", 1)[1])
    assert rows[0]["harm_mechanism"] == (
        f"A servicer applies an unexpected fee {first_number}."
    )

    candidates = verify._candidates(con, signals_run)
    selected_ids = {row["cluster_id"] for row in rows}
    coverage: dict[str, set[str]] = {}
    for candidate in candidates:
        if candidate.cluster_id not in selected_ids:
            continue
        for dimension, value in verify._strata(candidate):
            coverage.setdefault(dimension, set()).add(value)
    assert set(coverage) == {
        "fired_status", "product_family", "confidence",
        "template_suspicion", "taxonomy_distinctness",
    }
    assert all(len(values) >= 2 for values in coverage.values())


def test_export_writes_immutable_provenance_sidecar(verification_fixture, tmp_path):
    """The review artifact carries the exact source/model/population fingerprint."""
    con, signals_run = verification_fixture
    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    metadata = verify.load_worklist_metadata(con, path)
    csv_ids = sorted(row["cluster_id"] for row in csv.DictReader(path.open()))

    assert metadata.signals_run == signals_run
    assert metadata.cluster_run == "0000000000001-cluster1"
    assert metadata.model == "claude-opus-5"
    assert metadata.prompt_version == "v1"
    assert metadata.seed == 7
    assert list(metadata.cluster_ids) == csv_ids
    assert len(metadata.worklist_version) == 64
    assert verify.worklist_sidecar_path(path).exists()


def _complete_exported_worklist(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update({
            "mechanism_accuracy": "agree",
            "taxonomy_distinctness_accuracy": "agree",
            "template_accuracy": "agree",
            "should_have_abstained": "false",
            "failure_category": "none",
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=verify.HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_record_worklist_uses_sidecar_after_later_run_and_config_change(
    verification_fixture, tmp_path, monkeypatch,
):
    """Recording must use export-time provenance, never latest/current defaults."""
    con, signals_run = verification_fixture
    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    _complete_exported_worklist(path)
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES ('9999999999999-signals', 'signals', 'test', "
        "'test', ?, now(), 'ok')",
        [json.dumps({"params": {"cluster_run": "later-cluster-run"}})],
    )
    monkeypatch.setattr(
        verify,
        "CONFIG",
        replace(verify.CONFIG, llm=replace(
            verify.CONFIG.llm, model="future-model", prompt_version="v2",
        )),
    )

    count, metadata = verify.record_worklist(con, path, "reviewer-1")

    assert count == 50
    assert metadata.signals_run == signals_run
    assert metadata.model == "claude-opus-5"
    assert con.execute(
        "SELECT DISTINCT signals_run, worklist_version FROM label_verifications"
    ).fetchall() == [(signals_run, metadata.worklist_version)]


def test_record_worklist_rejects_csv_population_tampering(
    verification_fixture, tmp_path,
):
    """A changed review population cannot retain the export's trusted digest."""
    con, signals_run = verification_fixture
    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    _complete_exported_worklist(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["cluster_id"] = "tampered-cluster"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=verify.HEADER)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="cluster IDs.*sidecar"):
        verify.record_worklist(con, path, "reviewer-1")


def test_record_worklist_rejects_sidecar_digest_tampering(
    verification_fixture, tmp_path,
):
    """Editing provenance without recomputing its digest is detected before ingest."""
    con, signals_run = verification_fixture
    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    sidecar = verify.worklist_sidecar_path(path)
    payload = json.loads(sidecar.read_text())
    payload["model"] = "tampered-model"
    sidecar.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="digest"):
        verify.record_worklist(con, path, "reviewer-1")


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@"])
def test_export_neutralises_spreadsheet_formula_prefixes(
    verification_fixture, tmp_path, prefix,
):
    """CSV quoting alone cannot make untrusted spreadsheet cells inert."""
    con, signals_run = verification_fixture
    con.execute(
        "UPDATE cluster_labels SET harm_mechanism = ?, actors = ?, preconditions = ?, "
        "consumer_impact = ?",
        [prefix + "mechanism", prefix + "actors", prefix + "condition", prefix + "impact"],
    )
    con.execute("UPDATE cluster_novelty SET dominant_label = ?", [prefix + "taxonomy"])
    con.execute("UPDATE narratives SET text_redacted = ?", [prefix + "narrative"])

    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    first = next(csv.DictReader(path.open()))

    for field in (
        "harm_mechanism", "actors", "preconditions", "consumer_impact",
        "dominant_taxonomy", "narrative_1",
    ):
        assert first[field].startswith("'" + prefix)
    assert verify.spreadsheet_safe_text(prefix + "reviewer note").startswith("'" + prefix)


def test_export_caps_narratives_at_1200_characters(verification_fixture, tmp_path):
    """Reviewer excerpts retain the configured privacy/readability bound."""
    con, signals_run = verification_fixture
    con.execute("UPDATE narratives SET text_redacted = ?", ["x" * 1_300])

    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")

    assert all(
        len(row[f"narrative_{number}"]) == 1_200
        for row in csv.DictReader(path.open())
        for number in range(1, 11)
    )


def test_candidate_requires_ten_available_redacted_narratives(
    verification_fixture,
):
    """Member counts cannot substitute for reviewable redacted evidence."""
    con, signals_run = verification_fixture
    missing_cluster = "0000000000001-cluster1:mortgage:0"
    con.execute("DELETE FROM narratives WHERE complaint_id = 1")

    candidate_ids = {row.cluster_id for row in verify._candidates(con, signals_run)}

    assert missing_cluster not in candidate_ids


def test_worklist_narratives_skip_empty_redacted_text(verification_fixture):
    """Eligibility and the ten displayed excerpts use the same availability rule."""
    con, _signals_run = verification_fixture
    cluster_id = "0000000000001-cluster1:mortgage:0"
    con.execute(
        "INSERT INTO complaints "
        "(complaint_id, date_received, period_month, product_family, has_narrative) "
        "VALUES (10000, DATE '2020-01-01', DATE '2020-01-01', 'credit_card', true)"
    )
    con.execute(
        "INSERT INTO narratives "
        "(complaint_id, text_redacted, text_hash, redaction_count) "
        "VALUES (10000, 'eleventh available narrative', 'hash-10000', 0)"
    )
    con.execute(
        "INSERT INTO cluster_members (cluster_id, complaint_id) VALUES (?, 10000)",
        [cluster_id],
    )
    con.execute("UPDATE narratives SET text_redacted = '   ' WHERE complaint_id = 1")

    narratives = verify._narratives(con, cluster_id)

    assert len(narratives) == 10
    assert all(narrative for narrative in narratives)
    assert narratives[-1] == "eleventh available narrative"


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


def _reviewed_rows(
    agree: int,
    total: int,
    *,
    start: int = 0,
    reviewer_id: str = "reviewer-1",
    reviewer_origin: str = "human",
):
    rows = []
    for offset, number in enumerate(range(start, start + total)):
        mechanism = "agree" if offset < agree else "disagree"
        rows.append(verify.Verification(
            cluster_id=f"0000000000001-cluster1:mortgage:{number}",
            reviewer_id=reviewer_id,
            mechanism_accuracy=mechanism,
            taxonomy_distinctness_accuracy="agree",
            template_accuracy="agree",
            should_have_abstained=False,
            failure_category="none" if mechanism == "agree" else "other",
            notes=None,
            reviewer_origin=reviewer_origin,
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
    assert report.unique_reviewed_clusters == 50
    assert report.gate_passed is True
    assert report.gate_eligible is True


def test_report_render_exposes_human_rates_and_breakdowns(verification_fixture):
    """The CLI report must retain its denominator and hidden-strata context."""
    con, signals_run = verification_fixture
    verify.record(con, _reviewed_rows(agree=39, total=50), signals_run, "wl-v1")

    rendered = verify.report(con, "wl-v1").render().lower()

    for term in ("39/50", "78.0%", "wilson", "fired", "control", "confidence"):
        assert term in rendered


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


def test_gate_uses_unique_human_clusters_not_rows_or_model_reviews(
    verification_fixture,
):
    """Duplicates and model judges cannot inflate 49 human labels past the gate."""
    con, signals_run = verification_fixture
    human_rows = _reviewed_rows(agree=49, total=49)
    duplicate_human = replace(human_rows[0], reviewer_id="reviewer-2")
    model_rows = _reviewed_rows(
        agree=50,
        total=50,
        reviewer_id="model-judge",
        reviewer_origin="model",
    )
    verify.record(con, [*human_rows, duplicate_human, *model_rows], signals_run, "wl-v1")

    report = verify.report(con, "wl-v1")

    assert report.overall.mechanism.total == 50
    assert report.unique_reviewed_clusters == 49
    assert report.gate_eligible is True
    assert report.gate_passed is False


def test_unscoped_report_cannot_pool_worklists_to_pass_the_gate(verification_fixture):
    """Fifty labels split across worklists are never one eligible review population."""
    con, signals_run = verification_fixture
    verify.record(con, _reviewed_rows(25, 25), signals_run, "wl-a")
    verify.record(con, _reviewed_rows(25, 25, start=25), signals_run, "wl-b")

    assert verify.report(con, "wl-a").gate_passed is False
    assert verify.report(con, "wl-b").gate_passed is False
    pooled = verify.report(con)
    assert pooled.unique_reviewed_clusters == 50
    assert pooled.gate_eligible is False
    assert pooled.gate_passed is False
    assert "not gate eligible" in pooled.render().lower()


def test_wilson_is_bounded_for_empty_and_complete_samples():
    """Intervals remain valid at the statistical boundaries."""
    assert verify.wilson(0, 0) == (0.0, 0.0)
    low, high = verify.wilson(50, 50)
    assert 0.0 < low < 1.0
    assert high == pytest.approx(1.0)
