"""Blinded human verification for descriptive cluster labels."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest

from src.llm import verify


@pytest.fixture(autouse=True)
def private_review_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        verify,
        "PATHS",
        SimpleNamespace(interim=tmp_path),
        raising=False,
    )


@pytest.fixture
def verification_fixture(con):
    cluster_run = "0000000000001-cluster1"
    signals_run = "0000000000002-signal01"
    for run_id, phase in ((cluster_run, "cluster"), (signals_run, "signals")):
        params = (
            json.dumps({"params": {"cluster_run": cluster_run}})
            if phase == "signals"
            else json.dumps({"params": {"model": "embed-m"}})
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
            "(cluster_id, run_id, product_family, n_members, centroid_idx, coherence, as_of) "
            "VALUES (?, ?, ?, 30, ?, 0.9, ?)",
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
                "VALUES (?, ?, ?, '__ALL__', ?, 'ewma', 1.0, NULL, 30, 30, ?)",
                [
                    f"signal-{number}",
                    signals_run,
                    cluster_id,
                    date(2020, 1, 1),
                    date(2020, 1, 1),
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
    assert rows == list(
        csv.DictReader(
            verify.export_worklist(con, signals_run, 50, 7, tmp_path / "again.csv").open()
        )
    )
    assert all(row["narrative_10"] for row in rows)
    assert all("  " not in row["narrative_1"] for row in rows)
    assert all(not row["harm_mechanism"].startswith("'") for row in rows)
    first_number = int(rows[0]["cluster_id"].rsplit(":", 1)[1])
    assert rows[0]["harm_mechanism"] == (f"A servicer applies an unexpected fee {first_number}.")

    candidates = verify._candidates(con, signals_run)
    selected_ids = {row["cluster_id"] for row in rows}
    coverage: dict[str, set[str]] = {}
    for candidate in candidates:
        if candidate.cluster_id not in selected_ids:
            continue
        for dimension, value in verify._strata(candidate):
            coverage.setdefault(dimension, set()).add(value)
    assert set(coverage) == {
        "fired_status",
        "product_family",
        "confidence",
        "template_suspicion",
        "taxonomy_distinctness",
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
    assert set(metadata.fired_cluster_ids) == {
        cluster_id for cluster_id in csv_ids if int(cluster_id.rsplit(":", 1)[1]) % 2
    }
    assert len(metadata.worklist_version) == 64
    assert verify.worklist_sidecar_path(path).exists()


def test_failed_signals_run_cannot_export_candidates_or_controls(
    verification_fixture,
    tmp_path,
):
    con, signals_run = verification_fixture
    con.execute("UPDATE runs SET status = 'failed' WHERE run_id = ?", [signals_run])

    with pytest.raises(ValueError, match="successful signals run"):
        verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")

    assert not (tmp_path / "review.csv").exists()
    assert not verify.worklist_sidecar_path(tmp_path / "review.csv").exists()


def test_label_review_export_rejects_parent_symlink_swap(
    verification_fixture,
    tmp_path,
    monkeypatch,
):
    con, signals_run = verification_fixture
    moved_parent = tmp_path.with_name(tmp_path.name + "-original")
    outside = tmp_path.with_name(tmp_path.name + "-outside")
    outside.mkdir()
    original_candidates = verify._candidates

    def swap_parent(*args):
        candidates = original_candidates(*args)
        tmp_path.rename(moved_parent)
        tmp_path.symlink_to(outside, target_is_directory=True)
        return candidates

    monkeypatch.setattr(verify, "_candidates", swap_parent)

    try:
        with pytest.raises((ValueError, OSError), match="symlink|path|parent|interim"):
            verify.export_worklist(
                con,
                signals_run,
                50,
                7,
                tmp_path / "review.csv",
            )
        assert not (outside / "review.csv").exists()
    finally:
        if tmp_path.is_symlink():
            tmp_path.unlink()
        if moved_parent.exists():
            moved_parent.rename(tmp_path)
        outside.rmdir()


def test_label_review_rejects_nested_private_destination_before_database_read(
    verification_fixture,
    tmp_path,
    monkeypatch,
):
    con, signals_run = verification_fixture
    touched: list[str] = []

    def fail_candidates(*_args, **_kwargs):
        touched.append("candidates")
        raise AssertionError("private output path must be rejected before database reads")

    monkeypatch.setattr(verify, "_candidates", fail_candidates)

    with pytest.raises(ValueError, match="direct.*child|nested"):
        verify.export_worklist(con, signals_run, 50, 7, tmp_path / "nested" / "review.csv")

    assert touched == []


def test_label_review_sidecar_failure_preserves_existing_artifact_pair(
    verification_fixture,
    tmp_path,
    monkeypatch,
):
    con, signals_run = verification_fixture
    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    sidecar = verify.worklist_sidecar_path(path)
    original_csv = path.read_bytes()
    original_sidecar = sidecar.read_bytes()
    con.execute("UPDATE narratives SET text_redacted = 'changed private evidence'")
    original_write = verify.atomic_write_bytes

    def fail_sidecar(target, root, data):
        if target.name.endswith(".metadata.json"):
            raise OSError("simulated sidecar write failure")
        return original_write(target, root, data)

    monkeypatch.setattr(verify, "atomic_write_bytes", fail_sidecar)
    with pytest.raises(OSError, match="sidecar write failure"):
        verify.export_worklist(con, signals_run, 50, 7, path)

    assert path.read_bytes() == original_csv
    assert sidecar.read_bytes() == original_sidecar


@pytest.mark.parametrize("old_csv", [False, True])
@pytest.mark.parametrize("old_sidecar", [False, True])
@pytest.mark.parametrize("failed_component", ["sidecar", "csv"])
def test_worklist_export_failure_restores_each_prior_component_exactly(
    verification_fixture,
    tmp_path,
    monkeypatch,
    old_csv,
    old_sidecar,
    failed_component,
):
    """A failed paired publish restores present components and removes new ones."""
    con, signals_run = verification_fixture
    path = tmp_path / "review.csv"
    sidecar = verify.worklist_sidecar_path(path)
    expected_csv = b"old csv bytes\n" if old_csv else None
    expected_sidecar = b"old sidecar bytes\n" if old_sidecar else None
    if expected_csv is not None:
        path.write_bytes(expected_csv)
    if expected_sidecar is not None:
        sidecar.write_bytes(expected_sidecar)

    original_write = verify.atomic_write_bytes
    failed = False

    def fail_after_replace(target, root, data):
        nonlocal failed
        written = original_write(target, root, data)
        is_target = (failed_component == "csv" and target.name == path.name) or (
            failed_component == "sidecar" and target.name == sidecar.name
        )
        if is_target and not failed:
            failed = True
            raise OSError(f"simulated {failed_component} fsync failure")
        return written

    monkeypatch.setattr(verify, "atomic_write_bytes", fail_after_replace)
    with pytest.raises(OSError, match="fsync failure"):
        verify.export_worklist(con, signals_run, 50, 7, path)

    assert (path.read_bytes() if path.exists() else None) == expected_csv
    assert (sidecar.read_bytes() if sidecar.exists() else None) == expected_sidecar


def _complete_exported_worklist(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(
            {
                "mechanism_accuracy": "agree",
                "taxonomy_distinctness_accuracy": "agree",
                "template_accuracy": "agree",
                "should_have_abstained": "false",
                "failure_category": "none",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=verify.HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_record_worklist_uses_sidecar_after_later_run_and_config_change(
    verification_fixture,
    tmp_path,
    monkeypatch,
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
        replace(
            verify.CONFIG,
            llm=replace(
                verify.CONFIG.llm,
                model="future-model",
                prompt_version="v2",
            ),
        ),
    )

    count, metadata = verify.record_worklist(con, path, "reviewer-1")

    assert count == 50
    assert metadata.signals_run == signals_run
    assert metadata.model == "claude-opus-5"
    assert con.execute(
        "SELECT DISTINCT signals_run, worklist_version FROM label_verifications"
    ).fetchall() == [(signals_run, metadata.worklist_version)]


def test_record_worklist_uses_export_time_fired_snapshot_after_config_change(
    verification_fixture,
    tmp_path,
    monkeypatch,
):
    con, signals_run = verification_fixture
    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    metadata = verify.load_worklist_metadata(con, path)
    _complete_exported_worklist(path)
    monkeypatch.setattr(verify, "canonical_fired_cluster_ids", lambda *_args: set())

    verify.record_worklist(con, path, "reviewer-1")

    stored = {
        cluster_id: bool(is_fired)
        for cluster_id, is_fired in con.execute(
            "SELECT cluster_id, is_fired FROM label_verifications"
        ).fetchall()
    }
    assert {cluster_id for cluster_id, fired in stored.items() if fired} == set(
        metadata.fired_cluster_ids
    )


class _MutateLabelOnBeginConnection:
    def __init__(self, con, cluster_id: str):
        self._con = con
        self._cluster_id = cluster_id
        self._mutated = False

    def execute(self, query, parameters=None):
        if query.strip().upper() == "BEGIN TRANSACTION" and not self._mutated:
            self._mutated = True
            self._con.execute(
                "UPDATE cluster_labels SET input_hash = 'post-precheck-change' "
                "WHERE cluster_id = ?",
                [self._cluster_id],
            )
        if parameters is None:
            return self._con.execute(query)
        return self._con.execute(query, parameters)

    def __getattr__(self, name):
        return getattr(self._con, name)


def test_record_worklist_rechecks_label_provenance_inside_owned_transaction(
    verification_fixture,
    tmp_path,
):
    con, signals_run = verification_fixture
    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    metadata = verify.load_worklist_metadata(con, path)
    _complete_exported_worklist(path)
    wrapped = _MutateLabelOnBeginConnection(con, metadata.cluster_ids[0])

    with pytest.raises(ValueError, match="label source|provenance"):
        verify.record_worklist(wrapped, path, "reviewer-1")

    assert con.execute("SELECT count(*) FROM label_verifications").fetchone() == (0,)


def test_record_rejects_caller_transaction_without_mutating_it(verification_fixture):
    con, signals_run = verification_fixture
    row = _reviewed_rows(agree=1, total=1)[0]
    con.execute("BEGIN TRANSACTION")
    try:
        transaction_id = con.execute("SELECT current_transaction_id()").fetchone()[0]
        with pytest.raises(verify.VerificationTransactionError, match="autocommit"):
            verify.record(con, [row], signals_run, "wl-v1")
        assert con.execute("SELECT count(*) FROM label_verifications").fetchone() == (0,)
        assert con.execute("SELECT current_transaction_id()").fetchone() == (transaction_id,)
    finally:
        con.execute("ROLLBACK")


def test_record_worklist_rejects_csv_population_tampering(
    verification_fixture,
    tmp_path,
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


@pytest.mark.parametrize("field", ["harm_mechanism", "narrative_1"])
def test_record_worklist_rejects_reviewer_visible_source_tampering(
    verification_fixture,
    tmp_path,
    field,
):
    """Only decision and note fields may change after export."""
    con, signals_run = verification_fixture
    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    _complete_exported_worklist(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = "tampered reviewer-visible source"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=verify.HEADER)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="source|tamper"):
        verify.record_worklist(con, path, "reviewer-1")
    assert con.execute("SELECT count(*) FROM label_verifications").fetchone() == (0,)


def test_human_review_records_are_immutable_but_exact_replay_is_idempotent(
    verification_fixture,
):
    con, signals_run = verification_fixture
    original = _reviewed_rows(agree=1, total=1)[0]
    assert verify.record(con, [original], signals_run, "wl-v1") == 1
    reviewed_at = con.execute("SELECT reviewed_at FROM label_verifications").fetchone()[0]

    assert verify.record(con, [original], signals_run, "wl-v1") == 1
    assert con.execute("SELECT reviewed_at FROM label_verifications").fetchone() == (reviewed_at,)

    changed = replace(
        original,
        mechanism_accuracy="disagree",
        failure_category="other",
    )
    with pytest.raises(ValueError, match="immutable|different"):
        verify.record(con, [changed], signals_run, "wl-v1")
    assert con.execute("SELECT mechanism_accuracy FROM label_verifications").fetchone() == (
        "agree",
    )


def test_record_worklist_rejects_sidecar_digest_tampering(
    verification_fixture,
    tmp_path,
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


def test_record_worklist_rejects_csv_sidecar_renaming_before_sql(
    verification_fixture,
    tmp_path,
):
    con, signals_run = verification_fixture
    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    renamed = tmp_path / "renamed.csv"
    path.rename(renamed)
    verify.worklist_sidecar_path(path).rename(verify.worklist_sidecar_path(renamed))
    _complete_exported_worklist(renamed)

    with pytest.raises(ValueError, match="filename|path"):
        verify.record_worklist(con, renamed, "reviewer-1")
    assert con.execute("SELECT count(*) FROM label_verifications").fetchone() == (0,)


def test_record_preflight_rejects_unsafe_note_before_sql(
    verification_fixture,
    tmp_path,
):
    con, signals_run = verification_fixture
    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    _complete_exported_worklist(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["notes"] = "=unsafe reviewer note"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=verify.HEADER)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="notes.*spreadsheet-safe"):
        verify.prepare_worklist_record(path, "reviewer-1")
    assert con.execute("SELECT count(*) FROM label_verifications").fetchone() == (0,)


@pytest.mark.parametrize(
    "unsafe",
    [
        "\x1b=SUM(1,1)",
        "\u202e=SUM(1,1)",
        "\x1b]0;hidden title\x07=SUM(1,1)",
    ],
)
def test_record_preflight_rejects_control_prefixed_formula_notes(
    verification_fixture,
    tmp_path,
    unsafe,
):
    con, signals_run = verification_fixture
    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    _complete_exported_worklist(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["notes"] = unsafe
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=verify.HEADER)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="notes.*spreadsheet-safe"):
        verify.prepare_worklist_record(path, "reviewer-1")
    assert con.execute("SELECT count(*) FROM label_verifications").fetchone() == (0,)


def test_label_review_record_rejects_parent_symlink_before_read_or_sql(
    verification_fixture,
    tmp_path,
):
    con, signals_run = verification_fixture
    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    _complete_exported_worklist(path)
    moved_parent = tmp_path.with_name(tmp_path.name + "-original")
    outside = tmp_path.with_name(tmp_path.name + "-outside")
    outside.mkdir()
    tmp_path.rename(moved_parent)
    tmp_path.symlink_to(outside, target_is_directory=True)

    try:
        with pytest.raises((ValueError, OSError), match="symlink|path|parent|interim"):
            verify.prepare_worklist_record(tmp_path / "review.csv", "reviewer-1")
        assert con.execute("SELECT count(*) FROM label_verifications").fetchone() == (0,)
    finally:
        if tmp_path.is_symlink():
            tmp_path.unlink()
        if moved_parent.exists():
            moved_parent.rename(tmp_path)
        outside.rmdir()


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@"])
def test_export_neutralises_spreadsheet_formula_prefixes(
    verification_fixture,
    tmp_path,
    prefix,
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
        "harm_mechanism",
        "actors",
        "preconditions",
        "consumer_impact",
        "dominant_taxonomy",
        "narrative_1",
    ):
        assert first[field].startswith("'" + prefix)
    assert verify.spreadsheet_safe_text(prefix + "reviewer note").startswith("'" + prefix)


def test_export_strips_terminal_and_bidi_controls_before_formula_neutralisation(
    verification_fixture,
    tmp_path,
):
    con, signals_run = verification_fixture
    unsafe = "\x1b\u202e=SUM(1,1)"
    con.execute("UPDATE cluster_labels SET harm_mechanism = ?", [unsafe])
    con.execute("UPDATE narratives SET text_redacted = ?", [unsafe])

    path = verify.export_worklist(con, signals_run, 50, 7, tmp_path / "review.csv")
    first = next(csv.DictReader(path.open()))

    assert first["harm_mechanism"] == "'=SUM(1,1)"
    assert first["narrative_1"] == "'=SUM(1,1)"
    assert "\x1b" not in first["harm_mechanism"]
    assert "\u202e" not in first["harm_mechanism"]


def test_spreadsheet_sanitizer_preserves_safe_unicode_joiners_and_combining_marks():
    safe = "café 👩\u200d💻 فارسی\u200c e\u0301"
    assert verify.spreadsheet_safe_text(safe) == safe


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
    verification_fixture,
    tmp_path,
):
    """A smaller worklist cannot be presented as satisfying the human-review gate."""
    con, signals_run = verification_fixture

    with pytest.raises(ValueError, match="at least 50"):
        verify.export_worklist(con, signals_run, 49, 7, tmp_path / "review.csv")


def test_export_rejects_an_eligible_population_smaller_than_requested(
    verification_fixture,
    tmp_path,
):
    """The exporter must not silently downgrade the required human denominator."""
    con, signals_run = verification_fixture

    with pytest.raises(ValueError, match="eligible.*60.*requested 61"):
        verify.export_worklist(con, signals_run, 61, 7, tmp_path / "review.csv")


def test_historical_cluster_run_cannot_enter_worklist_or_human_report(
    verification_fixture,
    tmp_path,
):
    """Controls belong to the selected signals run's cluster refit, never history."""
    con, signals_run = verification_fixture
    historical_cluster = _add_historical_cluster(con)

    rows = list(
        csv.DictReader(
            verify.export_worklist(con, signals_run, 60, 7, tmp_path / "review.csv").open()
        )
    )

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
    row.update(
        {
            "cluster_id": "cluster-1",
            "mechanism_accuracy": "agree",
            "taxonomy_distinctness_accuracy": "agree",
            "template_accuracy": "agree",
            "should_have_abstained": "false",
            "failure_category": "none",
        }
    )
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


@pytest.mark.parametrize("malformation", ["extra", "missing"])
def test_parse_rejects_wrong_physical_row_arity(tmp_path, malformation):
    path = _filled_worklist(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if malformation == "extra":
        lines[1] += ",unexpected"
    else:
        lines[1] = lines[1].rsplit(",", 1)[0]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="field count"):
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
        rows.append(
            verify.Verification(
                cluster_id=f"0000000000001-cluster1:mortgage:{number}",
                reviewer_id=reviewer_id,
                mechanism_accuracy=mechanism,
                taxonomy_distinctness_accuracy="agree",
                template_accuracy="agree",
                should_have_abstained=False,
                failure_category="none" if mechanism == "agree" else "other",
                notes=None,
                reviewer_origin=reviewer_origin,
            )
        )
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


def test_report_uses_stored_fired_snapshot_after_config_change(
    verification_fixture,
    monkeypatch,
):
    con, signals_run = verification_fixture
    verify.record(con, _reviewed_rows(agree=39, total=50), signals_run, "wl-v1")
    original = verify.report(con, "wl-v1")
    monkeypatch.setattr(verify, "canonical_fired_cluster_ids", lambda *_args: set())

    drifted = verify.report(con, "wl-v1")

    assert drifted.by_fired_status == original.by_fired_status
    assert drifted.by_fired_status["fired"].mechanism.total == 25


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
