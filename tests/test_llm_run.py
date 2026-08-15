"""Integration tests for resumable, observable LLM cluster labelling."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from src import pipeline
from src.llm import label as label_mod
from src.llm import run as llm_run
from src.llm.client import ModelCallError, ModelCallResult, TokenUsage


def complete_label() -> dict:
    return {
        "harm_mechanism": "A servicer applies fees after a payment.",
        "actors": ["servicer"],
        "preconditions": "The consumer makes a payment.",
        "consumer_impact": "The consumer pays an unexpected fee.",
        "distinct_from_taxonomy": True,
        "distinctness_rationale": "The existing label does not describe fees.",
        "confidence": "high",
        "is_likely_template": False,
    }


def result(payload: dict) -> ModelCallResult:
    return ModelCallResult(
        payload=payload,
        model="claude-opus-5",
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=12, output_tokens=4),
        attempts=1,
        latency_seconds=0.25,
        estimated_cost_usd=0.00016,
    )


class FakeModelClient:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0
        self.preflights = 0

    def preflight(self, _model):
        self.preflights += 1

    def call_json(self, **_kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def label_fixture(con):
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
    cluster_ids = [f"{cluster_run}:mortgage:{number}" for number in (1, 2)]
    for offset, cluster_id in enumerate(cluster_ids):
        con.execute(
            "INSERT INTO clusters "
            "(cluster_id, run_id, product_family, n_members, centroid_idx, as_of) "
            "VALUES (?, ?, 'mortgage', 30, ?, ?)",
            [cluster_id, cluster_run, offset * 2, date(2020, 1, 1)],
        )
        for complaint_id in (offset * 2 + 1, offset * 2 + 2):
            con.execute(
                "INSERT INTO complaints "
                "(complaint_id, date_received, period_month, product_family, "
                "has_narrative) VALUES (?, ?, ?, 'mortgage', true)",
                [complaint_id, date(2020, 1, 1), date(2020, 1, 1)],
            )
            con.execute(
                "INSERT INTO narratives "
                "(complaint_id, text_redacted, text_hash, redaction_count) "
                "VALUES (?, ?, ?, 0)",
                [complaint_id, f"complaint {complaint_id}", f"hash-{complaint_id}"],
            )
            con.execute(
                "INSERT INTO embedding_map (complaint_id, row_idx, model, dim) "
                "VALUES (?, ?, 'm', 2)",
                [complaint_id, complaint_id - 1],
            )
            con.execute(
                "INSERT INTO cluster_members (cluster_id, complaint_id) VALUES (?, ?)",
                [cluster_id, complaint_id],
            )
    con.execute(
        "INSERT INTO signals "
        "(signal_id, run_id, cluster_id, company_id, period_month, method, statistic, "
        "q_value, n_supporting, n_supporting_groups, as_of) "
        "VALUES ('signal-1', ?, ?, '__ALL__', ?, 'ewma', 1.0, 0.01, 2, 2, ?)",
        [signals_run, cluster_ids[0], date(2020, 1, 1), date(2020, 1, 1)],
    )
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    return con, vectors, cluster_run, signals_run


def test_label_job_resumes_from_cache_without_second_call(label_fixture, tmp_path):
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([result(complete_label())])

    first = llm_run.run(
        con, cluster_run, signals_run, control_n=0, limit=1,
        embed_model="m", client=client, vectors=vectors, cache_dir=tmp_path,
    )
    second = llm_run.run(
        con, cluster_run, signals_run, control_n=0, limit=1,
        embed_model="m", client=client, vectors=vectors, cache_dir=tmp_path,
    )

    assert first.labelled == 1
    assert second.cached == 1
    assert client.calls == 1
    assert client.preflights == 1
    outcomes = con.execute(
        "SELECT cache_status, outcome FROM llm_usage ORDER BY created_at, usage_id"
    ).fetchall()
    assert outcomes == [("miss", "ok"), ("hit", "ok")]


def test_terminal_billing_failure_stops_remaining_population(label_fixture, tmp_path):
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([ModelCallError("billing", 1, False)])

    with pytest.raises(ModelCallError, match="billing"):
        llm_run.run(
            con, cluster_run, signals_run, control_n=1, limit=None,
            embed_model="m", client=client, vectors=vectors, cache_dir=tmp_path,
        )

    assert client.calls == 1
    assert con.execute(
        "SELECT count(*) FROM llm_usage WHERE error_category = 'billing'"
    ).fetchone()[0] == 1


def test_scalar_model_payload_is_recorded_and_next_target_continues(
    label_fixture, tmp_path,
):
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([result(None), result(complete_label())])

    stats = llm_run.run(
        con, cluster_run, signals_run, control_n=1, limit=None,
        embed_model="m", client=client, vectors=vectors, cache_dir=tmp_path,
    )

    assert stats.failed == 1
    assert stats.labelled == 1
    assert client.calls == 2
    assert con.execute(
        "SELECT outcome, error_category FROM llm_usage ORDER BY created_at, usage_id"
    ).fetchall() == [("failed", "schema"), ("ok", None)]


def test_cache_only_replay_never_constructs_a_default_client(
    label_fixture, tmp_path, monkeypatch,
):
    con, vectors, cluster_run, signals_run = label_fixture
    key = label_mod.input_hash("v1", "claude-opus-5", [1, 2])
    label_mod.write_cache(tmp_path, key, complete_label())

    def fail_construction():
        raise AssertionError("cache-only replay must not construct a provider client")

    monkeypatch.setattr(llm_run, "AnthropicModelClient", fail_construction)

    stats = llm_run.run(
        con, cluster_run, signals_run, control_n=0, limit=1,
        embed_model="m", vectors=vectors, cache_dir=tmp_path,
    )

    assert stats.cached == 1
    assert stats.labelled == 1


def _record_review(con, cluster_id: str, signals_run: str) -> None:
    con.execute(
        "INSERT INTO label_verifications "
        "(cluster_id, reviewer_id, reviewer_origin, worklist_version, signals_run, "
        "is_fired, mechanism_accuracy, taxonomy_distinctness_accuracy, "
        "template_accuracy, should_have_abstained, failure_category, reviewed_at) "
        "VALUES (?, 'reviewer-1', 'human', 'wl-v1', ?, true, 'agree', 'agree', "
        "'agree', false, 'none', now())",
        [cluster_id, signals_run],
    )


def test_cache_replay_after_verification_leaves_label_and_review_untouched(
    label_fixture, tmp_path,
):
    """A same-input resume must not delete a label referenced by human review."""
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([result(complete_label())])
    llm_run.run(
        con, cluster_run, signals_run, control_n=0, limit=1,
        embed_model="m", client=client, vectors=vectors, cache_dir=tmp_path,
    )
    cluster_id, generated_at = con.execute(
        "SELECT cluster_id, generated_at FROM cluster_labels"
    ).fetchone()
    _record_review(con, cluster_id, signals_run)

    stats = llm_run.run(
        con, cluster_run, signals_run, control_n=0, limit=1,
        embed_model="m", client=client, vectors=vectors, cache_dir=tmp_path,
    )

    assert stats.cached == 1
    assert stats.labelled == 1
    assert client.calls == 1
    assert con.execute(
        "SELECT generated_at FROM cluster_labels WHERE cluster_id = ?", [cluster_id]
    ).fetchone() == (generated_at,)
    assert con.execute(
        "SELECT count(*) FROM label_verifications WHERE cluster_id = ?", [cluster_id]
    ).fetchone() == (1,)


def test_unreviewed_label_change_updates_in_place(
    label_fixture, tmp_path, monkeypatch,
):
    """An unreviewed cluster may be refreshed without replacement semantics."""
    con, vectors, cluster_run, signals_run = label_fixture
    first_client = FakeModelClient([result(complete_label())])
    llm_run.run(
        con, cluster_run, signals_run, control_n=0, limit=1,
        embed_model="m", client=first_client, vectors=vectors, cache_dir=tmp_path,
    )
    changed_config = replace(
        llm_run.CONFIG,
        llm=replace(llm_run.CONFIG.llm, prompt_version="v2"),
    )
    monkeypatch.setattr(llm_run, "CONFIG", changed_config)
    changed_label = complete_label() | {"confidence": "medium"}
    second_client = FakeModelClient([result(changed_label)])

    stats = llm_run.run(
        con, cluster_run, signals_run, control_n=0, limit=1,
        embed_model="m", client=second_client, vectors=vectors, cache_dir=tmp_path,
    )

    assert stats.labelled == 1
    assert second_client.calls == 1
    assert con.execute(
        "SELECT prompt_version, confidence FROM cluster_labels"
    ).fetchall() == [("v2", "medium")]


def test_reviewed_label_change_is_rejected_before_provider_call(
    label_fixture, tmp_path, monkeypatch,
):
    """A changed key cannot silently attach an old review to a new label."""
    con, vectors, cluster_run, signals_run = label_fixture
    first_client = FakeModelClient([result(complete_label())])
    llm_run.run(
        con, cluster_run, signals_run, control_n=0, limit=1,
        embed_model="m", client=first_client, vectors=vectors, cache_dir=tmp_path,
    )
    cluster_id = con.execute("SELECT cluster_id FROM cluster_labels").fetchone()[0]
    _record_review(con, cluster_id, signals_run)
    changed_config = replace(
        llm_run.CONFIG,
        llm=replace(llm_run.CONFIG.llm, prompt_version="v2"),
    )
    monkeypatch.setattr(llm_run, "CONFIG", changed_config)
    second_client = FakeModelClient([])

    with pytest.raises(llm_run.ReviewedLabelChangeError, match="reviewed cluster"):
        llm_run.run(
            con, cluster_run, signals_run, control_n=0, limit=1,
            embed_model="m", client=second_client, vectors=vectors, cache_dir=tmp_path,
        )

    assert second_client.calls == 0
    assert con.execute(
        "SELECT prompt_version FROM cluster_labels WHERE cluster_id = ?", [cluster_id]
    ).fetchone() == ("v1",)


def test_signals_cluster_run_mismatch_fails_before_population_or_client(
    label_fixture, tmp_path, monkeypatch,
):
    """Label provenance is validated before population work or provider setup."""
    con, vectors, cluster_run, signals_run = label_fixture
    con.execute(
        "UPDATE runs SET params_json = ? WHERE run_id = ?",
        [json.dumps({"params": {"cluster_run": "different-cluster-run"}}), signals_run],
    )

    def fail_population(*_args, **_kwargs):
        raise AssertionError("population must not run after provenance mismatch")

    monkeypatch.setattr(llm_run, "population", fail_population)

    with pytest.raises(ValueError, match="different-cluster-run.*requested"):
        llm_run.run(
            con, cluster_run, signals_run, control_n=0, limit=1,
            embed_model="m", vectors=vectors, cache_dir=tmp_path,
        )


def test_unknown_nonretryable_provider_error_stops_the_batch(
    label_fixture, tmp_path,
):
    """Every terminal provider error stops, including categories unknown to us."""
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([ModelCallError("unexpected", 1, False)])

    with pytest.raises(ModelCallError, match="unexpected"):
        llm_run.run(
            con, cluster_run, signals_run, control_n=1, limit=None,
            embed_model="m", client=client, vectors=vectors, cache_dir=tmp_path,
        )

    assert client.calls == 1
    assert con.execute(
        "SELECT outcome, error_category FROM llm_usage"
    ).fetchall() == [("failed", "unexpected")]


def test_malformed_response_usage_is_recorded_and_next_target_continues(
    label_fixture, tmp_path,
):
    """Paid malformed output remains accounted while later clusters continue."""
    con, vectors, cluster_run, signals_run = label_fixture
    error = ModelCallError(
        "malformed_response", 1, False,
        usage=TokenUsage(input_tokens=21, output_tokens=8),
        latency_seconds=0.75,
        estimated_cost_usd=0.002,
        response_received=True,
    )
    client = FakeModelClient([error, result(complete_label())])

    stats = llm_run.run(
        con, cluster_run, signals_run, control_n=1, limit=None,
        embed_model="m", client=client, vectors=vectors, cache_dir=tmp_path,
    )

    assert stats.failed == 1
    assert stats.labelled == 1
    assert stats.input_tokens == 33
    assert stats.output_tokens == 12
    assert stats.latency_seconds == pytest.approx(1.0)
    assert stats.estimated_cost_usd == pytest.approx(0.00216)
    assert con.execute(
        "SELECT input_tokens, output_tokens, latency_seconds, estimated_cost_usd, "
        "error_category FROM llm_usage ORDER BY created_at, usage_id"
    ).fetchall()[0] == (21, 8, 0.75, 0.002, "malformed_response")


def test_refusal_records_paid_usage_without_writing_a_label(label_fixture, tmp_path):
    """A refusal is auditable per-cluster paid usage, not a batch failure."""
    con, vectors, cluster_run, signals_run = label_fixture
    refusal = result({"refused": True, "stop_reason": "refusal"})
    client = FakeModelClient([refusal])

    stats = llm_run.run(
        con, cluster_run, signals_run, control_n=0, limit=1,
        embed_model="m", client=client, vectors=vectors, cache_dir=tmp_path,
    )

    assert stats.refused == 1
    assert stats.labelled == 0
    assert stats.failed == 0
    assert con.execute("SELECT count(*) FROM cluster_labels").fetchone() == (0,)
    assert con.execute(
        "SELECT input_tokens, output_tokens, outcome FROM llm_usage"
    ).fetchall() == [(12, 4, "refused")]


def test_cache_publication_failure_records_paid_usage_before_reraising(
    label_fixture, tmp_path, monkeypatch,
):
    """A disk failure after a response cannot erase the provider charge."""
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([result(complete_label())])

    def fail_cache(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(label_mod, "write_cache", fail_cache)

    with pytest.raises(OSError, match="disk full"):
        llm_run.run(
            con, cluster_run, signals_run, control_n=0, limit=1,
            embed_model="m", client=client, vectors=vectors, cache_dir=tmp_path,
        )

    assert con.execute(
        "SELECT input_tokens, output_tokens, latency_seconds, estimated_cost_usd, "
        "outcome, error_category FROM llm_usage"
    ).fetchall() == [(12, 4, 0.25, 0.00016, "failed", "cache_write")]


def test_label_phase_passes_its_run_id_to_the_typed_runner(monkeypatch):
    """Usage rows must be attributable to the surrounding pipeline run."""
    captured = {}

    @contextmanager
    def fake_run(*_args, **_kwargs):
        yield SimpleNamespace(run_id="label-run", finish=lambda **_kwargs: None)

    def fake_label_run(*_args, **kwargs):
        captured.update(kwargs)
        return llm_run.LabelRunStats()

    monkeypatch.setattr(pipeline.db, "bootstrap", lambda: object())
    monkeypatch.setattr(pipeline.db, "run", fake_run)
    monkeypatch.setattr(llm_run, "run", fake_label_run)
    args = SimpleNamespace(
        run_id="cluster-run", signals_run="signals-run", model=None,
        control_n=0, limit=1,
    )

    pipeline.phase_label(args)

    assert captured["run_id"] == "label-run"
