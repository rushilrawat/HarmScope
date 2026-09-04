"""Integration tests for resumable, observable LLM cluster labelling."""

from __future__ import annotations

import json
import stat
from contextlib import contextmanager, suppress
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


def _staged_usage_events(cache_dir):
    return sorted(cache_dir.glob("label-usage-*.json"))


@pytest.fixture
def label_fixture(con):
    cluster_run = "0000000000001-cluster1"
    signals_run = "0000000000002-signal01"
    for run_id, phase in ((cluster_run, "cluster"), (signals_run, "signals")):
        params = (
            json.dumps({"params": {"cluster_run": cluster_run}})
            if phase == "signals"
            else json.dumps({"params": {"model": "m"}})
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
            "(cluster_id, run_id, product_family, n_members, centroid_idx, coherence, as_of) "
            "VALUES (?, ?, 'mortgage', 30, ?, 0.9, ?)",
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
        "VALUES ('signal-1', ?, ?, '__ALL__', ?, 'ewma', 1.0, NULL, 20, 15, ?)",
        [signals_run, cluster_ids[0], date(2020, 1, 1), date(2020, 1, 1)],
    )
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    return con, vectors, cluster_run, signals_run


def test_label_job_resumes_from_cache_without_second_call(label_fixture, tmp_path):
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([result(complete_label())])

    first = llm_run.run(
        con,
        cluster_run,
        signals_run,
        control_n=0,
        limit=1,
        embed_model="m",
        client=client,
        vectors=vectors,
        cache_dir=tmp_path,
    )
    second = llm_run.run(
        con,
        cluster_run,
        signals_run,
        control_n=0,
        limit=1,
        embed_model="m",
        client=client,
        vectors=vectors,
        cache_dir=tmp_path,
    )

    assert first.labelled == 1
    assert second.cached == 1
    assert client.calls == 1
    assert client.preflights == 1
    outcomes = con.execute(
        "SELECT cache_status, outcome FROM llm_usage ORDER BY created_at, usage_id"
    ).fetchall()
    assert outcomes == [("miss", "ok"), ("hit", "ok")]
    assert _staged_usage_events(tmp_path) == []


def test_population_uses_support_coherence_and_q_or_changepoint(label_fixture):
    """EWMA-only coherent/supportable scopes fire; weak q-only scopes stay controls."""
    con, _vectors, cluster_run, signals_run = label_fixture
    quiet_cluster = f"{cluster_run}:mortgage:2"
    con.execute(
        "INSERT INTO signals "
        "(signal_id, run_id, cluster_id, company_id, period_month, method, statistic, "
        "q_value, n_supporting, n_supporting_groups, as_of) "
        "VALUES ('signal-weak-q', ?, ?, 'company-2', DATE '2020-01-01', "
        "'ebgm', 1.0, 0.01, 20, 14, DATE '2020-01-01')",
        [signals_run, quiet_cluster],
    )

    rows, n_fired, n_control = llm_run.population(con, cluster_run, signals_run, 2)

    status = {row[0]: bool(row[5]) for row in rows}
    assert status == {
        f"{cluster_run}:mortgage:1": True,
        quiet_cluster: False,
    }
    assert (n_fired, n_control) == (1, 1)


def test_label_rejects_wrong_embedding_model_before_vectors_cache_or_provider(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    con, _vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([])
    touched = []
    monkeypatch.setattr(llm_run, "population", lambda *_args: touched.append("population"))

    with pytest.raises(ValueError, match="records embedding model 'm'.*requested 'wrong'"):
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="wrong",
            client=client,
            cache_dir=tmp_path,
        )

    assert touched == []
    assert client.calls == 0
    assert list(tmp_path.glob("label_usage_outbox")) == []
    assert list(tmp_path.glob("*.json")) == []
    assert con.execute("SELECT count(*) FROM llm_usage").fetchone() == (0,)


def test_label_input_hash_binds_evidence_embedding_model():
    assert label_mod.input_hash("v1", "llm", [1, 2], "embed-a") != label_mod.input_hash(
        "v1", "llm", [1, 2], "embed-b"
    )


def test_terminal_billing_failure_stops_remaining_population(label_fixture, tmp_path):
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([ModelCallError("billing", 1, False)])

    with pytest.raises(ModelCallError, match="billing"):
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=1,
            limit=None,
            embed_model="m",
            client=client,
            vectors=vectors,
            cache_dir=tmp_path,
        )

    assert client.calls == 1
    assert (
        con.execute("SELECT count(*) FROM llm_usage WHERE error_category = 'billing'").fetchone()[0]
        == 1
    )


def test_scalar_model_payload_is_recorded_and_next_target_continues(
    label_fixture,
    tmp_path,
):
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([result(None), result(complete_label())])

    stats = llm_run.run(
        con,
        cluster_run,
        signals_run,
        control_n=1,
        limit=None,
        embed_model="m",
        client=client,
        vectors=vectors,
        cache_dir=tmp_path,
    )

    assert stats.failed == 1
    assert stats.labelled == 1
    assert client.calls == 2
    assert con.execute(
        "SELECT outcome, error_category FROM llm_usage ORDER BY created_at, usage_id"
    ).fetchall() == [("failed", "schema"), ("ok", None)]


def test_cache_only_replay_never_constructs_a_default_client(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    con, vectors, cluster_run, signals_run = label_fixture
    key = label_mod.input_hash("v1", "claude-opus-5", [1, 2], "m")
    label_mod.write_cache(tmp_path, key, complete_label())

    def fail_construction():
        raise AssertionError("cache-only replay must not construct a provider client")

    monkeypatch.setattr(llm_run, "AnthropicModelClient", fail_construction)

    stats = llm_run.run(
        con,
        cluster_run,
        signals_run,
        control_n=0,
        limit=1,
        embed_model="m",
        vectors=vectors,
        cache_dir=tmp_path,
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
    label_fixture,
    tmp_path,
):
    """A same-input resume must not delete a label referenced by human review."""
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([result(complete_label())])
    llm_run.run(
        con,
        cluster_run,
        signals_run,
        control_n=0,
        limit=1,
        embed_model="m",
        client=client,
        vectors=vectors,
        cache_dir=tmp_path,
    )
    cluster_id, generated_at = con.execute(
        "SELECT cluster_id, generated_at FROM cluster_labels"
    ).fetchone()
    _record_review(con, cluster_id, signals_run)

    stats = llm_run.run(
        con,
        cluster_run,
        signals_run,
        control_n=0,
        limit=1,
        embed_model="m",
        client=client,
        vectors=vectors,
        cache_dir=tmp_path,
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
    label_fixture,
    tmp_path,
    monkeypatch,
):
    """An unreviewed cluster may be refreshed without replacement semantics."""
    con, vectors, cluster_run, signals_run = label_fixture
    first_client = FakeModelClient([result(complete_label())])
    llm_run.run(
        con,
        cluster_run,
        signals_run,
        control_n=0,
        limit=1,
        embed_model="m",
        client=first_client,
        vectors=vectors,
        cache_dir=tmp_path,
    )
    changed_config = replace(
        llm_run.CONFIG,
        llm=replace(llm_run.CONFIG.llm, prompt_version="v2"),
    )
    monkeypatch.setattr(llm_run, "CONFIG", changed_config)
    changed_label = complete_label() | {"confidence": "medium"}
    second_client = FakeModelClient([result(changed_label)])

    stats = llm_run.run(
        con,
        cluster_run,
        signals_run,
        control_n=0,
        limit=1,
        embed_model="m",
        client=second_client,
        vectors=vectors,
        cache_dir=tmp_path,
    )

    assert stats.labelled == 1
    assert second_client.calls == 1
    assert con.execute("SELECT prompt_version, confidence FROM cluster_labels").fetchall() == [
        ("v2", "medium")
    ]


def test_reviewed_label_change_is_rejected_before_provider_call(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    """A changed key cannot silently attach an old review to a new label."""
    con, vectors, cluster_run, signals_run = label_fixture
    first_client = FakeModelClient([result(complete_label())])
    llm_run.run(
        con,
        cluster_run,
        signals_run,
        control_n=0,
        limit=1,
        embed_model="m",
        client=first_client,
        vectors=vectors,
        cache_dir=tmp_path,
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
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=second_client,
            vectors=vectors,
            cache_dir=tmp_path,
        )

    assert second_client.calls == 0
    assert con.execute(
        "SELECT prompt_version FROM cluster_labels WHERE cluster_id = ?", [cluster_id]
    ).fetchone() == ("v1",)


def test_signals_cluster_run_mismatch_fails_before_population_or_client(
    label_fixture,
    tmp_path,
    monkeypatch,
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
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            vectors=vectors,
            cache_dir=tmp_path,
        )


def test_failed_signals_run_is_rejected_before_population_or_client(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    """A failed historical signals run cannot define labels or quiet controls."""
    con, vectors, cluster_run, signals_run = label_fixture
    con.execute("UPDATE runs SET status = 'failed' WHERE run_id = ?", [signals_run])
    touched: list[str] = []

    def fail_population(*_args, **_kwargs):
        touched.append("population")
        raise AssertionError("population must not run for a failed signals run")

    monkeypatch.setattr(llm_run, "population", fail_population)

    with pytest.raises(ValueError, match="successful signals run"):
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=FakeModelClient([]),
            vectors=vectors,
            cache_dir=tmp_path,
        )

    assert touched == []
    assert con.execute("SELECT count(*) FROM llm_usage").fetchone() == (0,)


def test_unknown_nonretryable_provider_error_stops_the_batch(
    label_fixture,
    tmp_path,
):
    """Every terminal provider error stops, including categories unknown to us."""
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([ModelCallError("unexpected", 1, False)])

    with pytest.raises(ModelCallError, match="unexpected"):
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=1,
            limit=None,
            embed_model="m",
            client=client,
            vectors=vectors,
            cache_dir=tmp_path,
        )

    assert client.calls == 1
    assert con.execute("SELECT outcome, error_category FROM llm_usage").fetchall() == [
        ("failed", "unexpected")
    ]


def test_malformed_response_usage_is_recorded_and_next_target_continues(
    label_fixture,
    tmp_path,
):
    """Paid malformed output remains accounted while later clusters continue."""
    con, vectors, cluster_run, signals_run = label_fixture
    error = ModelCallError(
        "malformed_response",
        1,
        False,
        usage=TokenUsage(input_tokens=21, output_tokens=8),
        latency_seconds=0.75,
        estimated_cost_usd=0.002,
        response_received=True,
    )
    client = FakeModelClient([error, result(complete_label())])

    stats = llm_run.run(
        con,
        cluster_run,
        signals_run,
        control_n=1,
        limit=None,
        embed_model="m",
        client=client,
        vectors=vectors,
        cache_dir=tmp_path,
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
        con,
        cluster_run,
        signals_run,
        control_n=0,
        limit=1,
        embed_model="m",
        client=client,
        vectors=vectors,
        cache_dir=tmp_path,
    )

    assert stats.refused == 1
    assert stats.labelled == 0
    assert stats.failed == 0
    assert con.execute("SELECT count(*) FROM cluster_labels").fetchone() == (0,)
    assert con.execute("SELECT input_tokens, output_tokens, outcome FROM llm_usage").fetchall() == [
        (12, 4, "refused")
    ]


def test_cache_publication_failure_records_paid_usage_before_reraising(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    """A disk failure after a response cannot erase the provider charge."""
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([result(complete_label())])

    def fail_cache(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(label_mod, "write_cache", fail_cache)

    with pytest.raises(OSError, match="disk full"):
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=client,
            vectors=vectors,
            cache_dir=tmp_path,
        )

    assert con.execute(
        "SELECT input_tokens, output_tokens, latency_seconds, estimated_cost_usd, "
        "outcome, error_category FROM llm_usage"
    ).fetchall() == [(12, 4, 0.25, 0.00016, "failed", "cache_write")]


def test_schema_failure_falls_back_to_database_when_outbox_stage_fails(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    """A paid malformed response remains charged when staging is unavailable."""
    con, vectors, cluster_run, signals_run = label_fixture

    def fail_stage(*_args, **_kwargs):
        raise OSError("outbox unavailable")

    monkeypatch.setattr(llm_run, "_stage_usage", fail_stage)

    with pytest.raises(label_mod.LabelSchemaError, match="label must be an object") as raised:
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=FakeModelClient([result(None)]),
            vectors=vectors,
            cache_dir=tmp_path,
        )

    assert any(
        "staging failed after durable database fallback" in note for note in raised.value.__notes__
    )
    assert con.execute(
        "SELECT input_tokens, output_tokens, estimated_cost_usd, outcome, error_category "
        "FROM llm_usage"
    ).fetchall() == [(12, 4, 0.00016, "failed", "schema")]
    assert con.execute("SELECT count(*) FROM cluster_labels").fetchone() == (0,)


def test_schema_failure_preserves_original_error_when_stage_and_database_fail(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    con, vectors, cluster_run, signals_run = label_fixture

    monkeypatch.setattr(
        llm_run,
        "_stage_usage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("outbox unavailable")),
    )
    monkeypatch.setattr(
        llm_run,
        "record_usage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("database unavailable")),
    )

    with pytest.raises(label_mod.LabelSchemaError, match="label must be an object") as raised:
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=FakeModelClient([result(None)]),
            vectors=vectors,
            cache_dir=tmp_path,
        )

    assert any("staging and database fallback failed" in note for note in raised.value.__notes__)
    assert con.execute("SELECT count(*) FROM llm_usage").fetchone() == (0,)


def test_cache_failure_falls_back_to_database_when_failed_restaging_fails(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    con, vectors, cluster_run, signals_run = label_fixture
    original_stage = llm_run._stage_usage
    stage_calls = 0

    def fail_second_stage(*args, **kwargs):
        nonlocal stage_calls
        stage_calls += 1
        if stage_calls == 2:
            raise OSError("failed-event restage unavailable")
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(llm_run, "_stage_usage", fail_second_stage)
    monkeypatch.setattr(
        label_mod,
        "write_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cache unavailable")),
    )

    with pytest.raises(OSError, match="cache unavailable"):
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=FakeModelClient([result(complete_label())]),
            vectors=vectors,
            cache_dir=tmp_path,
        )

    assert con.execute(
        "SELECT input_tokens, output_tokens, estimated_cost_usd, outcome, error_category "
        "FROM llm_usage"
    ).fetchall() == [(12, 4, 0.00016, "failed", "cache_write")]


def test_cache_failure_keeps_correct_failed_event_when_restaging_and_database_fail(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    """A later replay must not turn a known cache failure into an `ok` event."""
    con, vectors, cluster_run, signals_run = label_fixture
    original_stage = llm_run._stage_usage
    stage_calls = 0

    def fail_second_stage(*args, **kwargs):
        nonlocal stage_calls
        stage_calls += 1
        if stage_calls == 2:
            raise OSError("failed-event restage unavailable")
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(llm_run, "_stage_usage", fail_second_stage)
    monkeypatch.setattr(
        label_mod,
        "write_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cache unavailable")),
    )
    original_record_usage = llm_run.record_usage
    monkeypatch.setattr(
        llm_run,
        "record_usage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("database unavailable")),
    )

    with pytest.raises(OSError, match="cache unavailable"):
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=FakeModelClient([result(complete_label())]),
            vectors=vectors,
            cache_dir=tmp_path,
        )

    staged = _staged_usage_events(tmp_path)
    assert len(staged) == 1
    payload = json.loads(staged[0].read_text(encoding="utf-8"))
    assert (payload["outcome"], payload["error_category"]) == ("failed", "cache_write")
    assert con.execute("SELECT count(*) FROM llm_usage").fetchone() == (0,)

    monkeypatch.setattr(llm_run, "record_usage", original_record_usage)
    assert llm_run.drain_usage_outbox(con, tmp_path) == 1
    assert llm_run.drain_usage_outbox(con, tmp_path) == 0
    assert con.execute(
        "SELECT count(*), min(outcome), min(error_category), sum(input_tokens), "
        "sum(estimated_cost_usd) FROM llm_usage"
    ).fetchone() == (1, "failed", "cache_write", 12, 0.00016)
    assert _staged_usage_events(tmp_path) == []


def test_cache_failure_upgrades_same_usage_id_already_persisted_as_ok(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    con, vectors, cluster_run, signals_run = label_fixture

    def persist_ok_then_fail(*_args, **_kwargs):
        staged = _staged_usage_events(tmp_path)
        assert len(staged) == 1
        usage = llm_run._usage_from_payload(json.loads(staged[0].read_text(encoding="utf-8")))
        llm_run.record_usage(con, usage)
        raise OSError("cache unavailable after accounting race")

    monkeypatch.setattr(label_mod, "write_cache", persist_ok_then_fail)

    with pytest.raises(OSError, match="cache unavailable"):
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=FakeModelClient([result(complete_label())]),
            vectors=vectors,
            cache_dir=tmp_path,
        )

    assert con.execute(
        "SELECT count(*), min(outcome), min(error_category), sum(input_tokens), "
        "sum(estimated_cost_usd) FROM llm_usage"
    ).fetchone() == (1, "failed", "cache_write", 12, 0.00016)
    assert _staged_usage_events(tmp_path) == []


def test_paid_label_usage_replays_after_cache_publication_and_database_failure(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    """A published cache can never turn a paid response into a zero-cost replay."""
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([result(complete_label())])
    original_record_usage = llm_run.record_usage

    def fail_usage(*_args, **_kwargs):
        raise OSError("database unavailable")

    monkeypatch.setattr(llm_run, "record_usage", fail_usage)
    with pytest.raises(OSError, match="database unavailable"):
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=client,
            vectors=vectors,
            cache_dir=tmp_path,
        )

    assert (
        len([path for path in tmp_path.glob("*.json") if not path.name.startswith("label-usage-")])
        == 1
    )
    staged = _staged_usage_events(tmp_path)
    assert len(staged) == 1
    assert "complaint" not in staged[0].read_text(encoding="utf-8").lower()
    assert con.execute("SELECT count(*) FROM llm_usage").fetchone() == (0,)

    monkeypatch.setattr(llm_run, "record_usage", original_record_usage)
    replay = llm_run.run(
        con,
        cluster_run,
        signals_run,
        control_n=0,
        limit=1,
        embed_model="m",
        client=client,
        vectors=vectors,
        cache_dir=tmp_path,
    )

    assert replay.cached == 1
    assert client.calls == 1
    assert con.execute(
        "SELECT cache_status, input_tokens, output_tokens FROM llm_usage "
        "ORDER BY input_tokens DESC, cache_status"
    ).fetchall() == [("miss", 12, 4), ("hit", 0, 0)]
    assert _staged_usage_events(tmp_path) == []


def test_label_usage_outbox_rejects_caller_transaction_before_file_mutation(
    label_fixture,
    tmp_path,
):
    con, vectors, cluster_run, signals_run = label_fixture
    con.execute("BEGIN TRANSACTION")
    try:
        with pytest.raises(llm_run.LabelTransactionError, match="autocommit"):
            llm_run.run(
                con,
                cluster_run,
                signals_run,
                control_n=0,
                limit=1,
                embed_model="m",
                client=FakeModelClient([]),
                vectors=vectors,
                cache_dir=tmp_path,
            )
        assert list(tmp_path.glob("label_usage_outbox")) == []
        assert list(tmp_path.glob("*.json")) == []
    finally:
        con.execute("ROLLBACK")


def test_label_usage_outbox_replay_is_idempotent_after_commit_before_unlink(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    """A crash after the DB commit may replay, but cannot double-count the charge."""
    con, vectors, cluster_run, signals_run = label_fixture
    client = FakeModelClient([result(complete_label())])
    original_remove = llm_run._remove_staged_usage

    def crash_before_unlink(_path):
        raise RuntimeError("simulated crash before unlink")

    monkeypatch.setattr(llm_run, "_remove_staged_usage", crash_before_unlink)
    with pytest.raises(RuntimeError, match="crash before unlink"):
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=client,
            vectors=vectors,
            cache_dir=tmp_path,
        )

    assert con.execute("SELECT count(*) FROM llm_usage WHERE input_tokens = 12").fetchone() == (1,)
    assert len(_staged_usage_events(tmp_path)) == 1

    monkeypatch.setattr(llm_run, "_remove_staged_usage", original_remove)
    replay = llm_run.run(
        con,
        cluster_run,
        signals_run,
        control_n=0,
        limit=1,
        embed_model="m",
        client=client,
        vectors=vectors,
        cache_dir=tmp_path,
    )

    assert replay.cached == 1
    assert client.calls == 1
    assert con.execute(
        "SELECT input_tokens, count(*) FROM llm_usage GROUP BY input_tokens ORDER BY 1"
    ).fetchall() == [(0, 1), (12, 1)]
    assert _staged_usage_events(tmp_path) == []


def test_corrupt_label_usage_outbox_fails_closed_before_cache_or_provider(
    label_fixture,
    tmp_path,
):
    con, vectors, cluster_run, signals_run = label_fixture
    event = tmp_path / "label-usage-opaque.json"
    event.write_text('{"unexpected":"private narrative must not appear in errors"}')
    client = FakeModelClient([])

    with pytest.raises(llm_run.LabelUsageOutboxError) as raised:
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=client,
            vectors=vectors,
            cache_dir=tmp_path,
        )

    assert "private narrative" not in str(raised.value)
    assert event.exists()
    assert client.calls == 0
    assert con.execute("SELECT count(*) FROM llm_usage").fetchone() == (0,)


def test_label_usage_outbox_directory_creation_can_retry_after_fsync_failure(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    original_fsync = llm_run.os.fsync
    calls = 0

    def fail_once(descriptor):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated parent fsync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(llm_run.os, "fsync", fail_once)
    with pytest.raises(OSError, match="parent fsync failure"):
        llm_run._stage_usage(usage, cache)

    staged = llm_run._stage_usage(usage, cache)
    assert staged.name == f"label-usage-{usage.usage_id}.json"
    assert "private" not in staged.read_text(encoding="utf-8")


def test_label_usage_outbox_rejects_symlinked_directory_without_outside_write(
    tmp_path,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "label_usage_outbox").symlink_to(outside, target_is_directory=True)

    staged = llm_run._stage_usage(usage, tmp_path)

    assert staged.parent == tmp_path
    assert list(outside.iterdir()) == []


def test_label_usage_outbox_rejects_symlinked_event_without_database_write(
    label_fixture,
    tmp_path,
):
    con, _vectors, _cluster_run, _signals_run = label_fixture
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    outside_event = tmp_path / "outside-event.json"
    outside_event.write_text(
        json.dumps(llm_run._usage_payload(usage), sort_keys=True), encoding="utf-8"
    )
    (tmp_path / f"label-usage-{usage.usage_id}.json").symlink_to(outside_event)

    with pytest.raises(llm_run.LabelUsageOutboxError):
        llm_run.drain_usage_outbox(con, tmp_path)

    assert con.execute("SELECT count(*) FROM llm_usage").fetchone() == (0,)
    assert outside_event.exists()


def test_label_usage_outbox_rejects_cache_root_swap_before_event_write(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    moved = tmp_path / "outside-moved-cache"
    original_verify = llm_run._verify_cache_root
    calls = 0

    def swap_root(parent_fd, root_fd, root_name):
        nonlocal calls
        calls += 1
        if calls == 2:
            cache.rename(moved)
            cache.mkdir()
        return original_verify(parent_fd, root_fd, root_name)

    monkeypatch.setattr(llm_run, "_verify_cache_root", swap_root)

    with pytest.raises(llm_run.LabelUsageOutboxError, match="cache root changed"):
        llm_run._stage_usage(usage, cache)

    assert _staged_usage_events(moved) == []
    assert _staged_usage_events(cache) == []


@pytest.mark.parametrize("previous", [None, b"exact previous event bytes\n"])
def test_label_usage_stage_rolls_back_when_cache_root_moves_inside_replace(
    tmp_path,
    monkeypatch,
    previous,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    if previous is not None:
        event.write_bytes(previous)
    moved = tmp_path / "outside-moved-cache"
    original_replace = llm_run.os.replace
    swapped = False

    def swap_then_replace(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            cache.rename(moved)
            cache.mkdir()
        return original_replace(*args, **kwargs)

    monkeypatch.setattr(llm_run.os, "replace", swap_then_replace)

    with pytest.raises(llm_run.LabelUsageOutboxError, match="cache root changed"):
        llm_run._stage_usage(usage, cache)

    moved_event = moved / event.name
    assert (moved_event.read_bytes() if moved_event.exists() else None) == previous
    assert not (cache / event.name).exists()
    assert sorted(item.name for item in moved.iterdir()) == ([event.name] if previous else [])
    assert list(cache.iterdir()) == []


def test_label_usage_stage_fsyncs_each_backup_namespace_change_in_order(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    event.write_bytes(b"old\n")
    operations: list[str] = []
    original_link = llm_run.os.link
    original_replace = llm_run.os.replace
    original_unlink = llm_run.os.unlink
    original_fsync = llm_run.os.fsync
    original_verify = llm_run._verify_cache_root

    def observe_link(*args, **kwargs):
        operations.append("backup-link")
        return original_link(*args, **kwargs)

    def observe_replace(*args, **kwargs):
        operations.append("publish-replace")
        return original_replace(*args, **kwargs)

    def observe_unlink(name, *args, **kwargs):
        if str(name).endswith(".bak"):
            operations.append("backup-unlink")
        return original_unlink(name, *args, **kwargs)

    def observe_fsync(descriptor):
        if stat.S_ISDIR(llm_run.os.fstat(descriptor).st_mode):
            operations.append("directory-fsync")
        return original_fsync(descriptor)

    def observe_verify(*args, **kwargs):
        if operations[-2:] == ["backup-unlink", "directory-fsync"]:
            operations.append("final-verify")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(llm_run.os, "link", observe_link)
    monkeypatch.setattr(llm_run.os, "replace", observe_replace)
    monkeypatch.setattr(llm_run.os, "unlink", observe_unlink)
    monkeypatch.setattr(llm_run.os, "fsync", observe_fsync)
    monkeypatch.setattr(llm_run, "_verify_cache_root", observe_verify)

    llm_run._stage_usage_direct(usage, cache)

    assert operations == [
        "directory-fsync",
        "backup-link",
        "directory-fsync",
        "publish-replace",
        "directory-fsync",
        "backup-unlink",
        "directory-fsync",
        "final-verify",
    ]


def test_label_usage_stage_fsyncs_prepublication_backup_cleanup(tmp_path, monkeypatch):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    event.write_bytes(b"old\n")
    operations: list[str] = []
    original_link = llm_run.os.link
    original_unlink = llm_run.os.unlink
    original_fsync = llm_run.os.fsync

    def observe_link(*args, **kwargs):
        operations.append("backup-link")
        return original_link(*args, **kwargs)

    def fail_replace(*_args, **_kwargs):
        operations.append("publish-replace")
        raise OSError("publication failed")

    def observe_unlink(name, *args, **kwargs):
        if str(name).endswith(".bak"):
            operations.append("backup-unlink")
        return original_unlink(name, *args, **kwargs)

    def observe_fsync(descriptor):
        if stat.S_ISDIR(llm_run.os.fstat(descriptor).st_mode):
            operations.append("directory-fsync")
        return original_fsync(descriptor)

    monkeypatch.setattr(llm_run.os, "link", observe_link)
    monkeypatch.setattr(llm_run.os, "replace", fail_replace)
    monkeypatch.setattr(llm_run.os, "unlink", observe_unlink)
    monkeypatch.setattr(llm_run.os, "fsync", observe_fsync)

    with pytest.raises(OSError, match="publication failed"):
        llm_run._stage_usage_direct(usage, cache)

    assert operations == [
        "directory-fsync",
        "backup-link",
        "directory-fsync",
        "publish-replace",
        "backup-unlink",
        "directory-fsync",
    ]
    assert event.read_bytes() == b"old\n"
    assert sorted(item.name for item in cache.iterdir()) == [event.name]


def test_label_usage_stage_reports_final_backup_cleanup_fsync_failure_consistently(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    event.write_bytes(b"old\n")
    original_unlink = llm_run.os.unlink
    original_fsync = llm_run.os.fsync
    backup_unlinked = False
    failed = False

    def observe_unlink(name, *args, **kwargs):
        nonlocal backup_unlinked
        result = original_unlink(name, *args, **kwargs)
        if str(name).endswith(".bak"):
            backup_unlinked = True
        return result

    def fail_cleanup_fsync(descriptor):
        nonlocal failed
        if backup_unlinked and not failed and stat.S_ISDIR(llm_run.os.fstat(descriptor).st_mode):
            failed = True
            raise OSError("final cleanup fsync failed")
        return original_fsync(descriptor)

    monkeypatch.setattr(llm_run.os, "unlink", observe_unlink)
    monkeypatch.setattr(llm_run.os, "fsync", fail_cleanup_fsync)

    with pytest.raises(OSError, match="final cleanup fsync failed"):
        llm_run._stage_usage_direct(usage, cache)

    assert event.read_text() != "old\n"
    assert sorted(item.name for item in cache.iterdir()) == [event.name]


def test_label_usage_stage_preserves_backup_collision_when_temp_cleanup_also_fails(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    event.write_bytes(b"old\n")
    nonce = "fixed-nonce"
    backup = cache / f".{event.name}.{nonce}.bak"
    backup.write_bytes(b"preexisting collision\n")
    original_unlink = llm_run.os.unlink
    original_fsync = llm_run.os.fsync
    directory_fsyncs = 0

    def fail_temp_unlink(name, *args, **kwargs):
        if str(name).endswith(".tmp"):
            raise OSError("secret prose and /private/path must not leak")
        return original_unlink(name, *args, **kwargs)

    def observe_fsync(descriptor):
        nonlocal directory_fsyncs
        if stat.S_ISDIR(llm_run.os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
        return original_fsync(descriptor)

    monkeypatch.setattr(llm_run.db, "new_run_id", lambda: nonce)
    monkeypatch.setattr(llm_run.os, "unlink", fail_temp_unlink)
    monkeypatch.setattr(llm_run.os, "fsync", observe_fsync)

    with pytest.raises(FileExistsError) as raised:
        llm_run._stage_usage_direct(usage, cache)

    notes = getattr(raised.value, "__notes__", [])
    assert notes == ["label usage publication recovery also failed: OSError"]
    assert "secret prose" not in notes[0]
    assert "/private/path" not in notes[0]
    assert directory_fsyncs == 2
    assert event.read_bytes() == b"old\n"
    assert backup.read_bytes() == b"preexisting collision\n"
    assert len(list(cache.glob(f".{event.name}.{nonce}.tmp"))) == 1


def test_label_usage_stage_rolls_back_when_root_moves_during_backup_unlink(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    previous = b"exact previous event bytes\n"
    event.write_bytes(previous)
    moved = tmp_path / "outside-moved-cache"
    original_unlink = llm_run.os.unlink
    original_fsync = llm_run.os.fsync
    swapped = False
    post_swap_directory_fsyncs = 0

    def unlink_then_swap_root(name, *args, **kwargs):
        nonlocal swapped
        result = original_unlink(name, *args, **kwargs)
        if str(name).endswith(".bak") and not swapped:
            swapped = True
            cache.rename(moved)
            cache.mkdir()
        return result

    def observe_fsync(descriptor):
        nonlocal post_swap_directory_fsyncs
        if swapped and stat.S_ISDIR(llm_run.os.fstat(descriptor).st_mode):
            post_swap_directory_fsyncs += 1
        return original_fsync(descriptor)

    monkeypatch.setattr(llm_run.os, "unlink", unlink_then_swap_root)
    monkeypatch.setattr(llm_run.os, "fsync", observe_fsync)

    with pytest.raises(llm_run.LabelUsageOutboxError, match="cache root changed"):
        llm_run._stage_usage_direct(usage, cache)

    assert list(cache.iterdir()) == []
    assert (moved / event.name).read_bytes() == previous
    assert sorted(item.name for item in moved.iterdir()) == [event.name]
    assert post_swap_directory_fsyncs == 2


def test_label_usage_stage_final_verify_removes_new_event_after_no_old_root_swap(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    moved = tmp_path / "outside-moved-cache"
    original_replace = llm_run.os.replace
    original_fsync = llm_run.os.fsync
    original_verify = llm_run._verify_cache_root
    published = False
    swapped = False
    recovery_directory_fsyncs = 0

    def observe_replace(*args, **kwargs):
        nonlocal published
        result = original_replace(*args, **kwargs)
        published = True
        return result

    def verify_then_swap_root(*args, **kwargs):
        nonlocal swapped
        result = original_verify(*args, **kwargs)
        if published and not swapped:
            swapped = True
            cache.rename(moved)
            cache.mkdir()
        return result

    def observe_fsync(descriptor):
        nonlocal recovery_directory_fsyncs
        if swapped and stat.S_ISDIR(llm_run.os.fstat(descriptor).st_mode):
            recovery_directory_fsyncs += 1
        return original_fsync(descriptor)

    monkeypatch.setattr(llm_run.os, "replace", observe_replace)
    monkeypatch.setattr(llm_run.os, "fsync", observe_fsync)
    monkeypatch.setattr(llm_run, "_verify_cache_root", verify_then_swap_root)

    with pytest.raises(llm_run.LabelUsageOutboxError, match="cache root changed"):
        llm_run._stage_usage_direct(usage, cache)

    assert list(cache.iterdir()) == []
    assert list(moved.iterdir()) == []
    assert recovery_directory_fsyncs == 1


def test_label_usage_recovery_uses_original_inode_not_replaced_backup_name(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    previous = b"exact original event bytes\n"
    event.write_bytes(previous)
    moved = tmp_path / "outside-moved-cache"
    original_open = llm_run.os.open
    original_unlink = llm_run.os.unlink
    substituted = False
    swapped = False

    def substitute_backup_before_open(name, flags, *args, **kwargs):
        nonlocal substituted
        if str(name).endswith(".bak") and not substituted:
            substituted = True
            original_unlink(name, dir_fd=kwargs["dir_fd"])
            substitute_fd = original_open(
                name,
                llm_run.os.O_WRONLY | llm_run.os.O_CREAT | llm_run.os.O_EXCL,
                0o600,
                dir_fd=kwargs["dir_fd"],
            )
            llm_run.os.write(substitute_fd, b"substitute event bytes\n")
            llm_run.os.close(substitute_fd)
        return original_open(name, flags, *args, **kwargs)

    def unlink_then_swap_root(name, *args, **kwargs):
        nonlocal swapped
        result = original_unlink(name, *args, **kwargs)
        if str(name).endswith(".bak") and not swapped:
            swapped = True
            cache.rename(moved)
            cache.mkdir()
        return result

    monkeypatch.setattr(llm_run.os, "open", substitute_backup_before_open)
    monkeypatch.setattr(llm_run.os, "unlink", unlink_then_swap_root)

    with pytest.raises(llm_run.LabelUsageOutboxError, match="cache root changed"):
        llm_run._stage_usage_direct(usage, cache)

    assert list(cache.iterdir()) == []
    assert (moved / event.name).read_bytes() == previous
    assert sorted(item.name for item in moved.iterdir()) == [event.name]


def test_label_usage_rejects_destination_swap_between_pin_and_backup_link(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    previous = b"exact original event bytes\n"
    event.write_bytes(previous)
    original_open = llm_run.os.open
    original_link = llm_run.os.link
    original_pinned = False

    def observe_original_open(name, flags, *args, **kwargs):
        nonlocal original_pinned
        result = original_open(name, flags, *args, **kwargs)
        if name == event.name and flags & llm_run.os.O_ACCMODE == llm_run.os.O_RDONLY:
            original_pinned = True
        return result

    def swap_then_link(*args, **kwargs):
        if original_pinned:
            substitute = cache / "substitute"
            substitute.write_bytes(b"substitute event bytes\n")
            llm_run.os.replace(substitute, event)
        return original_link(*args, **kwargs)

    monkeypatch.setattr(llm_run.os, "open", observe_original_open)
    monkeypatch.setattr(llm_run.os, "link", swap_then_link)

    with pytest.raises(llm_run.LabelUsageOutboxError, match="changed during publication"):
        llm_run._stage_usage_direct(usage, cache)

    assert event.read_bytes() == previous
    assert sorted(item.name for item in cache.iterdir()) == [event.name]


def test_label_usage_rejects_in_place_mutation_before_backup_link_and_restores_snapshot(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    previous = b"exact original event bytes\n"
    event.write_bytes(previous)
    original_link = llm_run.os.link
    mutated = False

    def mutate_then_link(*args, **kwargs):
        nonlocal mutated
        if not mutated:
            mutated = True
            with event.open("r+b") as source:
                source.truncate(0)
                source.write(b"in-place mutated event bytes\n")
                source.flush()
                llm_run.os.fsync(source.fileno())
        return original_link(*args, **kwargs)

    monkeypatch.setattr(llm_run.os, "link", mutate_then_link)

    with pytest.raises(llm_run.LabelUsageOutboxError, match="changed during publication"):
        llm_run._stage_usage_direct(usage, cache)

    assert event.read_bytes() == previous
    assert sorted(item.name for item in cache.iterdir()) == [event.name]


def test_label_usage_root_failure_restores_independent_snapshot_after_linked_inode_mutation(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    previous = b"exact original event bytes\n"
    event.write_bytes(previous)
    moved = tmp_path / "outside-moved-cache"
    original_replace = llm_run.os.replace
    swapped = False

    def mutate_move_then_publish(source, destination, *args, **kwargs):
        nonlocal swapped
        if not swapped and str(source).endswith(".tmp") and destination == event.name:
            swapped = True
            with event.open("r+b") as old_destination:
                old_destination.truncate(0)
                old_destination.write(b"in-place mutated event bytes\n")
                old_destination.flush()
                llm_run.os.fsync(old_destination.fileno())
            cache.rename(moved)
            cache.mkdir()
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(llm_run.os, "replace", mutate_move_then_publish)

    with pytest.raises(llm_run.LabelUsageOutboxError, match="cache root changed"):
        llm_run._stage_usage_direct(usage, cache)

    assert list(cache.iterdir()) == []
    assert (moved / event.name).read_bytes() == previous
    assert sorted(item.name for item in moved.iterdir()) == [event.name]


def test_label_usage_rejects_destination_appearance_after_absence_was_pinned(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    original_open = llm_run.os.open
    original_link = llm_run.os.link
    absence_pinned = False

    def observe_original_absence(name, flags, *args, **kwargs):
        nonlocal absence_pinned
        try:
            return original_open(name, flags, *args, **kwargs)
        except FileNotFoundError:
            if name == event.name:
                absence_pinned = True
            raise

    def create_then_link(*args, **kwargs):
        if absence_pinned:
            event.write_bytes(b"concurrently created event bytes\n")
        return original_link(*args, **kwargs)

    monkeypatch.setattr(llm_run.os, "open", observe_original_absence)
    monkeypatch.setattr(llm_run.os, "link", create_then_link)

    with pytest.raises(llm_run.LabelUsageOutboxError, match="changed during publication"):
        llm_run._stage_usage_direct(usage, cache)

    assert event.read_bytes() == b"concurrently created event bytes\n"
    assert sorted(item.name for item in cache.iterdir()) == [event.name]


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_label_usage_rejects_nonregular_old_destination(tmp_path, kind):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    if kind == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_bytes(b"outside event bytes\n")
        event.symlink_to(outside)
    else:
        event.mkdir()

    with pytest.raises(llm_run.LabelUsageOutboxError, match="regular file"):
        llm_run._stage_usage_direct(usage, cache)

    assert not list(cache.glob(f".{event.name}.*"))


def test_label_usage_snapshot_close_failure_does_not_mask_root_change(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    previous = b"exact original event bytes\n"
    event.write_bytes(previous)
    moved = tmp_path / "outside-moved-cache"
    original_open = llm_run.os.open
    original_close = llm_run.os.close
    original_unlink = llm_run.os.unlink
    descriptors: dict[str, int] = {}
    swapped = False
    close_failed = False

    def observe_open(name, flags, *args, **kwargs):
        descriptor = original_open(name, flags, *args, **kwargs)
        if name == cache.name:
            descriptors["root"] = descriptor
        elif name == event.name and flags & llm_run.os.O_ACCMODE == llm_run.os.O_RDONLY:
            descriptors["snapshot"] = descriptor
        return descriptor

    def unlink_then_swap_root(name, *args, **kwargs):
        nonlocal swapped
        result = original_unlink(name, *args, **kwargs)
        if str(name).endswith(".bak") and not swapped:
            swapped = True
            cache.rename(moved)
            cache.mkdir()
        return result

    def fail_snapshot_close(descriptor):
        nonlocal close_failed
        if descriptor == descriptors.get("snapshot") and not close_failed:
            close_failed = True
            raise OSError("secret prose and /private/path must not leak")
        return original_close(descriptor)

    monkeypatch.setattr(llm_run.os, "open", observe_open)
    monkeypatch.setattr(llm_run.os, "unlink", unlink_then_swap_root)
    monkeypatch.setattr(llm_run.os, "close", fail_snapshot_close)

    try:
        with pytest.raises(llm_run.LabelUsageOutboxError, match="cache root changed") as raised:
            llm_run._stage_usage_direct(usage, cache)

        notes = getattr(raised.value, "__notes__", [])
        assert notes == ["label usage descriptor cleanup also failed: OSError"]
        assert "secret prose" not in notes[0]
        assert "/private/path" not in notes[0]
        assert list(cache.iterdir()) == []
        assert (moved / event.name).read_bytes() == previous
        assert sorted(item.name for item in moved.iterdir()) == [event.name]
        with pytest.raises(OSError):
            llm_run.os.fstat(descriptors["root"])
    finally:
        with suppress(OSError):
            original_close(descriptors["snapshot"])


def test_label_usage_surfaces_snapshot_close_failure_after_verified_success(
    tmp_path,
    monkeypatch,
):
    usage = llm_run._usage_from_result(
        "label-run", "cluster-1", "input-hash", "miss", "ok", result(complete_label())
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    event = cache / f"label-usage-{usage.usage_id}.json"
    event.write_bytes(b"old event bytes\n")
    original_open = llm_run.os.open
    original_close = llm_run.os.close
    descriptors: dict[str, int] = {}
    close_failed = False

    def observe_open(name, flags, *args, **kwargs):
        descriptor = original_open(name, flags, *args, **kwargs)
        if name == cache.name:
            descriptors["root"] = descriptor
        elif name == event.name and flags & llm_run.os.O_ACCMODE == llm_run.os.O_RDONLY:
            descriptors["snapshot"] = descriptor
        return descriptor

    def fail_snapshot_close(descriptor):
        nonlocal close_failed
        if descriptor == descriptors.get("snapshot") and not close_failed:
            close_failed = True
            raise OSError("snapshot close failed")
        return original_close(descriptor)

    monkeypatch.setattr(llm_run.os, "open", observe_open)
    monkeypatch.setattr(llm_run.os, "close", fail_snapshot_close)

    try:
        with pytest.raises(OSError, match="snapshot close failed"):
            llm_run._stage_usage_direct(usage, cache)

        payload = json.loads(event.read_text())
        assert payload["usage_id"] == usage.usage_id
        assert sorted(item.name for item in cache.iterdir()) == [event.name]
        with pytest.raises(OSError):
            llm_run.os.fstat(descriptors["root"])
    finally:
        with suppress(OSError):
            original_close(descriptors["snapshot"])


def _swap_cache_root_inside_replace(monkeypatch, cache, moved):
    original_replace = llm_run.os.replace
    swapped = False

    def swap_then_replace(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            cache.rename(moved)
            cache.mkdir()
        return original_replace(*args, **kwargs)

    monkeypatch.setattr(llm_run.os, "replace", swap_then_replace)


def test_paid_result_root_swap_falls_back_to_database_before_cache_publication(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    con, vectors, cluster_run, signals_run = label_fixture
    cache = tmp_path / "cache"
    cache.mkdir()
    moved = tmp_path / "outside-moved-cache"
    _swap_cache_root_inside_replace(monkeypatch, cache, moved)
    monkeypatch.setattr(
        label_mod,
        "write_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache cannot publish before durable paid accounting")
        ),
    )

    with pytest.raises(llm_run.LabelUsageOutboxError, match="cache root changed"):
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=FakeModelClient([result(complete_label())]),
            vectors=vectors,
            cache_dir=cache,
        )

    assert con.execute(
        "SELECT outcome, error_category, input_tokens, output_tokens, estimated_cost_usd "
        "FROM llm_usage"
    ).fetchall() == [("ok", None, 12, 4, 0.00016)]
    assert con.execute("SELECT count(*) FROM cluster_labels").fetchone() == (1,)
    assert list(cache.iterdir()) == []
    assert list(moved.iterdir()) == []


def test_paid_provider_error_root_swap_falls_back_to_database_and_preserves_error(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    con, vectors, cluster_run, signals_run = label_fixture
    cache = tmp_path / "cache"
    cache.mkdir()
    moved = tmp_path / "outside-moved-cache"
    _swap_cache_root_inside_replace(monkeypatch, cache, moved)
    provider_error = ModelCallError(
        "malformed_response",
        1,
        False,
        usage=TokenUsage(input_tokens=21, output_tokens=8),
        latency_seconds=0.75,
        estimated_cost_usd=0.002,
        response_received=True,
    )

    with pytest.raises(ModelCallError, match="malformed_response") as raised:
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=FakeModelClient([provider_error]),
            vectors=vectors,
            cache_dir=cache,
        )

    assert isinstance(raised.value.__cause__, llm_run.LabelUsageOutboxError)
    assert any("after durable database insert" in note for note in raised.value.__notes__)
    assert con.execute(
        "SELECT outcome, error_category, input_tokens, output_tokens, estimated_cost_usd "
        "FROM llm_usage"
    ).fetchall() == [("failed", "malformed_response", 21, 8, 0.002)]
    assert con.execute("SELECT count(*) FROM cluster_labels").fetchone() == (0,)
    assert list(cache.iterdir()) == []
    assert list(moved.iterdir()) == []


def test_paid_result_root_swap_preserves_stage_error_when_database_fallback_fails(
    label_fixture,
    tmp_path,
    monkeypatch,
):
    con, vectors, cluster_run, signals_run = label_fixture
    cache = tmp_path / "cache"
    cache.mkdir()
    moved = tmp_path / "outside-moved-cache"
    _swap_cache_root_inside_replace(monkeypatch, cache, moved)
    monkeypatch.setattr(
        llm_run,
        "_persist_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("database unavailable")),
    )

    with pytest.raises(llm_run.LabelUsageOutboxError, match="cache root changed") as raised:
        llm_run.run(
            con,
            cluster_run,
            signals_run,
            control_n=0,
            limit=1,
            embed_model="m",
            client=FakeModelClient([result(complete_label())]),
            vectors=vectors,
            cache_dir=cache,
        )

    assert any("database persistence also failed" in note for note in raised.value.__notes__)
    assert con.execute("SELECT count(*) FROM llm_usage").fetchone() == (0,)
    assert con.execute("SELECT count(*) FROM cluster_labels").fetchone() == (0,)
    assert list(cache.iterdir()) == []
    assert list(moved.iterdir()) == []


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
    monkeypatch.setattr(llm_run, "embedding_model_for_cluster_run", lambda *_args: "m")
    args = SimpleNamespace(
        run_id="cluster-run",
        signals_run="signals-run",
        model=None,
        control_n=0,
        limit=1,
    )

    pipeline.phase_label(args)

    assert captured["run_id"] == "label-run"
