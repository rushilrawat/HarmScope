"""Integration tests for resumable, observable LLM cluster labelling."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from src import pipeline
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
        con.execute(
            "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
            "started_at, status) VALUES (?, ?, 'test', 'test', '{}', now(), 'ok')",
            [run_id, phase],
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
