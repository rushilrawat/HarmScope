"""CLI coverage for Phase 8 labelling and human verification."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

import pytest

from src import pipeline
from src.llm import answer, verify
from src.llm import run as llm_run
from src.llm.client import TokenUsage
from src.llm.retrieve import RetrievedEvidence


def _ask_result(*, insufficient_evidence: bool = False) -> answer.AnswerResult:
    evidence = RetrievedEvidence(
        complaint_id=10,
        date_received=date(2020, 1, 2),
        company_id="company-1",
        company_name="Scope Company",
        product_family="mortgage",
        text_redacted="Consumers reported a delayed refund.",
        company_public_response="The company states that it resolved the complaint.",
        dense_rank=1,
        dense_score=0.9,
        sparse_rank=1,
        sparse_score=0.8,
        fused_score=0.1,
    )
    answer_value = answer.GroundedAnswer(
        answer="",
        claims=(),
        insufficient_evidence=True,
        limitations=("No relevant complaint evidence was retrieved.",),
    ) if insufficient_evidence else answer.GroundedAnswer(
        answer="Consumers allege delayed refunds.",
        claims=(answer.Claim("Consumers allege delayed refunds.", (10,)),),
        insufficient_evidence=False,
        limitations=("The evidence does not establish frequency.",),
    )
    return answer.AnswerResult(
        answer=answer_value,
        evidence=() if insufficient_evidence else (evidence,),
        enforcement_context=(),
        cached=False,
        usage=TokenUsage(input_tokens=12, output_tokens=4),
        latency_seconds=0.25,
        estimated_cost_usd=0.00016,
    )


def test_ask_command_requires_scope_question_and_parses_optional_flags():
    """Removing any analyst scope would allow an answer outside its evidence."""
    parser = pipeline.build_parser()

    args = parser.parse_args([
        "ask", "--cluster-id", "c1", "--company-id", "co1",
        "--question", "Why were funds unavailable?", "--model", "embed-m",
        "--include-enforcement-context",
    ])

    assert (args.cluster_id, args.company_id, args.question, args.model) == (
        "c1", "co1", "Why were funds unavailable?", "embed-m",
    )
    assert args.include_enforcement_context is True
    defaults = parser.parse_args([
        "ask", "--cluster-id", "c1", "--company-id", "co1", "--question", "Why?",
    ])
    assert defaults.model is None
    assert defaults.include_enforcement_context is False
    with pytest.raises(SystemExit):
        parser.parse_args(["ask", "--cluster-id", "c1", "--question", "Why?"])


def test_ask_wires_scope_to_answer_and_renders_distinct_sections(monkeypatch, capsys):
    """Collapsing provenance sections would blur allegation and company context."""
    captured = {}
    sentinel_connection = object()

    def fake_answer_question(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _ask_result()

    monkeypatch.setattr(pipeline.db, "bootstrap", lambda: sentinel_connection)
    monkeypatch.setattr(answer, "answer_question", fake_answer_question)

    status = pipeline.cmd_ask(SimpleNamespace(
        cluster_id="c1",
        company_id="co1",
        question="Why were funds unavailable?",
        model="embed-m",
        include_enforcement_context=True,
    ))

    assert status == 0
    assert captured == {
        "args": (sentinel_connection, "c1", "co1", "Why were funds unavailable?", "embed-m"),
        "kwargs": {"include_enforcement_context": True},
    }
    out = capsys.readouterr().out
    assert out.index("Claims") < out.index("Retrieved complaint evidence")
    assert out.index("Retrieved complaint evidence") < out.index("Company public responses")
    assert "Complaint 10" in out
    assert "Company public response" in out
    assert pipeline.DISCLAIMER in out
    assert out.count(pipeline.DISCLAIMER) == 1


def test_ask_renders_abstention_without_fabricating_complaint_evidence(monkeypatch, capsys):
    """An empty retrieval must stay visibly insufficient instead of looking answered."""
    captured = {}

    def fake_answer_question(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _ask_result(insufficient_evidence=True)

    monkeypatch.setattr(pipeline.db, "bootstrap", lambda: object())
    monkeypatch.setattr(answer, "answer_question", fake_answer_question)

    pipeline.cmd_ask(SimpleNamespace(
        cluster_id="c1",
        company_id="co1",
        question="Why were funds unavailable?",
        model=None,
        include_enforcement_context=False,
    ))

    out = capsys.readouterr().out
    assert "Insufficient complaint evidence" in out
    assert "Complaint 10" not in out
    assert out.count(pipeline.DISCLAIMER) == 1
    assert captured["args"][-1] == pipeline.CONFIG.embed.dev_model
    assert captured["kwargs"] == {"include_enforcement_context": False}


def test_label_verify_subcommands_parse():
    """Removing the verification CLI would make human review inaccessible."""
    parser = pipeline.build_parser()

    args = parser.parse_args([
        "label-verify", "export", "--n", "50", "--output", "review.csv",
    ])

    assert args.verify_action == "export"
    assert args.n == 50
    assert args.output == "review.csv"


def test_label_summary_reports_all_operational_totals(monkeypatch, capsys):
    """An incomplete summary would conceal failed calls or paid-run usage."""
    @contextmanager
    def fake_run(*_args, **_kwargs):
        yield SimpleNamespace(run_id="label-run", finish=lambda **_kwargs: None)

    monkeypatch.setattr(pipeline.db, "bootstrap", lambda: object())
    monkeypatch.setattr(pipeline.db, "run", fake_run)
    monkeypatch.setattr(
        llm_run,
        "run",
        lambda *_args, **_kwargs: llm_run.LabelRunStats(
            labelled=17,
            cached=2,
            refused=1,
            failed=1,
            skipped=1,
            input_tokens=1_000,
            output_tokens=100,
            latency_seconds=4.2,
            estimated_cost_usd=0.01,
        ),
    )

    pipeline.phase_label(SimpleNamespace(
        run_id="cluster-run",
        signals_run="signals-run",
        model=None,
        control_n=0,
        limit=20,
    ))

    out = capsys.readouterr().out.lower()
    for term in (
        "labelled", "cache", "refused", "failed", "skipped", "input tokens",
        "output tokens", "latency", "estimated cost", "estimated, not invoice",
    ):
        assert term in out


def test_label_verify_rejects_duplicate_cluster_ids(tmp_path):
    """One cluster cannot count twice toward the human-review denominator."""
    row = dict.fromkeys(verify.HEADER, "")
    row.update({
        "cluster_id": "cluster-1",
        "mechanism_accuracy": "agree",
        "taxonomy_distinctness_accuracy": "agree",
        "template_accuracy": "agree",
        "should_have_abstained": "false",
        "failure_category": "none",
    })
    path = tmp_path / "duplicate.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=verify.HEADER)
        writer.writeheader()
        writer.writerows([row, row])

    with pytest.raises(ValueError, match="duplicate cluster_id"):
        verify.parse_worklist(path, "reviewer-1")
