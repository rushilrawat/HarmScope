"""CLI coverage for Phase 8 labelling and human verification."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from src import pipeline
from src.llm import run as llm_run
from src.llm import verify


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


def test_default_worklist_version_is_stable_across_row_order():
    """Sorting a completed sheet must not orphan the exported worklist version."""
    original = pipeline._worklist_version(["cluster-b", "cluster-a"])

    assert original == pipeline._worklist_version(["cluster-a", "cluster-b"])
    assert original == pipeline._worklist_version([
        "cluster-a", "cluster-b", "cluster-a",
    ])


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
