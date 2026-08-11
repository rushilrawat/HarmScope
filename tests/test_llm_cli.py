"""CLI coverage for Phase 8 labelling and human verification."""

from __future__ import annotations

import csv
import hashlib
import json
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

import pytest

from src import pipeline
from src.llm import answer, verify
from src.llm import eval as llm_eval
from src.llm import run as llm_run
from src.llm.client import TokenUsage
from src.llm.retrieve import RetrievedEvidence


def _ask_result(*, insufficient_evidence: bool = False) -> answer.AnswerResult:
    evidence = RetrievedEvidence(
        complaint_id=10,
        cluster_id="c1",
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
    answer_value = (
        answer.GroundedAnswer(
            answer="",
            claims=(),
            insufficient_evidence=True,
            limitation_reasons=("no_relevant_complaint_evidence",),
        )
        if insufficient_evidence
        else answer.GroundedAnswer(
            answer="Consumers allege delayed refunds.",
            claims=(answer.Claim("Consumers allege delayed refunds.", (10,)),),
            insufficient_evidence=False,
            limitation_reasons=("retrieved_complaints_do_not_establish_frequency",),
        )
    )
    return answer.AnswerResult(
        answer=answer_value,
        evidence=() if insufficient_evidence else (evidence,),
        enforcement_context=(),
        cache_status="miss",
        usage=TokenUsage(input_tokens=12, output_tokens=4),
        latency_seconds=0.25,
        estimated_cost_usd=0.00016,
    )


def test_ask_command_requires_scope_question_and_parses_optional_flags():
    """Removing any analyst scope would allow an answer outside its evidence."""
    parser = pipeline.build_parser()

    args = parser.parse_args(
        [
            "ask",
            "--cluster-id",
            "c1",
            "--company-id",
            "co1",
            "--question",
            "Why were funds unavailable?",
            "--model",
            "embed-m",
            "--include-enforcement-context",
        ]
    )

    assert (args.cluster_id, args.company_id, args.question, args.model) == (
        "c1",
        "co1",
        "Why were funds unavailable?",
        "embed-m",
    )
    assert args.include_enforcement_context is True
    defaults = parser.parse_args(
        [
            "ask",
            "--cluster-id",
            "c1",
            "--company-id",
            "co1",
            "--question",
            "Why?",
        ]
    )
    assert defaults.model is None
    assert defaults.include_enforcement_context is False
    with pytest.raises(SystemExit):
        parser.parse_args(["ask", "--cluster-id", "c1", "--question", "Why?"])


def test_ask_wires_scope_to_answer_and_renders_distinct_sections(monkeypatch, capsys):
    """Collapsing provenance sections would blur allegation and company context."""
    captured = {}
    closed = []
    sentinel_connection = SimpleNamespace(close=lambda: closed.append(True))

    def fake_answer_question(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _ask_result()

    monkeypatch.setattr(pipeline.db, "bootstrap", lambda: sentinel_connection)
    monkeypatch.setattr(answer, "answer_question", fake_answer_question)

    status = pipeline.cmd_ask(
        SimpleNamespace(
            cluster_id="c1",
            company_id="co1",
            question="Why were funds unavailable?",
            model="embed-m",
            include_enforcement_context=True,
        )
    )

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
    assert closed == [True]


def test_ask_renders_abstention_without_fabricating_complaint_evidence(monkeypatch, capsys):
    """An empty retrieval must stay visibly insufficient instead of looking answered."""
    captured = {}

    def fake_answer_question(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _ask_result(insufficient_evidence=True)

    closed = []
    monkeypatch.setattr(
        pipeline.db,
        "bootstrap",
        lambda: SimpleNamespace(close=lambda: closed.append(True)),
    )
    monkeypatch.setattr(answer, "answer_question", fake_answer_question)

    pipeline.cmd_ask(
        SimpleNamespace(
            cluster_id="c1",
            company_id="co1",
            question="Why were funds unavailable?",
            model=None,
            include_enforcement_context=False,
        )
    )

    out = capsys.readouterr().out
    assert "Insufficient complaint evidence" in out
    assert "Complaint 10" not in out
    assert out.count(pipeline.DISCLAIMER) == 1
    assert captured["args"][-1] is None
    assert captured["kwargs"] == {"include_enforcement_context": False}
    assert closed == [True]


def test_ask_closes_database_when_answer_generation_fails(monkeypatch):
    closed = []
    connection = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(pipeline.db, "bootstrap", lambda: connection)

    def fail(*_args, **_kwargs):
        raise RuntimeError("answer failed")

    monkeypatch.setattr(answer, "answer_question", fail)

    with pytest.raises(RuntimeError, match="answer failed"):
        pipeline.cmd_ask(
            SimpleNamespace(
                cluster_id="c1",
                company_id="co1",
                question="Why?",
                model=None,
                include_enforcement_context=False,
            )
        )

    assert closed == [True]


def test_label_verify_subcommands_parse():
    """Removing the verification CLI would make human review inaccessible."""
    parser = pipeline.build_parser()

    args = parser.parse_args(
        [
            "label-verify",
            "export",
            "--n",
            "50",
            "--output",
            "review.csv",
        ]
    )

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

    pipeline.phase_label(
        SimpleNamespace(
            run_id="cluster-run",
            signals_run="signals-run",
            model=None,
            control_n=0,
            limit=20,
        )
    )

    out = capsys.readouterr().out.lower()
    for term in (
        "labelled",
        "cache",
        "refused",
        "failed",
        "skipped",
        "input tokens",
        "output tokens",
        "latency",
        "estimated cost",
        "estimated, not invoice",
    ):
        assert term in out


def test_label_verify_rejects_duplicate_cluster_ids(tmp_path):
    """One cluster cannot count twice toward the human-review denominator."""
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
    path = tmp_path / "duplicate.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=verify.HEADER)
        writer.writeheader()
        writer.writerows([row, row])

    with pytest.raises(ValueError, match="duplicate cluster_id"):
        verify.parse_worklist(path, "reviewer-1")


class _ConnectionSpy:
    """Keep the real test DB inspectable while proving the CLI closes its handle."""

    def __init__(self, connection):
        self.connection = connection
        self.closed = False

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def close(self):
        self.closed = True


def _rag_args(**overrides):
    values = {
        "eval_action": "run",
        "retrieval_only": False,
        "output": None,
        "input": None,
        "run_id": None,
        "reviewer": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_rag_eval_parser_defaults_to_full_run_and_exposes_every_action():
    """Removing a subcommand would make a reviewed workflow unreachable."""
    parser = pipeline.build_parser()

    default = parser.parse_args(["rag-eval"])
    assert default.eval_action == "run"
    assert default.retrieval_only is False
    assert default.func is pipeline.cmd_rag_eval

    retrieval = parser.parse_args(["rag-eval", "--retrieval-only"])
    assert retrieval.eval_action == "run"
    assert retrieval.retrieval_only is True

    author = parser.parse_args(["rag-eval", "author", "--output", "draft.csv"])
    assert (author.eval_action, author.output) == ("author", "draft.csv")
    imported = parser.parse_args(["rag-eval", "import", "--input", "draft.csv"])
    assert (imported.eval_action, imported.input) == ("import", "draft.csv")
    exported = parser.parse_args(
        ["rag-eval", "claims-export", "--run-id", "eval-1", "--output", "claims.csv"]
    )
    assert (exported.eval_action, exported.run_id, exported.output) == (
        "claims-export",
        "eval-1",
        "claims.csv",
    )
    recorded = parser.parse_args(
        [
            "rag-eval",
            "claims-record",
            "--run-id",
            "eval-1",
            "--input",
            "claims.reviewer-1.csv",
            "--reviewer",
            "reviewer-1",
        ]
    )
    assert (recorded.eval_action, recorded.run_id, recorded.reviewer) == (
        "claims-record",
        "eval-1",
        "reviewer-1",
    )


def test_rag_eval_loads_manifest_before_opening_writable_database(monkeypatch, tmp_path):
    """A missing human freeze must not create a run or mutate a database."""
    paths = SimpleNamespace(
        ground_truth=tmp_path / "ground_truth",
        interim=tmp_path / "interim",
    )
    monkeypatch.setattr(pipeline, "PATHS", paths)
    opened = []
    monkeypatch.setattr(pipeline.db, "bootstrap", lambda: opened.append(True))

    with pytest.raises(FileNotFoundError):
        pipeline.cmd_rag_eval(_rag_args())

    assert opened == []


def test_rag_eval_registers_exact_manifest_identity_and_finishes_ninety_rows(
    monkeypatch, con, tmp_path, capsys
):
    """Changing the manifest/model/mode provenance must change the registered run."""
    manifest = tmp_path / "ground_truth" / "rag_eval_questions.csv"
    manifest.parent.mkdir()
    manifest.write_bytes(b"frozen benchmark bytes\n")
    interim = tmp_path / "interim"
    interim.mkdir()
    monkeypatch.setattr(
        pipeline,
        "PATHS",
        SimpleNamespace(ground_truth=manifest.parent, interim=interim),
    )
    questions = [object()] * 30
    events = []
    spy = _ConnectionSpy(con)
    monkeypatch.setattr(pipeline.db, "bootstrap", lambda: events.append("bootstrap") or spy)
    monkeypatch.setattr(
        llm_eval,
        "load_manifest",
        lambda path: events.append(("load", path)) or questions,
    )
    monkeypatch.setattr(
        llm_eval,
        "validate_manifest",
        lambda connection, rows: events.append(("validate", connection, rows)),
    )
    monkeypatch.setattr(
        llm_eval,
        "manifest_embed_model",
        lambda connection, rows: events.append(("model", connection, rows)) or "embed-m",
    )
    retrieval = SimpleNamespace(render=lambda: "retrieval metrics")
    answers = SimpleNamespace(render=lambda: "answer metrics")
    monkeypatch.setattr(
        llm_eval,
        "run_retrieval_eval",
        lambda connection, rows, model, run_id: events.append(
            ("retrieval", connection, rows, model, run_id)
        )
        or retrieval,
    )
    monkeypatch.setattr(
        llm_eval,
        "run_answer_eval",
        lambda connection, rows, model, run_id: events.append(
            ("answers", connection, rows, model, run_id)
        )
        or answers,
    )
    monkeypatch.setattr(
        llm_eval,
        "render_evaluation_run",
        lambda run_id, got_retrieval, got_answers: (
            f"eval run ID: {run_id}\nretrieval metrics\nanswer metrics\n"
            "human groundedness gate: PENDING"
        ),
    )

    assert pipeline.cmd_rag_eval(_rag_args()) == 0

    row = con.execute(
        "SELECT run_id, status, output_rows, params_json FROM runs WHERE phase = 'rag-eval'"
    ).fetchone()
    run_id, status, output_rows, raw_params = row
    params = json.loads(raw_params)["params"]
    assert (status, output_rows) == ("ok", 90)
    assert params == {
        "embed_model": "embed-m",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "retrieval_only": False,
    }
    assert events[:4] == [
        ("load", manifest),
        "bootstrap",
        ("validate", spy, questions),
        ("model", spy, questions),
    ]
    assert events[4][0:4] == ("retrieval", spy, questions, "embed-m")
    assert events[4][4] == run_id
    assert events[5][0:4] == ("answers", spy, questions, "embed-m")
    assert events[5][4] == run_id
    output = capsys.readouterr().out
    assert f"eval run ID: {run_id}" in output
    assert "human groundedness gate: PENDING" in output
    assert spy.closed is True


def test_rag_eval_retrieval_only_never_reaches_answer_provider(monkeypatch, con, tmp_path):
    """Retrieval-only must remain runnable without constructing a paid client."""
    manifest = tmp_path / "ground_truth" / "rag_eval_questions.csv"
    manifest.parent.mkdir()
    manifest.write_text("manifest")
    interim = tmp_path / "interim"
    interim.mkdir()
    monkeypatch.setattr(
        pipeline,
        "PATHS",
        SimpleNamespace(ground_truth=manifest.parent, interim=interim),
    )
    questions = [object()] * 30
    spy = _ConnectionSpy(con)
    monkeypatch.setattr(pipeline.db, "bootstrap", lambda: spy)
    monkeypatch.setattr(llm_eval, "load_manifest", lambda _path: questions)
    monkeypatch.setattr(llm_eval, "validate_manifest", lambda *_args: None)
    monkeypatch.setattr(llm_eval, "manifest_embed_model", lambda *_args: "embed-m")
    retrieval = SimpleNamespace(render=lambda: "retrieval metrics")
    monkeypatch.setattr(llm_eval, "run_retrieval_eval", lambda *_args: retrieval)
    monkeypatch.setattr(
        llm_eval,
        "run_answer_eval",
        lambda *_args: pytest.fail("retrieval-only reached answer generation"),
    )
    rendered = []
    monkeypatch.setattr(
        llm_eval,
        "render_evaluation_run",
        lambda run_id, got_retrieval, got_answers: rendered.append(
            (run_id, got_retrieval, got_answers)
        )
        or "answer metrics: n/a\nhuman groundedness: PENDING",
    )

    pipeline.cmd_rag_eval(_rag_args(retrieval_only=True))

    status, output_rows, params = con.execute(
        "SELECT status, output_rows, params_json FROM runs WHERE phase = 'rag-eval'"
    ).fetchone()
    assert (status, output_rows) == ("ok", 90)
    assert json.loads(params)["params"]["retrieval_only"] is True
    assert rendered[0][1:] == (retrieval, None)
    assert spy.closed is True


def test_rag_eval_failure_stays_failed_without_output_declaration(monkeypatch, con, tmp_path):
    """A partial retrieval cannot masquerade as a completed 90-row evaluation."""
    manifest = tmp_path / "ground_truth" / "rag_eval_questions.csv"
    manifest.parent.mkdir()
    manifest.write_text("manifest")
    interim = tmp_path / "interim"
    interim.mkdir()
    monkeypatch.setattr(
        pipeline,
        "PATHS",
        SimpleNamespace(ground_truth=manifest.parent, interim=interim),
    )
    questions = [object()] * 30
    spy = _ConnectionSpy(con)
    monkeypatch.setattr(pipeline.db, "bootstrap", lambda: spy)
    monkeypatch.setattr(llm_eval, "load_manifest", lambda _path: questions)
    monkeypatch.setattr(llm_eval, "validate_manifest", lambda *_args: None)
    monkeypatch.setattr(llm_eval, "manifest_embed_model", lambda *_args: "embed-m")

    def fail(*_args):
        raise RuntimeError("retrieval interrupted")

    monkeypatch.setattr(llm_eval, "run_retrieval_eval", fail)

    with pytest.raises(RuntimeError, match="retrieval interrupted"):
        pipeline.cmd_rag_eval(_rag_args(retrieval_only=True))

    status, output_rows, error = con.execute(
        "SELECT status, output_rows, error FROM runs WHERE phase = 'rag-eval'"
    ).fetchone()
    assert status == "failed"
    assert output_rows is None
    assert error == "RuntimeError: retrieval interrupted"
    assert spy.closed is True


def test_rag_eval_render_failure_cannot_leave_a_successful_run(monkeypatch, con, tmp_path):
    """A run whose required stable report cannot be produced is not complete."""
    manifest = tmp_path / "ground_truth" / "rag_eval_questions.csv"
    manifest.parent.mkdir()
    manifest.write_text("manifest")
    interim = tmp_path / "interim"
    interim.mkdir()
    monkeypatch.setattr(
        pipeline,
        "PATHS",
        SimpleNamespace(ground_truth=manifest.parent, interim=interim),
    )
    questions = [object()] * 30
    spy = _ConnectionSpy(con)
    monkeypatch.setattr(pipeline.db, "bootstrap", lambda: spy)
    monkeypatch.setattr(llm_eval, "load_manifest", lambda _path: questions)
    monkeypatch.setattr(llm_eval, "validate_manifest", lambda *_args: None)
    monkeypatch.setattr(llm_eval, "manifest_embed_model", lambda *_args: "embed-m")
    monkeypatch.setattr(llm_eval, "run_retrieval_eval", lambda *_args: object())

    def fail_render(*_args):
        raise ValueError("summary corrupt")

    monkeypatch.setattr(llm_eval, "render_evaluation_run", fail_render)

    with pytest.raises(ValueError, match="summary corrupt"):
        pipeline.cmd_rag_eval(_rag_args(retrieval_only=True))

    status, output_rows, error = con.execute(
        "SELECT status, output_rows, error FROM runs WHERE phase = 'rag-eval'"
    ).fetchone()
    assert status == "failed"
    assert output_rows is None
    assert error == "ValueError: summary corrupt"
    assert spy.closed is True


def test_rag_eval_author_import_and_claim_actions_wire_exact_contracts(
    monkeypatch, tmp_path, capsys
):
    """Each non-run action must preserve its reviewed path/count/provenance boundary."""
    ground_truth = tmp_path / "ground_truth"
    interim = tmp_path / "interim"
    ground_truth.mkdir()
    interim.mkdir()
    manifest = ground_truth / "rag_eval_questions.csv"
    source = interim / "rag_eval_authoring.csv"
    source.write_text("private")
    claims = interim / "claims.csv"
    completed = interim / "claims.reviewer-1.csv"
    completed.write_text("human decisions")
    monkeypatch.setattr(
        pipeline,
        "PATHS",
        SimpleNamespace(ground_truth=ground_truth, interim=interim),
    )
    connections = []

    def open_connection():
        connection = SimpleNamespace(close=lambda: setattr(connection, "closed", True))
        connection.closed = False
        connections.append(connection)
        return connection

    monkeypatch.setattr(pipeline.db, "bootstrap", open_connection)
    calls = []
    monkeypatch.setattr(
        llm_eval,
        "export_authoring_worklist",
        lambda connection, seed, path: calls.append(("author", connection, seed, path)) or path,
    )
    monkeypatch.setattr(
        llm_eval,
        "import_authoring_worklist",
        lambda connection, got_source, destination: calls.append(
            ("import", connection, got_source, destination)
        )
        or destination,
    )
    monkeypatch.setattr(llm_eval, "load_manifest", lambda path: [object()] * 30)
    monkeypatch.setattr(
        llm_eval,
        "export_claim_review",
        lambda connection, run_id, n, seed, path: calls.append(
            ("claims-export", connection, run_id, n, seed, path)
        )
        or path,
    )
    reviews = [object()] * 50
    monkeypatch.setattr(
        llm_eval,
        "parse_claim_review",
        lambda path, reviewer: calls.append(("parse", path, reviewer)) or reviews,
    )
    report = SimpleNamespace(reviewed=50, render=lambda: "grounded claims: 44/50")
    monkeypatch.setattr(
        llm_eval,
        "record_claim_review",
        lambda connection, run_id, got_reviews: calls.append(
            ("claims-record", connection, run_id, got_reviews)
        )
        or report,
    )

    pipeline.cmd_rag_eval(_rag_args(eval_action="author", output=str(source)))
    author_output = capsys.readouterr().out.splitlines()
    pipeline.cmd_rag_eval(_rag_args(eval_action="import", input=str(source)))
    import_output = capsys.readouterr().out.splitlines()
    manifest.write_text("frozen")
    pipeline.cmd_rag_eval(
        _rag_args(eval_action="claims-export", run_id="eval-1", output=str(claims))
    )
    export_output = capsys.readouterr().out.splitlines()
    pipeline.cmd_rag_eval(
        _rag_args(
            eval_action="claims-record",
            run_id="eval-1",
            input=str(completed),
            reviewer="reviewer-1",
        )
    )
    record_output = capsys.readouterr().out

    assert calls[0][0:4] == (
        "author",
        connections[0],
        pipeline.CONFIG.llm.verification_seed,
        source,
    )
    assert calls[1] == ("import", connections[1], source, manifest)
    assert calls[2] == ("claims-export", connections[2], "eval-1", 50, 20260809, claims)
    assert calls[3] == ("parse", completed, "reviewer-1")
    assert calls[4] == ("claims-record", connections[3], "eval-1", reviews)
    assert all(connection.closed for connection in connections)
    assert len(author_output) == 2
    assert "worklist" in author_output[0].lower()
    assert "rows" in author_output[1].lower()
    assert len(import_output) == 2
    assert "manifest" in import_output[0].lower()
    assert "questions" in import_output[1].lower()
    assert len(export_output) == 2
    assert "claims" in export_output[0].lower()
    assert "rows" in export_output[1].lower()
    assert "recorded" in record_output.lower()
    assert "44/50" in record_output


def test_rag_eval_claim_record_parses_before_opening_database(monkeypatch, tmp_path):
    """A bad private filename/content must fail before a writable connection exists."""
    interim = tmp_path / "interim"
    interim.mkdir()
    bad = interim / "claims.csv"
    bad.write_text("bad")
    monkeypatch.setattr(
        pipeline,
        "PATHS",
        SimpleNamespace(ground_truth=tmp_path / "ground_truth", interim=interim),
    )
    monkeypatch.setattr(llm_eval, "PATHS", SimpleNamespace(interim=interim))
    opened = []
    monkeypatch.setattr(pipeline.db, "bootstrap", lambda: opened.append(True))

    with pytest.raises(ValueError, match="filename.*reviewer_id"):
        pipeline.cmd_rag_eval(
            _rag_args(
                eval_action="claims-record",
                run_id="eval-1",
                input=str(bad),
                reviewer="reviewer-1",
            )
        )

    assert opened == []


def test_rag_eval_rejects_private_artifacts_outside_interim_before_database(monkeypatch, tmp_path):
    """Generated or consumer prose must not escape the ignored interim directory."""
    interim = tmp_path / "interim"
    interim.mkdir()
    ground_truth = tmp_path / "ground_truth"
    ground_truth.mkdir()
    monkeypatch.setattr(
        pipeline,
        "PATHS",
        SimpleNamespace(ground_truth=ground_truth, interim=interim),
    )
    opened = []
    monkeypatch.setattr(pipeline.db, "bootstrap", lambda: opened.append(True))

    with pytest.raises(ValueError, match="data/interim"):
        pipeline.cmd_rag_eval(_rag_args(eval_action="author", output=str(tmp_path / "tracked.csv")))

    assert opened == []
