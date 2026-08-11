"""Phase 8D benchmark manifest and private authoring-workflow contracts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from src.llm import eval as llm_eval


def _token_text(prefix: str, count: int = 12) -> str:
    """Create non-prose synthetic tokens without committing narrative fixtures."""
    return " ".join(f"{prefix}{index}" for index in range(count))


def _manifest_row(
    index: int,
    category: str,
    *,
    cluster_id: str = "cluster-1",
    company_id: str = "company-1",
    answerable: bool | None = None,
    relevant_ids: str | None = None,
) -> dict[str, str]:
    is_answerable = category != "unanswerable" if answerable is None else answerable
    if relevant_ids is None:
        relevant_ids = str(index) if is_answerable else ""
    return {
        "question_id": f"rag-{index:03d}",
        "question": f"Synthetic analyst question {index}?",
        "cluster_id": cluster_id,
        "company_id": company_id,
        "category": category,
        "answerable": str(is_answerable).lower(),
        "relevant_complaint_ids": relevant_ids,
    }


def _write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _small_manifest_rows() -> list[dict[str, str]]:
    return [
        _manifest_row(index, category)
        for index, category in enumerate(llm_eval.CATEGORY_ORDER, start=1)
    ]


def test_manifest_parser_accepts_only_the_strict_balanced_contract(tmp_path):
    path = _write_csv(
        tmp_path / "manifest.csv",
        list(llm_eval.MANIFEST_HEADER),
        _small_manifest_rows(),
    )

    rows = llm_eval.load_manifest(path, expected_n=6)

    assert len(rows) == 6
    assert Counter(row.category for row in rows) == dict.fromkeys(llm_eval.CATEGORIES, 1)
    assert rows[0].relevant_complaint_ids == frozenset({1})
    assert rows[-1].answerable is False
    assert rows[-1].relevant_complaint_ids == frozenset()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda rows: rows[0].update(answerable="TRUE"), "true or false"),
        (lambda rows: rows[0].update(question=""), "question"),
        (lambda rows: rows[0].update(cluster_id=""), "cluster_id"),
        (lambda rows: rows[0].update(category="other"), "category"),
        (lambda rows: rows[1].update(question_id=rows[0]["question_id"]), "duplicate"),
        (
            lambda rows: rows[0].update(relevant_complaint_ids="2;1"),
            "ascending",
        ),
        (
            lambda rows: rows[0].update(relevant_complaint_ids="1;1"),
            "duplicate",
        ),
        (lambda rows: rows[0].update(relevant_complaint_ids=""), "answerable"),
        (
            lambda rows: rows[-1].update(relevant_complaint_ids="6"),
            "unanswerable",
        ),
    ],
)
def test_manifest_parser_rejects_malformed_rows(tmp_path, mutate, match):
    rows = _small_manifest_rows()
    mutate(rows)
    path = _write_csv(tmp_path / "manifest.csv", list(llm_eval.MANIFEST_HEADER), rows)

    with pytest.raises(llm_eval.ManifestError, match=match):
        llm_eval.load_manifest(path, expected_n=6)


def test_manifest_parser_rejects_wrong_header_and_distribution(tmp_path):
    rows = _small_manifest_rows()
    wrong_header = list(llm_eval.MANIFEST_HEADER) + ["helper"]
    with pytest.raises(llm_eval.ManifestError, match="columns"):
        llm_eval.load_manifest(
            _write_csv(tmp_path / "header.csv", wrong_header, rows),
            expected_n=6,
        )

    rows[1]["category"] = "mechanism"
    with pytest.raises(llm_eval.ManifestError, match="balanced"):
        llm_eval.load_manifest(
            _write_csv(
                tmp_path / "distribution.csv",
                list(llm_eval.MANIFEST_HEADER),
                rows,
            ),
            expected_n=6,
        )


@pytest.fixture
def validation_fixture(con):
    run_id = "0000000000001-cluster1"
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'cluster', 'test', 'test', '{}', now(), 'ok')",
        [run_id],
    )
    for cluster_id in ("cluster-1", "cluster-2"):
        con.execute(
            "INSERT INTO clusters "
            "(cluster_id, run_id, product_family, n_members, as_of) "
            "VALUES (?, ?, 'family-1', 3, ?)",
            [cluster_id, run_id, date(2020, 1, 1)],
        )
    for company_id in ("company-1", "company-2", "company-3"):
        con.execute(
            "INSERT INTO company_canonical "
            "(company_id, canonical_name, verified_by) VALUES (?, ?, 'manual')",
            [company_id, company_id],
        )
    records = [
        (10, "cluster-1", "company-1", _token_text("alpha")),
        (20, "cluster-1", "company-2", _token_text("beta")),
        (30, "cluster-2", "company-1", _token_text("gamma")),
    ]
    for complaint_id, cluster_id, company_id, text in records:
        con.execute(
            "INSERT INTO complaints "
            "(complaint_id, date_received, period_month, company_id, "
            "product_family, has_narrative) VALUES (?, ?, ?, ?, 'family-1', true)",
            [complaint_id, date(2020, 1, 1), date(2020, 1, 1), company_id],
        )
        con.execute(
            "INSERT INTO narratives "
            "(complaint_id, text_redacted, text_hash, redaction_count) "
            "VALUES (?, ?, ?, 0)",
            [complaint_id, text, f"hash-{complaint_id}"],
        )
        con.execute(
            "INSERT INTO cluster_members (cluster_id, complaint_id) VALUES (?, ?)",
            [cluster_id, complaint_id],
        )
    con.execute(
        "INSERT INTO complaints "
        "(complaint_id, date_received, period_month, company_id, "
        "product_family, has_narrative) VALUES (40, ?, ?, 'company-3', 'family-1', false)",
        [date(2020, 1, 1), date(2020, 1, 1)],
    )
    con.execute("INSERT INTO cluster_members (cluster_id, complaint_id) VALUES ('cluster-1', 40)")
    return con


def _question(
    *,
    question: str = "Short synthetic question?",
    cluster_id: str = "cluster-1",
    company_id: str | None = "company-1",
    relevant_ids: frozenset[int] = frozenset({10}),
) -> llm_eval.EvalQuestion:
    return llm_eval.EvalQuestion(
        question_id="rag-001",
        question=question,
        cluster_id=cluster_id,
        company_id=company_id,
        category="mechanism",
        relevant_complaint_ids=relevant_ids,
        answerable=True,
    )


@pytest.mark.parametrize(
    ("question", "match"),
    [
        (_question(cluster_id="missing"), "cluster"),
        (_question(company_id="missing"), "company"),
        (_question(relevant_ids=frozenset({20})), "outside"),
        (_question(relevant_ids=frozenset({30})), "outside"),
    ],
)
def test_database_validation_enforces_exact_cluster_company_scope(
    validation_fixture, question, match
):
    with pytest.raises(llm_eval.ManifestError, match=match):
        llm_eval.validate_manifest(validation_fixture, [question])


def test_privacy_guard_rejects_eight_token_overlap_without_echoing_text(
    validation_fixture,
):
    private_sequence = " ".join(f"alpha{index}" for index in range(8))
    bad = _question(question=private_sequence)

    with pytest.raises(llm_eval.ManifestError, match="eight-token") as caught:
        llm_eval.validate_manifest(validation_fixture, [bad])

    message = str(caught.value)
    assert "rag-001" in message
    assert "10" in message
    assert private_sequence not in message


def test_privacy_guard_accepts_shorter_overlap(validation_fixture):
    seven_tokens = " ".join(f"alpha{index}" for index in range(7))
    llm_eval.validate_manifest(validation_fixture, [_question(question=seven_tokens)])


def test_company_membership_does_not_require_a_narrative(validation_fixture):
    question = llm_eval.EvalQuestion(
        question_id="rag-006",
        question="Synthetic unavailable fact?",
        cluster_id="cluster-1",
        company_id="company-3",
        category="unanswerable",
        relevant_complaint_ids=frozenset(),
        answerable=False,
    )

    llm_eval.validate_manifest(validation_fixture, [question])


@pytest.fixture
def authoring_fixture(con):
    cluster_run = "0000000000001-cluster1"
    signals_run = "0000000000002-signal01"
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'cluster', 'test', 'test', '{}', now(), 'ok')",
        [cluster_run],
    )
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'signals', 'test', 'test', ?, now(), 'ok')",
        [signals_run, json.dumps({"params": {"cluster_run": cluster_run}})],
    )
    for company_number in range(36):
        company_id = f"company-{company_number:02d}"
        con.execute(
            "INSERT INTO company_canonical "
            "(company_id, canonical_name, verified_by) VALUES (?, ?, 'manual')",
            [company_id, company_id],
        )
    complaint_id = 1_000
    for cluster_number in range(36):
        cluster_id = f"{cluster_run}:family-{cluster_number % 6}:{cluster_number:02d}"
        family = f"family-{cluster_number % 6}"
        company_id = f"company-{cluster_number:02d}"
        con.execute(
            "INSERT INTO clusters "
            "(cluster_id, run_id, product_family, n_members, as_of) "
            "VALUES (?, ?, ?, 10, ?)",
            [cluster_id, cluster_run, family, date(2020, 1, 1)],
        )
        if cluster_number < 18:
            con.execute(
                "INSERT INTO signals "
                "(signal_id, run_id, cluster_id, company_id, period_month, method, "
                "statistic, q_value, n_supporting, n_supporting_groups, as_of) "
                "VALUES (?, ?, ?, ?, ?, 'ewma', 1.0, 0.01, 10, 10, ?)",
                [
                    f"signal-{cluster_number}",
                    signals_run,
                    cluster_id,
                    company_id,
                    date(2020, 1, 1),
                    date(2020, 1, 1),
                ],
            )
        for excerpt_number in range(10):
            text = _token_text(f"z{cluster_number}x{excerpt_number}y")
            if cluster_number == 0 and excerpt_number == 0:
                text = "=FORMULA " + text
            con.execute(
                "INSERT INTO complaints "
                "(complaint_id, date_received, period_month, company_id, "
                "product_family, has_narrative) VALUES (?, ?, ?, ?, ?, true)",
                [
                    complaint_id,
                    date(2020, 1, 1),
                    date(2020, 1, 1),
                    company_id,
                    family,
                ],
            )
            con.execute(
                "INSERT INTO narratives "
                "(complaint_id, text_redacted, text_hash, redaction_count) "
                "VALUES (?, ?, ?, 0)",
                [complaint_id, text, f"hash-{complaint_id}"],
            )
            con.execute(
                "INSERT INTO cluster_members (cluster_id, complaint_id) VALUES (?, ?)",
                [cluster_id, complaint_id],
            )
            complaint_id += 1
    return con


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def test_authoring_export_is_deterministic_balanced_and_spreadsheet_safe(
    authoring_fixture, tmp_path
):
    first = llm_eval.export_authoring_worklist(authoring_fixture, 7, tmp_path / "first.csv")
    second = llm_eval.export_authoring_worklist(authoring_fixture, 7, tmp_path / "second.csv")

    assert first.read_bytes() == second.read_bytes()
    header, rows = _read_rows(first)
    assert header == list(llm_eval.AUTHORING_HEADER)
    assert len(rows) == 30
    assert len({row["cluster_id"] for row in rows}) == 30
    assert Counter(row["category"] for row in rows) == dict.fromkeys(llm_eval.CATEGORIES, 5)
    assert Counter(row["fired_status"] for row in rows) == {
        "fired": 15,
        "control": 15,
    }
    assert all(row["question"] == "" for row in rows)
    assert all(row["relevant_complaint_ids"] == "" for row in rows)
    assert all(row["answerable"] == "" for row in rows)
    assert all(row["privacy_reviewed"] == "" for row in rows)
    for row in rows:
        assert all(row[f"evidence_{index}_complaint_id"] for index in range(1, 11))
        for value in row.values():
            assert not value.startswith(("=", "+", "-", "@"))


def _completed_worklist(path: Path) -> list[dict[str, str]]:
    _, rows = _read_rows(path)
    for index, row in enumerate(rows, start=1):
        row["question"] = f"Synthetic scoped analyst query {index}?"
        if row["category"] == "unanswerable":
            row["answerable"] = "false"
            row["relevant_complaint_ids"] = ""
        else:
            row["answerable"] = "true"
            row["relevant_complaint_ids"] = row["evidence_1_complaint_id"]
        row["privacy_reviewed"] = "y" + "es"
    return rows


def test_authoring_import_strips_private_columns_and_is_stable(authoring_fixture, tmp_path):
    draft = llm_eval.export_authoring_worklist(authoring_fixture, 13, tmp_path / "draft.csv")
    completed = _completed_worklist(draft)
    _write_csv(draft, list(llm_eval.AUTHORING_HEADER), completed)

    first = llm_eval.import_authoring_worklist(
        authoring_fixture, draft, tmp_path / "manifest-1.csv"
    )
    second = llm_eval.import_authoring_worklist(
        authoring_fixture, draft, tmp_path / "manifest-2.csv"
    )

    assert first.read_bytes() == second.read_bytes()
    header, rows = _read_rows(first)
    assert header == list(llm_eval.MANIFEST_HEADER)
    assert [row["question_id"] for row in rows] == sorted(row["question_id"] for row in rows)
    assert all(set(row) == set(llm_eval.MANIFEST_HEADER) for row in rows)
    loaded = llm_eval.load_manifest(first)
    llm_eval.validate_manifest(authoring_fixture, loaded)


def test_authoring_import_requires_human_privacy_review(authoring_fixture, tmp_path):
    draft = llm_eval.export_authoring_worklist(authoring_fixture, 17, tmp_path / "draft.csv")
    completed = _completed_worklist(draft)
    completed[0]["privacy_reviewed"] = ""
    _write_csv(draft, list(llm_eval.AUTHORING_HEADER), completed)

    with pytest.raises(llm_eval.ManifestError, match="privacy_reviewed"):
        llm_eval.import_authoring_worklist(authoring_fixture, draft, tmp_path / "manifest.csv")


def test_authoring_import_rejects_formula_question_and_out_of_scope_id(authoring_fixture, tmp_path):
    draft = llm_eval.export_authoring_worklist(authoring_fixture, 19, tmp_path / "draft.csv")
    completed = _completed_worklist(draft)
    completed[0]["question"] = "=FORMULA"
    _write_csv(draft, list(llm_eval.AUTHORING_HEADER), completed)
    with pytest.raises(llm_eval.ManifestError, match="spreadsheet"):
        llm_eval.import_authoring_worklist(authoring_fixture, draft, tmp_path / "manifest.csv")

    completed[0]["question"] = "Synthetic scoped analyst query?"
    completed[0]["relevant_complaint_ids"] = "999999999"
    _write_csv(draft, list(llm_eval.AUTHORING_HEADER), completed)
    with pytest.raises(llm_eval.ManifestError, match="outside"):
        llm_eval.import_authoring_worklist(authoring_fixture, draft, tmp_path / "manifest.csv")


def test_authoring_export_refuses_a_tracked_repository_destination(
    authoring_fixture,
):
    with pytest.raises(llm_eval.ManifestError, match="interim"):
        llm_eval.export_authoring_worklist(
            authoring_fixture,
            23,
            Path("data/ground_truth/private-authoring.csv"),
        )
