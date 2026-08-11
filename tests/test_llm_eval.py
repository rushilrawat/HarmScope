"""Phase 8D benchmark manifest and private authoring-workflow contracts."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from src.llm import answer, retrieve
from src.llm import eval as llm_eval
from src.llm.client import ModelCallError, TokenUsage


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


@pytest.mark.parametrize("malformation", ["extra", "missing"])
def test_manifest_parser_rejects_wrong_row_arity(tmp_path, malformation):
    path = _write_csv(
        tmp_path / "manifest.csv",
        list(llm_eval.MANIFEST_HEADER),
        _small_manifest_rows(),
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    if malformation == "extra":
        lines[1] += ",unexpected"
    else:
        lines[-1] = lines[-1].rsplit(",", 1)[0]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(llm_eval.ManifestError, match="field count"):
        llm_eval.load_manifest(path, expected_n=6)


@pytest.fixture
def validation_fixture(con):
    dedup_run = "0000000000000-dedup001"
    run_id = "0000000000001-cluster1"
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'dedup', 'test', 'test', '{}', now(), 'ok')",
        [dedup_run],
    )
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'cluster', 'test', 'test', ?, now(), 'ok')",
        [
            run_id,
            json.dumps(
                {
                    "params": {
                        "dedup_run": dedup_run,
                        "model": "embed-m",
                        "cutoff": "2020-02-01",
                    }
                }
            ),
        ],
    )
    for cluster_id in ("cluster-1", "cluster-2"):
        con.execute(
            "INSERT INTO clusters "
            "(cluster_id, run_id, product_family, n_members, as_of) "
            "VALUES (?, ?, 'family-1', 8, ?)",
            [cluster_id, run_id, date(2020, 1, 1)],
        )
    for company_id in ("company-1", "company-2", "company-3"):
        con.execute(
            "INSERT INTO company_canonical "
            "(company_id, canonical_name, verified_by) VALUES (?, ?, 'manual')",
            [company_id, company_id],
        )
    records = [
        (10, "cluster-1", "company-1", "family-1", _token_text("alpha")),
        (20, "cluster-1", "company-2", "family-1", _token_text("beta")),
        (30, "cluster-2", "company-1", "family-1", _token_text("gamma")),
        (50, None, "company-2", "family-1", _token_text("delta")),
        (60, "cluster-1", "company-1", "family-1", _token_text("campaign")),
        (70, "cluster-1", "company-1", "family-1", _token_text("future")),
        (80, "cluster-1", "company-1", "family-2", _token_text("family")),
        (90, "cluster-1", "company-1", "family-1", _token_text("unembedded")),
    ]
    for row_idx, (complaint_id, cluster_id, company_id, family, text) in enumerate(records):
        received = date(2020, 3, 1) if complaint_id == 70 else date(2020, 1, 1)
        con.execute(
            "INSERT INTO complaints "
            "(complaint_id, date_received, period_month, company_id, "
            "product_family, has_narrative) VALUES (?, ?, ?, ?, ?, true)",
            [complaint_id, received, received.replace(day=1), company_id, family],
        )
        con.execute(
            "INSERT INTO narratives "
            "(complaint_id, text_redacted, text_hash, redaction_count) "
            "VALUES (?, ?, ?, 0)",
            [complaint_id, text, f"hash-{complaint_id}"],
        )
        if complaint_id != 90:
            con.execute(
                "INSERT INTO embedding_map (complaint_id, row_idx, model, dim) "
                "VALUES (?, ?, 'embed-m', 2)",
                [complaint_id, row_idx],
            )
        group_id = "group-10" if complaint_id in {10, 50} else f"group-{complaint_id}"
        con.execute(
            "INSERT INTO dup_groups "
            "(run_id, complaint_id, group_id, is_representative, group_size, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                dedup_run,
                complaint_id,
                group_id,
                complaint_id != 50,
                2 if complaint_id in {10, 50} else 1,
                date(2020, 3, 1),
            ],
        )
        if cluster_id is not None:
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
    con.execute(
        "INSERT INTO campaigns "
        "(campaign_id, run_id, n_complaints, n_groups, product_family, n_signals, "
        "flagged, as_of) VALUES ('campaign-1', ?, 1, 1, 'family-1', 3, true, ?)",
        [dedup_run, date(2020, 1, 1)],
    )
    con.execute(
        "INSERT INTO campaign_members (complaint_id, campaign_id) VALUES (60, 'campaign-1')"
    )
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


def test_retrieval_scope_requires_retrievable_narrative_evidence(validation_fixture):
    question = llm_eval.EvalQuestion(
        question_id="rag-006",
        question="Synthetic unavailable fact?",
        cluster_id="cluster-1",
        company_id="company-3",
        category="unanswerable",
        relevant_complaint_ids=frozenset(),
        answerable=False,
    )

    with pytest.raises(llm_eval.ManifestError, match="retrievable evidence"):
        llm_eval.validate_manifest(validation_fixture, [question])


def test_validation_includes_nonrepresentative_cross_company_duplicate(
    validation_fixture,
):
    corpus = retrieve.load_corpus(validation_fixture, "cluster-1", "company-2", "embed-m")
    assert [row.complaint_id for row in corpus.rows] == [20, 50]

    question = _question(company_id="company-2", relevant_ids=frozenset({50}))
    llm_eval.validate_manifest(validation_fixture, [question])

    overlap = " ".join(f"delta{index}" for index in range(8))
    with pytest.raises(llm_eval.ManifestError, match="eight-token"):
        llm_eval.validate_manifest(
            validation_fixture,
            [_question(question=overlap, company_id="company-2", relevant_ids=frozenset({50}))],
        )


@pytest.mark.parametrize("excluded_id", [60, 70, 80, 90])
def test_validation_rejects_ids_excluded_by_retrieval_scope(validation_fixture, excluded_id):
    with pytest.raises(llm_eval.ManifestError, match="outside"):
        llm_eval.validate_manifest(
            validation_fixture,
            [_question(company_id="company-1", relevant_ids=frozenset({excluded_id}))],
        )


@pytest.fixture
def authoring_fixture(con):
    dedup_run = "0000000000000-dedup001"
    cluster_run = "0000000000001-cluster1"
    signals_run = "0000000000002-signal01"
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'dedup', 'test', 'test', '{}', now(), 'ok')",
        [dedup_run],
    )
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'cluster', 'test', 'test', ?, now(), 'ok')",
        [
            cluster_run,
            json.dumps(
                {
                    "params": {
                        "dedup_run": dedup_run,
                        "model": "embed-m",
                        "cutoff": "",
                    }
                }
            ),
        ],
    )
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'signals', 'test', 'test', ?, now(), 'ok')",
        [signals_run, json.dumps({"params": {"cluster_run": cluster_run}})],
    )
    for company_number in range(42):
        company_id = f"company-{company_number:02d}"
        con.execute(
            "INSERT INTO company_canonical "
            "(company_id, canonical_name, verified_by) VALUES (?, ?, 'manual')",
            [company_id, company_id],
        )
    complaint_id = 1_000
    for cluster_number in range(42):
        cluster_id = f"{cluster_run}:family-{cluster_number % 6}:{cluster_number:02d}"
        family = f"family-{cluster_number % 6}"
        company_id = f"company-{cluster_number:02d}"
        con.execute(
            "INSERT INTO clusters "
            "(cluster_id, run_id, product_family, n_members, coherence, as_of) "
            "VALUES (?, ?, ?, 10, 0.9, ?)",
            [cluster_id, cluster_run, family, date(2020, 1, 1)],
        )
        if cluster_number < 9:
            con.execute(
                "INSERT INTO signals "
                "(signal_id, run_id, cluster_id, company_id, period_month, method, "
                "statistic, q_value, n_supporting, n_supporting_groups, as_of) "
                "VALUES (?, ?, ?, ?, ?, 'ebgm', 1.0, 0.01, 20, 15, ?)",
                [
                    f"signal-{cluster_number}",
                    signals_run,
                    cluster_id,
                    company_id,
                    date(2020, 1, 1),
                    date(2020, 1, 1),
                ],
            )
        elif cluster_number < 18:
            con.execute(
                "INSERT INTO signals "
                "(signal_id, run_id, cluster_id, company_id, period_month, method, "
                "statistic, q_value, n_supporting, n_supporting_groups, as_of) "
                "VALUES (?, ?, ?, ?, ?, 'ewma', 1.0, NULL, 20, 15, ?)",
                [
                    f"signal-{cluster_number}",
                    signals_run,
                    cluster_id,
                    company_id,
                    date(2020, 1, 1),
                    date(2020, 1, 1),
                ],
            )
        elif cluster_number == 18:
            con.execute(
                "INSERT INTO signals "
                "(signal_id, run_id, cluster_id, company_id, period_month, method, "
                "statistic, q_value, n_supporting, n_supporting_groups, as_of) "
                "VALUES (?, ?, ?, ?, ?, 'ebgm', 1.0, 0.01, 20, 14, ?)",
                [
                    "signal-noncanonical",
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
                "INSERT INTO embedding_map (complaint_id, row_idx, model, dim) "
                "VALUES (?, ?, 'embed-m', 2)",
                [complaint_id, complaint_id - 1_000],
            )
            con.execute(
                "INSERT INTO dup_groups "
                "(run_id, complaint_id, group_id, is_representative, group_size, as_of) "
                "VALUES (?, ?, ?, true, 1, ?)",
                [dedup_run, complaint_id, f"group-{complaint_id}", date(2020, 1, 1)],
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


def test_authoring_status_matches_canonical_alert_gate(authoring_fixture, tmp_path):
    path = llm_eval.export_authoring_worklist(authoring_fixture, 29, tmp_path / "canonical.csv")
    _, rows = _read_rows(path)
    canonical = {
        (cluster_id, company_id)
        for cluster_id, company_id in authoring_fixture.execute(
            """
            WITH grouped AS (
              SELECT cluster_id, company_id,
                     min(CASE WHEN method = 'ebgm' THEN q_value END) AS q_value,
                     max(CASE WHEN method IN ('ewma', 'pelt') THEN 1 ELSE 0 END) AS changed,
                     max(n_supporting_groups) AS n_groups
              FROM signals WHERE run_id = '0000000000002-signal01'
              GROUP BY 1, 2
            )
            SELECT g.cluster_id, g.company_id
            FROM grouped g JOIN clusters c USING (cluster_id)
            WHERE c.coherence >= 0.45 AND g.n_groups >= 15
              AND (g.q_value <= 0.05 OR g.changed = 1)
            """
        ).fetchall()
    }
    canonical_clusters = {cluster_id for cluster_id, _ in canonical}
    for row in rows:
        scope = (row["cluster_id"], row["company_id"] or "__ALL__")
        if row["fired_status"] == "fired":
            assert scope in canonical
        else:
            assert row["cluster_id"] not in canonical_clusters


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


@pytest.mark.parametrize("malformation", ["extra", "missing"])
def test_authoring_import_rejects_wrong_row_arity(authoring_fixture, tmp_path, malformation):
    draft = llm_eval.export_authoring_worklist(authoring_fixture, 31, tmp_path / "draft.csv")
    completed = _completed_worklist(draft)
    _write_csv(draft, list(llm_eval.AUTHORING_HEADER), completed)
    lines = draft.read_text(encoding="utf-8").splitlines()
    if malformation == "extra":
        lines[1] += ",unexpected"
    else:
        lines[-1] = lines[-1].rsplit(",", 1)[0]
    draft.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(llm_eval.ManifestError, match="field count"):
        llm_eval.import_authoring_worklist(authoring_fixture, draft, tmp_path / "manifest.csv")


def test_export_write_failure_preserves_destination_and_cleans_temporary_file(
    authoring_fixture, tmp_path, monkeypatch
):
    destination = tmp_path / "draft.csv"
    original = b"existing valid destination\n"
    destination.write_bytes(original)

    def fail_mid_write(writer, rows):
        writer.writerow(rows[0])
        raise OSError("simulated mid-write failure")

    monkeypatch.setattr(csv.DictWriter, "writerows", fail_mid_write)
    with pytest.raises(OSError, match="simulated mid-write failure"):
        llm_eval.export_authoring_worklist(authoring_fixture, 37, destination)

    assert destination.read_bytes() == original
    assert list(tmp_path.glob(".draft.csv.*.tmp")) == []


def test_authoring_export_refuses_a_tracked_repository_destination(
    authoring_fixture,
):
    with pytest.raises(llm_eval.ManifestError, match="interim"):
        llm_eval.export_authoring_worklist(
            authoring_fixture,
            23,
            Path("data/ground_truth/private-authoring.csv"),
        )


def _retrieval_question(
    question_id: str,
    *,
    relevant_ids: frozenset[int] = frozenset({10}),
    answerable: bool = True,
    question_text: str | None = None,
) -> llm_eval.EvalQuestion:
    return llm_eval.EvalQuestion(
        question_id=question_id,
        question=question_text or f"Synthetic retrieval query {question_id}?",
        cluster_id="cluster-1",
        company_id="company-1",
        category="mechanism" if answerable else "unanswerable",
        relevant_complaint_ids=relevant_ids,
        answerable=answerable,
    )


def _retrieval_result(
    *,
    dense_ids: tuple[int, ...] = (10, 20, 30),
    bm25_ids: tuple[int, ...] = (20, 10, 30),
    latencies: tuple[float, float, float] = (0.01, 0.02, 0.003),
) -> retrieve.RetrievalResult:
    dense = tuple(
        retrieve.RankedHit(complaint_id, rank, 1.0 / rank)
        for rank, complaint_id in enumerate(dense_ids, start=1)
    )
    sparse = tuple(
        retrieve.RankedHit(complaint_id, rank, 1.0 / rank)
        for rank, complaint_id in enumerate(bm25_ids, start=1)
    )
    fused = tuple(
        retrieve.reciprocal_rank_fusion(
            list(dense),
            list(sparse),
            llm_eval.CONFIG.llm.rrf_k,
            llm_eval.CONFIG.llm.rag_top_k,
        )
    )
    return retrieve.RetrievalResult(
        corpus=retrieve.ScopedCorpus("cluster-1", "company-1", "embed-m", ()),
        dense=dense,
        sparse=sparse,
        fused=fused,
        evidence=(),
        dense_seconds=latencies[0],
        sparse_seconds=latencies[1],
        fusion_seconds=latencies[2],
    )


def _valid_rrf_result(
    *,
    dense_ids: tuple[int, ...] = (10, 20),
    bm25_ids: tuple[int, ...] = (20, 10),
) -> retrieve.RetrievalResult:
    return _retrieval_result(dense_ids=dense_ids, bm25_ids=bm25_ids)


class _VariantRetriever:
    def __init__(self, results: dict[str, retrieve.RetrievalResult]):
        self.results = results
        self.calls: list[tuple[str, str | None, str, str]] = []

    def __call__(self, con, cluster_id, company_id, question, embed_model):
        del con
        self.calls.append((cluster_id, company_id, question, embed_model))
        return self.results[question]


class _NoDatabaseConnection:
    def __init__(self):
        self.calls: list[str] = []

    def execute(self, query, parameters=None):
        del parameters
        self.calls.append(query)
        raise AssertionError("invalid evaluation input touched the database")


def test_score_ranking_uses_all_relevant_ids_as_denominator_and_unique_hits():
    got = llm_eval.score_ranking([9, 2, 2, 4, 6], {2, 4, 6, 8}, k=4)

    assert got == llm_eval.RetrievalMetrics(
        rank_first_relevant=2,
        relevant_retrieved_count=2,
        recall_at_10=pytest.approx(0.5),
        reciprocal_rank=pytest.approx(0.5),
    )


def test_score_ranking_returns_zero_metrics_when_nothing_is_relevant():
    assert llm_eval.score_ranking([1, 2, 3], set()) == llm_eval.RetrievalMetrics(None, 0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("ranked_ids", "relevant_ids", "k", "match"),
    [
        ((1, 2), {1}, 10, "list"),
        ([1, True], {1}, 10, "positive integers"),
        ([1, 0], {1}, 10, "positive integers"),
        ([1, 2], [1], 10, "set"),
        ([1, 2], {True}, 10, "positive integers"),
        ([1, 2], {1}, True, "positive integer"),
        ([1, 2], {1}, 0, "positive integer"),
    ],
)
def test_score_ranking_rejects_invalid_boundary_values(ranked_ids, relevant_ids, k, match):
    with pytest.raises((TypeError, ValueError), match=match):
        llm_eval.score_ranking(ranked_ids, relevant_ids, k=k)


def test_evaluate_retrieval_question_scores_exactly_three_orderings():
    got = llm_eval.evaluate_retrieval_question(
        _retrieval_question("rag-001"),
        _retrieval_result(),
    )

    assert set(got) == {"dense", "bm25", "fused"}
    assert got["dense"].rank_first_relevant == 1
    assert got["bm25"].rank_first_relevant == 2
    assert got["fused"].rank_first_relevant == 1


def test_evaluate_retrieval_question_rejects_wrong_scope_and_rank_shapes():
    question = _retrieval_question("rag-001")
    wrong_scope = _retrieval_result()
    object.__setattr__(
        wrong_scope,
        "corpus",
        retrieve.ScopedCorpus("other-cluster", "company-1", "embed-m", ()),
    )
    with pytest.raises(ValueError, match="scope"):
        llm_eval.evaluate_retrieval_question(question, wrong_scope)

    malformed = _retrieval_result()
    object.__setattr__(
        malformed,
        "dense",
        (
            retrieve.RankedHit(10, 2, 1.0),
            retrieve.RankedHit(10, 1, 0.5),
        ),
    )
    with pytest.raises(ValueError, match="dense.*rank|dense.*duplicate"):
        llm_eval.evaluate_retrieval_question(question, malformed)


@pytest.mark.parametrize(
    "malformation",
    [
        "outside-component-union",
        "dense-rank",
        "sparse-score",
        "rank-score-presence",
        "reversed-order",
        "wrong-fused-score",
        "truncated",
    ],
)
def test_evaluate_retrieval_question_rejects_fused_output_not_produced_by_rrf(
    malformation,
):
    question = _retrieval_question("rag-001")
    result = _valid_rrf_result()
    fused = result.fused
    if malformation == "outside-component-union":
        malformed = (
            retrieve.FusedHit(999, fused[0].fused_score, None, None, None, None),
            fused[1],
        )
    elif malformation == "dense-rank":
        malformed = (replace(fused[0], dense_rank=2), fused[1])
    elif malformation == "sparse-score":
        malformed = (replace(fused[0], sparse_score=999.0), fused[1])
    elif malformation == "rank-score-presence":
        malformed = (replace(fused[0], sparse_rank=None), fused[1])
    elif malformation == "reversed-order":
        malformed = tuple(reversed(fused))
    elif malformation == "wrong-fused-score":
        malformed = (replace(fused[0], fused_score=0.99), fused[1])
    else:
        malformed = fused[:-1]
    object.__setattr__(result, "fused", malformed)

    with pytest.raises(ValueError, match="fused"):
        llm_eval.evaluate_retrieval_question(question, result)


def test_evaluate_retrieval_question_rejects_fused_rows_when_components_are_empty():
    result = _valid_rrf_result(dense_ids=(), bm25_ids=())
    object.__setattr__(
        result,
        "fused",
        (retrieve.FusedHit(999, 0.5, None, None, None, None),),
    )

    with pytest.raises(ValueError, match="fused"):
        llm_eval.evaluate_retrieval_question(_retrieval_question("rag-001"), result)


@pytest.mark.parametrize("length_change", ["truncated", "extra"])
def test_evaluate_retrieval_question_requires_configured_fused_top_k(length_change):
    dense_ids = tuple(range(1, llm_eval.CONFIG.llm.rag_top_k + 3))
    result = _valid_rrf_result(dense_ids=dense_ids, bm25_ids=())
    assert len(result.fused) == llm_eval.CONFIG.llm.rag_top_k
    if length_change == "truncated":
        malformed = result.fused[:-1]
    else:
        full = retrieve.reciprocal_rank_fusion(
            list(result.dense),
            list(result.sparse),
            llm_eval.CONFIG.llm.rrf_k,
            len(dense_ids),
        )
        malformed = tuple(full)
    object.__setattr__(result, "fused", malformed)

    with pytest.raises(ValueError, match="fused"):
        llm_eval.evaluate_retrieval_question(
            _retrieval_question("rag-001", relevant_ids=frozenset({1})), result
        )


def test_retrieval_eval_persists_three_variants_once_and_aggregates_answerable_only(con):
    questions = [
        _retrieval_question("rag-001"),
        _retrieval_question(
            "rag-002",
            relevant_ids=frozenset(),
            answerable=False,
        ),
    ]
    results = {question.question: _retrieval_result() for question in questions}
    retriever = _VariantRetriever(results)

    summary = llm_eval.run_retrieval_eval(
        con,
        questions,
        "embed-m",
        "eval-1",
        retriever=retriever,
    )

    rows = con.execute(
        "SELECT question_id, retrieval_method, rank_first_relevant, "
        "recall_at_10, reciprocal_rank, latency_seconds "
        "FROM rag_eval_results ORDER BY question_id, retrieval_method"
    ).fetchall()
    assert len(rows) == 6
    assert {method for _, method, *_ in rows} == {"dense", "bm25", "fused"}
    first_latencies = {method: latency for qid, method, *_, latency in rows if qid == "rag-001"}
    assert first_latencies == {"dense": 0.01, "bm25": 0.02, "fused": 0.003}
    assert len(retriever.calls) == 2
    assert all(call[-1] == "embed-m" for call in retriever.calls)
    assert summary.answerable_count == 1
    assert summary.unanswerable_count == 1
    assert summary.methods["dense"].recall_at_10 == pytest.approx(1.0)
    assert summary.methods["dense"].reciprocal_rank == pytest.approx(1.0)
    assert summary.methods["bm25"].reciprocal_rank == pytest.approx(0.5)
    assert summary.methods["fused"].reciprocal_rank == pytest.approx(1.0)
    assert summary.fused_vs_dense == llm_eval.WinTieLoss(wins=0, ties=2, losses=0)
    assert summary.fused_vs_bm25 == llm_eval.WinTieLoss(wins=1, ties=1, losses=0)


class _FailingInsertConnection:
    def __init__(self, con, fail_on_insert: int):
        self._con = con
        self._fail_on_insert = fail_on_insert
        self._inserts = 0

    def execute(self, query, parameters=None):
        if query.lstrip().startswith("INSERT INTO rag_eval_results"):
            self._inserts += 1
            if self._inserts == self._fail_on_insert:
                raise RuntimeError("simulated second-method insert failure")
        return (
            self._con.execute(query) if parameters is None else self._con.execute(query, parameters)
        )


def test_retrieval_eval_rolls_back_one_question_but_preserves_prior_question(con):
    questions = [_retrieval_question("rag-001"), _retrieval_question("rag-002")]
    retriever = _VariantRetriever(
        {question.question: _retrieval_result() for question in questions}
    )
    wrapped = _FailingInsertConnection(con, fail_on_insert=5)

    with pytest.raises(RuntimeError, match="second-method"):
        llm_eval.run_retrieval_eval(
            wrapped,
            questions,
            "embed-m",
            "eval-atomic",
            retriever=retriever,
        )

    assert con.execute(
        "SELECT question_id, retrieval_method FROM rag_eval_results ORDER BY 1, 2"
    ).fetchall() == [
        ("rag-001", "bm25"),
        ("rag-001", "dense"),
        ("rag-001", "fused"),
    ]


def test_retrieval_eval_replay_updates_metrics_without_erasing_answer_review_fields(con):
    question = _retrieval_question("rag-001")
    first = _VariantRetriever({question.question: _retrieval_result()})
    llm_eval.run_retrieval_eval(con, [question], "embed-m", "eval-replay", retriever=first)
    con.execute(
        "UPDATE rag_eval_results SET citation_valid = true, citation_coverage = 0.75, "
        "abstention_correct = true, grounded_claims = 7, reviewed_claims = 8 "
        "WHERE eval_run_id = 'eval-replay' AND retrieval_method = 'fused'"
    )
    created_at = con.execute(
        "SELECT created_at FROM rag_eval_results WHERE eval_run_id = 'eval-replay' "
        "AND retrieval_method = 'fused'"
    ).fetchone()[0]
    changed = _VariantRetriever(
        {
            question.question: _retrieval_result(
                dense_ids=(20, 10),
                bm25_ids=(10, 20),
                latencies=(0.5, 0.6, 0.7),
            )
        }
    )

    llm_eval.run_retrieval_eval(
        con,
        [question],
        "embed-m",
        "eval-replay",
        retriever=changed,
    )

    assert con.execute(
        "SELECT count(*) FROM rag_eval_results WHERE eval_run_id = 'eval-replay'"
    ).fetchone() == (3,)
    assert con.execute(
        "SELECT rank_first_relevant, latency_seconds, citation_valid, citation_coverage, "
        "abstention_correct, grounded_claims, reviewed_claims, created_at "
        "FROM rag_eval_results WHERE eval_run_id = 'eval-replay' "
        "AND retrieval_method = 'fused'"
    ).fetchone() == (1, 0.7, True, 0.75, True, 7, 8, created_at)


@pytest.mark.parametrize("resolution", ["COMMIT", "ROLLBACK"])
def test_retrieval_eval_rejects_caller_transaction_before_retrieval(con, resolution):
    con.execute("CREATE TABLE caller_eval_work (value INTEGER)")
    con.execute("BEGIN TRANSACTION")
    con.execute("INSERT INTO caller_eval_work VALUES (1)")
    question = _retrieval_question("rag-001")
    retriever = _VariantRetriever({question.question: _retrieval_result()})

    with pytest.raises(llm_eval.EvaluationTransactionError, match="autocommit"):
        llm_eval.run_retrieval_eval(
            con,
            [question],
            "embed-m",
            "eval-caller-tx",
            retriever=retriever,
        )

    assert retriever.calls == []
    assert con.execute("SELECT value FROM caller_eval_work").fetchall() == [(1,)]
    assert con.execute("SELECT count(*) FROM rag_eval_results").fetchone() == (0,)
    con.execute(resolution)
    expected = [(1,)] if resolution == "COMMIT" else []
    assert con.execute("SELECT value FROM caller_eval_work").fetchall() == expected


@pytest.mark.parametrize(
    ("questions", "embed_model", "eval_run_id", "match"),
    [
        (
            [_retrieval_question("rag-001"), _retrieval_question("rag-001")],
            "embed-m",
            "eval-1",
            "duplicate question_id",
        ),
        ([_retrieval_question("rag-001")], "", "eval-1", "embed_model"),
        ([_retrieval_question("rag-001")], "embed-m", "", "eval_run_id"),
        ([_retrieval_question("rag-001")], "embed-m", "unsafe run id\n", "eval_run_id"),
        ([_retrieval_question("rag-001")], "embed-m", " eval-1", "eval_run_id"),
        ([_retrieval_question("rag-001")], "embed-m", "=eval-1", "eval_run_id"),
        ([_retrieval_question("rag-001")], "embed-m", "eval/1", "eval_run_id"),
        (
            [_retrieval_question("rag-001", relevant_ids=frozenset())],
            "embed-m",
            "eval-1",
            "answerable",
        ),
        (
            [
                _retrieval_question(
                    "rag-001",
                    relevant_ids=frozenset({10}),
                    answerable=False,
                )
            ],
            "embed-m",
            "eval-1",
            "unanswerable",
        ),
    ],
)
def test_retrieval_eval_validates_batch_before_retrieval(
    con, questions, embed_model, eval_run_id, match
):
    retriever = _VariantRetriever(
        {question.question: _retrieval_result() for question in questions}
    )

    with pytest.raises(ValueError, match=match):
        llm_eval.run_retrieval_eval(
            con,
            questions,
            embed_model,
            eval_run_id,
            retriever=retriever,
        )

    assert retriever.calls == []


@pytest.mark.parametrize(
    ("questions", "eval_run_id", "match"),
    [
        ([], "eval-1", "non-empty"),
        ([_retrieval_question("rag-001")], "unsafe run id", "eval_run_id"),
        (
            [_retrieval_question("rag-001"), _retrieval_question("rag-001")],
            "eval-1",
            "duplicate question_id",
        ),
    ],
)
def test_invalid_retrieval_batch_touches_neither_database_nor_retriever(
    questions, eval_run_id, match
):
    con = _NoDatabaseConnection()
    retriever = _VariantRetriever(
        {question.question: _retrieval_result() for question in questions}
    )

    with pytest.raises(ValueError, match=match):
        llm_eval.run_retrieval_eval(
            con,
            questions,
            "embed-m",
            eval_run_id,
            retriever=retriever,
        )

    assert con.calls == []
    assert retriever.calls == []


def test_retrieval_summary_reports_linear_p95_and_every_fusion_loss_without_narratives(con):
    questions: list[llm_eval.EvalQuestion] = []
    results: dict[str, retrieve.RetrievalResult] = {}
    for index in range(1, 21):
        question = _retrieval_question(
            f"rag-{index:03d}",
            question_text=f"private narrative phrase {index}",
        )
        questions.append(question)
        dense_ids = (10, 20)
        bm25_ids = (10, 20)
        if index == 19:
            bm25_ids = (20, 30)
        elif index == 20:
            dense_ids = (20, 30)
        results[question.question] = _retrieval_result(
            dense_ids=dense_ids,
            bm25_ids=bm25_ids,
            latencies=(index / 1000, index / 500, index / 2000),
        )
    summary = llm_eval.run_retrieval_eval(
        con,
        questions,
        "embed-m",
        "eval-report",
        retriever=_VariantRetriever(results),
    )

    assert summary.methods["dense"].median_latency_seconds == pytest.approx(0.0105)
    assert summary.methods["dense"].p95_latency_seconds == pytest.approx(0.01905)
    assert summary.fused_vs_dense == llm_eval.WinTieLoss(wins=1, ties=18, losses=1)
    report = summary.render()
    assert report == summary.render()
    assert "answerable: 20" in report
    assert "unanswerable: 0" in report
    assert "Recall@10" in report
    assert "MRR" in report
    assert "fused vs dense win/tie/loss: 1/18/1" in report
    assert "fused vs BM25 win/tie/loss: 1/18/1" in report
    assert "rag-019" in report and "rag-020" in report and "loss" in report
    assert "private narrative phrase" not in report
    assert all(math.isfinite(method.p95_latency_seconds) for method in summary.methods.values())


def _grounded_answer(
    *,
    claims: tuple[answer.Claim, ...] = (
        answer.Claim("Consumers allege the first problem.", (10,)),
        answer.Claim("Consumers allege the second problem.", (20,)),
    ),
    insufficient_evidence: bool = False,
) -> answer.GroundedAnswer:
    return answer.GroundedAnswer(
        answer=" ".join(claim.text for claim in claims),
        claims=claims,
        insufficient_evidence=insufficient_evidence,
        limitation_reasons=(),
    )


def test_citation_metrics_are_deterministic_and_use_all_claims():
    grounded = _grounded_answer()

    assert llm_eval.citation_validity(grounded, {10, 20}) is True
    assert llm_eval.citation_validity(grounded, {10}) is False
    assert llm_eval.citation_coverage(grounded) == 1.0

    partially_cited = _grounded_answer(
        claims=(
            answer.Claim("Consumers allege the first problem.", (10,)),
            answer.Claim("Consumers allege the second problem.", ()),
        )
    )
    assert llm_eval.citation_validity(partially_cited, {10}) is False
    assert llm_eval.citation_coverage(partially_cited) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "complaint_ids",
    [(), (10, 10), (0,), (-1,), (True,), ("10",)],
)
def test_citation_validity_fails_closed_for_malformed_claim_citations(complaint_ids):
    grounded = _grounded_answer(
        claims=(answer.Claim("Consumers allege a problem.", complaint_ids),)
    )

    assert llm_eval.citation_validity(grounded, {10}) is False


def test_zero_claim_coverage_distinguishes_abstention_from_empty_answer():
    abstention = _grounded_answer(claims=(), insufficient_evidence=True)
    empty_answer = _grounded_answer(claims=(), insufficient_evidence=False)

    assert llm_eval.citation_validity(abstention, set()) is True
    assert llm_eval.citation_coverage(abstention) == 1.0
    assert llm_eval.citation_coverage(empty_answer) == 0.0


def test_citation_coverage_counts_presence_separately_from_citation_validity():
    duplicate_citation = _grounded_answer(
        claims=(answer.Claim("Consumers allege a problem.", (10, 10)),)
    )

    assert llm_eval.citation_validity(duplicate_citation, {10}) is False
    assert llm_eval.citation_coverage(duplicate_citation) == 1.0


@pytest.mark.parametrize(
    ("answerable", "insufficient", "expected"),
    [(True, False, True), (True, True, False), (False, True, True), (False, False, False)],
)
def test_abstention_accuracy_matrix(answerable, insufficient, expected):
    assert llm_eval.abstention_correct(answerable, insufficient) is expected


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: llm_eval.citation_validity("answer", {10}), "GroundedAnswer"),
        (lambda: llm_eval.citation_validity(_grounded_answer(), [10]), "set"),
        (lambda: llm_eval.citation_coverage("answer"), "GroundedAnswer"),
        (lambda: llm_eval.abstention_correct(1, False), "boolean"),
        (lambda: llm_eval.abstention_correct(True, 0), "boolean"),
    ],
)
def test_answer_metrics_reject_wrong_boundary_types(call, match):
    with pytest.raises(TypeError, match=match):
        call()


def _seed_retrieval_rows(
    con,
    eval_run_id: str,
    questions: list[llm_eval.EvalQuestion],
) -> None:
    for question in questions:
        for method in ("dense", "bm25", "fused"):
            con.execute(
                "INSERT INTO rag_eval_results "
                "(eval_run_id, question_id, retrieval_method, rank_first_relevant, "
                "relevant_retrieved_count, recall_at_10, reciprocal_rank, latency_seconds, "
                "created_at) VALUES (?, ?, ?, 1, 1, 0.5, 0.25, 0.125, now())",
                [eval_run_id, question.question_id, method],
            )


def _answer_evidence(
    question: llm_eval.EvalQuestion,
    complaint_id: int = 10,
    *,
    cluster_id: str | None = None,
    company_id: str | None = None,
) -> retrieve.RetrievedEvidence:
    return retrieve.RetrievedEvidence(
        complaint_id=complaint_id,
        cluster_id=question.cluster_id if cluster_id is None else cluster_id,
        date_received=date(2020, 1, 2),
        company_id=question.company_id if company_id is None else company_id,
        company_name="Private company name",
        product_family="family-1",
        text_redacted="Private narrative phrase that must never enter the report.",
        company_public_response=None,
        dense_rank=1,
        dense_score=0.9,
        sparse_rank=1,
        sparse_score=0.8,
        fused_score=0.1,
    )


def _answer_result(
    question: llm_eval.EvalQuestion,
    *,
    grounded: answer.GroundedAnswer | None = None,
    evidence_rows: tuple[retrieve.RetrievedEvidence, ...] | None = None,
    cache_status: str = "miss",
) -> answer.AnswerResult:
    if evidence_rows is None:
        evidence_rows = (_answer_evidence(question),)
    if grounded is None:
        grounded = _grounded_answer(
            claims=(answer.Claim("Consumers allege a problem.", (10,)),),
            insufficient_evidence=False,
        )
    return answer.AnswerResult(
        answer=grounded,
        evidence=evidence_rows,
        enforcement_context=(),
        cache_status=cache_status,
        usage=TokenUsage(),
        latency_seconds=0.0,
        estimated_cost_usd=0.0,
    )


def _insert_answer_usage(
    con,
    *,
    usage_id: str,
    eval_run_id: str,
    question: llm_eval.EvalQuestion,
    cache_status: str,
    outcome: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    latency_seconds: float = 0.0,
    estimated_cost_usd: float = 0.0,
    operation: str = "answer",
) -> None:
    con.execute(
        "INSERT INTO llm_usage "
        "(usage_id, run_id, operation, cluster_id, question_hash, model, prompt_version, "
        "input_hash, cache_status, attempts, input_tokens, output_tokens, "
        "cache_read_input_tokens, cache_creation_input_tokens, latency_seconds, "
        "estimated_cost_usd, outcome, error_category, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'model-private', 'prompt-private', ?, ?, 1, ?, ?, ?, ?, "
        "?, ?, ?, NULL, now())",
        [
            usage_id,
            eval_run_id,
            operation,
            question.cluster_id,
            f"hash-{question.question_id}",
            f"input-{question.question_id}",
            cache_status,
            input_tokens,
            output_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
            latency_seconds,
            estimated_cost_usd,
            outcome,
        ],
    )


class _Answerer:
    def __init__(
        self,
        outcomes: dict[str, answer.AnswerResult | BaseException],
        usage: dict[str, dict[str, object]] | None = None,
    ):
        self.outcomes = outcomes
        self.usage = usage or {}
        self.calls: list[tuple[str, str, str, str, str]] = []

    def __call__(
        self,
        con,
        cluster_id,
        company_id,
        question,
        embed_model,
        *,
        run_id,
    ):
        self.calls.append((cluster_id, company_id, question, embed_model, run_id))
        outcome = self.outcomes[question]
        usage = self.usage.get(question)
        if usage is not None:
            usage_values = dict(usage)
            usage_question = usage_values.pop("question")
            _insert_answer_usage(
                con,
                eval_run_id=run_id,
                question=usage_question,
                **usage_values,
            )
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_answer_eval_updates_only_fused_rows_and_aggregates_exact_usage(con):
    answerable = _retrieval_question("rag-001")
    unanswerable = _retrieval_question("rag-002", relevant_ids=frozenset(), answerable=False)
    bypassed = _retrieval_question("rag-003", relevant_ids=frozenset(), answerable=False)
    questions = [answerable, unanswerable, bypassed]
    _seed_retrieval_rows(con, "eval-answer", questions)
    con.execute(
        "UPDATE rag_eval_results SET grounded_claims = 7, reviewed_claims = 8 "
        "WHERE eval_run_id = 'eval-answer' AND question_id = 'rag-001' "
        "AND retrieval_method = 'fused'"
    )
    abstention = _grounded_answer(claims=(), insufficient_evidence=True)
    answerer = _Answerer(
        {
            answerable.question: _answer_result(answerable),
            unanswerable.question: _answer_result(unanswerable, grounded=abstention),
            bypassed.question: _answer_result(
                bypassed,
                grounded=abstention,
                evidence_rows=(),
                cache_status="bypass",
            ),
        },
        {
            answerable.question: {
                "usage_id": "usage-1",
                "question": answerable,
                "cache_status": "miss",
                "outcome": "ok",
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 7,
                "latency_seconds": 0.4,
                "estimated_cost_usd": 0.03,
            },
            unanswerable.question: {
                "usage_id": "usage-2",
                "question": unanswerable,
                "cache_status": "hit",
                "outcome": "ok",
            },
            bypassed.question: {
                "usage_id": "usage-3",
                "question": bypassed,
                "cache_status": "bypass",
                "outcome": "skipped",
            },
        },
    )

    summary = llm_eval.run_answer_eval(con, questions, "embed-m", "eval-answer", answerer=answerer)

    assert answerer.calls == [
        (
            question.cluster_id,
            question.company_id,
            question.question,
            "embed-m",
            "eval-answer",
        )
        for question in questions
    ]
    assert con.execute(
        "SELECT question_id, retrieval_method, citation_valid, citation_coverage, "
        "abstention_correct, grounded_claims, reviewed_claims "
        "FROM rag_eval_results WHERE eval_run_id = 'eval-answer' ORDER BY 1, 2"
    ).fetchall() == [
        ("rag-001", "bm25", None, None, None, None, None),
        ("rag-001", "dense", None, None, None, None, None),
        ("rag-001", "fused", True, 1.0, True, 7, 8),
        ("rag-002", "bm25", None, None, None, None, None),
        ("rag-002", "dense", None, None, None, None, None),
        ("rag-002", "fused", True, 1.0, True, None, None),
        ("rag-003", "bm25", None, None, None, None, None),
        ("rag-003", "dense", None, None, None, None, None),
        ("rag-003", "fused", True, 1.0, True, None, None),
    ]
    assert summary.attempted_count == 3
    assert summary.completed_count == 3
    assert summary.failed_count == 0
    assert summary.citation_valid_count == 3
    assert summary.citation_validity_rate == 1.0
    assert summary.citation_coverage == 1.0
    assert summary.abstention_accuracy == 1.0
    assert summary.input_tokens == 100
    assert summary.output_tokens == 20
    assert summary.cache_read_input_tokens == 5
    assert summary.cache_creation_input_tokens == 7
    assert summary.total_latency_seconds == pytest.approx(0.4)
    assert summary.estimated_cost_usd == pytest.approx(0.03)
    assert (summary.cache_hits, summary.cache_misses, summary.cache_bypasses) == (1, 1, 1)
    assert (summary.outcome_ok, summary.outcome_refused, summary.outcome_failed) == (2, 0, 0)
    assert summary.outcome_skipped == 1


def test_answer_eval_continues_only_typed_per_question_failures_and_keeps_nulls(con):
    questions = [_retrieval_question(f"rag-{index:03d}") for index in range(1, 5)]
    _seed_retrieval_rows(con, "eval-failures", questions)
    failures: list[BaseException] = [
        answer.CitationError("private unsupported citation 999"),
        answer.AnswerSchemaError("private malformed prose"),
        answer.AnswerRefusalError(
            answer.ModelCallResult(
                payload={"refused": True, "stop_reason": "refusal"},
                model="private-model",
                stop_reason="refusal",
                usage=TokenUsage(),
                attempts=1,
                latency_seconds=0.0,
                estimated_cost_usd=0.0,
            )
        ),
    ]
    answerer = _Answerer(
        {
            questions[0].question: failures[0],
            questions[1].question: failures[1],
            questions[2].question: failures[2],
            questions[3].question: _answer_result(questions[3]),
        }
    )

    summary = llm_eval.run_answer_eval(
        con, questions, "embed-m", "eval-failures", answerer=answerer
    )

    assert [(failure.question_id, failure.category) for failure in summary.failures] == [
        ("rag-001", "citation"),
        ("rag-002", "schema"),
        ("rag-003", "refusal"),
    ]
    assert summary.completed_count == 1
    assert summary.failed_count == 3
    assert con.execute(
        "SELECT question_id, citation_valid, citation_coverage, abstention_correct "
        "FROM rag_eval_results WHERE eval_run_id = 'eval-failures' "
        "AND retrieval_method = 'fused' ORDER BY question_id"
    ).fetchall() == [
        ("rag-001", None, None, None),
        ("rag-002", None, None, None),
        ("rag-003", None, None, None),
        ("rag-004", True, 1.0, True),
    ]
    rendered = summary.render()
    assert rendered == summary.render()
    assert "rag-001 | citation" in rendered
    assert "rag-002 | schema" in rendered
    assert "rag-003 | refusal" in rendered
    assert "private" not in rendered.lower()
    assert questions[0].question not in rendered


def test_answer_eval_terminal_provider_error_stops_and_preserves_prior_question(con):
    questions = [_retrieval_question(f"rag-{index:03d}") for index in range(1, 4)]
    _seed_retrieval_rows(con, "eval-terminal", questions)
    terminal = ModelCallError("billing", 1, False)
    answerer = _Answerer(
        {
            questions[0].question: _answer_result(questions[0]),
            questions[1].question: terminal,
            questions[2].question: _answer_result(questions[2]),
        }
    )

    with pytest.raises(ModelCallError) as caught:
        llm_eval.run_answer_eval(con, questions, "embed-m", "eval-terminal", answerer=answerer)

    assert caught.value is terminal
    assert [call[2] for call in answerer.calls] == [
        questions[0].question,
        questions[1].question,
    ]
    assert con.execute(
        "SELECT question_id, citation_valid FROM rag_eval_results "
        "WHERE eval_run_id = 'eval-terminal' AND retrieval_method = 'fused' "
        "ORDER BY question_id"
    ).fetchall() == [("rag-001", True), ("rag-002", None), ("rag-003", None)]


def test_answer_eval_rejects_missing_company_before_database_or_provider():
    question = _retrieval_question("rag-001")
    question = replace(question, company_id=None)
    con = _NoDatabaseConnection()
    answerer = _Answerer({question.question: _answer_result(question)})

    with pytest.raises(ValueError, match="company_id"):
        llm_eval.run_answer_eval(con, [question], "embed-m", "eval-company", answerer=answerer)

    assert con.calls == []
    assert answerer.calls == []


@pytest.mark.parametrize(
    "question",
    [
        replace(
            _retrieval_question("rag-001", relevant_ids=frozenset(), answerable=False),
            category="mechanism",
        ),
        replace(_retrieval_question("rag-001"), category="unanswerable"),
    ],
)
def test_answer_eval_rejects_answerability_category_mismatch_before_database(question):
    con = _NoDatabaseConnection()
    answerer = _Answerer({question.question: _answer_result(question)})

    with pytest.raises(ValueError, match="category.*answerable|answerable.*category"):
        llm_eval.run_answer_eval(
            con,
            [question],
            "embed-m",
            "eval-category",
            answerer=answerer,
        )

    assert con.calls == []
    assert answerer.calls == []


def test_answer_eval_requires_every_fused_row_before_calling_provider(con):
    questions = [_retrieval_question("rag-001"), _retrieval_question("rag-002")]
    _seed_retrieval_rows(con, "eval-missing", questions[:1])
    answerer = _Answerer({question.question: _answer_result(question) for question in questions})

    with pytest.raises(ValueError, match="fused.*rag-002"):
        llm_eval.run_answer_eval(con, questions, "embed-m", "eval-missing", answerer=answerer)

    assert answerer.calls == []


@pytest.mark.parametrize("resolution", ["COMMIT", "ROLLBACK"])
def test_answer_eval_rejects_caller_transaction_without_altering_it(con, resolution):
    question = _retrieval_question("rag-001")
    _seed_retrieval_rows(con, "eval-caller-answer", [question])
    con.execute("CREATE TABLE caller_answer_work (value INTEGER)")
    con.execute("BEGIN TRANSACTION")
    con.execute("INSERT INTO caller_answer_work VALUES (1)")
    answerer = _Answerer({question.question: _answer_result(question)})

    with pytest.raises(llm_eval.EvaluationTransactionError, match="autocommit"):
        llm_eval.run_answer_eval(
            con, [question], "embed-m", "eval-caller-answer", answerer=answerer
        )

    assert answerer.calls == []
    assert con.execute("SELECT value FROM caller_answer_work").fetchall() == [(1,)]
    con.execute(resolution)
    expected = [(1,)] if resolution == "COMMIT" else []
    assert con.execute("SELECT value FROM caller_answer_work").fetchall() == expected


@pytest.mark.parametrize("malformation", ["wrong-cluster", "wrong-company", "duplicate-id"])
def test_answer_eval_rejects_returned_evidence_outside_exact_scope(con, malformation):
    question = _retrieval_question("rag-001")
    _seed_retrieval_rows(con, "eval-evidence", [question])
    first = _answer_evidence(
        question,
        cluster_id="other-cluster" if malformation == "wrong-cluster" else None,
        company_id="other-company" if malformation == "wrong-company" else None,
    )
    rows = (first, first) if malformation == "duplicate-id" else (first,)
    answerer = _Answerer({question.question: _answer_result(question, evidence_rows=rows)})

    with pytest.raises(ValueError, match="evidence"):
        llm_eval.run_answer_eval(con, [question], "embed-m", "eval-evidence", answerer=answerer)

    assert con.execute(
        "SELECT citation_valid, citation_coverage, abstention_correct "
        "FROM rag_eval_results WHERE eval_run_id = 'eval-evidence' "
        "AND retrieval_method = 'fused'"
    ).fetchone() == (None, None, None)


class _FailingAnswerUpdateConnection:
    def __init__(self, con, fail_on_update: int):
        self._con = con
        self._fail_on_update = fail_on_update
        self._updates = 0

    def execute(self, query, parameters=None):
        if query.lstrip().startswith("UPDATE rag_eval_results"):
            self._updates += 1
            if self._updates == self._fail_on_update:
                raise RuntimeError("simulated answer metric update failure")
        return (
            self._con.execute(query) if parameters is None else self._con.execute(query, parameters)
        )


def test_answer_eval_rolls_back_current_metric_update_and_preserves_prior_question(con):
    questions = [_retrieval_question("rag-001"), _retrieval_question("rag-002")]
    _seed_retrieval_rows(con, "eval-answer-atomic", questions)
    answerer = _Answerer({question.question: _answer_result(question) for question in questions})
    wrapped = _FailingAnswerUpdateConnection(con, fail_on_update=2)

    with pytest.raises(RuntimeError, match="metric update"):
        llm_eval.run_answer_eval(
            wrapped,
            questions,
            "embed-m",
            "eval-answer-atomic",
            answerer=answerer,
        )

    assert con.execute(
        "SELECT question_id, citation_valid FROM rag_eval_results "
        "WHERE eval_run_id = 'eval-answer-atomic' AND retrieval_method = 'fused' "
        "ORDER BY question_id"
    ).fetchall() == [("rag-001", True), ("rag-002", None)]


def test_answer_eval_replay_preserves_retrieval_human_and_prior_metrics_on_failure(con):
    question = _retrieval_question("rag-001")
    _seed_retrieval_rows(con, "eval-answer-replay", [question])
    con.execute(
        "UPDATE rag_eval_results SET citation_valid = false, citation_coverage = 0.25, "
        "abstention_correct = false, grounded_claims = 9, reviewed_claims = 10 "
        "WHERE eval_run_id = 'eval-answer-replay' AND retrieval_method = 'fused'"
    )
    before = con.execute(
        "SELECT rank_first_relevant, relevant_retrieved_count, recall_at_10, "
        "reciprocal_rank, latency_seconds, grounded_claims, reviewed_claims, created_at "
        "FROM rag_eval_results WHERE eval_run_id = 'eval-answer-replay' "
        "AND retrieval_method = 'fused'"
    ).fetchone()

    llm_eval.run_answer_eval(
        con,
        [question],
        "embed-m",
        "eval-answer-replay",
        answerer=_Answerer({question.question: _answer_result(question)}),
    )
    after_success = con.execute(
        "SELECT rank_first_relevant, relevant_retrieved_count, recall_at_10, "
        "reciprocal_rank, latency_seconds, grounded_claims, reviewed_claims, created_at, "
        "citation_valid, citation_coverage, abstention_correct "
        "FROM rag_eval_results WHERE eval_run_id = 'eval-answer-replay' "
        "AND retrieval_method = 'fused'"
    ).fetchone()
    assert after_success == (*before, True, 1.0, True)

    summary = llm_eval.run_answer_eval(
        con,
        [question],
        "embed-m",
        "eval-answer-replay",
        answerer=_Answerer({question.question: answer.AnswerSchemaError("private failure")}),
    )
    assert summary.failed_count == 1
    assert con.execute(
        "SELECT citation_valid, citation_coverage, abstention_correct, "
        "grounded_claims, reviewed_claims FROM rag_eval_results "
        "WHERE eval_run_id = 'eval-answer-replay' AND retrieval_method = 'fused'"
    ).fetchone() == (True, 1.0, True, 9, 10)


def test_answer_usage_aggregation_filters_exact_run_and_operation(con):
    question = _retrieval_question("rag-001")
    _seed_retrieval_rows(con, "eval-usage", [question])
    _insert_answer_usage(
        con,
        usage_id="usage-current",
        eval_run_id="eval-usage",
        question=question,
        cache_status="miss",
        outcome="ok",
        input_tokens=11,
        output_tokens=3,
        latency_seconds=0.2,
        estimated_cost_usd=0.01,
    )
    _insert_answer_usage(
        con,
        usage_id="usage-other-run",
        eval_run_id="other-run",
        question=question,
        cache_status="hit",
        outcome="ok",
        input_tokens=999,
    )
    _insert_answer_usage(
        con,
        usage_id="usage-label",
        eval_run_id="eval-usage",
        question=question,
        cache_status="miss",
        outcome="ok",
        input_tokens=888,
        operation="label",
    )

    summary = llm_eval.run_answer_eval(
        con,
        [question],
        "embed-m",
        "eval-usage",
        answerer=_Answerer({question.question: _answer_result(question)}),
    )

    assert summary.input_tokens == 11
    assert summary.output_tokens == 3
    assert summary.total_latency_seconds == pytest.approx(0.2)
    assert summary.estimated_cost_usd == pytest.approx(0.01)
    assert (summary.cache_hits, summary.cache_misses, summary.cache_bypasses) == (0, 1, 0)


@pytest.mark.parametrize(
    ("field", "value"),
    [("input_tokens", -1), ("latency_seconds", -0.1), ("estimated_cost_usd", float("inf"))],
)
def test_answer_usage_aggregation_rejects_corrupt_values(con, field, value):
    question = _retrieval_question("rag-001")
    _seed_retrieval_rows(con, "eval-corrupt", [question])
    values = {
        "input_tokens": 1,
        "latency_seconds": 0.1,
        "estimated_cost_usd": 0.01,
    }
    values[field] = value
    _insert_answer_usage(
        con,
        usage_id="usage-corrupt",
        eval_run_id="eval-corrupt",
        question=question,
        cache_status="miss",
        outcome="ok",
        **values,
    )

    with pytest.raises(ValueError, match="usage"):
        llm_eval.run_answer_eval(
            con,
            [question],
            "embed-m",
            "eval-corrupt",
            answerer=_Answerer({question.question: _answer_result(question)}),
        )
