"""Phase 8D benchmark manifest and private authoring-workflow contracts."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from src.llm import eval as llm_eval
from src.llm import retrieve


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
    fused_ids: tuple[int, ...] = (30, 10, 20),
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
        retrieve.FusedHit(
            complaint_id,
            1.0 / (60 + rank),
            next((hit.rank for hit in dense if hit.complaint_id == complaint_id), None),
            next((hit.score for hit in dense if hit.complaint_id == complaint_id), None),
            next((hit.rank for hit in sparse if hit.complaint_id == complaint_id), None),
            next((hit.score for hit in sparse if hit.complaint_id == complaint_id), None),
        )
        for rank, complaint_id in enumerate(fused_ids, start=1)
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


class _VariantRetriever:
    def __init__(self, results: dict[str, retrieve.RetrievalResult]):
        self.results = results
        self.calls: list[tuple[str, str | None, str, str]] = []

    def __call__(self, con, cluster_id, company_id, question, embed_model):
        del con
        self.calls.append((cluster_id, company_id, question, embed_model))
        return self.results[question]


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
    assert got["fused"].rank_first_relevant == 2


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
    assert summary.methods["fused"].reciprocal_rank == pytest.approx(0.5)
    assert summary.fused_vs_dense == llm_eval.WinTieLoss(wins=0, ties=1, losses=1)
    assert summary.fused_vs_bm25 == llm_eval.WinTieLoss(wins=0, ties=2, losses=0)


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
                fused_ids=(10, 20),
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


def test_retrieval_summary_reports_linear_p95_and_every_fusion_loss_without_narratives(con):
    questions: list[llm_eval.EvalQuestion] = []
    results: dict[str, retrieve.RetrievalResult] = {}
    for index in range(1, 21):
        question = _retrieval_question(
            f"rag-{index:03d}",
            question_text=f"private narrative phrase {index}",
        )
        questions.append(question)
        fused_ids = (20, 10) if index == 20 else (10, 20)
        results[question.question] = _retrieval_result(
            dense_ids=(10, 20),
            bm25_ids=(10, 20),
            fused_ids=fused_ids,
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
    assert summary.fused_vs_dense == llm_eval.WinTieLoss(wins=0, ties=19, losses=1)
    report = summary.render()
    assert report == summary.render()
    assert "answerable: 20" in report
    assert "unanswerable: 0" in report
    assert "Recall@10" in report
    assert "MRR" in report
    assert "fused vs dense win/tie/loss: 0/19/1" in report
    assert "fused vs BM25 win/tie/loss: 0/19/1" in report
    assert "rag-020" in report and "loss" in report
    assert "private narrative phrase" not in report
    assert all(math.isfinite(method.p95_latency_seconds) for method in summary.methods.values())
