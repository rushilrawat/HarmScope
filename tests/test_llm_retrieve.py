"""Scoped, auditable evidence loading for hybrid retrieval."""

from __future__ import annotations

from datetime import date

import pytest

from src.llm import retrieve


@pytest.fixture
def retrieval_fixture(con):
    wanted_cluster = "0000000000001-test:mortgage:1"
    other_cluster = "0000000000001-test:mortgage:2"
    wanted_company = "acme-bank"
    other_company = "other-bank"
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES (?, 'test', 'test', 'test', '{}', now(), 'ok')",
        ["0000000000001-test"],
    )
    for cluster_id in (wanted_cluster, other_cluster):
        con.execute(
            "INSERT INTO clusters "
            "(cluster_id, run_id, product_family, n_members, as_of) "
            "VALUES (?, '0000000000001-test', 'mortgage', 2, ?)",
            [cluster_id, date(2020, 1, 1)],
        )
    for company_id, company_name in (
        (wanted_company, "Acme Bank"),
        (other_company, "Other Bank"),
    ):
        con.execute(
            "INSERT INTO company_canonical "
            "(company_id, canonical_name, verified_by) VALUES (?, ?, 'manual')",
            [company_id, company_name],
        )

    records = (
        (20, wanted_cluster, wanted_company, "Acme Bank", "second redacted narrative", " "),
        (10, wanted_cluster, wanted_company, "Acme Bank", "first redacted narrative", "Company disputes the allegation."),
        (30, wanted_cluster, other_company, "Other Bank", "other-company redacted narrative", None),
        (40, other_cluster, wanted_company, "Acme Bank", "other-cluster redacted narrative", None),
    )
    for row_idx, (complaint_id, cluster_id, company_id, company_name, text, response) in enumerate(records):
        con.execute(
            "INSERT INTO complaints_raw "
            "(complaint_id, date_received, company_raw, company_public_response) "
            "VALUES (?, ?, ?, ?)",
            [complaint_id, date(2020, 1, complaint_id // 10), company_name, response],
        )
        con.execute(
            "INSERT INTO complaints "
            "(complaint_id, date_received, period_month, company_id, product_family, has_narrative) "
            "VALUES (?, ?, ?, ?, 'mortgage', true)",
            [
                complaint_id,
                date(2020, 1, complaint_id // 10),
                date(2020, 1, 1),
                company_id,
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
            [complaint_id, row_idx],
        )
        con.execute(
            "INSERT INTO cluster_members (cluster_id, complaint_id) VALUES (?, ?)",
            [cluster_id, complaint_id],
        )
    return con, wanted_cluster, wanted_company


def test_load_corpus_cannot_escape_cluster_or_company(retrieval_fixture):
    """Removing either scope predicate would leak unrelated complaint evidence."""
    con, wanted_cluster, wanted_company = retrieval_fixture

    corpus = retrieve.load_corpus(con, wanted_cluster, wanted_company, "embed-m")

    assert {row.cluster_id for row in corpus.rows} == {wanted_cluster}
    assert {row.company_id for row in corpus.rows} == {wanted_company}
    assert [row.complaint_id for row in corpus.rows] == [10, 20]


def test_load_corpus_carries_auditable_evidence_fields(retrieval_fixture):
    """Evidence exposes redacted narrative and metadata without mixing context into text."""
    con, cluster_id, company_id = retrieval_fixture

    row = retrieve.load_corpus(con, cluster_id, company_id, "embed-m").rows[0]

    assert row.complaint_id == 10
    assert row.date_received == date(2020, 1, 1)
    assert row.company_id == company_id
    assert row.company_name == "Acme Bank"
    assert row.product_family == "mortgage"
    assert row.text_redacted == "first redacted narrative"
    assert row.row_idx == 1
    assert row.company_public_response == "Company disputes the allegation."


def test_load_corpus_normalizes_blank_public_response_without_changing_text(retrieval_fixture):
    """Public response remains separate context and blank context becomes absent."""
    con, cluster_id, company_id = retrieval_fixture

    row = retrieve.load_corpus(con, cluster_id, company_id, "embed-m").rows[1]

    assert row.text_redacted == "second redacted narrative"
    assert row.company_public_response is None


def test_unknown_scope_fails_instead_of_searching_globally(retrieval_fixture):
    """An unknown scope must fail rather than silently broadening retrieval."""
    con, _, _ = retrieval_fixture

    with pytest.raises(ValueError, match="no evidence"):
        retrieve.load_corpus(con, "missing-cluster", None, "embed-m")


def test_membership_hash_is_deterministic_and_scope_sensitive(retrieval_fixture):
    """Changing retrieval scope changes the cache key while identical inputs are stable."""
    con, cluster_id, company_id = retrieval_fixture
    corpus = retrieve.load_corpus(con, cluster_id, company_id, "embed-m")

    first = retrieve.membership_hash(corpus, "word-v1")
    second = retrieve.membership_hash(corpus, "word-v1")
    all_companies = retrieve.load_corpus(con, cluster_id, None, "embed-m")

    assert first == second
    assert retrieve.membership_hash(all_companies, "word-v1") != first
    assert retrieve.membership_hash(corpus, "word-v2") != first
