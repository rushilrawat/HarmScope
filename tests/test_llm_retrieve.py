"""Scoped, auditable evidence loading for hybrid retrieval."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from src.llm import retrieve


class FakeEncoder:
    def __init__(self, encoded: np.ndarray):
        self.encoded = encoded
        self.calls: list[list[str]] = []
        self.encode_kwargs: list[dict[str, object]] = []

    def encode(self, questions, **kwargs):
        self.calls.append(questions)
        self.encode_kwargs.append(kwargs)
        return self.encoded


def evidence(complaint_id: int, row_idx: int = 0) -> retrieve.EvidenceRecord:
    return retrieve.EvidenceRecord(
        complaint_id=complaint_id,
        cluster_id="scope-cluster",
        row_idx=row_idx,
        date_received=date(2020, 1, 1),
        company_id="scope-company",
        company_name="Scope Company",
        product_family="mortgage",
        text_redacted="redacted evidence",
        company_public_response=None,
    )


@pytest.fixture
def scoped_corpus():
    return retrieve.ScopedCorpus(
        cluster_id="scope-cluster",
        company_id="scope-company",
        embed_model="embed-m",
        rows=(evidence(complaint_id=1, row_idx=0),),
    )


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


def test_encode_query_is_unit_normalized():
    """Dense query embeddings are normalized in the corpus vector space."""
    encoder = FakeEncoder(np.array([[3.0, 4.0]], dtype=np.float32))

    got = retrieve.encode_query(encoder, "where did the refund go?")

    np.testing.assert_allclose(got, [[0.6, 0.8]], atol=1e-6)
    assert encoder.calls == [["where did the refund go?"]]
    assert encoder.encode_kwargs == [
        {
            "convert_to_numpy": True,
            "normalize_embeddings": False,
            "show_progress_bar": False,
        }
    ]


def test_encode_query_rejects_zero_norm_embedding():
    """A zero query vector cannot produce meaningful inner-product ranking."""
    encoder = FakeEncoder(np.array([[0.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="zero norm"):
        retrieve.encode_query(encoder, "where did the refund go?")


def test_dense_rank_uses_only_scoped_row_indices(scoped_corpus):
    """A nearest vector outside the corpus scope is never returned."""
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    corpus = replace(
        scoped_corpus,
        rows=(
            evidence(complaint_id=11, row_idx=2),
            evidence(complaint_id=12, row_idx=1),
        ),
    )

    got = retrieve.dense_rank(
        corpus, vectors, np.array([[1.0, 0.0]], dtype=np.float32), limit=10
    )

    assert [(hit.complaint_id, hit.rank) for hit in got] == [(11, 1), (12, 2)]
    assert {hit.complaint_id for hit in got} == {11, 12}


def test_dense_ties_break_on_complaint_id(scoped_corpus):
    """Equal dense scores resolve deterministically on complaint ID."""
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    corpus = replace(
        scoped_corpus,
        rows=(
            evidence(complaint_id=20, row_idx=0),
            evidence(complaint_id=10, row_idx=1),
        ),
    )

    got = retrieve.dense_rank(
        corpus, vectors, np.array([[1.0, 0.0]], dtype=np.float32), 2
    )

    assert [hit.complaint_id for hit in got] == [10, 20]


def test_dense_rank_rejects_out_of_bounds_scoped_row_index(scoped_corpus):
    """An invalid embedding row mapping fails instead of selecting another vector."""
    corpus = replace(scoped_corpus, rows=(evidence(complaint_id=1, row_idx=2),))
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="row_idx"):
        retrieve.dense_rank(
            corpus, vectors, np.array([[1.0, 0.0]], dtype=np.float32), limit=1
        )


def test_dense_rank_rejects_query_dimension_mismatch(scoped_corpus):
    """FAISS is never asked to rank a query from a different vector space."""
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="dimension"):
        retrieve.dense_rank(
            scoped_corpus,
            vectors,
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            limit=1,
        )


def test_dense_rank_rejects_zero_norm_scoped_embedding(scoped_corpus):
    """A zero corpus vector violates the normalized FAISS similarity contract."""
    vectors = np.array([[0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="zero norm"):
        retrieve.dense_rank(
            scoped_corpus, vectors, np.array([[1.0, 0.0]], dtype=np.float32), limit=1
        )


def test_dense_rank_rejects_zero_norm_query(scoped_corpus):
    """A zero query vector cannot produce meaningful inner-product ranking."""
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="zero norm"):
        retrieve.dense_rank(
            scoped_corpus, vectors, np.array([[0.0, 0.0]], dtype=np.float32), limit=1
        )
