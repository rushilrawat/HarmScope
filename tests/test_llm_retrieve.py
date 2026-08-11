"""Scoped, auditable evidence loading for hybrid retrieval."""

from __future__ import annotations

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

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
        embed_dim=2,
        embedding_rows=1,
    )


@pytest.fixture
def fake_encoder():
    return FakeEncoder(np.array([[1.0, 0.0]], dtype=np.float32))


@pytest.fixture
def vectors():
    return np.array(
        [
            [0.0, 1.0],
            [0.8, 0.6],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def retrieval_fixture(con):
    dedup_run = "0000000000000-dedup"
    cluster_run = "0000000000001-test"
    wanted_cluster = "0000000000001-test:mortgage:1"
    other_cluster = "0000000000001-test:mortgage:2"
    wanted_company = "acme-bank"
    other_company = "other-bank"
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
                        "model": "embed-m",
                        "dedup_run": dedup_run,
                        "cutoff": "",
                    }
                }
            ),
        ],
    )
    for cluster_id in (wanted_cluster, other_cluster):
        con.execute(
            "INSERT INTO clusters "
            "(cluster_id, run_id, product_family, n_members, as_of) "
            "VALUES (?, ?, 'mortgage', 2, ?)",
            [cluster_id, cluster_run, date(2020, 1, 1)],
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
        (
            10,
            wanted_cluster,
            wanted_company,
            "Acme Bank",
            "first redacted narrative",
            "Company disputes the allegation.",
        ),
        (
            30,
            wanted_cluster,
            other_company,
            "Other Bank",
            "escrow refund escrow refund strongest lexical match",
            None,
        ),
        (
            40,
            other_cluster,
            wanted_company,
            "Acme Bank",
            "escrow refund other-cluster narrative",
            None,
        ),
    )
    group_by_complaint = {10: ("group-10", True, 2), 30: ("group-10", False, 2)}
    for row_idx, (complaint_id, cluster_id, company_id, company_name, text, response) in enumerate(
        records
    ):
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
        group_id, is_representative, group_size = group_by_complaint.get(
            complaint_id, (f"group-{complaint_id}", True, 1)
        )
        con.execute(
            "INSERT INTO dup_groups "
            "(run_id, complaint_id, group_id, is_representative, group_size, as_of) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                dedup_run,
                complaint_id,
                group_id,
                is_representative,
                group_size,
                date(2020, 1, 31),
            ],
        )
        if is_representative:
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


def test_load_corpus_rejects_a_model_not_recorded_by_the_cluster_run(retrieval_fixture):
    """Caller-selected rows cannot silently change the cluster's vector space."""
    con, wanted_cluster, wanted_company = retrieval_fixture
    con.execute(
        "INSERT INTO embedding_map (complaint_id, row_idx, model, dim) "
        "VALUES (10, 99, 'competing-model', 2)"
    )

    requested = retrieve.load_corpus(con, wanted_cluster, wanted_company, "embed-m")
    assert [(row.complaint_id, row.row_idx) for row in requested.rows] == [
        (10, 1),
        (20, 0),
    ]
    with pytest.raises(ValueError, match="cluster run.*embed-m.*competing-model"):
        retrieve.load_corpus(con, wanted_cluster, wanted_company, "competing-model")


def test_load_corpus_expands_cross_company_duplicates_then_applies_company_scope(
    retrieval_fixture,
):
    """A sibling company in the representative's dup group remains retrievable."""
    con, cluster_id, _ = retrieval_fixture

    corpus = retrieve.load_corpus(con, cluster_id, "other-bank", "embed-m")

    assert [row.complaint_id for row in corpus.rows] == [30]
    assert corpus.rows[0].company_id == "other-bank"


def test_load_corpus_uses_the_cluster_runs_recorded_dedup_run(retrieval_fixture):
    """A newer incompatible grouping cannot rebind an existing cluster's population."""
    con, cluster_id, _ = retrieval_fixture
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES "
        "('0000000000002-dedup', 'dedup', 'test', 'test', '{}', now(), 'ok')"
    )
    con.executemany(
        "INSERT INTO dup_groups "
        "(run_id, complaint_id, group_id, is_representative, group_size, as_of) "
        "VALUES ('0000000000002-dedup', ?, ?, true, 1, DATE '2020-01-31')",
        [(10, "new-group-10"), (30, "new-group-30")],
    )

    corpus = retrieve.load_corpus(con, cluster_id, "other-bank", "embed-m")

    assert [row.complaint_id for row in corpus.rows] == [30]


def test_load_corpus_selects_one_deterministic_evidence_row_per_group_and_company(
    retrieval_fixture,
):
    """Duplicate filings cannot overweight a group/company evidence unit."""
    con, cluster_id, _ = retrieval_fixture
    con.execute(
        "INSERT INTO complaints_raw "
        "(complaint_id, date_received, company_raw, company_public_response) "
        "VALUES (31, DATE '2020-01-05', 'Other Bank', NULL)"
    )
    con.execute(
        "INSERT INTO complaints "
        "(complaint_id, date_received, period_month, company_id, product_family, has_narrative) "
        "VALUES (31, DATE '2020-01-05', DATE '2020-01-01', "
        "'other-bank', 'mortgage', true)"
    )
    con.execute(
        "INSERT INTO narratives (complaint_id, text_redacted, text_hash, redaction_count) "
        "VALUES (31, 'later duplicate', 'hash-31', 0)"
    )
    con.execute(
        "INSERT INTO embedding_map (complaint_id, row_idx, model, dim) VALUES (31, 4, 'embed-m', 2)"
    )
    con.execute(
        "INSERT INTO dup_groups "
        "(run_id, complaint_id, group_id, is_representative, group_size, as_of) "
        "VALUES ('0000000000000-dedup', 31, 'group-10', false, 3, DATE '2020-01-31')"
    )

    corpus = retrieve.load_corpus(con, cluster_id, "other-bank", "embed-m")

    assert [row.complaint_id for row in corpus.rows] == [30]


def test_load_corpus_excludes_flagged_campaign_groups(retrieval_fixture):
    """Evidence uses the same campaign-excluded population as signals."""
    con, cluster_id, _ = retrieval_fixture
    con.execute(
        "INSERT INTO campaigns "
        "(campaign_id, run_id, n_complaints, n_groups, product_family, "
        "n_signals, flagged, as_of) VALUES "
        "('0000000000000-dedup:campaign:1', '0000000000000-dedup', 2, 1, "
        "'mortgage', 3, true, DATE '2020-01-31')"
    )
    con.executemany(
        "INSERT INTO campaign_members (complaint_id, campaign_id) VALUES "
        "(?, '0000000000000-dedup:campaign:1')",
        [(10,), (30,)],
    )

    corpus = retrieve.load_corpus(con, cluster_id, None, "embed-m")

    assert [row.complaint_id for row in corpus.rows] == [20]


def test_load_corpus_carries_consistent_embedding_map_dimension(retrieval_fixture):
    """The vector dimension recorded by embedding_map reaches dense validation."""
    con, cluster_id, company_id = retrieval_fixture

    corpus = retrieve.load_corpus(con, cluster_id, company_id, "embed-m")

    assert corpus.embed_dim == 2
    assert corpus.embedding_rows == 4


def test_load_corpus_rejects_inconsistent_embedding_map_dimensions(retrieval_fixture):
    """Mixed dimensions for one model are corrupt provenance, not evidence."""
    con, cluster_id, company_id = retrieval_fixture
    con.execute("UPDATE embedding_map SET dim = 3 WHERE complaint_id = 20 AND model = 'embed-m'")

    with pytest.raises(ValueError, match="embedding_map.*dimension"):
        retrieve.load_corpus(con, cluster_id, company_id, "embed-m")


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

    got = retrieve.dense_rank(corpus, vectors, np.array([[1.0, 0.0]], dtype=np.float32), limit=10)

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

    got = retrieve.dense_rank(corpus, vectors, np.array([[1.0, 0.0]], dtype=np.float32), 2)

    assert [hit.complaint_id for hit in got] == [10, 20]


def test_dense_rank_normalizes_corpus_vectors_at_the_cosine_boundary(scoped_corpus):
    """Vector magnitude cannot outrank a more directionally similar complaint."""
    vectors = np.array([[1.0, 1.0], [0.9, 0.0]], dtype=np.float32)
    corpus = replace(
        scoped_corpus,
        rows=(
            evidence(complaint_id=10, row_idx=0),
            evidence(complaint_id=20, row_idx=1),
        ),
        embedding_rows=2,
    )

    got = retrieve.dense_rank(
        corpus,
        vectors,
        np.array([[2.0, 0.0]], dtype=np.float32),
        limit=2,
    )

    assert [hit.complaint_id for hit in got] == [20, 10]
    assert got[0].score == pytest.approx(1.0)


def test_dense_rank_rejects_out_of_bounds_scoped_row_index(scoped_corpus):
    """An invalid embedding row mapping fails instead of selecting another vector."""
    corpus = replace(scoped_corpus, rows=(evidence(complaint_id=1, row_idx=2),))
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="row_idx"):
        retrieve.dense_rank(corpus, vectors, np.array([[1.0, 0.0]], dtype=np.float32), limit=1)


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


@pytest.mark.parametrize(
    ("vectors", "query", "message"),
    [
        (
            np.array([[np.nan, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
            "embedding matrix",
        ),
        (
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[np.inf, 0.0]], dtype=np.float32),
            "query vector",
        ),
    ],
)
def test_dense_rank_rejects_non_finite_inputs(scoped_corpus, vectors, query, message):
    """NaN or infinity must fail before reaching FAISS ranking."""
    with pytest.raises(ValueError, match=message):
        retrieve.dense_rank(scoped_corpus, vectors, query, limit=1)


@pytest.mark.parametrize("limit", [0, -1])
def test_dense_rank_rejects_non_positive_limit(scoped_corpus, limit):
    """A non-positive dense candidate count is invalid configuration."""
    with pytest.raises(ValueError, match="limit must be positive"):
        retrieve.dense_rank(
            scoped_corpus,
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
            limit=limit,
        )


def test_tokenizer_is_lowercase_and_deterministic():
    """Changing token normalization would change lexical retrieval results."""
    assert retrieve.tokenize("APR fees, APR-refund!") == [
        "apr",
        "fees",
        "apr-refund",
    ]


def test_sparse_corpus_is_immutable():
    """Cached tokens must not be mutable after the corpus is loaded."""
    sparse = retrieve.SparseCorpus(
        complaint_ids=(10,),
        tokens=(("mortgage", "escrow"),),
    )

    with pytest.raises(AttributeError):
        sparse.complaint_ids = (20,)


def test_bm25_finds_exact_product_language():
    """Dropping lexical scoring would miss the document with both exact terms."""
    sparse = retrieve.SparseCorpus(
        complaint_ids=(10, 20, 30),
        tokens=(
            ("mortgage", "escrow", "shortage"),
            ("credit", "report", "dispute"),
            ("generic", "customer", "service"),
        ),
    )

    got = retrieve.bm25_rank(sparse, "escrow shortage", limit=3)

    assert got[0].complaint_id == 10
    assert got[0].rank == 1


def test_bm25_ties_break_on_complaint_id():
    """Equal lexical scores have a stable complaint-ID ordering."""
    sparse = retrieve.SparseCorpus(
        complaint_ids=(20, 10),
        tokens=(("escrow",), ("escrow",)),
    )

    got = retrieve.bm25_rank(sparse, "escrow", limit=2)

    assert [(hit.complaint_id, hit.rank) for hit in got] == [(10, 1), (20, 2)]


def test_bm25_empty_query_returns_zero_score_ids_in_order():
    """Punctuation-only questions have deterministic, score-free candidates."""
    sparse = retrieve.SparseCorpus(
        complaint_ids=(20, 10, 30),
        tokens=(("escrow",), ("mortgage",), ("refund",)),
    )

    got = retrieve.bm25_rank(sparse, "?!", limit=2)

    assert [(hit.complaint_id, hit.rank, hit.score) for hit in got] == [
        (10, 1, 0.0),
        (20, 2, 0.0),
    ]


def test_bm25_all_empty_documents_return_zero_score_ids_in_order():
    """Fully redacted documents cannot make BM25 divide by zero."""
    sparse = retrieve.SparseCorpus(
        complaint_ids=(20, 10),
        tokens=((), ()),
    )

    got = retrieve.bm25_rank(sparse, "refund", limit=2)

    assert [(hit.complaint_id, hit.rank, hit.score) for hit in got] == [
        (10, 1, 0.0),
        (20, 2, 0.0),
    ]


def test_sparse_cache_invalidates_on_membership_or_tokenizer(scoped_corpus, tmp_path):
    """Scope membership and tokenizer changes must rebuild sparse token data."""
    first, hit1 = retrieve.load_sparse_corpus(scoped_corpus, tmp_path, "word-v1")
    second, hit2 = retrieve.load_sparse_corpus(scoped_corpus, tmp_path, "word-v1")
    changed = replace(scoped_corpus, rows=scoped_corpus.rows + (evidence(complaint_id=999),))
    _, hit3 = retrieve.load_sparse_corpus(changed, tmp_path, "word-v1")
    _, hit4 = retrieve.load_sparse_corpus(scoped_corpus, tmp_path, "word-v2")

    assert first == second
    assert (hit1, hit2, hit3, hit4) == (False, True, False, False)


def test_sparse_cache_is_json_only_and_written_atomically(scoped_corpus, tmp_path):
    """Sparse caches persist only the validated, replayable token payload."""
    sparse, cache_hit = retrieve.load_sparse_corpus(scoped_corpus, tmp_path, "word-v1")
    key = retrieve.membership_hash(scoped_corpus, "word-v1")
    path = tmp_path / f"bm25.{key}.json"

    assert cache_hit is False
    assert sparse == retrieve.SparseCorpus(complaint_ids=(1,), tokens=(("redacted", "evidence"),))
    assert json.loads(path.read_text()) == {
        "complaint_ids": [1],
        "tokenizer_version": "word-v1",
        "tokens": [["redacted", "evidence"]],
    }
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        json.dumps({"tokenizer_version": "word-v1", "complaint_ids": [1]}),
        json.dumps(
            {
                "tokenizer_version": "word-v2",
                "complaint_ids": [1],
                "tokens": [["redacted", "evidence"]],
            }
        ),
        json.dumps(
            {
                "tokenizer_version": "word-v1",
                "complaint_ids": [999],
                "tokens": [["redacted", "evidence"]],
            }
        ),
        json.dumps(
            {
                "tokenizer_version": "word-v1",
                "complaint_ids": [1],
                "tokens": [["redacted", 1]],
            }
        ),
    ],
)
def test_malformed_sparse_cache_is_quarantined(scoped_corpus, tmp_path, payload):
    """Bad cache JSON cannot be trusted and is moved aside before rebuilding."""
    key = retrieve.membership_hash(scoped_corpus, "word-v1")
    path = tmp_path / f"bm25.{key}.json"
    path.write_text(payload)

    sparse, cache_hit = retrieve.load_sparse_corpus(scoped_corpus, tmp_path, "word-v1")

    assert cache_hit is False
    assert sparse.complaint_ids == (1,)
    assert path.exists()
    assert len(list(tmp_path.glob(f"{path.name}.corrupt-*"))) == 1


def test_concurrent_malformed_sparse_cache_quarantine_is_race_safe(
    scoped_corpus, tmp_path, monkeypatch
):
    """Two readers observing one corrupt cache cannot race into FileNotFoundError."""
    key = retrieve.membership_hash(scoped_corpus, "word-v1")
    path = tmp_path / f"bm25.{key}.json"
    path.write_text("{")
    barrier = threading.Barrier(2)
    quarantine = retrieve._quarantine_sparse_cache

    def synchronized_quarantine(cache_path):
        barrier.wait(timeout=5)
        quarantine(cache_path)

    monkeypatch.setattr(retrieve, "_quarantine_sparse_cache", synchronized_quarantine)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                retrieve.load_sparse_corpus,
                scoped_corpus,
                tmp_path,
                "word-v1",
            )
            for _ in range(2)
        ]
        results = [future.result(timeout=5) for future in futures]

    assert all(sparse.complaint_ids == (1,) for sparse, _ in results)
    assert path.exists()


def test_rrf_combines_component_ranks_and_breaks_ties_by_id():
    """Equal reciprocal-rank totals resolve deterministically on source ID."""
    dense = [retrieve.RankedHit(10, 1, 0.9), retrieve.RankedHit(20, 2, 0.8)]
    sparse = [retrieve.RankedHit(20, 1, 4.0), retrieve.RankedHit(10, 2, 3.0)]

    got = retrieve.reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=10)

    assert [hit.complaint_id for hit in got] == [10, 20]
    assert got[0].fused_score == pytest.approx(got[1].fused_score)
    assert got[0].dense_rank == 1
    assert got[0].sparse_rank == 2
    assert got[1].dense_score == pytest.approx(0.8)
    assert got[1].sparse_score == pytest.approx(4.0)
    assert got[0].fused_score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_preserves_disjoint_component_maps_and_uses_configured_constant():
    """One-channel candidates keep missing fields and the supplied RRF constant."""
    dense = [retrieve.RankedHit(10, 1, 0.9)]
    sparse = [retrieve.RankedHit(20, 1, 4.0)]

    got = retrieve.reciprocal_rank_fusion(dense, sparse, rrf_k=7, top_k=10)

    assert got == [
        retrieve.FusedHit(10, 1 / 8, 1, 0.9, None, None),
        retrieve.FusedHit(20, 1 / 8, None, None, 1, 4.0),
    ]


@pytest.mark.parametrize(("rrf_k", "top_k"), [(0, 10), (-1, 10), (60, 0), (60, -1)])
def test_rrf_rejects_non_positive_configuration(rrf_k, top_k):
    """Invalid fusion constants fail instead of producing meaningless ranks."""
    with pytest.raises(ValueError, match="positive"):
        retrieve.reciprocal_rank_fusion([], [], rrf_k=rrf_k, top_k=top_k)


@pytest.mark.parametrize(
    "hits",
    [
        [retrieve.RankedHit(10, 0, 1.0)],
        [retrieve.RankedHit(10, 1, math.nan)],
        [retrieve.RankedHit(10, 1, 1.0), retrieve.RankedHit(10, 2, 0.5)],
        [retrieve.RankedHit(10, 1, 1.0), retrieve.RankedHit(20, 1, 0.5)],
    ],
)
def test_rrf_rejects_malformed_component_rankings(hits):
    """Fusion refuses invalid ranks, scores, duplicate IDs, and duplicate ranks."""
    with pytest.raises(ValueError, match="rank|score|duplicate"):
        retrieve.reciprocal_rank_fusion(hits, [], rrf_k=60, top_k=10)


def test_retrieval_result_types_are_immutable():
    """Evaluation inputs cannot drift after retrieval has been measured."""
    fused = retrieve.FusedHit(10, 0.1, 1, 0.9, None, None)
    evidence_row = retrieve.RetrievedEvidence(
        complaint_id=10,
        cluster_id="scope-cluster",
        date_received=date(2020, 1, 1),
        company_id="acme-bank",
        company_name="Acme Bank",
        product_family="mortgage",
        text_redacted="evidence",
        company_public_response=None,
        dense_rank=1,
        dense_score=0.9,
        sparse_rank=None,
        sparse_score=None,
        fused_score=0.1,
    )
    result = retrieve.RetrievalResult(
        corpus=retrieve.ScopedCorpus("c", None, "m", ()),
        dense=(),
        sparse=(),
        fused=(fused,),
        evidence=(evidence_row,),
        dense_seconds=0.0,
        sparse_seconds=0.0,
        fusion_seconds=0.0,
    )

    with pytest.raises(FrozenInstanceError):
        fused.fused_score = 0.2
    with pytest.raises(FrozenInstanceError):
        evidence_row.text_redacted = "changed"
    with pytest.raises(FrozenInstanceError):
        result.evidence = ()


def test_retrieve_variants_is_scoped_auditable_cached_and_observable(
    retrieval_fixture, fake_encoder, vectors, tmp_path
):
    """The full hybrid path exposes every stage without leaking top external matches."""
    con, cluster_id, company_id = retrieval_fixture

    first = retrieve.retrieve_variants(
        con,
        cluster_id,
        company_id,
        "escrow refund",
        "embed-m",
        encoder=fake_encoder,
        vectors=vectors,
        cache_dir=tmp_path,
        top_k=10,
    )
    cache_path = next(tmp_path.glob("bm25.*.json"))
    first_cache_identity = (cache_path.stat().st_ino, cache_path.stat().st_mtime_ns)
    second = retrieve.retrieve_variants(
        con,
        cluster_id,
        company_id,
        "escrow refund",
        "embed-m",
        encoder=fake_encoder,
        vectors=vectors,
        cache_dir=tmp_path,
        top_k=10,
    )

    assert [hit.complaint_id for hit in first.dense] == [10, 20]
    assert [hit.complaint_id for hit in first.sparse] == [10, 20]
    assert [hit.complaint_id for hit in first.fused] == [10, 20]
    assert [row.complaint_id for row in first.evidence] == [10, 20]
    assert {row.cluster_id for row in first.evidence} == {cluster_id}
    assert {30, 40}.isdisjoint(hit.complaint_id for hit in first.dense)
    assert {30, 40}.isdisjoint(hit.complaint_id for hit in first.sparse)
    assert {30, 40}.isdisjoint(hit.complaint_id for hit in first.fused)
    assert [(row.complaint_id, row.date_received) for row in first.evidence] == [
        (10, date(2020, 1, 1)),
        (20, date(2020, 1, 2)),
    ]
    assert all(row.text_redacted for row in first.evidence)
    assert all(row.dense_rank is not None or row.sparse_rank is not None for row in first.evidence)
    assert all(
        math.isfinite(seconds) and seconds >= 0.0
        for seconds in (first.dense_seconds, first.sparse_seconds, first.fusion_seconds)
    )
    assert second.dense == first.dense
    assert second.sparse == first.sparse
    assert second.fused == first.fused
    assert second.evidence == first.evidence
    assert (cache_path.stat().st_ino, cache_path.stat().st_mtime_ns) == first_cache_identity


def _configure_provider_artifact(
    con,
    cluster_id: str,
    tmp_path: Path,
    model: str = "provider/embed-m",
    *,
    metadata_model: str | None = None,
    n_done: int = 4,
    n_total: int = 4,
    dim: int = 2,
):
    from src.embed import encode

    cluster_run = con.execute(
        "SELECT run_id FROM clusters WHERE cluster_id = ?", [cluster_id]
    ).fetchone()[0]
    con.execute(
        "UPDATE runs SET params_json = ? WHERE run_id = ?",
        [
            json.dumps(
                {
                    "params": {
                        "model": model,
                        "dedup_run": "0000000000000-dedup",
                        "cutoff": "",
                    }
                }
            ),
            cluster_run,
        ],
    )
    con.execute(
        "INSERT INTO embedding_map "
        "SELECT complaint_id, row_idx, ?, dim FROM embedding_map WHERE model = 'embed-m'",
        [model],
    )
    artifacts = tmp_path / "artifacts"
    artifact = encode.embedding_artifact_paths(artifacts, model).memmap
    artifact.parent.mkdir(parents=True, exist_ok=True)
    encode.Progress(
        artifact.with_suffix(".progress.json"),
        n_done=n_done,
        n_total=n_total,
        dim=dim,
        model=model if metadata_model is None else metadata_model,
    ).write()
    return artifact


def test_retrieve_variants_uses_default_embedding_artifact_contract(
    retrieval_fixture, fake_encoder, vectors, tmp_path, monkeypatch
):
    """Default vectors use full-model addressing and validated encode metadata."""
    con, cluster_id, company_id = retrieval_fixture
    loaded_models: list[str] = []
    loaded_vectors: list[tuple[Path, str | None]] = []
    artifact = _configure_provider_artifact(con, cluster_id, tmp_path)
    monkeypatch.setattr(
        retrieve,
        "PATHS",
        SimpleNamespace(artifacts=tmp_path / "artifacts", llm_cache=tmp_path / "cache"),
    )

    def fake_load_model(model_name):
        loaded_models.append(model_name)
        return fake_encoder, "cpu"

    def fake_load(path, mmap_mode=None):
        loaded_vectors.append((Path(path), mmap_mode))
        return vectors

    from src.embed import encode

    monkeypatch.setattr(encode, "load_model", fake_load_model)
    monkeypatch.setattr(retrieve.np, "load", fake_load)

    got = retrieve.retrieve_variants(
        con,
        cluster_id,
        company_id,
        "escrow refund",
        "provider/embed-m",
        cache_dir=tmp_path,
        top_k=1,
    )

    assert len(got.evidence) == 1
    assert loaded_models == ["provider/embed-m"]
    assert loaded_vectors == [(artifact, "r")]
    assert fake_encoder.calls == [["escrow refund"]]


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"metadata_model": "other/embed-m"}, "model"),
        ({"n_done": 3}, "complete"),
        ({"n_done": 3, "n_total": 3}, "row count"),
        ({"dim": 3}, "dimension"),
    ],
)
def test_default_embedding_artifact_rejects_metadata_mismatch(
    retrieval_fixture, fake_encoder, vectors, tmp_path, monkeypatch, metadata, message
):
    """Sidecar provenance must match the cluster-bound map before vectors load."""
    con, cluster_id, company_id = retrieval_fixture
    _configure_provider_artifact(con, cluster_id, tmp_path, **metadata)
    monkeypatch.setattr(
        retrieve,
        "PATHS",
        SimpleNamespace(artifacts=tmp_path / "artifacts", llm_cache=tmp_path / "cache"),
    )
    monkeypatch.setattr(retrieve.np, "load", lambda *_args, **_kwargs: vectors)

    with pytest.raises(ValueError, match=message):
        retrieve.retrieve_variants(
            con,
            cluster_id,
            company_id,
            "escrow refund",
            "provider/embed-m",
            encoder=fake_encoder,
            cache_dir=tmp_path / "cache",
            top_k=1,
        )


def test_default_embedding_artifact_rejects_array_dimension_mismatch(
    retrieval_fixture, fake_encoder, vectors, tmp_path, monkeypatch
):
    """A correctly named artifact with the wrong array shape cannot be searched."""
    con, cluster_id, company_id = retrieval_fixture
    _configure_provider_artifact(con, cluster_id, tmp_path)
    monkeypatch.setattr(
        retrieve,
        "PATHS",
        SimpleNamespace(artifacts=tmp_path / "artifacts", llm_cache=tmp_path / "cache"),
    )
    monkeypatch.setattr(retrieve.np, "load", lambda *_args, **_kwargs: vectors[:, :1])

    with pytest.raises(ValueError, match="artifact.*dimension"):
        retrieve.retrieve_variants(
            con,
            cluster_id,
            company_id,
            "escrow refund",
            "provider/embed-m",
            encoder=fake_encoder,
            cache_dir=tmp_path / "cache",
            top_k=1,
        )


def test_retrieve_variants_honors_configured_candidate_and_default_top_k(
    retrieval_fixture, fake_encoder, vectors, tmp_path, monkeypatch
):
    """Candidate stages and default fusion output use their distinct config limits."""
    con, cluster_id, company_id = retrieval_fixture
    configured_llm = replace(
        retrieve.CONFIG.llm,
        rag_candidate_k=1,
        rag_top_k=1,
    )
    monkeypatch.setattr(retrieve, "CONFIG", SimpleNamespace(llm=configured_llm))

    got = retrieve.retrieve_variants(
        con,
        cluster_id,
        company_id,
        "escrow refund",
        "embed-m",
        encoder=fake_encoder,
        vectors=vectors,
        cache_dir=tmp_path,
    )

    assert len(got.dense) == 1
    assert len(got.sparse) == 1
    assert len(got.fused) == 1
    assert len(got.evidence) == 1


def test_retrieve_variants_uses_configured_rrf_tokenizer_and_default_cache_path(
    retrieval_fixture, fake_encoder, vectors, tmp_path, monkeypatch
):
    """Orchestration passes measured config and defaults cache writes to PATHS."""
    con, cluster_id, company_id = retrieval_fixture
    configured_llm = replace(
        retrieve.CONFIG.llm,
        rrf_k=7,
        bm25_tokenizer_version="word-v99",
    )
    default_cache = tmp_path / "default-cache"
    monkeypatch.setattr(retrieve, "CONFIG", SimpleNamespace(llm=configured_llm))
    monkeypatch.setattr(
        retrieve,
        "PATHS",
        SimpleNamespace(artifacts=tmp_path / "artifacts", llm_cache=default_cache),
    )

    got = retrieve.retrieve_variants(
        con,
        cluster_id,
        company_id,
        "escrow refund",
        "embed-m",
        encoder=fake_encoder,
        vectors=vectors,
        top_k=1,
    )

    assert got.fused[0].fused_score == pytest.approx(2 / 8)
    cache_key = retrieve.membership_hash(got.corpus, "word-v99")
    assert (default_cache / f"bm25.{cache_key}.json").exists()


def test_retrieve_evidence_is_a_thin_list_wrapper(
    retrieval_fixture, fake_encoder, vectors, tmp_path
):
    """Ordinary callers get a list without a second query encoding or generation call."""
    con, cluster_id, company_id = retrieval_fixture

    got = retrieve.retrieve_evidence(
        con,
        cluster_id,
        company_id,
        "escrow refund",
        "embed-m",
        encoder=fake_encoder,
        vectors=vectors,
        cache_dir=tmp_path,
        top_k=1,
    )

    assert isinstance(got, list)
    assert [row.complaint_id for row in got] == [10]
    assert fake_encoder.calls == [["escrow refund"]]
