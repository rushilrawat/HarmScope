# Phase 8B Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox ( - [ ] ) syntax for tracking.

**Goal:** Retrieve the ten best complaint records supporting an analyst's question by combining exact dense similarity and lexical BM25 inside one cluster/company scope.

**Architecture:** A scoped corpus loader is the only database boundary. Dense and sparse rankers consume the same immutable evidence records and remain independently measurable. Reciprocal-rank fusion combines ranks deterministically, while a membership-hash cache avoids repeated BM25 tokenization.

**Tech Stack:** Python 3.11+, DuckDB 1.5.5, NumPy 2.3.4, FAISS CPU 1.13.0, sentence-transformers 5.1.2, rank-bm25 0.2.2, pytest

## Global Constraints

- Retrieval is always restricted to one cluster and, when supplied, one company.
- Retrieval never searches the full complaint corpus after the signal scope is known.
- Only narratives.text_redacted is indexed or returned as complaint evidence.
- Dense retrieval uses the same embedding model and vector space recorded for the cluster run.
- Exact FAISS IndexFlatIP is used over unit-normalized vectors; no approximate search parameters are introduced.
- BM25 tokenization is deterministic, lowercased, versioned, and cached by cluster membership hash.
- Fusion uses CONFIG.llm.rrf_k and breaks every tie on complaint_id.
- Every returned evidence item contains complaint_id, date, company, product family, component ranks/scores, fused score, and public response when available.
- Retrieval returns evidence only; it does not generate a conclusion.

## File map

- Modify src/config.py — adds candidate-pool and tokenizer-version parameters.
- Create src/llm/retrieve.py — scoped loading, dense rank, BM25 rank, RRF, cache, and orchestration.
- Create tests/test_llm_retrieve.py — unit and integration tests for all retrieval stages.
- Modify tests/test_config.py — pins measured retrieval parameters in the config fingerprint.
- Modify tests/test_llm.py — prevents retrieval code from leaking into detection packages.

---

### Task 1: Define the scoped evidence corpus

**Files:**
- Modify: src/config.py in LLMConfig
- Modify: tests/test_config.py
- Create: src/llm/retrieve.py
- Create: tests/test_llm_retrieve.py

**Interfaces:**
- Consumes: cluster_members, embedding_map, narratives, complaints, complaints_raw, company_canonical
- Produces:
  - EvidenceRecord
  - ScopedCorpus
  - load_corpus(con, cluster_id: str, company_id: str | None, embed_model: str) -> ScopedCorpus
  - membership_hash(corpus: ScopedCorpus, tokenizer_version: str) -> str

- [ ] **Step 1: Write failing scope and field-completeness tests**

~~~python
def test_load_corpus_cannot_escape_cluster_or_company(retrieval_fixture):
    con, wanted_cluster, wanted_company = retrieval_fixture
    corpus = retrieve.load_corpus(con, wanted_cluster, wanted_company, "embed-m")
    assert {row.cluster_id for row in corpus.rows} == {wanted_cluster}
    assert {row.company_id for row in corpus.rows} == {wanted_company}
    assert [row.complaint_id for row in corpus.rows] == sorted(
        row.complaint_id for row in corpus.rows
    )


def test_load_corpus_carries_auditable_evidence_fields(retrieval_fixture):
    con, cluster_id, company_id = retrieval_fixture
    row = retrieve.load_corpus(con, cluster_id, company_id, "embed-m").rows[0]
    assert row.complaint_id
    assert row.date_received
    assert row.company_id == company_id
    assert row.company_name
    assert row.product_family
    assert row.text_redacted
    assert row.row_idx >= 0
    assert row.company_public_response == "Company disputes the allegation."


def test_unknown_scope_fails_instead_of_searching_globally(retrieval_fixture):
    con, _, _ = retrieval_fixture
    with pytest.raises(ValueError, match="no evidence"):
        retrieve.load_corpus(con, "missing-cluster", None, "embed-m")
~~~

- [ ] **Step 2: Run focused tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_retrieve.py -k "load_corpus or unknown_scope" -v

Expected: FAIL because src.llm.retrieve does not exist.

- [ ] **Step 3: Add explicit candidate and tokenizer config**

~~~python
rag_candidate_k: int = 50
bm25_tokenizer_version: str = "word-v1"
~~~

Assert rag_candidate_k >= rag_top_k in LLMConfig.__post_init__. Preserve any existing post-init validation by adding to it rather than replacing it.

- [ ] **Step 4: Define immutable corpus types**

~~~python
@dataclass(frozen=True)
class EvidenceRecord:
    complaint_id: int
    cluster_id: str
    row_idx: int
    date_received: date
    company_id: str
    company_name: str
    product_family: str
    text_redacted: str
    company_public_response: str | None


@dataclass(frozen=True)
class ScopedCorpus:
    cluster_id: str
    company_id: str | None
    embed_model: str
    rows: tuple[EvidenceRecord, ...]
~~~

- [ ] **Step 5: Implement one parameterized scoped query**

The query starts from cluster_members m WHERE m.cluster_id = ?. Join narratives, complaints, embedding_map filtered by model, optional company_canonical, and complaints_raw for company_public_response. Add AND c.company_id = ? only when company_id is supplied. Order by complaint_id. Raise ValueError when no rows are returned. Do not accept a fallback scope.

Normalize blank public responses to None. Public responses are context and remain a separate field; never concatenate them into text_redacted.

- [ ] **Step 6: Implement the deterministic membership hash**

~~~python
def membership_hash(corpus: ScopedCorpus, tokenizer_version: str) -> str:
    payload = {
        "cluster_id": corpus.cluster_id,
        "company_id": corpus.company_id,
        "embed_model": corpus.embed_model,
        "tokenizer_version": tokenizer_version,
        "complaint_ids": [row.complaint_id for row in corpus.rows],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
~~~

- [ ] **Step 7: Run focused tests and commit**

Run: .venv/bin/python -m pytest tests/test_llm_retrieve.py -k "load_corpus or unknown_scope or membership_hash" -v

Expected: PASS.

~~~bash
git add src/config.py src/llm/retrieve.py tests/test_config.py tests/test_llm_retrieve.py
git commit -m "Add scoped RAG evidence corpus"
~~~

---

### Task 2: Implement exact dense retrieval

**Files:**
- Modify: src/llm/retrieve.py
- Modify: tests/test_llm_retrieve.py

**Interfaces:**
- Consumes: ScopedCorpus and the existing normalized embedding memmap
- Produces:
  - RankedHit(complaint_id: int, rank: int, score: float)
  - encode_query(encoder, question: str) -> np.ndarray
  - dense_rank(corpus: ScopedCorpus, vectors: np.ndarray, query_vector: np.ndarray, limit: int) -> list[RankedHit]

- [ ] **Step 1: Write failing exact-ranking and encoder-normalization tests**

~~~python
def test_encode_query_is_unit_normalized():
    encoder = FakeEncoder(np.array([[3.0, 4.0]], dtype=np.float32))
    got = retrieve.encode_query(encoder, "where did the refund go?")
    np.testing.assert_allclose(got, [[0.6, 0.8]], atol=1e-6)
    assert encoder.calls == [["where did the refund go?"]]


def test_dense_rank_uses_only_scoped_row_indices(scoped_corpus):
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.6],
        [-1.0, 0.0],
    ], dtype=np.float32)
    corpus = replace(scoped_corpus, rows=(
        evidence(complaint_id=11, row_idx=2),
        evidence(complaint_id=12, row_idx=1),
    ))
    got = retrieve.dense_rank(
        corpus, vectors, np.array([[1.0, 0.0]], dtype=np.float32), limit=10
    )
    assert [(hit.complaint_id, hit.rank) for hit in got] == [(11, 1), (12, 2)]
    assert {hit.complaint_id for hit in got} == {11, 12}


def test_dense_ties_break_on_complaint_id(scoped_corpus):
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    corpus = replace(scoped_corpus, rows=(
        evidence(complaint_id=20, row_idx=0),
        evidence(complaint_id=10, row_idx=1),
    ))
    got = retrieve.dense_rank(
        corpus, vectors, np.array([[1.0, 0.0]], dtype=np.float32), 2
    )
    assert [hit.complaint_id for hit in got] == [10, 20]
~~~

- [ ] **Step 2: Run dense tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_retrieve.py -k "encode_query or dense" -v

Expected: FAIL because the ranker does not exist.

- [ ] **Step 3: Implement query encoding**

Call encoder.encode([question], convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False). Convert to contiguous float32 shape (1, dim), reject zero norm, and normalize after encoding.

- [ ] **Step 4: Implement exact scoped FAISS search**

~~~python
@dataclass(frozen=True)
class RankedHit:
    complaint_id: int
    rank: int
    score: float


def dense_rank(corpus, vectors, query_vector, limit):
    import faiss

    matrix = np.ascontiguousarray(
        np.asarray([vectors[row.row_idx] for row in corpus.rows], dtype=np.float32)
    )
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    scores, positions = index.search(
        np.ascontiguousarray(query_vector, dtype=np.float32),
        len(corpus.rows),
    )
    pairs = [
        (corpus.rows[int(pos)].complaint_id, float(score))
        for pos, score in zip(positions[0], scores[0], strict=True)
        if pos >= 0
    ]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    ranked = [
        RankedHit(complaint_id=complaint_id, rank=rank, score=score)
        for rank, (complaint_id, score) in enumerate(pairs, start=1)
    ]
    return ranked[:limit]
~~~

- [ ] **Step 5: Run dense tests and commit**

Run: .venv/bin/python -m pytest tests/test_llm_retrieve.py -k "encode_query or dense" -v

Expected: PASS.

~~~bash
git add src/llm/retrieve.py tests/test_llm_retrieve.py
git commit -m "Add exact dense evidence retrieval"
~~~

---

### Task 3: Implement versioned BM25 retrieval and cache

**Files:**
- Modify: src/llm/retrieve.py
- Modify: tests/test_llm_retrieve.py

**Interfaces:**
- Consumes: ScopedCorpus, PATHS.artifacts, CONFIG.llm.bm25_tokenizer_version
- Produces:
  - tokenize(text: str) -> list[str]
  - SparseCorpus(complaint_ids: tuple[int, ...], tokens: tuple[tuple[str, ...], ...])
  - load_sparse_corpus(corpus: ScopedCorpus, cache_dir: Path, tokenizer_version: str) -> tuple[SparseCorpus, bool]
  - bm25_rank(sparse: SparseCorpus, question: str, limit: int) -> list[RankedHit]

- [ ] **Step 1: Write failing tokenization, lexical-ranking, and cache tests**

~~~python
def test_tokenizer_is_lowercase_and_deterministic():
    assert retrieve.tokenize("APR fees, APR-refund!") == [
        "apr", "fees", "apr-refund"
    ]


def test_bm25_finds_exact_product_language():
    sparse = SparseCorpus(
        complaint_ids=(10, 20, 30),
        tokens=(
            ("mortgage", "escrow", "shortage"),
            ("credit", "report", "dispute"),
            ("generic", "customer", "service"),
        ),
    )
    got = retrieve.bm25_rank(sparse, "escrow shortage", limit=3)
    assert got[0].complaint_id == 10


def test_sparse_cache_invalidates_on_membership_or_tokenizer(scoped_corpus, tmp_path):
    first, hit1 = retrieve.load_sparse_corpus(scoped_corpus, tmp_path, "word-v1")
    second, hit2 = retrieve.load_sparse_corpus(scoped_corpus, tmp_path, "word-v1")
    changed = replace(
        scoped_corpus, rows=scoped_corpus.rows + (evidence(complaint_id=999),)
    )
    _, hit3 = retrieve.load_sparse_corpus(changed, tmp_path, "word-v1")
    _, hit4 = retrieve.load_sparse_corpus(scoped_corpus, tmp_path, "word-v2")
    assert first == second
    assert (hit1, hit2, hit3, hit4) == (False, True, False, False)
~~~

- [ ] **Step 2: Run sparse tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_retrieve.py -k "tokenizer or bm25 or sparse_cache" -v

Expected: FAIL because sparse retrieval does not exist.

- [ ] **Step 3: Implement the tokenizer and safe JSON cache**

Use re.findall with pattern [a-z0-9]+(?:[-'][a-z0-9]+)* over text.lower(). Cache only complaint IDs and token lists as JSON; do not use pickle. Write atomically with NamedTemporaryFile plus os.replace. The filename is bm25.<membership_hash>.json.

Validate a cache hit by checking version, exact complaint ID order, and that every token is a string. Quarantine malformed cache files using the same corrupt-suffix convention as label caching.

- [ ] **Step 4: Implement BM25 ranking with deterministic ties**

Build rank_bm25.BM25Okapi from the cached token lists. Score tokenize(question). Sort (-score, complaint_id), enumerate ranks from one, and return at most limit hits. If the query token list is empty, return complaint IDs in ascending order with zero scores; later answer logic will abstain if no evidence is relevant.

- [ ] **Step 5: Run sparse tests and commit**

Run: .venv/bin/python -m pytest tests/test_llm_retrieve.py -k "tokenizer or bm25 or sparse_cache" -v

Expected: PASS.

~~~bash
git add src/llm/retrieve.py tests/test_llm_retrieve.py
git commit -m "Add cached BM25 evidence retrieval"
~~~

---

### Task 4: Fuse ranks and expose one retrieval function

**Files:**
- Modify: src/llm/retrieve.py
- Modify: tests/test_llm_retrieve.py
- Modify: tests/test_llm.py

**Interfaces:**
- Consumes: dense_rank, bm25_rank, ScopedCorpus, CONFIG.llm.rrf_k
- Produces:
  - RetrievedEvidence
  - RetrievalResult(corpus: ScopedCorpus, dense: tuple[RankedHit, ...], sparse: tuple[RankedHit, ...], fused: tuple[FusedHit, ...], evidence: tuple[RetrievedEvidence, ...], dense_seconds: float, sparse_seconds: float, fusion_seconds: float)
  - reciprocal_rank_fusion(dense: list[RankedHit], sparse: list[RankedHit], rrf_k: int, top_k: int) -> list[FusedHit]
  - retrieve_variants(con, cluster_id: str, company_id: str | None, question: str, embed_model: str, encoder=None, vectors=None, cache_dir=None, top_k=None) -> RetrievalResult
  - retrieve_evidence(con, cluster_id: str, company_id: str | None, question: str, embed_model: str, encoder=None, vectors=None, cache_dir=None, top_k=None) -> list[RetrievedEvidence]

- [ ] **Step 1: Write failing RRF and end-to-end tests**

~~~python
def test_rrf_combines_component_ranks_and_breaks_ties_by_id():
    dense = [RankedHit(10, 1, 0.9), RankedHit(20, 2, 0.8)]
    sparse = [RankedHit(20, 1, 4.0), RankedHit(10, 2, 3.0)]
    got = retrieve.reciprocal_rank_fusion(dense, sparse, rrf_k=60, top_k=10)
    assert [hit.complaint_id for hit in got] == [10, 20]
    assert got[0].fused_score == pytest.approx(got[1].fused_score)


def test_retrieve_evidence_returns_auditable_top_ten(
    retrieval_fixture, fake_encoder, vectors, tmp_path
):
    con, cluster_id, company_id = retrieval_fixture
    got = retrieve.retrieve_evidence(
        con, cluster_id, company_id, "escrow refund", "embed-m",
        encoder=fake_encoder, vectors=vectors, cache_dir=tmp_path, top_k=10,
    )
    assert len(got) <= 10
    assert got == sorted(got, key=lambda row: (-row.fused_score, row.complaint_id))
    assert all(row.date_received and row.text_redacted for row in got)
    assert all(row.dense_rank or row.sparse_rank for row in got)
~~~

- [ ] **Step 2: Run fusion tests and verify they fail**

Run: .venv/bin/python -m pytest tests/test_llm_retrieve.py -k "rrf or retrieve_evidence" -v

Expected: FAIL because fusion and orchestration do not exist.

- [ ] **Step 3: Define fused and returned evidence types**

~~~python
@dataclass(frozen=True)
class FusedHit:
    complaint_id: int
    fused_score: float
    dense_rank: int | None
    dense_score: float | None
    sparse_rank: int | None
    sparse_score: float | None


@dataclass(frozen=True)
class RetrievedEvidence:
    complaint_id: int
    date_received: date
    company_id: str
    company_name: str
    product_family: str
    text_redacted: str
    company_public_response: str | None
    dense_rank: int | None
    dense_score: float | None
    sparse_rank: int | None
    sparse_score: float | None
    fused_score: float
~~~

- [ ] **Step 4: Implement RRF**

For every dense hit add 1 / (rrf_k + dense_rank); for every sparse hit add 1 / (rrf_k + sparse_rank). Preserve component rank and score in maps. Sort by descending fused score then ascending complaint_id. Reject rrf_k <= 0 and top_k <= 0.

- [ ] **Step 5: Implement retrieval orchestration with injectable heavy dependencies**

When encoder is None, call src.embed.encode.load_model(embed_model) once. When vectors is None, load PATHS.artifacts / embeddings.<model-tail>.npy. retrieve_variants loads the scoped corpus, encodes the query, ranks CONFIG.llm.rag_candidate_k dense and sparse candidates, fuses to top_k or CONFIG.llm.rag_top_k, and joins FusedHit fields back to EvidenceRecord by complaint_id. Measure dense_seconds around query encoding plus FAISS search, sparse_seconds around cache load plus BM25 ranking, and fusion_seconds around RRF plus evidence joining. Return those values on RetrievalResult. retrieve_evidence is a thin wrapper returning list(retrieve_variants(...).evidence), so ordinary callers stay simple while evaluation can score all three variants without rerunning the encoder.

Never import retrieve.py outside function-local Phase 8 CLI handlers. Extend the existing AST test so detection packages also fail if they query llm_usage, rag_answers, or rag_eval_results.

- [ ] **Step 6: Run retrieval and leakage suites**

Run: .venv/bin/python -m pytest tests/test_llm_retrieve.py tests/test_llm.py -v

Expected: PASS.

Run: .venv/bin/python -m pytest tests/test_signals.py tests/test_leakage.py -v

Expected: PASS.

- [ ] **Step 7: Run Ruff and commit Phase 8B**

Run: .venv/bin/ruff check src/llm/retrieve.py tests/test_llm_retrieve.py tests/test_llm.py

Expected: PASS.

~~~bash
git add src/llm/retrieve.py tests/test_llm_retrieve.py tests/test_llm.py
git commit -m "Add hybrid RAG evidence retrieval"
~~~

## Phase 8B completion gate

- Dense, BM25, and fused rankings are separately returned and testable.
- An out-of-scope complaint cannot appear even if it is the closest vector or strongest lexical match.
- Repeated sparse retrieval hits the membership/version cache.
- Every tie resolves on complaint_id.
- Every final evidence record contains its source ID and date.
- No model call is made anywhere in this subsystem.
