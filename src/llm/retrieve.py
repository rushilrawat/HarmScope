"""Load the explicit, privacy-safe evidence population for RAG retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np

from src.config import CONFIG, PATHS
from src.population import EXPANDED_SELECT_SQL


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
    embed_dim: int | None = None
    embedding_rows: int | None = None


@dataclass(frozen=True)
class RankedHit:
    complaint_id: int
    rank: int
    score: float


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
    cluster_id: str
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


@dataclass(frozen=True)
class RetrievalResult:
    corpus: ScopedCorpus
    dense: tuple[RankedHit, ...]
    sparse: tuple[RankedHit, ...]
    fused: tuple[FusedHit, ...]
    evidence: tuple[RetrievedEvidence, ...]
    dense_seconds: float
    sparse_seconds: float
    fusion_seconds: float


@dataclass(frozen=True)
class SparseCorpus:
    """Immutable tokenized representation of one scoped evidence corpus."""

    complaint_ids: tuple[int, ...]
    tokens: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        complaint_ids = tuple(self.complaint_ids)
        tokens = tuple(tuple(document) for document in self.tokens)
        if len(complaint_ids) != len(tokens):
            raise ValueError("sparse corpus IDs and token lists must have equal length")
        if any(
            not isinstance(complaint_id, int) or isinstance(complaint_id, bool)
            for complaint_id in complaint_ids
        ):
            raise ValueError("sparse corpus complaint IDs must be integers")
        if len(set(complaint_ids)) != len(complaint_ids):
            raise ValueError("sparse corpus complaint IDs must be unique")
        if any(not isinstance(token, str) for document in tokens for token in document):
            raise ValueError("sparse corpus tokens must be strings")
        object.__setattr__(self, "complaint_ids", complaint_ids)
        object.__setattr__(self, "tokens", tokens)


class SparseCacheError(ValueError):
    """Raised when a persisted sparse corpus does not match its expected schema."""


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Split text into deterministic lowercase lexical-retrieval terms."""
    return _TOKEN_PATTERN.findall(text.lower())


def _sparse_cache_path(corpus: ScopedCorpus, cache_dir: Path, tokenizer_version: str) -> Path:
    return cache_dir / f"bm25.{membership_hash(corpus, tokenizer_version)}.json"


def _validate_sparse_cache(
    payload: object,
    expected_ids: tuple[int, ...],
    tokenizer_version: str,
) -> SparseCorpus:
    if not isinstance(payload, dict):
        raise SparseCacheError("sparse cache must be an object")
    required_fields = {"tokenizer_version", "complaint_ids", "tokens"}
    if set(payload) != required_fields:
        raise SparseCacheError("sparse cache fields do not match the schema")
    if payload["tokenizer_version"] != tokenizer_version:
        raise SparseCacheError("sparse cache tokenizer version does not match")

    complaint_ids = payload["complaint_ids"]
    token_lists = payload["tokens"]
    if not isinstance(complaint_ids, list) or any(
        not isinstance(complaint_id, int) or isinstance(complaint_id, bool)
        for complaint_id in complaint_ids
    ):
        raise SparseCacheError("sparse cache complaint IDs must be integer lists")
    if tuple(complaint_ids) != expected_ids:
        raise SparseCacheError("sparse cache complaint IDs do not match the scope")
    if not isinstance(token_lists, list) or len(token_lists) != len(complaint_ids):
        raise SparseCacheError("sparse cache token lists do not match complaint IDs")
    if any(
        not isinstance(tokens, list) or any(not isinstance(token, str) for token in tokens)
        for tokens in token_lists
    ):
        raise SparseCacheError("sparse cache tokens must be string lists")
    return SparseCorpus(tuple(complaint_ids), tuple(tuple(tokens) for tokens in token_lists))


def _quarantine_sparse_cache(path: Path) -> None:
    """Move an invalid cache aside using the label-cache corrupt-file convention."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    corrupt = path.with_name(f"{path.name}.corrupt-{stamp}")
    suffix = 1
    while corrupt.exists():
        corrupt = path.with_name(f"{path.name}.corrupt-{stamp}-{suffix}")
        suffix += 1
    try:
        os.replace(path, corrupt)
    except FileNotFoundError:
        # Another reader may have observed and quarantined the same bad file
        # after this caller read it. Both callers may safely rebuild because
        # writes use independent temporary files and atomic replacement.
        return


def _write_sparse_cache(path: Path, sparse: SparseCorpus, tokenizer_version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "complaint_ids": list(sparse.complaint_ids),
        "tokenizer_version": tokenizer_version,
        "tokens": [list(tokens) for tokens in sparse.tokens],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as temporary:
        json.dump(payload, temporary, sort_keys=True)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def load_sparse_corpus(
    corpus: ScopedCorpus, cache_dir: Path, tokenizer_version: str
) -> tuple[SparseCorpus, bool]:
    """Load or atomically build the membership- and tokenizer-keyed BM25 cache."""
    expected_ids = tuple(row.complaint_id for row in corpus.rows)
    path = _sparse_cache_path(corpus, cache_dir, tokenizer_version)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _validate_sparse_cache(payload, expected_ids, tokenizer_version), True
        except (json.JSONDecodeError, SparseCacheError, TypeError, UnicodeDecodeError):
            _quarantine_sparse_cache(path)

    sparse = SparseCorpus(
        complaint_ids=expected_ids,
        tokens=tuple(tuple(tokenize(row.text_redacted)) for row in corpus.rows),
    )
    _write_sparse_cache(path, sparse, tokenizer_version)
    return sparse, False


def bm25_rank(sparse: SparseCorpus, question: str, limit: int) -> list[RankedHit]:
    """Rank one sparse corpus with deterministic lexical score tie-breaking."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not sparse.complaint_ids:
        return []

    query_tokens = tokenize(question)
    if not query_tokens or not any(sparse.tokens):
        pairs = [(complaint_id, 0.0) for complaint_id in sparse.complaint_ids]
    else:
        from rank_bm25 import BM25Okapi

        scorer = BM25Okapi([list(tokens) for tokens in sparse.tokens])
        pairs = [
            (complaint_id, float(score))
            for complaint_id, score in zip(
                sparse.complaint_ids, scorer.get_scores(query_tokens), strict=True
            )
        ]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return [
        RankedHit(complaint_id=complaint_id, rank=rank, score=score)
        for rank, (complaint_id, score) in enumerate(pairs[:limit], start=1)
    ]


@dataclass(frozen=True)
class _ClusterProvenance:
    run_id: str
    dedup_run: str
    embed_model: str
    cutoff: date | None


def _cluster_provenance(
    con: duckdb.DuckDBPyConnection,
    cluster_id: str,
    requested_model: str,
) -> _ClusterProvenance:
    row = con.execute(
        """
        SELECT
            c.run_id,
            json_extract_string(r.params_json, '$.params.dedup_run'),
            json_extract_string(r.params_json, '$.params.model'),
            nullif(json_extract_string(r.params_json, '$.params.cutoff'), '')
        FROM clusters c
        JOIN runs r ON r.run_id = c.run_id
        WHERE c.cluster_id = ?
        """,
        [cluster_id],
    ).fetchone()
    if row is None:
        raise ValueError("no evidence exists for the requested cluster/company/model scope")
    run_id, dedup_run, recorded_model, cutoff_text = row
    if not isinstance(dedup_run, str) or not dedup_run.strip():
        raise ValueError(f"cluster run {run_id} does not record a dedup_run")
    if not isinstance(recorded_model, str) or not recorded_model.strip():
        raise ValueError(f"cluster run {run_id} does not record an embedding model")
    if requested_model != recorded_model:
        raise ValueError(
            f"cluster run {run_id} records embedding model {recorded_model!r}; "
            f"requested {requested_model!r}"
        )
    try:
        cutoff = None if cutoff_text is None else date.fromisoformat(cutoff_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cluster run {run_id} records an invalid cutoff {cutoff_text!r}") from exc
    return _ClusterProvenance(run_id, dedup_run, recorded_model, cutoff)


def _embedding_map_metadata(con: duckdb.DuckDBPyConnection, model: str) -> tuple[int, int]:
    n_dims, embed_dim, n_rows, max_row, min_row = con.execute(
        """
        SELECT count(DISTINCT dim), min(dim), count(DISTINCT row_idx),
               max(row_idx), min(row_idx)
        FROM embedding_map
        WHERE model = ?
        """,
        [model],
    ).fetchone()
    if n_dims != 1 or not isinstance(embed_dim, int) or embed_dim <= 0:
        raise ValueError(f"embedding_map has inconsistent dimension metadata for model {model!r}")
    expected_rows = int(max_row) + 1
    if min_row != 0 or n_rows != expected_rows:
        raise ValueError(f"embedding_map row indices are not contiguous for model {model!r}")
    return int(embed_dim), expected_rows


def load_corpus(
    con: duckdb.DuckDBPyConnection,
    cluster_id: str,
    company_id: str | None,
    embed_model: str,
) -> ScopedCorpus:
    """Load one signal-consistent, deduplicated redacted evidence scope."""
    provenance = _cluster_provenance(con, cluster_id, embed_model)
    embed_dim, embedding_rows = _embedding_map_metadata(con, provenance.embed_model)
    query = f"""
        WITH expanded AS (
            {EXPANDED_SELECT_SQL}
        ),
        scoped AS (
            SELECT expanded.*, dates.date_received,
                   row_number() OVER (
                       PARTITION BY expanded.group_id, expanded.company_id
                       ORDER BY dates.date_received, expanded.complaint_id
                   ) AS evidence_rank
            FROM expanded
            JOIN complaints dates ON dates.complaint_id = expanded.complaint_id
            WHERE expanded.cluster_id = ?
              AND expanded.product_family = expanded.cluster_family
              AND (? IS NULL OR expanded.company_id = ?)
        )
        SELECT
            s.complaint_id,
            s.cluster_id,
            e.row_idx,
            c.date_received,
            c.company_id,
            COALESCE(cc.canonical_name, c.company_id, 'Unknown company') AS company_name,
            c.product_family,
            n.text_redacted,
            r.company_public_response
        FROM scoped s
        JOIN narratives n USING (complaint_id)
        JOIN complaints c USING (complaint_id)
        JOIN embedding_map e
          ON e.complaint_id = s.complaint_id AND e.model = ? AND e.dim = ?
        LEFT JOIN company_canonical cc ON cc.company_id = c.company_id
        LEFT JOIN complaints_raw r ON r.complaint_id = s.complaint_id
        WHERE s.evidence_rank = 1
        ORDER BY s.complaint_id
    """  # noqa: S608 - interpolation is the static shared population SQL
    parameters = [
        provenance.run_id,
        provenance.dedup_run,
        provenance.dedup_run,
        provenance.cutoff,
        provenance.cutoff,
        cluster_id,
        company_id,
        company_id,
        provenance.embed_model,
        embed_dim,
    ]
    records = con.execute(query, parameters).fetchall()
    if not records:
        raise ValueError("no evidence exists for the requested cluster/company/model scope")

    rows = tuple(
        EvidenceRecord(
            complaint_id=complaint_id,
            cluster_id=row_cluster_id,
            row_idx=row_idx,
            date_received=date_received,
            company_id=row_company_id,
            company_name=company_name,
            product_family=product_family,
            text_redacted=text_redacted,
            company_public_response=(response.strip() or None) if response is not None else None,
        )
        for (
            complaint_id,
            row_cluster_id,
            row_idx,
            date_received,
            row_company_id,
            company_name,
            product_family,
            text_redacted,
            response,
        ) in records
    )
    return ScopedCorpus(
        cluster_id,
        company_id,
        provenance.embed_model,
        rows,
        embed_dim=embed_dim,
        embedding_rows=embedding_rows,
    )


def membership_hash(corpus: ScopedCorpus, tokenizer_version: str) -> str:
    payload = {
        "cluster_id": corpus.cluster_id,
        "company_id": corpus.company_id,
        "embed_model": corpus.embed_model,
        "tokenizer_version": tokenizer_version,
        "complaint_ids": [row.complaint_id for row in corpus.rows],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def encode_query(encoder, question: str) -> np.ndarray:
    """Encode one question as a unit vector in the evidence vector space."""
    encoded = encoder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    query_vector = np.ascontiguousarray(np.asarray(encoded, dtype=np.float32))
    if query_vector.ndim != 2 or query_vector.shape[0] != 1 or query_vector.shape[1] == 0:
        raise ValueError("encoder must return one query vector with shape (1, dim)")
    if not np.isfinite(query_vector).all():
        raise ValueError("query vector contains non-finite values")

    norm = float(np.linalg.norm(query_vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("query vector has zero norm")
    return np.ascontiguousarray(query_vector / norm, dtype=np.float32)


def dense_rank(
    corpus: ScopedCorpus,
    vectors: np.ndarray,
    query_vector: np.ndarray,
    limit: int,
) -> list[RankedHit]:
    """Rank only the requested evidence scope with exact FAISS inner products."""
    import faiss

    if limit <= 0:
        raise ValueError("limit must be positive")
    if not corpus.rows:
        raise ValueError("cannot rank an empty scoped corpus")

    all_vectors = np.asarray(vectors, dtype=np.float32)
    if all_vectors.ndim != 2 or all_vectors.shape[0] == 0 or all_vectors.shape[1] == 0:
        raise ValueError("vectors must have shape (n, dim)")
    if corpus.embed_dim is not None and all_vectors.shape[1] != corpus.embed_dim:
        raise ValueError(
            "embedding vector dimension does not match embedding_map dimension "
            f"({all_vectors.shape[1]} != {corpus.embed_dim})"
        )

    row_indices = [row.row_idx for row in corpus.rows]
    if any(row_idx < 0 or row_idx >= all_vectors.shape[0] for row_idx in row_indices):
        raise ValueError("scoped corpus row_idx is outside the embedding matrix")
    matrix = np.ascontiguousarray(all_vectors[row_indices], dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("scoped embedding matrix contains non-finite values")
    matrix_norms = np.linalg.norm(matrix, axis=1)
    if np.any(matrix_norms == 0.0):
        raise ValueError("scoped embedding matrix contains a zero norm vector")
    matrix = np.ascontiguousarray(matrix / matrix_norms[:, None], dtype=np.float32)

    query = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32))
    if query.ndim != 2 or query.shape[0] != 1:
        raise ValueError("query vector must have shape (1, dim)")
    if query.shape[1] != matrix.shape[1]:
        raise ValueError("query vector dimension must match embedding dimension")
    if not np.isfinite(query).all():
        raise ValueError("query vector contains non-finite values")
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        raise ValueError("query vector has zero norm")
    query = np.ascontiguousarray(query / query_norm, dtype=np.float32)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    scores, positions = index.search(query, len(corpus.rows))
    pairs = [
        (corpus.rows[int(position)].complaint_id, float(score))
        for position, score in zip(positions[0], scores[0], strict=True)
        if position >= 0
    ]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    ranked = [
        RankedHit(complaint_id=complaint_id, rank=rank, score=score)
        for rank, (complaint_id, score) in enumerate(pairs, start=1)
    ]
    return ranked[:limit]


def _validate_component_ranking(hits: list[RankedHit], component: str) -> None:
    complaint_ids: set[int] = set()
    ranks: set[int] = set()
    for hit in hits:
        if not isinstance(hit.complaint_id, int) or isinstance(hit.complaint_id, bool):
            raise ValueError(f"{component} complaint IDs must be integers")
        if not isinstance(hit.rank, int) or isinstance(hit.rank, bool) or hit.rank <= 0:
            raise ValueError(f"{component} ranks must be positive integers")
        if not np.isfinite(hit.score):
            raise ValueError(f"{component} scores must be finite")
        if hit.complaint_id in complaint_ids:
            raise ValueError(f"{component} ranking contains a duplicate complaint ID")
        if hit.rank in ranks:
            raise ValueError(f"{component} ranking contains a duplicate rank")
        complaint_ids.add(hit.complaint_id)
        ranks.add(hit.rank)


def reciprocal_rank_fusion(
    dense: list[RankedHit],
    sparse: list[RankedHit],
    rrf_k: int,
    top_k: int,
) -> list[FusedHit]:
    """Fuse dense and sparse component ranks with deterministic source-ID ties."""
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    _validate_component_ranking(dense, "dense")
    _validate_component_ranking(sparse, "sparse")

    dense_by_id = {hit.complaint_id: hit for hit in dense}
    sparse_by_id = {hit.complaint_id: hit for hit in sparse}
    fused = []
    for complaint_id in dense_by_id.keys() | sparse_by_id.keys():
        dense_hit = dense_by_id.get(complaint_id)
        sparse_hit = sparse_by_id.get(complaint_id)
        score = 0.0
        if dense_hit is not None:
            score += 1.0 / (rrf_k + dense_hit.rank)
        if sparse_hit is not None:
            score += 1.0 / (rrf_k + sparse_hit.rank)
        fused.append(
            FusedHit(
                complaint_id=complaint_id,
                fused_score=score,
                dense_rank=None if dense_hit is None else dense_hit.rank,
                dense_score=None if dense_hit is None else dense_hit.score,
                sparse_rank=None if sparse_hit is None else sparse_hit.rank,
                sparse_score=None if sparse_hit is None else sparse_hit.score,
            )
        )
    fused.sort(key=lambda hit: (-hit.fused_score, hit.complaint_id))
    return fused[:top_k]


def _load_default_vectors(corpus: ScopedCorpus) -> np.ndarray:
    """Load the cluster-bound vector artifact after validating its sidecar."""
    from src.embed.encode import Progress, embedding_artifact_paths

    if corpus.embed_dim is None or corpus.embedding_rows is None:
        raise ValueError("scoped corpus is missing embedding_map metadata")
    artifact = embedding_artifact_paths(PATHS.artifacts, corpus.embed_model).memmap
    metadata_path = artifact.with_suffix(".progress.json")
    try:
        metadata = Progress.read(metadata_path)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid embedding artifact metadata at {metadata_path}") from exc
    if metadata is None:
        raise ValueError(f"missing embedding artifact metadata at {metadata_path}")
    if metadata.model != corpus.embed_model:
        raise ValueError(
            "embedding artifact metadata model does not match the cluster run "
            f"({metadata.model!r} != {corpus.embed_model!r})"
        )
    if metadata.n_total != corpus.embedding_rows:
        raise ValueError(
            "embedding artifact metadata row count does not match embedding_map "
            f"({metadata.n_total} != {corpus.embedding_rows})"
        )
    if metadata.dim != corpus.embed_dim:
        raise ValueError(
            "embedding artifact metadata dimension does not match embedding_map "
            f"({metadata.dim} != {corpus.embed_dim})"
        )
    if metadata.n_done != metadata.n_total:
        raise ValueError(
            "embedding artifact metadata does not describe a complete encode "
            f"({metadata.n_done} of {metadata.n_total})"
        )

    vectors = np.load(artifact, mmap_mode="r")
    if vectors.ndim != 2:
        raise ValueError("embedding artifact must be a two-dimensional array")
    if vectors.shape[0] != metadata.n_total:
        raise ValueError(
            "embedding artifact row count does not match metadata "
            f"({vectors.shape[0]} != {metadata.n_total})"
        )
    if vectors.shape[1] != metadata.dim:
        raise ValueError(
            "embedding artifact dimension does not match metadata "
            f"({vectors.shape[1]} != {metadata.dim})"
        )
    return vectors


def retrieve_variants(
    con: duckdb.DuckDBPyConnection,
    cluster_id: str,
    company_id: str | None,
    question: str,
    embed_model: str,
    encoder=None,
    vectors=None,
    cache_dir=None,
    top_k=None,
) -> RetrievalResult:
    """Retrieve independently observable dense, sparse, and fused evidence ranks."""
    corpus = load_corpus(con, cluster_id, company_id, embed_model)
    if encoder is None:
        from src.embed.encode import load_model

        encoder, _device = load_model(embed_model)
    if vectors is None:
        vectors = _load_default_vectors(corpus)

    dense_started = time.perf_counter()
    query_vector = encode_query(encoder, question)
    dense = dense_rank(corpus, vectors, query_vector, CONFIG.llm.rag_candidate_k)
    dense_seconds = time.perf_counter() - dense_started

    sparse_started = time.perf_counter()
    sparse_corpus, _cache_hit = load_sparse_corpus(
        corpus,
        PATHS.llm_cache if cache_dir is None else Path(cache_dir),
        CONFIG.llm.bm25_tokenizer_version,
    )
    sparse = bm25_rank(sparse_corpus, question, CONFIG.llm.rag_candidate_k)
    sparse_seconds = time.perf_counter() - sparse_started

    fusion_started = time.perf_counter()
    fused = reciprocal_rank_fusion(
        dense,
        sparse,
        CONFIG.llm.rrf_k,
        CONFIG.llm.rag_top_k if top_k is None else top_k,
    )
    rows_by_id = {row.complaint_id: row for row in corpus.rows}
    evidence_rows = tuple(
        RetrievedEvidence(
            complaint_id=hit.complaint_id,
            cluster_id=corpus.cluster_id,
            date_received=rows_by_id[hit.complaint_id].date_received,
            company_id=rows_by_id[hit.complaint_id].company_id,
            company_name=rows_by_id[hit.complaint_id].company_name,
            product_family=rows_by_id[hit.complaint_id].product_family,
            text_redacted=rows_by_id[hit.complaint_id].text_redacted,
            company_public_response=rows_by_id[hit.complaint_id].company_public_response,
            dense_rank=hit.dense_rank,
            dense_score=hit.dense_score,
            sparse_rank=hit.sparse_rank,
            sparse_score=hit.sparse_score,
            fused_score=hit.fused_score,
        )
        for hit in fused
    )
    fusion_seconds = time.perf_counter() - fusion_started
    return RetrievalResult(
        corpus=corpus,
        dense=tuple(dense),
        sparse=tuple(sparse),
        fused=tuple(fused),
        evidence=evidence_rows,
        dense_seconds=dense_seconds,
        sparse_seconds=sparse_seconds,
        fusion_seconds=fusion_seconds,
    )


def retrieve_evidence(
    con: duckdb.DuckDBPyConnection,
    cluster_id: str,
    company_id: str | None,
    question: str,
    embed_model: str,
    encoder=None,
    vectors=None,
    cache_dir=None,
    top_k=None,
) -> list[RetrievedEvidence]:
    """Return only final evidence for callers that do not need stage diagnostics."""
    return list(
        retrieve_variants(
            con,
            cluster_id,
            company_id,
            question,
            embed_model,
            encoder=encoder,
            vectors=vectors,
            cache_dir=cache_dir,
            top_k=top_k,
        ).evidence
    )
