"""Load the explicit, privacy-safe evidence population for RAG retrieval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date

import duckdb
import numpy as np


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


@dataclass(frozen=True)
class RankedHit:
    complaint_id: int
    rank: int
    score: float


def load_corpus(
    con: duckdb.DuckDBPyConnection,
    cluster_id: str,
    company_id: str | None,
    embed_model: str,
) -> ScopedCorpus:
    """Load redacted narratives from exactly one requested evidence scope."""
    query = """
        SELECT
            m.complaint_id,
            m.cluster_id,
            e.row_idx,
            c.date_received,
            c.company_id,
            COALESCE(cc.canonical_name, c.company_id, 'Unknown company') AS company_name,
            c.product_family,
            n.text_redacted,
            r.company_public_response
        FROM cluster_members m
        JOIN narratives n USING (complaint_id)
        JOIN complaints c USING (complaint_id)
        JOIN embedding_map e
          ON e.complaint_id = m.complaint_id AND e.model = ?
        LEFT JOIN company_canonical cc USING (company_id)
        LEFT JOIN complaints_raw r USING (complaint_id)
        WHERE m.cluster_id = ?
    """
    parameters: list[str] = [embed_model, cluster_id]
    if company_id is not None:
        query += " AND c.company_id = ?"
        parameters.append(company_id)
    query += " ORDER BY m.complaint_id"

    records = con.execute(query, parameters).fetchall()
    if not records:
        raise ValueError(
            "no evidence exists for the requested cluster/company/model scope"
        )

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
            company_public_response=(response.strip() or None)
            if response is not None
            else None,
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
    return ScopedCorpus(cluster_id, company_id, embed_model, rows)


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

    row_indices = [row.row_idx for row in corpus.rows]
    if any(row_idx < 0 or row_idx >= all_vectors.shape[0] for row_idx in row_indices):
        raise ValueError("scoped corpus row_idx is outside the embedding matrix")
    matrix = np.ascontiguousarray(all_vectors[row_indices], dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("scoped embedding matrix contains non-finite values")
    if np.any(np.linalg.norm(matrix, axis=1) == 0.0):
        raise ValueError("scoped embedding matrix contains a zero norm vector")

    query = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32))
    if query.ndim != 2 or query.shape[0] != 1:
        raise ValueError("query vector must have shape (1, dim)")
    if query.shape[1] != matrix.shape[1]:
        raise ValueError("query vector dimension must match embedding dimension")
    if not np.isfinite(query).all():
        raise ValueError("query vector contains non-finite values")
    if float(np.linalg.norm(query)) == 0.0:
        raise ValueError("query vector has zero norm")

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
