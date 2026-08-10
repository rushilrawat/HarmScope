"""Load the explicit, privacy-safe evidence population for RAG retrieval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date

import duckdb


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
