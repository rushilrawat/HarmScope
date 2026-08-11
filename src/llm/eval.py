"""Reproducible, privacy-guarded RAG evaluation authoring contracts.

The committed manifest is deliberately ID-only. Consumer narratives are read
from DuckDB only while validating a human-authored question or preparing a
gitignored authoring worklist.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import tempfile
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from numbers import Real
from pathlib import Path
from statistics import median

from src.config import CONFIG, PATHS
from src.llm import retrieve
from src.population import EXPANDED_SELECT_SQL

CATEGORY_ORDER = (
    "mechanism",
    "actors_preconditions",
    "consumer_consequence",
    "time_sequence",
    "company_response",
    "unanswerable",
)
CATEGORIES = frozenset(CATEGORY_ORDER)
MANIFEST_HEADER = (
    "question_id",
    "question",
    "cluster_id",
    "company_id",
    "category",
    "answerable",
    "relevant_complaint_ids",
)
_EVIDENCE_HEADER = tuple(
    field
    for index in range(1, 11)
    for field in (f"evidence_{index}_complaint_id", f"evidence_{index}_text_redacted")
)
AUTHORING_HEADER = (
    *MANIFEST_HEADER,
    "product_family",
    "fired_status",
    *_EVIDENCE_HEADER,
    "privacy_reviewed",
)

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_FORMULA_PREFIXES = ("=", "+", "-", "@")
_SHINGLE_SIZE = 8
_EXCERPTS_PER_ROW = 10
_QUESTIONS_PER_CATEGORY = 5
_RETRIEVAL_METHODS = ("dense", "bm25", "fused")


class ManifestError(ValueError):
    """The evaluation manifest or private authoring artifact is invalid."""


class EvaluationTransactionError(RuntimeError):
    """The evaluation runner requires an autocommit DuckDB connection."""


@dataclass(frozen=True)
class EvalQuestion:
    question_id: str
    question: str
    cluster_id: str
    company_id: str | None
    category: str
    relevant_complaint_ids: frozenset[int]
    answerable: bool


@dataclass(frozen=True)
class RetrievalMetrics:
    rank_first_relevant: int | None
    relevant_retrieved_count: int
    recall_at_10: float
    reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalMethodSummary:
    recall_at_10: float
    reciprocal_rank: float
    median_latency_seconds: float
    p95_latency_seconds: float


@dataclass(frozen=True)
class WinTieLoss:
    wins: int
    ties: int
    losses: int


@dataclass(frozen=True)
class QuestionRetrievalEvaluation:
    question_id: str
    answerable: bool
    dense: RetrievalMetrics
    bm25: RetrievalMetrics
    fused: RetrievalMetrics
    dense_seconds: float
    bm25_seconds: float
    fused_seconds: float

    @property
    def methods(self) -> dict[str, RetrievalMetrics]:
        return {"dense": self.dense, "bm25": self.bm25, "fused": self.fused}

    @property
    def latencies(self) -> dict[str, float]:
        return {
            "dense": self.dense_seconds,
            "bm25": self.bm25_seconds,
            "fused": self.fused_seconds,
        }


@dataclass(frozen=True)
class EvaluationSummary:
    answerable_count: int
    unanswerable_count: int
    dense: RetrievalMethodSummary
    bm25: RetrievalMethodSummary
    fused: RetrievalMethodSummary
    fused_vs_dense: WinTieLoss
    fused_vs_bm25: WinTieLoss
    questions: tuple[QuestionRetrievalEvaluation, ...]

    @property
    def methods(self) -> dict[str, RetrievalMethodSummary]:
        return {"dense": self.dense, "bm25": self.bm25, "fused": self.fused}

    def render(self) -> str:
        """Render stable aggregate and per-question metrics without narrative text."""
        lines = [
            "RAG retrieval evaluation",
            f"answerable: {self.answerable_count}",
            f"unanswerable: {self.unanswerable_count}",
            "method | Recall@10 | MRR | median latency (s) | p95 latency (s)",
        ]
        for method in _RETRIEVAL_METHODS:
            aggregate = self.methods[method]
            lines.append(
                f"{method} | {aggregate.recall_at_10:.6f} | "
                f"{aggregate.reciprocal_rank:.6f} | "
                f"{aggregate.median_latency_seconds:.6f} | "
                f"{aggregate.p95_latency_seconds:.6f}"
            )
        dense_comparison = self.fused_vs_dense
        bm25_comparison = self.fused_vs_bm25
        lines.extend(
            [
                "fused vs dense win/tie/loss: "
                f"{dense_comparison.wins}/{dense_comparison.ties}/{dense_comparison.losses}",
                "fused vs BM25 win/tie/loss: "
                f"{bm25_comparison.wins}/{bm25_comparison.ties}/{bm25_comparison.losses}",
                "question_id | answerable | dense R@10/MRR | BM25 R@10/MRR | "
                "fused R@10/MRR | fused-vs-dense | fused-vs-BM25",
            ]
        )
        for question in sorted(self.questions, key=lambda row: row.question_id):
            dense_relation = _relation(question.fused, question.dense)
            bm25_relation = _relation(question.fused, question.bm25)
            lines.append(
                f"{question.question_id} | {'yes' if question.answerable else 'no'} | "
                f"{question.dense.recall_at_10:.6f}/{question.dense.reciprocal_rank:.6f} | "
                f"{question.bm25.recall_at_10:.6f}/{question.bm25.reciprocal_rank:.6f} | "
                f"{question.fused.recall_at_10:.6f}/{question.fused.reciprocal_rank:.6f} | "
                f"{dense_relation} | {bm25_relation}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class _AuthoringCandidate:
    cluster_id: str
    company_id: str | None
    product_family: str
    did_fire: bool


def _positive_complaint_ids(values, label: str) -> None:
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f"{label} must contain positive integers")


def score_ranking(
    ranked_ids: list[int], relevant_ids: set[int] | frozenset[int], k: int = 10
) -> RetrievalMetrics:
    """Compute top-k recall and reciprocal rank from hand-marked relevance IDs."""
    if type(ranked_ids) is not list:
        raise TypeError("ranked_ids must be a list")
    if not isinstance(relevant_ids, (set, frozenset)):
        raise TypeError("relevant_ids must be a set")
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    _positive_complaint_ids(ranked_ids, "ranked_ids")
    _positive_complaint_ids(relevant_ids, "relevant_ids")
    if not relevant_ids:
        return RetrievalMetrics(None, 0, 0.0, 0.0)

    top_k = ranked_ids[:k]
    first_rank = next(
        (rank for rank, complaint_id in enumerate(top_k, start=1) if complaint_id in relevant_ids),
        None,
    )
    retrieved = len(set(top_k) & relevant_ids)
    return RetrievalMetrics(
        rank_first_relevant=first_rank,
        relevant_retrieved_count=retrieved,
        recall_at_10=retrieved / len(relevant_ids),
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
    )


def _validate_component_ranking(hits: tuple[retrieve.RankedHit, ...], label: str) -> list[int]:
    if type(hits) is not tuple:
        raise ValueError(f"{label} ranking must be a tuple")
    complaint_ids: list[int] = []
    for expected_rank, hit in enumerate(hits, start=1):
        if not isinstance(hit, retrieve.RankedHit):
            raise ValueError(f"{label} ranking contains an invalid hit")
        if hit.rank != expected_rank:
            raise ValueError(f"{label} rank sequence must be one-based and contiguous")
        if type(hit.complaint_id) is not int or hit.complaint_id <= 0:
            raise ValueError(f"{label} complaint IDs must be positive integers")
        if (
            not isinstance(hit.score, Real)
            or isinstance(hit.score, bool)
            or not math.isfinite(hit.score)
        ):
            raise ValueError(f"{label} scores must be finite numbers")
        complaint_ids.append(hit.complaint_id)
    if len(complaint_ids) != len(set(complaint_ids)):
        raise ValueError(f"{label} ranking contains duplicate complaint IDs")
    return complaint_ids


def _validate_fused_ranking(hits: tuple[retrieve.FusedHit, ...]) -> list[int]:
    if type(hits) is not tuple:
        raise ValueError("fused ranking must be a tuple")
    complaint_ids: list[int] = []
    for hit in hits:
        if not isinstance(hit, retrieve.FusedHit):
            raise ValueError("fused ranking contains an invalid hit")
        if type(hit.complaint_id) is not int or hit.complaint_id <= 0:
            raise ValueError("fused complaint IDs must be positive integers")
        if (
            not isinstance(hit.fused_score, Real)
            or isinstance(hit.fused_score, bool)
            or not math.isfinite(hit.fused_score)
        ):
            raise ValueError("fused scores must be finite numbers")
        complaint_ids.append(hit.complaint_id)
    if len(complaint_ids) != len(set(complaint_ids)):
        raise ValueError("fused ranking contains duplicate complaint IDs")
    return complaint_ids


def evaluate_retrieval_question(
    question: EvalQuestion, result: retrieve.RetrievalResult
) -> dict[str, RetrievalMetrics]:
    """Score dense, BM25, and fused orderings from one retrieval result."""
    if not isinstance(question, EvalQuestion):
        raise TypeError("question must be an EvalQuestion")
    if not isinstance(result, retrieve.RetrievalResult):
        raise TypeError("result must be a RetrievalResult")
    if (
        result.corpus.cluster_id != question.cluster_id
        or result.corpus.company_id != question.company_id
    ):
        raise ValueError("retrieval result scope does not match the evaluation question")
    dense_ids = _validate_component_ranking(result.dense, "dense")
    bm25_ids = _validate_component_ranking(result.sparse, "bm25")
    fused_ids = _validate_fused_ranking(result.fused)
    relevant = set(question.relevant_complaint_ids)
    return {
        "dense": score_ranking(dense_ids, relevant),
        "bm25": score_ranking(bm25_ids, relevant),
        "fused": score_ranking(fused_ids, relevant),
    }


def _latency(value: object, label: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} latency must be a finite non-negative number")
    return float(value)


def _require_autocommit(con) -> None:
    first_id = con.execute("SELECT current_transaction_id()").fetchone()[0]
    second_id = con.execute("SELECT current_transaction_id()").fetchone()[0]
    if first_id == second_id:
        raise EvaluationTransactionError("run_retrieval_eval requires an autocommit connection")


def _validate_retrieval_batch(
    questions: list[EvalQuestion], embed_model: str, eval_run_id: str
) -> None:
    if type(questions) is not list or not questions:
        raise ValueError("questions must be a non-empty list")
    if type(embed_model) is not str or not embed_model.strip():
        raise ValueError("embed_model is required")
    if type(eval_run_id) is not str or not eval_run_id.strip():
        raise ValueError("eval_run_id is required")
    seen: set[str] = set()
    for question in questions:
        if not isinstance(question, EvalQuestion):
            raise TypeError("questions must contain EvalQuestion values")
        if not _SAFE_ID.fullmatch(question.question_id):
            raise ValueError("question_id must be a safe non-blank identifier")
        if question.question_id in seen:
            raise ValueError(f"duplicate question_id {question.question_id!r}")
        seen.add(question.question_id)
        if question.answerable and not question.relevant_complaint_ids:
            raise ValueError("answerable questions require relevant complaint IDs")
        if not question.answerable and question.relevant_complaint_ids:
            raise ValueError("unanswerable questions cannot have relevant complaint IDs")


def _write_retrieval_rows(
    con,
    eval_run_id: str,
    question: EvalQuestion,
    metrics: dict[str, RetrievalMetrics],
    latencies: dict[str, float],
) -> None:
    if tuple(metrics) != _RETRIEVAL_METHODS or tuple(latencies) != _RETRIEVAL_METHODS:
        raise ValueError("retrieval rows require exactly dense, bm25, and fused methods")
    con.execute("BEGIN TRANSACTION")
    try:
        for method in _RETRIEVAL_METHODS:
            metric = metrics[method]
            con.execute(
                "INSERT INTO rag_eval_results "
                "(eval_run_id, question_id, retrieval_method, rank_first_relevant, "
                "relevant_retrieved_count, recall_at_10, reciprocal_rank, "
                "latency_seconds, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (eval_run_id, question_id, retrieval_method) DO UPDATE SET "
                "rank_first_relevant = excluded.rank_first_relevant, "
                "relevant_retrieved_count = excluded.relevant_retrieved_count, "
                "recall_at_10 = excluded.recall_at_10, "
                "reciprocal_rank = excluded.reciprocal_rank, "
                "latency_seconds = excluded.latency_seconds",
                [
                    eval_run_id,
                    question.question_id,
                    method,
                    metric.rank_first_relevant,
                    metric.relevant_retrieved_count,
                    metric.recall_at_10,
                    metric.reciprocal_rank,
                    latencies[method],
                    datetime.now(),
                ],
            )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


def _relation(left: RetrievalMetrics, right: RetrievalMetrics) -> str:
    left_pair = (left.recall_at_10, left.reciprocal_rank)
    right_pair = (right.recall_at_10, right.reciprocal_rank)
    if left_pair > right_pair:
        return "win"
    if left_pair < right_pair:
        return "loss"
    return "tie"


def _comparison(questions: list[QuestionRetrievalEvaluation], component: str) -> WinTieLoss:
    counts = Counter(
        _relation(question.fused, getattr(question, component)) for question in questions
    )
    return WinTieLoss(counts["win"], counts["tie"], counts["loss"])


def _percentile(values: list[float], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _aggregate_method(
    questions: list[QuestionRetrievalEvaluation], method: str
) -> RetrievalMethodSummary:
    answerable = [question for question in questions if question.answerable]
    recall = (
        sum(getattr(question, method).recall_at_10 for question in answerable) / len(answerable)
        if answerable
        else 0.0
    )
    reciprocal_rank = (
        sum(getattr(question, method).reciprocal_rank for question in answerable) / len(answerable)
        if answerable
        else 0.0
    )
    latencies = [question.latencies[method] for question in questions]
    return RetrievalMethodSummary(
        recall_at_10=recall,
        reciprocal_rank=reciprocal_rank,
        median_latency_seconds=float(median(latencies)) if latencies else 0.0,
        p95_latency_seconds=_percentile(latencies, 0.95),
    )


def _summarize_retrieval(
    questions: list[QuestionRetrievalEvaluation],
) -> EvaluationSummary:
    answerable_count = sum(question.answerable for question in questions)
    return EvaluationSummary(
        answerable_count=answerable_count,
        unanswerable_count=len(questions) - answerable_count,
        dense=_aggregate_method(questions, "dense"),
        bm25=_aggregate_method(questions, "bm25"),
        fused=_aggregate_method(questions, "fused"),
        fused_vs_dense=_comparison(questions, "dense"),
        fused_vs_bm25=_comparison(questions, "bm25"),
        questions=tuple(questions),
    )


def run_retrieval_eval(
    con,
    questions: list[EvalQuestion],
    embed_model: str,
    eval_run_id: str,
    retriever=retrieve.retrieve_variants,
) -> EvaluationSummary:
    """Evaluate and persist all retrieval variants with one call per question."""
    _require_autocommit(con)
    _validate_retrieval_batch(questions, embed_model, eval_run_id)
    evaluated: list[QuestionRetrievalEvaluation] = []
    for question in questions:
        result = retriever(
            con,
            question.cluster_id,
            question.company_id,
            question.question,
            embed_model,
        )
        if result.corpus.embed_model != embed_model:
            raise ValueError("retrieval result embedding model does not match embed_model")
        metrics = evaluate_retrieval_question(question, result)
        latencies = {
            "dense": _latency(result.dense_seconds, "dense"),
            "bm25": _latency(result.sparse_seconds, "bm25"),
            "fused": _latency(result.fusion_seconds, "fused"),
        }
        _write_retrieval_rows(con, eval_run_id, question, metrics, latencies)
        evaluated.append(
            QuestionRetrievalEvaluation(
                question_id=question.question_id,
                answerable=question.answerable,
                dense=metrics["dense"],
                bm25=metrics["bm25"],
                fused=metrics["fused"],
                dense_seconds=latencies["dense"],
                bm25_seconds=latencies["bm25"],
                fused_seconds=latencies["fused"],
            )
        )
    return _summarize_retrieval(evaluated)


def _required(value: str | None, field: str, line_number: int) -> str:
    if value is None or not value.strip():
        raise ManifestError(f"line {line_number}: {field} is required")
    return value.strip()


def _safe_identifier(value: str | None, field: str, line_number: int) -> str:
    identifier = _required(value, field, line_number)
    if not _SAFE_ID.fullmatch(identifier):
        raise ManifestError(f"line {line_number}: {field} is not a safe identifier")
    return identifier


def _relevant_ids(value: str | None, line_number: int) -> frozenset[int]:
    raw = value or ""
    if not raw:
        return frozenset()
    pieces = raw.split(";")
    try:
        ids = [int(piece) for piece in pieces]
    except ValueError as exc:
        raise ManifestError(
            f"line {line_number}: relevant_complaint_ids must be positive integers"
        ) from exc
    if any(complaint_id <= 0 for complaint_id in ids):
        raise ManifestError(f"line {line_number}: relevant_complaint_ids must be positive integers")
    if any(str(complaint_id) != piece for complaint_id, piece in zip(ids, pieces, strict=True)):
        raise ManifestError(
            f"line {line_number}: relevant_complaint_ids must use canonical integers"
        )
    if len(ids) != len(set(ids)):
        raise ManifestError(f"line {line_number}: duplicate relevant_complaint_ids are forbidden")
    if ids != sorted(ids):
        raise ManifestError(f"line {line_number}: relevant_complaint_ids must be ascending")
    return frozenset(ids)


def _parse_rows(rows: list[dict[str, str]], expected_n: int) -> list[EvalQuestion]:
    if (
        not isinstance(expected_n, int)
        or isinstance(expected_n, bool)
        or expected_n <= 0
        or expected_n % len(CATEGORY_ORDER)
    ):
        raise ManifestError("expected_n must be a positive multiple of six")
    if len(rows) != expected_n:
        raise ManifestError(f"manifest must contain exactly {expected_n} questions")

    parsed: list[EvalQuestion] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        question_id = _safe_identifier(row.get("question_id"), "question_id", line_number)
        if question_id in seen_ids:
            raise ManifestError(f"line {line_number}: duplicate question_id {question_id!r}")
        seen_ids.add(question_id)
        question = _required(row.get("question"), "question", line_number)
        if question.lstrip().startswith(_FORMULA_PREFIXES):
            raise ManifestError(f"line {line_number}: question is unsafe for spreadsheet display")
        cluster_id = _safe_identifier(row.get("cluster_id"), "cluster_id", line_number)
        company_raw = (row.get("company_id") or "").strip()
        company_id = (
            _safe_identifier(company_raw, "company_id", line_number) if company_raw else None
        )
        category = _required(row.get("category"), "category", line_number)
        if category not in CATEGORIES:
            raise ManifestError(f"line {line_number}: unknown category {category!r}")
        raw_answerable = row.get("answerable")
        if raw_answerable not in {"true", "false"}:
            raise ManifestError(f"line {line_number}: answerable must be true or false")
        answerable = raw_answerable == "true"
        relevant = _relevant_ids(row.get("relevant_complaint_ids"), line_number)
        if answerable and not relevant:
            raise ManifestError(
                f"line {line_number}: answerable questions require relevant complaint IDs"
            )
        if not answerable and relevant:
            raise ManifestError(
                f"line {line_number}: unanswerable questions cannot have relevant complaint IDs"
            )
        if not answerable and category != "unanswerable":
            raise ManifestError(
                f"line {line_number}: unanswerable rows must use the unanswerable category"
            )
        if category == "unanswerable" and answerable:
            raise ManifestError(
                f"line {line_number}: the unanswerable category must set answerable=false"
            )
        parsed.append(
            EvalQuestion(
                question_id,
                question,
                cluster_id,
                company_id,
                category,
                relevant,
                answerable,
            )
        )

    required_per_category = expected_n // len(CATEGORY_ORDER)
    distribution = Counter(row.category for row in parsed)
    expected = dict.fromkeys(CATEGORY_ORDER, required_per_category)
    if distribution != expected:
        raise ManifestError(
            "manifest categories must be balanced: "
            + ", ".join(f"{category}={required_per_category}" for category in CATEGORY_ORDER)
        )
    return parsed


def load_manifest(path: Path, expected_n: int = 30) -> list[EvalQuestion]:
    """Load the exact seven-column, category-balanced evaluation manifest."""
    rows = _read_strict_csv(path, MANIFEST_HEADER, "manifest")
    return _parse_rows(rows, expected_n)


def _read_strict_csv(path: Path, header: tuple[str, ...], artifact: str) -> list[dict[str, str]]:
    """Read a CSV only when every physical row has exactly the header's arity."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(header):
                raise ManifestError(f"{artifact} columns do not match the contract")
            rows: list[dict[str, str]] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ManifestError(
                        f"line {line_number}: {artifact} field count does not match its header"
                    )
                rows.append(row)
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{artifact} must be UTF-8") from exc
    return rows


def _shingles(tokens: list[str]) -> set[tuple[str, ...]]:
    return {
        tuple(tokens[index : index + _SHINGLE_SIZE])
        for index in range(len(tokens) - _SHINGLE_SIZE + 1)
    }


def _embed_model_for_cluster(con, question_id: str, cluster_id: str) -> str:
    row = con.execute(
        """
        SELECT json_extract_string(r.params_json, '$.params.model')
        FROM clusters c JOIN runs r ON r.run_id = c.run_id
        WHERE c.cluster_id = ?
        """,
        [cluster_id],
    ).fetchone()
    if row is None:
        raise ManifestError(f"question {question_id!r} references an unknown cluster_id")
    model = row[0]
    if not isinstance(model, str) or not model.strip():
        raise ManifestError(
            f"question {question_id!r} has no retrievable evidence for its cluster/model scope"
        )
    return model


def _scope_rows(con, question: EvalQuestion) -> list[tuple[int, str]]:
    model = _embed_model_for_cluster(con, question.question_id, question.cluster_id)
    try:
        corpus = retrieve.load_corpus(con, question.cluster_id, question.company_id, model)
    except ValueError:
        raise ManifestError(
            f"question {question.question_id!r} has no retrievable evidence for "
            "its cluster/company/model scope"
        ) from None
    return [(row.complaint_id, row.text_redacted) for row in corpus.rows]


def validate_manifest(con, questions: list[EvalQuestion]) -> None:
    """Validate exact DB scope, relevant IDs, and normalized quote privacy."""
    for question in questions:
        rows = _scope_rows(con, question)
        scoped_ids = {complaint_id for complaint_id, _ in rows}
        outside = question.relevant_complaint_ids - scoped_ids
        if outside:
            raise ManifestError(
                f"question {question.question_id!r} has relevant complaint IDs outside "
                "its exact cluster/company scope"
            )
        question_shingles = _shingles(retrieve.tokenize(question.question))
        if not question_shingles:
            continue
        for complaint_id, text_redacted in rows:
            if question_shingles & _shingles(retrieve.tokenize(text_redacted)):
                raise ManifestError(
                    f"question {question.question_id!r} has an eight-token overlap "
                    f"with complaint_id {complaint_id}"
                )


def _latest_signals_provenance(con) -> tuple[str, str]:
    row = con.execute(
        """
        SELECT run_id, json_extract_string(params_json, '$.params.cluster_run')
        FROM runs
        WHERE phase = 'signals' AND status = 'ok'
          AND coalesce(json_extract(params_json, '$.params.shuffle'), '0') = '0'
          AND coalesce(json_extract(params_json, '$.params.limit'), 'null') = 'null'
          AND json_extract_string(params_json, '$.params.cluster_run') IS NOT NULL
        ORDER BY started_at DESC, run_id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise ManifestError("no successful full-input signals run is available")
    return row[0], row[1]


def _authoring_candidates(con) -> list[_AuthoringCandidate]:
    signals_run, cluster_run = _latest_signals_provenance(con)
    provenance = con.execute(
        """
        SELECT json_extract_string(params_json, '$.params.dedup_run'),
               json_extract_string(params_json, '$.params.model'),
               nullif(json_extract_string(params_json, '$.params.cutoff'), '')
        FROM runs WHERE run_id = ? AND phase = 'cluster' AND status = 'ok'
        """,
        [cluster_run],
    ).fetchone()
    if provenance is None or not all(
        isinstance(value, str) and value.strip() for value in provenance[:2]
    ):
        raise ManifestError("the authoring cluster run lacks retrieval provenance")
    dedup_run, embed_model, cutoff = provenance
    query = f"""
        WITH expanded AS (
          {EXPANDED_SELECT_SQL}
        ),
        ranked AS (
          SELECT expanded.*, dates.date_received,
                 row_number() OVER (
                   PARTITION BY expanded.group_id, expanded.company_id
                   ORDER BY dates.date_received, expanded.complaint_id
                 ) AS evidence_rank
          FROM expanded
          JOIN complaints dates USING (complaint_id)
          WHERE expanded.cluster_id IS NOT NULL
            AND expanded.product_family = expanded.cluster_family
        ),
        retrievable AS (
          SELECT ranked.cluster_id, ranked.company_id, ranked.group_id,
                 ranked.complaint_id
          FROM ranked
          JOIN narratives n USING (complaint_id)
          JOIN embedding_map e
            ON e.complaint_id = ranked.complaint_id AND e.model = ?
          WHERE ranked.evidence_rank = 1
        ),
        scope_counts AS (
          SELECT cluster_id, company_id, count(*) AS n_evidence
          FROM retrievable GROUP BY cluster_id, company_id
        ),
        cluster_counts AS (
          SELECT cluster_id, sum(n_evidence) AS n_evidence
          FROM scope_counts GROUP BY cluster_id
        ),
        grouped_signals AS (
          SELECT cluster_id, company_id,
                 min(CASE WHEN method = 'ebgm' THEN q_value END) AS q_value,
                 max(CASE WHEN method IN ('ewma', 'pelt') THEN 1 ELSE 0 END) AS changed,
                 max(n_supporting_groups) AS n_groups
          FROM signals WHERE run_id = ? GROUP BY cluster_id, company_id
        ),
        canonical_fired AS (
          SELECT g.cluster_id, g.company_id
          FROM grouped_signals g JOIN clusters c USING (cluster_id)
          WHERE c.run_id = ? AND c.coherence >= ? AND g.n_groups >= ?
            AND (g.q_value <= ? OR g.changed = 1)
        ),
        fired AS (
          SELECT fs.cluster_id,
                 CASE WHEN fs.company_id = '__ALL__' THEN NULL ELSE fs.company_id END
                   AS company_id,
                 cl.product_family,
                 true AS did_fire
          FROM canonical_fired fs
          JOIN clusters cl ON cl.cluster_id = fs.cluster_id AND cl.run_id = ?
          LEFT JOIN scope_counts sc
            ON sc.cluster_id = fs.cluster_id AND sc.company_id = fs.company_id
          LEFT JOIN cluster_counts cc ON cc.cluster_id = fs.cluster_id
          WHERE CASE WHEN fs.company_id = '__ALL__'
                     THEN coalesce(cc.n_evidence, 0)
                     ELSE coalesce(sc.n_evidence, 0)
                END >= ?
        ),
        controls AS (
          SELECT cl.cluster_id, NULL AS company_id, cl.product_family, false AS did_fire
          FROM clusters cl
          JOIN cluster_counts cc USING (cluster_id)
          WHERE cl.run_id = ? AND cc.n_evidence >= ?
            AND NOT EXISTS (
              SELECT 1 FROM canonical_fired f WHERE f.cluster_id = cl.cluster_id
            )
        )
        SELECT DISTINCT cluster_id, company_id, product_family, did_fire FROM fired
        UNION ALL
        SELECT cluster_id, company_id, product_family, did_fire FROM controls
        ORDER BY cluster_id, company_id
        """  # noqa: S608 - interpolation is the static shared population SQL
    rows = con.execute(
        query,
        [
            cluster_run,
            dedup_run,
            dedup_run,
            cutoff,
            cutoff,
            embed_model,
            signals_run,
            cluster_run,
            CONFIG.novelty.min_coherence,
            CONFIG.signals.min_supporting_groups,
            CONFIG.signals.fdr_alpha,
            cluster_run,
            _EXCERPTS_PER_ROW,
            cluster_run,
            _EXCERPTS_PER_ROW,
        ],
    ).fetchall()
    return [_AuthoringCandidate(*row) for row in rows]


def _tie_break(seed: int, category: str, candidate: _AuthoringCandidate) -> str:
    material = f"{seed}\0{category}\0{candidate.cluster_id}\0{candidate.company_id or ''}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _select_authoring_candidates(
    candidates: list[_AuthoringCandidate], seed: int
) -> list[tuple[str, _AuthoringCandidate]]:
    if len({candidate.cluster_id for candidate in candidates}) < 30:
        raise ManifestError("authoring requires at least 30 eligible distinct clusters")

    remaining = list(candidates)
    used_clusters: set[str] = set()
    selected: list[tuple[str, _AuthoringCandidate]] = []
    global_product: Counter[str] = Counter()
    global_status: Counter[bool] = Counter()

    for category_index, category in enumerate(CATEGORY_ORDER):
        category_product: Counter[str] = Counter()
        category_status: Counter[bool] = Counter()
        for slot in range(_QUESTIONS_PER_CATEGORY):
            preferred_fired = (slot + category_index) % 2 == 0
            available = [
                candidate for candidate in remaining if candidate.cluster_id not in used_clusters
            ]
            preferred = [
                candidate for candidate in available if candidate.did_fire == preferred_fired
            ]
            pool = preferred or available
            if not pool:
                raise ManifestError("eligible authoring population was exhausted")
            chosen = min(
                pool,
                key=lambda candidate: (
                    category_product[candidate.product_family],
                    global_product[candidate.product_family],
                    category_status[candidate.did_fire],
                    global_status[candidate.did_fire],
                    _tie_break(seed, category, candidate),
                ),
            )
            selected.append((category, chosen))
            used_clusters.add(chosen.cluster_id)
            category_product[chosen.product_family] += 1
            category_status[chosen.did_fire] += 1
            global_product[chosen.product_family] += 1
            global_status[chosen.did_fire] += 1
    return selected


def _spreadsheet_safe(value: str | None) -> str:
    normalized = " ".join((value or "").split())
    if normalized.startswith(_FORMULA_PREFIXES):
        normalized = "'" + normalized
    return normalized[: CONFIG.llm.max_narrative_chars]


def _evidence(con, candidate: _AuthoringCandidate) -> list[tuple[int, str]]:
    model = _embed_model_for_cluster(con, "authoring", candidate.cluster_id)
    try:
        corpus = retrieve.load_corpus(con, candidate.cluster_id, candidate.company_id, model)
    except ValueError:
        raise ManifestError(
            f"candidate cluster {candidate.cluster_id!r} no longer has retrievable evidence"
        ) from None
    return [(row.complaint_id, row.text_redacted) for row in corpus.rows[:_EXCERPTS_PER_ROW]]


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _ensure_private_destination(path: Path) -> None:
    resolved = path.resolve()
    allowed = PATHS.interim.resolve()
    repository_roots = [PATHS.root.resolve()]
    if PATHS.root.parent.name == ".worktrees":
        repository_roots.append(PATHS.root.parent.parent.resolve())
    if any(_inside(resolved, root) for root in repository_roots) and not _inside(resolved, allowed):
        raise ManifestError("private authoring worklists must stay under data/interim")


def _write_rows(path: Path, header: tuple[str, ...], rows: list[dict[str, str]]) -> Path:
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not parent_existed:
        _fsync_directory(path.parent.parent)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=list(header), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
        return path
    except BaseException:
        if temporary_path is not None:
            with suppress(FileNotFoundError):
                temporary_path.unlink()
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def export_authoring_worklist(con, seed: int, path: Path) -> Path:
    """Export thirty deterministic candidates with private redacted evidence."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ManifestError("seed must be an integer")
    _ensure_private_destination(path)
    selected = _select_authoring_candidates(_authoring_candidates(con), seed)
    rows: list[dict[str, str]] = []
    for question_number, (category, candidate) in enumerate(selected, start=1):
        evidence = _evidence(con, candidate)
        if len(evidence) != _EXCERPTS_PER_ROW:
            raise ManifestError(
                f"candidate cluster {candidate.cluster_id!r} no longer has ten excerpts"
            )
        row = dict.fromkeys(AUTHORING_HEADER, "")
        row.update(
            {
                "question_id": f"rag-{question_number:03d}",
                "cluster_id": candidate.cluster_id,
                "company_id": candidate.company_id or "",
                "category": category,
                "product_family": _spreadsheet_safe(candidate.product_family),
                "fired_status": "fired" if candidate.did_fire else "control",
            }
        )
        for index, (complaint_id, text_redacted) in enumerate(evidence, start=1):
            row[f"evidence_{index}_complaint_id"] = str(complaint_id)
            row[f"evidence_{index}_text_redacted"] = _spreadsheet_safe(text_redacted)
        rows.append(row)
    return _write_rows(path, AUTHORING_HEADER, rows)


def import_authoring_worklist(con, source: Path, destination: Path) -> Path:
    """Validate a completed human worklist and emit only committed ID columns."""
    source_rows = _read_strict_csv(source, AUTHORING_HEADER, "authoring worklist")
    if len(source_rows) != len(CATEGORY_ORDER) * _QUESTIONS_PER_CATEGORY:
        raise ManifestError("authoring worklist must contain exactly 30 completed rows")

    manifest_rows: list[dict[str, str]] = []
    for line_number, row in enumerate(source_rows, start=2):
        if row.get("privacy_reviewed") != "yes":
            raise ManifestError(
                f"line {line_number}: privacy_reviewed must equal yes after human review"
            )
        manifest_rows.append({field: row.get(field, "") for field in MANIFEST_HEADER})

    manifest_rows.sort(key=lambda row: row["question_id"])
    questions = _parse_rows(manifest_rows, expected_n=30)
    validate_manifest(con, questions)
    return _write_rows(destination, MANIFEST_HEADER, manifest_rows)
