"""Reproducible, privacy-guarded RAG evaluation authoring contracts.

The committed manifest is deliberately ID-only. Consumer narratives are read
from DuckDB only while validating a human-authored question or preparing a
gitignored authoring worklist.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from numbers import Real
from pathlib import Path
from statistics import median

from src.alert_scope import canonical_alert_scopes
from src.config import CONFIG, PATHS
from src.llm import answer, retrieve, verify
from src.population import EXPANDED_SELECT_SQL
from src.private_artifacts import (
    PrivatePathError,
    atomic_write_bytes,
    canonical_private_path,
    read_private_bytes,
    read_stable_bytes,
)

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
_CLAIM_REVIEW_VERSION = 2
_CLAIM_FAILURE_CATEGORIES = frozenset(
    {"unsupported", "contradicted", "overgeneralized", "citation_mismatch", "other"}
)
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_DISPLAY_CONTROLS = frozenset("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")
CLAIM_REVIEW_HEADER = (
    "review_id",
    "eval_run_id",
    "question_id",
    "question",
    "claim_position",
    "claim_text",
    "cited_complaint_ids",
    "cited_evidence_json",
    "grounded",
    "failure_category",
    "notes",
)


class ManifestError(ValueError):
    """The evaluation manifest or private authoring artifact is invalid."""


class EvaluationTransactionError(RuntimeError):
    """The evaluation runner requires an autocommit DuckDB connection."""


@dataclass(frozen=True)
class ClaimEvidence:
    complaint_id: int
    text_redacted: str


@dataclass(frozen=True)
class ClaimReview:
    """One completed decision from a blinded, run-bound claim worklist."""

    review_id: str
    eval_run_id: str
    question_id: str
    question: str
    claim_position: int
    claim_text: str
    cited_complaint_ids: tuple[int, ...]
    cited_evidence: tuple[ClaimEvidence, ...]
    grounded: bool
    failure_category: str
    notes: str | None
    reviewer_id: str


@dataclass(frozen=True)
class GroundednessReport:
    grounded: int
    reviewed: int
    rate: float
    ci_low: float
    ci_high: float
    reviewer_id: str

    @property
    def gate_complete(self) -> bool:
        """Only a real minimum-size denominator can complete the human gate."""
        return self.reviewed >= CONFIG.llm.human_verify_n

    def render(self) -> str:
        status = "COMPLETE" if self.gate_complete else "PENDING"
        return "\n".join(
            [
                f"human groundedness gate: {status}",
                f"reviewer: {self.reviewer_id}",
                f"grounded claims: {self.grounded}/{self.reviewed} ({self.rate:.1%})",
                f"Wilson 95% interval: {self.ci_low:.1%}..{self.ci_high:.1%}",
            ]
        )


@dataclass(frozen=True)
class _ClaimCandidate:
    review_id: str
    eval_run_id: str
    question_id: str
    category: str
    answerable: bool
    company_id: str
    question: str
    claim_position: int
    claim_text: str
    cited_complaint_ids: tuple[int, ...]
    cited_evidence: tuple[ClaimEvidence, ...]


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
class PreparedAuthoringImport:
    """One exact, fully parsed private worklist identity awaiting DB validation."""

    source: Path
    source_sha256: str
    source_bytes: bytes = field(repr=False)
    questions: tuple[EvalQuestion, ...] = field(repr=False)
    manifest_rows: tuple[tuple[str, ...], ...] = field(repr=False)


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
class ScoredRetrieval:
    """One question's immutable in-memory link to its persisted retrieval score."""

    question_id: str
    result: retrieve.RetrievalResult = field(repr=False)


@dataclass(frozen=True)
class RetrievalEvaluationRun:
    """A retrieval summary plus exact evidence retained for same-run answering."""

    summary: EvaluationSummary
    scored: tuple[ScoredRetrieval, ...]


def _grounded_claims(value: object) -> tuple[answer.Claim, ...]:
    if not isinstance(value, answer.GroundedAnswer):
        raise TypeError("answer must be a GroundedAnswer")
    if type(value.claims) is not tuple or any(
        not isinstance(claim, answer.Claim) for claim in value.claims
    ):
        raise TypeError("GroundedAnswer claims must be a tuple of Claim values")
    if type(value.insufficient_evidence) is not bool:
        raise TypeError("insufficient_evidence must be a boolean")
    return value.claims


def citation_validity(
    value: answer.GroundedAnswer,
    retrieved_ids: set[int],
    *,
    evidence_text_by_id: object,
) -> bool:
    """Return whether the complete answer is valid against exact retrieved text."""
    _grounded_claims(value)
    if type(retrieved_ids) is not set:
        raise TypeError("retrieved_ids must be a set")
    _positive_complaint_ids(retrieved_ids, "retrieved_ids")
    try:
        answer.validate_answer(
            answer._answer_payload(value),
            retrieved_ids,
            evidence_text_by_id=evidence_text_by_id,
        )
    except (answer.AnswerSchemaError, answer.CitationError):
        return False
    return True


def citation_coverage(value: answer.GroundedAnswer) -> float:
    """Return the share of claims carrying at least one citation."""
    claims = _grounded_claims(value)
    if not claims:
        return 1.0 if value.insufficient_evidence else 0.0
    cited = 0
    for claim in claims:
        complaint_ids = claim.complaint_ids
        if type(complaint_ids) is tuple and complaint_ids:
            cited += 1
    return cited / len(claims)


def abstention_correct(answerable: bool, insufficient_evidence: bool) -> bool:
    """Return whether the answer's abstention state matches benchmark answerability."""
    if type(answerable) is not bool or type(insufficient_evidence) is not bool:
        raise TypeError("answerable and insufficient_evidence must be booleans")
    return answerable != insufficient_evidence


@dataclass(frozen=True)
class AnswerQuestionEvaluation:
    question_id: str
    citation_valid: bool
    citation_coverage: float
    abstention_correct: bool


@dataclass(frozen=True)
class AnswerEvaluationFailure:
    question_id: str
    category: str


@dataclass(frozen=True)
class AnswerEvaluationSummary:
    attempted_count: int
    completed_count: int
    citation_valid_count: int
    citation_validity_rate: float | None
    citation_coverage: float | None
    abstention_correct_count: int
    abstention_accuracy: float | None
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    total_latency_seconds: float
    estimated_cost_usd: float
    cache_hits: int
    cache_misses: int
    cache_bypasses: int
    outcome_ok: int
    outcome_refused: int
    outcome_failed: int
    outcome_skipped: int
    questions: tuple[AnswerQuestionEvaluation, ...]
    failures: tuple[AnswerEvaluationFailure, ...]

    @property
    def failed_count(self) -> int:
        return len(self.failures)

    def render(self) -> str:
        """Render stable aggregate and ID-only per-question answer results."""
        lines = [
            "RAG answer evaluation",
            f"attempted: {self.attempted_count}",
            f"completed: {self.completed_count}",
            f"failed: {self.failed_count}",
            (
                "citation validity: n/a"
                if self.citation_validity_rate is None
                else "citation validity: "
                f"{self.citation_valid_count}/{self.completed_count} "
                f"({self.citation_validity_rate:.6f})"
            ),
            (
                "citation coverage: n/a"
                if self.citation_coverage is None
                else f"citation coverage: {self.citation_coverage:.6f}"
            ),
            (
                "abstention accuracy: n/a"
                if self.abstention_accuracy is None
                else "abstention accuracy: "
                f"{self.abstention_correct_count}/{self.completed_count} "
                f"({self.abstention_accuracy:.6f})"
            ),
            f"input tokens: {self.input_tokens}",
            f"output tokens: {self.output_tokens}",
            f"prompt cache read tokens: {self.cache_read_input_tokens}",
            f"prompt cache creation tokens: {self.cache_creation_input_tokens}",
            f"total answer latency (s): {self.total_latency_seconds:.6f}",
            f"estimated cost (USD): {self.estimated_cost_usd:.6f}",
            f"cache hit/miss/bypass: {self.cache_hits}/{self.cache_misses}/{self.cache_bypasses}",
            "outcome ok/refused/failed/skipped: "
            f"{self.outcome_ok}/{self.outcome_refused}/"
            f"{self.outcome_failed}/{self.outcome_skipped}",
            "question_id | citation valid | citation coverage | abstention correct",
        ]
        for question in sorted(self.questions, key=lambda row: row.question_id):
            lines.append(
                f"{question.question_id} | "
                f"{'yes' if question.citation_valid else 'no'} | "
                f"{question.citation_coverage:.6f} | "
                f"{'yes' if question.abstention_correct else 'no'}"
            )
        lines.append("failed question_id | category")
        if self.failures:
            lines.extend(
                f"{failure.question_id} | {failure.category}"
                for failure in sorted(self.failures, key=lambda row: row.question_id)
            )
        else:
            lines.append("(none)")
        return "\n".join(lines)


def render_evaluation_run(
    eval_run_id: str,
    retrieval_summary: EvaluationSummary,
    answer_summary: AnswerEvaluationSummary | None,
) -> str:
    """Render one stable run report while keeping unrun gates visibly unavailable."""
    if type(eval_run_id) is not str or not _SAFE_ID.fullmatch(eval_run_id):
        raise ValueError("eval_run_id must be a safe non-blank identifier")
    if not isinstance(retrieval_summary, EvaluationSummary):
        raise TypeError("retrieval_summary must be an EvaluationSummary")
    if answer_summary is not None and not isinstance(answer_summary, AnswerEvaluationSummary):
        raise TypeError("answer_summary must be an AnswerEvaluationSummary or None")

    if answer_summary is None:
        answer_section = "\n".join(
            [
                "RAG answer evaluation: n/a (retrieval-only run)",
                "citation validity: n/a",
                "citation coverage: n/a",
                "abstention accuracy: n/a",
                "tokens: n/a",
                "cache outcomes: n/a",
                "answer latency: n/a",
                "estimated cost: n/a",
            ]
        )
    else:
        answer_section = answer_summary.render()
    human_section = "\n".join(
        [
            "human groundedness gate: PENDING",
            f"claim review run ID: {eval_run_id}",
            f"required human-reviewed claims: at least {CONFIG.llm.human_verify_n}",
        ]
    )
    return "\n\n".join(
        [
            f"eval run ID: {eval_run_id}",
            retrieval_summary.render(),
            answer_section,
            human_section,
        ]
    )


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


def _validate_fused_ranking(
    hits: tuple[retrieve.FusedHit, ...],
    dense: tuple[retrieve.RankedHit, ...],
    sparse: tuple[retrieve.RankedHit, ...],
) -> list[int]:
    if type(hits) is not tuple:
        raise ValueError("fused ranking must be a tuple")
    expected = tuple(
        retrieve.reciprocal_rank_fusion(
            list(dense),
            list(sparse),
            CONFIG.llm.rrf_k,
            CONFIG.llm.rag_top_k,
        )
    )
    if hits != expected:
        raise ValueError("fused ranking does not match the configured reciprocal-rank fusion")
    return [hit.complaint_id for hit in hits]


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
    fused_ids = _validate_fused_ranking(result.fused, result.dense, result.sparse)
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
    if type(eval_run_id) is not str or not _SAFE_ID.fullmatch(eval_run_id):
        raise ValueError("eval_run_id must be a safe non-blank identifier")
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
    *,
    retain_results: bool = False,
) -> EvaluationSummary | RetrievalEvaluationRun:
    """Evaluate and persist all retrieval variants with one call per question."""
    _validate_retrieval_batch(questions, embed_model, eval_run_id)
    if type(retain_results) is not bool:
        raise TypeError("retain_results must be a boolean")
    _require_autocommit(con)
    evaluated: list[QuestionRetrievalEvaluation] = []
    scored: list[ScoredRetrieval] = []
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
        if retain_results:
            scored.append(ScoredRetrieval(question.question_id, result))
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
    summary = _summarize_retrieval(evaluated)
    if retain_results:
        return RetrievalEvaluationRun(summary, tuple(scored))
    return summary


def _validate_answer_batch(
    questions: list[EvalQuestion],
    embed_model: str,
    eval_run_id: str,
    answerer,
) -> None:
    _validate_retrieval_batch(questions, embed_model, eval_run_id)
    if not callable(answerer):
        raise TypeError("answerer must be callable")
    for question in questions:
        if type(question.question) is not str or not question.question.strip():
            raise ValueError("evaluation question text must be nonblank")
        if type(question.cluster_id) is not str or not _SAFE_ID.fullmatch(question.cluster_id):
            raise ValueError("cluster_id must be a safe non-blank identifier")
        if type(question.company_id) is not str or not _SAFE_ID.fullmatch(question.company_id):
            raise ValueError("answer evaluation requires a concrete safe non-blank company_id")
        if question.category not in CATEGORIES:
            raise ValueError("evaluation question category is invalid")
        if type(question.answerable) is not bool:
            raise TypeError("evaluation question answerable must be a boolean")
        if (question.category == "unanswerable") != (not question.answerable):
            raise ValueError("evaluation question category and answerable state disagree")
        if type(question.relevant_complaint_ids) is not frozenset:
            raise TypeError("relevant_complaint_ids must be a frozenset")
        _positive_complaint_ids(
            question.relevant_complaint_ids,
            "relevant_complaint_ids",
        )


def _require_fused_rows(con, questions: list[EvalQuestion], eval_run_id: str) -> None:
    present = {
        question_id
        for (question_id,) in con.execute(
            "SELECT question_id FROM rag_eval_results "
            "WHERE eval_run_id = ? AND retrieval_method = 'fused'",
            [eval_run_id],
        ).fetchall()
    }
    missing = sorted(
        question.question_id for question in questions if question.question_id not in present
    )
    if missing:
        raise ValueError("fused retrieval rows are missing for question IDs: " + ", ".join(missing))


def _retrievable_answer_scopes(
    con,
    questions: list[EvalQuestion],
    embed_model: str,
) -> dict[tuple[str, str], tuple[str, dict[int, str]]]:
    scopes: dict[tuple[str, str], tuple[str, dict[int, str]]] = {}
    keys = sorted({(question.cluster_id, question.company_id) for question in questions})
    for cluster_id, company_id in keys:
        corpus = retrieve.load_corpus(con, cluster_id, company_id, embed_model)
        if corpus.cluster_id != cluster_id or corpus.company_id != company_id:
            raise ValueError("retrievable answer corpus scope does not match the question")
        product_families = {row.product_family for row in corpus.rows}
        if len(product_families) != 1:
            raise ValueError("retrievable answer corpus has inconsistent product families")
        evidence_text_by_id = {row.complaint_id: row.text_redacted for row in corpus.rows}
        if len(evidence_text_by_id) != len(corpus.rows) or any(
            type(complaint_id) is not int or complaint_id <= 0
            for complaint_id in evidence_text_by_id
        ):
            raise ValueError("retrievable answer corpus has invalid complaint IDs")
        if any(type(text) is not str for text in evidence_text_by_id.values()):
            raise ValueError("retrievable answer corpus has invalid evidence text")
        scopes[(cluster_id, company_id)] = (product_families.pop(), evidence_text_by_id)
    return scopes


def _index_scored_retrievals(
    questions: list[EvalQuestion],
    scored_retrievals: object,
) -> dict[str, retrieve.RetrievalResult]:
    if type(scored_retrievals) is not tuple:
        raise TypeError("scored_retrievals must be a tuple")
    indexed: dict[str, retrieve.RetrievalResult] = {}
    for scored in scored_retrievals:
        if type(scored) is not ScoredRetrieval:
            raise TypeError("scored_retrievals must contain ScoredRetrieval values")
        if type(scored.question_id) is not str or not _SAFE_ID.fullmatch(scored.question_id):
            raise ValueError("scored retrieval question_id must be a safe identifier")
        if scored.question_id in indexed:
            raise ValueError("scored retrieval question IDs must be unique")
        if type(scored.result) is not retrieve.RetrievalResult:
            raise TypeError("scored retrieval result must be a RetrievalResult")
        indexed[scored.question_id] = scored.result
    expected = {question.question_id for question in questions}
    if set(indexed) != expected:
        raise ValueError("scored retrievals must exactly cover evaluation question IDs")
    return indexed


def _fused_evidence(result: retrieve.RetrievalResult) -> tuple[retrieve.RetrievedEvidence, ...]:
    rows_by_id = {row.complaint_id: row for row in result.corpus.rows}
    if len(rows_by_id) != len(result.corpus.rows):
        raise ValueError("scored retrieval corpus has duplicate complaint IDs")
    expected: list[retrieve.RetrievedEvidence] = []
    for hit in result.fused:
        row = rows_by_id.get(hit.complaint_id)
        if row is None:
            raise ValueError("scored fused ranking references evidence outside its corpus")
        expected.append(
            retrieve.RetrievedEvidence(
                complaint_id=hit.complaint_id,
                cluster_id=result.corpus.cluster_id,
                date_received=row.date_received,
                company_id=row.company_id,
                company_name=row.company_name,
                product_family=row.product_family,
                text_redacted=row.text_redacted,
                company_public_response=row.company_public_response,
                dense_rank=hit.dense_rank,
                dense_score=hit.dense_score,
                sparse_rank=hit.sparse_rank,
                sparse_score=hit.sparse_score,
                fused_score=hit.fused_score,
            )
        )
    return tuple(expected)


def _require_persisted_retrieval_identity(
    con,
    eval_run_id: str,
    question: EvalQuestion,
    metrics: dict[str, RetrievalMetrics],
    latencies: dict[str, float],
) -> None:
    rows = con.execute(
        "SELECT retrieval_method, rank_first_relevant, relevant_retrieved_count, "
        "recall_at_10, reciprocal_rank, latency_seconds FROM rag_eval_results "
        "WHERE eval_run_id = ? AND question_id = ? ORDER BY retrieval_method",
        [eval_run_id, question.question_id],
    ).fetchall()
    actual = {method: values for method, *values in rows}
    if set(actual) != set(_RETRIEVAL_METHODS):
        raise ValueError("persisted retrieval rows do not exactly cover all scored methods")
    for method in _RETRIEVAL_METHODS:
        metric = metrics[method]
        expected = [
            metric.rank_first_relevant,
            metric.relevant_retrieved_count,
            metric.recall_at_10,
            metric.reciprocal_rank,
            latencies[method],
        ]
        if actual[method] != expected:
            raise ValueError("persisted retrieval rows do not match the scored retrieval identity")


def _validated_scored_answer_inputs(
    con,
    questions: list[EvalQuestion],
    embed_model: str,
    eval_run_id: str,
    indexed: dict[str, retrieve.RetrievalResult],
) -> tuple[
    dict[tuple[str, str], tuple[str, dict[int, str]]],
    dict[str, tuple[retrieve.RetrievedEvidence, ...]],
]:
    scopes: dict[tuple[str, str], tuple[str, dict[int, str]]] = {}
    evidence_by_question: dict[str, tuple[retrieve.RetrievedEvidence, ...]] = {}
    for question in questions:
        result = indexed[question.question_id]
        if result.corpus.embed_model != embed_model:
            raise ValueError("scored retrieval embedding model does not match embed_model")
        metrics = evaluate_retrieval_question(question, result)
        latencies = {
            "dense": _latency(result.dense_seconds, "dense"),
            "bm25": _latency(result.sparse_seconds, "bm25"),
            "fused": _latency(result.fusion_seconds, "fused"),
        }
        live_corpus = retrieve.load_corpus(
            con,
            question.cluster_id,
            question.company_id,
            embed_model,
        )
        if result.corpus != live_corpus:
            raise ValueError("scored retrieval corpus does not match exact live provenance")
        exact_evidence = _fused_evidence(result)
        if type(result.evidence) is not tuple or result.evidence != exact_evidence:
            raise ValueError("scored retrieval does not contain the exact fused evidence")
        _require_persisted_retrieval_identity(
            con,
            eval_run_id,
            question,
            metrics,
            latencies,
        )
        product_families = {row.product_family for row in live_corpus.rows}
        if len(product_families) != 1:
            raise ValueError("retrievable answer corpus has inconsistent product families")
        evidence_text_by_id = {row.complaint_id: row.text_redacted for row in live_corpus.rows}
        if len(evidence_text_by_id) != len(live_corpus.rows) or any(
            type(complaint_id) is not int or complaint_id <= 0
            for complaint_id in evidence_text_by_id
        ):
            raise ValueError("retrievable answer corpus has invalid complaint IDs")
        if any(type(text) is not str for text in evidence_text_by_id.values()):
            raise ValueError("retrievable answer corpus has invalid evidence text")
        scopes[(question.cluster_id, question.company_id)] = (
            product_families.pop(),
            evidence_text_by_id,
        )
        evidence_by_question[question.question_id] = exact_evidence
    return scopes, evidence_by_question


def _bound_evidence_retriever(
    question: EvalQuestion,
    embed_model: str,
    evidence: tuple[retrieve.RetrievedEvidence, ...],
):
    def bound(con, cluster_id, company_id, raw_question, raw_embed_model):
        del con
        if (
            cluster_id != question.cluster_id
            or company_id != question.company_id
            or raw_question != question.question
            or raw_embed_model != embed_model
        ):
            raise ValueError("answerer requested evidence outside the scored retrieval identity")
        return list(evidence)

    return bound


def _validated_returned_evidence(
    result: object,
    question: EvalQuestion,
    product_family: str,
    live_evidence_text_by_id: dict[int, str],
    expected_evidence: tuple[retrieve.RetrievedEvidence, ...] | None = None,
) -> list[retrieve.RetrievedEvidence]:
    if type(result) is not answer.AnswerResult:
        raise TypeError("answerer must return an AnswerResult")
    evidence = result.evidence
    if type(evidence) is not tuple or any(
        type(row) is not retrieve.RetrievedEvidence for row in evidence
    ):
        raise ValueError("answer evidence must be a tuple of RetrievedEvidence values")
    # Keep evaluation aligned with the reviewed Phase 8C evidence boundary
    # instead of maintaining a second, inevitably drifting validation copy.
    validated = answer._validate_retrieved_evidence(
        list(evidence),
        cluster_id=question.cluster_id,
        company_id=question.company_id,
        product_family=product_family,
        top_k=CONFIG.llm.rag_top_k,
    )
    if expected_evidence is not None and tuple(validated) != expected_evidence:
        raise ValueError("answer evidence does not match the exact scored fused evidence")
    complaint_ids = [row.complaint_id for row in validated]
    if any(type(complaint_id) is not int or complaint_id <= 0 for complaint_id in complaint_ids):
        raise ValueError("answer evidence complaint IDs must be positive integers")
    if not set(complaint_ids).issubset(live_evidence_text_by_id):
        raise ValueError("answer evidence IDs are outside the exact retrievable scope")
    if any(type(row.text_redacted) is not str for row in validated):
        raise ValueError("answer evidence text must be exact strings")
    if any(
        row.text_redacted.encode("utf-8")
        != live_evidence_text_by_id[row.complaint_id].encode("utf-8")
        for row in validated
    ):
        raise ValueError("answer evidence text does not match exact live retrievable evidence")
    return validated


def _write_answer_metrics(
    con,
    eval_run_id: str,
    metrics: AnswerQuestionEvaluation,
) -> None:
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "UPDATE rag_eval_results SET citation_valid = ?, citation_coverage = ?, "
            "abstention_correct = ? WHERE eval_run_id = ? AND question_id = ? "
            "AND retrieval_method = 'fused'",
            [
                metrics.citation_valid,
                metrics.citation_coverage,
                metrics.abstention_correct,
                eval_run_id,
                metrics.question_id,
            ],
        )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


def _usage_nonnegative_integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"answer usage {field} must be a non-negative integer")
    return value


def _usage_nonnegative_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"answer usage {field} must be a finite non-negative number")
    return float(value)


def _answer_usage(con, eval_run_id: str) -> dict[str, object]:
    rows = con.execute(
        "SELECT cache_status, outcome, input_tokens, output_tokens, "
        "cache_read_input_tokens, cache_creation_input_tokens, latency_seconds, "
        "estimated_cost_usd FROM llm_usage "
        "WHERE run_id = ? AND operation = 'answer' ORDER BY usage_id",
        [eval_run_id],
    ).fetchall()
    cache_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    input_tokens = 0
    output_tokens = 0
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0
    latencies: list[float] = []
    costs: list[float] = []
    for row in rows:
        cache_status, outcome, raw_input, raw_output, raw_read, raw_creation, latency, cost = row
        if cache_status not in {"hit", "miss", "bypass"}:
            raise ValueError("answer usage cache_status is invalid")
        if outcome not in {"ok", "refused", "failed", "skipped"}:
            raise ValueError("answer usage outcome is invalid")
        cache_counts[cache_status] += 1
        outcome_counts[outcome] += 1
        input_tokens += _usage_nonnegative_integer(raw_input, "input_tokens")
        output_tokens += _usage_nonnegative_integer(raw_output, "output_tokens")
        cache_read_input_tokens += _usage_nonnegative_integer(raw_read, "cache_read_input_tokens")
        cache_creation_input_tokens += _usage_nonnegative_integer(
            raw_creation, "cache_creation_input_tokens"
        )
        latencies.append(_usage_nonnegative_number(latency, "latency_seconds"))
        costs.append(_usage_nonnegative_number(cost, "estimated_cost_usd"))
    total_latency = math.fsum(latencies)
    estimated_cost = math.fsum(costs)
    if not math.isfinite(total_latency) or not math.isfinite(estimated_cost):
        raise ValueError("answer usage aggregate must be finite")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "total_latency_seconds": total_latency,
        "estimated_cost_usd": estimated_cost,
        "cache_hits": cache_counts["hit"],
        "cache_misses": cache_counts["miss"],
        "cache_bypasses": cache_counts["bypass"],
        "outcome_ok": outcome_counts["ok"],
        "outcome_refused": outcome_counts["refused"],
        "outcome_failed": outcome_counts["failed"],
        "outcome_skipped": outcome_counts["skipped"],
    }


def _summarize_answers(
    attempted_count: int,
    evaluated: list[AnswerQuestionEvaluation],
    failures: list[AnswerEvaluationFailure],
    usage: dict[str, object],
) -> AnswerEvaluationSummary:
    completed = len(evaluated)
    valid_count = sum(question.citation_valid for question in evaluated)
    abstention_count = sum(question.abstention_correct for question in evaluated)
    return AnswerEvaluationSummary(
        attempted_count=attempted_count,
        completed_count=completed,
        citation_valid_count=valid_count,
        citation_validity_rate=valid_count / completed if completed else None,
        citation_coverage=(
            math.fsum(question.citation_coverage for question in evaluated) / completed
            if completed
            else None
        ),
        abstention_correct_count=abstention_count,
        abstention_accuracy=abstention_count / completed if completed else None,
        questions=tuple(evaluated),
        failures=tuple(failures),
        **usage,
    )


def run_answer_eval(
    con,
    questions: list[EvalQuestion],
    embed_model: str,
    eval_run_id: str,
    answerer=answer.answer_question,
    *,
    scored_retrievals: tuple[ScoredRetrieval, ...] | None = None,
) -> AnswerEvaluationSummary:
    """Evaluate grounded answers against existing fused retrieval results."""
    _validate_answer_batch(questions, embed_model, eval_run_id, answerer)
    indexed = (
        None
        if scored_retrievals is None
        else _index_scored_retrievals(questions, scored_retrievals)
    )
    _require_autocommit(con)
    _require_fused_rows(con, questions, eval_run_id)
    if indexed is None:
        retrievable_scopes = _retrievable_answer_scopes(con, questions, embed_model)
        evidence_by_question: dict[str, tuple[retrieve.RetrievedEvidence, ...]] = {}
    else:
        retrievable_scopes, evidence_by_question = _validated_scored_answer_inputs(
            con,
            questions,
            embed_model,
            eval_run_id,
            indexed,
        )
    evaluated: list[AnswerQuestionEvaluation] = []
    failures: list[AnswerEvaluationFailure] = []
    for question in questions:
        try:
            answer_kwargs = {"run_id": eval_run_id}
            if indexed is not None:
                answer_kwargs["retriever"] = _bound_evidence_retriever(
                    question,
                    embed_model,
                    evidence_by_question[question.question_id],
                )
            result = answerer(
                con,
                question.cluster_id,
                question.company_id,
                question.question,
                embed_model,
                **answer_kwargs,
            )
        except answer.CitationError:
            failures.append(AnswerEvaluationFailure(question.question_id, "citation"))
            continue
        except answer.AnswerSchemaError:
            failures.append(AnswerEvaluationFailure(question.question_id, "schema"))
            continue
        except answer.AnswerRefusalError:
            failures.append(AnswerEvaluationFailure(question.question_id, "refusal"))
            continue

        product_family, live_evidence_text_by_id = retrievable_scopes[
            (question.cluster_id, question.company_id)
        ]
        returned_evidence = _validated_returned_evidence(
            result,
            question,
            product_family,
            live_evidence_text_by_id,
            evidence_by_question.get(question.question_id),
        )
        try:
            canonical_answer = answer._validate_grounded_answer(
                result.answer,
                returned_evidence,
            )
        except answer.CitationError:
            failures.append(AnswerEvaluationFailure(question.question_id, "citation"))
            continue
        except answer.AnswerSchemaError:
            failures.append(AnswerEvaluationFailure(question.question_id, "schema"))
            continue
        returned_text_by_id = {row.complaint_id: row.text_redacted for row in returned_evidence}
        retrieved_ids = set(returned_text_by_id)
        metrics = AnswerQuestionEvaluation(
            question_id=question.question_id,
            citation_valid=citation_validity(
                canonical_answer,
                retrieved_ids,
                evidence_text_by_id=returned_text_by_id,
            ),
            citation_coverage=citation_coverage(canonical_answer),
            abstention_correct=abstention_correct(
                question.answerable,
                canonical_answer.insufficient_evidence,
            ),
        )
        _write_answer_metrics(con, eval_run_id, metrics)
        evaluated.append(metrics)
    return _summarize_answers(
        len(questions),
        evaluated,
        failures,
        _answer_usage(con, eval_run_id),
    )


def _review_text(value: object, label: str) -> str:
    """Render untrusted prose inertly without changing its identity digest."""
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be nonblank text")
    safe = "".join(
        " "
        if ord(character) <= 0x1F
        or 0x7F <= ord(character) <= 0x9F
        or character in _DISPLAY_CONTROLS
        else character
        for character in value
    )
    safe = " ".join(safe.split())
    if safe.startswith(_FORMULA_PREFIXES):
        safe = "'" + safe
    if not safe:
        raise ValueError(f"{label} becomes blank after display sanitization")
    return safe


def _cache_input_hash(
    *,
    question_hash: str,
    evidence_hash: str,
    cluster_id: str,
    company_id: str,
    model: str,
    prompt_version: str,
) -> str:
    payload = json.dumps(
        {
            "cluster_id": cluster_id,
            "company_id": company_id,
            "evidence_hash": evidence_hash,
            "model": model,
            "prompt_version": prompt_version,
            "question_hash": question_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stored_json(value: object, label: str, question_id: str) -> object:
    if type(value) is not str:
        raise ValueError(f"question {question_id!r} has invalid cached {label}")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise ValueError(f"question {question_id!r} has invalid cached {label}") from None


def _cached_evidence_ids(
    value: object,
    evidence_hash: object,
    question_id: str,
) -> tuple[int, ...]:
    parsed = _stored_json(value, "evidence identity", question_id)
    if (
        type(parsed) is not list
        or not parsed
        or any(type(complaint_id) is not int or complaint_id <= 0 for complaint_id in parsed)
        or len(parsed) != len(set(parsed))
    ):
        raise ValueError(f"question {question_id!r} has invalid cached evidence identity")
    canonical = json.dumps(parsed, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if evidence_hash != expected:
        raise ValueError(f"question {question_id!r} has invalid cached evidence identity")
    return tuple(parsed)


def _retrievable_evidence_text_by_id(
    con,
    question: EvalQuestion,
    complaint_ids: tuple[int, ...],
    corpus_cache: dict[tuple[str, str, str], dict[int, str]],
) -> dict[int, str]:
    if question.company_id is None:
        raise ValueError(f"question {question.question_id!r} has no concrete evidence scope")
    embed_model = _embed_model_for_cluster(con, question.question_id, question.cluster_id)
    scope_key = (question.cluster_id, question.company_id, embed_model)
    evidence_by_id = corpus_cache.get(scope_key)
    if evidence_by_id is None:
        try:
            corpus = retrieve.load_corpus(
                con,
                question.cluster_id,
                question.company_id,
                embed_model,
            )
        except ValueError:
            raise ValueError(
                f"question {question.question_id!r} has no exact retrievable evidence scope"
            ) from None
        if (
            corpus.cluster_id != question.cluster_id
            or corpus.company_id != question.company_id
            or corpus.embed_model != embed_model
        ):
            raise ValueError(
                f"question {question.question_id!r} has mismatched retrievable evidence scope"
            )
        evidence_by_id = {row.complaint_id: row.text_redacted for row in corpus.rows}
        if len(evidence_by_id) != len(corpus.rows):
            raise ValueError(
                f"question {question.question_id!r} has duplicate retrievable evidence IDs"
            )
        corpus_cache[scope_key] = evidence_by_id

    requested: dict[int, str] = {}
    for complaint_id in complaint_ids:
        text_redacted = evidence_by_id.get(complaint_id)
        if type(text_redacted) is not str or not text_redacted.strip():
            raise ValueError(
                f"question {question.question_id!r} has unavailable cited evidence "
                f"for complaint_id {complaint_id}"
            )
        requested[complaint_id] = text_redacted
    return requested


def _claim_evidence(
    con,
    question: EvalQuestion,
    complaint_ids: tuple[int, ...],
    corpus_cache: dict[tuple[str, str, str], dict[int, str]],
) -> tuple[tuple[ClaimEvidence, ...], tuple[tuple[int, str, str], ...]]:
    evidence_by_id = _retrievable_evidence_text_by_id(
        con,
        question,
        complaint_ids,
        corpus_cache,
    )
    visible: list[ClaimEvidence] = []
    identity: list[tuple[int, str, str]] = []
    for complaint_id in complaint_ids:
        text_redacted = evidence_by_id[complaint_id]
        text_hash_row = con.execute(
            "SELECT text_hash FROM narratives WHERE complaint_id = ?",
            [complaint_id],
        ).fetchone()
        if text_hash_row is None or type(text_hash_row[0]) is not str:
            raise ValueError(
                f"question {question.question_id!r} has unavailable cited evidence "
                f"for complaint_id {complaint_id}"
            )
        visible.append(ClaimEvidence(complaint_id, _review_text(text_redacted, "evidence excerpt")))
        identity.append(
            (
                complaint_id,
                text_hash_row[0],
                hashlib.sha256(text_redacted.encode("utf-8")).hexdigest(),
            )
        )
    return tuple(visible), tuple(identity)


def _review_identity(
    *,
    eval_run_id: str,
    question_id: str,
    question_hash: str,
    claim_position: int,
    claim_text: str,
    complaint_ids: tuple[int, ...],
    evidence_hash: str,
    cited_evidence_identity: tuple[tuple[int, str, str], ...],
    input_hash: str,
) -> str:
    rendered_claim = answer.render_attributed_claim(answer.Claim(claim_text, complaint_ids))
    material = json.dumps(
        {
            "version": _CLAIM_REVIEW_VERSION,
            "eval_run_id": eval_run_id,
            "question_id": question_id,
            "question_hash": question_hash,
            "claim_position": claim_position,
            "claim_text_hash": hashlib.sha256(claim_text.encode("utf-8")).hexdigest(),
            "rendered_claim_hash": hashlib.sha256(rendered_claim.encode("utf-8")).hexdigest(),
            "renderer_version": answer.ATTRIBUTED_CLAIM_RENDERER_VERSION,
            "cited_complaint_ids": list(complaint_ids),
            "evidence_hash": evidence_hash,
            "cited_evidence_identity": [list(value) for value in cited_evidence_identity],
            "input_hash": input_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _claim_candidates(con, eval_run_id: str) -> list[_ClaimCandidate]:
    manifest_path = PATHS.ground_truth / "rag_eval_questions.csv"
    try:
        questions, manifest_sha256 = load_manifest_snapshot(manifest_path)
    except FileNotFoundError:
        raise ValueError("RAG evaluation manifest is unavailable") from None
    run_identity = con.execute(
        "SELECT phase, status, "
        "json_extract_string(params_json, '$.params.manifest_sha256') "
        "FROM runs WHERE run_id = ?",
        [eval_run_id],
    ).fetchone()
    if run_identity != ("rag-eval", "ok", manifest_sha256):
        raise ValueError("evaluation run manifest identity is unavailable or does not match")
    questions_by_id = {question.question_id: question for question in questions}
    source_questions: dict[tuple[str, str, str | None], str] = {}
    for question in questions:
        source_key = (
            answer.question_hash(question.question),
            question.cluster_id,
            question.company_id,
        )
        previous = source_questions.get(source_key)
        if previous is not None:
            raise ValueError(
                f"questions {previous!r} and {question.question_id!r} share one "
                "answer usage identity"
            )
        source_questions[source_key] = question.question_id
    fused_rows = con.execute(
        "SELECT question_id FROM rag_eval_results WHERE eval_run_id = ? "
        "AND retrieval_method = 'fused' ORDER BY question_id",
        [eval_run_id],
    ).fetchall()
    fused_ids = [row[0] for row in fused_rows]
    expected_ids = sorted(questions_by_id)
    if fused_ids != expected_ids:
        raise ValueError("evaluation run fused rows do not match the frozen manifest question IDs")

    usage_rows = con.execute(
        "SELECT question_hash, cluster_id, model, prompt_version, input_hash, "
        "cache_status, outcome FROM llm_usage WHERE run_id = ? AND operation = 'answer' "
        "ORDER BY usage_id",
        [eval_run_id],
    ).fetchall()
    candidates: list[_ClaimCandidate] = []
    corpus_cache: dict[tuple[str, str, str], dict[int, str]] = {}
    for question in questions:
        question_hash = answer.question_hash(question.question)
        matching_usage = [
            row for row in usage_rows if row[0] == question_hash and row[1] == question.cluster_id
        ]
        if not matching_usage:
            raise ValueError(
                f"question {question.question_id!r} has no answer usage for this evaluation run"
            )
        cache_identities = {
            (row[2], row[3], row[4])
            for row in matching_usage
            if row[5] in {"hit", "miss"} and row[6] == "ok"
        }
        if len(cache_identities) > 1:
            raise ValueError(
                f"question {question.question_id!r} has ambiguous answer cache identities"
            )
        if not cache_identities:
            continue
        if question.company_id is None:
            raise ValueError(
                f"question {question.question_id!r} has no concrete company cache scope"
            )
        model, prompt_version, input_hash = next(iter(cache_identities))
        cache_rows = con.execute(
            "SELECT evidence_hash, evidence_ids_json, answer_json, citation_valid "
            "FROM rag_answers WHERE question_hash = ? AND cluster_id = ? "
            "AND company_id = ? AND model = ? AND prompt_version = ? "
            "ORDER BY evidence_hash",
            [
                question_hash,
                question.cluster_id,
                question.company_id,
                model,
                prompt_version,
            ],
        ).fetchall()
        linked: list[tuple[str, tuple[int, ...], answer.GroundedAnswer]] = []
        for evidence_hash, evidence_ids_json, answer_json, citation_valid in cache_rows:
            evidence_ids = _cached_evidence_ids(
                evidence_ids_json,
                evidence_hash,
                question.question_id,
            )
            candidate_input_hash = _cache_input_hash(
                question_hash=question_hash,
                evidence_hash=evidence_hash,
                cluster_id=question.cluster_id,
                company_id=question.company_id,
                model=model,
                prompt_version=prompt_version,
            )
            if candidate_input_hash != input_hash:
                continue
            if citation_valid is not True:
                raise ValueError(f"question {question.question_id!r} has an invalid cached answer")
            payload = _stored_json(answer_json, "answer", question.question_id)
            try:
                grounded = answer.validate_answer(
                    payload,
                    set(evidence_ids),
                    evidence_text_by_id=_retrievable_evidence_text_by_id(
                        con,
                        question,
                        evidence_ids,
                        corpus_cache,
                    ),
                )
            except (answer.AnswerSchemaError, TypeError, ValueError):
                raise ValueError(
                    f"question {question.question_id!r} has an invalid cached answer"
                ) from None
            linked.append((evidence_hash, evidence_ids, grounded))
        if len(linked) != 1:
            raise ValueError(
                f"question {question.question_id!r} cannot be linked unambiguously "
                "to its cached answer"
            )
        evidence_hash, _, grounded = linked[0]
        for claim_position, claim in enumerate(grounded.claims, start=1):
            cited_evidence, cited_identity = _claim_evidence(
                con,
                question,
                claim.complaint_ids,
                corpus_cache,
            )
            review_id = _review_identity(
                eval_run_id=eval_run_id,
                question_id=question.question_id,
                question_hash=question_hash,
                claim_position=claim_position,
                claim_text=claim.text,
                complaint_ids=claim.complaint_ids,
                evidence_hash=evidence_hash,
                cited_evidence_identity=cited_identity,
                input_hash=input_hash,
            )
            candidates.append(
                _ClaimCandidate(
                    review_id=review_id,
                    eval_run_id=eval_run_id,
                    question_id=question.question_id,
                    category=question.category,
                    answerable=question.answerable,
                    company_id=question.company_id,
                    question=_review_text(question.question, "question"),
                    claim_position=claim_position,
                    claim_text=_review_text(
                        answer.render_attributed_claim(claim),
                        "rendered claim",
                    ),
                    cited_complaint_ids=claim.complaint_ids,
                    cited_evidence=cited_evidence,
                )
            )
    if len({candidate.review_id for candidate in candidates}) != len(candidates):
        raise ValueError("evaluation run produced duplicate claim review identities")
    return candidates


def _claim_tie_break(seed: int, candidate: _ClaimCandidate) -> str:
    material = f"{seed}\0{candidate.review_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _sample_claims(
    candidates: list[_ClaimCandidate],
    n: int,
    seed: int,
) -> list[_ClaimCandidate]:
    remaining = list(candidates)
    selected: list[_ClaimCandidate] = []
    category_counts: Counter[str] = Counter()
    answerable_counts: Counter[bool] = Counter()
    company_counts: Counter[str] = Counter()
    position_counts: Counter[int] = Counter()
    while len(selected) < n:
        chosen = min(
            remaining,
            key=lambda candidate: (
                category_counts[candidate.category],
                answerable_counts[candidate.answerable],
                company_counts[candidate.company_id],
                position_counts[candidate.claim_position],
                _claim_tie_break(seed, candidate),
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        category_counts[chosen.category] += 1
        answerable_counts[chosen.answerable] += 1
        company_counts[chosen.company_id] += 1
        position_counts[chosen.claim_position] += 1
    return sorted(selected, key=lambda candidate: (candidate.question_id, candidate.claim_position))


def _claim_review_row(candidate: _ClaimCandidate) -> dict[str, str]:
    evidence_json = json.dumps(
        [
            {
                "complaint_id": evidence.complaint_id,
                "text_redacted": evidence.text_redacted,
            }
            for evidence in candidate.cited_evidence
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "review_id": candidate.review_id,
        "eval_run_id": candidate.eval_run_id,
        "question_id": candidate.question_id,
        "question": candidate.question,
        "claim_position": str(candidate.claim_position),
        "claim_text": candidate.claim_text,
        "cited_complaint_ids": ";".join(map(str, candidate.cited_complaint_ids)),
        "cited_evidence_json": evidence_json,
        "grounded": "",
        "failure_category": "",
        "notes": "",
    }


def export_claim_review(
    con,
    eval_run_id: str,
    n: int,
    seed: int,
    path: Path,
) -> Path:
    """Export an exact, blinded, run-bound sample of cached answer claims."""
    if type(eval_run_id) is not str or not _SAFE_ID.fullmatch(eval_run_id):
        raise ValueError("eval_run_id must be a safe non-blank identifier")
    if type(n) is not int or isinstance(n, bool) or n < CONFIG.llm.human_verify_n:
        raise ValueError(f"human groundedness requires at least {CONFIG.llm.human_verify_n} claims")
    if type(seed) is not int or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    canonical_private_path(path, PATHS.interim, create_parents=True)
    candidates = _claim_candidates(con, eval_run_id)
    if len(candidates) < n:
        raise ValueError(
            f"eligible claim population {len(candidates)} is smaller than requested {n}"
        )
    selected = _sample_claims(candidates, n, seed)
    return _write_rows(
        path,
        CLAIM_REVIEW_HEADER,
        [_claim_review_row(candidate) for candidate in selected],
        root=PATHS.interim,
    )


def _parse_review_citations(value: str, line_number: int) -> tuple[int, ...]:
    if not value:
        raise ValueError(f"line {line_number}: cited_complaint_ids is required")
    pieces = value.split(";")
    try:
        complaint_ids = tuple(int(piece) for piece in pieces)
    except ValueError:
        raise ValueError(
            f"line {line_number}: cited_complaint_ids must be canonical integers"
        ) from None
    if (
        any(complaint_id <= 0 for complaint_id in complaint_ids)
        or any(
            str(complaint_id) != piece
            for complaint_id, piece in zip(complaint_ids, pieces, strict=True)
        )
        or len(complaint_ids) != len(set(complaint_ids))
    ):
        raise ValueError(
            f"line {line_number}: cited_complaint_ids must be unique canonical integers"
        )
    return complaint_ids


def _parse_review_evidence(
    value: str,
    complaint_ids: tuple[int, ...],
    line_number: int,
) -> tuple[ClaimEvidence, ...]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        raise ValueError(f"line {line_number}: cited_evidence_json is invalid") from None
    if type(payload) is not list or len(payload) != len(complaint_ids):
        raise ValueError(f"line {line_number}: cited evidence does not match citations")
    evidence: list[ClaimEvidence] = []
    for expected_id, item in zip(complaint_ids, payload, strict=True):
        if type(item) is not dict or set(item) != {"complaint_id", "text_redacted"}:
            raise ValueError(f"line {line_number}: cited evidence fields are invalid")
        text = item["text_redacted"]
        if item["complaint_id"] != expected_id or type(text) is not str or not text.strip():
            raise ValueError(f"line {line_number}: cited evidence does not match citations")
        if _review_text(text, "evidence excerpt") != text:
            raise ValueError(f"line {line_number}: cited evidence contains unsafe text")
        evidence.append(ClaimEvidence(expected_id, text))
    return tuple(evidence)


def parse_claim_review(path: Path, reviewer_id: str) -> list[ClaimReview]:
    """Parse completed human judgments without trusting artifact identities."""
    if type(reviewer_id) is not str or not _SAFE_ID.fullmatch(reviewer_id):
        raise ValueError("reviewer_id must be a safe non-blank identifier")
    canonical_path, source_bytes = read_private_bytes(path, PATHS.interim)
    if not canonical_path.stem.endswith(f".{reviewer_id}"):
        raise ValueError("claim review filename must end with the reviewer_id")
    rows = _read_strict_csv_bytes(source_bytes, CLAIM_REVIEW_HEADER, "claim review")
    parsed: list[ClaimReview] = []
    seen_ids: set[str] = set()
    seen_claims: set[tuple[str, int]] = set()
    for line_number, row in enumerate(rows, start=2):
        review_id = row["review_id"]
        if not _HEX_DIGEST.fullmatch(review_id):
            raise ValueError(f"line {line_number}: review_id is invalid")
        if review_id in seen_ids:
            raise ValueError(f"line {line_number}: duplicate review_id")
        seen_ids.add(review_id)
        eval_run_id = _safe_identifier(row["eval_run_id"], "eval_run_id", line_number)
        question_id = _safe_identifier(row["question_id"], "question_id", line_number)
        try:
            claim_position = int(row["claim_position"])
        except ValueError:
            raise ValueError(f"line {line_number}: claim_position must be positive") from None
        if claim_position <= 0 or row["claim_position"] != str(claim_position):
            raise ValueError(f"line {line_number}: claim_position must be canonical and positive")
        claim_key = (question_id, claim_position)
        if claim_key in seen_claims:
            raise ValueError(f"line {line_number}: duplicate question claim position")
        seen_claims.add(claim_key)
        question = _required(row["question"], "question", line_number)
        claim_text = _required(row["claim_text"], "claim_text", line_number)
        if _review_text(question, "question") != question:
            raise ValueError(f"line {line_number}: question contains unsafe text")
        if _review_text(claim_text, "claim text") != claim_text:
            raise ValueError(f"line {line_number}: claim text contains unsafe text")
        complaint_ids = _parse_review_citations(row["cited_complaint_ids"], line_number)
        cited_evidence = _parse_review_evidence(
            row["cited_evidence_json"],
            complaint_ids,
            line_number,
        )
        grounded_value = row["grounded"]
        if grounded_value not in {"yes", "no"}:
            raise ValueError(f"line {line_number}: grounded must be yes or no")
        failure_category = row["failure_category"]
        if grounded_value == "yes":
            if failure_category != "none":
                raise ValueError(
                    f"line {line_number}: failure_category must be none when grounded=yes"
                )
        elif failure_category in {"", "none"}:
            raise ValueError(f"line {line_number}: failure_category is required when grounded=no")
        elif failure_category not in _CLAIM_FAILURE_CATEGORIES:
            choices = ", ".join(sorted(_CLAIM_FAILURE_CATEGORIES))
            raise ValueError(f"line {line_number}: failure_category must be one of {choices}")
        raw_notes = row["notes"].strip()
        if raw_notes and _review_text(raw_notes, "notes") != raw_notes:
            raise ValueError(f"line {line_number}: notes contain unsafe text")
        parsed.append(
            ClaimReview(
                review_id=review_id,
                eval_run_id=eval_run_id,
                question_id=question_id,
                question=question,
                claim_position=claim_position,
                claim_text=claim_text,
                cited_complaint_ids=complaint_ids,
                cited_evidence=cited_evidence,
                grounded=grounded_value == "yes",
                failure_category=failure_category,
                notes=raw_notes or None,
                reviewer_id=reviewer_id,
            )
        )
    return parsed


def _validate_constructed_review(review: ClaimReview) -> None:
    if type(review.review_id) is not str or not _HEX_DIGEST.fullmatch(review.review_id):
        raise ValueError("review_id must be a lowercase SHA-256 digest")
    for field_name, value in (
        ("eval_run_id", review.eval_run_id),
        ("question_id", review.question_id),
        ("reviewer_id", review.reviewer_id),
    ):
        if type(value) is not str or not _SAFE_ID.fullmatch(value):
            raise ValueError(f"{field_name} must be a safe non-blank identifier")
    if type(review.claim_position) is not int or review.claim_position <= 0:
        raise ValueError("claim_position must be a positive integer")
    for field_name, value in (("question", review.question), ("claim text", review.claim_text)):
        if type(value) is not str or _review_text(value, field_name) != value:
            raise ValueError(f"{field_name} contains unsafe text")
    if review.notes is not None and (
        type(review.notes) is not str or _review_text(review.notes, "notes") != review.notes
    ):
        raise ValueError("review notes contain unsafe text")
    if (
        type(review.cited_complaint_ids) is not tuple
        or not review.cited_complaint_ids
        or any(
            type(complaint_id) is not int or complaint_id <= 0
            for complaint_id in review.cited_complaint_ids
        )
        or len(review.cited_complaint_ids) != len(set(review.cited_complaint_ids))
    ):
        raise ValueError(
            "cited_complaint_ids must be a non-empty tuple of unique positive integers"
        )
    if type(review.cited_evidence) is not tuple or len(review.cited_evidence) != len(
        review.cited_complaint_ids
    ):
        raise ValueError("cited_evidence must be a tuple matching cited_complaint_ids")
    for expected_id, evidence in zip(
        review.cited_complaint_ids,
        review.cited_evidence,
        strict=True,
    ):
        if (
            type(evidence) is not ClaimEvidence
            or type(evidence.complaint_id) is not int
            or evidence.complaint_id <= 0
            or evidence.complaint_id != expected_id
        ):
            raise ValueError("cited evidence IDs must exactly match cited_complaint_ids")
        if (
            type(evidence.text_redacted) is not str
            or _review_text(evidence.text_redacted, "evidence excerpt") != evidence.text_redacted
        ):
            raise ValueError("cited evidence contains unsafe text")
    if type(review.grounded) is not bool:
        raise TypeError("grounded must be a boolean")
    if type(review.failure_category) is not str:
        raise TypeError("failure_category must be a string")
    if review.grounded and review.failure_category != "none":
        raise ValueError("failure_category must be none when grounded=yes")
    if not review.grounded and review.failure_category not in _CLAIM_FAILURE_CATEGORIES:
        raise ValueError("failure_category is required and must be valid when grounded=no")


def _validate_review_batch(eval_run_id: str, reviews: list[ClaimReview]) -> str:
    if type(eval_run_id) is not str or not _SAFE_ID.fullmatch(eval_run_id):
        raise ValueError("eval_run_id must be a safe non-blank identifier")
    if type(reviews) is not list or any(type(review) is not ClaimReview for review in reviews):
        raise TypeError("reviews must be a list of ClaimReview values")
    if len(reviews) < CONFIG.llm.human_verify_n:
        raise ValueError(
            f"human groundedness requires at least {CONFIG.llm.human_verify_n} reviews"
        )
    for review in reviews:
        _validate_constructed_review(review)
    review_ids = [review.review_id for review in reviews]
    claim_keys = [(review.question_id, review.claim_position) for review in reviews]
    if len(review_ids) != len(set(review_ids)) or len(claim_keys) != len(set(claim_keys)):
        raise ValueError("duplicate claim reviews are forbidden")
    if any(review.eval_run_id != eval_run_id for review in reviews):
        raise ValueError("review eval_run_id does not match the requested eval_run_id")
    reviewer_ids = {review.reviewer_id for review in reviews}
    if len(reviewer_ids) != 1:
        raise ValueError("claim reviews must contain exactly one reviewer_id")
    return next(iter(reviewer_ids))


def record_claim_review(
    con,
    eval_run_id: str,
    reviews: list[ClaimReview],
) -> GroundednessReport:
    """Verify and atomically persist one complete human-review artifact."""
    reviewer_id = _validate_review_batch(eval_run_id, reviews)
    _require_autocommit(con)
    candidates = {
        candidate.review_id: candidate for candidate in _claim_candidates(con, eval_run_id)
    }
    for review in reviews:
        candidate = candidates.get(review.review_id)
        if candidate is None or (
            review.eval_run_id,
            review.question_id,
            review.question,
            review.claim_position,
            review.claim_text,
            review.cited_complaint_ids,
            review.cited_evidence,
        ) != (
            candidate.eval_run_id,
            candidate.question_id,
            candidate.question,
            candidate.claim_position,
            candidate.claim_text,
            candidate.cited_complaint_ids,
            candidate.cited_evidence,
        ):
            raise ValueError("claim review identity failed closed after artifact tampering")

    counts: dict[str, list[int]] = {}
    for review in reviews:
        values = counts.setdefault(review.question_id, [0, 0])
        values[0] += int(review.grounded)
        values[1] += 1
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "UPDATE rag_eval_results SET grounded_claims = NULL, reviewed_claims = NULL "
            "WHERE eval_run_id = ? AND retrieval_method = 'fused'",
            [eval_run_id],
        )
        for question_id, (grounded, reviewed) in sorted(counts.items()):
            con.execute(
                "UPDATE rag_eval_results SET grounded_claims = ?, reviewed_claims = ? "
                "WHERE eval_run_id = ? AND question_id = ? AND retrieval_method = 'fused'",
                [grounded, reviewed, eval_run_id, question_id],
            )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise

    grounded = sum(review.grounded for review in reviews)
    reviewed = len(reviews)
    ci_low, ci_high = verify.wilson(grounded, reviewed)
    return GroundednessReport(
        grounded=grounded,
        reviewed=reviewed,
        rate=grounded / reviewed,
        ci_low=ci_low,
        ci_high=ci_high,
        reviewer_id=reviewer_id,
    )


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
    return load_manifest_bytes(read_stable_bytes(path), expected_n)


def load_manifest_bytes(data: bytes, expected_n: int = 30) -> list[EvalQuestion]:
    """Parse questions from the exact bytes used for the manifest identity."""
    rows = _read_strict_csv_bytes(data, MANIFEST_HEADER, "manifest")
    return _parse_rows(rows, expected_n)


def load_manifest_snapshot(
    path: Path,
    expected_n: int = 30,
) -> tuple[list[EvalQuestion], str]:
    """Load and hash one stable manifest byte buffer."""
    data = read_stable_bytes(path)
    return load_manifest_bytes(data, expected_n), hashlib.sha256(data).hexdigest()


def _read_strict_csv_bytes(
    data: bytes,
    header: tuple[str, ...],
    artifact: str,
) -> list[dict[str, str]]:
    """Parse exact UTF-8 CSV bytes only when every row has the header's arity."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{artifact} must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != list(header):
        raise ManifestError(f"{artifact} columns do not match the contract")
    rows: list[dict[str, str]] = []
    for line_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ManifestError(
                f"line {line_number}: {artifact} field count does not match its header"
            )
        rows.append(row)
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


def manifest_embed_model(con, questions: list[EvalQuestion]) -> str:
    """Return the one cluster-run model shared by a validated evaluation manifest."""
    if type(questions) is not list or not questions:
        raise ManifestError("evaluation manifest must contain questions")
    if any(not isinstance(question, EvalQuestion) for question in questions):
        raise TypeError("questions must contain EvalQuestion values")
    models = {
        _embed_model_for_cluster(con, question.question_id, question.cluster_id)
        for question in questions
    }
    if len(models) != 1:
        raise ManifestError("evaluation manifest must use exactly one embedding model")
    return next(iter(models))


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
        )
        SELECT sc.cluster_id, sc.company_id, sc.n_evidence, cl.product_family
        FROM scope_counts sc
        JOIN clusters cl ON cl.cluster_id = sc.cluster_id AND cl.run_id = ?
        ORDER BY sc.cluster_id, sc.company_id
        """  # noqa: S608 - interpolation is the static shared population SQL
    scope_rows = con.execute(
        query,
        [
            cluster_run,
            dedup_run,
            dedup_run,
            cutoff,
            cutoff,
            embed_model,
            cluster_run,
        ],
    ).fetchall()
    scope_counts = {
        (cluster_id, company_id): int(n_evidence)
        for cluster_id, company_id, n_evidence, _family in scope_rows
    }
    cluster_counts: Counter[str] = Counter()
    cluster_families: dict[str, str] = {}
    for cluster_id, _company_id, n_evidence, product_family in scope_rows:
        cluster_counts[cluster_id] += int(n_evidence)
        cluster_families[cluster_id] = product_family

    canonical = canonical_alert_scopes(con, signals_run)
    fired_clusters = {scope.cluster_id for scope in canonical}
    candidates: set[_AuthoringCandidate] = set()
    for scope in canonical:
        if scope.company_id == "__ALL__":
            company_id = None
            n_evidence = cluster_counts[scope.cluster_id]
        else:
            company_id = scope.company_id
            n_evidence = scope_counts.get((scope.cluster_id, scope.company_id), 0)
        if n_evidence >= _EXCERPTS_PER_ROW:
            candidates.add(
                _AuthoringCandidate(
                    scope.cluster_id,
                    company_id,
                    scope.product_family,
                    True,
                )
            )

    for cluster_id, n_evidence in cluster_counts.items():
        if n_evidence >= _EXCERPTS_PER_ROW and cluster_id not in fired_clusters:
            candidates.add(
                _AuthoringCandidate(
                    cluster_id,
                    None,
                    cluster_families[cluster_id],
                    False,
                )
            )
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.cluster_id,
            candidate.company_id or "",
            candidate.product_family,
            candidate.did_fire,
        ),
    )


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


def _write_rows(
    path: Path,
    header: tuple[str, ...],
    rows: list[dict[str, str]],
    *,
    root: Path,
) -> Path:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(header), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return atomic_write_bytes(path, root, buffer.getvalue().encode("utf-8"))


def export_authoring_worklist(con, seed: int, path: Path) -> Path:
    """Export thirty deterministic candidates with private redacted evidence."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ManifestError("seed must be an integer")
    try:
        canonical_private_path(path, PATHS.interim, create_parents=True)
    except PrivatePathError as exc:
        raise ManifestError(str(exc)) from None
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
    return _write_rows(path, AUTHORING_HEADER, rows, root=PATHS.interim)


def _prepare_authoring_bytes(source: Path, source_bytes: bytes) -> PreparedAuthoringImport:
    source_rows = _read_strict_csv_bytes(
        source_bytes,
        AUTHORING_HEADER,
        "authoring worklist",
    )
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
    questions = tuple(_parse_rows(manifest_rows, expected_n=30))
    rows = tuple(tuple(row[field] for field in MANIFEST_HEADER) for row in manifest_rows)
    return PreparedAuthoringImport(
        source=source,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_bytes=source_bytes,
        questions=questions,
        manifest_rows=rows,
    )


def prepare_authoring_import(source: Path) -> PreparedAuthoringImport:
    """Fully parse one exact private worklist before opening a writable database."""
    if not isinstance(source, Path):
        raise TypeError("source must be a Path")
    canonical_source, source_bytes = read_private_bytes(source, PATHS.interim)
    return _prepare_authoring_bytes(canonical_source, source_bytes)


def import_authoring_worklist(
    con,
    source: Path,
    destination: Path,
    *,
    preflight: PreparedAuthoringImport | None = None,
) -> Path:
    """DB-validate one preflighted byte identity and emit committed ID columns."""
    prepared = prepare_authoring_import(source) if preflight is None else preflight
    if type(prepared) is not PreparedAuthoringImport:
        raise TypeError("preflight must be a PreparedAuthoringImport")
    canonical_source, current_bytes = read_private_bytes(source, PATHS.interim)
    if prepared.source != canonical_source:
        raise ManifestError("authoring worklist path does not match its preflight")
    rebuilt = _prepare_authoring_bytes(canonical_source, prepared.source_bytes)
    if rebuilt != prepared:
        raise ManifestError("authoring worklist preflight identity is invalid")
    if (
        current_bytes != prepared.source_bytes
        or hashlib.sha256(current_bytes).hexdigest() != prepared.source_sha256
    ):
        raise ManifestError("authoring worklist changed after preflight")
    questions = list(prepared.questions)
    validate_manifest(con, questions)
    manifest_rows = [dict(zip(MANIFEST_HEADER, row, strict=True)) for row in prepared.manifest_rows]
    return _write_rows(
        destination,
        MANIFEST_HEADER,
        manifest_rows,
        root=PATHS.ground_truth,
    )
