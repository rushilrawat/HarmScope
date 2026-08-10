"""Blinded human review for descriptive LLM cluster labels.

The exported CSV intentionally excludes every signal-strength field and model
confidence.  Sampling may use those fields to obtain coverage; reviewers may
not see them.  Only after review is recorded do reports rejoin the hidden
strata for honest denominator-bearing breakdowns.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.config import CONFIG

HEADER = [
    "cluster_id", "product_family", "harm_mechanism", "actors",
    "preconditions", "consumer_impact", "dominant_taxonomy",
    "model_distinct_from_taxonomy", "model_is_likely_template",
    "narrative_1", "narrative_2", "narrative_3", "narrative_4",
    "narrative_5", "narrative_6", "narrative_7", "narrative_8",
    "narrative_9", "narrative_10", "mechanism_accuracy",
    "taxonomy_distinctness_accuracy", "template_accuracy",
    "should_have_abstained", "failure_category", "notes",
]

MECHANISM_DECISIONS = {"agree", "partial", "disagree"}
BINARY_DECISIONS = {"agree", "disagree"}
FAILURE_CATEGORIES = {
    "none", "incoherent_cluster", "overgeneralized", "overspecific",
    "missed_submechanism", "taxonomy_error", "template_error",
    "unsupported_claim", "other",
}


@dataclass(frozen=True)
class Verification:
    """One reviewer decision, parsed from a blinded worklist CSV."""

    cluster_id: str
    reviewer_id: str
    mechanism_accuracy: str
    taxonomy_distinctness_accuracy: str
    template_accuracy: str
    should_have_abstained: bool
    failure_category: str
    notes: str | None


@dataclass(frozen=True)
class Rate:
    """A rate with its actual reviewed denominator and Wilson interval."""

    successes: int
    total: int
    rate: float
    ci_low: float
    ci_high: float

    @property
    def numerator(self) -> int:
        """The number of reviewed rows satisfying the rate definition."""
        return self.successes

    @property
    def denominator(self) -> int:
        """The actual number of human-reviewed rows in this population."""
        return self.total


@dataclass(frozen=True)
class VerificationMetrics:
    """All accuracy rates reported for a review population."""

    mechanism: Rate
    mechanism_lenient: Rate
    taxonomy_distinctness: Rate
    template: Rate


@dataclass(frozen=True)
class VerificationReport:
    """Review metrics overall and across the source-only strata."""

    overall: VerificationMetrics
    by_fired_status: dict[str, VerificationMetrics]
    by_confidence: dict[str, VerificationMetrics]
    failure_categories: dict[str, int]

    @property
    def mechanism_agree(self) -> int:
        return self.overall.mechanism.successes

    @property
    def mechanism_total(self) -> int:
        return self.overall.mechanism.total

    @property
    def mechanism_rate(self) -> float:
        return self.overall.mechanism.rate

    @property
    def mechanism_ci_low(self) -> float:
        return self.overall.mechanism.ci_low

    @property
    def mechanism_ci_high(self) -> float:
        return self.overall.mechanism.ci_high


@dataclass(frozen=True)
class _Candidate:
    cluster_id: str
    product_family: str
    harm_mechanism: str | None
    actors: str | None
    preconditions: str | None
    consumer_impact: str | None
    dominant_taxonomy: str | None
    distinct_from_taxonomy: bool | None
    is_likely_template: bool | None
    did_fire: bool
    confidence: str | None


def _normalise_text(text: str | None) -> str:
    return " ".join((text or "").split())[:CONFIG.llm.max_narrative_chars]


def _strata(candidate: _Candidate) -> tuple[tuple[str, str], ...]:
    """Hidden strata used only to balance a reviewer-blinded sample."""
    return (
        ("fired_status", "fired" if candidate.did_fire else "control"),
        ("product_family", candidate.product_family or "(unknown)"),
        ("confidence", candidate.confidence or "(unknown)"),
        ("template_suspicion", str(bool(candidate.is_likely_template)).lower()),
        ("taxonomy_distinctness", str(bool(candidate.distinct_from_taxonomy)).lower()),
    )


def _tie_break(seed: int, cluster_id: str) -> str:
    return hashlib.sha256(f"{seed}{cluster_id}".encode()).hexdigest()


def _sample(candidates: list[_Candidate], n: int, seed: int) -> list[_Candidate]:
    """Greedily cover the least-represented value of every hidden stratum."""
    remaining = list(candidates)
    selected: list[_Candidate] = []
    coverage: Counter[tuple[str, str]] = Counter()

    while remaining and len(selected) < n:
        def score(candidate: _Candidate) -> int:
            # A candidate is valuable once for every one of its values that is
            # currently least represented within that stratum dimension.
            value_score = 0
            for dimension, value in _strata(candidate):
                dimension_counts = [
                    coverage[(dimension, other_value)]
                    for other_dimension, other_value in (
                        stratum for row in candidates for stratum in _strata(row)
                    )
                    if other_dimension == dimension
                ]
                if coverage[(dimension, value)] == min(dimension_counts):
                    value_score += 1
            return value_score

        best_score = max(score(candidate) for candidate in remaining)
        tied = [candidate for candidate in remaining if score(candidate) == best_score]
        chosen = min(tied, key=lambda candidate: _tie_break(seed, candidate.cluster_id))
        selected.append(chosen)
        coverage.update(_strata(chosen))
        remaining.remove(chosen)
    return selected


def _candidates(con, signals_run: str) -> list[_Candidate]:
    rows = con.execute(
        """
        SELECT l.cluster_id, c.product_family, l.harm_mechanism, l.actors,
               l.preconditions, l.consumer_impact, n.dominant_label,
               l.distinct_from_taxonomy, l.is_likely_template,
               EXISTS (
                   SELECT 1 FROM signals s
                   WHERE s.run_id = ? AND s.cluster_id = l.cluster_id
                     AND s.q_value <= ?
               ) AS did_fire,
               l.confidence
        FROM cluster_labels l
        JOIN clusters c USING (cluster_id)
        LEFT JOIN cluster_novelty n USING (cluster_id)
        WHERE (
            SELECT count(*) FROM cluster_members m WHERE m.cluster_id = l.cluster_id
        ) >= 10
        ORDER BY l.cluster_id
        """,
        [signals_run, CONFIG.signals.fdr_alpha],
    ).fetchall()
    return [_Candidate(*row) for row in rows]


def _narratives(con, cluster_id: str) -> list[str]:
    rows = con.execute(
        """
        SELECT n.text_redacted
        FROM cluster_members m
        JOIN narratives n USING (complaint_id)
        WHERE m.cluster_id = ?
        ORDER BY m.complaint_id
        LIMIT 10
        """,
        [cluster_id],
    ).fetchall()
    return [_normalise_text(text) for (text,) in rows]


def export_worklist(con, signals_run: str, n: int, seed: int, path: Path) -> Path:
    """Write a deterministic, stratified but signal-blinded CSV worklist."""
    if n < 0:
        raise ValueError("n must be non-negative")
    selected = _sample(_candidates(con, signals_run), n, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, extrasaction="raise")
        writer.writeheader()
        for candidate in selected:
            narratives = _narratives(con, candidate.cluster_id)
            row = {
                "cluster_id": candidate.cluster_id,
                "product_family": candidate.product_family,
                "harm_mechanism": _normalise_text(candidate.harm_mechanism),
                "actors": _normalise_text(candidate.actors),
                "preconditions": _normalise_text(candidate.preconditions),
                "consumer_impact": _normalise_text(candidate.consumer_impact),
                "dominant_taxonomy": _normalise_text(candidate.dominant_taxonomy),
                "model_distinct_from_taxonomy": str(
                    bool(candidate.distinct_from_taxonomy)
                ).lower(),
                "model_is_likely_template": str(
                    bool(candidate.is_likely_template)
                ).lower(),
                "mechanism_accuracy": "",
                "taxonomy_distinctness_accuracy": "",
                "template_accuracy": "",
                "should_have_abstained": "",
                "failure_category": "",
                "notes": "",
            }
            row.update({f"narrative_{index}": text for index, text in enumerate(narratives, 1)})
            writer.writerow(row)
    return path


def _decision(row: dict[str, str], field: str, allowed: set[str]) -> str:
    value = row.get(field, "").strip().lower()
    if value not in allowed:
        choices = (
            "agree, partial, disagree"
            if allowed == MECHANISM_DECISIONS
            else "agree, disagree"
        )
        raise ValueError(f"{field} must be one of {choices}")
    return value


def _boolean(row: dict[str, str], field: str) -> bool:
    value = row.get(field, "").strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{field} must be true or false")


def _failure_category(row: dict[str, str], fully_agrees: bool) -> str:
    value = row.get("failure_category", "").strip().lower()
    if fully_agrees:
        if value != "none":
            raise ValueError("failure_category must be none when all labels agree")
        return value
    if value == "none" or not value:
        raise ValueError("failure_category is required when a label does not agree")
    if value not in FAILURE_CATEGORIES:
        choices = ", ".join(sorted(FAILURE_CATEGORIES - {"none"}))
        raise ValueError(f"failure_category must be one of {choices}")
    return value


def parse_worklist(path: Path, reviewer_id: str) -> list[Verification]:
    """Parse completed reviewer decisions with a deliberately strict contract."""
    if not reviewer_id.strip():
        raise ValueError("reviewer_id is required")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADER:
            raise ValueError("worklist columns do not match the verification contract")
        parsed: list[Verification] = []
        for line_number, row in enumerate(reader, start=2):
            cluster_id = row.get("cluster_id", "").strip()
            if not cluster_id:
                raise ValueError(f"line {line_number}: cluster_id is required")
            mechanism = _decision(row, "mechanism_accuracy", MECHANISM_DECISIONS)
            taxonomy = _decision(
                row, "taxonomy_distinctness_accuracy", BINARY_DECISIONS,
            )
            template = _decision(row, "template_accuracy", BINARY_DECISIONS)
            abstained = _boolean(row, "should_have_abstained")
            category = _failure_category(
                row,
                mechanism == taxonomy == template == "agree" and not abstained,
            )
            notes = row.get("notes", "").strip() or None
            parsed.append(Verification(
                cluster_id, reviewer_id, mechanism, taxonomy, template, abstained,
                category, notes,
            ))
    return parsed


def _is_fired(con, cluster_id: str, signals_run: str) -> bool:
    return bool(con.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM signals
            WHERE run_id = ? AND cluster_id = ? AND q_value <= ?
        )
        """,
        [signals_run, cluster_id, CONFIG.signals.fdr_alpha],
    ).fetchone()[0])


def record(
    con, rows: list[Verification], signals_run: str, worklist_version: str,
) -> int:
    """Persist completed reviews, deriving hidden signal status after review."""
    if not worklist_version.strip():
        raise ValueError("worklist_version is required")
    con.execute("BEGIN TRANSACTION")
    try:
        for row in rows:
            labelled = con.execute(
                "SELECT 1 FROM cluster_labels WHERE cluster_id = ?", [row.cluster_id],
            ).fetchone()
            if labelled is None:
                raise ValueError(f"unknown labelled cluster_id: {row.cluster_id}")
            con.execute(
                "INSERT OR REPLACE INTO label_verifications "
                "(cluster_id, reviewer_id, worklist_version, signals_run, is_fired, "
                "mechanism_accuracy, taxonomy_distinctness_accuracy, template_accuracy, "
                "should_have_abstained, failure_category, notes, reviewed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    row.cluster_id, row.reviewer_id, worklist_version, signals_run,
                    _is_fired(con, row.cluster_id, signals_run), row.mechanism_accuracy,
                    row.taxonomy_distinctness_accuracy, row.template_accuracy,
                    row.should_have_abstained, row.failure_category, row.notes,
                    datetime.now(),
                ],
            )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return len(rows)


def wilson(
    successes: int, total: int, z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Two-sided Wilson score interval, including an explicit empty sample."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    centre = (proportion + z_squared / (2 * total)) / denominator
    margin = z * (
        (proportion * (1 - proportion) / total + z_squared / (4 * total * total))
        ** 0.5
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _rate(successes: int, total: int) -> Rate:
    low, high = wilson(successes, total)
    return Rate(successes, total, successes / total if total else 0.0, low, high)


def _metrics(rows: list[tuple]) -> VerificationMetrics:
    total = len(rows)
    mechanism = sum(row[0] == "agree" for row in rows)
    lenient = sum(row[0] in {"agree", "partial"} for row in rows)
    taxonomy = sum(row[1] == "agree" for row in rows)
    template = sum(row[2] == "agree" for row in rows)
    return VerificationMetrics(
        _rate(mechanism, total), _rate(lenient, total), _rate(taxonomy, total),
        _rate(template, total),
    )


def report(con, worklist_version: str | None = None) -> VerificationReport:
    """Report overall and hidden-stratum agreement from recorded human reviews."""
    rows = con.execute(
        """
        SELECT v.mechanism_accuracy, v.taxonomy_distinctness_accuracy,
               v.template_accuracy, v.failure_category,
               CASE WHEN EXISTS (
                   SELECT 1 FROM signals s
                   WHERE s.run_id = v.signals_run AND s.cluster_id = v.cluster_id
                     AND s.q_value <= ?
               ) THEN 'fired' ELSE 'control' END AS fired_status,
               coalesce(l.confidence, '(unknown)') AS confidence
        FROM label_verifications v
        JOIN cluster_labels l USING (cluster_id)
        WHERE ? IS NULL OR v.worklist_version = ?
        ORDER BY v.cluster_id, v.reviewer_id, v.worklist_version
        """,
        [CONFIG.signals.fdr_alpha, worklist_version, worklist_version],
    ).fetchall()
    metric_rows = [(row[0], row[1], row[2]) for row in rows]
    by_fired: dict[str, list[tuple]] = defaultdict(list)
    by_confidence: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        by_fired[row[4]].append((row[0], row[1], row[2]))
        by_confidence[row[5]].append((row[0], row[1], row[2]))
    return VerificationReport(
        overall=_metrics(metric_rows),
        by_fired_status={
            status: _metrics(by_fired[status]) for status in ("fired", "control")
        },
        by_confidence={key: _metrics(value) for key, value in sorted(by_confidence.items())},
        failure_categories=dict(sorted(Counter(row[3] for row in rows).items())),
    )
