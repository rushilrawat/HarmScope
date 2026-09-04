"""Blinded human review for descriptive LLM cluster labels.

The exported CSV intentionally excludes every signal-strength field and model
confidence.  Sampling may use those fields to obtain coverage; reviewers may
not see them.  Only after review is recorded do reports rejoin the hidden
strata for honest denominator-bearing breakdowns.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from pathlib import Path

from src.alert_scope import canonical_fired_cluster_ids
from src.config import CONFIG, PATHS
from src.private_artifacts import (
    atomic_write_bytes,
    canonical_private_path,
    read_private_bytes,
    unlink_private_file,
)
from src.text_safety import sanitize_display_text

HEADER = [
    "cluster_id",
    "product_family",
    "harm_mechanism",
    "actors",
    "preconditions",
    "consumer_impact",
    "dominant_taxonomy",
    "model_distinct_from_taxonomy",
    "model_is_likely_template",
    "narrative_1",
    "narrative_2",
    "narrative_3",
    "narrative_4",
    "narrative_5",
    "narrative_6",
    "narrative_7",
    "narrative_8",
    "narrative_9",
    "narrative_10",
    "mechanism_accuracy",
    "taxonomy_distinctness_accuracy",
    "template_accuracy",
    "should_have_abstained",
    "failure_category",
    "notes",
]

MECHANISM_DECISIONS = {"agree", "partial", "disagree"}
BINARY_DECISIONS = {"agree", "disagree"}
FAILURE_CATEGORIES = {
    "none",
    "incoherent_cluster",
    "overgeneralized",
    "overspecific",
    "missed_submechanism",
    "taxonomy_error",
    "template_error",
    "unsupported_claim",
    "other",
}
REVIEWER_ORIGINS = {"human", "model"}
REVIEW_FIELDS = frozenset(
    {
        "mechanism_accuracy",
        "taxonomy_distinctness_accuracy",
        "template_accuracy",
        "should_have_abstained",
        "failure_category",
        "notes",
    }
)


class VerificationTransactionError(RuntimeError):
    """The caller already owns a transaction required by review recording."""


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
    reviewer_origin: str = "human"


@dataclass(frozen=True)
class WorklistMetadata:
    """Non-reviewer-facing immutable provenance for one exported worklist."""

    signals_run: str
    cluster_run: str
    model: str
    prompt_version: str
    seed: int
    artifact_name: str
    worklist_version: str
    cluster_ids: tuple[str, ...]
    fired_cluster_ids: tuple[str, ...]
    source_sha256: str
    label_input_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PreparedWorklistRecord:
    """One exact private CSV/sidecar identity parsed before writable DB access."""

    path: Path
    sidecar_path: Path
    metadata: WorklistMetadata
    source_bytes: bytes = dataclass_field(repr=False)
    sidecar_bytes: bytes = dataclass_field(repr=False)
    rows: tuple[Verification, ...] = dataclass_field(repr=False)


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
    unique_reviewed_clusters: int
    gate_eligible: bool
    gate_passed: bool

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

    def render(self) -> str:
        """Render denominator-bearing human-review metrics for the pipeline CLI."""

        def rate(name: str, value: Rate) -> str:
            return (
                f"  {name:<24} {value.successes}/{value.total} "
                f"({value.rate:.1%}; Wilson {value.ci_low:.1%}..{value.ci_high:.1%})"
            )

        def metrics(title: str, value: VerificationMetrics) -> list[str]:
            return [
                title,
                rate("mechanism", value.mechanism),
                rate("mechanism (lenient)", value.mechanism_lenient),
                rate("taxonomy distinctness", value.taxonomy_distinctness),
                rate("template", value.template),
            ]

        if self.gate_eligible:
            gate = "PASS" if self.gate_passed else "FAIL"
            lines = [
                f"human-review gate: {gate} "
                f"({self.unique_reviewed_clusters}/{CONFIG.llm.human_verify_n} "
                "unique clusters in this worklist version)"
            ]
        else:
            lines = [
                "human-review gate: NOT GATE ELIGIBLE "
                "(unscoped aggregate; specify one worklist version)"
            ]
        lines.extend(metrics("human label verification", self.overall))
        lines.append("fired/control breakdown")
        for status in ("fired", "control"):
            lines.extend(metrics(f"  {status}", self.by_fired_status[status]))
        lines.append("confidence breakdown")
        for confidence, value in self.by_confidence.items():
            lines.extend(metrics(f"  {confidence}", value))
        lines.append("failure categories")
        if self.failure_categories:
            lines.extend(
                f"  {category:<24} {count}" for category, count in self.failure_categories.items()
            )
        else:
            lines.append("  (none recorded)")
        return "\n".join(lines)


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


SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@")


def spreadsheet_safe_text(text: str | None) -> str:
    """Normalize, bound, and neutralize an untrusted spreadsheet text cell."""
    normalized = sanitize_display_text(text)
    if normalized.startswith(SPREADSHEET_FORMULA_PREFIXES):
        normalized = "'" + normalized
    return normalized[: CONFIG.llm.max_narrative_chars]


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


def _cluster_run_for(con, signals_run: str) -> str:
    """Return the cluster refit that the selected signals run actually scored."""
    row = con.execute(
        """
        SELECT json_extract_string(params_json, '$.params.cluster_run')
        FROM runs WHERE run_id = ? AND phase = 'signals' AND status = 'ok'
        """,
        [signals_run],
    ).fetchone()
    if row is None or not row[0]:
        raise ValueError(
            f"signals run {signals_run!r} is not a successful signals run with cluster provenance"
        )
    return row[0]


def _candidates(con, signals_run: str) -> list[_Candidate]:
    cluster_run = _cluster_run_for(con, signals_run)
    fired_ids = canonical_fired_cluster_ids(con, signals_run)
    rows = con.execute(
        """
        SELECT l.cluster_id, c.product_family, l.harm_mechanism, l.actors,
               l.preconditions, l.consumer_impact, n.dominant_label,
               l.distinct_from_taxonomy, l.is_likely_template,
               l.confidence
        FROM cluster_labels l
        JOIN clusters c USING (cluster_id)
        LEFT JOIN cluster_novelty n USING (cluster_id)
        WHERE c.run_id = ? AND (
            SELECT count(*)
            FROM cluster_members m
            JOIN narratives review_n USING (complaint_id)
            WHERE m.cluster_id = l.cluster_id
              AND length(trim(review_n.text_redacted)) > 0
        ) >= 10 AND l.model = ? AND l.prompt_version = ?
        ORDER BY l.cluster_id
        """,
        [
            cluster_run,
            CONFIG.llm.model,
            CONFIG.llm.prompt_version,
        ],
    ).fetchall()
    return [_Candidate(*row[:9], row[0] in fired_ids, row[9]) for row in rows]


def _narratives(con, cluster_id: str) -> list[str]:
    rows = con.execute(
        """
        SELECT n.text_redacted
        FROM cluster_members m
        JOIN narratives n USING (complaint_id)
        WHERE m.cluster_id = ? AND length(trim(n.text_redacted)) > 0
        ORDER BY m.complaint_id
        LIMIT 10
        """,
        [cluster_id],
    ).fetchall()
    return [spreadsheet_safe_text(text) for (text,) in rows]


def worklist_sidecar_path(path: Path) -> Path:
    """Return the non-reviewer-facing provenance file beside a CSV worklist."""
    return path.with_name(path.name + ".metadata.json")


def _strict_rows(data: bytes, artifact: str) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{artifact} must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != HEADER:
        raise ValueError("worklist columns do not match the verification contract")
    rows: list[dict[str, str]] = []
    for line_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(
                f"line {line_number}: {artifact} field count does not match its header"
            )
        rows.append(row)
    return rows


def _immutable_source_sha256(rows: list[dict[str, str]]) -> str:
    material = [
        {field: row[field] for field in HEADER if field not in REVIEW_FIELDS} for row in rows
    ]
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=HEADER, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _worklist_version(
    signals_run: str,
    cluster_run: str,
    model: str,
    prompt_version: str,
    seed: int,
    artifact_name: str,
    cluster_ids: tuple[str, ...],
    fired_cluster_ids: tuple[str, ...],
    source_sha256: str,
    label_input_hashes: tuple[tuple[str, str], ...],
) -> str:
    material = json.dumps(
        {
            "signals_run": signals_run,
            "cluster_run": cluster_run,
            "model": model,
            "prompt_version": prompt_version,
            "seed": seed,
            "artifact_name": artifact_name,
            "cluster_ids": list(cluster_ids),
            "fired_cluster_ids": list(fired_cluster_ids),
            "source_sha256": source_sha256,
            "label_input_hashes": [list(value) for value in label_input_hashes],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _metadata(
    signals_run: str,
    cluster_run: str,
    model: str,
    prompt_version: str,
    seed: int,
    artifact_name: str,
    cluster_ids: list[str],
    fired_cluster_ids: list[str],
    source_sha256: str,
    label_input_hashes: list[tuple[str, str]],
) -> WorklistMetadata:
    canonical_ids = tuple(sorted(set(cluster_ids)))
    canonical_fired_ids = tuple(sorted(set(fired_cluster_ids)))
    if not set(canonical_fired_ids).issubset(canonical_ids):
        raise ValueError("worklist fired-cluster snapshot must be within the worklist population")
    canonical_hashes = tuple(sorted(label_input_hashes))
    return WorklistMetadata(
        signals_run=signals_run,
        cluster_run=cluster_run,
        model=model,
        prompt_version=prompt_version,
        seed=seed,
        artifact_name=artifact_name,
        worklist_version=_worklist_version(
            signals_run,
            cluster_run,
            model,
            prompt_version,
            seed,
            artifact_name,
            canonical_ids,
            canonical_fired_ids,
            source_sha256,
            canonical_hashes,
        ),
        cluster_ids=canonical_ids,
        fired_cluster_ids=canonical_fired_ids,
        source_sha256=source_sha256,
        label_input_hashes=canonical_hashes,
    )


def _metadata_payload(metadata: WorklistMetadata) -> dict[str, object]:
    return {
        "signals_run": metadata.signals_run,
        "cluster_run": metadata.cluster_run,
        "model": metadata.model,
        "prompt_version": metadata.prompt_version,
        "seed": metadata.seed,
        "artifact_name": metadata.artifact_name,
        "worklist_version": metadata.worklist_version,
        "cluster_ids": list(metadata.cluster_ids),
        "fired_cluster_ids": list(metadata.fired_cluster_ids),
        "source_sha256": metadata.source_sha256,
        "label_input_hashes": [list(value) for value in metadata.label_input_hashes],
    }


def _write_worklist_metadata(path: Path, metadata: WorklistMetadata) -> None:
    payload = json.dumps(_metadata_payload(metadata), indent=2, sort_keys=True) + "\n"
    atomic_write_bytes(
        worklist_sidecar_path(path),
        PATHS.interim,
        payload.encode("utf-8"),
    )


def _validate_metadata_labels(con, metadata: WorklistMetadata) -> None:
    rows = con.execute(
        "SELECT l.cluster_id, c.run_id, l.model, l.prompt_version, l.input_hash "
        "FROM cluster_labels l JOIN clusters c USING (cluster_id) "
        "WHERE c.run_id = ?",
        [metadata.cluster_run],
    ).fetchall()
    labels = {row[0]: row[1:] for row in rows}
    for cluster_id in metadata.cluster_ids:
        source = labels.get(cluster_id)
        if source is None:
            raise ValueError(f"sidecar cluster {cluster_id!r} is not currently labelled")
        expected_hash = dict(metadata.label_input_hashes).get(cluster_id)
        if source != (
            metadata.cluster_run,
            metadata.model,
            metadata.prompt_version,
            expected_hash,
        ):
            raise ValueError(f"label provenance changed for sidecar cluster {cluster_id!r}")


def _metadata_from_bytes(
    csv_bytes: bytes,
    sidecar_bytes: bytes,
) -> WorklistMetadata:
    """Validate one exact reviewer-visible CSV and sidecar byte pair."""
    try:
        payload = json.loads(sidecar_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("worklist provenance sidecar is invalid JSON") from exc
    required = {
        "signals_run",
        "cluster_run",
        "model",
        "prompt_version",
        "seed",
        "artifact_name",
        "worklist_version",
        "cluster_ids",
        "fired_cluster_ids",
        "source_sha256",
        "label_input_hashes",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("worklist provenance sidecar fields do not match the contract")
    for field in (
        "signals_run",
        "cluster_run",
        "model",
        "prompt_version",
        "artifact_name",
    ):
        if type(payload[field]) is not str or not payload[field].strip():
            raise ValueError(f"worklist provenance sidecar {field} is invalid")
    if Path(payload["artifact_name"]).name != payload["artifact_name"]:
        raise ValueError("worklist provenance artifact filename is invalid")
    cluster_ids = payload["cluster_ids"]
    if (
        not isinstance(cluster_ids, list)
        or not all(isinstance(cluster_id, str) and cluster_id for cluster_id in cluster_ids)
        or cluster_ids != sorted(set(cluster_ids))
    ):
        raise ValueError("worklist provenance sidecar cluster_ids are not canonical")
    fired_cluster_ids = payload["fired_cluster_ids"]
    if (
        not isinstance(fired_cluster_ids, list)
        or not all(isinstance(cluster_id, str) and cluster_id for cluster_id in fired_cluster_ids)
        or fired_cluster_ids != sorted(set(fired_cluster_ids))
        or not set(fired_cluster_ids).issubset(cluster_ids)
    ):
        raise ValueError("worklist provenance fired-cluster snapshot is invalid")
    if type(payload["seed"]) is not int:
        raise ValueError("worklist provenance sidecar seed must be an integer")
    label_input_hashes = payload["label_input_hashes"]
    if (
        type(label_input_hashes) is not list
        or any(
            type(value) is not list
            or len(value) != 2
            or type(value[0]) is not str
            or not value[0]
            or type(value[1]) is not str
            or not value[1]
            for value in label_input_hashes
        )
        or label_input_hashes != sorted(label_input_hashes)
        or [value[0] for value in label_input_hashes] != cluster_ids
    ):
        raise ValueError("worklist provenance label input hashes are invalid")
    source_sha256 = payload["source_sha256"]
    if (
        type(source_sha256) is not str
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("worklist provenance source digest is invalid")
    metadata = _metadata(
        payload["signals_run"],
        payload["cluster_run"],
        payload["model"],
        payload["prompt_version"],
        payload["seed"],
        payload["artifact_name"],
        cluster_ids,
        fired_cluster_ids,
        source_sha256,
        [tuple(value) for value in label_input_hashes],
    )
    if payload["worklist_version"] != metadata.worklist_version:
        raise ValueError("worklist provenance sidecar digest does not match its contents")
    rows = _strict_rows(csv_bytes, "worklist")
    csv_ids = [row["cluster_id"].strip() for row in rows]
    if len(csv_ids) != len(set(csv_ids)) or sorted(csv_ids) != list(metadata.cluster_ids):
        raise ValueError("worklist CSV cluster IDs do not match the provenance sidecar")
    if _immutable_source_sha256(rows) != metadata.source_sha256:
        raise ValueError("worklist reviewer-visible source was tampered after export")
    return metadata


def _read_worklist_artifact(path: Path) -> tuple[Path, Path, bytes, bytes]:
    sidecar = worklist_sidecar_path(path)
    try:
        canonical_path, csv_bytes = read_private_bytes(path, PATHS.interim)
        canonical_sidecar, sidecar_bytes = read_private_bytes(sidecar, PATHS.interim)
    except FileNotFoundError as exc:
        raise ValueError("worklist CSV or provenance sidecar is missing") from exc
    if canonical_sidecar != worklist_sidecar_path(canonical_path):
        raise ValueError("worklist provenance sidecar path does not match the CSV")
    return canonical_path, canonical_sidecar, csv_bytes, sidecar_bytes


def load_worklist_metadata(con, path: Path) -> WorklistMetadata:
    """Load and validate a CSV's immutable provenance sidecar against the DB."""
    canonical_path, _sidecar, csv_bytes, sidecar_bytes = _read_worklist_artifact(path)
    metadata = _metadata_from_bytes(csv_bytes, sidecar_bytes)
    if canonical_path.name != metadata.artifact_name:
        raise ValueError("worklist artifact filename changed after export")
    actual_cluster_run = _cluster_run_for(con, metadata.signals_run)
    if actual_cluster_run != metadata.cluster_run:
        raise ValueError("signals-run cluster provenance changed since worklist export")
    _validate_metadata_labels(con, metadata)
    return metadata


def export_worklist(con, signals_run: str, n: int, seed: int, path: Path) -> Path:
    """Write a deterministic, stratified but signal-blinded CSV worklist."""
    canonical_path = canonical_private_path(path, PATHS.interim)
    if n < CONFIG.llm.human_verify_n:
        raise ValueError(f"human verification requires at least {CONFIG.llm.human_verify_n} rows")
    candidates = _candidates(con, signals_run)
    if len(candidates) < n:
        raise ValueError(f"eligible population {len(candidates)} is smaller than requested {n}")
    selected = _sample(candidates, n, seed)
    cluster_run = _cluster_run_for(con, signals_run)
    rows: list[dict[str, str]] = []
    for candidate in selected:
        narratives = _narratives(con, candidate.cluster_id)
        row = {
            "cluster_id": candidate.cluster_id,
            "product_family": spreadsheet_safe_text(candidate.product_family),
            "harm_mechanism": spreadsheet_safe_text(candidate.harm_mechanism),
            "actors": spreadsheet_safe_text(candidate.actors),
            "preconditions": spreadsheet_safe_text(candidate.preconditions),
            "consumer_impact": spreadsheet_safe_text(candidate.consumer_impact),
            "dominant_taxonomy": spreadsheet_safe_text(candidate.dominant_taxonomy),
            "model_distinct_from_taxonomy": str(bool(candidate.distinct_from_taxonomy)).lower(),
            "model_is_likely_template": str(bool(candidate.is_likely_template)).lower(),
            "mechanism_accuracy": "",
            "taxonomy_distinctness_accuracy": "",
            "template_accuracy": "",
            "should_have_abstained": "",
            "failure_category": "",
            "notes": "",
        }
        row.update({f"narrative_{index}": text for index, text in enumerate(narratives, 1)})
        row.update(
            {
                field: spreadsheet_safe_text(value)
                for field, value in row.items()
                if field != "cluster_id" and isinstance(value, str)
            }
        )
        rows.append(row)
    source_sha256 = _immutable_source_sha256(rows)
    label_hashes = con.execute(
        "SELECT cluster_id, input_hash FROM cluster_labels "
        "WHERE cluster_id IN (SELECT unnest(?)) ORDER BY cluster_id",
        [[candidate.cluster_id for candidate in selected]],
    ).fetchall()
    metadata = _metadata(
        signals_run,
        cluster_run,
        CONFIG.llm.model,
        CONFIG.llm.prompt_version,
        seed,
        canonical_path.name,
        [candidate.cluster_id for candidate in selected],
        [candidate.cluster_id for candidate in selected if candidate.did_fire],
        source_sha256,
        label_hashes,
    )
    old_csv: bytes | None = None
    old_sidecar: bytes | None = None
    with suppress(FileNotFoundError):
        _old_path, old_csv = read_private_bytes(canonical_path, PATHS.interim)
    with suppress(FileNotFoundError):
        _old_sidecar_path, old_sidecar = read_private_bytes(
            worklist_sidecar_path(canonical_path),
            PATHS.interim,
        )
    try:
        # The sidecar is staged first; the reviewer-visible CSV is the pair's
        # commit point. A crash before that point leaves no newly reusable CSV.
        _write_worklist_metadata(canonical_path, metadata)
        canonical_path = atomic_write_bytes(canonical_path, PATHS.interim, _csv_bytes(rows))
    except BaseException as write_error:
        restoration_errors: list[str] = []
        for target, previous in (
            (canonical_path, old_csv),
            (worklist_sidecar_path(canonical_path), old_sidecar),
        ):
            try:
                if previous is None:
                    unlink_private_file(target, PATHS.interim, missing_ok=True)
                else:
                    try:
                        _current_path, current = read_private_bytes(target, PATHS.interim)
                    except FileNotFoundError:
                        current = None
                    if current != previous:
                        atomic_write_bytes(target, PATHS.interim, previous)
            except Exception as restoration_error:
                restoration_errors.append(type(restoration_error).__name__)
        if restoration_errors:
            write_error.add_note(
                "private worklist restoration also failed: " + ", ".join(restoration_errors)
            )
        raise
    return canonical_path


def _decision(row: dict[str, str], field: str, allowed: set[str]) -> str:
    value = row.get(field, "").strip().lower()
    if value not in allowed:
        choices = (
            "agree, partial, disagree" if allowed == MECHANISM_DECISIONS else "agree, disagree"
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


def _reviewer_origin(reviewer_origin: str) -> str:
    if type(reviewer_origin) is not str:
        raise TypeError("reviewer_origin must be a string")
    origin = reviewer_origin.strip().lower()
    if origin not in REVIEWER_ORIGINS:
        raise ValueError("reviewer_origin must be human or model")
    return origin


def _parse_worklist_bytes(
    data: bytes,
    reviewer_id: str,
    reviewer_origin: str,
) -> list[Verification]:
    if type(reviewer_id) is not str or not reviewer_id.strip():
        raise ValueError("reviewer_id is required")
    origin = _reviewer_origin(reviewer_origin)
    rows = _strict_rows(data, "worklist")
    parsed: list[Verification] = []
    seen_cluster_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        cluster_id = row["cluster_id"].strip()
        if not cluster_id:
            raise ValueError(f"line {line_number}: cluster_id is required")
        if cluster_id in seen_cluster_ids:
            raise ValueError(f"line {line_number}: duplicate cluster_id {cluster_id!r}")
        seen_cluster_ids.add(cluster_id)
        mechanism = _decision(row, "mechanism_accuracy", MECHANISM_DECISIONS)
        taxonomy = _decision(
            row,
            "taxonomy_distinctness_accuracy",
            BINARY_DECISIONS,
        )
        template = _decision(row, "template_accuracy", BINARY_DECISIONS)
        abstained = _boolean(row, "should_have_abstained")
        category = _failure_category(
            row,
            mechanism == taxonomy == template == "agree" and not abstained,
        )
        notes = row["notes"].strip() or None
        verification = Verification(
            cluster_id,
            reviewer_id,
            mechanism,
            taxonomy,
            template,
            abstained,
            category,
            notes,
            origin,
        )
        _validate_verification(verification)
        parsed.append(verification)
    return parsed


def parse_worklist(
    path: Path,
    reviewer_id: str,
    reviewer_origin: str = "human",
) -> list[Verification]:
    """Parse one stable private reviewer artifact under data/interim."""
    _canonical_path, data = read_private_bytes(path, PATHS.interim)
    return _parse_worklist_bytes(data, reviewer_id, reviewer_origin)


def _is_fired(con, cluster_id: str, signals_run: str) -> bool:
    return cluster_id in canonical_fired_cluster_ids(con, signals_run)


def _require_autocommit(con) -> None:
    first_id = con.execute("SELECT current_transaction_id()").fetchone()[0]
    second_id = con.execute("SELECT current_transaction_id()").fetchone()[0]
    if first_id == second_id:
        raise VerificationTransactionError(
            "verification recording requires an autocommit connection"
        )


def _validate_verification(row: Verification) -> None:
    if type(row) is not Verification:
        raise TypeError("rows must contain Verification values")
    if type(row.cluster_id) is not str or not row.cluster_id.strip():
        raise ValueError("cluster_id is required")
    if type(row.reviewer_id) is not str or not row.reviewer_id.strip():
        raise ValueError("reviewer_id is required")
    if row.mechanism_accuracy not in MECHANISM_DECISIONS:
        raise ValueError("mechanism_accuracy is invalid")
    if row.taxonomy_distinctness_accuracy not in BINARY_DECISIONS:
        raise ValueError("taxonomy_distinctness_accuracy is invalid")
    if row.template_accuracy not in BINARY_DECISIONS:
        raise ValueError("template_accuracy is invalid")
    if type(row.should_have_abstained) is not bool:
        raise TypeError("should_have_abstained must be boolean")
    fully_agrees = (
        row.mechanism_accuracy
        == row.taxonomy_distinctness_accuracy
        == row.template_accuracy
        == "agree"
        and not row.should_have_abstained
    )
    if fully_agrees and row.failure_category != "none":
        raise ValueError("failure_category must be none when all labels agree")
    if not fully_agrees and row.failure_category not in FAILURE_CATEGORIES - {"none"}:
        raise ValueError("failure_category is required when a label does not agree")
    if row.notes is not None and (
        type(row.notes) is not str or spreadsheet_safe_text(row.notes) != row.notes
    ):
        raise ValueError("notes must be normalized, bounded, and spreadsheet-safe")
    _reviewer_origin(row.reviewer_origin)


def record(
    con,
    rows: list[Verification],
    signals_run: str,
    worklist_version: str,
    *,
    provenance: WorklistMetadata | None = None,
) -> int:
    """Persist completed reviews, deriving hidden signal status after review."""
    if not worklist_version.strip():
        raise ValueError("worklist_version is required")
    if type(rows) is not list:
        raise TypeError("rows must be a list")
    for row in rows:
        _validate_verification(row)
    if provenance is not None and (
        signals_run != provenance.signals_run or worklist_version != provenance.worklist_version
    ):
        raise ValueError("record provenance does not match the worklist sidecar")
    _require_autocommit(con)
    con.execute("BEGIN TRANSACTION")
    try:
        cluster_run = _cluster_run_for(con, signals_run)
        if provenance is not None:
            if cluster_run != provenance.cluster_run:
                raise ValueError("signals-run cluster provenance changed since worklist export")
            _validate_metadata_labels(con, provenance)
            fired_ids = frozenset(provenance.fired_cluster_ids)
        else:
            fired_ids = canonical_fired_cluster_ids(con, signals_run)
        for row in rows:
            labelled = con.execute(
                """
                SELECT c.run_id FROM cluster_labels l
                JOIN clusters c USING (cluster_id)
                WHERE l.cluster_id = ?
                """,
                [row.cluster_id],
            ).fetchone()
            if labelled is None:
                raise ValueError(f"unknown labelled cluster_id: {row.cluster_id}")
            if labelled[0] != cluster_run:
                raise ValueError(
                    f"cluster {row.cluster_id!r} does not belong to signals run {signals_run!r}"
                )
            origin = _reviewer_origin(row.reviewer_origin)
            desired = (
                origin,
                signals_run,
                row.cluster_id in fired_ids,
                row.mechanism_accuracy,
                row.taxonomy_distinctness_accuracy,
                row.template_accuracy,
                row.should_have_abstained,
                row.failure_category,
                row.notes,
            )
            existing = con.execute(
                "SELECT reviewer_origin, signals_run, is_fired, mechanism_accuracy, "
                "taxonomy_distinctness_accuracy, template_accuracy, should_have_abstained, "
                "failure_category, notes FROM label_verifications "
                "WHERE cluster_id = ? AND reviewer_id = ? AND worklist_version = ?",
                [row.cluster_id, row.reviewer_id, worklist_version],
            ).fetchone()
            if existing is not None:
                if existing != desired:
                    raise ValueError("human verification record is immutable and differs")
                continue
            con.execute(
                "INSERT INTO label_verifications "
                "(cluster_id, reviewer_id, reviewer_origin, worklist_version, signals_run, is_fired, "
                "mechanism_accuracy, taxonomy_distinctness_accuracy, template_accuracy, "
                "should_have_abstained, failure_category, notes, reviewed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    row.cluster_id,
                    row.reviewer_id,
                    origin,
                    worklist_version,
                    signals_run,
                    row.cluster_id in fired_ids,
                    row.mechanism_accuracy,
                    row.taxonomy_distinctness_accuracy,
                    row.template_accuracy,
                    row.should_have_abstained,
                    row.failure_category,
                    row.notes,
                    datetime.now(),
                ],
            )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return len(rows)


def prepare_worklist_record(path: Path, reviewer_id: str) -> PreparedWorklistRecord:
    """Parse exact CSV/sidecar bytes before a writable database is opened."""
    canonical_path, canonical_sidecar, source_bytes, sidecar_bytes = _read_worklist_artifact(path)
    metadata = _metadata_from_bytes(source_bytes, sidecar_bytes)
    if canonical_path.name != metadata.artifact_name:
        raise ValueError("worklist artifact filename changed after export")
    rows = _parse_worklist_bytes(source_bytes, reviewer_id, "human")
    return PreparedWorklistRecord(
        path=canonical_path,
        sidecar_path=canonical_sidecar,
        metadata=metadata,
        source_bytes=source_bytes,
        sidecar_bytes=sidecar_bytes,
        rows=tuple(rows),
    )


def record_worklist(
    con,
    path: Path,
    reviewer_id: str,
    *,
    preflight: PreparedWorklistRecord | None = None,
) -> tuple[int, WorklistMetadata]:
    """Validate one preflighted byte identity and persist its human review."""
    prepared = prepare_worklist_record(path, reviewer_id) if preflight is None else preflight
    if type(prepared) is not PreparedWorklistRecord:
        raise TypeError("preflight must be a PreparedWorklistRecord")
    canonical_path, canonical_sidecar, source_bytes, sidecar_bytes = _read_worklist_artifact(path)
    if (
        canonical_path != prepared.path
        or canonical_sidecar != prepared.sidecar_path
        or source_bytes != prepared.source_bytes
        or sidecar_bytes != prepared.sidecar_bytes
    ):
        raise ValueError("worklist artifact changed after record preflight")
    metadata = _metadata_from_bytes(source_bytes, sidecar_bytes)
    if metadata != prepared.metadata:
        raise ValueError("worklist provenance changed after record preflight")
    if tuple(_parse_worklist_bytes(source_bytes, reviewer_id, "human")) != prepared.rows:
        raise ValueError("worklist decisions changed after record preflight")
    rows = list(prepared.rows)
    count = record(
        con,
        rows,
        metadata.signals_run,
        metadata.worklist_version,
        provenance=metadata,
    )
    return count, metadata


def wilson(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
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
    margin = (
        z
        * ((proportion * (1 - proportion) / total + z_squared / (4 * total * total)) ** 0.5)
        / denominator
    )
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
        _rate(mechanism, total),
        _rate(lenient, total),
        _rate(taxonomy, total),
        _rate(template, total),
    )


def report(con, worklist_version: str | None = None) -> VerificationReport:
    """Report overall and hidden-stratum agreement from recorded human reviews."""
    rows = con.execute(
        """
        SELECT v.cluster_id, v.mechanism_accuracy, v.taxonomy_distinctness_accuracy,
               v.template_accuracy, v.failure_category,
               v.is_fired,
               coalesce(l.confidence, '(unknown)') AS confidence
        FROM label_verifications v
        JOIN cluster_labels l USING (cluster_id)
        JOIN clusters c USING (cluster_id)
        JOIN runs source_run ON source_run.run_id = v.signals_run
                            AND source_run.phase = 'signals'
                            AND source_run.status = 'ok'
        WHERE v.reviewer_origin = 'human'
          AND c.run_id = json_extract_string(source_run.params_json, '$.params.cluster_run')
          AND (? IS NULL OR v.worklist_version = ?)
        ORDER BY v.cluster_id, v.reviewer_id, v.worklist_version
        """,
        [worklist_version, worklist_version],
    ).fetchall()
    metric_rows = [(row[1], row[2], row[3]) for row in rows]
    by_fired: dict[str, list[tuple]] = defaultdict(list)
    by_confidence: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        status = "fired" if row[5] else "control"
        by_fired[status].append((row[1], row[2], row[3]))
        by_confidence[row[6]].append((row[1], row[2], row[3]))
    unique_reviewed_clusters = len({row[0] for row in rows})
    gate_eligible = worklist_version is not None and bool(worklist_version.strip())
    return VerificationReport(
        overall=_metrics(metric_rows),
        by_fired_status={status: _metrics(by_fired[status]) for status in ("fired", "control")},
        by_confidence={key: _metrics(value) for key, value in sorted(by_confidence.items())},
        failure_categories=dict(sorted(Counter(row[4] for row in rows).items())),
        unique_reviewed_clusters=unique_reviewed_clusters,
        gate_eligible=gate_eligible,
        gate_passed=(gate_eligible and unique_reviewed_clusters >= CONFIG.llm.human_verify_n),
    )
