"""Phase 8: the labelling job.

docs/LLM_LAYER.md §2.4 — label **lazily**. Labelling every cluster in every
backtest refit is a large avoidable bill and produces labels nobody reads, so the
default population is the clusters that actually fired a signal, plus a seeded
random control sample so §2.5's verification is not drawn only from alerts.

This module is the only one in `src/llm/` that touches the database, and it only
ever writes `cluster_labels` — a table no detection stage reads. The §1
determinism contract is a property of that: `tests/test_llm.py` asserts no
detection module can import this package at all.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime

from src import db
from src.config import CONFIG, PATHS
from src.llm import label as label_mod
from src.llm import select as select_mod
from src.llm.client import AnthropicModelClient, ModelCallError, ModelCallResult

# Deliberately not `SELECT *`: a cluster is labelled from its members' text and
# its dominant taxonomy labels, never from anything that says whether it fired.
# The prompt cannot leak signal strength if the query never retrieves it.
POPULATION_SQL = """
WITH fired AS (
  SELECT DISTINCT cluster_id FROM signals
  WHERE run_id = ? AND q_value <= ? AND cluster_id IS NOT NULL
)
SELECT c.cluster_id, c.product_family, c.n_members, c.centroid_idx,
       coalesce(n.dominant_label, '') AS dominant_label,
       (c.cluster_id IN (SELECT cluster_id FROM fired)) AS did_fire
FROM clusters c
LEFT JOIN cluster_novelty n USING (cluster_id)
WHERE c.run_id = ? AND c.n_members >= ?
ORDER BY c.cluster_id
"""


@dataclass
class LabelRunStats:
    labelled: int = 0
    cached: int = 0
    refused: int = 0
    failed: int = 0
    skipped: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_seconds: float = 0.0


@dataclass(frozen=True)
class UsageRecord:
    run_id: str | None
    cluster_id: str
    input_hash: str
    cache_status: str
    attempts: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    latency_seconds: float
    estimated_cost_usd: float
    outcome: str
    error_category: str | None = None


def record_usage(con, usage: UsageRecord) -> None:
    """Persist one privacy-safe accounting event for one label target."""
    con.execute(
        "INSERT INTO llm_usage "
        "(usage_id, run_id, operation, cluster_id, model, prompt_version, "
        "input_hash, cache_status, attempts, input_tokens, output_tokens, "
        "cache_read_input_tokens, cache_creation_input_tokens, latency_seconds, "
        "estimated_cost_usd, outcome, error_category, created_at) "
        "VALUES (?, ?, 'label', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            db.new_run_id(), usage.run_id, usage.cluster_id, CONFIG.llm.model,
            CONFIG.llm.prompt_version, usage.input_hash, usage.cache_status,
            usage.attempts, usage.input_tokens, usage.output_tokens,
            usage.cache_read_input_tokens, usage.cache_creation_input_tokens,
            usage.latency_seconds, usage.estimated_cost_usd, usage.outcome,
            usage.error_category, datetime.now(),
        ],
    )


def population(con, cluster_run: str, signals_run: str, control_n: int):
    """Clusters that fired, plus a seeded random control sample of those that did not."""
    rows = con.execute(
        POPULATION_SQL,
        [signals_run, CONFIG.signals.fdr_alpha, cluster_run,
         CONFIG.llm.min_cluster_size_for_label],
    ).fetchall()
    fired = [r for r in rows if r[5]]
    quiet = [r for r in rows if not r[5]]
    rng = random.Random(CONFIG.seed)  # noqa: S311 - sampling, not cryptography
    control = rng.sample(quiet, min(control_n, len(quiet)))
    return fired + control, len(fired), len(control)


def narratives_for(con, vectors, cluster_id: str, medoid_idx, model: str):
    """The k narratives §2.1 selects, and the complaint ids they came from."""
    import numpy as np

    rows = con.execute(
        """
        SELECT m.complaint_id, e.row_idx, n.text_redacted
        FROM cluster_members m
        JOIN embedding_map e USING (complaint_id)
        JOIN narratives n USING (complaint_id)
        WHERE m.cluster_id = ? AND e.model = ?
        ORDER BY m.complaint_id
        """,
        [cluster_id, model],
    ).fetchnumpy()
    if not len(rows["complaint_id"]):
        return [], []

    text_by_id = dict(zip(rows["complaint_id"], rows["text_redacted"], strict=True))
    picked = select_mod.select_for_label(
        vectors,
        rows["row_idx"].astype(np.int64),
        rows["complaint_id"].astype(np.int64),
        medoid_idx,
        CONFIG.llm.label_sample_k,
        CONFIG.llm.label_medoid_k,
    )
    return picked, [text_by_id[i] for i in picked]


def write_label(con, cluster_id: str, key: str, payload: dict) -> None:
    con.execute("DELETE FROM cluster_labels WHERE cluster_id = ?", [cluster_id])
    con.execute(
        "INSERT INTO cluster_labels (cluster_id, harm_mechanism, actors, "
        "preconditions, consumer_impact, distinct_from_taxonomy, rationale, "
        "confidence, is_likely_template, model, prompt_version, input_hash, "
        "generated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            cluster_id,
            payload.get("harm_mechanism"),
            ", ".join(payload.get("actors") or []),
            payload.get("preconditions"),
            payload.get("consumer_impact"),
            payload.get("distinct_from_taxonomy"),
            payload.get("distinctness_rationale"),
            payload.get("confidence"),
            payload.get("is_likely_template"),
            CONFIG.llm.model,
            CONFIG.llm.prompt_version,
            key,
            datetime.now(),
        ],
    )


def _usage_from_result(
    run_id: str | None, cluster_id: str, key: str, cache_status: str,
    outcome: str, result: ModelCallResult | None = None,
    error_category: str | None = None, attempts: int = 0,
) -> UsageRecord:
    if result is None:
        return UsageRecord(
            run_id, cluster_id, key, cache_status, attempts, 0, 0, 0, 0,
            0.0, 0.0, outcome, error_category,
        )
    return UsageRecord(
        run_id, cluster_id, key, cache_status, result.attempts,
        result.usage.input_tokens, result.usage.output_tokens,
        result.usage.cache_read_input_tokens,
        result.usage.cache_creation_input_tokens, result.latency_seconds,
        result.estimated_cost_usd, outcome, error_category,
    )


def _add_result_totals(stats: LabelRunStats, result: ModelCallResult) -> None:
    stats.input_tokens += result.usage.input_tokens
    stats.output_tokens += result.usage.output_tokens
    stats.estimated_cost_usd += result.estimated_cost_usd
    stats.latency_seconds += result.latency_seconds


def _persist_target(con, cluster_id: str, key: str, payload: dict | None,
                    usage: UsageRecord) -> None:
    """Commit the label and its accounting row together, or neither."""
    con.execute("BEGIN TRANSACTION")
    try:
        if payload is not None and not payload.get("refused"):
            write_label(con, cluster_id, key, payload)
        record_usage(con, usage)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


def run(con, cluster_run: str, signals_run: str, control_n: int, limit: int | None,
        embed_model: str, log=print, *, client=None, vectors=None, cache_dir=None,
        run_id: str | None = None) -> LabelRunStats:
    """Label the lazy population with resumable cache and per-target accounting."""
    import numpy as np

    if vectors is None:
        memmap = PATHS.artifacts / f"embeddings.{embed_model.split('/')[-1]}.npy"
        vectors = np.load(memmap, mmap_mode="r")
    cache = cache_dir or PATHS.llm_cache

    targets, n_fired, n_control = population(con, cluster_run, signals_run, control_n)
    if limit:
        targets = targets[:limit]
    log(f"population : {len(targets):,} clusters "
        f"({n_fired:,} fired, {n_control:,} control) "
        f"at n_members >= {CONFIG.llm.min_cluster_size_for_label}")
    log(f"model      : {CONFIG.llm.model}   prompt {CONFIG.llm.prompt_version}")

    stats = LabelRunStats()
    preflight_done = False
    for cluster_id, family, _n, medoid_idx, dominant, _fired in targets:
        ids, texts = narratives_for(con, vectors, cluster_id, medoid_idx, embed_model)
        key = label_mod.input_hash(CONFIG.llm.prompt_version, CONFIG.llm.model, ids)
        if not texts:
            stats.skipped += 1
            _persist_target(
                con, cluster_id, key, None,
                _usage_from_result(run_id, cluster_id, key, "bypass", "skipped"),
            )
            continue

        payload = label_mod.cached(cache, key)
        if payload is None:
            if client is None:
                client = AnthropicModelClient()
            if not preflight_done:
                try:
                    client.preflight(CONFIG.llm.model)
                except ModelCallError as error:
                    stats.failed += 1
                    _persist_target(
                        con, cluster_id, key, None,
                        _usage_from_result(
                            run_id, cluster_id, key, "miss", "failed",
                            error_category=error.category, attempts=error.attempts,
                        ),
                    )
                    if error.category in {
                        "authentication", "billing", "permission", "invalid_request",
                    }:
                        raise
                    continue
                preflight_done = True
            try:
                result = client.call_json(
                    model=CONFIG.llm.model,
                    system=label_mod.SYSTEM,
                    prompt=label_mod.build_prompt(
                        texts, [d for d in [dominant] if d],
                        CONFIG.llm.max_narrative_chars,
                    ),
                    schema=label_mod.LABEL_SCHEMA,
                    max_tokens=2000,
                )
            except ModelCallError as error:
                stats.failed += 1
                _persist_target(
                    con, cluster_id, key, None,
                    _usage_from_result(
                        run_id, cluster_id, key, "miss", "failed",
                        error_category=error.category, attempts=error.attempts,
                    ),
                )
                if error.category in {
                    "authentication", "billing", "permission", "invalid_request",
                }:
                    raise
                continue
            _add_result_totals(stats, result)
            try:
                label_mod.write_cache(cache, key, result.payload)
            except label_mod.LabelSchemaError:
                stats.failed += 1
                _persist_target(
                    con, cluster_id, key, None,
                    _usage_from_result(run_id, cluster_id, key, "miss", "failed", result,
                                       "schema"),
                )
                continue
            payload = result.payload
            usage = _usage_from_result(
                run_id, cluster_id, key, "miss",
                "refused" if payload.get("refused") else "ok", result,
            )
        else:
            stats.cached += 1
            usage = _usage_from_result(
                run_id, cluster_id, key, "hit",
                "refused" if payload.get("refused") else "ok",
            )

        if payload.get("refused"):
            stats.refused += 1
            _persist_target(con, cluster_id, key, None, usage)
            continue
        _persist_target(con, cluster_id, key, payload, usage)
        stats.labelled += 1
        if stats.labelled % 25 == 0:
            log(f"  {stats.labelled:,} labelled ({family})")
    return stats
