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
from datetime import datetime

from src.config import CONFIG, PATHS
from src.llm import label as label_mod
from src.llm import select as select_mod

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


def run(con, cluster_run: str, signals_run: str, control_n: int, limit: int | None,
        embed_model: str, log=print) -> dict:
    """Label the lazy population. Returns counts; writes `cluster_labels`."""
    import anthropic
    import numpy as np

    memmap = PATHS.artifacts / f"embeddings.{embed_model.split('/')[-1]}.npy"
    vectors = np.load(memmap, mmap_mode="r")
    client = anthropic.Anthropic()
    cache = PATHS.llm_cache

    targets, n_fired, n_control = population(con, cluster_run, signals_run, control_n)
    if limit:
        targets = targets[:limit]
    log(f"population : {len(targets):,} clusters "
        f"({n_fired:,} fired, {n_control:,} control) "
        f"at n_members >= {CONFIG.llm.min_cluster_size_for_label}")
    log(f"model      : {CONFIG.llm.model}   prompt {CONFIG.llm.prompt_version}")

    stats = {"labelled": 0, "cached": 0, "refused": 0, "skipped": 0}
    for cluster_id, family, _n, medoid_idx, dominant, _fired in targets:
        ids, texts = narratives_for(con, vectors, cluster_id, medoid_idx, embed_model)
        if not texts:
            stats["skipped"] += 1
            continue

        key = label_mod.input_hash(CONFIG.llm.prompt_version, CONFIG.llm.model, ids)
        payload = label_mod.cached(cache, key)
        if payload is None:
            payload = label_mod.label_cluster(
                client, CONFIG.llm.model, texts,
                [d for d in [dominant] if d], CONFIG.llm.max_narrative_chars,
            )
            label_mod.write_cache(cache, key, payload)
        else:
            stats["cached"] += 1

        if payload.get("refused"):
            stats["refused"] += 1
            continue
        write_label(con, cluster_id, key, payload)
        stats["labelled"] += 1
        if stats["labelled"] % 25 == 0:
            log(f"  {stats['labelled']:,} labelled ({family})")
    return stats
