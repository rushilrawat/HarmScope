"""The 300-pair evaluation set for the Phase 2 gate.

docs/METHODOLOGY.md §2.4: stratified 100 obvious duplicates / 100 hard
near-duplicates / 100 unrelated, stored as `data/ground_truth/dedup_eval_pairs.csv`.
Target precision >= 0.95 — a false merge destroys real signal, so precision is
the binding constraint and recall is reported rather than optimised.

**Ids only, never narrative text** (docs/DATA.md §6): the file is committed, and
`tests/test_ground_truth.py` enforces the schema.

## What the reference label is, and what it is not

`true_jaccard` is the exact character-5-shingle overlap, computed from the
narratives directly. That is an *independent* reference for the detector, which
sees only a 128-permutation MinHash estimate and an LSH bucketing — so scoring
against it genuinely measures approximation error and LSH recall.

It does **not** answer "is this the same filing?", which is a judgement. That
judgement is what the `label` column holds, and on a machine-generated set it
is machine-generated. `label_source` records this per row so nobody reads a
precision figure without knowing what produced the labels.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import duckdb

from src.config import PATHS
from src.dedup.minhash import shingles

EVAL_CSV = "dedup_eval_pairs.csv"
NEAR_MISS_CSV = "dedup_near_misses.csv"
HEADER = [
    "complaint_id_a", "complaint_id_b", "label", "stratum",
    "true_jaccard", "label_source", "notes",
]


def write_near_misses(
    ids, pairs, sims, threshold: float, n: int = 300, band: float = 0.30,
    path: Path | None = None,
) -> Path:
    """Sample LSH candidates the verifier **rejected**, during the run.

    Every pair in `EVAL_CSV` was drawn from `dup_pairs`, which only ever holds
    pairs that already cleared the threshold. Recall measured from that file
    alone can only miss pairs lost in grouping — never a pair the verifier
    threw away — and is biased upward by construction.

    The rejected candidates exist only as an in-memory array inside the dedup
    run, so they are sampled here or they are gone. Deterministic stride sample
    over the near-miss band, ordered by similarity.

    Written to `data/interim/`, **not** `data/ground_truth/`: this is detector
    output regenerated on every run, and the ground-truth directory is frozen
    before any detection runs (trap T5). A file that rewrites itself has no
    business living behind a freeze protocol.
    """
    path = path or PATHS.interim / NEAR_MISS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    lo = threshold - band
    keep = [i for i, s in enumerate(sims) if lo <= s < threshold]
    keep.sort(key=lambda i: (float(sims[i]), int(pairs[i, 0]), int(pairs[i, 1])))
    step = max(1, len(keep) // n)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["complaint_id_a", "complaint_id_b", "est_similarity"])
        for i in keep[::step][:n]:
            a, b = int(ids[pairs[i, 0]]), int(ids[pairs[i, 1]])
            w.writerow([min(a, b), max(a, b), round(float(sims[i]), 4)])
    return path


def true_jaccard(text_a: str, text_b: str, k: int) -> float:
    """Exact Jaccard over character k-shingles — no estimation involved."""
    sa, sb = shingles(text_a, k), shingles(text_b, k)
    if not sa and not sb:
        return 1.0
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def sample_pairs(
    con: duckdb.DuckDBPyConnection, run_id: str, k: int, seed: int, per_stratum: int = 100
) -> list[dict]:
    """Draw the three strata.

    `hard` deliberately straddles the 0.85 threshold in both directions: a set
    of only easy cases would report a precision that says nothing about where
    the detector actually makes mistakes.
    """
    rng = random.Random(seed)  # noqa: S311 - sampling, not cryptography
    rows: list[dict] = []

    # Seeded hash ordering rather than USING SAMPLE: DuckDB will not bind
    # parameters into a sample clause, and this is reproducible from the seed
    # alone, which the run registry records.
    strata = {
        "obvious": "similarity >= 0.98",
        "hard": "similarity BETWEEN 0.70 AND 0.93",
    }
    for stratum, where in strata.items():
        for a, b in con.execute(
            f"SELECT complaint_id_a, complaint_id_b FROM dup_pairs WHERE {where} "  # noqa: S608
            f"ORDER BY hash(complaint_id_a * 1000003 + complaint_id_b + ?) LIMIT ?",
            [seed, per_stratum],
        ).fetchall():
            rows.append({"complaint_id_a": a, "complaint_id_b": b, "stratum": stratum})

    # Unrelated: same product family (so the pair is plausible), but no
    # detected relationship. Same-family keeps the negatives non-trivial.
    unrelated = con.execute(
        """
        WITH pool AS (
          SELECT c.complaint_id, c.product_family
          FROM complaints c JOIN narratives USING (complaint_id)
          ORDER BY hash(c.complaint_id + ?) LIMIT ?
        )
        SELECT x.complaint_id, y.complaint_id
        FROM pool x JOIN pool y
          ON x.product_family = y.product_family AND x.complaint_id < y.complaint_id
        WHERE NOT EXISTS (
          SELECT 1 FROM dup_pairs p
          WHERE p.complaint_id_a = x.complaint_id AND p.complaint_id_b = y.complaint_id)
        LIMIT ?
        """,
        [seed, per_stratum * 40, per_stratum],
    ).fetchall()
    for a, b in unrelated:
        rows.append({"complaint_id_a": a, "complaint_id_b": b, "stratum": "unrelated"})

    rng.shuffle(rows)
    return rows


def label(
    con: duckdb.DuckDBPyConnection, rows: list[dict], k: int, threshold: float
) -> list[dict]:
    """Attach the exact-Jaccard reference and a label to each pair."""
    out = []
    for row in rows:
        texts = dict(
            con.execute(
                "SELECT complaint_id, text_redacted FROM narratives "
                "WHERE complaint_id IN (?, ?)",
                [row["complaint_id_a"], row["complaint_id_b"]],
            ).fetchall()
        )
        if len(texts) < 2:
            continue
        tj = true_jaccard(
            texts[row["complaint_id_a"]], texts[row["complaint_id_b"]], k
        )
        out.append({
            **row,
            "true_jaccard": round(tj, 4),
            "label": "dup" if tj >= threshold else "not_dup",
            "label_source": "exact_jaccard_reference",
            "notes": "",
        })
    return out


def write(rows: list[dict], path: Path | None = None) -> Path:
    path = path or PATHS.ground_truth / EVAL_CSV
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        for row in rows:
            w.writerow({key: row.get(key, "") for key in HEADER})
    return path


def score(
    con: duckdb.DuckDBPyConnection, rows: list[dict], run_id: str
) -> dict[str, float]:
    """Precision / recall / F1 of the detector against the labelled set.

    "Predicted duplicate" means the detector put the two complaints in the same
    `dup_group` — the thing that actually has downstream consequences — not
    merely that the pair survived verification.

    `run_id` is required, not optional. `dup_groups` is keyed
    `(run_id, complaint_id)` and holds one row per complaint per refit, so an
    unscoped query counts group ids across runs: `count(DISTINCT group_id) = 1`
    is then almost never true, `tp + fp` is 0, and `precision = 1.0` is
    returned by the empty-denominator guard. A precision gate that reports a
    perfect pass when the query is wrong is trap T1 wearing a rosette — hence
    the asserts below.
    """
    tp = fp = fn = tn = 0
    for row in rows:
        # Both ids must be present: with only one row matching, the distinct
        # count is trivially 1 and a missing complaint scores as a merge.
        n_rows, n_groups = con.execute(
            """
            SELECT count(*), count(DISTINCT group_id) FROM dup_groups
            WHERE run_id = ? AND complaint_id IN (?, ?)
            """,
            [run_id, row["complaint_id_a"], row["complaint_id_b"]],
        ).fetchone()
        same_group = n_rows == 2 and n_groups == 1
        actual = row["label"] == "dup"
        if same_group and actual:
            tp += 1
        elif same_group and not actual:
            fp += 1
        elif not same_group and actual:
            fn += 1
        else:
            tn += 1
    if tp + fp == 0 or tp + fn == 0:
        raise ValueError(
            f"degenerate scoring: tp={tp} fp={fp} fn={fn} tn={tn}. Either the "
            f"run_id {run_id!r} has no dup_groups rows or the eval pairs are "
            f"not in this corpus — not a precision of 1.0."
        )
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def read(path: Path | None = None) -> list[dict]:
    """The committed eval pairs, ids coerced to int."""
    path = path or PATHS.ground_truth / EVAL_CSV
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["complaint_id_a"] = int(row["complaint_id_a"])
        row["complaint_id_b"] = int(row["complaint_id_b"])
        row["true_jaccard"] = float(row["true_jaccard"])
    return rows
