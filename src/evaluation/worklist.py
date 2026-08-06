"""Phase 6: build a blinded adjudication worklist.

docs/EVALUATION.md §1.3. This module is the *only* place that reads signal
strength. It selects the candidates and then strips every trace of why they
were selected, writing a worklist that carries cluster text and nothing else.

The split matters. `adjudicate.py` — the surface a human actually looks at —
cannot import from here and cannot reach `signals`, so the blinding is a
property of the code's shape rather than of the adjudicator's discipline.
`tests/test_leakage.py` item 5 asserts exactly that, by reading the source.

Decoys are drawn from unrelated companies and shuffled in with the real
candidates. Without them an adjudicator who sees twenty clusters for one
company knows the system fired on that company, and "did any of these match?"
becomes "which of these matched?" — a different and much easier question.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import duckdb

HEADER = [
    "action_id", "slot", "cluster_id", "product_family", "n_members",
    "exemplar_1", "exemplar_2", "exemplar_3",
    "match_quality", "notes",
]


@dataclass(frozen=True)
class Candidate:
    cluster_id: str
    product_family: str
    n_members: int
    exemplars: list[str]


def _exemplars(con, cluster_id: str, k: int, chars: int) -> list[str]:
    rows = con.execute(
        """
        SELECT n.text_redacted FROM cluster_members m
        JOIN narratives n USING (complaint_id)
        WHERE m.cluster_id = ?
        ORDER BY m.is_exemplar DESC, m.membership_prob DESC LIMIT ?
        """,
        [cluster_id, k],
    ).fetchall()
    return [" ".join(r[0].split())[:chars] for r in rows]


def select(
    con: duckdb.DuckDBPyConnection,
    signals_run: str,
    cluster_run: str,
    company_id: str,
    top_k: int,
    n_decoys: int,
    seed: int,
) -> tuple[list[Candidate], set[str]]:
    """Top clusters for a company by signal strength, plus decoys.

    Returns `(shuffled candidates, the real cluster ids)`. The caller writes the
    worklist; the truth set stays here and never reaches the surface.
    """
    real = con.execute(
        """
        SELECT DISTINCT s.cluster_id, c.product_family, c.n_members
        FROM signals s JOIN clusters c USING (cluster_id)
        WHERE s.run_id = ? AND s.company_id = ?
        ORDER BY c.n_members DESC LIMIT ?
        """,
        [signals_run, company_id, top_k],
    ).fetchall()

    decoys = con.execute(
        """
        SELECT c.cluster_id, c.product_family, c.n_members
        FROM clusters c
        WHERE c.run_id = ? AND c.cluster_id NOT IN (
            SELECT cluster_id FROM signals WHERE run_id = ? AND company_id = ?)
        ORDER BY hash(c.cluster_id || ?) LIMIT ?
        """,
        [cluster_run, signals_run, company_id, str(seed), n_decoys],
    ).fetchall()

    truth = {r[0] for r in real}
    pool = [
        Candidate(cid, fam, n, _exemplars(con, cid, 3, 260))
        for cid, fam, n in [*real, *decoys]
    ]
    random.Random(seed).shuffle(pool)  # noqa: S311 - presentation order, not crypto
    return pool, truth


def write(
    action_id: str, candidates: list[Candidate], path: Path
) -> Path:
    """Emit the worklist. No column here says whether anything fired."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        for slot, cand in enumerate(candidates, start=1):
            padded = (cand.exemplars + ["", "", ""])[:3]
            writer.writerow({
                "action_id": action_id, "slot": slot,
                "cluster_id": cand.cluster_id,
                "product_family": cand.product_family,
                "n_members": cand.n_members,
                "exemplar_1": padded[0], "exemplar_2": padded[1],
                "exemplar_3": padded[2],
                "match_quality": "", "notes": "",
            })
    return path
