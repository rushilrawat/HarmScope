"""Phase 8: which narratives a cluster gets labelled from.

docs/LLM_LAYER.md §2.1 — k = 20 per cluster: 12 nearest the medoid, 8 chosen for
diversity by maximal marginal relevance against the first 12.

**The diversity half is not a nicety.** 20 near-medoid narratives from a
template-adjacent cluster all read identically, and the model will faithfully
describe the template rather than the harm mechanism. The medoid half says what
the cluster is centrally about; the MMR half is what stops a confident label
being written about a mail merge.

Everything here is deterministic and touches no API. That matters for the §1
contract: the selection, the cache key, and the prompt are all reproducible
without a network call, so the only non-deterministic step in the layer is the
model's own response.
"""

from __future__ import annotations

import numpy as np


def mmr(
    candidates: np.ndarray,
    selected: np.ndarray,
    k: int,
    lambda_: float = 0.5,
) -> list[int]:
    """Indices into `candidates` maximising relevance minus redundancy.

    Relevance is similarity to the cluster's own centre, redundancy is the
    largest similarity to anything already picked. Greedy and O(k * n), which is
    fine at the k=20 this is called with and keeps the result independent of any
    library's tie-breaking.

    Vectors are unit length (METHODOLOGY §3), so a dot product is cosine.
    """
    if k <= 0 or not len(candidates):
        return []
    centre = candidates.mean(axis=0)
    centre = centre / max(np.linalg.norm(centre), 1e-12)
    relevance = candidates @ centre

    # Similarity to the already-selected set, updated as picks are made rather
    # than recomputed: the max over a growing set only ever increases.
    redundancy = (
        (candidates @ selected.T).max(axis=1)
        if len(selected)
        else np.zeros(len(candidates))
    )

    picked: list[int] = []
    for _ in range(min(k, len(candidates))):
        score = lambda_ * relevance - (1 - lambda_) * redundancy
        score[picked] = -np.inf
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            break
        picked.append(best)
        redundancy = np.maximum(redundancy, candidates @ candidates[best])
    return picked


def select_for_label(
    vectors,
    rows: np.ndarray,
    complaint_ids: np.ndarray,
    medoid_row: int | None,
    k: int,
    medoid_k: int,
) -> list[int]:
    """`complaint_id`s to show the model, medoid-nearest first then MMR.

    Returns them in selection order but the caller sorts before hashing — the
    cache key must not depend on which half a narrative came from
    (docs/LLM_LAYER.md §2.4).
    """
    from src.cluster.fit import read_vectors

    if len(rows) == 0:
        return []
    X = read_vectors(vectors, rows)

    # The medoid is a real narrative (clusters.centroid_idx), not the mean —
    # §2.1 wants the 12 nearest an actual complaint a human could read.
    if medoid_row is not None and medoid_row in set(rows.tolist()):
        anchor = X[int(np.flatnonzero(rows == medoid_row)[0])]
    else:
        anchor = X.mean(axis=0)
        anchor = anchor / max(np.linalg.norm(anchor), 1e-12)

    near = np.argsort(-(X @ anchor))[: min(medoid_k, len(X))]
    diverse = mmr(X, X[near], k - len(near))
    order = list(near) + [i for i in diverse if i not in set(near.tolist())]
    return [int(complaint_ids[i]) for i in order[:k]]
