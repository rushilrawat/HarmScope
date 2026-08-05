"""Phase 4: assign every representative to a cluster, or to noise.

docs/METHODOLOGY.md §4.1 offers `approximate_predict` or "FAISS nearest-exemplar
+ threshold". This is the second. `approximate_predict` would require running
`umap.transform` on the ~1.4M representatives outside the fit sample, which is
both slow and conceptually shaky: UMAP is fit to one sample, so a transformed
coordinate for an unseen point is an extrapolation, and HDBSCAN's density
estimate in that space was never fit to it either.

Cosine against a cluster centroid in the original embedding space has neither
problem. It is the space Phase 3 unit-normalized for, `assign_max_distance` is
interpretable in it, and the same rule applies to sampled and unsampled points
alike — see the note in `fit.py` on why that symmetry matters.

Centroid rather than a single exemplar point: a mean over hundreds of members is
a far more stable target than whichever member happened to be picked, and it
costs one vector per cluster either way. The medoid is still recorded, as
`clusters.centroid_idx`, because a human reading a cluster wants a real
narrative rather than an average of narratives.
"""

from __future__ import annotations

import numpy as np

from src.cluster.fit import read_vectors

NOISE = -1


def assign(
    vectors,
    rows: np.ndarray,
    centroids: np.ndarray,
    max_distance: float,
    chunk: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest centroid per row, or `NOISE` when none is close enough.

    Returns `(labels, similarity)`. Vectors and centroids are unit length, so
    the inner product FAISS maximizes *is* cosine similarity and the cosine
    distance bound is a plain `1 - sim` comparison — no metric conversion, and
    `IndexFlatIP` makes the search exact rather than approximate.
    """
    import faiss

    if len(centroids) == 0:
        return np.full(len(rows), NOISE, dtype=np.int32), np.zeros(len(rows))

    index = faiss.IndexFlatIP(centroids.shape[1])
    index.add(np.ascontiguousarray(centroids))
    min_similarity = 1.0 - max_distance

    labels = np.empty(len(rows), dtype=np.int32)
    sims = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), chunk):
        sl = slice(start, start + chunk)
        block = read_vectors(vectors, rows[sl])
        score, idx = index.search(np.ascontiguousarray(block), 1)
        labels[sl] = idx[:, 0]
        sims[sl] = score[:, 0]
    labels[sims < min_similarity] = NOISE
    return labels, sims


def coherence(labels: np.ndarray, sims: np.ndarray, n_clusters: int) -> np.ndarray:
    """Mean cosine of each cluster's assigned members to its centroid.

    METHODOLOGY §5.2 requires this as the guard against calling an *incoherent*
    cluster novel. Member-to-centroid rather than all-pairs: the all-pairs mean
    is O(n^2) per cluster and, for unit vectors, is a monotone function of this
    anyway — the squared norm of the mean vector — so the ranking §5.2 needs is
    unchanged and the cost is not.
    """
    out = np.zeros(n_clusters, dtype=np.float64)
    for c in range(n_clusters):
        member = labels == c
        if member.any():
            out[c] = float(sims[member].mean())
    return out


def medoids(
    vectors, rows: np.ndarray, labels: np.ndarray, sims: np.ndarray, n_clusters: int
) -> np.ndarray:
    """The single member closest to each centroid — a readable stand-in.

    `sims` already holds each member's cosine to its own centroid, so the medoid
    is an argmax over a column we have rather than a second pass over the data.
    """
    out = np.full(n_clusters, -1, dtype=np.int64)
    for c in range(n_clusters):
        member = np.flatnonzero(labels == c)
        if len(member):
            out[c] = int(rows[member[np.argmax(sims[member])]])
    return out


def related(
    centroids: dict[str, np.ndarray], min_similarity: float
) -> list[tuple[str, int, str, int, float]]:
    """Cross-family cluster pairs whose centroids are close (METHODOLOGY §4.2).

    Only across families: within a family, two clusters being similar means
    HDBSCAN split one region, which is a clustering question rather than the
    cross-product harm this is looking for.
    """
    out = []
    families = sorted(centroids)
    for i, fa in enumerate(families):
        for fb in families[i + 1 :]:
            a, b = centroids[fa], centroids[fb]
            if not len(a) or not len(b):
                continue
            sim = a @ b.T
            for ia, ib in zip(*np.where(sim >= min_similarity), strict=True):
                out.append((fa, int(ia), fb, int(ib), float(sim[ia, ib])))
    return out
