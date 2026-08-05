"""Phase 4 gate: is the cluster structure a property of the data or of the sample?

docs/METHODOLOGY.md §4.3 asks for ARI "on the intersection of assigned points".
For two fits on **disjoint** halves that intersection is empty by construction,
so the comparison has to be made somewhere neither fit saw: a held-out
evaluation set is carved off first, both models assign it, and ARI is computed
over the points both models assigned to some cluster.

That is a stricter test than comparing overlapping samples, and it is the one
worth passing — it asks whether two independent views of the corpus induce the
same partition on unseen complaints, which is what "these clusters are real"
has to mean if Phase 5 is going to track them over time.
"""

from __future__ import annotations

import numpy as np

from src.cluster import assign as assign_mod
from src.cluster import fit as fit_mod
from src.config import ClusterConfig


def ari(labels_a: np.ndarray, labels_b: np.ndarray) -> tuple[float, int]:
    """ARI over points both runs assigned. Returns `(ari, n_compared)`.

    Noise is excluded rather than treated as a shared cluster: two runs agreeing
    that a point belongs nowhere is not agreement about structure, and counting
    it as such would let a mostly-noise clustering post a high ARI.
    """
    from sklearn.metrics import adjusted_rand_score

    both = (labels_a != assign_mod.NOISE) & (labels_b != assign_mod.NOISE)
    n = int(both.sum())
    if n < 2:
        return float("nan"), n
    return float(adjusted_rand_score(labels_a[both], labels_b[both])), n


def _fit_and_assign(
    vectors, fit_rows, eval_rows, family, cfg, seed, size, log
) -> tuple[np.ndarray, int, float] | None:
    result = fit_mod.fit_family(
        vectors, fit_rows, family, cfg, seed, sample_size=size, log=log
    )
    if result is None or result.n_clusters == 0:
        return None
    labels, _ = assign_mod.assign(
        vectors, eval_rows, result.centroids, cfg.assign_max_distance
    )
    return labels, result.n_clusters, result.fit_noise_fraction


def sample_size_sweep(
    vectors,
    rows: np.ndarray,
    family: str,
    cfg: ClusterConfig,
    seed: int,
    sizes: tuple[int, ...],
    eval_size: int = 50_000,
    log=print,
) -> dict:
    """Cluster count and ARI as a function of fit sample size.

    Every fit is compared against the largest one on the same held-out set, so
    the numbers answer "does adding data change the partition" rather than
    "do two arbitrary runs agree".
    """
    rng = np.random.default_rng(seed)
    held = np.sort(rng.choice(rows, size=min(eval_size, len(rows) // 4), replace=False))
    pool = np.setdiff1d(rows, held, assume_unique=False)

    runs = []
    for size in sizes:
        if size > len(pool):
            log(f"  size {size:,} exceeds the {len(pool):,} available — skipped")
            continue
        got = _fit_and_assign(vectors, pool, held, family, cfg, seed, size, log)
        if got is None:
            continue
        labels, n_clusters, fit_noise = got
        runs.append({
            "size": size, "n_clusters": n_clusters, "fit_noise": fit_noise,
            "assigned_fraction": float((labels != assign_mod.NOISE).mean()),
            "labels": labels,
        })

    for run in runs:
        value, n = ari(run["labels"], runs[-1]["labels"])
        run["ari_vs_largest"] = value
        run["n_compared"] = n
        del run["labels"]
    return {"family": family, "eval_size": len(held), "runs": runs}


def disjoint_halves(
    vectors,
    rows: np.ndarray,
    family: str,
    cfg: ClusterConfig,
    seed: int,
    eval_size: int = 50_000,
    log=print,
) -> dict:
    """The gate's headline: two independent halves, one unseen evaluation set.

    The evaluation set is carved out **before** the split, so neither half
    contains any of it and neither model has an advantage on it.
    """
    rng = np.random.default_rng(seed)
    held = np.sort(rng.choice(rows, size=min(eval_size, len(rows) // 4), replace=False))
    pool = np.setdiff1d(rows, held, assume_unique=False)
    shuffled = rng.permutation(pool)
    half_a, half_b = np.sort(shuffled[: len(shuffled) // 2]), np.sort(
        shuffled[len(shuffled) // 2 :]
    )

    out = {"family": family, "eval_size": len(held),
           "half_sizes": (len(half_a), len(half_b))}
    a = _fit_and_assign(vectors, half_a, held, family, cfg, seed, len(half_a), log)
    b = _fit_and_assign(vectors, half_b, held, family, cfg, seed + 1, len(half_b), log)
    if a is None or b is None:
        out["ari"] = None
        return out
    labels_a, n_a, noise_a = a
    labels_b, n_b, noise_b = b
    value, n = ari(labels_a, labels_b)
    out.update({
        "n_clusters": (n_a, n_b), "fit_noise": (noise_a, noise_b),
        "ari": value, "n_compared": n,
        "assigned_fraction": (
            float((labels_a != assign_mod.NOISE).mean()),
            float((labels_b != assign_mod.NOISE).mean()),
        ),
    })
    return out
