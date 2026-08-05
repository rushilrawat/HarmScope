"""Phase 4: UMAP -> HDBSCAN on a per-family sample, then cluster centroids.

docs/METHODOLOGY.md §4.1. HDBSCAN's job here is **discovery** — find where the
dense regions are and how many there are. It is not the final word on
membership; `assign.py` is, and it applies one rule to every representative.

The reason is a sampling artifact that would otherwise be baked in. HDBSCAN
labels 54-61% of a credit-reporting fit sample as noise (measured 2026-08-05).
If sampled points kept that `-1` while unsampled points were assigned by
nearest centroid within `assign_max_distance`, two identical narratives would
get different fates according to which one happened to be drawn — and cluster
sizes would carry that coin flip into every Phase 5 growth statistic. So the
noise call is used to *report* discovery quality (§4.3 asks for it) and the
threshold decides membership for everyone.

`random_state` is set on UMAP, which disables its parallelism and costs real
wall time. README standing rule 3 requires determinism, and an unreproducible
clustering makes the stability numbers in §4.3 meaningless — they would measure
UMAP's seed as much as the data.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np

from src.config import ClusterConfig


@dataclass(frozen=True)
class FitResult:
    """Outcome of one family fit. `centroids` is what assignment consumes."""

    family: str
    sample_rows: np.ndarray      # memmap row indices the fit saw
    labels: np.ndarray           # HDBSCAN label per sampled point, -1 = noise
    centroids: np.ndarray        # (n_clusters, dim), unit length, original space
    persistence: np.ndarray      # HDBSCAN cluster persistence, per cluster
    n_sampled: int
    fit_noise_fraction: float    # HDBSCAN's own call, before assignment
    seconds: float

    @property
    def n_clusters(self) -> int:
        return len(self.centroids)


def sample_rows(rows: np.ndarray, size: int, seed: int) -> np.ndarray:
    """A reproducible subsample, sorted so memmap reads stay sequential."""
    if len(rows) <= size:
        return np.sort(rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=size, replace=False))


def read_vectors(vectors, rows: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Materialize `rows` from the memmap. Chunked to bound peak memory."""
    out = np.empty((len(rows), vectors.shape[1]), dtype=np.float32)
    for start in range(0, len(rows), chunk):
        sl = slice(start, start + chunk)
        out[sl] = vectors[rows[sl]]
    return out


def centroids_from(
    X: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Unit-length mean of each cluster's members, in the ORIGINAL space.

    Not the UMAP space: UMAP is a non-metric embedding fit to this sample only,
    so a distance in it means nothing for a point the fit never saw. The
    original space is where `assign_max_distance` is interpretable and where
    cosine is the similarity Phase 3 normalized for.
    """
    out = np.zeros((n_clusters, X.shape[1]), dtype=np.float32)
    for c in range(n_clusters):
        members = X[labels == c]
        if len(members):
            out[c] = members.mean(axis=0)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)


def fit_family(
    vectors,
    rows: np.ndarray,
    family: str,
    cfg: ClusterConfig,
    seed: int,
    sample_size: int | None = None,
    log=print,
) -> FitResult | None:
    """UMAP -> HDBSCAN over one product family. `None` if it is too small.

    A family needs more points than UMAP has neighbours, and enough of them that
    a `min_cluster_size` cluster is not most of the family. Returning None is
    the honest outcome for `other` (290 representatives) rather than inventing
    structure in it.
    """
    import hdbscan
    import umap

    size = sample_size or cfg.fit_sample_size
    take = sample_rows(rows, size, seed)
    floor = max(cfg.umap_n_neighbors + 1, cfg.hdbscan_min_cluster_size * 2)
    if len(take) < floor:
        log(f"  {family:<18} {len(take):>7,} representatives — below the {floor} "
            f"floor, not clustered")
        return None

    t0 = time.time()
    X = read_vectors(vectors, take)
    with warnings.catch_warnings():
        # UMAP warns that `random_state` forces n_jobs=1. That is the trade this
        # module exists to make — see the module docstring — so the warning is
        # reporting intended behaviour, once per family. Matched narrowly: any
        # other UMAP warning still surfaces.
        warnings.filterwarnings(
            "ignore", message=r"n_jobs value .* overridden", category=UserWarning
        )
        reduced = umap.UMAP(
            n_neighbors=cfg.umap_n_neighbors,
            n_components=cfg.umap_n_components,
            min_dist=cfg.umap_min_dist,
            metric=cfg.umap_metric,
            random_state=seed,
            verbose=False,
        ).fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        cluster_selection_method=cfg.cluster_selection_method,
        core_dist_n_jobs=-1,
    ).fit(reduced)

    labels = clusterer.labels_
    n_clusters = int(labels.max()) + 1
    persistence = np.asarray(
        getattr(clusterer, "cluster_persistence_", np.zeros(n_clusters)),
        dtype=np.float64,
    )
    if len(persistence) != n_clusters:  # defensive: keep indexing aligned
        persistence = np.resize(persistence, n_clusters)

    result = FitResult(
        family=family,
        sample_rows=take,
        labels=labels,
        centroids=centroids_from(X, labels, n_clusters),
        persistence=persistence,
        n_sampled=len(take),
        fit_noise_fraction=float((labels == -1).mean()),
        seconds=time.time() - t0,
    )
    log(f"  {family:<18} {len(take):>7,} sampled  {n_clusters:>5} clusters  "
        f"fit-noise {result.fit_noise_fraction:5.1%}  {result.seconds / 60:5.1f} min")
    return result
