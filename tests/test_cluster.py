"""Assignment, novelty scoring, and the stability comparison.

No UMAP or HDBSCAN here — those are third-party and slow. What is tested is the
project's own decisions: that the assignment threshold means what it says, that
novelty is high exactly when no label describes a cluster, and that ARI is not
inflated by two runs agreeing a point is noise.
"""

from __future__ import annotations

import numpy as np

from src.cluster import assign, novelty, stability
from src.config import NoveltyConfig

CFG = NoveltyConfig()


def _unit(*rows) -> np.ndarray:
    a = np.asarray(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_assignment_threshold_is_cosine_distance():
    """max_distance 0.35 must mean cosine similarity 0.65, not 0.35."""
    centroids = _unit([1, 0], [0, 1])
    near = _unit([1, 0.2])          # cosine to centroid 0 is ~0.98
    far = _unit([1, 1])             # ~0.707 to both
    vectors = np.vstack([near, far])
    rows = np.array([0, 1])

    labels, sims = assign.assign(vectors, rows, centroids, max_distance=0.35)
    assert labels[0] == 0
    assert labels[1] != assign.NOISE, "0.707 similarity is inside a 0.35 distance"

    labels, _ = assign.assign(vectors, rows, centroids, max_distance=0.1)
    assert labels[0] == 0, "0.98 similarity is still inside a 0.1 distance"
    assert labels[1] == assign.NOISE, "0.707 similarity is outside a 0.1 distance"


def test_no_centroids_means_everything_is_noise():
    vectors = _unit([1, 0], [0, 1])
    labels, _ = assign.assign(vectors, np.array([0, 1]), np.empty((0, 2)), 0.35)
    assert (labels == assign.NOISE).all()


def test_coherence_is_per_cluster_and_ignores_noise():
    labels = np.array([0, 0, assign.NOISE, 1])
    sims = np.array([0.9, 0.7, 0.99, 0.5])
    got = assign.coherence(labels, sims, n_clusters=2)
    assert np.isclose(got[0], 0.8)   # the 0.99 noise point must not count
    assert np.isclose(got[1], 0.5)


def test_related_links_only_across_families():
    centroids = {"a": _unit([1, 0], [0, 1]), "b": _unit([1, 0.02])}
    links = assign.related(centroids, min_similarity=0.9)
    assert [(fa, fb) for fa, _, fb, _, _ in links] == [("a", "b")]
    # Two near-identical clusters inside one family are a clustering question,
    # not a cross-product harm, and must not be linked.
    assert not assign.related({"a": _unit([1, 0], [1, 0.01])}, 0.9)


def test_a_cluster_with_no_surviving_label_is_maximally_novel():
    """The case the whole ablation test turns on."""
    assert novelty.score_cluster([], family_label_space=100, cfg=CFG).score == 1.0


def test_one_label_everywhere_is_not_novel():
    got = novelty.score_cluster(["x"] * 50, family_label_space=100, cfg=CFG)
    assert got.dominant_share == 1.0
    assert got.entropy == 0.0
    assert got.score == 0.0


def test_entropy_is_normalized_by_the_family_space_not_the_cluster():
    """A 50/50 split over two labels is not 'scattered across many labels'.

    Normalizing by observed support would score this 1.0 — the standard Shannon
    evenness measure, and wrong for what METHODOLOGY §5 defines novelty to be.
    """
    two = novelty.score_cluster(["a"] * 10 + ["b"] * 10, 100, CFG)
    assert two.entropy < 0.2, two.entropy
    many = novelty.score_cluster([str(i) for i in range(100)], 100, CFG)
    assert many.entropy > 0.99
    assert many.score > two.score


def test_ablation_hides_a_label_without_removing_the_member():
    members = {
        "hidden": [("fraud", "s1")] * 20,
        "kept": [("billing", "s2")] * 20,
    }
    plain = novelty.score_all(members, 100, CFG)
    assert plain["hidden"].score == plain["kept"].score == 0.0

    ablated = novelty.score_all(members, 100, CFG, hidden_issue="fraud")
    assert ablated["hidden"].score == 1.0, "hiding its only issue must max novelty"
    assert ablated["kept"].score == 0.0, "an unrelated cluster must be unaffected"


def test_ablation_auc_separates_hidden_from_matched():
    members = {f"c{i}": [("fraud", "s")] * 30 for i in range(3)}
    members.update({f"d{i}": [("billing", "s")] * 30 for i in range(3)})
    report = novelty.ablation_auc(members, 100, CFG)
    assert report["n_issues"] == 2
    assert report["mean_auc"] == 1.0
    assert report["passes"]


def test_novelty_requires_coherence_and_persistence():
    """§5.2 — incoherent-and-novel is a clustering defect, not a finding."""
    nov = novelty.Novelty("x", 0.0, 1.0, 1.0, 5)
    assert novelty.is_novel(nov, coherence=0.9, persistence=0.5, cfg=CFG)
    assert not novelty.is_novel(nov, coherence=0.1, persistence=0.5, cfg=CFG)
    assert not novelty.is_novel(nov, coherence=0.9, persistence=0.0, cfg=CFG)


def test_ari_ignores_points_either_run_called_noise():
    """Agreeing a point belongs nowhere is not agreement about structure."""
    a = np.array([0, 0, 1, 1, assign.NOISE, assign.NOISE])
    b = np.array([1, 1, 0, 0, assign.NOISE, 5])
    value, n = stability.ari(a, b)
    assert n == 4, "the two noise columns must be dropped"
    assert np.isclose(value, 1.0), "a relabelling is perfect agreement"


def test_ari_is_nan_when_nothing_is_comparable():
    a = np.array([assign.NOISE, assign.NOISE])
    value, n = stability.ari(a, np.array([0, 1]))
    assert n == 0 and np.isnan(value)
