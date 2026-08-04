"""MinHash correctness, including agreement with the library the spec named."""

from __future__ import annotations

import numpy as np
import pytest

from src.dedup.minhash import (
    BANDS,
    ROWS,
    band_hashes,
    jaccard,
    permutations,
    shingles,
    signature,
    signatures,
    star_cluster,
)

K = 5
NUM_PERM = 128


@pytest.fixture
def perms():
    return permutations(NUM_PERM, seed=20260803)


def _true_jaccard(a: str, b: str, k: int = K) -> float:
    sa, sb = shingles(a, k), shingles(b, k)
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------
# Shingling
# --------------------------------------------------------------------------
def test_shingles_are_character_ngrams():
    assert shingles("abcdefg", 5) == {"abcde", "bcdef", "cdefg"}


def test_short_and_empty_documents():
    assert shingles("abc", 5) == {"abc"}
    assert shingles("", 5) == set()


def test_empty_document_gets_a_signature(perms):
    a, b = perms
    assert signature("", K, a, b).shape == (NUM_PERM,)


# --------------------------------------------------------------------------
# Signature semantics
# --------------------------------------------------------------------------
def test_identical_documents_have_identical_signatures(perms):
    a, b = perms
    text = "I dispute this account under the Fair Credit Reporting Act."
    assert np.array_equal(signature(text, K, a, b), signature(text, K, a, b))


def test_signature_is_deterministic_across_permutation_rebuilds():
    """A run registry that records a seed but yields different signatures on a
    re-run would make every dedup result unreproducible."""
    text = "the quick brown fox jumps over the lazy dog, repeatedly"
    s1 = signature(text, K, *permutations(NUM_PERM, seed=7))
    s2 = signature(text, K, *permutations(NUM_PERM, seed=7))
    assert np.array_equal(s1, s2)


def test_signature_is_stable_across_processes():
    """The bug this guards: CPython salts `hash()` on strings per process, so
    signatures built with it differ on every run while the run registry
    faithfully records a fixed seed. Caught in the smoke test as the LSH
    candidate count moving between two identical runs."""
    import subprocess
    import sys

    code = (
        "from src.dedup.minhash import permutations, signature;"
        "a,b = permutations(128, 7);"
        "print(int(signature('dispute this account under the FCRA now', 5, a, b).sum()))"
    )
    outs = {
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],  # noqa: S607
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(outs) == 1, f"signature varies with PYTHONHASHSEED: {outs}"


def test_estimated_jaccard_tracks_true_jaccard(perms):
    """The estimate is what the 0.85 threshold is applied to, so it has to be
    close to the real thing across the range that matters."""
    a, b = perms
    base = (
        "In accordance with the Fair Credit Reporting Act, the accounts listed "
        "below have violated my federally protected consumer rights to privacy "
        "and confidentiality under 15 USC 1681. Please investigate and remove."
    )
    variants = [
        base,
        base.replace("investigate and remove", "investigate and delete"),
        base.replace("15 USC 1681", "15 U.S.C. 1681b").replace("privacy", "privacy rights"),
        "My mortgage servicer lost my escrow payment and will not return calls.",
    ]
    for v in variants:
        est = jaccard(signature(base, K, a, b), signature(v, K, a, b))
        true = _true_jaccard(base, v)
        assert abs(est - true) < 0.10, f"est {est:.3f} vs true {true:.3f} for {v[:40]!r}"


def test_signatures_matrix_matches_per_document(perms):
    a, b = perms
    docs = ["alpha beta gamma delta", "alpha beta gamma epsilon", "unrelated text here"]
    mat = signatures(docs, K, a, b)
    assert mat.shape == (3, NUM_PERM)
    assert mat.dtype == np.uint32
    for i, d in enumerate(docs):
        assert np.array_equal(mat[i], signature(d, K, a, b))


# --------------------------------------------------------------------------
# Agreement with datasketch — the library METHODOLOGY §2.2 names
# --------------------------------------------------------------------------
def test_agrees_with_datasketch_on_jaccard_estimates(perms):
    """The numpy implementation replaces datasketch for memory reasons only.
    If the two disagree on similarity, the substitution is not equivalent and
    the deviation recorded in ENGINEERING_NOTES is not defensible."""
    datasketch = pytest.importorskip("datasketch")
    a, b = perms

    pairs = [
        ("the same sentence exactly", "the same sentence exactly"),
        ("dispute account 1234 under FCRA", "dispute account 9876 under FCRA"),
        ("please remove this collection entry", "please delete this collection item"),
        ("my mortgage escrow was miscalculated", "unrelated credit card dispute here"),
    ]
    for x, y in pairs:
        mine = jaccard(signature(x, K, a, b), signature(y, K, a, b))

        mx, my = (datasketch.MinHash(num_perm=NUM_PERM) for _ in range(2))
        mx.update_batch([s.encode() for s in shingles(x, K)])
        my.update_batch([s.encode() for s in shingles(y, K)])
        theirs = mx.jaccard(my)

        true = _true_jaccard(x, y)
        # Both are estimators of the same quantity with the same permutation
        # count, so they must agree with each other about as well as each
        # agrees with the truth.
        assert abs(mine - theirs) < 0.12, f"mine {mine:.3f} datasketch {theirs:.3f}"
        assert abs(mine - true) < 0.12


# --------------------------------------------------------------------------
# Banding
# --------------------------------------------------------------------------
def test_band_shape_and_identical_docs_share_every_band(perms):
    a, b = perms
    sig = signatures(["a template sentence about credit reporting"] * 2, K, a, b)
    bh = band_hashes(sig)
    assert bh.shape == (2, BANDS)
    assert np.array_equal(bh[0], bh[1])


def test_band_count_matches_permutation_count(perms):
    a, b = perms
    assert BANDS * ROWS == NUM_PERM
    with pytest.raises(ValueError, match="num_perm must be"):
        band_hashes(np.zeros((2, 64), dtype=np.uint32))


def test_near_duplicates_share_at_least_one_band(perms):
    """The point of banding: a pair above threshold must become a candidate."""
    a, b = perms
    base = (
        "In accordance with the Fair Credit Reporting Act the accounts listed "
        "below have violated my federally protected consumer rights. Account "
        "number 44556677 must be investigated within 30 days."
    )
    near = base.replace("44556677", "99887766")
    assert _true_jaccard(base, near) > 0.85
    bh = band_hashes(signatures([base, near], K, a, b))
    assert (bh[0] == bh[1]).sum() >= 1


def test_unrelated_documents_rarely_share_a_band(perms):
    a, b = perms
    docs = [
        "my mortgage servicer lost the escrow payment and will not call back",
        "a debt collector keeps calling about an account that is not mine at all",
    ]
    bh = band_hashes(signatures(docs, K, a, b))
    assert (bh[0] == bh[1]).sum() == 0


# --------------------------------------------------------------------------
# Star clustering — bounded group diameter
# --------------------------------------------------------------------------
def test_star_cluster_does_not_chain():
    """The Phase 2 gate failure in one test.

    A~B~C~D with each consecutive pair above threshold but A and D unrelated.
    Connected components merge all four; star clustering must not.
    """
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    label = star_cluster(4, edges)
    assert label[0] != label[3], "endpoints of a chain must not share a group"
    assert len({int(x) for x in label}) >= 2


def test_star_cluster_keeps_a_true_star_together():
    """A template and all its variants share a seed, so they stay one group."""
    edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.int64)
    label = star_cluster(5, edges)
    assert len({int(x) for x in label}) == 1


def test_star_cluster_seeds_highest_degree_first():
    """Node 1 has three neighbours, node 0 has one; 1 should seed."""
    edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
    label = star_cluster(4, edges)
    assert label[1] == 1
    assert label[0] == label[2] == label[3] == 1


def test_star_cluster_isolated_nodes_are_singletons():
    label = star_cluster(4, np.array([[0, 1]], dtype=np.int64))
    assert label[2] == 2 and label[3] == 3
    assert label[0] == label[1]


def test_star_cluster_no_edges():
    label = star_cluster(3, np.empty((0, 2), dtype=np.int64))
    assert list(label) == [0, 1, 2]


def test_star_cluster_is_deterministic():
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [1, 4]], dtype=np.int64)
    a = star_cluster(5, edges)
    b = star_cluster(5, edges[::-1].copy())
    assert list(a) == list(b)


def test_star_cluster_bounds_group_diameter():
    """Every member must have a direct edge to its seed — the invariant that
    makes an 84,657-member chain impossible."""
    rng = np.random.default_rng(0)
    n = 200
    e = rng.integers(0, n, (600, 2))
    e = e[e[:, 0] != e[:, 1]]
    label = star_cluster(n, e)
    adj = {(int(x), int(y)) for x, y in e} | {(int(y), int(x)) for x, y in e}
    for node, seed in enumerate(label):
        if node != seed:
            assert (node, int(seed)) in adj, f"{node} not adjacent to seed {seed}"
