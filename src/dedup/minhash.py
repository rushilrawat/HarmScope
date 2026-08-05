"""Tier 2 near-duplicate detection: MinHash + LSH banding.

docs/METHODOLOGY.md §2.2. Character 5-shingles, 128 permutations, Jaccard
threshold ~0.85, blocked by `product_family`.

**Deviation from the spec, recorded here and in ENGINEERING_NOTES.**
METHODOLOGY names `datasketch`. The signatures are computed with numpy instead
and the LSH banding is done in DuckDB. Reason is memory, not preference: at
2.48M representatives a `datasketch` `MinHashLSH` holds ~40M Python dict
entries across its band tables, which does not fit in 19 GB, and the per-object
`MinHash` model is also 2x slower to build (measured 1,126 vs 2,128 docs/s).
The algorithm is unchanged — same shingling, same permutation count, same
Jaccard semantics — and `tests/test_minhash.py` asserts agreement with
`datasketch` on a sample, so the substitution is verified rather than asserted.

Banding: 128 permutations = 16 bands x 8 rows. Detection probability at the
0.85 threshold is 1-(1-0.85^8)^16 = 0.993; at Jaccard 0.5 it is 0.06. Two docs
that share any band become a candidate pair and are then verified against the
full signature, so band collisions cost time, never correctness.
"""

from __future__ import annotations

import zlib

import numpy as np

# Largest prime below 2^32, so signatures fit in uint32: 2.48M x 128 x 4 bytes
# is 1.27 GB, against 2.54 GB for uint64. At this scale that is the difference
# between fitting in memory and not.
MERSENNE_P = np.uint64((1 << 32) - 5)
MAX_HASH = np.uint64((1 << 32) - 1)

BANDS = 16
ROWS = 8  # BANDS * ROWS must equal num_perm


def shingles(text: str, k: int) -> set[str]:
    """Character k-shingles.

    Character shingles beat word shingles here because templates vary mainly in
    inserted account numbers and dates (docs/METHODOLOGY.md §2.2) — a word
    shingle spanning the varying token is destroyed, a character shingle mostly
    is not.
    """
    if len(text) <= k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def permutations(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`(a, b)` coefficients for h_i(x) = (a_i*x + b_i) mod p. Seeded, so the
    signatures are reproducible across runs — a run registry that records a
    seed but produces different signatures would be worthless."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, MERSENNE_P, num_perm, dtype=np.uint64)
    b = rng.integers(0, MERSENNE_P, num_perm, dtype=np.uint64)
    return a, b


def _shingle_hashes(shingle_set: set[str]) -> np.ndarray:
    """Stable 32-bit hash per shingle.

    `zlib.crc32`, NOT Python's builtin `hash()`. String hashing in CPython is
    salted per process (PYTHONHASHSEED), so `hash()` produces different
    signatures on every run — which showed up as the LSH candidate count moving
    between two otherwise identical runs (299,034 vs 262,338). Every dedup
    result would have been unreproducible while the run registry faithfully
    recorded a fixed seed, which is worse than an obvious failure.

    crc32's distribution is weaker than a cryptographic digest, but the
    permutation step (a*x + b mod p) supplies the randomness that MinHash
    actually depends on, and crc32 is roughly an order of magnitude faster.
    """
    return np.fromiter(
        (zlib.crc32(s.encode("utf-8")) for s in shingle_set),
        dtype=np.uint64,
        count=len(shingle_set),
    )


def signature(text: str, k: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """128-element uint32 MinHash signature for one document."""
    sh = shingles(text, k)
    if not sh:
        return np.full(a.shape[0], MAX_HASH, dtype=np.uint32)
    h = _shingle_hashes(sh)
    return ((np.multiply.outer(h, a) + b) % MERSENNE_P).min(axis=0).astype(np.uint32)


def signatures(
    texts: list[str], k: int, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """`(n, num_perm)` uint32 signature matrix."""
    out = np.empty((len(texts), a.shape[0]), dtype=np.uint32)
    for i, text in enumerate(texts):
        out[i] = signature(text, k, a, b)
    return out


def band_hashes(sig: np.ndarray, seed: int = 0) -> np.ndarray:
    """`(n, BANDS)` uint64 band hashes.

    A dot product mod a prime rather than a cryptographic digest: this runs
    40M times and collisions are harmless, because every candidate pair is
    verified against the full signature afterwards.
    """
    n, num_perm = sig.shape
    if num_perm != BANDS * ROWS:
        raise ValueError(f"num_perm must be {BANDS * ROWS}, got {num_perm}")
    rng = np.random.default_rng(seed)
    coef = rng.integers(1, MERSENNE_P, ROWS, dtype=np.uint64)
    bands = sig.reshape(n, BANDS, ROWS).astype(np.uint64)
    return (bands * coef).sum(axis=2) % MERSENNE_P


def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> np.ndarray:
    """Estimated Jaccard similarity: the fraction of matching signature slots.

    Works on a single pair or on aligned `(m, num_perm)` matrices.
    """
    return (sig_a == sig_b).mean(axis=-1)


def star_cluster(n: int, edges: np.ndarray) -> np.ndarray:
    """Group nodes so every member is within one hop of its seed.

    Returns `label[i]` = the seed index of node i's group.

    This replaces connected components, and the reason is the whole Phase 2
    gate. Union-find merges A and Z whenever a chain A~B~...~Z exists with each
    consecutive pair above threshold, even though A and Z are unrelated. On the
    2026-08-04 corpus that produced a "duplicate group" of 84,657 members and a
    mean group size of 3,709 — chains, not groups. Raising the pairwise
    threshold cannot fix it: it only requires more hops, and with 12.9M edges
    hops are abundant.

    Star clustering bounds group diameter at 1 by construction: a member has to
    clear the threshold against the *seed*, not against some other member. The
    cost is that a genuine duplicate cluster wider than one hop gets split into
    several groups — which is the safe direction, because METHODOLOGY §2.4 is
    explicit that a false merge destroys real signal while a miss does not.

    Seeds are taken highest-degree first, ties by lowest index: the canonical
    copy of a template has the most neighbours, so it seeds its own group, and
    the ordering is deterministic so the same corpus yields the same groups.
    """
    label = np.arange(n, dtype=np.int64)
    if len(edges) == 0:
        return label

    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]

    nodes = np.arange(n)
    starts = np.searchsorted(src, nodes, side="left")
    ends = np.searchsorted(src, nodes, side="right")
    degree = ends - starts

    label[:] = -1
    # Only nodes with neighbours can seed a multi-member group; the rest are
    # singletons and are filled in at the end.
    seeds = np.lexsort((nodes, -degree))
    seeds = seeds[degree[seeds] > 0]

    for seed in seeds:
        if label[seed] != -1:
            continue
        label[seed] = seed
        neighbours = dst[starts[seed] : ends[seed]]
        free = neighbours[label[neighbours] == -1]
        label[free] = seed

    singletons = label == -1
    label[singletons] = nodes[singletons]
    return label

