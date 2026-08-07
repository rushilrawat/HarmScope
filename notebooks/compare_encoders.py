"""Is the partition encoder-sensitive? MiniLM vs bge-base, same texts, same config.

The README has said since Phase 4 that the gate numbers are "provisional" on the
dev model. This answers the question that flag is actually about — whether the
encoder can move the partition the whole comparison rests on — without a 16.8 h
re-encode of the full corpus and a re-run of Phases 4-7.

Two guards, both because of mistakes this project already made:

**The cross-encoder ARI is meaningless without its within-encoder baseline.**
Disjoint-halves ARI on MiniLM is already 0.505 (README): the partition disagrees
with *itself* that much under resampling alone. So a cross-encoder ARI of 0.45
is not evidence the encoder matters. The decisive quantity is cross-encoder ARI
against within-encoder ARI **on this same sample**. Both use
`src.cluster.stability` — the same held-out protocol and the same noise-excluding
`ari()` that produced the recorded 0.505 — because two ARIs computed differently
are not comparable, which is the whole point of the exercise.

**`assign_max_distance = 0.35` is calibrated to MiniLM's geometry.** bge-base
sits on a different cosine scale, so a fixed 0.65 similarity floor moves the
assignment rate for reasons unrelated to partition quality — the same defect
class as an absolute `min_supporting_groups` floor across systems of different
granularity. The nearest-centroid similarity distributions are printed first; if
they are shifted, assignment rate and noise fraction are not comparable at a
fixed threshold and the matched-quantile figures are the honest ones.
"""
import os

import numpy as np

from src.cluster import assign as A
from src.cluster import fit as F
from src.cluster import stability as S
from src.config import CONFIG

OUT = os.environ.get("ENCODER_PROBE_DIR", "data/artifacts/encoder_probe")
FAMILY, SEED, CFG = "credit_reporting", CONFIG.seed, CONFIG.cluster

row_idx = np.load(f"{OUT}/bge_row_idx.npy")
X = {
    "MiniLM": F.read_vectors(
        np.load("data/artifacts/embeddings.all-MiniLM-L6-v2.npy", mmap_mode="r"),
        row_idx,
    ),
    "bge-base": np.ascontiguousarray(np.load(f"{OUT}/bge_vectors.npy")),
}
n = len(row_idx)
rows_all = np.arange(n)
print(f"{n:,} {FAMILY} representatives at the 2024 cutoff")
print(f"dims: MiniLM {X['MiniLM'].shape[1]}, bge-base {X['bge-base'].shape[1]}\n")

# --- granularity and the similarity scale ----------------------------------
full = {}
for name, V in X.items():
    res = F.fit_family(V, rows_all, FAMILY, CFG, SEED, log=lambda *a: None)
    labels, sims = A.assign(V, rows_all, res.centroids, CFG.assign_max_distance)
    full[name] = (res, labels, sims)
    q = np.quantile(sims, [0.10, 0.50, 0.90])
    print(f"{name:<9} clusters {res.n_clusters:>5}   fit-noise "
          f"{res.fit_noise_fraction:5.1%}   assigned {(labels != -1).mean():5.1%}")
    print(f"{'':<9} nearest-centroid cosine  p10 {q[0]:.3f}  p50 {q[1]:.3f}  "
          f"p90 {q[2]:.3f}   (fixed floor {1 - CFG.assign_max_distance:.2f})")

print("\nsame comparison at a matched quantile of each encoder's own scale:")
for name, (_, _, sims) in full.items():
    thr = [f"q{int(q * 100)} {np.quantile(sims, q):.3f}" for q in (0.25, 0.50, 0.75)]
    print(f"  {name:<9} {'  '.join(thr)}")

# --- ARI: cross-encoder against the within-encoder baseline ----------------
# Cross-encoder mirrors disjoint_halves' protocol: carve the held-out set first,
# fit both encoders on the remaining pool, assign the same held-out points.
rng = np.random.default_rng(SEED)
held = np.sort(rng.choice(rows_all, size=min(50_000, n // 4), replace=False))
pool = np.setdiff1d(rows_all, held)

cross = {}
for name, V in X.items():
    res = F.fit_family(V, pool, FAMILY, CFG, SEED, sample_size=len(pool),
                       log=lambda *a: None)
    cross[name], _ = A.assign(V, held, res.centroids, CFG.assign_max_distance)

value, n_cmp = S.ari(cross["MiniLM"], cross["bge-base"])
print(f"\nARI, noise-excluded, on {len(held):,} held-out points:")
print(f"  cross-encoder  MiniLM vs bge-base      {value:.3f}  (n={n_cmp:,})")
for name, V in X.items():
    d = S.disjoint_halves(V, rows_all, FAMILY, CFG, SEED, log=lambda *a: None)
    print(f"  within-encoder disjoint halves, {name:<9} {d['ari']:.3f}  "
          f"(n={d['n_compared']:,}, {d['n_clusters'][0]} vs {d['n_clusters'][1]} clusters)")
print("\nREADME records 0.505 for this family on the full MiniLM population.")
