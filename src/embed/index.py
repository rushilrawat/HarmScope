"""Phase 3: FAISS index over the embedding memmap.

docs/METHODOLOGY.md §3 specifies unit-normalized vectors so cosine == inner
product, which makes `IndexFlatIP` exact. ARCHITECTURE §5 named IVF-Flat for
scale; at 2.48M x 768 a flat index is 7.3 GB and a brute-force query is seconds,
which is fine for the handful of sanity queries Phase 3 accepts on and for
Phase 4's exemplar assignment. IVF is added when a query path actually needs
sub-second latency — see the ponytail note below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_vectors(memmap_path: Path) -> np.ndarray:
    return np.load(memmap_path, mmap_mode="r")


def build(memmap_path: Path, index_path: Path, batch: int = 100_000) -> int:
    """Build an exact inner-product index and write it to disk.

    ponytail: IndexFlatIP, not IVF-Flat. Exact, no training step, no nlist/nprobe
    to tune, and no recall loss to account for in Phase 4's ARI. Swap in IVF when
    a latency requirement exists to justify the recall trade — the callers here
    only need `search`, so the change stays inside this module.
    """
    import faiss

    vectors = load_vectors(memmap_path)
    index = faiss.IndexFlatIP(vectors.shape[1])
    for start in range(0, len(vectors), batch):
        index.add(np.ascontiguousarray(vectors[start : start + batch]))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    return index.ntotal


def load(index_path: Path):
    import faiss

    return faiss.read_index(str(index_path))


def neighbours(index, vectors: np.ndarray, row_idx: int, k: int = 5):
    """The `k` nearest rows to `row_idx`, excluding itself.

    Searches for k+1 and drops the self-hit rather than assuming it lands first:
    with duplicate vectors the ordering among ties is not defined.
    """
    query = np.ascontiguousarray(vectors[row_idx : row_idx + 1])
    scores, idx = index.search(query, k + 1)
    out = [(int(i), float(s)) for i, s in zip(idx[0], scores[0], strict=True)
           if int(i) != row_idx]
    return out[:k]
