"""Phase 3: encode narratives to a float32 memmap.

docs/METHODOLOGY.md §3. Two decisions here are not in that section and are
worth stating, because both are about *what gets encoded* rather than how.

**The unit is a distinct text, not a complaint and not a representative.**
METHODOLOGY §3 says "representatives only". But representative selection is
refit per backtest cutoff (ENGINEERING_NOTES, reversed decision 2026-08-03),
while the same entry says embeddings are computed once and date-filtered — and
those two cannot both be true of a representative-keyed memmap. An embedding is
a pure function of its input text, so `text_hash` is the honest key: it is a
superset of every cutoff's representative set, it never needs refitting, and it
costs 2,477,937 encodes against 1,883,062 for a single cutoff's representatives
— cheaper than re-encoding at each of eight cutoffs, and immune to leakage for
the same reason MinHash is.

**Long narratives are encoded as first + last window, mean-pooled.** Also §3.
The p99 narrative is 6,116 characters against a 512-token window, and complaint
narratives routinely state the problem once at the top and again in the closing
demand, so truncation drops the half that is often more specific.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

# Characters, not tokens: the tokenizer truncates to the model window anyway, so
# an over-long slice is harmless and a char split avoids tokenizing twice. ~4
# chars/token for English prose against a 512-token window.
WINDOW_CHARS = 2000


@dataclass(frozen=True)
class Progress:
    """Sidecar next to the memmap, so a crash at 2.4M does not cost the run."""

    path: Path
    n_done: int
    n_total: int
    dim: int
    model: str

    @classmethod
    def read(cls, path: Path) -> Progress | None:
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        return cls(path=path, n_done=d["n_done"], n_total=d["n_total"],
                   dim=d["dim"], model=d["model"])

    def write(self) -> None:
        self.path.write_text(json.dumps({
            "n_done": self.n_done, "n_total": self.n_total,
            "dim": self.dim, "model": self.model,
        }))


def texts_to_encode(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """`(text_hash, text)` for every distinct narrative, in a stable order.

    Ordered by `text_hash` so row indices are reproducible: the memmap is
    written in this order and resumed by offset, which is only sound if the
    order does not depend on anything that can change between runs.
    """
    return con.execute(
        """
        SELECT text_hash, any_value(text_redacted)
        FROM narratives GROUP BY text_hash ORDER BY text_hash
        """
    ).fetchall()


def windows(text: str) -> list[str]:
    """One window, or first and last for a narrative longer than the model sees."""
    if len(text) <= WINDOW_CHARS:
        return [text]
    return [text[:WINDOW_CHARS], text[-WINDOW_CHARS:]]


def encode_batch(model, texts: list[str], forward_batch: int = 64) -> np.ndarray:
    """Encode with first+last mean-pooling, returned unit-normalized.

    Both windows of a long narrative go into the same `model.encode` call, so
    the two halves are never split across calls. But `forward_batch` bounds what
    goes through the GPU at once: passing `batch_size=len(flat)` here made a
    512-text batch one forward pass over 1,024 sequences of 512 tokens, which
    OOMs Metal on an M3 Pro. The outer batch size controls checkpoint spacing
    and memmap writes; this controls GPU memory, and they are not the same knob.
    """
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for text in texts:
        chunks = windows(text)
        spans.append((len(flat), len(flat) + len(chunks)))
        flat.extend(chunks)

    raw = model.encode(
        flat, batch_size=forward_batch, convert_to_numpy=True,
        normalize_embeddings=False, show_progress_bar=False,
    )
    out = np.empty((len(texts), raw.shape[1]), dtype=np.float32)
    for i, (lo, hi) in enumerate(spans):
        out[i] = raw[lo] if hi - lo == 1 else raw[lo:hi].mean(axis=0)
    # Unit length so cosine == inner product and FAISS IndexFlatIP is exact
    # (METHODOLOGY §3). Mean-pooling two unit vectors does not preserve norm,
    # so this has to happen after pooling, not inside model.encode.
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    np.divide(out, np.maximum(norms, 1e-12), out=out)
    return out


def load_model(name: str, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    if device is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(name, device=device), device


def encode_all(
    con: duckdb.DuckDBPyConnection,
    model_name: str,
    memmap_path: Path,
    batch_size: int = 128,
    device: str | None = None,
    limit: int | None = None,
    checkpoint_every: int = 50_000,
    forward_batch: int = 64,
    log=print,
) -> dict:
    """Encode every distinct narrative into `memmap_path`. Resumable, idempotent.

    Returns a stats dict. Re-running after a complete pass encodes nothing —
    ROADMAP Phase 3 requires that, and it falls out of the progress sidecar
    rather than needing a separate check.
    """
    rows = texts_to_encode(con)
    if limit:
        rows = rows[:limit]
    n_total = len(rows)

    model, device = load_model(model_name, device)
    dim = model.get_sentence_embedding_dimension()
    progress_path = memmap_path.with_suffix(".progress.json")
    prior = Progress.read(progress_path)

    if prior and (prior.model != model_name or prior.dim != dim
                  or prior.n_total != n_total):
        raise ValueError(
            f"{progress_path} describes a different encode "
            f"({prior.model}, dim {prior.dim}, {prior.n_total:,} texts) than this "
            f"one ({model_name}, dim {dim}, {n_total:,}). Delete it to re-encode."
        )
    start_at = prior.n_done if prior else 0
    if start_at >= n_total:
        log(f"already encoded: {n_total:,} texts, nothing to do")
        return {"encoded": 0, "n_total": n_total, "dim": dim, "device": device,
                "seconds": 0.0, "resumed_at": start_at}

    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+" if memmap_path.exists() and prior else "w+"
    out = np.lib.format.open_memmap(
        memmap_path, mode=mode, dtype=np.float32, shape=(n_total, dim)
    )
    if start_at:
        log(f"resuming at {start_at:,} of {n_total:,}")

    t0 = time.time()
    done = start_at
    for begin in range(start_at, n_total, batch_size):
        chunk = rows[begin : begin + batch_size]
        out[begin : begin + len(chunk)] = encode_batch(
            model, [t for _, t in chunk], forward_batch
        )
        done = begin + len(chunk)
        if done % checkpoint_every < batch_size or done == n_total:
            out.flush()
            Progress(progress_path, done, n_total, dim, model_name).write()
            rate = (done - start_at) / max(time.time() - t0, 1e-9)
            eta = (n_total - done) / max(rate, 1e-9)
            log(f"  {done:,}/{n_total:,}  {rate:,.0f} texts/s  eta {eta / 60:.1f} min")

    out.flush()
    Progress(progress_path, n_total, n_total, dim, model_name).write()
    elapsed = time.time() - t0
    return {
        "encoded": n_total - start_at, "n_total": n_total, "dim": dim,
        "device": device, "seconds": elapsed, "resumed_at": start_at,
        "rate": (n_total - start_at) / max(elapsed, 1e-9),
    }


def build_map(
    con: duckdb.DuckDBPyConnection, model_name: str, dim: int, n_rows: int
) -> int:
    """Populate `embedding_map`: every complaint_id -> its text's memmap row.

    Identical narratives share a row. The join goes through `text_hash` in the
    same `ORDER BY text_hash` the encoder used, so `row_idx` is defined by that
    ordering and by nothing else.

    `n_rows` is the number of rows actually in the memmap, and the filter on it
    is not optional. Without it a `--limit 3000` run mapped all 3,830,002
    complaints to indices computed over all 2,477,937 texts, so almost every row
    pointed past the end of a 3,000-row memmap — no error anywhere, and a
    downstream read would have silently returned whatever numpy found there.

    Returns `(n_mapped, n_unmapped)`. The filter turns that bug into a quieter
    one — complaints dropped from the map rather than pointing at nothing — so
    the caller is handed the count instead of being left to assume it is zero.
    A full encode must leave nothing unmapped; a `--limit` run legitimately does.
    """
    con.execute("DELETE FROM embedding_map WHERE model = ?", [model_name])
    con.execute(
        """
        INSERT INTO embedding_map (complaint_id, row_idx, model, dim)
        SELECT n.complaint_id, r.row_idx, ?, ?
        FROM narratives n
        JOIN (
          SELECT text_hash, row_number() OVER (ORDER BY text_hash) - 1 AS row_idx
          FROM (SELECT DISTINCT text_hash FROM narratives)
        ) r USING (text_hash)
        WHERE r.row_idx < ?
        """,
        [model_name, dim, n_rows],
    )
    n_mapped, n_distinct, hi = con.execute(
        "SELECT count(*), count(DISTINCT row_idx), coalesce(max(row_idx), -1) "
        "FROM embedding_map WHERE model = ?",
        [model_name],
    ).fetchone()
    if n_distinct != n_rows or hi != n_rows - 1:
        raise ValueError(
            f"embedding_map covers {n_distinct:,} rows with max index {hi:,}, "
            f"but the memmap has {n_rows:,} — every encoded row must be reachable."
        )
    n_narratives = con.execute("SELECT count(*) FROM narratives").fetchone()[0]
    return n_mapped, n_narratives - n_mapped
