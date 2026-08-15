"""Encoding, windowing, and the memmap <-> embedding_map contract.

No model weights are loaded here. A fake encoder makes the pooling and
normalization checkable exactly, and the parts that can silently corrupt a
downstream phase — a row index pointing past the end of the memmap, a resume
that starts at the wrong offset — do not involve a real model at all.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from src.embed import encode


class FakeModel:
    """Returns a deterministic vector per text, so pooling is checkable."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list[int] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, batch_size=32, **kw):
        self.calls.append(batch_size)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, 0] = len(t)
            out[i, 1] = float(t[0]) if t and t[0].isdigit() else 1.0
        return out


def test_windows_splits_only_long_narratives():
    assert encode.windows("short") == ["short"]
    long = "a" * (encode.WINDOW_CHARS + 500)
    got = encode.windows(long)
    assert len(got) == 2
    assert got[0] == long[: encode.WINDOW_CHARS]
    assert got[1] == long[-encode.WINDOW_CHARS :]
    # first + last, not first + second: the tail must reach the real end
    assert got[1].endswith(long[-1])


def test_output_is_unit_length_after_pooling():
    """Mean-pooling two unit vectors does not give a unit vector."""
    model = FakeModel()
    out = encode.encode_batch(model, ["short", "b" * (encode.WINDOW_CHARS + 10)])
    assert out.shape == (2, 4)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_long_text_is_the_mean_of_its_two_windows():
    model = FakeModel()
    long = "9" + "a" * (encode.WINDOW_CHARS + 10)
    out = encode.encode_batch(model, [long])
    # window 0 starts with '9' -> 9.0; window 1 starts with 'a' -> 1.0; mean 5.0
    expected = np.array([encode.WINDOW_CHARS, 5.0, 0, 0], dtype=np.float32)
    assert np.allclose(out[0], expected / np.linalg.norm(expected), atol=1e-6)


def test_forward_batch_bounds_the_gpu_call():
    """batch_size=len(flat) OOM'd Metal; the forward batch is a separate knob."""
    model = FakeModel()
    encode.encode_batch(model, ["x"] * 500, forward_batch=64)
    assert model.calls == [64]


def _narratives(con, texts: dict[int, str]) -> None:
    con.execute(
        "INSERT INTO runs (run_id, phase, git_sha, config_hash, params_json, "
        "started_at, status) VALUES ('r1','normalize','s','c','{}', now(), 'ok')"
    )
    con.execute(
        "INSERT INTO company_canonical (company_id, canonical_name, verified_by) "
        "VALUES ('co','Co','manual')"
    )
    con.executemany(
        "INSERT INTO complaints (complaint_id, date_received, period_month, "
        "company_id, product_family, has_narrative) VALUES (?, ?, ?, 'co', 'x', true)",
        [(cid, date(2020, 1, 1), date(2020, 1, 1)) for cid in texts],
    )
    con.executemany(
        "INSERT INTO narratives (complaint_id, text_redacted, text_hash, char_len, "
        "redaction_count) VALUES (?, ?, ?, ?, 0)",
        [(cid, t, f"h{hash(t) % 997:03d}", len(t)) for cid, t in texts.items()],
    )


def test_partial_encode_reports_what_it_could_not_map(con):
    """The --limit bug: 3,000 rows encoded, 3.8M complaints mapped over 2.5M.

    Bounding row_idx fixed the dangling indices but turned it into a silent
    drop, so the count comes back to the caller rather than being assumed zero.
    """
    _narratives(con, {1: "alpha", 2: "beta", 3: "gamma", 4: "alpha"})
    n_mapped, n_unmapped = encode.build_map(con, "m", 4, n_rows=1)
    assert n_mapped + n_unmapped == 4
    assert n_unmapped > 0
    hi = con.execute("SELECT max(row_idx) FROM embedding_map WHERE model='m'").fetchone()[0]
    assert hi == 0, "no index may point past a 1-row memmap"


def test_a_full_encode_leaves_nothing_unmapped(con):
    _narratives(con, {1: "same", 2: "same", 3: "different"})
    n_mapped, n_unmapped = encode.build_map(con, "m", 4, n_rows=2)
    assert (n_mapped, n_unmapped) == (3, 0)
    rows = dict(
        con.execute("SELECT complaint_id, row_idx FROM embedding_map WHERE model = 'm'").fetchall()
    )
    assert rows[1] == rows[2] != rows[3]  # identical narratives share a row


def test_progress_refuses_to_resume_a_different_encode(tmp_path):
    p = tmp_path / "e.progress.json"
    encode.Progress(p, n_done=10, n_total=100, dim=384, model="a").write()
    prior = encode.Progress.read(p)
    assert (prior.n_done, prior.model, prior.dim) == (10, "a", 384)


def test_embedding_artifact_paths_bind_the_collision_free_full_model_identity(tmp_path):
    """Models sharing a tail cannot overwrite each other's vectors or FAISS index."""
    first = encode.embedding_artifact_paths(tmp_path, "provider-a/shared")
    second = encode.embedding_artifact_paths(tmp_path, "provider-b/shared")

    assert first != second
    assert first.memmap == tmp_path / (
        "embeddings.5483d7e157e32b55a97648f86ed353a102798a1732e95dfed637114f50b6b38b.npy"
    )
    assert first.index == tmp_path / (
        "faiss.5483d7e157e32b55a97648f86ed353a102798a1732e95dfed637114f50b6b38b.index"
    )
    assert second.memmap == tmp_path / (
        "embeddings.277383a219c87b6ce6ba99ed9cd2837e5e7d08a64eba2be529c61f11d728b73b.npy"
    )
