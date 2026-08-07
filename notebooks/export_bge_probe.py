"""Export what the bge-base comparison needs, in one short DB window.

Everything after this runs off files, so the encode can proceed in parallel with
B3 holding the DuckDB write lock. Paired by construction: the SAME texts are
encoded by both models and clustered with the SAME config, so any difference in
the partition is the encoder's.
"""
import os

import duckdb
import numpy as np

CUTOFF, FAMILY, N = "2024-01-01", "credit_reporting", 150_000
OUT = os.environ.get("ENCODER_PROBE_DIR", "data/artifacts/encoder_probe")
MINILM = "sentence-transformers/all-MiniLM-L6-v2"

con = duckdb.connect("data/harmscope.duckdb", read_only=True)
dedup = con.execute(
    """SELECT run_id FROM runs WHERE phase='dedup' AND status='ok'
       AND json_extract_string(params_json,'$.params.cutoff') = ?
       ORDER BY started_at DESC LIMIT 1""", [CUTOFF]).fetchone()[0]

rows = con.execute(
    """SELECT d.complaint_id, e.row_idx, n.text_redacted
       FROM dup_groups d
       JOIN complaints c USING (complaint_id)
       JOIN narratives n USING (complaint_id)
       JOIN embedding_map e USING (complaint_id)
       WHERE d.run_id = ? AND d.is_representative AND c.product_family = ?
         AND c.date_received < ? AND e.model = ?
       ORDER BY d.complaint_id""",
    [dedup, FAMILY, CUTOFF, MINILM]).fetchnumpy()
con.close()

n_all = len(rows["complaint_id"])
rng = np.random.default_rng(42)
pick = np.sort(rng.choice(n_all, size=min(N, n_all), replace=False))
print(f"{FAMILY} {CUTOFF}: {n_all:,} reps, sampling {len(pick):,}")

np.save(f"{OUT}/bge_row_idx.npy", rows["row_idx"][pick].astype(np.int64))
np.save(f"{OUT}/bge_complaint_id.npy", rows["complaint_id"][pick].astype(np.int64))
# Object array, not a text file: newline-joining would make bge see whitespace-
# normalized text while MiniLM encoded `text_redacted` verbatim, and that
# difference would land in the cross-encoder ARI as if it were the encoder's.
np.save(f"{OUT}/bge_texts.npy", rows["text_redacted"][pick], allow_pickle=True)
print("exported")
