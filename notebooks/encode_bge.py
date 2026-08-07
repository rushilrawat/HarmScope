"""Encode the exported sample on bge-base-en-v1.5, into a scratch memmap.

Reuses `src.embed.encode.encode_batch` rather than calling `model.encode`
directly: the first+last 2000-char windowing and mean-pooling must be identical
on both sides, or that difference lands in the cross-encoder ARI as if it were
the encoder's. Touches no database, so it runs while B3 holds the write lock.
"""
import os
import time

import numpy as np

from src.embed.encode import encode_batch, load_model

OUT = os.environ.get("ENCODER_PROBE_DIR", "data/artifacts/encoder_probe")
MODEL = "BAAI/bge-base-en-v1.5"
BATCH = 128

# allow_pickle: this file is a numpy object array of strings written by
# export_bge_probe.py in this same scratch directory a few minutes earlier. Not
# untrusted input.
texts = np.load(f"{OUT}/bge_texts.npy", allow_pickle=True)
model, device = load_model(MODEL)
dim = model.get_sentence_embedding_dimension()
print(f"{len(texts):,} texts, {MODEL}, dim {dim}, device {device}", flush=True)

path = f"{OUT}/bge_vectors.npy"
out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                shape=(len(texts), dim))
t0 = time.time()
for i in range(0, len(texts), BATCH):
    chunk = list(texts[i:i + BATCH])
    out[i:i + len(chunk)] = encode_batch(model, chunk)
    if i % 12_800 == 0 and i:
        rate = i / (time.time() - t0)
        print(f"  {i:,}/{len(texts):,}  {rate:,.0f}/s  "
              f"eta {(len(texts) - i) / rate / 60:.0f} min", flush=True)
out.flush()
print(f"done in {(time.time() - t0) / 60:.1f} min -> {path}")
