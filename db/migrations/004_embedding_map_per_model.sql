-- 2026-08-05. Re-key embedding_map on (complaint_id, model).
--
-- complaint_id alone was the primary key, so the table could hold exactly one
-- model's row per complaint. METHODOLOGY §3 names two — bge-base-en-v1.5 as the
-- default and all-MiniLM-L6-v2 for dev iteration — and Phase 4's stability work
-- is the reason the dev model exists. With the old key, encoding with the second
-- model meant deleting the first, so the two could never be compared and the
-- documented dev/default split did not work at all.
--
-- Safe to drop and recreate: nothing has been encoded yet.

DROP TABLE IF EXISTS embedding_map;

CREATE TABLE embedding_map (
  complaint_id BIGINT NOT NULL,
  row_idx      BIGINT NOT NULL,   -- row in the .npy memmap and the FAISS index
  model        VARCHAR NOT NULL,
  dim          INTEGER NOT NULL,
  PRIMARY KEY (complaint_id, model)
);

-- Identical narratives share a row: the memmap is keyed on text_hash, since an
-- embedding is a pure function of the text. 3,830,002 narratives collapse to
-- 2,477,937 distinct texts.
CREATE INDEX IF NOT EXISTS idx_embedding_row ON embedding_map(model, row_idx);
