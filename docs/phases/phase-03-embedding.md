# Phase 3 — Embedding & index

**State:** ✅ complete, on the dev model · **Effort:** ~3 days

## Question

Can complaint narratives be placed in a space where "the same harm mechanism"
means "nearby"?

## Context

Two constraints shaped this more than model choice did.

**Embeddings must not be refit per cutoff.** The backtest refits everything
date-dependent at eight annual cutoffs. An embedding is a pure function of its
input text, so it is not date-dependent — but *representative selection is*.
Those two facts cannot both be true of a representative-keyed memmap, which is
why the memmap is keyed on `text_hash` instead.

**The encoder choice is a 8× cost decision.** `bge-base-en-v1.5` is the declared
default; `all-MiniLM-L6-v2` is what `METHODOLOGY §3` names for iteration. 16.8 h
against 2.1 h for the corpus.

## What runs

1. `texts_to_encode` — one row per distinct `text_hash`, ordered by hash so
   memmap row indices are reproducible and resumable by offset.
2. Encode into a float32 memmap with a progress sidecar every 50k texts.
3. Build the FAISS index.
4. Ten hand-picked narratives; inspect their five nearest neighbours.

## Tech

| Choice | Why this one |
|---|---|
| **`sentence-transformers` 5.1.2** | Standard, CPU/MPS-viable. |
| **`all-MiniLM-L6-v2`** (384-d) in use | 386 texts/s vs bge's 41. See the standing caveat. |
| **FAISS `IndexFlatIP`** | Vectors are unit-length, so inner product *is* cosine and the index is **exact**, not approximate. |
| **First + last 2000-char windows, mean-pooled** | The p99 narrative is 6,116 chars against a 512-token window, and complaints routinely state the problem at the top and again in the closing demand. Truncation drops the half that is often more specific. |
| **`forward_batch` separate from batch size** | Passing `batch_size=len(flat)` made a 512-text batch one forward pass over 1,024 sequences of 512 tokens, which OOMs Metal on an M3 Pro. Checkpoint spacing and GPU memory are not the same knob. |

## Acceptance

> 10 narratives' 5 nearest neighbours topically correct; throughput logged;
> re-running encode is a no-op.

**Met.** 2,477,937 vectors, 10/10 neighbour checks correct, idempotence test
passes.

## Findings

**A `--limit` run silently corrupted the index and did not error.** It mapped
3.8M complaints to row indices over a memmap holding only 2.5M texts. Every index
past the end was out of range, with no exception raised — numpy memmap reads do
not bounds-check the way you want here. Caught by an explicit check, not by a
crash.

**The encoder recovers company identity through redaction.** A USAA deposit-hold
complaint returns five USAA deposit-hold complaints, despite the company name
being redacted. This is expected — institution-specific process language survives
PII removal — and it is why `related_clusters` is load-bearing: neighbours cross
`product_family` constantly.

**Length-sorting to cut padding waste was measured at a 7% gain and not built.**

## Standing caveat

Phases 3–5 ran on the dev model. Phase 9 has since measured what that costs —
see [Phase 9](phase-09-evaluation.md): cluster *identity* is encoder-sensitive
(cross-encoder ARI 0.232 against a within-encoder 0.469), while cluster
*granularity* is not (336 vs 350 clusters). Since granularity drives the Phase 7
comparison, the swap is unlikely to move the ordering — but the full encode
remains outstanding.
