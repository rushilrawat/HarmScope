# Phase 8 Answer-Safety Decision (Approved and Implemented)

**Date:** 2026-08-11
**Approved:** 2026-08-12
**Status:** Implemented with network-free synthetic/fake-provider verification

## Problem

The prior answer validator proved citation membership, but it did not prove
that a cited complaint supports the model-written claim. A payload whose claim
is `The company violated federal law.` and whose complaint ID is in the
retrieved scope previously passed local validation. Citation scope alone cannot
turn generated prose into grounded evidence.

## Approved contract

Adopt extractive claims. Each model claim must select an exact contiguous span
from at least one of its cited redacted complaint narratives. The wire schema
continues to carry claim text and complaint IDs, but local validation receives
the cited evidence text and rejects any claim that is not an allowed extract.
Deterministic synthesis, not the model, adds explicit attribution such as
`Complaints allege: “…”` and renders the cited complaint IDs.

The contract should be precise:

- normalize Unicode to NFC and collapse whitespace for matching, without
  case-folding or semantic rewriting;
- require one contiguous span from at least one cited complaint for a multi-ID
  claim, while retaining all cited IDs as declared support;
- reject empty, control-only, over-length, or non-contiguous generated text;
- cap extracts at the existing bounded claim length and reject rather than
  truncate a model response;
- pass the exact retrieved redacted evidence into validation; IDs alone are
  insufficient;
- set `answer_prompt_version` to `rag-v3` so legacy cached answers remain
  separated and fail extractive validation if loaded explicitly;
- bind human claim-review identity to the validated extract, cited source
  bytes, prompt version, and deterministic rendered attribution.

This does not establish that the underlying allegation is true. It establishes
only that generated answer text is an attributed extract of the cited redacted
complaint evidence.

## Decision and implementation

The extractive contract was approved on 2026-08-12. `validate_answer` now
requires exact evidence text, cache/provider/evaluation paths propagate that
text, the prompt requests verbatim contiguous excerpts without model-added
attribution, and deterministic local rendering supplies the allegation frame
and IDs. One pure, versioned canonical renderer now supplies that exact string to
both CLI output and human claim-review export. Claim-review identity contains only
the renderer version and rendered-output hash, not private rendered prose, so a
prefix, escaping, or complaint-ID layout change invalidates prior review IDs. The
wire schema and database schema remain unchanged.

Network-free tests cover normalization, case sensitivity, contiguity, evidence
map shape, Unicode/control safety, bounded length, legacy cache rejection, paid
invalid-response accounting, renderer output, evaluation propagation, and
claim-review identity. No live provider benchmark or human groundedness review
has run, so answer quality and real-world groundedness remain pending.
