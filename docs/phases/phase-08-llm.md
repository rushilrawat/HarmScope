# Phase 8 — LLM layer

**State:** 🔄 deterministic core built and its acceptance criterion met; labelling
blocked on credentials · **Effort:** ~1 week

## Question

Can a cluster be given a human-readable harm description without the LLM
touching detection?

## Context

This is the phase most likely to quietly invalidate everything before it, and the
architecture is built to make that impossible rather than unlikely.

**The LLM labels; it never detects.** Every number in the results table comes
from `signals` and `baseline_results`, both computed by deterministic statistics
over `clusters`. If an LLM call could influence which clusters fire, the backtest
would be measuring a non-reproducible system and no threshold sweep or negative
control would mean anything.

**The determinism claim has to be exactly scoped.** The honest pair of
statements:

- `signals` is byte-identical with `src/llm/` deleted. **Detection is fully
  deterministic and LLM-free.**
- `baseline_results` is byte-identical *given the same `backtest_links`*.
  Adjudication is a deliberate human step; LLM output reaches it through a
  person, by design.

Claiming byte-identical `baseline_results` unconditionally would be a claim the
pipeline cannot honour, and overclaiming here would undermine the one
architectural guarantee this project actually has.

> **Note:** ROADMAP Phase 8's criterion and `LLM_LAYER.md` both said
> `backtest_results` until 2026-08-06. Nothing writes that table —
> `src/evaluation/backtest.py` writes `baseline_results`. Phase 8 would have been
> built to prove a property of a table that is never populated.

## What runs (planned)

1. Cluster labelling with caching, guardrails, structured output.
2. Hybrid retrieval (BM25 + dense) + grounded answers.
3. Human verification of ≥ 50 labels.

## Tech (planned)

| Choice | Why this one |
|---|---|
| **`claude-sonnet-5`** via the Anthropic API | Labelling and evidence synthesis only. |
| **On-disk response cache** (`data/artifacts/llm_cache/`) | Labels must be reproducible across runs without re-billing, and a cache hit makes the determinism test cheap to run. |
| **`rank-bm25`** alongside the FAISS index | Complaint narratives carry statute names and product terms that lexical search finds and dense retrieval blurs. |
| **Structured output** | A label that does not parse is a failure, not a string to regex. |

## Acceptance

- **Determinism test passes:** removing `src/llm/` leaves `signals` and
  `baseline_results` byte-identical.
- Label agreement rate reported on ≥ 50 human-verified labels.
- RAG Recall@10 and groundedness on the 30-question set.

## What is built (2026-08-07)

`src/llm/select.py` and `src/llm/label.py` — everything in the layer that does
not need a network call:

- **Input selection** (§2.1): 12 nearest the medoid + 8 by maximal marginal
  relevance. The medoid half says what the cluster is centrally about; the MMR
  half is what stops a confident label being written about a mail merge.
- **The cache key** (§2.4): `sha256(prompt_version + model + sorted ids)`. Sorted,
  so it does not depend on selection order; `prompt_version` is inside it, which
  is what makes §4's bump-on-edit rule actually invalidate anything.
- **The output contract** (§2.2): enforced by the API through structured outputs
  (`output_config.format`) rather than requested in prose. The older way to force
  JSON — prefilling an assistant turn with `{"` — returns a 400 on every current
  model, and the "output ONLY valid JSON" prompt-begging that accompanied it is
  dead weight once the schema is enforced.
- **The guardrails** (§2.3) in a system prompt with a `cache_control` breakpoint,
  so they are written once and read at ~0.1× for every subsequent cluster. That
  is what makes §2.4's "label lazily, batch where you can" affordable.

**The acceptance criterion that matters is met.** The determinism test is
asserted structurally — no module in `signals`, `cluster`, `dedup`, `embed`, or
`evaluation` may import `src/llm/` or `anthropic` — and verified empirically:
with `src/llm/` physically deleted, the signals, cluster, and leakage suites all
still pass.

It is deliberately *not* a two-run byte-comparison. That needs hours of refit, so
it would be skipped in CI and therefore never run — and a leakage check that
never runs is the exact defect this repo shipped once already (`de2dbc5`).

## What is blocked

Label generation and the RAG layer need a working API credential. `ant auth
status` reports a profile for this account whose token **expired 2026-07-20**, so
nothing can call the API until it is refreshed with `ant auth login`. Everything
above was built and tested without one; §2.5's human verification and §3.5's
RAG metrics cannot start until it exists.

## Open risk

The adjudicator in Phase 6 was already a model (`claude-opus-5`), recorded as
`model_adjudicated_blind` rather than passed off as human labelling. Adding an
LLM labelling layer increases the surface where model output could reach a
number that is reported as measurement. The determinism test is the guard, and it
is the reason it is scoped to `signals` rather than waved at generally.
