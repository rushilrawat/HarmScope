# Phase 8 — LLM layer

**State:** 🔄 engineering built and automated gates pass; frozen benchmark,
live provider, and human review remain open · **Effort:** ~1 week

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

## What runs

1. Resumable structured cluster labeling with atomic cache, retries, usage, and
   a blinded ≥50-label review workflow.
2. Cluster/company-scoped exact dense retrieval, BM25, reciprocal-rank fusion,
   grounded answers, citation validation, and separate response/context display.
3. A private 30-question authoring/import workflow, three-variant retrieval and
   answer metrics, run provenance, and blinded ≥50-claim human review.
4. Lazy CLI commands: `run --phase label`, `label-verify`, `ask`, and the
   `rag-eval` run/author/import/claims actions.

## Tech

| Choice | Why this one |
|---|---|
| **`claude-opus-5`** via one typed Anthropic boundary | Configured for labels and evidence synthesis only; live billing gate pending. |
| **Atomic caches + usage outbox** (`data/artifacts/llm_cache/`) | Resume without rebilling, validate hits, and preserve paid usage across failures. |
| **Exact FAISS + `rank-bm25` + RRF** | Dense semantics and lexical detail over the same scoped evidence, with component losses visible. |
| **DuckDB run/cache/evaluation tables** | Bind outputs to model, prompt, manifest, evidence, config, Git SHA, tokens, latency, and cost. |
| **Closed structured output + deterministic validators** | Schema/citation/scope failures are visible typed outcomes, never regex-repaired prose. |
| **pytest/TDD + blinded human worklists** | Software behavior is automated; label quality and groundedness retain real human denominators. |

## Acceptance

- **Automated — passes:** no detection package imports/queries the LLM layer;
  `signals` is byte-identical with `src/llm` absent. `baseline_results` is
  byte-identical given the same human-written `backtest_links`.
- **Built/tested — passes:** resume/accounting, retrieval provenance, grounded
  answer validation, three-variant metrics, evaluation registry, and blinded
  review workflows.
- **Live — pending:** paid 20-label pilot, full label population, and full
  answer evaluation.
- **Human — pending:** ≥50 label reviews and ≥50 grounded-claim reviews.
- **Measured RAG — pending:** frozen human-authored 30-question manifest plus
  real dense/BM25/fused and answer metrics.

## What is built (through 2026-08-11)

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

The later Phase 8 work adds:

- `client.py`, `run.py`, and `verify.py`: terminal/retryable provider taxonomy,
  exact usage/cost accounting, resumable label population, and version-bound
  human label review.
- `retrieve.py`: the same deduplicated/campaign-filtered population as signals,
  cluster-run model provenance, SHA-addressed embedding artifacts, exact dense
  ranking, versioned BM25, and deterministic RRF.
- `answer.py`: structured complaint-cited claims, deterministic abstention,
  cache/resume, fsynced usage outbox, and allegation/context separation.
- `eval.py`: human-authored manifest/privacy contract, Recall@10/MRR and
  win/tie/loss, answer/cost metrics, and unforgeable blinded claim review.
- `pipeline.py`: lazy evaluation actions. Manifest bytes are loaded before a
  writable DB, run parameters contain exact manifest SHA/model/mode, partial
  failures remain failed, retrieval-only cannot touch the provider, and 90 rows
  are declared only after stable report rendering.

## What is blocked

The old “credentials expired” diagnosis was incorrect. The token auto-refreshed
and the 2026-08-07 request authenticated; Anthropic then returned terminal
`400 invalid_request_error` because the organization had insufficient credit.

Billing is only one prerequisite. The human-authored/privacy-reviewed
30-question manifest is absent. The private blank draft has 18/30 rows without
the concrete company scope grounded answers require, and its five
`company_response` candidates expose complaint narratives rather than the
independent response evidence needed to author that category honestly. The
real MiniLM vectors also retain the old model-tail filename and need validated
regeneration under the full-model SHA contract.

The exact network-free command was attempted on 2026-08-11. It exited 1 with
`FileNotFoundError` for `data/ground_truth/rag_eval_questions.csv` before
opening the database and left zero `rag-eval` run rows. That is an honest
prerequisite failure, not a retrieval score.

After those decisions/artifacts exist: fund the provider, run the paid label
pilot, record ≥50 human label reviews, run the full answer benchmark, and record
≥50 human claim reviews. Until then, agreement, Recall@10/MRR, citation,
abstention, token/cost, and groundedness values are PENDING.

## Open risk

The adjudicator in Phase 6 was already a model (`claude-opus-5`), recorded as
`model_adjudicated_blind` rather than passed off as human labelling. Adding an
LLM labelling layer increases the surface where model output could reach a
number that is reported as measurement. The determinism test is the guard, and it
is the reason it is scoped to `signals` rather than waved at generally.
