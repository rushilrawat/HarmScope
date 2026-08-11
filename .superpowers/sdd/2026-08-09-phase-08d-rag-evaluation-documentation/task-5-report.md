# Phase 8D Task 5 implementation report

Status: **DONE_WITH_CONCERNS** — the lazy CLI, exact run provenance, combined
observability, and documentation reconciliation are implemented and verified.
The engineering surface is complete; the real benchmark, paid-provider, and
human-review gates remain deliberately open.

Implementation commit: this report's enclosing `Complete Phase 8 RAG
evaluation` commit.

## Implemented

- Added lazy `rag-eval` orchestration with full mode as the parser default plus
  `--retrieval-only`, `author`, `import`, `claims-export`, and `claims-record`.
  The detection pipeline still imports no LLM evaluation module at startup.
- Run mode reads and hashes the exact frozen manifest before database bootstrap,
  rechecks the bytes against concurrent replacement, validates every row against
  the database, and derives one recorded embedding model from the referenced
  cluster runs.
- One `db.run(..., phase='rag-eval')` records exactly `manifest_sha256`,
  `embed_model`, and `retrieval_only`. Retrieval and full-mode answering share
  its run ID; the run finishes with 90 output rows only after requested work and
  stable rendering both succeed. Typed failures preserve their identity and
  leave the run failed.
- Retrieval-only mode never constructs or preflights a provider. It calls the
  reviewed Task 2 runner once for the 30-question batch, preserving the
  one-retrieval-call-per-question query-encoder contract.
- Combined rendering includes the run ID, answerable/unanswerable counts, every
  dense/BM25/fused retrieval metric and comparison, answer metrics/accounting or
  deterministic `n/a`, and an explicit pending 50-claim human-groundedness gate.
- Author/import output contains only the artifact path and row count. Import
  targets exactly `data/ground_truth/rag_eval_questions.csv`.
- Claim export binds the current manifest to a completed run. Claim record
  validates interim containment and the required `.<reviewer_id>.csv` suffix
  before database access, then prints the denominator-bearing Wilson report.
- Every database-owning handler closes its connection in `finally`; argument,
  containment, artifact, identifier, and manifest preflights occur before a
  writable run is created.

## Strict TDD evidence

- RED 1: eight CLI tests failed because `rag-eval` and its actions did not exist.
  GREEN 1: parser defaults, exact run parameters/output rows, all actions,
  provider isolation, closure, and failure state passed.
- RED 2: two evaluator tests failed because manifest model derivation and the
  combined renderer did not exist. GREEN 2 added the single-model provenance
  check and stable full/retrieval-only output.
- RED 3: a rendering exception occurred after the run had already been marked
  successful. GREEN 3 moved rendering inside the run context before `finish`,
  and the regression observes a failed run with no success output.
- RED 4: import printed extra human guidance instead of the exact path/count-only
  contract. GREEN 4 reduced every artifact action to the required stable output.

## Network-free real-data exercise

With `HARMSCOPE_DATA_DIR=/Users/rushilrawat/HarmScope/data`, the exact command
`.venv/bin/python -m src.pipeline rag-eval --retrieval-only` exited 1 at the
honest first prerequisite: `FileNotFoundError` for the absent frozen
`data/ground_truth/rag_eval_questions.csv`. This happened before database
bootstrap; a read-only follow-up found zero `runs` rows for phase `rag-eval`.
No provider was contacted, no paid request occurred, and no run, hash, or metric
was fabricated.

## Fresh verification

- Required CLI/eval/answer/retrieval/run/schema/ground-truth/isolation selection:
  427 passed, 1 expected missing-human-manifest skip in 42.45s.
- Full repository: 659 passed, 6 expected real-data skips in 43.43s.
- `ruff check src tests`: passed.
- Scoped `ruff format --check`: passed.
- `git diff --check`: passed.

## Documentation reconciliation and open gates

README, LLM layer, engineering notes, Phase 8, roadmap, and a dated design-spec
amendment now distinguish built/tested engineering from live model evidence and
human judgment. They preserve the detector/LLM boundary and name only the stack
actually present in code.

The following remain external prerequisites, not implementation passes:

1. Human authorship, privacy review, and freeze of the 30-question manifest.
2. Human/controller resolution or regeneration of the 18 blank company scopes.
3. A reviewed evidence source or design decision for the unsupported
   `company_response` category; the current worklist has complaint evidence only.
4. Validated migration or regeneration of legacy short-name embedding artifacts
   to the current full model-SHA filename contract.
5. Anthropic billing, live labeling, and a full paid answer evaluation.
6. The blinded 50-label agreement review and 50-claim groundedness review.

The ignored private authoring draft was neither committed nor treated as ground
truth. Fake-client tests were not reported as model quality, and skipped
real-data tests were not represented as passing external gates.
