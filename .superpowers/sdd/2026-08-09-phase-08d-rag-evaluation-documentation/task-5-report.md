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

## Fix round 1/5 — exact scored evidence reuse and pure import preflight

The Task 5 review found two Important orchestration gaps and one Minor evidence
wording error. The original full CLI scored one retrieval result and then let
`answer_question` retrieve and query-encode a second time; import opened the
database before it parsed the private artifact; and the engineering notes
mislocated the 2026-08-07 insufficient-credit error at provider preflight.

### RED evidence

- The focused five-test regression selection failed 5/5 on `207ab7b`.
- Evaluation rejected the requested `retain_results` and
  `scored_retrievals` interfaces, demonstrating that no scored evidence crossed
  the retrieval/answer boundary.
- Full CLI orchestration did not request or forward retained retrieval results.
- A malformed authoring header opened the database before raising, and its
  invalid fake connection then masked the original `ManifestError` during close.
- Replacing one valid authoring worklist with a different valid worklist during
  `db.bootstrap()` silently imported the replacement instead of rejecting the
  byte-identity change.
- A subsequent two-test privacy-hardening RED showed that default dataclass
  representations echoed private worklist bytes/questions and retrieved
  narratives from the new in-memory carriers.

### Fix

- Full evaluation now asks `run_retrieval_eval` for a frozen
  `RetrievalEvaluationRun`: the existing summary plus one immutable
  `ScoredRetrieval(question_id, RetrievalResult)` per question. Retrieval-only
  retains the original summary-only path and never retains or forwards answer
  evidence.
- Before any answer/provider work, answer evaluation requires exact question-ID
  coverage, revalidates component rankings and configured RRF, compares the
  carried corpus to the exact live cluster/company/model corpus, reconstructs
  the fused evidence from that corpus, and compares all three metric/latency rows
  to the same evaluation run's persisted identity. Omission, mutation, live
  corpus forgery, or persisted-metric tampering fails closed.
- The default Phase 8C answerer receives a fresh list copy of the exact carried
  fused tuple through its existing `retriever` dependency. Its full evidence,
  cache, schema, citation, usage, and outbox validators remain unchanged. The
  returned `AnswerResult.evidence` must still equal the scored fused tuple, so a
  custom caller cannot answer from a silently dropped subset.
- `prepare_authoring_import` reads exact bytes once and fully validates UTF-8,
  header, physical arity, 30-row count, privacy decisions, manifest field
  semantics, category balance, and relevance/answerability before bootstrap. It
  returns a frozen object containing the source path, SHA-256, exact bytes,
  parsed questions, and ID-only rows.
- DB-backed import rebuilds and verifies that prepared identity, compares the
  current source bytes and SHA-256 to reject bootstrap-time TOCTOU, validates
  only the prepared questions against database scope/privacy, and atomically
  writes only the prepared ID-only rows. Direct callers retain the original
  convenience API through an internal preflight.
- Sensitive prepared bytes/questions/rows and retained retrieval results are
  excluded from dataclass representations, while safe source/hash/question IDs
  remain available for diagnostics.
- Engineering notes now state the exact live sequence: authentication and model
  preflight succeeded on 2026-08-07, the subsequent messages request returned
  insufficient credit, and Phase 8D did not recheck the current balance on
  2026-08-11.

### Fresh fix verification

- Review regressions: 5 passed after the initial 5 expected RED failures;
  sensitive-representation hardening passed 2 after 2 expected RED failures.
- Required LLM/CLI/eval/answer/retrieval/run/SDK/schema/ground-truth/isolation
  selection: 434 passed, 1 expected missing-human-manifest skip in 40.65s.
- Full repository: 665 passed, 6 expected real-data skips in 42.10s.
- `ruff check src tests`, scoped format, and `git diff --check`: passed.
- No provider call, balance check, paid request, private artifact import, or
  human decision occurred. All previously documented external gates remain
  open. Fix commit pending scoped re-review.
