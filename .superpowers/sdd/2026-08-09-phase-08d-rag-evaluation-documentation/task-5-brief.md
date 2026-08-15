# Phase 8D Task 5 — evaluation CLI, run provenance, observability, and honest documentation

Baseline: `8e08a4b721daa6a80786cf75645bdb5573280152`.

Governing plan: `docs/superpowers/plans/2026-08-09-phase-08d-rag-evaluation-documentation.md`, Task 5 and Phase 8D completion gate.

## Goal

Expose all evaluation/authoring/review workflows through lazy CLI handlers, register reproducible evaluation runs with exact manifest identity, render every metric and pending gate, and reconcile project documentation to measured reality without inventing live or human results.

## Files

- Modify `src/pipeline.py`, `tests/test_llm_cli.py`, and `src/llm/eval.py`/`tests/test_llm_eval.py` only as needed for orchestration/combined summaries.
- Modify `README.md`, `docs/LLM_LAYER.md`, `docs/ENGINEERING_NOTES.md`, `docs/phases/phase-08-llm.md`, `docs/ROADMAP.md`, and the approved Phase 8 design spec only through explicit amendment sections for real deviations.
- Do not create or import a committed benchmark, paid labels/answers, human reviews, or measured RAG metrics without their real prerequisites.

## CLI and run contracts

- Lazy-import commands: `rag-eval` (default full run), `rag-eval --retrieval-only`, `rag-eval author --output PATH`, `rag-eval import --input PATH`, `rag-eval claims-export --run-id ID --output PATH`, `rag-eval claims-record --run-id ID --input PATH --reviewer ID`.
- Pure argument/path/manifest validation occurs before writable DB/run creation or provider work. Every handler closes its connection in `finally` and preserves typed original errors.
- Run loads and DB-validates the exact frozen 30-question manifest, computes the file SHA-256, and creates `db.run(con, 'rag-eval', CONFIG, params=...)` with exact `manifest_sha256`, embed model, and `retrieval_only`. It finishes only after all requested work succeeds, with `output_rows=90`; failed/interrupted runs remain visibly failed/incomplete and cannot export claims.
- Retrieval-only never constructs/preflights a provider. Full mode runs retrieval then answer evaluation under the same run ID. Do not rerun the query encoder beyond the Task 2 one-call-per-question contract.
- Combined stable rendering names answerable/unanswerable counts; Recall@10/MRR/latency for dense, BM25, fused; both fusion win/tie/loss comparisons and per-question losses; citation validity/coverage/abstention; token/cache/outcome/latency/cost totals; human groundedness status; and eval run ID. Pending answer/human gates render `n/a`/PENDING, never zero/pass.
- Author/import print only paths/counts and never narrative text. Import targets the exact ground-truth manifest and cannot set human provenance itself. Claims export requires a completed run whose `manifest_sha256` matches the current manifest. Claims record requires input under configured interim with filename ending `.<reviewer_id>.csv` and prints denominator-bearing Wilson output.
- Parser defaults and help are deterministic; evaluation imports remain function-local so deleting `src/llm` leaves detection import/execution boundaries intact.

## Required honest status and documentation

- Engineering implementation may be called complete only where independently reviewed and tested. Explicitly list open gates: human author/privacy review and freeze of the 30-question manifest; resolution/regeneration of 18 blank company scopes; resolution of unsupported `company_response` category; legacy embedding artifact uses the pre-SHA filename contract and requires a validated migration/regeneration before the network-free real run; Anthropic billing/live labeling/full-answer run; 50-label and 50-claim human review.
- Never present the current private draft as ground truth, current old embedding file as accepted by the new loader, fake-client metrics as measured model performance, or skipped real-DB gates as passes.
- README explains the deterministic detector versus descriptive LLM boundary and names the implemented AI/ML stack naturally: structured outputs, caching/resume, dense embeddings, exact vector search/FAISS, BM25, reciprocal-rank fusion, DuckDB provenance, TDD/evaluation, and blinded human review.
- LLM_LAYER documents exact current interfaces, error/cache/outbox/run identities, benchmark/review protocol, model/pricing provenance, and why live metrics are pending.
- ENGINEERING_NOTES records commands, hashes/run IDs/metrics only when real, plus exact blockers/cost state.
- Phase 8 phase doc uses a built/automated/live/human matrix. ROADMAP corrects acceptance to signals byte-identical with `src/llm` absent and baseline results identical given the same `backtest_links`.
- Design spec uses a dated amendment for final deviations; do not rewrite history.

## TDD and verification

Use strict RED→GREEN for parser defaults, lazy import, every action, connection closure, preflight ordering, run status/params/hash/output rows, retrieval-only provider isolation, partial failure, summary labels/pending values, filename/path contracts, and detection isolation. Run CLI/eval/answer/retrieval/run/schema/ground-truth/isolation suites, full pytest, Ruff, scoped format, and diff check. Run only network-free commands whose prerequisites truly exist; when blocked, capture exact failure without mutating ground truth or weakening provenance. Self-review docs against code and Git history, write `task-5-report.md`, update ledger, commit, and leave a clean tree.
