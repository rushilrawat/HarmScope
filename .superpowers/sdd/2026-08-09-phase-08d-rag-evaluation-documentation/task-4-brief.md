# Phase 8D Task 4 — blinded fifty-claim human groundedness gate

Baseline: `af2e4e727d2361dfe3cc1e17632c76801ca42bc7`.

Governing plan: `docs/superpowers/plans/2026-08-09-phase-08d-rag-evaluation-documentation.md`, Task 4 and Global Constraints.

## Goal

Implement a deterministic, privacy-safe local worklist for at least fifty generated claims, strict human review parsing, and idempotent question-level groundedness persistence with a Wilson interval. Never substitute an automated judge for the human denominator.

## Files

- Modify `src/llm/eval.py`.
- Modify `tests/test_llm_eval.py`.
- Do not change schema, answer, or verification code unless a reproduced incompatibility is approved.

## Required interfaces and semantics

- `export_claim_review(con, eval_run_id: str, n: int, seed: int, path: Path) -> Path`.
- `parse_claim_review(path: Path, reviewer_id: str) -> list[ClaimReview]`.
- `record_claim_review(con, eval_run_id: str, reviews: list[ClaimReview]) -> GroundednessReport`.
- Resolve evaluation questions through the frozen manifest plus exact run-linked answer usage/cache identity. Fail clearly and ID-only when the manifest is absent, a question cannot be linked unambiguously to its cached answer, the eval run lacks fused rows, or fewer than `n` eligible claims exist. Do not guess among cache rows.
- Sample exactly `n` unique claims with deterministic seed and hash tie-break, stratifying across category, answerable status, company, and claim position as available. Require at least 50 for the human gate.
- Export question, claim text, cited complaint IDs, and only the cited redacted evidence excerpts. Exclude model confidence, expected answerability, retrieval scores/ranks, model outcome, and any field that reveals the expected judgment.
- The private worklist must stay under the configured ignored `data/interim`, use strict exact columns/row arity, neutralize spreadsheet formulas/control hazards, write atomically and durably, and never echo claim/evidence/question prose in logs/errors/tests/Git.
- Give every review row a deterministic unforgeable identity bound to eval run, question ID, claim position/text hash, citations, and evidence identity so edited or cross-run worklists fail closed without storing prose in the identity.
- Human columns: `grounded=yes|no`, `failure_category`, `notes`. `yes` requires `failure_category=none`; `no` requires one of `unsupported`, `contradicted`, `overgeneralized`, `citation_mismatch`, `other`. Reviewer ID must be a safe nonblank identifier and appear in the report; filename/reviewer provenance must follow the plan without a new table.
- `record_claim_review` requires at least 50 unique valid reviews, verifies exact eval-run/question/claim identity against local cached answers and existing fused rows, and atomically upserts only `grounded_claims`/`reviewed_claims` on fused rows. Preserve retrieval, answer, and prior unrelated columns. Replays are idempotent and cannot double count.
- `GroundednessReport` reports grounded, reviewed, rate, Wilson low/high, reviewer ID, and explicit gate status. Never call the gate complete for fewer than 50 reviews; malformed or mixed-reviewer artifacts fail before DB mutation.
- The engineering workflow may use synthetic private fixtures, but no review may be marked human unless a person actually completed it. No real provider/human review is required for code verification; those remain external gates.

## TDD and verification

Use strict RED→GREEN. Cover manifest/cache ambiguity, insufficient claims, deterministic/stratified sampling, exact 50, cited-only evidence, excluded blinding fields, formula/control safety, row arity, tamper/cross-run detection, reviewer ID, yes/no/failure rules, duplicate reviews, fewer-than-50 rejection, Wilson values, fused-only atomic persistence, replay, rollback, and no prose leakage. Run Task 4 tests, eval/answer/verify/schema/isolation suites, full pytest, Ruff, scoped format, and diff check. Self-review, write `task-4-report.md`, update ledger, commit, and leave a clean tree.
