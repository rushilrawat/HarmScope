# Phase 8 Extractive Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every non-abstaining HarmScope answer claim a locally verified extract from its cited redacted complaint evidence, then render it as an allegation rather than a fact.

**Architecture:** Keep the closed answer wire schema, but make `validate_answer` require the exact cited evidence text in addition to allowed IDs. Normalize claim and evidence text with NFC plus whitespace collapse for contiguous-span matching, reject unsafe/empty/oversized extracts, and render validated claims through deterministic allegation framing. Bump the answer prompt/cache contract to `rag-v3`; reuse the existing cache, usage, retrieval, evaluation, and human-review identity boundaries with exact evidence bytes.

**Tech Stack:** Python 3.12, DuckDB, Anthropic structured JSON, pytest, Ruff.

## Global Constraints

- Claims are allegations from redacted complaint narratives; the software must not present them as established facts or legal conclusions.
- A claim is valid only when its NFC-and-whitespace-normalized text is one contiguous span of at least one of its cited complaint narratives. Do not case-fold or semantically rewrite for matching.
- Multi-ID claims retain all declared unique in-scope IDs and need an exact extract match in at least one cited narrative.
- Reject empty, control-only, unsafe-control-bearing, non-contiguous, and over-`CONFIG.llm.max_narrative_chars` claim text; never truncate model output.
- Validation must receive exact retrieved redacted evidence text. An ID-only call is not a valid production boundary.
- The model cannot add allegation framing. Deterministic local rendering must display `Complaints allege: “<extract>”` with the declared complaint IDs.
- Set `LLMConfig.answer_prompt_version` to exactly `rag-v3`; legacy cache rows remain separated by prompt version and fail extractive validation if explicitly loaded.
- Claim-review identity must remain bound to claim text, cited source bytes, prompt version/input identity, and citation IDs. Do not add private prose to usage rows or reports.
- Do not change retrieval ranking, provider accounting, outbox durability, enforcement display-only behavior, database schema, or Phase 8A labeling behavior.
- The current worktree contains approved unstaged Phase 8 fixes. Preserve them. Do not stage or commit because the active Codex approval-layer Git block explicitly forbids retry/workarounds.

---

### Task 1: Enforce and propagate extractive grounded claims

**Files:**
- Modify: `src/config.py`
- Modify: `src/llm/answer.py`
- Modify: `src/llm/eval.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_llm_answer.py`
- Modify: `tests/test_llm_eval.py`
- Modify: `tests/test_llm_cli.py` only where fixtures must represent valid extracts
- Modify: `README.md`
- Modify: `docs/LLM_LAYER.md`
- Modify: `docs/superpowers/specs/2026-08-11-phase-08-extractive-answer-safety-decision.md`
- Modify: `.superpowers/sdd/2026-08-09-phase-08d-rag-evaluation-documentation/progress.md`

**Interfaces:**
- Consumes: `RetrievedEvidence.complaint_id`, `RetrievedEvidence.text_redacted`, `CONFIG.llm.max_narrative_chars`, existing cache helpers, `_claim_evidence`, and the existing `Claim`/`GroundedAnswer` types.
- Produces: `validate_answer(payload, allowed_ids, *, evidence_text_by_id)` as the required local semantic boundary; deterministic allegation rendering; default answer prompt/cache identity `rag-v3`.

- [ ] **Step 1: Establish strict RED behavior tests**

  Remove the expected-failure marker from `test_validator_rejects_cited_but_nonextractive_legal_conclusion`, and add behavior tests with literal expectations for:

  - a normalized exact span accepted across NFC and collapsed whitespace;
  - case-only mismatch rejected;
  - disjoint words/non-contiguous paraphrases rejected;
  - multi-ID claim accepted when one cited narrative contains the span and rejected when none do;
  - evidence mappings with missing, extra, boolean, duplicate-equivalent, blank, or non-string IDs/text rejected before accepting claims;
  - unsafe C0/C1/bidi control-bearing, control-only, and `max_narrative_chars + 1` claims rejected without truncation, while safe ZWJ/ZWNJ/combining text remains matchable;
  - ID-only validation no longer satisfying the substantive-answer contract;
  - cached legacy/nonextractive payload deletion and provider-response accounting as `schema` or `citation` according to the existing typed error taxonomy;
  - deterministic renderer output containing `Complaints allege: “<extract>” [Complaint IDs: ...]` and never rendering a manually injected `GroundedAnswer.answer` legal conclusion;
  - the system/prompt asking for verbatim contiguous extracts without telling the model to add allegation prose;
  - `answer_question`, cache reads/writes, and claim-review export passing exact evidence text into validation;
  - claim-review IDs changing when prompt version, extract bytes, source bytes, or citations change.

- [ ] **Step 2: Run the focused tests and confirm the intended failures**

  Run:

  ```bash
  .venv/bin/python -m pytest -q tests/test_llm_answer.py tests/test_llm_eval.py tests/test_config.py
  ```

  Expected: the new extractive cases fail because `validate_answer` still accepts ID-only cited prose, `rag-v2` is still configured, and the renderer lacks deterministic allegation framing. Existing unrelated setup must remain green.

- [ ] **Step 3: Implement minimal extract normalization and validation**

  In `src/llm/answer.py`, add small private helpers that:

  ```python
  def _normalize_extract(value: object, label: str) -> str: ...
  def _evidence_text_map(allowed_ids: set[int], value: object) -> dict[int, str]: ...
  def _claim_is_extract(text: str, complaint_ids: tuple[int, ...], evidence: dict[int, str]) -> bool: ...
  ```

  Use `unicodedata.normalize("NFC", value)` followed by whitespace collapse. Validate exact `int` IDs (never `bool`), exact key equality with `allowed_ids`, nonblank string narratives, and safe characters. Require each sufficient claim to be a contiguous substring of at least one cited normalized narrative. Raise `CitationError` for unsupported extracts/citation-evidence mismatches and `AnswerSchemaError` for malformed claim text/shape. Make `evidence_text_by_id` a required keyword-only argument to `validate_answer`; abstentions still receive and validate the exact evidence map.

- [ ] **Step 4: Propagate evidence text through generation and caches**

  Build `{row.complaint_id: row.text_redacted}` from the already validated `RetrievedEvidence` list. Pass it to provider-response validation, cache-load validation, and cache-write validation. Keep evidence ordering and hashes unchanged. A corrupt legacy row must be deleted by the existing fail-closed cache path rather than replayed.

- [ ] **Step 5: Add deterministic allegation synthesis**

  Preserve the closed wire requirement that `payload["answer"]` exactly equals normalized extract texts joined in claim order. Separately render every validated claim locally as:

  ```text
  Complaints allege: “<extract>” [Complaint IDs: 10, 20]
  ```

  Use this deterministic attribution in both the Answer and Claims CLI sections. Continue applying `_console_text` at the display boundary. The model/system prompt must request exact contiguous excerpts only and must not ask the model to generate the allegation prefix.

- [ ] **Step 6: Preserve evaluation and human-review provenance**

  In `src/llm/eval.py`, resolve the exact retrievable corpus text before cached answer validation and pass the exact evidence map to `validate_answer`. Reuse the same corpus cache used by `_claim_evidence`; do not introduce a second population definition. Keep `_review_identity` binding `claim_text`, citations, cited source hashes, evidence hash, and `input_hash`; prove through tests that the `rag-v3` input identity changes review IDs while replay remains deterministic.

- [ ] **Step 7: Bump identity and reconcile documentation**

  Change `answer_prompt_version` from `rag-v2` to `rag-v3` in `src/config.py`, config tests, and `docs/LLM_LAYER.md`. Mark the decision document approved/implemented, explain the allegation-vs-truth limitation, and update the Phase 8 progress/report without claiming a live provider or human evaluation.

- [ ] **Step 8: Verify focused and full behavior**

  Run, read, and record exact outcomes for:

  ```bash
  .venv/bin/python -m pytest -q tests/test_llm_answer.py tests/test_llm_eval.py tests/test_llm_cli.py tests/test_config.py
  .venv/bin/python -m pytest -q
  .venv/bin/python -m ruff check .
  .venv/bin/python -m ruff format --check src/llm/answer.py src/llm/eval.py src/config.py tests/test_llm_answer.py tests/test_llm_eval.py tests/test_llm_cli.py tests/test_config.py
  git diff --check
  ```

  The former strict xfail must be a normal passing test. The full suite may retain only the documented real-data/human-artifact skips; no finding-7 xfail may remain.

- [ ] **Step 9: Self-review and report without Git mutation**

  Mutation-check case sensitivity, contiguity, missing cited evidence, control handling, prompt version, legacy cache rejection, provider accounting, and claim-review identity. Write the task report in the SDD workspace and append its result to the ledger. Leave the Git index empty and do not attempt staging or commit while the approval-layer block remains active.
