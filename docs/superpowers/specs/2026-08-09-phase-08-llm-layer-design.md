# Phase 8 LLM Layer Design

**Date:** 2026-08-09
**Status:** Approved scope; implementation pending
**Audience:** HarmScope maintainer and technical reviewers

## 1. Goal

Complete HarmScope's descriptive LLM layer without allowing any model output to
affect cluster membership, signal detection, statistics, rankings, or backtest
results.

Phase 8 has two user-facing jobs:

1. Give a statistically selected complaint cluster a cautious, readable harm
   description.
2. Let an analyst ask a question about a signal and receive the exact supporting
   complaints plus a synthesis whose claims cite complaint IDs.

The layer should also demonstrate production-quality AI/ML practices suitable
for the project's portfolio goal: structured output, deterministic evidence
selection, hybrid retrieval, reciprocal-rank fusion, grounded generation,
evaluation, caching, cost tracking, retry behavior, and human review.

## 2. Non-goals

- The LLM does not decide whether a signal fires.
- The LLM does not change cluster membership, novelty, q-values, rankings, or
  baseline results.
- Phase 8 does not add FastAPI routes or the final React interface. It exposes
  stable Python functions and CLI commands that Phase 10 can wrap.
- Phase 8 does not label every cluster from every historical cutoff. It labels
  fired clusters plus a seeded control sample.
- Phase 8 does not use an LLM as the judge of its own headline quality metrics.
- Narrative text and model responses containing consumer text are never
  committed to Git. Evaluation questions must be sanitized, synthetic analyst
  questions that neither quote nor closely paraphrase complaint narratives.

## 3. Existing foundation

The following code remains and is extended rather than replaced:

- `src/llm/select.py`: deterministic selection of 12 medoid-nearest and 8
  MMR-diverse narratives.
- `src/llm/label.py`: prompt, strict JSON schema, guardrails, cache key, and one
  structured Anthropic call.
- `src/llm/run.py`: fired-cluster plus control population, narrative loading,
  cache use, and `cluster_labels` persistence.
- `tests/test_llm.py`: structural separation between the detection path and the
  LLM package.
- `python -m src.pipeline run --phase label`: the labeling entry point.

No implementation may weaken the existing import-boundary test.

## 4. Architecture

```text
statistical signal + cluster
          |
          v
deterministic evidence selection
  12 central + 8 MMR-diverse narratives
          |
          v
structured cluster label --------------------+
  allegation framing, confidence, template   |
          |                                   |
          v                                   v
human verification                     cached label + usage

analyst question + company + cluster
          |
          +----------+-----------+
          |                      |
          v                      v
 dense retrieval             BM25 retrieval
          |                      |
          +----------+-----------+
                     v
          reciprocal-rank fusion
                     |
                     v
       top evidence with complaint IDs
                     |
                     v
          grounded answer generation
                     |
                     v
     citation validation + RAG evaluation
```

Detection packages must not import anything under `src/llm/`. The CLI may
import Phase 8 functions lazily inside Phase 8 command handlers.

## 5. Components

### 5.1 Reliable model client

Add `src/llm/client.py` as the only module that knows Anthropic transport
details.

Responsibilities:

- Execute structured model calls.
- Retry transient connection errors, 429 responses, and 5xx responses with
  bounded exponential backoff and jitter.
- Do not retry authentication, billing, invalid-request, or schema errors.
- Return a typed result containing payload, model, stop reason, input tokens,
  output tokens, cache-read tokens, cache-write tokens, attempts, latency, and
  error category.
- Estimate cost from configuration-owned prices. Estimated cost is clearly
  labeled and never treated as an invoice.
- Accept an injected client and clock/sleeper so all behavior is unit-testable
  without network access.

The default remains `claude-opus-5`, as already configured. Model name and
prompt version stay in every cache key.

### 5.2 Label job

Extend the existing job rather than creating a second runner.

Required behavior:

- Resolve one explicit cluster run and one explicit signals run.
- Select fired clusters plus a seeded non-firing control sample.
- Support `--limit`, with `--limit 20` as the paid pilot command.
- Check disk cache before making a request.
- Write cache entries atomically using a temporary file followed by rename.
- Validate cached payloads against the same schema used for live output.
- Record refusals and terminal failures without aborting the entire batch.
- Persist one usage record for each cache hit or attempted API call.
- Print a final summary of labeled, cached, refused, failed, skipped, tokens,
  latency, and estimated cost.
- Resume safely: a repeated run with the same inputs must not rebill successful
  clusters.

The database transaction is per completed cluster, not one transaction around
the whole population. An interruption must preserve prior completed labels and
usage records.

### 5.3 Human label verification

Add `src/llm/verify.py` and CLI commands following the existing blinded
adjudication pattern.

The exporter chooses at least 50 labels using a fixed seed and stratifies across:

- fired and control clusters;
- product families;
- high, medium, and low model confidence;
- model template suspicion;
- model taxonomy-distinctness decision.

The worklist shows the model label, dominant taxonomy label, and ten redacted
narratives. It does not show signal rank, statistic, q-value, or whether the
cluster was a control.

The reviewer records:

- mechanism accuracy: `agree`, `partial`, or `disagree`;
- taxonomy-distinctness accuracy: `agree` or `disagree`;
- template judgment accuracy: `agree` or `disagree`;
- whether the model should have abstained;
- one failure category and optional notes.

Failure categories are fixed: `incoherent_cluster`, `overgeneralized`,
`overspecific`, `missed_submechanism`, `taxonomy_error`, `template_error`,
`unsupported_claim`, and `other`.

The report command calculates agreement rates with numerators, denominators,
and Wilson intervals. It also reports results by fired/control status and model
confidence. A model-generated review may be stored, but it cannot satisfy the
ROADMAP's human-verification criterion.

### 5.4 Hybrid evidence retrieval

Add `src/llm/retrieve.py` with three separately testable stages.

**Scope.** Retrieval is always restricted to one cluster and, when supplied,
one company. It never searches the full corpus for an answer after a signal has
already established the relevant population.

**Dense retrieval.** Encode the analyst query with the same embedding model used
for the cluster run, normalize it, and search the normalized member vectors with
an exact FAISS `IndexFlatIP` index (equivalent to cosine similarity here).
Return a larger candidate set than the final top-k. Exact search is deliberate:
retrieval is already scoped to one cluster, so an approximate index would add
nondeterminism without a meaningful latency benefit.

**Sparse retrieval.** Build or load a BM25 index over the same scoped redacted
narratives. Tokenization is deterministic, lowercased, and versioned. Cache the
index by cluster membership hash and tokenizer version.

**Fusion.** Combine dense and sparse ranks with reciprocal-rank fusion using
`CONFIG.llm.rrf_k`. Ties break on complaint ID. The result includes complaint
ID, date, company, product family, dense rank/score, sparse rank/score, fused
score, and company public response when available.

Retrieval returns evidence; it never returns a model-written conclusion.

### 5.5 Grounded answer generation

Add `src/llm/answer.py`.

Input:

- analyst question;
- cluster/company scope;
- fused top-k evidence;
- optional relevant company public responses;
- optional enforcement context explicitly labeled as context rather than
  complaint evidence.

Structured output:

```json
{
  "answer": "Short synthesis using allegation language.",
  "claims": [
    {
      "text": "One independently checkable sentence.",
      "complaint_ids": [123, 456]
    }
  ],
  "insufficient_evidence": false,
  "limitations": ["What the retrieved complaints cannot establish"]
}
```

Every substantive claim requires at least one complaint ID. A deterministic
validator rejects any cited ID not present in the retrieved evidence. If there
is no relevant evidence, the output must set `insufficient_evidence` and avoid a
substantive answer. Answers use `complaints allege` language and never render a
legal verdict.

The answer cache key includes prompt version, model, normalized question,
cluster ID, company ID, and ordered evidence complaint IDs.

### 5.6 Evaluation harness

Add `src/llm/eval.py` and a committed evaluation manifest at
`data/ground_truth/rag_eval_questions.csv`.

The committed file contains question IDs, sanitized synthetic question text,
cluster IDs, optional company IDs, question-category labels, and hand-marked
relevant complaint IDs. An automated privacy guard rejects any normalized
eight-token sequence shared with a complaint narrative; a human privacy review
checks the remaining questions for close paraphrases before commit. Keeping the
safe question text with the IDs makes the benchmark reproducible on a fresh
checkout.

The 30 questions cover:

- mechanism identification;
- affected actors and preconditions;
- consumer consequences;
- time or sequence details;
- company-response retrieval;
- deliberately unanswerable questions.

Metrics:

- Recall@10 and MRR for dense, BM25, and fused retrieval;
- fusion win/tie/loss counts against both component retrievers;
- citation validity: cited IDs are a subset of retrieved IDs;
- citation coverage: substantive claims carrying at least one citation;
- abstention accuracy on answerable versus unanswerable questions;
- latency, token use, and estimated cost;
- groundedness from a human review of at least 50 generated claims.

The system must report all three retrieval variants even if fusion loses.

## 6. Data model

Add one migration with four tables.

### `llm_usage`

One row per cache lookup or attempted model call: operation, cluster ID,
question hash when applicable, model, prompt version, input hash, cache status,
attempts, token categories, latency, estimated cost, outcome, error category,
and timestamp.

### `label_verifications`

One row per reviewer and cluster: the four judgments, fixed failure category,
notes, blinded-worklist version, reviewer identifier, and timestamp. It does not
overwrite the model label.

### `rag_answers`

One row per cached answer: question hash, scope, model/prompt version, evidence
IDs as JSON, structured answer as JSON, citation-valid flag, and timestamp.

### `rag_eval_results`

One row per question, retrieval method, and evaluation run: rank of first
relevant item, relevant retrieved count, Recall@10, reciprocal rank, latency,
and answer-level validation fields where applicable.

All new tables are downstream-only. No detection query may read them.

## 7. CLI contract

The completed phase exposes these workflows:

```bash
# Paid pilot, then resumable full labeling
python -m src.pipeline run --phase label --limit 20
python -m src.pipeline run --phase label

# Human label review
python -m src.pipeline label-verify export --n 50 --output data/interim/label_review.csv
python -m src.pipeline label-verify record --input data/interim/label_review.csv
python -m src.pipeline label-verify report

# Retrieval and grounded-answer smoke test
python -m src.pipeline ask --cluster-id ID --company-id ID --question "..."

# Thirty-question evaluation
python -m src.pipeline rag-eval
```

Commands that require a live model validate credentials and configuration
before starting the batch. A billing failure from the provider is terminal: the
command stops after the first failed request and does not attempt the remaining
population. Retrieval-only evaluation remains runnable without an Anthropic key
once the local query encoder is available.

## 8. Failure handling and privacy

- Authentication, billing, and malformed-request errors stop the live job with
  an actionable message before repeated calls.
- Rate limits and transient server/network failures retry within a configured
  bound.
- One cluster refusal or terminal content failure is recorded and skipped.
- Cache corruption is quarantined and recomputed, never silently accepted.
- Database writes use parameterized SQL.
- Only `narratives.text_redacted` may reach retrieval or an external model.
- Raw narrative text is never logged, committed, or embedded in evaluation CSVs.
- Individual names are not displayed or reproduced in synthesized answers.
- Complaint allegations and company responses are visually and structurally
  distinguishable in returned data.

## 9. Testing strategy

### Unit tests

- deterministic MMR selection and tie-breaking;
- structured schema validation;
- atomic cache round-trip and corrupt-cache handling;
- retryable versus terminal API errors;
- token and estimated-cost accounting;
- BM25 ranking;
- dense cosine ranking;
- RRF calculation and deterministic ties;
- scoped retrieval cannot escape cluster/company boundaries;
- answer citations cannot reference unretrieved complaints;
- insufficient-evidence behavior;
- verification sampling and metric calculations.

### Integration tests

- label job with a fake client writes labels and usage records, resumes from
  cache, and does not duplicate charges;
- retrieval joins complaint IDs, dates, and company responses correctly;
- end-to-end fake RAG answer persists and validates;
- schema migration applies to an empty database and upgrades the current one;
- deleting `src/llm/` still leaves detection packages importable and their tests
  passing;
- no module in the detection path imports or queries Phase 8 tables.

### Live gates

1. Run `--limit 20`; inspect labels and usage/cost output.
2. If the pilot is acceptable, label the required verification population.
3. Human-review at least 50 labels and report agreement.
4. Run the 30-question retrieval benchmark.
5. Human-check at least 50 generated claims for groundedness.

## 10. Acceptance criteria

Phase 8 is complete only when:

- the import-boundary determinism test passes;
- interrupted labeling resumes without rebilling completed labels;
- token, latency, cache, refusal, failure, and estimated-cost totals are
  recorded;
- at least 50 labels have human verification and both required agreement rates
  are reported with denominators;
- dense, BM25, and fused Recall@10 and MRR are reported on all 30 questions;
- citation validity and human groundedness are reported;
- every returned evidence item includes a complaint ID and date;
- no raw narrative text is committed;
- `README.md`, `docs/LLM_LAYER.md`, `docs/ENGINEERING_NOTES.md`, and the Phase 8
  dossier reflect measured reality, including failures and incomplete gates.

If live API credit is unavailable, implementation can be complete and fully
tested with fake clients, but Phase 8 remains explicitly incomplete until the
live and human gates above run.

## 11. Implementation order

1. Migration and typed result models.
2. Reliable client, atomic cache, usage accounting, and label-run hardening.
3. Human verification export/record/report workflow.
4. BM25, dense retrieval, and RRF.
5. Grounded structured answers and deterministic citation validation.
6. Evaluation manifest, retrieval metrics, and evaluation CLI.
7. Full tests and documentation reconciliation.
8. Paid 20-label pilot, followed by live evaluation when credits are available.

## 12. Amendment — implemented boundaries and open gates (2026-08-11)

This section records deviations discovered during implementation. It does not
rewrite the approved design as though these decisions were present from the
start.

### 12.1 Retrieval and artifact provenance tightened

The implementation uses exact FAISS search as designed, but does not address a
vector artifact by model tail. It hashes the full model name and validates a
sidecar's model, row count, dimension, completion, and loaded array shape. This
was required because two providers can publish checkpoints with the same tail;
the former filename could silently select the wrong vector space. Legacy
tail-named artifacts fail closed and require regeneration or a separately
validated migration.

Retrieval also recreates the signal population rather than joining direct
cluster members naively: it pins the cluster run's dedup run/cutoff/model,
expands duplicate groups, excludes flagged campaigns, restricts to the cluster
family/company, and selects one deterministic complaint per group/company. The
neutral population SQL is shared with signals without adding an import from the
detection path into `src.llm`.

### 12.2 Answer generation separates support from display context

The approved design said company responses and optional enforcement context
should be included. The implemented answer prompt contains complaint evidence
only; company responses and enforcement matches are returned/rendered in
separate labeled sections. This prevents non-complaint context from becoming
the basis for a claim that cites a complaint ID. Every substantive generated
claim is deterministically rebuilt from validated complaint-cited claims, and
closed limitation reason codes replace unconstrained model prose.

The answer cache identity binds normalized question, ordered evidence, scope,
model, and effective prompt version. Paid usage is persisted through an atomic,
fsynced outbox; drain requires autocommit and deletes a staged event only after
its database insert is durable.

### 12.3 Benchmark freeze and evaluation execution are separate gates

The authoring exporter can prepare a private 30-row candidate worklist, but an
agent/model cannot set `privacy_reviewed=yes` or convert its relevance choices
into human ground truth. The committed manifest therefore remains absent until
a person authors and privacy-reviews it. Evaluation derives the one embedding
model from the referenced cluster-run provenance rather than assuming the
current config default.

One `rag-eval` run writes exact `manifest_sha256`, `embed_model`, and
`retrieval_only` parameters. It makes one retrieval call per question, persists
three method rows per question, and declares 90 outputs only after all requested
work and stable combined rendering succeed. Retrieval-only never invokes answer
generation/provider code. Full mode adds fused-only answer metrics under the
same run ID. Claim export requires that run to be complete and still match the
current manifest bytes.

The human claim artifact is private under configured `data/interim`. Export
creates an unreviewed name; the reviewer saves the completed artifact with the
required `.<reviewer_id>.csv` suffix. Reviewer ID is a trusted local audit label,
not authenticated identity; Phase 10 must bind it to an authenticated actor
before offering remote ingestion.

### 12.4 Current external/design blockers are not implementation results

The private candidate worklist has 18 cluster-wide rows without the concrete
company scope required by the answer cache/schema. Its `company_response`
category exposes only complaint excerpts, not independent public-response
evidence. No silent schema/category change was made; both require a human design
decision or regenerated candidates.

The live Anthropic request on 2026-08-07 authenticated and then failed on zero
organization credit. On 2026-08-11, the network-free retrieval command stopped
before DB creation because the human-frozen manifest was absent. Thus there is
no real evaluation run ID/hash, retrieval/answer metric, token/cost total, or
human denominator to report. Software verified with fake clients is labeled
automated evidence only; Phase 8 remains incomplete until the live and human
gates in §10 run.
