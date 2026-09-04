# LLM_LAYER

## 1. Contract

The LLM layer is **descriptive and retrieval-only**. It has exactly two jobs:

1. **Cluster labeling** — turn a set of narratives into a named harm mechanism.
2. **Evidence retrieval** — given a signal, retrieve and synthesize the supporting complaints.

It has zero authority over:
- whether a signal fires
- any statistic, p-value, q-value, or ranking
- cluster membership
- backtest outcomes

**Test:** deleting `src/llm/` entirely and re-running the pipeline must produce a
byte-identical **`signals`** table. Add this as an integration test. If it fails, the LLM
has leaked into the detection path.

The test is scoped to `signals` on purpose. `baseline_results` is derived from
`backtest_links`, which is written by a *human adjudicator* who is shown LLM
cluster labels alongside exemplar narratives (`EVALUATION.md` §1.3). LLM output
therefore reaches `baseline_results` through a person, by design. Claiming
byte-identical `baseline_results` would be a claim the pipeline cannot honour,
and an overclaim here would undermine the one architectural guarantee this
project actually has.

The honest pair of statements:

- `signals` is byte-identical with `src/llm/` deleted. **Detection is fully
  deterministic and LLM-free.**
- `baseline_results` is byte-identical *given the same `backtest_links`*.
  Adjudication is a deliberate human step, and its inputs, blinding protocol,
  and intra-rater agreement are reported (`EVALUATION.md` §1.3).

Rationale: LLM outputs are non-deterministic, version-drifting, and unauditable. A regulator-
facing signal detection system cannot have those properties in its decision path. This
architectural separation is the most defensible thing in the project — lead with it.

---

## 2. Cluster labeling

### 2.1 Input selection

Per cluster, select **k = 20** narratives:
- 12 nearest the medoid (representative)
- 8 sampled for diversity (maximal marginal relevance against the first 12)

Diversity sampling matters: 20 near-medoid narratives from a template-adjacent cluster all read
identically and the model will describe the template, not the mechanism.

Truncate each to ~1,200 chars. Always use `text_redacted`, never raw.

### 2.2 Output schema

Strict JSON, no prose, no markdown fences:

```json
{
  "harm_mechanism": "one sentence, active voice, describes what goes wrong mechanically",
  "actors": ["who is involved"],
  "preconditions": "what must be true for a consumer to be exposed",
  "consumer_impact": "concrete consequence described in the narratives",
  "distinct_from_taxonomy": true,
  "distinctness_rationale": "why the existing labels shown do not capture this, or why they do",
  "confidence": "high | medium | low",
  "is_likely_template": false
}
```

Include the cluster's dominant existing `(Issue, Sub-issue)` labels in the prompt so the model
can assess distinctness against something concrete rather than guessing.

`is_likely_template` is a **cross-check on the statistical campaign detector**, not a
replacement. Disagreements between the two go into a review queue — those disagreements are
genuinely useful and worth reporting as a finding.

### 2.3 Guardrails in the prompt

- Describe what complaints **allege**. Never assert that conduct occurred.
- Never name individuals. If a narrative contains a personal name, ignore it.
- Never speculate about legal violations or cite statutes as conclusions.
- If narratives are incoherent or the cluster has no common mechanism, say so via
  `confidence: low` — do not invent a unifying story. Reward abstention explicitly in the
  prompt; the default failure mode of an LLM given 20 unrelated texts is confident confabulation.

### 2.4 Caching and cost control

- Cache key: `sha256(prompt_version + LLM model + evidence embedding model +
  sorted(selected complaint_ids))`. Store in `cluster_labels.input_hash`, cache
  blobs on disk under `data/artifacts/llm_cache/`. The evidence model is read
  from the successful cluster run; an explicit mismatch fails before vectors,
  cache, provider, or label persistence are touched.
- Only label clusters with `n_members >= min_cluster_size_for_label` (config).
- Label lazily: signals first, label only clusters that fire, plus a random control sample for
  evaluation. Labeling every cluster in every backtest refit is a large avoidable bill.
- Batch where the API supports it.

### 2.5 Human verification (required)

Sample **≥ 50** labeled clusters. For each, read 10 narratives yourself and mark
`human_agrees`. Report:
- agreement rate on `harm_mechanism` being accurate
- agreement rate on `distinct_from_taxonomy`
- specific failure modes observed

An unverified LLM label layer is decoration. A verified one with a reported 78% agreement rate
is a result.

The gate is evaluated only for one explicit worklist version and counts distinct
human-reviewed clusters, not review rows. Export writes a non-reviewer-facing JSON
sidecar beside the CSV; it pins the signals run, cluster run, model, prompt version,
seed, canonical cluster IDs, the export-time fired-cluster snapshot, source label
input hashes, and a digest of every
reviewer-visible label and narrative field, as well as the exported filename.
Only the decision/note columns may
change. Recording parses the exact CSV/sidecar bytes before opening a writable
database, then rechecks successful run/cluster and current label provenance inside
the same owned transaction as the immutable inserts. Reports use the stored
`label_verifications.is_fired` snapshot rather than current alert configuration.
An exact replay is idempotent while rejecting a changed decision for the same key.

Phase 8's CLI is a trusted local, human-only ingestion boundary. Its `--reviewer`
value is an operator-supplied audit label, not authenticated identity, and model-origin
reviews are intentionally unavailable through that command. Before Phase 10 exposes
review ingestion through an API, bind both reviewer identity and `reviewer_origin` to
the authenticated actor on the server; never accept either as client-asserted authority.

---

## 3. Evidence retrieval (RAG)

### 3.1 Purpose

An analyst clicks a signal and asks: *"show me why."* The system must return the actual
complaints, not a summary that could be hallucinated.

### 3.2 Retrieval

Hybrid, scoped to the signal's cluster and company:

- **Dense:** FAISS over the cluster's member embeddings, query = analyst question.
- **Sparse:** BM25 over the same member narratives.
- **Fusion:** Reciprocal Rank Fusion. Return top 10 with complaint IDs and dates.

Scoping to the cluster is what makes this different from a generic document QA bot — retrieval
happens inside a statistically-identified population, not over the whole corpus.

### 3.3 Corpora

| Corpus | Use |
|---|---|
| Cluster member narratives | primary evidence |
| Enforcement action texts | context: has similar conduct been actioned before? |
| Company public responses | the other side of the story — include it |

Including company public responses is not decoration. A system that shows only consumer
allegations is an advocacy tool, not an analysis tool.

The implementation keeps that separation structural. Complaint narratives are
the only evidence supplied to answer generation and the only source that may
support a citation. Company public responses are returned and rendered in a
separate section; optional enforcement matches are labeled as context. Neither
is placed in the complaint-evidence prompt, so it cannot silently become the
basis for a complaint-cited claim.

### 3.4 Answer contract

- Every claim in the synthesized answer carries inline complaint IDs.
- If retrieval returns nothing relevant, say so. Do not answer from parametric knowledge.
- Answers are rendered next to the raw retrieved narratives, never instead of them.
- Standing footer: *"Complaints are consumer allegations. Publication does not indicate the
  CFPB verified the allegations or that the company acted unlawfully."*

### 3.5 Evaluation of the RAG layer

Modest but real:
- 30 hand-written analyst questions with hand-marked relevant complaint IDs.
- Report Recall@10 and MRR for dense / sparse / fused.
- Report groundedness: fraction of generated sentences with a valid supporting complaint ID
  (hand-check 50 sentences).

This is small. Keep it small. The RAG layer is a feature, not the thesis — do not let it eat
the schedule.

The implemented benchmark is stricter than the original three-line protocol:
it always reports dense, BM25, and fused Recall@10/MRR, fusion win/tie/loss
against both components, per-question losses, latency, citation validity,
citation coverage, answerable/unanswerable abstention, token/cache/outcome/cost
totals, and a separate human-groundedness gate. Unanswerable questions are
reported but excluded from retrieval macro means. A generated or LLM-judged
review cannot satisfy the human denominator.

---

## 4. Config

Config lives in `src/config.py` as `LLMConfig`, not as loose module constants —
`ARCHITECTURE.md` §6 requires every parameter that appears in a result to be
reachable from one frozen dataclass and serialized into `runs.params_json`.

```python
model                      = "claude-opus-5"
prompt_version             = "v1"
answer_prompt_version      = "rag-v3"
label_sample_k             = 20
label_medoid_k             = 12   # remainder sampled for diversity (MMR)
min_cluster_size_for_label = 30
max_narrative_chars        = 1200
rag_top_k                  = 10
rag_candidate_k            = 50
bm25_tokenizer_version     = "word-v1"
rrf_k                      = 60
human_verify_n             = 50
```

`prompt_version` is bumped on any prompt edit. Cached labels from an old version are never
silently reused — the cache key includes it.

**On the model.** This spec originally named `claude-sonnet-4-6`, then
`claude-sonnet-5` — the tier chosen deliberately because labelling is a bulk,
well-scoped, cost-sensitive job and §2.4 is explicit about cost control. It now
reads `claude-opus-5`.

That is the escalation this section already pre-registered ("if §2.5 human
verification comes back with a poor agreement rate, the first lever is a stronger
model"), pulled forward on 2026-08-07 at the operator's explicit instruction that
cost is not a constraint. It is recorded as an instructed change rather than an
evidence-driven one, because no verification has run yet — there is no agreement
rate to justify it, and pretending otherwise would misrepresent why the tier
moved.

Changing it now is free: **nothing has been labelled**, and the model string is
part of the cache key, so the same switch after a labelling run would have
invalidated every cached label. That property is why the model belongs in the
key, and the reason to make this change before the first run rather than after
it.

The remaining lever is the prompt. If §2.5 verification is still weak on
`claude-opus-5`, change the prompt and bump `prompt_version` — one at a time, and
record which, so an agreement-rate movement can be attributed.

---

## 5. Implemented interfaces and provenance

### 5.1 Provider, retry, and accounting boundary

`src.llm.client.AnthropicModelClient` is the only Anthropic transport boundary.
`preflight(model)` retrieves the configured model. `call_json(...)` applies the
closed JSON schema, bounded exponential backoff, prompt caching, typed usage,
latency, and configuration-owned price estimates. Rate limits, connections, and
server failures are retryable; authentication, permission, billing,
invalid-request, refusal/schema, and other terminal failures are not. Estimated
cost is provenance, not an invoice.

`llm_usage` records operation, run ID, question/input identities, cache status,
attempts, token categories, latency, estimated cost, outcome, and a closed error
category. Both label and answer accounting use fsynced atomic usage outboxes: each drain
requires autocommit, verifies an insert-or-ignore result against every stored
identity/accounting field, and deletes an event only after an exact insert or
idempotent replay is durable. A reused usage ID with different accounting raises
a typed outbox error and leaves the staged event intact. That
preserves paid-call accounting across process/database failures without logging
prompt or narrative prose. Label events are staged after a provider response and
before cache publication or label/usage persistence; stable usage IDs make a
crash-after-commit replay idempotent. Label events are fixed-prefix direct files
under a pinned, no-follow cache root (not a followable outbox subdirectory).
If that root is renamed during the descriptor-relative replace, staging fails
and publication-aware rollback removes the new event from the moved root or
restores the exact prior event before database fallback runs. Backup creation,
publication, and backup deletion are each followed by a pinned-directory
`fsync`. A failure in final backup-deletion durability is reported while the
already-verified new event remains the explicit filesystem state; cleanup
failures add exception types only and never replace the original staging error.
When a prior event exists, it is opened descriptor-relative with no-follow and
copied in bounded chunks into a distinct no-follow 0600 inode, which is file-fsynced
and unlinked while its descriptor remains pinned. Source metadata and pathname
identity are checked before and after the copy and again before/after the backup
hard link; source bytes must still equal the independent snapshot before
publication. The backup's device/inode must match the pinned source, but recovery
uses only the independent snapshot, never the mutable source inode or backup name.
A missing, mutated, or different destination/link identity fails before
publication and restores the snapshot where safe; a concurrently appeared event
after pinned absence is not deleted. A root rename during final cleanup restores
the snapshot on the pinned directory, or absence when there was no prior event.
This closes the checked publication lifecycle but cannot prevent a rename after
the final check has returned. The same-UID path-race model does not claim defense
against an actor that can inspect or modify this process's memory or anonymous
file descriptors.
If closing a pinned source/snapshot descriptor fails while a containment error is
already propagating, the containment error remains primary and receives only a
fixed, exception-type cleanup note; root descriptor cleanup still runs. A close
failure after otherwise successful verified publication is surfaced with the
new event left as the explicit state.
Malformed-result staging failures synchronously fall back to usage-only database
persistence, and cache-write failures durably correct the same usage ID to
`failed/cache_write` without double accounting.

### 5.2 Cache and resume identities

- Label cache: prompt version, LLM model, evidence embedding model, and sorted
  selected complaint IDs.
- Sparse cache: exact scoped-membership hash plus tokenizer version.
- Embedding artifact: SHA-256 of the full model name, with sidecar checks for
  model, row count, dimension, completion, and array shape.
- Answer cache: normalized question hash, ordered evidence identity, cluster,
  company, model, and effective answer-prompt version.
- Evaluation run: exact manifest bytes (`manifest_sha256`), the one embedding
  model recorded by all referenced cluster runs, and `retrieval_only`.
- Claim review: completed evaluation run, frozen manifest hash, question,
  cached answer/input/evidence identity, claim position/text, cited IDs,
  cited redacted-evidence digests, canonical attribution-renderer version, and
  the SHA-256 of its exact rendered output.

These identities make cache hits explainable and replays idempotent. Legacy
tail-named embedding files are intentionally not accepted by the current
loader: falling back to the ambiguous name would undo the model-identity fix.

#### Embedding artifact maintenance — completed and verified 2026-09-03

The only migration interface is deliberately explicit:

```bash
python -m src.pipeline migrate-embeddings \
  --model sentence-transformers/all-MiniLM-L6-v2
python -m src.pipeline migrate-embeddings \
  --model sentence-transformers/all-MiniLM-L6-v2 --execute
```

Without `--execute`, the command only validates and reports the planned legacy
and SHA-addressed basenames. It requires the exact model instead of borrowing a
configured default, checks that the database already exists, opens it read-only,
and never bootstraps the schema or records a run. Both modes validate the
sidecar, NumPy vectors, FAISS index, and database mapping; execute mode publishes
Darwin copy-on-write clones from the already-validated source descriptors,
independently validates targets, then atomically moves legacy basenames to
deterministic dot-prefixed retirement names. The automated command never unlinks
those retained entries. Per-role `legacy`, `legacy + target`, and
`target + retirement` crash phases replay by exact digest and semantic
equivalence; mismatches fail with every observed entry still named. Linux and
other platforms fail closed rather than path-copying. Target-only state and
same-inode dual aliases from the superseded hard-link protocol are rejected;
the retained comparison names must be different inodes. The current loader
remains SHA-only throughout—there is no legacy fallback—and later retirement
cleanup requires a separate quiescent operator.

The independently approved real migration completed on 2026-09-03 for
2,477,937 MiniLM rows × 384 dimensions. Execute returned `migrated`; immediate
replay returned `already-migrated`. Target/retirement sizes and SHA-256 digests
matched per role, target inodes differed from the retained original inodes, all
link counts were one, and legacy basenames were absent. The sidecar SHA-256 is
`5e6e54abb81cf8e173b8a9e47b8a2a34eaf63a6a4d66ad9066c2b6ede687484c`.
DuckDB file identity and mapping aggregates were unchanged, the strict loader
returned `(2477937, 384) float32`, and forced-offline hybrid retrieval returned
five evidence IDs for an existing cluster/company scope. No provider call ran.

### 5.3 Retrieval and answer APIs

`load_corpus(con, cluster_id, company_id, embed_model)` recreates the exact
signal-consistent population: the cluster run's recorded dedup run/cutoff,
campaign exclusion, family restriction, company scope, one item per dedup
group/company, and matching embedding-map provenance.

`retrieve_variants(...)` encodes a question once, performs exact normalized
FAISS inner-product ranking and deterministic BM25 over that same corpus, and
returns independent component rankings plus configured reciprocal-rank fusion.
`answer_question(...)` consumes fused evidence, validates every structural and
scope invariant, and requires every substantive claim to be an NFC-and-whitespace-
normalized contiguous excerpt from at least one of its cited redacted complaint
narratives. Matching is case-sensitive; multi-ID claims retain every declared
unique in-scope ID while requiring a match in at least one cited source. Empty,
unsafe-control-bearing, and over-length model text is rejected rather than
truncated. The wire answer remains the normalized extracts joined in claim
order. The versioned pure `render_attributed_claim` boundary—not the model—adds
`Complaints allege: “<extract>” [Complaint IDs: ...]` in both answer and claim
sections and in the human claim-review export. Review identity binds the renderer
version and exact rendered-output hash, so prefix, escaping, or ID-layout changes
cannot reuse prior judgments. This proves source fidelity only; it does not
establish that an underlying allegation is true or legally valid. Cache/provider
validation and claim-review export receive the exact retrieved redacted evidence
text, while usage rows, review IDs, and reports remain free of private prose.

### 5.4 Evaluation and human-review APIs

`src.llm.eval` exposes:

- `load_manifest` / `validate_manifest` for the exact seven-column, balanced
  30-question contract, cluster/company membership, and normalized eight-token
  privacy guard;
- `export_authoring_worklist` / `import_authoring_worklist` for private
  evidence-assisted human authoring and ID-only freeze;
- `run_retrieval_eval` for one-call-per-question dense/BM25/fused scoring and
  atomic three-row persistence;
- `run_answer_eval` for fused-only citation/coverage/abstention fields and
  exact run-linked usage aggregation;
- `export_claim_review` / `parse_claim_review` / `record_claim_review` for a
  deterministic blinded sample of at least 50 claims and denominator-bearing
  Wilson output.

The CLI is lazy-imported and available as `rag-eval`, `rag-eval
--retrieval-only`, `author`, `import`, `claims-export`, and `claims-record`.
All private authoring, claim-review, and label-review reads/writes stay under configured
`data/interim` as direct regular-file children through no-follow, configured-root
descriptor-pinned opens and atomic fsynced replacement. Nested private paths and
configured-root rename/symlink swaps fail closed. A rename precisely during
replacement can make the operation fail, but publication-aware rollback removes
the newly published private bytes from the moved root and restores an exact prior
destination when present; it does not leave a persistent new artifact. Backup
creation, publication, deletion, and recovery namespace changes are directory-
fsynced. If final backup-deletion fsync fails after successful root validation,
the operation reports failure and retains the verified new destination; cleanup
errors cannot mask the original write error and disclose exception types only.
The existing destination is opened descriptor-relative with no-follow and copied
in bounded chunks into a distinct no-follow 0600, file-fsynced inode. Source
metadata and name identity are verified around copying and backup linking, and
source bytes are compared with the independent snapshot before publication. The
linked backup must match the pinned source's device/inode, but recovery never
rereads that mutable inode or reopens the backup name. A missing, mutated, or
different destination-link identity fails before publication and restores the
independent snapshot where safe; a file appearing after pinned absence is
preserved. A rename in the final checked window restores exact prior bytes (or
absence) on the pinned directory, fsyncs recovery, and leaves the recreated
configured root empty. No same-identity guarantee is claimed after the final
check returns, and the same-UID model excludes actors able to alter process memory
or anonymous descriptors.
If closing a pinned source/snapshot descriptor fails during error propagation,
the original containment/write error remains primary with a type-only cleanup
note and root descriptor cleanup continues. A close failure after otherwise
successful verified publication is surfaced while retaining the verified new
artifact. Manifest parsing and hashing use
one unchanged byte buffer. Pure artifact validation precedes writable
connection/run creation. Successful
evaluation declares 90 output rows only after all requested work and stable
report rendering succeed; partial/interrupted runs stay failed. Claim export
accepts only a completed run whose manifest hash equals the current frozen
file, and completed review filenames must end in `.<reviewer_id>.csv` under
configured `data/interim`.

---

## 6. Current gate status — 2026-09-03

The software paths above are implemented and exercised with synthetic/fake
provider fixtures. That is not a measured LLM result. The following remain
open:

1. A human must author/privacy-review the private 30-row draft and freeze the
   ID-only manifest. No manifest hash or benchmark run ID exists yet.
2. Eighteen draft rows have no concrete company scope, while grounded answer
   evaluation deliberately requires one. The draft must be regenerated or the
   design explicitly changed; the evaluator fails before provider work.
3. `company_response` cannot be authored honestly from the current worklist's
   complaint excerpts alone; independent public-response evidence or a human
   scope decision is required.
4. The provider authenticated on 2026-08-07 but returned a terminal billing
   error because the organization had no credit. No live label or answer
   benchmark has run.
5. At least 50 labels and at least 50 answer claims still require human review.

Accordingly, Recall@10, MRR, fusion comparisons, citation/abstention results,
tokens, cost, and human groundedness are **pending**, not zero. On 2026-08-11,
the network-free command stopped before opening the database because the frozen
manifest does not yet exist; it created no `rag-eval` run.
