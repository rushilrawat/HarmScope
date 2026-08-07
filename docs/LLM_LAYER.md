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

- Cache key: `sha256(prompt_version + model + sorted(selected complaint_ids))`. Store in
  `cluster_labels.input_hash`, cache blobs on disk under `data/artifacts/llm_cache/`.
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

---

## 4. Config

Config lives in `src/config.py` as `LLMConfig`, not as loose module constants —
`ARCHITECTURE.md` §6 requires every parameter that appears in a result to be
reachable from one frozen dataclass and serialized into `runs.params_json`.

```python
model                      = "claude-sonnet-5"
prompt_version             = "v1"
label_sample_k             = 20
label_medoid_k             = 12   # remainder sampled for diversity (MMR)
min_cluster_size_for_label = 30
max_narrative_chars        = 1200
rag_top_k                  = 10
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
