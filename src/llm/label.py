"""Phase 8: turn a cluster's narratives into a named harm mechanism.

docs/LLM_LAYER.md §2. The layer is **descriptive only** — nothing here writes to
`signals`, `cluster_timeseries`, or `baseline_results`, and §1's determinism test
asserts that by deleting this package and re-running.

Three things are decided here rather than in the prompt, because a prompt is a
request and these are guarantees:

**The output shape is enforced by the API, not asked for.** §2.2 specifies strict
JSON with "no prose, no markdown fences". Structured outputs
(`output_config.format`) constrain the response to the schema, so a malformed
label is impossible rather than unlikely. The older way to get this — prefilling
an assistant turn with `{"` — is a 400 on every current model, and the prompt
begging that went with it ("output ONLY valid JSON") is dead weight once the
schema is enforced.

**The guardrails carry their reason.** §2.3 requires the model to describe what
complaints *allege*, never to assert conduct occurred, and to abstain via
`confidence: low` rather than invent a unifying story for 20 unrelated texts.
Confabulation is the documented default failure mode for this task, so abstention
is rewarded explicitly.

**The stable half of the prompt is cached.** The system prompt is byte-identical
across every cluster; only the narratives change. A `cache_control` breakpoint on
it means the guardrails are written once and read at ~0.1x for every cluster
after, which is what makes §2.4's "label lazily, batch where you can" affordable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# §2.2. `additionalProperties: false` and a full `required` list are what make
# the schema enforceable rather than advisory.
LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "harm_mechanism": {
            "type": "string",
            "description": "One sentence, active voice, describing what goes "
                           "wrong mechanically.",
        },
        "actors": {"type": "array", "items": {"type": "string"}},
        "preconditions": {
            "type": "string",
            "description": "What must be true for a consumer to be exposed.",
        },
        "consumer_impact": {
            "type": "string",
            "description": "Concrete consequence described in the narratives.",
        },
        "distinct_from_taxonomy": {"type": "boolean"},
        "distinctness_rationale": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "is_likely_template": {"type": "boolean"},
    },
    "required": [
        "harm_mechanism", "actors", "preconditions", "consumer_impact",
        "distinct_from_taxonomy", "distinctness_rationale", "confidence",
        "is_likely_template",
    ],
    "additionalProperties": False,
}

SYSTEM = """\
You are labelling clusters of US consumer-finance complaint narratives, to name \
the harm mechanism each cluster describes.

Complaints are allegations. Describe what consumers *allege* happened; never \
assert that any conduct occurred, and never characterise it as a legal \
violation. Do not name individuals — if a narrative contains a personal name, \
ignore it.

The narratives in one cluster may not share a mechanism. If they do not, say so \
with confidence "low" and describe only what they actually have in common. A \
label that admits the cluster is incoherent is more useful than a plausible \
story that unifies unrelated complaints, and abstaining costs you nothing here.

You are also shown the cluster's dominant existing CFPB taxonomy labels. Judge \
distinctness against those specific labels, not against a general impression of \
what the taxonomy covers.

Set is_likely_template when the narratives look like one document filed many \
times — near-identical phrasing, boilerplate structure, identical statutory \
citations. This is a cross-check on a separate statistical detector, so judge it \
from the text alone.\
"""


def build_prompt(narratives: list[str], taxonomy_labels: list[str],
                 max_chars: int) -> str:
    """The per-cluster user turn. Stable content lives in SYSTEM, not here."""
    labels = ", ".join(taxonomy_labels) if taxonomy_labels else "(none recorded)"
    body = "\n\n".join(
        f"--- narrative {i + 1} ---\n{text[:max_chars]}"
        for i, text in enumerate(narratives)
    )
    return (
        f"Dominant existing taxonomy labels for this cluster: {labels}\n\n"
        f"{len(narratives)} narratives from the cluster:\n\n{body}"
    )


def input_hash(prompt_version: str, model: str, complaint_ids: list[int]) -> str:
    """§2.4's cache key: sha256(prompt_version + model + sorted ids).

    Sorted, so the key does not depend on selection order — the same 20
    narratives always hash the same. `prompt_version` is in the key because a
    prompt edit invalidates every cached label, which is the whole reason §4
    requires bumping it.
    """
    payload = json.dumps(
        {"prompt_version": prompt_version, "model": model,
         "complaint_ids": sorted(int(c) for c in complaint_ids)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def cached(cache_dir: Path, key: str) -> dict | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    # Written by this module, so it is a trusted local artifact.
    return json.loads(path.read_text())


def write_cache(cache_dir: Path, key: str, label: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps(label, sort_keys=True))


def label_cluster(client, model: str, narratives: list[str],
                  taxonomy_labels: list[str], max_chars: int) -> dict:
    """One API call. The only non-deterministic step in this package.

    `cache_control` on the system block caches the guardrails across every
    cluster in the run; the narratives sit after it and vary per call, which is
    the ordering prompt caching requires — stable content first, volatile last.
    """
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=[{
            "type": "text",
            "text": SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        output_config={"format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
        messages=[{
            "role": "user",
            "content": build_prompt(narratives, taxonomy_labels, max_chars),
        }],
    )
    if response.stop_reason == "refusal":
        # Not an exception: a declined label is a fact about the cluster, and
        # the run should record it and continue rather than abort a batch.
        return {"refused": True, "stop_reason": "refusal"}
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)
