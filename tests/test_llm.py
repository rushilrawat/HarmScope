"""Phase 8's contract: the LLM layer is descriptive and cannot reach detection.

`docs/LLM_LAYER.md` §1 calls the separation "the most defensible thing in the
project", and its test is that deleting `src/llm/` leaves `signals`
byte-identical. That is asserted here structurally rather than by running the
pipeline twice: a real two-run comparison needs hours of refit, so it would be
skipped in CI and therefore never run — and a leakage check that never runs is
the defect this repo has already shipped once (see `test_leakage.py` item 6).

The structural version catches the thing that would actually break the contract:
an import edge from the detection path into `src/llm/`. If no detection module
can reach the package, deleting it cannot change their output.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.llm import label as label_mod
from src.llm import select as select_mod

ROOT = Path(__file__).resolve().parents[1]

# The detection path, as `LLM_LAYER.md` §1 defines it: everything that decides
# whether a signal fires or what any statistic is.
DETECTION = ["signals", "cluster", "dedup", "embed", "evaluation"]


def complete_label() -> dict:
    return {
        "harm_mechanism": "A servicer applies fees after a payment.",
        "actors": ["servicer"],
        "preconditions": "The consumer makes a payment.",
        "consumer_impact": "The consumer pays an unexpected fee.",
        "distinct_from_taxonomy": True,
        "distinctness_rationale": "The existing label does not describe fees.",
        "confidence": "high",
        "is_likely_template": False,
    }


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_detection_path_never_imports_the_llm_layer():
    """§1 — deleting src/llm/ must leave `signals` byte-identical."""
    offenders = []
    for package in DETECTION:
        for source in (ROOT / "src" / package).rglob("*.py"):
            for name in _imports(source):
                if name.startswith("src.llm") or name == "anthropic":
                    offenders.append(f"{source.relative_to(ROOT)} imports {name}")
    assert not offenders, (
        "the detection path can reach the LLM layer, so deleting src/llm/ "
        "could change `signals`:\n  " + "\n  ".join(offenders)
    )


def test_pipeline_imports_the_llm_layer_lazily():
    """The CLI may drive labelling, but importing it must not require anthropic.

    `src/pipeline.py` is imported by the detection phases, so a module-level
    `from src.llm import ...` would make the whole pipeline fail to import when
    `src/llm/` is deleted — turning the determinism test from a property into a
    crash.
    """
    tree = ast.parse((ROOT / "src" / "pipeline.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        names = (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
        )
        if any(n.startswith("src.llm") for n in names):
            assert node.col_offset > 0, (
                "src/pipeline.py imports src.llm at module level; it must be "
                "imported inside the function that uses it"
            )


# --- selection (deterministic, no API) --------------------------------------
def test_mmr_prefers_a_far_point_over_a_near_duplicate():
    """The reason §2.1 asks for diversity at all."""
    selected = np.array([[1.0, 0.0]])
    candidates = np.array([
        [0.9999, 0.0141],  # near-duplicate of what is already selected
        [0.0, 1.0],        # orthogonal — the informative one
    ])
    assert mmr_first(candidates, selected) == 1


def mmr_first(candidates, selected):
    return select_mod.mmr(candidates, selected, k=1)[0]


def test_selection_is_stable_and_bounded():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    rows = np.arange(50)
    ids = np.arange(1000, 1050)

    first = select_mod.select_for_label(X, rows, ids, 3, k=20, medoid_k=12)
    again = select_mod.select_for_label(X, rows, ids, 3, k=20, medoid_k=12)

    assert first == again, "selection must be reproducible"
    assert len(first) == 20 and len(set(first)) == 20


def test_selection_handles_a_cluster_smaller_than_k():
    X = np.eye(3, dtype=np.float32)
    got = select_mod.select_for_label(X, np.arange(3), np.array([7, 8, 9]),
                                      None, k=20, medoid_k=12)
    assert sorted(got) == [7, 8, 9]


# --- cache key --------------------------------------------------------------
def test_cache_key_ignores_selection_order_but_not_prompt_version():
    a = label_mod.input_hash("v1", "m", [3, 1, 2])
    b = label_mod.input_hash("v1", "m", [1, 2, 3])
    assert a == b, "the same 20 narratives must hash the same in any order"

    assert label_mod.input_hash("v2", "m", [1, 2, 3]) != a, (
        "a prompt edit must invalidate cached labels — that is why §4 requires "
        "bumping prompt_version"
    )
    assert label_mod.input_hash("v1", "other", [1, 2, 3]) != a


def test_cache_write_is_atomic_and_validated(tmp_path):
    key = label_mod.input_hash("v1", "m", [1])
    assert label_mod.cached(tmp_path, key) is None
    label_mod.write_cache(tmp_path, key, complete_label())
    assert label_mod.cached(tmp_path, key) == complete_label()
    assert list(tmp_path.glob("*.tmp")) == []


def test_corrupt_cache_is_quarantined(tmp_path):
    key = "broken"
    path = tmp_path / "broken.json"
    path.write_text("{")

    assert label_mod.cached(tmp_path, key) is None
    assert not path.exists()
    assert len(list(tmp_path.glob("broken.json.corrupt-*"))) == 1


def test_label_validation_rejects_missing_or_extra_fields(tmp_path):
    missing = complete_label()
    missing.pop("confidence")
    with pytest.raises(label_mod.LabelSchemaError, match="confidence"):
        label_mod.validate_label(missing)

    extra = complete_label() | {"extra": True}
    with pytest.raises(label_mod.LabelSchemaError, match="extra"):
        label_mod.validate_label(extra)

    with pytest.raises(label_mod.LabelSchemaError, match="object"):
        label_mod.write_cache(tmp_path, "scalar", None)


# --- output contract --------------------------------------------------------
def test_schema_is_enforceable():
    """§2.2 wants strict JSON. Structured outputs only guarantee that when the
    schema closes the object and requires every field."""
    assert label_mod.LABEL_SCHEMA["additionalProperties"] is False
    assert set(label_mod.LABEL_SCHEMA["required"]) == set(
        label_mod.LABEL_SCHEMA["properties"]
    )


def test_guardrails_are_in_the_system_prompt():
    """§2.3's four requirements, asserted on the text that carries them."""
    system = label_mod.SYSTEM.lower()
    assert "alleg" in system, "must frame complaints as allegations"
    assert "never assert" in system
    assert "do not name individuals" in system
    assert "low" in system and "abstain" in system, "must reward abstention"


def test_prompt_includes_the_taxonomy_labels_to_judge_against():
    """§2.2 — distinctness is judged against concrete labels, not a guess."""
    prompt = label_mod.build_prompt(["narrative text"], ["Incorrect information"], 100)
    assert "Incorrect information" in prompt
    assert label_mod.build_prompt(["x"], [], 100).count("(none recorded)") == 1


def test_narratives_are_truncated_to_the_configured_length():
    prompt = label_mod.build_prompt(["a" * 5000], [], 1200)
    assert "a" * 1200 in prompt and "a" * 1201 not in prompt


def test_refusal_is_recorded_rather_than_raised():
    """A declined label is a fact about the cluster; a batch must not abort."""

    class _Refusing:
        class messages:
            @staticmethod
            def create(**_):
                return type("R", (), {"stop_reason": "refusal", "content": []})()

    got = label_mod.label_cluster(_Refusing(), "m", ["n"], [], 1200)
    assert got == {"refused": True, "stop_reason": "refusal"}


def test_label_call_uses_structured_outputs_and_caches_the_system_prompt():
    """The two guarantees §2.2 and §2.4 turn on, asserted on the actual request."""
    seen = {}

    class _Recording:
        class messages:
            @staticmethod
            def create(**kwargs):
                seen.update(kwargs)
                payload = json.dumps({"harm_mechanism": "x"})
                block = type("B", (), {"type": "text", "text": payload})()
                return type("R", (), {"stop_reason": "end_turn",
                                      "content": [block]})()

    label_mod.label_cluster(_Recording(), "claude-sonnet-5", ["n"], ["L"], 1200)

    fmt = seen["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"] is label_mod.LABEL_SCHEMA
    assert seen["system"][0]["cache_control"] == {"type": "ephemeral"}
    # Sampling parameters are rejected on current models, and a prefilled
    # assistant turn is a 400 — neither may creep back in.
    assert not {"temperature", "top_p", "top_k"} & set(seen)
    assert seen["messages"][-1]["role"] == "user"


@pytest.mark.parametrize("field", ["confidence", "is_likely_template"])
def test_abstention_and_template_crosscheck_are_required_fields(field):
    assert field in label_mod.LABEL_SCHEMA["required"]
