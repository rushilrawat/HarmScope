"""Ground-truth file contracts.

docs/ARCHITECTURE.md §2 marks `dedup_eval_pairs.csv` as COMMITTED, while
docs/DATA.md §6 item 4 says narrative text is never committed to git. Those
only coexist if the file holds complaint *ids* and a label, with the text
joined from DuckDB at evaluation time.

This test is the enforcement. 300 hand-labelled narrative pairs landing in git
history cannot be walked back — the fix is to never let the first one land.

Company names and the curator's own `harm_summary` prose are explicitly fine
(docs/DATA.md §6), so `enforcement_actions.csv` is checked for shape only.
"""

from __future__ import annotations

import csv

import pytest

from src.config import PATHS

DEDUP_PAIRS = PATHS.ground_truth / "dedup_eval_pairs.csv"
ENFORCEMENT = PATHS.ground_truth / "enforcement_actions.csv"
RAG_MANIFEST = PATHS.ground_truth / "rag_eval_questions.csv"

# Any column that could carry consumer-written text.
FORBIDDEN_SUBSTRINGS = ("narrative", "text", "complaint_text", "body", "redacted")

# Nothing in this file should look like prose. Ids, labels, and short notes only.
MAX_CELL_CHARS = 200


def _read(path):
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames or [], list(reader)


@pytest.mark.skipif(not DEDUP_PAIRS.exists(), reason="not curated yet")
def test_dedup_pairs_carry_ids_not_narratives():
    header, rows = _read(DEDUP_PAIRS)
    lowered = [h.lower() for h in header]

    for col in lowered:
        assert not any(bad in col for bad in FORBIDDEN_SUBSTRINGS), (
            f"column {col!r} looks like narrative text. dedup_eval_pairs.csv is "
            f"committed to git; docs/DATA.md §6 forbids narrative text in git. "
            f"Store complaint ids and join the text from DuckDB at eval time."
        )

    assert {"complaint_id_a", "complaint_id_b", "label"} <= set(lowered), (
        f"expected id-based schema, got {header}"
    )

    for i, row in enumerate(rows, start=2):
        for col, value in row.items():
            assert len(value or "") <= MAX_CELL_CHARS, (
                f"{DEDUP_PAIRS.name} line {i}, column {col!r}: {len(value)} chars. "
                f"That is prose, not an id or a label."
            )


@pytest.mark.skipif(not ENFORCEMENT.exists(), reason="not curated yet")
def test_enforcement_actions_schema():
    header, rows = _read(ENFORCEMENT)
    required = {
        "action_id",
        "filed_date",
        "company_raw",
        "company_canonical_id",
        "product_family",
        "harm_summary",
        "harm_keywords",
        "conduct_start",
        "source_url",
        "usable",
    }
    present = {h.lower() for h in header}
    assert required <= present, f"missing columns: {required - present}"
    for i, row in enumerate(rows, start=2):
        assert row["usable"].strip().lower() in {"true", "false"}, (
            f"line {i}: usable must be true/false"
        )
        if row["usable"].strip().lower() == "false":
            assert row.get("exclusion_reason", "").strip(), (
                f"line {i}: usable=false requires an exclusion_reason "
                f"(docs/DATA.md §4 keeps excluded rows for transparency)"
            )


@pytest.mark.skipif(not RAG_MANIFEST.exists(), reason="human-authored benchmark pending")
def test_rag_manifest_is_id_only_and_matches_the_strict_benchmark_contract():
    from src.llm.eval import MANIFEST_HEADER, load_manifest

    header, rows = _read(RAG_MANIFEST)
    assert header == list(MANIFEST_HEADER)
    assert len(rows) == 30
    assert all(set(row) == set(MANIFEST_HEADER) for row in rows)
    assert all(len(value) <= MAX_CELL_CHARS for row in rows for value in row.values())
    assert len(load_manifest(RAG_MANIFEST)) == 30
