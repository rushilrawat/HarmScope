"""Curation of enforcement candidates into frozen ground truth.

Two properties are load-bearing and neither is about a count: that ambiguity
refuses rather than guesses, and that nothing in the module can read a signal.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from src.evaluation import curate


def _index(**names) -> curate.Index:
    class Fake:
        def execute(self, _sql):
            return self

        def fetchall(self):
            return list(names.items())

    return curate.Index.build(Fake())


def test_normalize_strips_legal_suffixes_repeatedly():
    assert curate.normalize("Acme Financial Services, Inc.") == "ACME FINANCIAL SERVICES"
    assert curate.normalize("Foo Holdings, LLC") == "FOO"
    assert curate.normalize("Bank of X, N.A.") == "BANK OF X"


def test_fragments_split_a_multi_entity_filing():
    got = curate.fragments("TransUnion Interactive, Inc., TransUnion, LLC, and TransUnion")
    assert got == ["TRANSUNION INTERACTIVE", "TRANSUNION"]


def test_exact_match_wins():
    idx = _index(a="Acme Financial Services, Inc.", b="Other Co")
    assert curate.resolve("ACME FINANCIAL SERVICES", idx) == ("a", "exact")


def test_unique_prefix_matches_but_ambiguity_refuses():
    """Trap T6: a wrong company makes every lead time from it meaningless."""
    # Suffix stripping makes "Credit Acceptance Corporation" an EXACT match, so
    # the prefix path needs a name with a real extra token.
    idx = _index(a="Credit Acceptance Services Group")
    assert curate.resolve("CREDIT ACCEPTANCE", idx) == ("a", "unique-prefix")

    both = _index(a="Credit Acceptance Services Group", b="Credit Acceptance Auto")
    company_id, how = curate.resolve("CREDIT ACCEPTANCE", both)
    assert company_id is None and how == "ambiguous"


def test_prefix_must_be_whole_tokens():
    """CREDIT must not match CREDITORS by character prefix."""
    idx = _index(a="Creditors Alliance")
    assert curate.resolve("CREDIT", idx)[0] is None


def test_usable_has_no_volume_threshold(tmp_path):
    """EVALUATION §1.4 Rule 1: a three-complaint company stays in and misses."""
    class Fake:
        def __init__(self):
            self.n = 0

        def execute(self, sql):
            self.sql = sql
            return self

        def fetchall(self):
            if "company_canonical" in self.sql:
                return [("tiny", "Tiny Bank")]
            return [("tiny", 3)]          # three narratives, far below any gate

    rows, tally = curate.curate(Fake(), [{
        "action_id": "a", "filed_date": "2020-05-01", "company_raw": "Tiny Bank",
        "source_url": "u", "cfpb_description": "d",
    }])
    assert rows[0]["usable"] == "true", tally


def test_window_and_unresolved_are_excluded_with_a_reason():
    class Fake:
        def execute(self, sql):
            self.sql = sql
            return self

        def fetchall(self):
            return [("known", "Known Bank")] if "company_canonical" in self.sql \
                else [("known", 100)]

    rows, _ = curate.curate(Fake(), [
        {"action_id": "old", "filed_date": "2016-05-01", "company_raw": "Known Bank",
         "source_url": "u", "cfpb_description": ""},
        {"action_id": "ghost", "filed_date": "2020-05-01", "company_raw": "Nobody At All",
         "source_url": "u", "cfpb_description": ""},
    ])
    assert rows[0]["exclusion_reason"] == "outside-evaluable-window"
    assert rows[1]["exclusion_reason"].startswith("company-unresolved")


def test_harm_keywords_are_never_populated():
    """EVALUATION §5 item 7 — they must not reach the detection path at all.

    The cheapest way to guarantee that is for the column to be empty in the
    committed file, so there is nothing to leak even by accident.
    """
    path = Path("data/ground_truth") / curate.ACTIONS_CSV
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "curated file is empty"
    assert all(not r["harm_keywords"].strip() for r in rows)


def test_curation_module_cannot_read_detection_output():
    """§1.4's contamination bound is only real if the code obeys it."""
    source = Path("src/evaluation/curate.py").read_text().lower()
    for table in ("from signals", "from clusters", "from campaigns",
                  "cluster_members", "cluster_novelty"):
        assert table not in source, f"curate.py references {table!r}"


def test_committed_ground_truth_is_within_the_evaluable_window():
    path = Path("data/ground_truth") / curate.ACTIONS_CSV
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        usable = [r for r in csv.DictReader(fh) if r["usable"] == "true"]
    assert len(usable) >= 20, "PROJECT_SPEC §6 requires >= 20 usable actions"
    for row in usable:
        filed = date.fromisoformat(row["filed_date"])
        assert curate.WINDOW_START <= filed <= curate.WINDOW_END
        assert row["company_canonical_id"], "a usable action must resolve to a company"
