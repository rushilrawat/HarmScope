"""Phase 6: the adjudication surface, and the record of what was decided.

docs/EVALUATION.md §1.3 step 3-4. This module presents a worklist and writes
verdicts. It deliberately **cannot** determine whether a cluster fired: it reads
the blinded CSV that `worklist.py` emits and never touches the detection tables.

That is enforced rather than intended — `tests/test_leakage.py` item 5 reads this
source and fails if it mentions a signal column, so the blinding survives a
future edit by someone who has forgotten why it matters.

A verdict is `strong`, `partial`, or `none`. Only `strong` counts as a detection
in the headline (§1.3 step 5); `partial` is reported separately rather than
quietly folded in, because folding it in is the easiest way to turn a null
result into a positive one.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import duckdb

VERDICTS = ("strong", "partial", "none")


@dataclass(frozen=True)
class Verdict:
    action_id: str
    cluster_id: str
    match_quality: str
    notes: str


def read_worklist(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def present(rows: list[dict], harm_summary: str, chars: int = 260) -> str:
    """Render one action's worklist for a human to read.

    The action's own description is shown once at the top, then the candidates
    in the order the worklist fixed. Nothing distinguishes a real candidate from
    a decoy, and nothing indicates rank.
    """
    out = [
        "=" * 72,
        "ACTION UNDER REVIEW",
        "=" * 72,
        " ".join(harm_summary.split())[:900],
        "",
        "Mark each candidate strong / partial / none against the description above.",
        "Order is randomized and some candidates are unrelated by construction.",
        "",
    ]
    for row in rows:
        out.append("-" * 72)
        out.append(f"[{row['slot']}] {row['product_family']}  "
                   f"{int(row['n_members']):,} members")
        for key in ("exemplar_1", "exemplar_2", "exemplar_3"):
            if row.get(key):
                out.append("   - " + row[key][:chars])
    return "\n".join(out)


def parse(rows: list[dict]) -> list[Verdict]:
    """Read back a filled-in worklist, rejecting anything unrecognised."""
    out = []
    for row in rows:
        quality = (row.get("match_quality") or "").strip().lower()
        if not quality:
            continue
        if quality not in VERDICTS:
            raise ValueError(
                f"slot {row['slot']}: {quality!r} is not one of {VERDICTS}"
            )
        out.append(Verdict(
            action_id=row["action_id"], cluster_id=row["cluster_id"],
            match_quality=quality, notes=(row.get("notes") or "").strip(),
        ))
    return out


def record(
    con: duckdb.DuckDBPyConnection, verdicts: list[Verdict], adjudicator: str
) -> int:
    """Write to `backtest_links`, stamped with who decided.

    §1.3: "If you are the only adjudicator, say so." The column exists so that a
    single-adjudicator study is visible in the data rather than only in a
    footnote — hidden single-adjudicator is the credibility problem, not single
    adjudicator itself.
    """
    con.executemany(
        "INSERT INTO backtest_links (action_id, cluster_id, match_quality, "
        "adjudicated_by, notes, adjudicated_at) VALUES (?, ?, ?, ?, ?, now()) "
        "ON CONFLICT DO NOTHING",
        [(v.action_id, v.cluster_id, v.match_quality, adjudicator, v.notes or None)
         for v in verdicts],
    )
    return len(verdicts)
