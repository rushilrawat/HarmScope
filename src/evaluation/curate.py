"""Phase 6: turn scraped enforcement candidates into the frozen ground truth.

Implements `docs/EVALUATION.md` §1.4 exactly, and that section was committed at
`baa0e8d` before this file existed. The ordering is the whole point: curation is
happening after Phases 4-5 ran, so the curator knows which companies alert, and
the only defence is that no step here involves a judgement call the curator
could have made differently.

Nothing in this module reads `signals`, `clusters`, or `campaigns`. It reads the
candidate CSV and `company_canonical`, and that is enforced by a test rather
than by intent.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

CANDIDATES_CSV = "enforcement_candidates.csv"
ACTIONS_CSV = "enforcement_actions.csv"

WINDOW_START = date(2017, 1, 1)
WINDOW_END = date(2024, 12, 31)

# Stripped from the end of a name, repeatedly. A filing writes "Acme Financial
# Services, Inc."; the complaint corpus writes "ACME FINANCIAL SERVICES".
LEGAL_SUFFIXES = {
    "INC", "LLC", "LP", "LTD", "CORP", "CORPORATION", "CO", "NA", "PLLC",
    "PC", "COMPANY", "HOLDINGS",
}
SPLIT = re.compile(r",|\band\b", flags=re.IGNORECASE)


def normalize(name: str) -> str:
    """Uppercase, drop punctuation, strip trailing legal suffixes.

    Periods are deleted rather than replaced by a space, so that `N.A.` — how
    national banks are actually written in filings — collapses to the `NA` the
    suffix list already contains. Replacing them with spaces left `N A`, two
    tokens matching nothing, and the pre-registered rule silently failed to fire
    on a whole class of bank names.
    """
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", (name or "").replace(".", "")).upper()
    tokens = [t for t in text.split() if t]
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def fragments(company_raw: str) -> list[str]:
    """The filing's company string split into individually matchable names."""
    out = []
    for piece in SPLIT.split(company_raw or ""):
        norm = normalize(piece)
        if norm and norm not in out:
            out.append(norm)
    return out


@dataclass(frozen=True)
class Index:
    """Normalized canonical names, for exact and unique-prefix lookup."""

    exact: dict[str, str]
    by_first_token: dict[str, list[tuple[str, str]]]

    @classmethod
    def build(cls, con: duckdb.DuckDBPyConnection) -> Index:
        exact: dict[str, str] = {}
        first: dict[str, list[tuple[str, str]]] = {}
        for company_id, name in con.execute(
            "SELECT company_id, canonical_name FROM company_canonical"
        ).fetchall():
            norm = normalize(name)
            if not norm:
                continue
            # First writer wins, so the mapping does not depend on row order.
            exact.setdefault(norm, company_id)
            first.setdefault(norm.split()[0], []).append((norm, company_id))
        return cls(exact=exact, by_first_token=first)


def resolve(fragment: str, index: Index) -> tuple[str | None, str]:
    """`(company_id, how)` for one fragment. Ambiguity resolves to no match.

    Conservative in the Phase 1 sense (trap T6): a wrong company attaches an
    action to the wrong complaint stream and every lead time computed from it is
    meaningless, so more than one candidate is a refusal rather than a coin flip.
    """
    if not fragment:
        return None, "empty"
    if fragment in index.exact:
        return index.exact[fragment], "exact"

    tokens = fragment.split()
    hits = {
        company_id
        for name, company_id in index.by_first_token.get(tokens[0], [])
        if name.split()[: len(tokens)] == tokens
    }
    if len(hits) == 1:
        return next(iter(hits)), "unique-prefix"
    return None, "ambiguous" if hits else "no-match"


def resolve_action(company_raw: str, index: Index) -> tuple[str | None, str]:
    """First fragment that resolves wins; filings list the parent first."""
    reasons = []
    for fragment in fragments(company_raw):
        company_id, how = resolve(fragment, index)
        if company_id:
            return company_id, how
        reasons.append(how)
    return None, reasons[0] if reasons else "empty"


def read_candidates(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def curate(
    con: duckdb.DuckDBPyConnection, candidates: list[dict]
) -> tuple[list[dict], dict[str, int]]:
    """Apply EVALUATION §1.4 Rules 1 and 2. No discretion anywhere in here."""
    index = Index.build(con)
    volume = dict(con.execute(
        "SELECT company_id, count(*) FROM complaints WHERE has_narrative GROUP BY 1"
    ).fetchall())

    rows, tally = [], {}
    for row in candidates:
        filed = date.fromisoformat(row["filed_date"])
        company_id, how = resolve_action(row["company_raw"], index)

        if not (WINDOW_START <= filed <= WINDOW_END):
            usable, reason = False, "outside-evaluable-window"
        elif company_id is None:
            usable, reason = False, f"company-unresolved:{how}"
        elif volume.get(company_id, 0) == 0:
            usable, reason = False, "no-narrative-complaints"
        else:
            usable, reason = True, None

        tally[reason or "usable"] = tally.get(reason or "usable", 0) + 1
        rows.append({
            "action_id": row["action_id"],
            "filed_date": row["filed_date"],
            "company_raw": row["company_raw"],
            "company_canonical_id": company_id or "",
            "product_family": "",
            # CFPB's own description, captured by the scraper before any signal
            # existed. Writing a summary "in the curator's words" now would put
            # post-hoc language into the file the adjudicator reads.
            "harm_summary": row.get("cfpb_description", ""),
            "harm_keywords": "",        # never populated; EVALUATION §5 item 7
            "conduct_start": "",
            "source_url": row["source_url"],
            "usable": "true" if usable else "false",
            "exclusion_reason": reason or "",
            "resolution_method": how,
        })
    return rows, tally


# Column names follow DATA.md §4, which names the CSV field
# `company_canonical_id` while db/schema.sql calls the table column
# `company_id`. Keeping the CSV aligned with the candidates file it is derived
# from matters more than matching the table it is loaded into; `load()` maps.
HEADER = [
    "action_id", "filed_date", "company_raw", "company_canonical_id", "product_family",
    "harm_summary", "harm_keywords", "conduct_start", "source_url",
    "usable", "exclusion_reason", "resolution_method",
]


def write(rows: list[dict], path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in HEADER})
    return path


def load(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> int:
    """Replace `enforcement_actions` with the curated set."""
    # Everything keyed on action_id goes first, or the foreign keys reject the
    # delete. Discarding them is correct rather than unfortunate: re-curating
    # can change which actions exist, and a backtest result for an action that
    # is no longer in the set is worse than no result. `update_summaries` is the
    # path for changing a description without touching any of this.
    con.execute("DELETE FROM baseline_results")
    con.execute("DELETE FROM backtest_links")
    con.execute("DELETE FROM enforcement_actions")
    con.executemany(
        "INSERT INTO enforcement_actions (action_id, filed_date, company_raw, "
        "company_id, product_family, harm_summary, harm_keywords, conduct_start, "
        "source_url, usable, exclusion_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            (r["action_id"], r["filed_date"], r["company_raw"],
             r["company_canonical_id"] or None, r["product_family"] or None,
             r["harm_summary"] or None, None, None, r["source_url"],
             r["usable"] == "true", r["exclusion_reason"] or None)
            for r in rows
        ],
    )
    return con.execute(
        "SELECT count(*) FROM enforcement_actions WHERE usable"
    ).fetchone()[0]


def update_summaries(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> int:
    """Update `harm_summary` in place, leaving every other column alone.

    Not a delete-and-reload. `baseline_results` and `backtest_links` both key on
    `action_id`, so replacing the table would either fail on the foreign key or
    require discarding backtest output to change a description field that no
    detection stage reads. An UPDATE also makes the invariant obvious: nothing
    that decides which actions are evaluated can move.
    """
    con.executemany(
        "UPDATE enforcement_actions SET harm_summary = ? WHERE action_id = ?",
        [(r["harm_summary"] or None, r["action_id"]) for r in rows],
    )
    return len(rows)
