"""Company name normalization.

docs/DATA.md §3.5 step 1. This is only the deterministic first step — case,
punctuation, and trailing corporate suffixes. Fuzzy blocking (step 2) and the
manual review of the top 300 companies by volume (step 3) come in Phase 1.

Deliberately conservative. Trap T6: an over-aggressive merge silently corrupts
every company-level statistic, and a corrupted PRR looks exactly like a real
one. So `WELLS FARGO BANK, N.A.` normalizes to `WELLS FARGO BANK`, not to
`WELLS FARGO` — those stay distinct strings and a human decides whether they
are the same legal entity. Precision over recall on merges, always.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import duckdb

from src.config import PATHS

MANUAL_CSV = "company_canonical_manual.csv"
REVIEW_CSV = "company_merge_review.csv"

# Trailing legal-form suffixes only. Multi-token entries are matched as a unit.
# Kept tight on purpose: every addition here is a potential silent merge.
SUFFIXES: tuple[tuple[str, ...], ...] = (
    ("NATIONAL", "ASSOCIATION"),
    ("INCORPORATED",),
    ("CORPORATION",),
    ("COMPANY",),
    ("LIMITED",),
    ("HOLDINGS",),
    ("INC",),
    ("LLC",),
    ("LLP",),
    ("PLC",),
    ("LTD",),
    ("CORP",),
    ("LP",),
    ("NA",),
    ("SA",),
    ("CO",),
    ("FSB",),
)

_DOTS = re.compile(r"\.")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_company(raw: str) -> str:
    """Uppercase, strip punctuation, drop trailing legal-form suffixes.

    Returns "" for empty or suffix-only input rather than raising — the caller
    decides how to handle an unmappable company string, and a crash inside a
    normalization pass over 10^7 rows is not the right failure mode.
    """
    if not raw:
        return ""

    # Periods are removed, not replaced with a space, so acronyms survive as
    # single tokens: "N.A." -> "NA" rather than "N A". Other punctuation becomes
    # a space so "WELLS FARGO & COMPANY" does not fuse into one word.
    text = _PUNCT.sub(" ", _DOTS.sub("", raw.upper()))
    tokens = _WHITESPACE.sub(" ", text).strip().split()

    # Strip repeatedly: "ACME CORP INC" -> "ACME".
    changed = True
    while changed and tokens:
        changed = False
        for suffix in SUFFIXES:
            n = len(suffix)
            if len(tokens) > n and tuple(tokens[-n:]) == suffix:
                tokens = tokens[:-n]
                changed = True
                break

    return " ".join(tokens)


def company_id(canonical_name: str) -> str:
    """Stable slug. Must not change between runs — it is a foreign key."""
    slug = re.sub(r"[^a-z0-9]+", "-", canonical_name.lower()).strip("-")
    return slug or "unknown"


def _manual_overrides(path: Path | None = None) -> dict[str, str]:
    """`alias_raw -> company_id` decisions made by a human.

    docs/DATA.md §3.5 step 3 is explicit that the top-300 review is not to be
    automated. This file is where those decisions live; it wins over anything
    derived automatically.
    """
    path = path or PATHS.ground_truth / MANUAL_CSV
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {
            r["alias_raw"]: r["company_id"]
            for r in csv.DictReader(fh)
            if r.get("alias_raw") and r.get("company_id")
        }


def build_canonical(
    con: duckdb.DuckDBPyConnection, manual_path: Path | None = None
) -> tuple[int, int]:
    """Populate `company_canonical` and `company_alias`. Returns (companies, aliases).

    Auto-merges **only** exact matches after normalization — same string once
    case, punctuation and trailing legal-form suffixes are removed. Fuzzy
    similarity is used to *propose* merges for human review (see
    `write_merge_review`), never to perform them.

    Trap T6: an over-aggressive merge silently corrupts every company-level
    statistic, and a corrupted PRR looks exactly like a real one. There is no
    downstream check that catches it, so the only defence is not doing it.
    """
    raw = con.execute(
        "SELECT company_raw, count(*) AS n FROM complaints_raw "
        "WHERE company_raw IS NOT NULL GROUP BY 1"
    ).fetchall()

    overrides = _manual_overrides(manual_path)

    aliases: list[tuple[str, str, float, str]] = []
    totals: dict[str, int] = {}
    names: dict[str, str] = {}

    for company_raw, n in raw:
        if company_raw in overrides:
            cid, method, score = overrides[company_raw], "manual", 1.0
            names.setdefault(cid, normalize_company(company_raw) or company_raw)
        else:
            canonical = normalize_company(company_raw) or company_raw
            cid, method, score = company_id(canonical), "fuzzy", 1.0
            names[cid] = canonical
        aliases.append((company_raw, cid, score, method))
        totals[cid] = totals.get(cid, 0) + n

    con.execute("DELETE FROM company_alias")
    con.execute("DELETE FROM company_canonical")
    con.executemany(
        "INSERT INTO company_canonical "
        "(company_id, canonical_name, verified_by, n_complaints) VALUES (?, ?, ?, ?)",
        [
            (cid, names[cid],
             "manual" if cid in set(overrides.values()) else "fuzzy",
             totals[cid])
            for cid in totals
        ],
    )
    con.executemany(
        "INSERT INTO company_alias (alias_raw, company_id, score, method) "
        "VALUES (?, ?, ?, ?)",
        aliases,
    )
    return len(totals), len(aliases)


def write_merge_review(
    con: duckdb.DuckDBPyConnection,
    out_path: Path | None = None,
    top_n: int = 300,
    threshold: int = 88,
) -> tuple[Path, int]:
    """Propose merge candidates among the top-N companies, for a human to decide.

    Output is a worklist, not a mapping: nothing here is applied until a human
    moves a row into `company_canonical_manual.csv`. Blocking by volume follows
    docs/DATA.md §3.5 — the top few hundred companies carry the vast majority
    of the corpus, so that is where a wrong merge does the most damage and
    where review effort pays.
    """
    # ponytail: pairwise loop over the upper triangle, not process.cdist —
    # cdist needs numpy, which is a Phase 3 dependency, and top_n=300 is only
    # ~45k comparisons. Revisit if this ever runs over all 8k companies.
    from rapidfuzz import fuzz

    out_path = out_path or PATHS.ground_truth / REVIEW_CSV
    rows = con.execute(
        "SELECT canonical_name, company_id, n_complaints FROM company_canonical "
        "ORDER BY n_complaints DESC LIMIT ?", [top_n]
    ).fetchall()
    names = [r[0] for r in rows]
    by_name = {r[0]: (r[1], r[2]) for r in rows}

    candidates = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            score = fuzz.token_set_ratio(a, b)
            if score >= threshold:
                candidates.append((int(score), a, by_name[a], b, by_name[b]))

    candidates.sort(key=lambda c: -c[0])
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "name_a", "company_id_a", "n_a",
                    "name_b", "company_id_b", "n_b", "decision"])
        for score, a, (aid, an), b, (bid, bn) in candidates:
            w.writerow([score, a, aid, an, b, bid, bn, ""])
    return out_path, len(candidates)
