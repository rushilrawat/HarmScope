"""Scrape the CFPB enforcement-action listing into a curation worklist.

docs/DATA.md §4. There is no clean bulk API for these, so the listing is
scraped and then **hand-curated**. This module does only the mechanical half —
slug, filed date, company string, description, URL, and a company match
attempt. The columns that require judgement (`harm_summary`, `product_family`,
`usable`, `harm_keywords`, `conduct_start`) are left blank for a human.

## Why this runs before any detection

Trap T5: adding enforcement actions after seeing results is p-hacking, so the
set must be selected and frozen first. Running the scrape while zero signals
exist makes cherry-picking impossible by construction rather than by promise.
`EVALUATION.md` §5 item 4 then verifies the freeze commit predates the first
detection run.

## Two things the scraper must not quietly get wrong

`filed_date` is when CFPB went public, not when CFPB knew (docs/DATA.md §4) —
lead time measured against it is lead time vs. public disclosure. The scraper
records it verbatim and the caveat belongs in every reported metric.

Parsed counts are checked against the total the page advertises. A listing
redesign that silently halves the yield would otherwise produce a smaller
ground-truth set and a quietly easier backtest.
"""

from __future__ import annotations

import csv
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from src.config import PATHS
from src.ingestion.download import USER_AGENT

LISTING_URL = "https://www.consumerfinance.gov/enforcement/actions/"
CANDIDATES_CSV = "enforcement_candidates.csv"
PAGE_DELAY_S = 1.0  # be a polite scraper against a public agency site

_ARTICLE = re.compile(r'<article class="o-post-preview".*?</article>', re.S)
_DATE = re.compile(r'<time datetime="(\d{4}-\d{2}-\d{2})')
_LINK = re.compile(r'<a href="(/enforcement/actions/([a-z0-9\-]+)/)">(.*?)</a>', re.S)
_DESC = re.compile(
    r'<div class="o-post-preview__description">(.*?)</div>', re.S
)
_TOTAL = re.compile(r"([\d,]+)\s+(?:results|filtered results)")
_TAGS = re.compile(r"<[^>]+>")


class ScrapeMismatch(RuntimeError):
    """Parsed row count disagrees with the total the listing advertises."""


@dataclass
class Action:
    action_id: str
    filed_date: str
    company_raw: str
    source_url: str
    description: str


def _clean(html: str) -> str:
    return re.sub(r"\s+", " ", _TAGS.sub("", html)).strip()


def fetch(url: str) -> str:
    # Guard before constructing the request, so a file:// or custom scheme can
    # never reach urlopen even if a caller passes one.
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https URL: {url!r}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def parse_page(html: str) -> list[Action]:
    out: list[Action] = []
    for block in _ARTICLE.findall(html):
        d = _DATE.search(block)
        link = _LINK.search(block)
        if not (d and link):
            continue
        desc = _DESC.search(block)
        out.append(
            Action(
                action_id=link.group(2),
                filed_date=d.group(1),
                company_raw=_clean(link.group(3)),
                source_url="https://www.consumerfinance.gov" + link.group(1),
                description=_clean(desc.group(1)) if desc else "",
            )
        )
    return out


def advertised_total(html: str) -> int | None:
    m = _TOTAL.search(_clean(html))
    return int(m.group(1).replace(",", "")) if m else None


def scrape(max_pages: int = 40) -> list[Action]:
    """Walk the listing until a page yields nothing new."""
    first = fetch(LISTING_URL)
    total = advertised_total(first)
    actions = parse_page(first)
    seen = {a.action_id for a in actions}

    for page in range(2, max_pages + 1):
        html = fetch(f"{LISTING_URL}?page={page}")
        fresh = [a for a in parse_page(html) if a.action_id not in seen]
        if not fresh:
            break
        seen.update(a.action_id for a in fresh)
        actions.extend(fresh)
        time.sleep(PAGE_DELAY_S)

    if total is not None and len(actions) < total * 0.9:
        raise ScrapeMismatch(
            f"parsed {len(actions)} actions but the listing advertises {total}. "
            f"The page layout probably changed — fix the parser rather than "
            f"accepting a smaller ground-truth set, which would quietly make "
            f"the backtest easier."
        )
    return actions


def in_window(actions: list[Action], start: date, end: date) -> list[Action]:
    return [a for a in actions if start <= date.fromisoformat(a.filed_date) <= end]


def write_candidates(
    actions: list[Action],
    company_match: dict[str, str] | None = None,
    path: Path | None = None,
) -> Path:
    """Write the curation worklist.

    Judgement columns are deliberately blank. `harm_summary` in the curator's
    own words is what docs/DATA.md §4 asks for; the scraped `description` is
    CFPB's boilerplate and is carried separately so it informs curation without
    being mistaken for it.
    """
    path = path or PATHS.ground_truth / CANDIDATES_CSV
    company_match = company_match or {}
    header = [
        "action_id", "filed_date", "company_raw", "company_canonical_id",
        "product_family", "harm_summary", "harm_keywords", "conduct_start",
        "source_url", "usable", "exclusion_reason", "cfpb_description",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for a in sorted(actions, key=lambda x: x.filed_date):
            row = dict.fromkeys(header, "")
            row.update(asdict(a))
            row.pop("description", None)
            row["cfpb_description"] = a.description
            row["company_canonical_id"] = company_match.get(a.action_id, "")
            w.writerow(row)
    return path


def match_companies(con, actions: list[Action]) -> dict[str, str]:
    """Best-effort `action_id -> company_id`, exact normalized match only.

    Deliberately conservative, for the same reason as company canonicalization
    (trap T6): a wrong company match silently attaches an enforcement action to
    the wrong complaint stream, and every lead-time number computed from it is
    then meaningless. Unmatched rows are left blank for a human.
    """
    from src.normalization.company import normalize_company

    lookup = {
        normalize_company(name): cid
        for name, cid in con.execute(
            "SELECT canonical_name, company_id FROM company_canonical"
        ).fetchall()
    }
    return {
        a.action_id: lookup[normalize_company(a.company_raw)]
        for a in actions
        if normalize_company(a.company_raw) in lookup
    }
