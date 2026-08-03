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

import re

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
