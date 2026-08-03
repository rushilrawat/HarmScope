"""Narrative text normalization and content hashing.

docs/METHODOLOGY.md §2.2 Tier 1: exact-duplicate detection is a group-by over
the sha256 of *normalized* text — lowercase, punctuation stripped, digit runs
collapsed to `#`, whitespace collapsed.

Collapsing digit runs is the point, not a side effect. Credit-repair templates
vary mainly in inserted account numbers and dates (docs/DATA.md §3.2), so two
narratives that differ only in those fields must hash identically or Tier 1
catches nothing. The words still have to match exactly.
"""

from __future__ import annotations

import hashlib
import re

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_DIGIT_RUN = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Canonical form used for exact-duplicate grouping.

    Punctuation is removed rather than replaced with a space so that formatting
    variants of the same string collapse together: `4111-1111` and `41111111`
    both become `#`.
    """
    if not text:
        return ""
    out = text.lower()
    out = _PUNCT.sub("", out)
    out = _DIGIT_RUN.sub("#", out)
    return _WHITESPACE.sub(" ", out).strip()


def text_hash(text: str) -> str:
    """sha256 of the normalized text. Stored as `narratives.text_hash`."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
