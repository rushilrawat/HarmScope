"""Secondary PII sweep over consumer narratives.

docs/DATA.md §6: CFPB scrubs PII, but scrubbing is imperfect and narratives are
consumer-written. Nothing reaches an embedding model, an external API, or a
screen until it has been through here.

Two design choices worth stating:

  - Replace, never delete. `[ACCOUNT]` preserves sentence structure so the
    embedding still sees a well-formed sentence. Deleting the span would
    silently change what the text means.
  - Count per pattern, not just per document. A single runaway regex is
    invisible in an aggregate redaction rate; per-pattern counts make drift
    attributable (docs/DATA.md §6 item 3).

Pattern order is load-bearing. Email runs before phone because addresses can
contain digit runs; card runs before the generic account-number rule because
otherwise a 16-digit card is redacted as `[ACCOUNT]` and the card counter never
moves.

KNOWN LIMITATION — personal names. `[NAME]` catches honorific-prefixed names
only (`Ms. Rivera`). General person-name detection is an NER problem and this
module does not pretend to solve it; CFPB's own masking is the primary defence
and the Phase 10 display path is the backstop. Do not describe this module as
name-safe.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field


def luhn_ok(digits: str) -> bool:
    """Luhn checksum — the structural difference between a payment card and an
    account number of the same length.

    A 13-digit account number and a 13-digit Visa are indistinguishable by
    shape alone, so without this the card rule swallows account numbers and the
    per-pattern counts stop meaning anything. Both are redacted either way; the
    check decides which counter moves, and therefore which drift is visible.
    """
    total, parity = 0, len(digits) % 2
    for i, char in enumerate(digits):
        n = int(char)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _is_card(span: str) -> bool:
    return luhn_ok(re.sub(r"\D", "", span))

# Street-type tokens, matched case-sensitively in Title or UPPER form only.
# Lowercase would make "3 days a week at work way" an address.
_STREET_TOKENS = (
    "Street", "St", "Avenue", "Ave", "Road", "Rd", "Boulevard", "Blvd",
    "Drive", "Dr", "Lane", "Ln", "Court", "Ct", "Circle", "Cir", "Way",
    "Place", "Pl", "Terrace", "Ter", "Parkway", "Pkwy", "Highway", "Hwy",
    "Suite", "Ste", "Apt", "Apartment",
)
_STREET_ALT = "|".join(
    sorted({t for tok in _STREET_TOKENS for t in (tok, tok.upper())},
           key=len, reverse=True)
)

# (name, pattern, placeholder, validator) — order matters, see module docstring.
# A validator that returns False leaves the span alone so a later pattern can
# claim it; `account` is the catch-all and runs last.
PATTERNS: tuple[tuple[str, re.Pattern[str], str, Callable[[str], bool] | None], ...] = (
    (
        "email",
        re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
        "[EMAIL]",
        None,
    ),
    (
        "ssn",
        re.compile(r"(?<!\d)\d{3}[-  ]\d{2}[-  ]\d{4}(?!\d)"),
        "[SSN]",
        None,
    ),
    (
        # 13-16 digits in 4-digit groups, optionally separated, Luhn-valid.
        # A same-length non-card falls through to the account rule below.
        "card",
        re.compile(r"(?<![\d-])(?:\d{4}[ -]?){3}\d{1,4}(?![\d-])"),
        "[CARD]",
        _is_card,
    ),
    (
        # Requires a separator or parens. A bare 10-digit run is ambiguous
        # between a phone and an account number; it falls through to the
        # account rule below, which still redacts it.
        "phone",
        re.compile(
            r"(?<![\d\-.])(?:\+?1[ .\-])?"
            r"(?:\(\d{3}\)\s?|\d{3}[ .\-])\d{3}[ .\-]\d{4}(?![\d\-])"
        ),
        "[PHONE]",
        None,
    ),
    (
        "address",
        re.compile(
            rf"(?<!\d)\d{{1,6}}\s+(?:[A-Za-z][A-Za-z.'\-]*\s+){{1,4}}"
            rf"(?:{_STREET_ALT})\b\.?"
        ),
        "[ADDRESS]",
        None,
    ),
    (
        "name",
        re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+"
                   r"(?:\s+[A-Z][a-z]+)?"),
        "[NAME]",
        None,
    ),
    (
        "account",
        re.compile(r"(?<![\d\-])\d{8,}(?![\d\-])"),
        "[ACCOUNT]",
        None,
    ),
)

PATTERN_NAMES: tuple[str, ...] = tuple(name for name, _, _, _ in PATTERNS)


@dataclass
class Redaction:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> Redaction:
    """Replace PII spans with typed placeholders.

    Idempotent: no placeholder contains a digit or `@`, so a second pass over
    already-redacted text is a no-op.
    """
    counts = dict.fromkeys(PATTERN_NAMES, 0)
    if not text:
        return Redaction(text=text or "", counts=counts)

    for name, pattern, placeholder, validator in PATTERNS:
        if validator is None:
            text, n = pattern.subn(placeholder, text)
        else:
            hits = 0

            def _replace(match: re.Match[str]) -> str:
                nonlocal hits
                span = match.group(0)
                if not validator(span):  # noqa: B023 - called before the next iteration
                    return span
                hits += 1
                return placeholder  # noqa: B023

            text = pattern.sub(_replace, text)
            n = hits
        counts[name] = n
    return Redaction(text=text, counts=counts)
