"""Neutral display-text sanitization shared by private review exporters."""

from __future__ import annotations

import re

_OSC_SEQUENCE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?")
_ESC_SEQUENCE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[ -/]*[@-~])?")
_BIDI_CONTROLS = frozenset("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def sanitize_display_text(value: str | None) -> str:
    """Strip terminal/bidi controls and normalize whitespace without flattening Unicode."""
    if value is not None and type(value) is not str:
        raise TypeError("display text must be a string or None")
    cleaned = _OSC_SEQUENCE.sub(" ", value or "")
    cleaned = _ESC_SEQUENCE.sub(" ", cleaned)
    cleaned = "".join(
        " "
        if ord(character) < 32 or 0x7F <= ord(character) <= 0x9F or character in _BIDI_CONTROLS
        else character
        for character in cleaned
    )
    return " ".join(cleaned.split())
