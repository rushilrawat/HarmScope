from __future__ import annotations

import pytest

from src.normalization.company import normalize_company
from src.normalization.text import normalize, text_hash


# --------------------------------------------------------------------------
# Narrative text
# --------------------------------------------------------------------------
def test_normalize_collapses_case_punctuation_and_digits():
    assert normalize("Account #4455-6677 was CLOSED!") == "account # was closed"


def test_template_variants_hash_identically():
    """The whole point of Tier 1. Credit-repair templates vary mainly in
    inserted account numbers and dates (docs/DATA.md §3.2)."""
    a = "Please remove account 4455667788 opened on 03/14/2019 from my report."
    b = "Please remove account 9911223344 opened on 11/02/2021 from my report."
    assert text_hash(a) == text_hash(b)


def test_formatting_variants_hash_identically():
    assert text_hash("Card 4111-1111") == text_hash("Card 41111111")


def test_genuinely_different_text_does_not_collide():
    a = "They charged me a late fee I never agreed to."
    b = "They closed my account without any notice at all."
    assert text_hash(a) != text_hash(b)


def test_word_changes_still_separate_documents():
    """Digits collapse; words must not."""
    a = "Please remove account 4455667788 from my report."
    b = "Please dispute account 4455667788 from my report."
    assert text_hash(a) != text_hash(b)


def test_normalize_handles_empty():
    assert normalize("") == ""
    assert text_hash("") == text_hash("   ")


# --------------------------------------------------------------------------
# Company names
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("EQUIFAX, INC.", "EQUIFAX"),
        ("Equifax Inc", "EQUIFAX"),
        ("TRANSUNION INTERMEDIATE HOLDINGS, INC.", "TRANSUNION INTERMEDIATE"),
        ("WELLS FARGO & COMPANY", "WELLS FARGO"),
        ("WELLS FARGO BANK, N.A.", "WELLS FARGO BANK"),
        ("Navient Solutions, LLC", "NAVIENT SOLUTIONS"),
        ("ACME CORP INC", "ACME"),
        ("BANK OF AMERICA, NATIONAL ASSOCIATION", "BANK OF AMERICA"),
    ],
)
def test_company_suffix_stripping(raw, expected):
    assert normalize_company(raw) == expected


def test_distinct_legal_entities_stay_distinct():
    """Trap T6. A wrong merge silently corrupts every company-level statistic,
    and a corrupted PRR looks exactly like a real one."""
    assert normalize_company("WELLS FARGO BANK, N.A.") != normalize_company("WELLS FARGO & CO")


def test_embedded_suffix_tokens_are_not_stripped():
    assert normalize_company("COSTCO WHOLESALE") == "COSTCO WHOLESALE"
    assert normalize_company("NATIONAL ASSOCIATION OF REALTORS") == (
        "NATIONAL ASSOCIATION OF REALTORS"
    )


def test_company_normalization_is_idempotent():
    once = normalize_company("EQUIFAX, INC.")
    assert normalize_company(once) == once


def test_empty_and_suffix_only_input():
    assert normalize_company("") == ""
    assert normalize_company("INC") == "INC"  # never strip to nothing
