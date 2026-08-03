from __future__ import annotations

import pytest

from src.normalization.pii import PATTERN_NAMES, luhn_ok, redact


@pytest.mark.parametrize(
    ("text", "pattern", "expected"),
    [
        ("write to jane.doe+cfpb@example.co.uk today", "email", "[EMAIL]"),
        ("my ssn is 123-45-6789 ok", "ssn", "[SSN]"),
        ("card 4111 1111 1111 1111 was cloned", "card", "[CARD]"),
        ("card 4111-1111-1111-1111 was cloned", "card", "[CARD]"),
        ("call (415) 555-0123 now", "phone", "[PHONE]"),
        ("call 415-555-0123 now", "phone", "[PHONE]"),
        ("call +1 415.555.0123 now", "phone", "[PHONE]"),
        ("lives at 1600 Pennsylvania Ave now", "address", "[ADDRESS]"),
        ("lives at 42 Cherry Tree LANE now", "address", "[ADDRESS]"),
        ("spoke to Ms. Rivera about it", "name", "[NAME]"),
        ("account 000123456789 is closed", "account", "[ACCOUNT]"),
    ],
)
def test_each_pattern_redacts_and_counts(text, pattern, expected):
    result = redact(text)
    assert expected in result.text
    assert result.counts[pattern] == 1
    assert result.total == 1, f"unexpected extra hits: {result.counts}"


@pytest.mark.parametrize(
    "text",
    [
        "I have 5 years of history and 3 accounts in good standing.",
        "They charged me $1,250.00 on 12/03 and again on 01/04.",
        "The account was opened in 2019 and closed in 2021.",
        "I called them 15 times over 6 months about 2 late fees.",
        "Reference XXXX XXXX was masked by the CFPB already.",
    ],
)
def test_benign_text_is_untouched(text):
    result = redact(text)
    assert result.text == text
    assert result.total == 0, f"false positives: {result.counts}"


def test_realistic_narrative():
    narrative = (
        "On XX/XX/XXXX I contacted the company about account 4455667788990. "
        "I spoke with Mr. Alvarez who asked me to email helpdesk@lender.example "
        "and confirm my SSN 987-65-4321. They also had my card 5500 0000 0000 0004 "
        "on file and mailed a notice to 88 Maple Street. I called 800-555-0199 "
        "three times over 2 weeks and paid $45.00 in fees."
    )
    r = redact(narrative)
    assert r.counts == {
        "email": 1, "ssn": 1, "card": 1, "phone": 1,
        "address": 1, "name": 1, "account": 1,
    }
    for leaked in ("4455667788990", "987-65-4321", "5500 0000 0000 0004",
                   "helpdesk@lender.example", "800-555-0199"):
        assert leaked not in r.text
    # Structure preserved, non-PII facts intact.
    assert "$45.00" in r.text and "2 weeks" in r.text and "XX/XX/XXXX" in r.text


def test_redaction_is_idempotent():
    once = redact("card 4111 1111 1111 1111, email a@b.co, at 5 Oak Road")
    twice = redact(once.text)
    assert twice.text == once.text
    assert twice.total == 0


def test_card_wins_over_account():
    """Pattern order is load-bearing: a 16-digit card must not be counted as an
    account number, or the card counter never moves and drift is invisible."""
    r = redact("card 4111111111111111 here")
    assert r.counts["card"] == 1
    assert r.counts["account"] == 0


def test_card_length_account_number_falls_through_to_account():
    """A 13-digit account number and a 13-digit Visa are the same shape. Luhn
    is the discriminator; both are redacted either way."""
    r = redact("account 4455667788990 was closed")
    assert r.counts == {**dict.fromkeys(PATTERN_NAMES, 0), "account": 1}
    assert "[ACCOUNT]" in r.text and "4455667788990" not in r.text


@pytest.mark.parametrize("number", ["4111111111111111", "5500000000000004",
                                    "4012888888881881", "378282246310005"])
def test_luhn_accepts_real_card_numbers(number):
    assert luhn_ok(number)


@pytest.mark.parametrize("number", ["4455667788990", "000123456789", "1234567890123456"])
def test_luhn_rejects_non_cards(number):
    assert not luhn_ok(number)


def test_empty_and_none_safe():
    assert redact("").text == ""
    assert redact("").total == 0
    assert set(redact("").counts) == set(PATTERN_NAMES)


def test_counts_cover_every_pattern():
    assert set(redact("hello").counts) == set(PATTERN_NAMES)
