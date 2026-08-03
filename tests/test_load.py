"""Loader contract, checked against a CFPB-shaped fixture.

The fixture header is the live CFPB export header verified 2026-08-03. The rows
carry the shapes that actually break naive loaders: a quoted comma inside
`Product`, an absent tag written as the literal string `None`, ISO timestamps
in one row and plain dates in another, and an empty narrative.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pytest

from src.ingestion.load import COLUMN_MAP, HeaderMismatch, csv_row_count, load_raw

HEADER = (
    "Date received,Product,Sub-product,Issue,Sub-issue,"
    "Consumer complaint narrative,Company public response,Company,State,"
    "ZIP code,Tags,Submitted via,Date sent to company,"
    "Company response to consumer,Timely response?,Complaint ID"
)

SECRET = "my account 4455667788990 and email me at leak@example.com"

ROWS = [
    # Quoted comma in Product; a narrative present; ISO timestamp dates.
    '2023-03-11T14:26:56.000Z,"Credit reporting, credit repair services, or other '
    'personal consumer reports",Credit reporting,Improper use of your report,'
    f'Reporting company used your report improperly,"{SECRET}",'
    "Company chooses not to provide a public response,Experian Information Solutions Inc.,"
    "VA,22003,None,Web,2023-03-11T14:49:28.000Z,Closed with explanation,Yes,6681519",
    # No narrative; plain dates; a real tag; blank state and ZIP.
    "2019-07-02,Mortgage,Conventional home mortgage,Trouble during payment process,,"
    ",,WELLS FARGO BANK; N.A.,,,Older American,Referral,2019-07-05,"
    "Closed with monetary relief,No,3300111",
    # Whitespace-only narrative must not count as a narrative.
    "2021-01-15,Student loan,Federal student loan servicing,Dealing with your lender,,"
    '"   ",,Navient Solutions; LLC,TX,73301,Servicemember,Web,2021-01-16,'
    "Closed with explanation,Yes,4100222",
]


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "complaints.csv"
    p.write_text(HEADER + "\n" + "\n".join(ROWS) + "\n", encoding="utf-8")
    return p


def test_row_counts_reconcile(con, csv_path):
    """ROADMAP Phase 1 acceptance: raw == loaded."""
    loaded = load_raw(con, csv_path)
    assert loaded == csv_row_count(con, csv_path) == 3


def test_narrative_text_never_enters_the_database(con, csv_path):
    """docs/DATA.md §6. complaints_raw records presence, never content.

    If this ever fails, raw consumer-written text is sitting in a .duckdb file
    that gets copied around — and `has_narrative` was the only thing that was
    ever supposed to be there.
    """
    load_raw(con, csv_path)
    dumped = str(con.execute("SELECT * FROM complaints_raw").fetchall())
    assert SECRET not in dumped
    assert "leak@example.com" not in dumped
    assert "4455667788990" not in dumped


def test_has_narrative_flags(con, csv_path):
    load_raw(con, csv_path)
    flags = dict(
        con.execute(
            "SELECT complaint_id, has_narrative FROM complaints_raw ORDER BY complaint_id"
        ).fetchall()
    )
    assert flags == {3300111: False, 4100222: False, 6681519: True}


def test_dates_parse_in_both_export_formats(con, csv_path):
    load_raw(con, csv_path)
    got = dict(
        con.execute(
            "SELECT complaint_id, date_received FROM complaints_raw"
        ).fetchall()
    )
    assert got[6681519] == date(2023, 3, 11)  # ISO timestamp
    assert got[3300111] == date(2019, 7, 2)  # plain date


def test_quoted_comma_in_product_survives(con, csv_path):
    load_raw(con, csv_path)
    product = con.execute(
        "SELECT product FROM complaints_raw WHERE complaint_id = 6681519"
    ).fetchone()[0]
    assert product.startswith("Credit reporting, credit repair")


def test_none_and_blank_become_null(con, csv_path):
    load_raw(con, csv_path)
    tags, state, zip_code = con.execute(
        "SELECT tags, state, zip_code FROM complaints_raw WHERE complaint_id = 6681519"
    ).fetchone()
    assert tags is None  # literal 'None' in the export
    row = con.execute(
        "SELECT state, zip_code, tags FROM complaints_raw WHERE complaint_id = 3300111"
    ).fetchone()
    assert row == (None, None, "Older American")
    assert (state, zip_code) == ("VA", "22003")


def test_timely_response_is_boolean(con, csv_path):
    load_raw(con, csv_path)
    got = dict(
        con.execute(
            "SELECT complaint_id, timely_response FROM complaints_raw"
        ).fetchall()
    )
    assert got[6681519] is True
    assert got[3300111] is False


def test_reload_is_a_clean_overwrite(con, csv_path):
    assert load_raw(con, csv_path) == 3
    assert load_raw(con, csv_path) == 3  # not 6


def test_missing_column_fails_loudly(con, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Date received,Product,Complaint ID\n2020-01-01,Mortgage,1\n")
    with pytest.raises(HeaderMismatch, match="missing expected columns"):
        load_raw(con, bad)


def test_extra_columns_are_tolerated(con, tmp_path):
    """CFPB has added columns before; that must not break the loader."""
    extra = tmp_path / "extra.csv"
    extra.write_text(
        HEADER + ",Consumer consent provided?\n" + ROWS[1] + ",Consent not provided\n"
    )
    assert load_raw(con, extra) == 1


def test_duplicate_complaint_id_is_rejected(con, tmp_path):
    dupe = tmp_path / "dupe.csv"
    dupe.write_text(HEADER + "\n" + ROWS[1] + "\n" + ROWS[1] + "\n")
    with pytest.raises(duckdb.ConstraintException):
        load_raw(con, dupe)


def test_failed_load_leaves_previous_contents_intact(con, csv_path, tmp_path):
    """A half-loaded table that downstream stages would happily read is trap T1."""
    load_raw(con, csv_path)
    dupe = tmp_path / "dupe.csv"
    dupe.write_text(HEADER + "\n" + ROWS[1] + "\n" + ROWS[1] + "\n")
    with pytest.raises(duckdb.ConstraintException):
        load_raw(con, dupe)
    assert con.execute("SELECT count(*) FROM complaints_raw").fetchone()[0] == 3


def test_failed_validation_rolls_back_the_load(con, csv_path):
    """Output that fails its own acceptance checks must not be committed.

    Committing first and asserting afterwards leaves the table full while the
    run is marked failed — downstream stages find rows and proceed happily.
    """
    def reject(_c):
        raise RuntimeError("acceptance failed")

    with pytest.raises(RuntimeError, match="acceptance failed"):
        load_raw(con, csv_path, validate=reject)
    assert con.execute("SELECT count(*) FROM complaints_raw").fetchone()[0] == 0


def test_validate_sees_the_loaded_rows(con, csv_path):
    seen = {}

    def peek(c):
        seen["n"] = c.execute("SELECT count(*) FROM complaints_raw").fetchone()[0]

    load_raw(con, csv_path, validate=peek)
    assert seen["n"] == 3


def test_column_map_covers_every_schema_column(con):
    """Every column in complaints_raw is populated by the loader."""
    schema_cols = {
        r[0] for r in con.execute("DESCRIBE complaints_raw").fetchall()
    }
    assert set(COLUMN_MAP.values()) == schema_cols
