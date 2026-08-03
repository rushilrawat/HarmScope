"""Crosswalk, company canonicalization, and the complaints/narratives build."""

from __future__ import annotations

import csv
from datetime import date

import pytest

from src.ingestion.build import build_complaints
from src.normalization import taxonomy
from src.normalization.company import build_canonical, company_id


@pytest.fixture
def loaded(con, tmp_path):
    """A tiny corpus in complaints_raw spanning both schema eras."""
    rows = [
        # pre-2017 labels
        (1, date(2016, 3, 1), "Credit reporting", None, "EQUIFAX, INC."),
        (2, date(2016, 5, 1), "Consumer Loan", "Vehicle loan", "ALLY FINANCIAL INC."),
        (3, date(2016, 6, 1), "Consumer Loan", "Installment loan", "ALLY FINANCIAL INC."),
        (4, date(2016, 7, 1), "Bank account or service", None, "Equifax Inc"),
        # 2017-2023 labels
        (5, date(2019, 2, 1),
         "Credit reporting, credit repair services, or other personal consumer reports",
         None, "EQUIFAX, INC."),
        (6, date(2019, 3, 1), "Credit card or prepaid card",
         "General-purpose credit card or charge card", "SYNCHRONY FINANCIAL"),
        (7, date(2019, 4, 1), "Credit card or prepaid card",
         "Gift card", "SYNCHRONY FINANCIAL"),
        # 2023+ labels
        (8, date(2024, 1, 1), "Credit reporting or other personal consumer reports",
         None, "EQUIFAX, INC."),
    ]
    for cid, d, product, sub, company in rows:
        con.execute(
            "INSERT INTO complaints_raw (complaint_id, date_received, product, "
            "sub_product, company_raw, has_narrative) VALUES (?, ?, ?, ?, ?, ?)",
            [cid, d, product, sub, company, cid in (1, 5)],
        )
    taxonomy.load_crosswalk(con)
    return con


# --------------------------------------------------------------------------
# Crosswalk
# --------------------------------------------------------------------------
def test_crosswalk_covers_the_whole_corpus(loaded):
    assert taxonomy.uncovered_labels(loaded) == []


def test_all_three_credit_reporting_eras_map_to_one_family(loaded):
    build_canonical(loaded)
    build_complaints(loaded, date(2015, 1, 1))
    fams = dict(
        loaded.execute(
            "SELECT complaint_id, product_family FROM complaints "
            "WHERE complaint_id IN (1, 5, 8)"
        ).fetchall()
    )
    assert set(fams.values()) == {"credit_reporting"}


def test_splits_route_on_sub_product(loaded):
    """`Consumer Loan` and `Credit card or prepaid card` are splits, not renames."""
    build_canonical(loaded)
    build_complaints(loaded, date(2015, 1, 1))
    fams = dict(
        loaded.execute(
            "SELECT complaint_id, product_family FROM complaints"
        ).fetchall()
    )
    assert fams[2] == "vehicle_loan"     # Consumer Loan / Vehicle loan
    assert fams[3] == "personal_loan"    # Consumer Loan / Installment loan
    assert fams[6] == "credit_card"      # .../ General-purpose credit card
    assert fams[7] == "prepaid_card"     # .../ Gift card


def test_uncovered_label_is_reported_not_defaulted(loaded):
    loaded.execute(
        "INSERT INTO complaints_raw (complaint_id, date_received, product, "
        "company_raw, has_narrative) VALUES (99, DATE '2024-01-01', "
        "'Crypto rug pull', 'ACME', false)"
    )
    uncovered = taxonomy.uncovered_labels(loaded)
    assert uncovered and uncovered[0][0] == "Crypto rug pull"


def test_committed_crosswalk_has_no_duplicate_keys():
    path = taxonomy.crosswalk_path()
    with path.open(newline="", encoding="utf-8") as fh:
        keys = [(r["product_raw"], r["sub_product_raw"]) for r in csv.DictReader(fh)]
    assert len(keys) == len(set(keys))


def test_every_crosswalk_row_names_a_family():
    path = taxonomy.crosswalk_path()
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            assert row["product_family"].strip(), row


# --------------------------------------------------------------------------
# Company canonicalization
# --------------------------------------------------------------------------
def test_exact_normalization_merges_only(loaded):
    """'EQUIFAX, INC.' and 'Equifax Inc' normalize identically -> one company."""
    n_co, n_alias = build_canonical(loaded)
    assert n_alias == 4  # four distinct raw strings
    rows = dict(
        loaded.execute(
            "SELECT alias_raw, company_id FROM company_alias"
        ).fetchall()
    )
    assert rows["EQUIFAX, INC."] == rows["Equifax Inc"] == company_id("EQUIFAX")
    assert rows["ALLY FINANCIAL INC."] != rows["SYNCHRONY FINANCIAL"]
    assert n_co == 3


def test_similar_but_distinct_names_are_not_merged(con):
    """Trap T6. A wrong merge silently corrupts every company-level statistic."""
    for i, name in enumerate(["WELLS FARGO BANK, N.A.", "WELLS FARGO & COMPANY"], 1):
        con.execute(
            "INSERT INTO complaints_raw (complaint_id, date_received, product, "
            "company_raw, has_narrative) VALUES (?, DATE '2020-01-01', 'Mortgage', "
            "?, false)", [i, name]
        )
    build_canonical(con)
    ids = {r[0] for r in con.execute("SELECT company_id FROM company_alias").fetchall()}
    assert len(ids) == 2, "fuzzy similarity must propose merges, never perform them"


def test_manual_overrides_win(con, tmp_path):
    for i, name in enumerate(["WELLS FARGO BANK, N.A.", "WELLS FARGO & COMPANY"], 1):
        con.execute(
            "INSERT INTO complaints_raw (complaint_id, date_received, product, "
            "company_raw, has_narrative) VALUES (?, DATE '2020-01-01', 'Mortgage', "
            "?, false)", [i, name]
        )
    manual = tmp_path / "manual.csv"
    manual.write_text(
        "alias_raw,company_id\n"
        '"WELLS FARGO BANK, N.A.",wells-fargo\n'
        '"WELLS FARGO & COMPANY",wells-fargo\n'
    )
    build_canonical(con, manual_path=manual)
    ids = {r[0] for r in con.execute("SELECT company_id FROM company_alias").fetchall()}
    assert ids == {"wells-fargo"}
    assert con.execute(
        "SELECT verified_by FROM company_canonical WHERE company_id = 'wells-fargo'"
    ).fetchone()[0] == "manual"


def test_company_id_is_stable_and_slug_shaped():
    assert company_id("WELLS FARGO BANK") == "wells-fargo-bank"
    assert company_id("WELLS FARGO BANK") == company_id("Wells Fargo Bank")
    assert company_id("") == "unknown"


# --------------------------------------------------------------------------
# complaints / narratives
# --------------------------------------------------------------------------
def test_window_filter_and_period_month(loaded):
    build_canonical(loaded)
    loaded.execute(
        "INSERT INTO complaints_raw (complaint_id, date_received, product, "
        "company_raw, has_narrative) VALUES (50, DATE '2013-05-02', 'Mortgage', "
        "'EQUIFAX, INC.', false)"
    )
    n = build_complaints(loaded, date(2015, 1, 1))
    assert n == 8  # the 2013 row is excluded
    pm = loaded.execute(
        "SELECT period_month FROM complaints WHERE complaint_id = 2"
    ).fetchone()[0]
    assert pm == date(2016, 5, 1)


def test_rebuild_is_a_clean_overwrite(loaded):
    build_canonical(loaded)
    assert build_complaints(loaded, date(2015, 1, 1)) == 8
    assert build_complaints(loaded, date(2015, 1, 1)) == 8
