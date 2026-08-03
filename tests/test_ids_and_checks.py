from __future__ import annotations

from datetime import date

import pytest

from src import checks
from src.ids import COMPANY_TOTAL, campaign_id, cluster_id, parse_cluster_id


# --------------------------------------------------------------------------
# Cluster ids
# --------------------------------------------------------------------------
def test_cluster_ids_do_not_collide_across_refits():
    """The backtest refits clustering once per annual cutoff. A locally unique
    id collides in every table keyed on cluster_id."""
    a = cluster_id("run-2019", "mortgage", 3)
    b = cluster_id("run-2020", "mortgage", 3)
    assert a != b
    assert parse_cluster_id(a)[0] == "run-2019"


def test_cluster_id_round_trip():
    cid = cluster_id("r1", "credit_reporting", 12)
    assert parse_cluster_id(cid) == ("r1", "credit_reporting", "12")


@pytest.mark.parametrize(
    ("run_id", "family"),
    [("", "mortgage"), ("r1", ""), ("r:1", "mortgage"), ("r1", "mort:gage")],
)
def test_cluster_id_rejects_ambiguous_parts(run_id, family):
    with pytest.raises(ValueError):
        cluster_id(run_id, family, 1)


def test_malformed_cluster_id_rejected():
    with pytest.raises(ValueError, match="malformed"):
        parse_cluster_id("mortgage-3")


def test_campaign_ids_do_not_collide_across_refits():
    """Campaign features are time-windowed, so campaigns are regenerated per
    cutoff for the same reason clusters are."""
    assert campaign_id("run-2019", 1) != campaign_id("run-2020", 1)
    assert parse_cluster_id(campaign_id("run-2019", 1)) == ("run-2019", "campaign", "1")


def test_company_total_sentinel_is_not_a_plausible_company_id():
    assert COMPANY_TOTAL.startswith("__") and COMPANY_TOTAL.endswith("__")


# --------------------------------------------------------------------------
# Stage assertions
# --------------------------------------------------------------------------
def test_expect_rows_bounds(con):
    con.execute(
        "INSERT INTO enforcement_actions (action_id, filed_date, usable) "
        "VALUES ('a1', DATE '2020-01-01', true), ('a2', DATE '2021-01-01', false)"
    )
    assert checks.expect_rows(con, "enforcement_actions", min=1, max=5) == 2
    assert checks.expect_rows(con, "enforcement_actions", where="usable") == 1

    with pytest.raises(checks.CheckFailed, match="expected at least 10"):
        checks.expect_rows(con, "enforcement_actions", min=10)
    with pytest.raises(checks.CheckFailed, match="expected at most 1"):
        checks.expect_rows(con, "enforcement_actions", max=1)


def test_empty_table_fails_a_min_check(con):
    """Trap T1: a stage that writes zero rows must not pass."""
    with pytest.raises(checks.CheckFailed, match="0 rows"):
        checks.expect_rows(con, "complaints", min=1)


def test_expect_no_nulls(con):
    con.execute(
        "INSERT INTO enforcement_actions (action_id, filed_date, company_raw, usable) "
        "VALUES ('a1', DATE '2020-01-01', NULL, true)"
    )
    checks.expect_no_nulls(con, "enforcement_actions", ["action_id"])
    with pytest.raises(checks.CheckFailed, match="NULL"):
        checks.expect_no_nulls(con, "enforcement_actions", ["company_raw"])


def test_expect_unique(con):
    con.execute(
        "INSERT INTO taxonomy_crosswalk (product_raw, product_family) "
        "VALUES ('Mortgage', 'mortgage'), ('Mortgage', 'mortgage')"
    )
    with pytest.raises(checks.CheckFailed, match="duplicated"):
        checks.expect_unique(con, "taxonomy_crosswalk", ["product_raw"])


def test_expect_scalar_range(con):
    val = checks.expect_scalar(con, "SELECT 0.42", lo=0.0, hi=1.0, label="coverage")
    assert val == pytest.approx(0.42)
    with pytest.raises(checks.CheckFailed, match="coverage"):
        checks.expect_scalar(con, "SELECT 0.42", lo=0.9, label="coverage")


def test_expect_no_leakage(seeded):
    con, run_id, cid = seeded
    con.execute(
        "INSERT INTO signals (signal_id, run_id, cluster_id, company_id, period_month, "
        "method, statistic, n_supporting, n_supporting_groups, as_of) "
        "VALUES ('s1', ?, ?, ?, DATE '2018-05-01', 'prr', 3.1, 40, 38, DATE '2019-01-01')",
        [run_id, cid, COMPANY_TOTAL],
    )
    checks.expect_no_leakage(con, "signals", date(2019, 1, 1))
    with pytest.raises(checks.CheckFailed, match="LEAKAGE"):
        checks.expect_no_leakage(con, "signals", date(2018, 1, 1))


def test_checks_reject_unsafe_identifiers(con):
    with pytest.raises(ValueError, match="unsafe SQL identifier"):
        checks.expect_rows(con, "runs; DROP TABLE runs")
