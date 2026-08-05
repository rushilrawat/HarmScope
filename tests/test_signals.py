"""Disproportionality, FDR, and the two changepoint detectors.

The tests that matter most here are the ones about *time*: a detector that
peeks at the future still produces plausible numbers, and the only thing that
catches it is an explicit check that a later observation cannot move an earlier
alarm.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from src.config import SignalConfig
from src.signals import changepoint
from src.signals import disproportionality as dp

CFG = SignalConfig()
MONTHS = [date(2015 + i // 12, i % 12 + 1, 1) for i in range(60)]


def test_prr_matches_the_definition():
    """PRR = [a/(a+b)] / [c/(c+d)] — 20% vs 10% is a PRR of 2."""
    prr, low, high = dp.prr_with_ci(a=20, b=80, c=10, d=90)
    assert np.isclose(prr, 2.0)
    assert low < prr < high, "the point estimate must sit inside its own CI"


def test_prr_ci_widens_as_evidence_shrinks():
    _, low_big, high_big = dp.prr_with_ci(200, 800, 100, 900)
    _, low_small, high_small = dp.prr_with_ci(5, 20, 100, 900)
    assert (high_small - low_small) > (high_big - low_big)


def test_ror_p_value_is_small_for_a_strong_association():
    ror, p = dp.ror_with_p(a=200, b=800, c=100, d=900)
    assert ror > 2.0
    assert p < 1e-8
    # No association at all must not be significant.
    _, p_null = dp.ror_with_p(a=100, b=900, c=100, d=900)
    assert p_null > 0.9


def test_benjamini_hochberg_is_monotone_and_never_below_p():
    p = np.array([0.001, 0.008, 0.02, 0.2, 0.7])
    q = dp.benjamini_hochberg(p)
    assert np.all(np.diff(q) >= -1e-12), "q must not decrease as p increases"
    assert np.all(q >= p - 1e-12), "BH can only make a p-value less significant"
    assert q[-1] <= 1.0


def test_bh_on_pure_noise_rejects_about_alpha():
    """The property the negative control checks at corpus scale."""
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20_000)
    q = dp.benjamini_hochberg(p)
    assert (q <= 0.05).mean() < 0.01, "uniform p-values must yield almost no rejections"


def test_shrinkage_pulls_small_counts_further_toward_the_null():
    """Why EB05 and not PRR: a=5 with a big ratio is mostly noise."""
    observed = np.array([5.0, 500.0])
    expected = np.array([1.0, 100.0])       # both have a raw ratio of 5
    alpha, beta = 2.0, 1.0
    ebgm, eb05 = dp.eb_shrink(observed, expected, alpha, beta)
    assert ebgm[0] < ebgm[1], "the small-count pair must shrink more"
    assert eb05[0] < eb05[1]
    assert eb05[0] < ebgm[0], "EB05 is a lower bound on the posterior"


def test_ewma_fires_on_a_sustained_step_and_not_on_noise():
    rng = np.random.default_rng(1)
    flat = np.abs(rng.normal(0.05, 0.005, size=60))
    assert changepoint.ewma_alarm(flat, MONTHS, CFG.ewma_lambda,
                                  CFG.ewma_control_limit, CFG.ewma_baseline_months) is None

    stepped = flat.copy()
    stepped[40:] += 0.05          # ten sigma, sustained
    got = changepoint.ewma_alarm(stepped, MONTHS, CFG.ewma_lambda,
                                 CFG.ewma_control_limit, CFG.ewma_baseline_months)
    assert got is not None and got.method == "ewma"
    assert MONTHS[40] <= got.period <= MONTHS[45], got.period


def test_ewma_alarm_cannot_be_moved_by_later_data():
    """The point-in-time guarantee that makes a lead time a lead time.

    Truncating the series just after the alarm must not change when it fired.
    If it did, the detector was reading the future and every Phase 6 lead time
    computed from it would be hindsight.
    """
    rng = np.random.default_rng(2)
    values = np.abs(rng.normal(0.05, 0.005, size=60))
    values[30:] += 0.05
    full = changepoint.ewma_alarm(values, MONTHS, CFG.ewma_lambda,
                                  CFG.ewma_control_limit, CFG.ewma_baseline_months)
    assert full is not None
    cut = MONTHS.index(full.period) + 1
    truncated = changepoint.ewma_alarm(values[:cut], MONTHS[:cut], CFG.ewma_lambda,
                                       CFG.ewma_control_limit, CFG.ewma_baseline_months)
    assert truncated is not None and truncated.period == full.period


def test_ewma_ignores_a_decrease():
    """A cluster shrinking is not a harm signal."""
    rng = np.random.default_rng(3)
    values = np.abs(rng.normal(0.10, 0.005, size=60))
    values[40:] -= 0.05
    assert changepoint.ewma_alarm(values, MONTHS, CFG.ewma_lambda,
                                  CFG.ewma_control_limit, CFG.ewma_baseline_months) is None


def test_pelt_finds_an_upward_regime_change_and_skips_downward():
    values = np.concatenate([np.full(30, 0.02), np.full(30, 0.20)])
    got = changepoint.pelt_alarm(values, MONTHS, CFG.pelt_penalty)
    assert got is not None and got.method == "pelt"
    assert abs((MONTHS.index(got.period)) - 30) <= 6, got.period
    assert got.statistic > 1.0, "a tenfold jump is a relative jump above 1"

    falling = np.concatenate([np.full(30, 0.20), np.full(30, 0.02)])
    assert changepoint.pelt_alarm(falling, MONTHS, CFG.pelt_penalty) is None


def test_pelt_is_quiet_on_a_constant_series():
    assert changepoint.pelt_alarm(np.full(60, 0.05), MONTHS, CFG.pelt_penalty) is None


def test_densify_treats_absent_months_as_real_zeros():
    points = [(MONTHS[2], 3, 100, 0.03), (MONTHS[5], 9, 100, 0.09)]
    got = changepoint.densify(points, MONTHS[:8])
    assert got.tolist() == [0.0, 0.0, 0.03, 0.0, 0.0, 0.09, 0.0, 0.0]


def test_analyse_drops_pairs_below_min_a():
    rows = [
        ("f", "co1", "cl1", 100, 900, 100, 9000),
        ("f", "co2", "cl1", 2, 900, 100, 9000),     # a = 2, below min_a
    ]
    got = dp.analyse(rows, min_a=5)
    assert [g.company_id for g in got] == ["co1"]
    assert 0.0 <= got[0].q_value <= 1.0
