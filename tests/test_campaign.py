"""Campaign scoring.

The flag decides what gets excluded from signal detection, so the failure that
matters is a signal that fires on something other than what it claims to
measure. Both concentration signals did exactly that.
"""

from __future__ import annotations

import pytest

from src.config import CONFIG
from src.dedup import campaign

CFG = CONFIG.dedup
STATE_BASELINE = 0.0674  # credit_reporting, measured 2026-08-04


def _group(**over) -> dict:
    """A candidate that fires nothing, so each test turns on one signal."""
    return {
        "product_family": "credit_reporting", "n_complaints": 100,
        "burstiness": 0.0, "length_cv": 1.0, "boilerplate_score": 0.0,
        "state_concentration": 0.0, "company_concentration": 0.0,
    } | over


def test_expected_hhi_is_the_null_not_the_family_average():
    """A group of n cannot have an HHI below 1/n, whatever it does."""
    # 20 members drawn at random from a 0.0674-concentrated family still show
    # 0.114 on average -- ABOVE the 1.5x0.0674 = 0.101 bar the old rule used.
    assert campaign.expected_hhi(STATE_BASELINE, 20) == pytest.approx(0.1141, abs=1e-4)
    assert campaign.expected_hhi(STATE_BASELINE, 20) > STATE_BASELINE * CFG.concentration_ratio
    # The bias vanishes as the group grows, which is why the old rule was
    # weakest on the large groups that are the real campaigns.
    assert campaign.expected_hhi(STATE_BASELINE, 25_000) == pytest.approx(
        STATE_BASELINE, abs=1e-4)


def test_small_group_at_chance_concentration_does_not_fire():
    """The regression: 84.7% of unflagged candidates fired this signal."""
    base = {"state_concentration:credit_reporting": STATE_BASELINE}
    at_chance = campaign.expected_hhi(STATE_BASELINE, 20)
    assert campaign.count_signals(
        _group(n_complaints=20, state_concentration=at_chance), CFG, base) == 0
    # Genuinely concentrated for its size still fires.
    assert campaign.count_signals(
        _group(n_complaints=20, state_concentration=at_chance * 2), CFG, base) == 1


def test_large_group_is_not_penalised_for_being_large():
    base = {"state_concentration:credit_reporting": STATE_BASELINE}
    concentrated = STATE_BASELINE * CFG.concentration_ratio * 1.1
    assert campaign.count_signals(
        _group(n_complaints=25_000, state_concentration=concentrated), CFG, base) == 1


def test_there_are_five_signals_not_six():
    """submitted_via is constant over narratives; it was dropped, not tuned."""
    all_on = _group(
        burstiness=CFG.burstiness_threshold + 1,
        length_cv=CFG.length_cv_threshold / 2,
        boilerplate_score=CFG.boilerplate_threshold + 0.1,
        state_concentration=1.0, company_concentration=1.0,
    )
    base = {
        "state_concentration:credit_reporting": STATE_BASELINE,
        "company_concentration:credit_reporting": 0.2674,
    }
    assert campaign.count_signals(all_on, CFG, base) == 5
    # A stale submitted_via value must not resurrect a sixth signal.
    assert campaign.count_signals(
        all_on | {"submitted_via_concentration": 1.0}, CFG, base) == 5


def test_size_gate_precedes_the_signal_count():
    base = {"state_concentration:credit_reporting": STATE_BASELINE}
    tiny = _group(
        n_complaints=CFG.campaign_min_size - 1,
        burstiness=CFG.burstiness_threshold + 1,
        length_cv=CFG.length_cv_threshold / 2,
        boilerplate_score=CFG.boilerplate_threshold + 0.1,
    )
    assert campaign.count_signals(tiny, CFG, base) >= CFG.campaign_min_signals
    assert campaign.is_flagged(tiny, CFG, base) is False


def test_missing_baseline_never_fires_a_concentration_signal():
    """An unknown family must not flag by accident."""
    assert campaign.count_signals(
        _group(product_family="unheard_of", state_concentration=1.0), CFG, {}) == 0
