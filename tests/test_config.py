from __future__ import annotations

import dataclasses

import pytest

from src.config import CONFIG, Config, NoveltyConfig, Paths, paths


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.seed = 1  # type: ignore[misc]


def test_fingerprint_is_stable_across_instances():
    assert Config().fingerprint == Config().fingerprint == CONFIG.fingerprint


def test_fingerprint_moves_when_a_threshold_moves():
    """Trap T4: threshold tuning on the backtest set must be detectable."""
    tweaked = dataclasses.replace(
        CONFIG, novelty=NoveltyConfig(threshold=CONFIG.novelty.threshold + 0.05)
    )
    assert tweaked.fingerprint != CONFIG.fingerprint


def test_novelty_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        NoveltyConfig(w_dominant_share=0.7, w_entropy=0.7)


def test_config_serialises_to_json_round_trip():
    assert CONFIG.to_dict()["novelty"]["threshold"] == CONFIG.novelty.threshold
    assert CONFIG.to_dict()["eval"]["cutoffs"][0] == "2017-01-01"


def test_ground_truth_window_starts_at_the_first_cutoff():
    """An action filed before the first cutoff has no cutoff strictly preceding
    it and cannot be evaluated. docs/EVALUATION.md §1.2."""
    assert CONFIG.eval.ground_truth_start == CONFIG.eval.cutoffs[0]
    assert CONFIG.eval.ground_truth_end.year == 2024


def test_every_action_can_be_assigned_a_cutoff():
    from datetime import date

    for filed in (date(2017, 6, 1), date(2020, 1, 2), date(2024, 12, 31)):
        prior = [c for c in CONFIG.eval.cutoffs if c < filed]
        assert prior, f"{filed} has no cutoff strictly before it"


def test_paths_can_be_redirected(monkeypatch, tmp_path):
    monkeypatch.setenv("HARMSCOPE_DATA_DIR", str(tmp_path))
    assert paths().data == tmp_path
    monkeypatch.delenv("HARMSCOPE_DATA_DIR")
    assert paths().data == Paths().data


def test_ensure_creates_every_directory(tmp_path):
    p = Paths(data=tmp_path / "d")
    p.ensure()
    for d in (p.raw, p.interim, p.artifacts, p.ground_truth, p.llm_cache):
        assert d.is_dir()
