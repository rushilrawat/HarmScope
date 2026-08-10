from __future__ import annotations

import dataclasses

import pytest

from src.config import CONFIG, Config, LLMConfig, NoveltyConfig, Paths, paths


def test_llm_operational_parameters_are_fingerprinted():
    """Retries and pricing must travel with every recorded configuration."""
    payload = CONFIG.to_dict()["llm"]
    assert payload["max_retries"] == 3
    assert payload["retry_base_seconds"] == 1.0
    assert payload["input_usd_per_million"] > 0
    assert payload["output_usd_per_million"] > 0
    assert payload["human_verify_n"] == 50
    assert payload["rag_candidate_k"] == 50
    assert payload["bm25_tokenizer_version"] == "word-v1"


def test_llm_candidate_pool_cannot_be_smaller_than_returned_evidence():
    """A truncation setting cannot silently make the requested top-k impossible."""
    with pytest.raises(ValueError, match="rag_candidate_k"):
        LLMConfig(rag_top_k=11, rag_candidate_k=10)


@pytest.mark.parametrize("field", ["rag_top_k", "rag_candidate_k", "rrf_k"])
@pytest.mark.parametrize("value", [0, -1, False, True, 1.5])
def test_llm_retrieval_integer_settings_must_be_positive_non_boolean(field, value):
    """Invalid retrieval limits cannot reach ranking or fusion at runtime."""
    kwargs = {field: value}
    if field == "rag_top_k" and isinstance(value, int) and not isinstance(value, bool):
        kwargs["rag_candidate_k"] = max(50, value)

    with pytest.raises(ValueError, match=field):
        LLMConfig(**kwargs)


@pytest.mark.parametrize("value", ["", "   ", None, 1])
def test_llm_tokenizer_version_must_be_a_nonblank_string(value):
    """Every sparse cache must be keyed by an explicit tokenizer contract."""
    with pytest.raises(ValueError, match="bm25_tokenizer_version"):
        LLMConfig(bm25_tokenizer_version=value)


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
