"""Single source of every tunable parameter.

Two rules from docs/ARCHITECTURE.md §6 shape this module:

  - No magic numbers in module code. If a number appears in a result, it is
    reachable from here and serialized into `runs.params_json`.
  - The config is frozen and fingerprinted. `Config.fingerprint` is a sha256
    over the whole science-parameter tree, recorded on every run. That turns
    trap T4 ("threshold tuning on the backtest set") from a promise into a
    checkable fact: if the fingerprint recorded at ground-truth freeze differs
    from the one on the backtest run, thresholds moved.

Environment-dependent values (filesystem paths) live in `Paths` and are
deliberately NOT part of the fingerprint — where the data sits is not science.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Environment (not fingerprinted)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT
    data: Path = REPO_ROOT / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def artifacts(self) -> Path:
        return self.data / "artifacts"

    @property
    def ground_truth(self) -> Path:
        return self.data / "ground_truth"

    @property
    def llm_cache(self) -> Path:
        return self.artifacts / "llm_cache"

    @property
    def db(self) -> Path:
        return self.data / "harmscope.duckdb"

    @property
    def schema_sql(self) -> Path:
        return self.root / "db" / "schema.sql"

    def ensure(self) -> None:
        """Create every directory this project writes to. Idempotent."""
        for p in (self.data, self.raw, self.interim, self.artifacts,
                  self.ground_truth, self.llm_cache):
            p.mkdir(parents=True, exist_ok=True)


def paths() -> Paths:
    """Paths, with the data directory overridable via HARMSCOPE_DATA_DIR."""
    override = os.environ.get("HARMSCOPE_DATA_DIR")
    if override:
        return Paths(data=Path(override).expanduser().resolve())
    return Paths()


# --------------------------------------------------------------------------
# Science parameters (fingerprinted)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    """docs/PROJECT_SPEC.md §4."""

    window_start: date = date(2015, 1, 1)
    # CFPB bulk CSV. Verified reachable before Phase 1 (docs/PROJECT_SPEC.md §5.4).
    bulk_csv_url: str = (
        "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
    )


@dataclass(frozen=True)
class DedupConfig:
    """docs/METHODOLOGY.md §2."""

    minhash_perms: int = 128
    shingle_size: int = 5               # character 5-shingles beat word shingles here
    # Tuned on the 300-pair labelled set (METHODOLOGY §2.2 sanctions this),
    # 2026-08-04. At the spec's 0.85 precision was 0.905 — the gate needs 0.95.
    # The cause was NOT a loose threshold but estimator noise: 128 permutations
    # give a standard error of ~0.088, so pairs with true Jaccard just under
    # 0.85 clear the bar about half the time. All 19 false positives were
    # direct edges with upward drift; zero came from transitive closure.
    #
    #   0.85 -> P 0.9050  R 1.0000   fail
    #   0.88 -> P 0.9632  R 0.8674   PASS  <- best recall among passing
    #   0.90 -> P 0.9921  R 0.6961
    #
    # Trade-off, stated plainly: at 0.88 roughly 13% of true duplicates are
    # missed. Those become separate groups, which INFLATES distinct-group
    # counts and makes a signal look stronger than it is (METHODOLOGY §2.3
    # weights 400 complaints in 380 groups as strong). That is the
    # anti-conservative direction, and it is accepted only because §2.4 is
    # explicit that a false merge destroys real signal outright.
    jaccard_threshold: float = 0.88
    # Campaign detection (Tier 3). Hand-tuned against the labelled pair set;
    # deliberately not a supervised model — too few labels, features are readable.
    campaign_min_size: int = 20
    # A campaign is flagged when at least this many of the six signals fire.
    # Not a trained model (docs/METHODOLOGY.md §2.2): too few labels, and a
    # human has to be able to audit why a flag fired.
    campaign_min_signals: int = 3
    burstiness_threshold: float = 3.0            # Fano factor of daily counts
    # Concentration signals are RELATIVE to the product family's own baseline,
    # not absolute. Measured on the 2026-08-03 snapshot, absolute thresholds
    # were degenerate: `submitted_via > 0.95` fired on 100% of candidates
    # (almost everything arrives via Web, so HHI is ~1.0 for any group) and
    # `company > 0.50` fired on 0% (credit reporting splits across three
    # bureaus, HHI ~0.33). A signal that always fires is worse than useless —
    # it was silently adding +1 to every group's n_signals, turning
    # campaign_min_signals=3 into an effective 2.
    #
    # "Organic harms spread, campaigns concentrate" (METHODOLOGY §2.2) is a
    # claim about concentrating MORE than the surrounding family does.
    concentration_ratio: float = 1.5   # group HHI / family baseline HHI
    boilerplate_threshold: float = 0.50   # bimodal in practice; any 0.1-0.9 works
    length_cv_threshold: float = 0.20     # LOW variance is the signal
    # Gate: docs/METHODOLOGY.md §2.4. False merges destroy real signal, so
    # precision is the binding constraint, not recall.
    min_precision: float = 0.95


@dataclass(frozen=True)
class EmbedConfig:
    """docs/METHODOLOGY.md §3."""

    model: str = "BAAI/bge-base-en-v1.5"
    dev_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 768
    batch_size: int = 256
    checkpoint_every: int = 50_000
    max_tokens: int = 512
    # Long narratives: embed first + last window and mean-pool. Complaint
    # narratives often state the core problem at both ends.
    dual_window: bool = True


@dataclass(frozen=True)
class ClusterConfig:
    """docs/METHODOLOGY.md §4."""

    umap_n_neighbors: int = 30
    umap_n_components: int = 10          # preprocessing for density clustering, not viz
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    hdbscan_min_cluster_size: int = 50
    hdbscan_min_samples: int = 10
    cluster_selection_method: str = "leaf"   # finer clusters => mechanisms, not topics
    fit_sample_size: int = 500_000
    stability_sample_sizes: tuple[int, ...] = (100_000, 250_000, 500_000)
    assign_max_distance: float = 0.35    # beyond this, a point becomes noise
    # METHODOLOGY §4.2: clusters in different families this close are the same
    # harm under two products. Higher than assign_max_distance implies, because
    # two centroids agreeing is a stronger claim than a point sitting near one.
    related_min_similarity: float = 0.80


@dataclass(frozen=True)
class NoveltyConfig:
    """docs/METHODOLOGY.md §5.

    novelty = w_dominant_share * (1 - dominant_label_share)
            + w_entropy       * normalized_entropy

    Weights are tuned on the label-ablation test (§5.1), never on the backtest
    set. The ablation test is a separate, legitimate tuning surface; recording
    that here is the T4 countermeasure.
    """

    w_dominant_share: float = 0.5
    w_entropy: float = 0.5
    threshold: float = 0.60
    # Guard against the trivial failure: incoherent-and-novel is a clustering
    # defect, not a finding. Both floors must be cleared.
    min_coherence: float = 0.45          # mean intra-cluster cosine similarity
    min_persistence: float = 0.05        # HDBSCAN cluster persistence
    ablation_n_issues: int = 10
    ablation_min_auc: float = 0.70       # Phase 4 gate

    def __post_init__(self) -> None:
        total = self.w_dominant_share + self.w_entropy
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"novelty weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class SignalConfig:
    """docs/METHODOLOGY.md §6."""

    min_a: int = 5                       # raw PRR is unusable below this
    fdr_alpha: float = 0.05              # Benjamini-Hochberg, within product family
    ewma_lambda: float = 0.2
    ewma_control_limit: float = 3.0      # L, in sigma
    ewma_baseline_months: int = 12
    pelt_penalty: float = 10.0
    min_supporting_groups: int = 15      # distinct dup-groups, not raw complaints
    rank_by: str = "eb05"                # never the point estimate


@dataclass(frozen=True)
class LLMConfig:
    """docs/LLM_LAYER.md §4.

    The layer is descriptive only. Deleting src/llm/ must leave `signals`
    byte-identical (docs/LLM_LAYER.md §1).
    """

    # docs/LLM_LAYER.md §4 specified claude-sonnet-5 and pre-registered
    # claude-opus-5 as the first lever if §2.5 verification came back weak.
    # Pulled forward on 2026-08-07 at the operator's instruction, before any
    # labelling run — free to change now because the model string is part of the
    # cache key, so switching after a run would have invalidated every label.
    model: str = "claude-opus-5"
    prompt_version: str = "v1"           # bumped on any prompt edit; part of the cache key
    label_sample_k: int = 20
    label_medoid_k: int = 12             # nearest-medoid share of label_sample_k
    min_cluster_size_for_label: int = 30
    max_narrative_chars: int = 1_200
    rag_top_k: int = 10
    rrf_k: int = 60
    human_verify_n: int = 50             # docs/LLM_LAYER.md §2.5, required
    max_retries: int = 3
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 30.0
    # Re-check official list pricing before a paid run. A change here produces
    # a new config fingerprint, preserving cost-accounting provenance.
    input_usd_per_million: float = 5.0
    output_usd_per_million: float = 25.0
    cache_write_usd_per_million: float = 6.25
    cache_read_usd_per_million: float = 0.50
    verification_seed: int = 20260809


@dataclass(frozen=True)
class EvalConfig:
    """docs/EVALUATION.md.

    Rolling annual cutoffs rather than per-action cutoffs: 8 refits instead of
    30+, and conservative — lead time is measured from a model up to 12 months
    staler than it needed to be. Understating our own lead time errs in the
    right direction.

    The ground-truth window starts at the first cutoff. An action filed before
    2017-01-01 has no cutoff strictly preceding it and cannot be evaluated;
    such rows stay in the CSV as usable=false so the exclusion is visible.
    """

    cutoffs: tuple[date, ...] = (
        date(2017, 1, 1), date(2018, 1, 1), date(2019, 1, 1), date(2020, 1, 1),
        date(2021, 1, 1), date(2022, 1, 1), date(2023, 1, 1), date(2024, 1, 1),
    )
    # Enforcement actions are only usable as ground truth through 2024: CFPB's
    # 2025 posture change means absence of an action no longer implies absence
    # of harm. docs/PROJECT_SPEC.md §5.
    ground_truth_end: date = date(2024, 12, 31)
    min_usable_actions: int = 20
    adjudication_top_k: int = 20
    adjudication_decoys: int = 10
    systems: tuple[str, ...] = ("harmscope", "B0", "B1", "B2", "B3")

    @property
    def ground_truth_start(self) -> date:
        return self.cutoffs[0]


@dataclass(frozen=True)
class Expectations:
    """Row-count and rate bounds asserted by every stage (trap T1).

    A stage that "completed" without producing valid output is a defect, not a
    pass. These are the bounds it is checked against. Order-of-magnitude values
    from docs/DATA.md §5; replace with measured values after Phase 1 and record
    the change in docs/ENGINEERING_NOTES.md.
    """

    complaints_raw_min: int = 1_000_000
    complaints_raw_max: int = 50_000_000
    # Rows in the CSV that do not reach complaints_raw. Only unkeyed rows (no
    # Complaint ID) are droppable, and they were 0.035% of the 2026-08-03
    # snapshot. A jump here means the export shape changed, not that the data
    # got worse — investigate before raising it.
    max_dropped_fraction: float = 0.001
    narrative_fraction_min: float = 0.05
    narrative_fraction_max: float = 0.95
    # Mean redactions per narrative. Measured 0.00781 on the 2026-08-03
    # snapshot — low because CFPB already masks aggressively, and this module
    # is the secondary sweep. The original 0.0 floor made the check vacuous: it
    # would have passed with redaction switched off entirely. docs/DATA.md §6
    # item 3 wants rate drift to be *detectable*, so the floor has to bite.
    redaction_rate_min: float = 0.002
    redaction_rate_max: float = 0.10
    # Fraction of narratives with at least one redaction. Measured 0.00296.
    redacted_doc_fraction_min: float = 0.001
    redacted_doc_fraction_max: float = 0.05


@dataclass(frozen=True)
class Config:
    seed: int = 20260803
    data: DataConfig = field(default_factory=DataConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    expect: Expectations = field(default_factory=Expectations)

    def to_dict(self) -> dict:
        return json.loads(self.to_json())

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)

    @property
    def fingerprint(self) -> str:
        """sha256 over every science parameter. Recorded on every run."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()


CONFIG = Config()
PATHS = paths()
