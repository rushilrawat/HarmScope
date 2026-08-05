"""Phase 4: does a cluster correspond to something the CFPB taxonomy names?

docs/METHODOLOGY.md §5. Two readings of that section had to be pinned down
before any number here means anything, and both are choices rather than
transcriptions:

**"normalized by log of support" — support of what.** Read as the number of
distinct label tuples available *in that product family*, not the number
observed in the cluster. Normalizing by the observed count is the standard
Shannon evenness measure and is wrong for this purpose: a cluster split evenly
between exactly two labels would score 1.0, maximum novelty, when §5 defines
high novelty as members "scattered across many existing labels". Per family
rather than corpus-wide because clustering is stratified by family, and a
mortgage cluster's entropy measured against credit-reporting's vocabulary is
not comparable to anything.

The cost is that the entropy term is compressed — reaching 1.0 needs uniformity
over every label in the family — so with `w_dominant_share == w_entropy` the
dominant-share term does most of the work. That is visible in the reported
components, and it is the honest consequence of making entropy comparable
across clusters.

**A cluster with no labelled members scores 1.0.** This only arises under
ablation (§5.1), and it is the case the whole test turns on: a cluster whose
every member carried the hidden issue now has nothing left to match against,
which is exactly "no existing category captures this".
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from src.config import NoveltyConfig

NO_LABEL = "∅"


def label_of(issue: str | None, sub_issue: str | None) -> str:
    """The `(issue_std, sub_issue_std)` tuple as one comparable string."""
    return f"{issue or NO_LABEL} | {sub_issue or NO_LABEL}"


@dataclass(frozen=True)
class Novelty:
    dominant_label: str
    dominant_share: float
    entropy: float            # normalized to [0, 1] by log(family label space)
    score: float
    n_labeled: int


def score_cluster(
    labels: list[str],
    family_label_space: int,
    cfg: NoveltyConfig,
    n_members: int | None = None,
) -> Novelty:
    """Novelty for one cluster from its members' label tuples.

    `labels` excludes members whose label was ablated away — that is how §5.1
    hides an issue, and why an empty list is meaningful rather than an error.
    `n_members` is the cluster's true size, which is larger than `len(labels)`
    exactly when labels have been ablated.

    **The denominator is the cluster, not the labelled part of it.** §5 defines
    `dominant_label_share` as the "fraction of members carrying the modal label
    tuple"; dividing by surviving labels instead answers "among the members we
    could match, how concentrated are they", which discards the one thing that
    matters — how much of the cluster no label explains at all.

    That deviation was not academic. On the 2026-08-05 run it gave a 101-member
    bank_account cluster, 99% of it the hidden issue, a novelty of **0.000**:
    one member survived, that member's label was trivially 100% of the
    survivors, and the most completely unexplained cluster in the family scored
    as perfectly explained. It is invisible outside ablation, because every
    complaint in the corpus carries an `issue_std`, so `n_members == len(labels)`
    and this changes no production score.
    """
    n_members = len(labels) if n_members is None else n_members
    if not labels:
        return Novelty(NO_LABEL, 0.0, 1.0, 1.0, 0)

    counts = Counter(labels)
    total = len(labels)
    dominant_label, dominant_n = counts.most_common(1)[0]
    dominant_share = dominant_n / max(n_members, 1)

    entropy = -sum(
        (n / total) * math.log(n / total) for n in counts.values() if n
    )
    ceiling = math.log(max(family_label_space, 2))
    normalized_entropy = min(entropy / ceiling, 1.0)

    score = (
        cfg.w_dominant_share * (1.0 - dominant_share)
        + cfg.w_entropy * normalized_entropy
    )
    return Novelty(dominant_label, dominant_share, normalized_entropy, score, total)


def score_all(
    members: dict[str, list[tuple[str | None, str | None]]],
    family_label_space: int,
    cfg: NoveltyConfig,
    hidden_issue: str | None = None,
) -> dict[str, Novelty]:
    """Score every cluster, optionally with one `issue_std` ablated.

    Ablation drops the *member's label*, not the member: the cluster keeps its
    size and loses its evidence, which is the situation a genuinely new harm
    presents — complaints exist, no category describes them.
    """
    out = {}
    for cluster_id, pairs in members.items():
        labels = [
            label_of(issue, sub)
            for issue, sub in pairs
            if hidden_issue is None or issue != hidden_issue
        ]
        out[cluster_id] = score_cluster(
            labels, family_label_space, cfg, n_members=len(pairs)
        )
    return out


def dominant_issue(
    members: dict[str, list[tuple[str | None, str | None]]]
) -> dict[str, str | None]:
    """Each cluster's modal `issue_std`, unablated — the ablation's answer key."""
    out: dict[str, str | None] = {}
    for cluster_id, pairs in members.items():
        counts = Counter(issue for issue, _ in pairs if issue)
        out[cluster_id] = counts.most_common(1)[0][0] if counts else None
    return out


def pick_hidden_issues(
    members: dict[str, list[tuple[str | None, str | None]]], n: int
) -> list[str]:
    """The `n` issues that dominate the most clusters.

    An ablation of an issue that dominates no cluster has no positives and no
    AUC, so the choice is driven by having something to detect rather than by
    which issues are interesting.
    """
    counts = Counter(v for v in dominant_issue(members).values() if v)
    return [issue for issue, _ in counts.most_common(n)]


def ablation_auc(
    members: dict[str, list[tuple[str | None, str | None]]],
    family_label_space: int,
    cfg: NoveltyConfig,
) -> dict:
    """§5.1 — can the score recover deliberately hidden `Issue` categories?

    Positives are the clusters the hidden issue dominates; negatives are the
    rest, scored under the same ablation (their labels are untouched by it).

    What this does and does not show: stripping a cluster's labels obviously
    raises its novelty, so a high AUC confirms the scoring machinery responds in
    the right direction and with enough separation to rank on. It is a
    necessary condition, not evidence that the score finds *unknown* harms —
    nothing available before Phase 6 can show that.
    """
    from sklearn.metrics import roc_auc_score

    truth = dominant_issue(members)
    hidden = pick_hidden_issues(members, cfg.ablation_n_issues)
    per_issue = []
    pooled_scores: list[float] = []
    pooled_truth: list[int] = []

    for issue in hidden:
        scored = score_all(members, family_label_space, cfg, hidden_issue=issue)
        y = [1 if truth[c] == issue else 0 for c in scored]
        s = [scored[c].score for c in scored]
        if not any(y) or all(y):
            continue
        per_issue.append({
            "issue": issue, "n_positive": sum(y),
            "auc": float(roc_auc_score(y, s)),
        })
        pooled_scores.extend(s)
        pooled_truth.extend(y)

    if not per_issue:
        return {"per_issue": [], "mean_auc": None, "pooled_auc": None,
                "n_issues": 0, "passes": False}
    mean_auc = sum(p["auc"] for p in per_issue) / len(per_issue)
    return {
        "per_issue": per_issue,
        "mean_auc": mean_auc,
        "pooled_auc": float(roc_auc_score(pooled_truth, pooled_scores)),
        "n_issues": len(per_issue),
        "passes": mean_auc >= cfg.ablation_min_auc,
    }


def is_novel(
    novelty: Novelty, coherence: float, persistence: float, cfg: NoveltyConfig
) -> bool:
    """§5.2 — novel *and* coherent. Incoherent-and-novel is a clustering defect."""
    return (
        novelty.score >= cfg.threshold
        and coherence >= cfg.min_coherence
        and persistence >= cfg.min_persistence
    )
