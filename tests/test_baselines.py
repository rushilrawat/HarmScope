"""Phase 7 baselines — the two properties B3 rests on.

B3's "no dedup" is expressed as data (a `dup_groups` population of singletons)
rather than as a flag through the panel, so the thing worth testing is that the
unchanged panel code actually reads it that way: every complaint carries its own
cluster, and nothing is collapsed. If that silently stopped holding, B3 would
still produce numbers — just HarmScope's deduplicated ones under a B3 label,
which is exactly the failure `--system was cosmetic` already cost this project
once.
"""

from __future__ import annotations

from datetime import date

from src.evaluation.baselines import bertopic_cluster_config, identity_groups
from src.ids import cluster_id
from src.signals.timeseries import EXPANDED_SQL


def _corpus(con, run_id):
    for cid, day in [(1, 5), (2, 6), (3, 7)]:
        con.execute(
            "INSERT INTO complaints (complaint_id, date_received, period_month, "
            "company_id, product_family, has_narrative) "
            "VALUES (?, ?, DATE '2020-01-01', NULL, 'mortgage', true)",
            [cid, date(2020, 1, day)],
        )
    # After the cutoff used below, and identical text to complaint 1 — a real
    # dedup run would merge them; an identity run must not.
    con.execute(
        "INSERT INTO complaints (complaint_id, date_received, period_month, "
        "company_id, product_family, has_narrative) "
        "VALUES (9, DATE '2021-06-01', DATE '2021-06-01', NULL, 'mortgage', true)"
    )


def test_identity_groups_are_singletons_and_respect_the_cutoff(seeded):
    con, run_id, _ = seeded
    _corpus(con, run_id)

    n = identity_groups(con, run_id, date(2020, 12, 31), cutoff=date(2021, 1, 1))

    assert n == 3, "the post-cutoff complaint must not be in scope"
    rows = con.execute(
        "SELECT count(*), count(DISTINCT group_id), sum(is_representative::INT), "
        "max(group_size) FROM dup_groups WHERE run_id = ?", [run_id]
    ).fetchone()
    assert rows == (3, 3, 3, 1), "every complaint is its own group and its own rep"


def test_the_panel_reads_an_identity_run_as_one_cluster_per_complaint(seeded):
    con, run_id, cid = seeded
    _corpus(con, run_id)
    identity_groups(con, run_id, date(2020, 12, 31), cutoff=date(2021, 1, 1))
    other = cluster_id(run_id, "mortgage", 4)
    con.execute(
        "INSERT INTO clusters (cluster_id, run_id, product_family, n_members, as_of) "
        "VALUES (?, ?, 'mortgage', 1, DATE '2020-12-31')", [other, run_id]
    )
    con.executemany(
        "INSERT INTO cluster_members (cluster_id, complaint_id, membership_prob, "
        "is_exemplar) VALUES (?, ?, 1.0, false)",
        [(cid, 1), (cid, 2), (other, 3)],
    )

    con.execute(EXPANDED_SQL, [run_id, run_id, run_id, date(2021, 1, 1), date(2021, 1, 1)])
    got = dict(con.execute(
        "SELECT complaint_id, cluster_id FROM _expanded ORDER BY 1"
    ).fetchall())

    # Under a real dedup run these three would share their representative's
    # cluster. Under the identity run each keeps its own, which is what makes
    # the panel's distinct-group count equal to its complaint count.
    assert got == {1: cid, 2: cid, 3: other}
    assert con.execute(
        "SELECT count(*) = count(DISTINCT group_id) FROM _expanded"
    ).fetchone()[0], "no dedup means one group per complaint"


def test_bertopic_config_changes_the_model_and_not_the_harness():
    from src.config import CONFIG

    cfg = bertopic_cluster_config()
    assert (cfg.umap_n_neighbors, cfg.umap_n_components) == (15, 5)
    assert (cfg.hdbscan_min_cluster_size, cfg.cluster_selection_method) == (10, "eom")
    # BERTopic sets neither of these, so B3 must inherit HarmScope's — otherwise
    # the gap mixes "default hyperparameters" with "different assignment rule".
    assert cfg.assign_max_distance == CONFIG.cluster.assign_max_distance
    assert cfg.fit_sample_size == CONFIG.cluster.fit_sample_size
