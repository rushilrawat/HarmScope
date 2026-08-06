"""Phase 7: the baselines the contribution is measured against.

docs/EVALUATION.md §2. Four systems, and B1 is the one that matters — if the
existing CFPB taxonomy gives the same lead time, this project has no
contribution and the README has to say so.

**B1 does not reimplement anything.** It materialises each
`(product_family, issue_std, sub_issue_std)` tuple as a row in `clusters` and
its complaints as `cluster_members`, under its own `run_id`. Every downstream
stage — the panel, the 2x2 margins, the empirical-Bayes shrinkage, BH within
family, EWMA, PELT, the alert criteria, the backtest harness — then runs over it
unchanged, because none of them knows or cares how a "cluster" was defined.

That is a stronger fairness guarantee than writing a parallel B1 pipeline could
ever be. §2 requires the baseline to use "the *same* statistical machinery — the
only difference is the unit being tracked", and a parallel implementation makes
that a claim to be trusted. Here it is the same code path, so the comparison
cannot be tilted by an implementation detail nobody noticed.

The taxonomy is a partition, so B1's "clusters" have properties HarmScope's do
not: every complaint belongs to exactly one, coherence is undefined, and noise
is empty. Coherence is set to 1.0 — a taxonomy label is perfectly coherent with
itself by definition — so the §6.3 alert criteria neither favour nor penalise it
on a dimension it cannot have.
"""

from __future__ import annotations

import dataclasses

import duckdb

from src.config import CONFIG
from src.ids import cluster_id as make_cluster_id

# BERTopic 0.17's constructor defaults, read from `bertopic/_bertopic.py` rather
# than from memory: UMAP(15, 5, min_dist=0.0, cosine) and HDBSCAN(
# min_cluster_size=min_topic_size=10, euclidean, eom, prediction_data=True).
# BERTopic does not pass `min_samples`, and hdbscan then defaults it to
# `min_cluster_size`, so 10 here is that default written out rather than a
# choice. Its default embedding model is `all-MiniLM-L6-v2` — the same model
# every system in this comparison runs on, which is why B3 is a fair off-the-
# shelf reference rather than an apples-to-oranges one.
BERTOPIC_DEFAULTS = {
    "umap_n_neighbors": 15,
    "umap_n_components": 5,
    "umap_min_dist": 0.0,
    "umap_metric": "cosine",
    "hdbscan_min_cluster_size": 10,
    "hdbscan_min_samples": 10,
    "cluster_selection_method": "eom",
}


def bertopic_cluster_config():
    """HarmScope's cluster config with BERTopic's defaults substituted in.

    Everything BERTopic does not set — `fit_sample_size`, `assign_max_distance`
    — stays at HarmScope's value on purpose. Those are harness parameters, not
    model parameters: changing them would make the B3 gap a mixture of "default
    hyperparameters" and "different assignment rule", and only the first is what
    §2 asks B3 to measure.
    """
    return dataclasses.replace(CONFIG.cluster, **BERTOPIC_DEFAULTS)


def identity_groups(
    con: duckdb.DuckDBPyConnection, run_id: str, as_of, cutoff=None
) -> int:
    """Every narrative-bearing complaint as its own dup group — B3's "no dedup".

    EVALUATION §2 requires B3 to run without dedup, and the honest way to
    express that is in the data rather than with a flag threaded through the
    panel. A `dup_groups` population of singletons makes the existing code path
    mean exactly "no dedup" with no code change at all:

      * `_expanded`'s self-join to the group representative resolves to the
        complaint itself, so each complaint carries its own cluster;
      * the panel's distinct-group count equals its complaint count, so
        `n_supporting_groups == n_supporting` and nothing is collapsed;
      * no `campaigns` rows exist under this run_id, so the campaign exclusion
        drops nothing.

    Registered under phase `dedup_identity`, never `dedup`. `latest_run` and
    `backtest.run_for_cutoff` both resolve dedup runs by recency within a phase,
    so a run of phase `dedup` at the same cutoff would be newer than the real
    refit and would silently shadow it for HarmScope, B0 and B1.
    """
    con.execute("DELETE FROM dup_groups WHERE run_id = ?", [run_id])
    con.execute(
        """
        INSERT INTO dup_groups (run_id, complaint_id, group_id, is_representative,
                                group_size, as_of)
        SELECT ?, complaint_id, 'g' || complaint_id, true, 1, ?
        FROM complaints
        WHERE has_narrative AND (? IS NULL OR date_received < ?)
        """,
        [run_id, as_of, cutoff, cutoff],
    )
    return con.execute(
        "SELECT count(*) FROM dup_groups WHERE run_id = ?", [run_id]
    ).fetchone()[0]


def build_b1(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    dedup_run: str,
    as_of,
    cutoff=None,
) -> tuple[int, int]:
    """Write the taxonomy as `clusters` + `cluster_members`.

    Uses the same dup-group representatives HarmScope clusters, so the two
    systems see an identical population — the only difference is how that
    population is partitioned. Without this, B1 would be counting complaints
    while HarmScope counts groups, and the comparison would measure dedup rather
    than discovery.
    """
    con.execute("DELETE FROM cluster_members WHERE cluster_id LIKE ?", [f"{run_id}:%"])
    con.execute("DELETE FROM clusters WHERE run_id = ?", [run_id])

    rows = con.execute(
        """
        SELECT c.product_family,
               coalesce(c.issue_std, '<none>')     AS issue,
               coalesce(c.sub_issue_std, '<none>') AS sub_issue,
               count(*)                            AS n_members,
               list(d.complaint_id)                AS members
        FROM dup_groups d
        JOIN complaints c USING (complaint_id)
        WHERE d.run_id = ? AND d.is_representative
          AND (? IS NULL OR c.date_received < ?)
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """,
        [dedup_run, cutoff, cutoff],
    ).fetchall()

    cluster_rows, member_rows = [], []
    for local, (family, _issue, _sub_issue, n_members, members) in enumerate(rows):
        cid = make_cluster_id(run_id, family, local)
        cluster_rows.append((
            cid, run_id, family, n_members,
            # persistence and coherence are HDBSCAN notions. A taxonomy label is
            # a definition, not a discovered density, so it is perfectly
            # coherent with itself and maximally persistent. Setting them to 1.0
            # keeps the §6.3 criteria from filtering B1 on a property it cannot
            # have, which would flatter HarmScope.
            1.0, 1.0, None, as_of,
        ))
        member_rows.extend((cid, int(m), 1.0, False) for m in members)

    con.executemany(
        "INSERT INTO clusters (cluster_id, run_id, product_family, n_members, "
        "persistence, coherence, centroid_idx, as_of) VALUES (?,?,?,?,?,?,?,?)",
        cluster_rows,
    )
    con.executemany(
        "INSERT INTO cluster_members (cluster_id, complaint_id, membership_prob, "
        "is_exemplar) VALUES (?, ?, ?, ?)",
        member_rows,
    )
    return len(cluster_rows), len(member_rows)


def build_b0(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    dedup_run: str,
    as_of,
    cutoff=None,
) -> tuple[int, int]:
    """B0 — volume only: one "cluster" per product family.

    The dumbest thing that could work. With a single unit per family, the
    disproportionality test reduces to "does this company file more than its
    share of this family", and the changepoint runs on total family volume. If
    B0 matches HarmScope, nothing after Phase 1 earned its keep.
    """
    con.execute("DELETE FROM cluster_members WHERE cluster_id LIKE ?", [f"{run_id}:%"])
    con.execute("DELETE FROM clusters WHERE run_id = ?", [run_id])

    rows = con.execute(
        """
        SELECT c.product_family, count(*), list(d.complaint_id)
        FROM dup_groups d JOIN complaints c USING (complaint_id)
        WHERE d.run_id = ? AND d.is_representative
          AND (? IS NULL OR c.date_received < ?)
        GROUP BY 1 ORDER BY 1
        """,
        [dedup_run, cutoff, cutoff],
    ).fetchall()

    cluster_rows, member_rows = [], []
    for local, (family, n_members, members) in enumerate(rows):
        cid = make_cluster_id(run_id, family, local)
        cluster_rows.append((cid, run_id, family, n_members, 1.0, 1.0, None, as_of))
        member_rows.extend((cid, int(m), 1.0, False) for m in members)

    con.executemany(
        "INSERT INTO clusters (cluster_id, run_id, product_family, n_members, "
        "persistence, coherence, centroid_idx, as_of) VALUES (?,?,?,?,?,?,?,?)",
        cluster_rows,
    )
    con.executemany(
        "INSERT INTO cluster_members (cluster_id, complaint_id, membership_prob, "
        "is_exemplar) VALUES (?, ?, ?, ?)",
        member_rows,
    )
    return len(cluster_rows), len(member_rows)


def harmscope_k(con: duckdb.DuckDBPyConnection, cutoff) -> dict[str, int]:
    """`{family: n_clusters}` from HarmScope's own refit at this cutoff.

    B2's topic count is the one free parameter that decides its granularity, and
    ENGINEERING_NOTES has already measured what granularity does to this
    comparison: an absolute `min_supporting_groups` floor is a filter on unit
    size, so a system with three times as many units clears it less often
    regardless of quality. Choosing `n_topics` by hand would therefore be
    choosing B2's detection rate.

    Matching HarmScope's discovered count per family removes that degree of
    freedom: whatever the floor does to HarmScope it does to B2. Fixed here,
    before any B2 detection rate exists, for the same reason the support floor
    is not being moved.
    """
    row = con.execute(
        """
        SELECT run_id FROM runs
        WHERE phase = 'cluster' AND status = 'ok'
          AND json_extract_string(params_json, '$.params.cutoff') = ?
          AND coalesce(json_extract_string(params_json, '$.params.system'),
                       'harmscope') = 'harmscope'
        ORDER BY started_at DESC LIMIT 1
        """,
        [str(cutoff)],
    ).fetchone()
    if row is None:
        raise SystemExit(
            f"no HarmScope cluster run at {cutoff} — B2's topic count is defined "
            "as HarmScope's cluster count per family, so that refit must exist first"
        )
    return dict(
        con.execute(
            "SELECT product_family, count(*) FROM clusters WHERE run_id = ? GROUP BY 1",
            [row[0]],
        ).fetchall()
    )


def build_b2(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    dedup_run: str,
    as_of,
    cutoff=None,
    log=print,
) -> tuple[int, int]:
    """B2 — TF-IDF + LDA over the same representatives, one fit per family.

    Tests whether the embeddings buy anything: same population, same downstream
    machinery, a classic bag-of-words topic model in place of
    embedding + UMAP + HDBSCAN.

    Three properties are deliberate.

    **Per family, not one global fit.** The panel restricts a cluster's
    numerator to members of the cluster's own family, so `product_family` is
    required on every unit. A global topic model would have to be given a modal
    family per topic, which silently drops its cross-family members. Family
    scoping is a harness requirement here, not an advantage handed to HarmScope.

    **TF-IDF, not counts, and `norm=None`.** LDA is a generative model over
    counts and is conventionally fit on them; EVALUATION §2 specifies
    "TF-IDF + LDA", so that is what runs. But `TfidfVectorizer` l2-normalizes by
    default, which leaves each document carrying about one unit of pseudo-count
    mass — and LDA fit on that has essentially no data. Measured at the 2017
    cutoff, credit_reporting, k=195: **5 of 195 topics ever won an argmax and one
    of them held 99.0% of the documents**, unchanged at max_iter 5, 20 and 50 and
    unchanged with `use_idf=False`, because normalization rather than weighting
    was the cause. With `norm=None` (213 mass per document) all 195 topics
    populate and the largest holds 5.4%. Raw counts sit in between: 92 of 195,
    largest 23.7%. So `norm=None` is both the spec's weighting and the variant
    that actually gives B2 the granularity `harmscope_k` intends.

    **Five online passes, not convergence.** Fit cost is linear in documents x
    topics, and `harmscope_k` puts k at 741 for credit_reporting at the 2024
    cutoff, so a converged batch fit over eight cutoffs is not affordable here.
    Fixed at 5 on the granularity diagnostic below — where 5, 20 and 50 passes
    were indistinguishable — and never on a detection rate. A better-converged
    B2 could only score higher, so this is a bound in HarmScope's favour and is
    reported as one.

    **No noise class.** LDA assigns every document a distribution over topics
    and argmax always returns one, whereas HDBSCAN + the assignment threshold
    leave ~30% of representatives unassigned. B2 therefore covers more of the
    population than HarmScope does, which if anything favours B2.
    """
    import numpy as np
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import TfidfVectorizer

    con.execute("DELETE FROM cluster_members WHERE cluster_id LIKE ?", [f"{run_id}:%"])
    con.execute("DELETE FROM clusters WHERE run_id = ?", [run_id])

    k_by_family = harmscope_k(con, cutoff)
    families = con.execute(
        """
        SELECT c.product_family, count(*) FROM dup_groups d
        JOIN complaints c USING (complaint_id)
        WHERE d.run_id = ? AND d.is_representative
          AND (? IS NULL OR c.date_received < ?)
        GROUP BY 1 ORDER BY 2 DESC
        """,
        [dedup_run, cutoff, cutoff],
    ).fetchall()

    n_clusters = n_members = 0
    for family, n_reps in families:
        k = k_by_family.get(family)
        if not k:
            # HarmScope found no clusters here (`other`, 290 representatives),
            # so there is no granularity to match and B2 gets no units either.
            log(f"  {family:<18} {n_reps:>7,} reps — HarmScope found no clusters, skipped")
            continue

        batch = con.execute(
            """
            SELECT d.complaint_id, n.text_redacted FROM dup_groups d
            JOIN complaints c USING (complaint_id)
            JOIN narratives n USING (complaint_id)
            WHERE d.run_id = ? AND d.is_representative AND c.product_family = ?
              AND (? IS NULL OR c.date_received < ?)
            ORDER BY d.complaint_id
            """,
            [dedup_run, family, cutoff, cutoff],
        ).fetchnumpy()
        ids, texts = batch["complaint_id"], batch["text_redacted"]
        if len(texts) < k:
            log(f"  {family:<18} {len(texts):>7,} reps — fewer than {k} topics, skipped")
            continue

        vec = TfidfVectorizer(max_features=50_000, min_df=5, max_df=0.5,
                              stop_words="english", norm=None)
        rng = np.random.default_rng(CONFIG.seed)
        size = min(CONFIG.cluster.fit_sample_size, len(texts))
        fit_idx = np.sort(rng.choice(len(texts), size=size, replace=False))
        X_fit = vec.fit_transform(texts[fit_idx])
        lda = LatentDirichletAllocation(
            n_components=k, max_iter=5, learning_method="online",
            batch_size=4096, random_state=CONFIG.seed, n_jobs=-1,
        ).fit(X_fit)

        # Chunked: one k-column dense array over every representative in
        # credit_reporting is several GB at the counts this runs at.
        topic = np.empty(len(texts), dtype=np.int32)
        for start in range(0, len(texts), 100_000):
            sl = slice(start, start + 100_000)
            topic[sl] = lda.transform(vec.transform(texts[sl])).argmax(axis=1)

        cluster_rows, member_rows = [], []
        for t in range(k):
            member = topic == t
            size_t = int(member.sum())
            if size_t == 0:
                continue
            cid = make_cluster_id(run_id, family, t)
            # Same reasoning as B1: coherence and persistence are HDBSCAN
            # notions a topic model has no analogue for, so they are set to 1.0
            # rather than left to filter B2 on a property it cannot have.
            cluster_rows.append((cid, run_id, family, size_t, 1.0, 1.0, None, as_of))
            member_rows.extend(
                (cid, int(i), 1.0, False) for i in ids[member]
            )
        con.executemany(
            "INSERT INTO clusters (cluster_id, run_id, product_family, n_members, "
            "persistence, coherence, centroid_idx, as_of) VALUES (?,?,?,?,?,?,?,?)",
            cluster_rows,
        )
        con.executemany(
            "INSERT INTO cluster_members (cluster_id, complaint_id, membership_prob, "
            "is_exemplar) VALUES (?, ?, ?, ?)",
            member_rows,
        )
        n_clusters += len(cluster_rows)
        n_members += len(member_rows)
        log(f"  {family:<18} {len(texts):>7,} reps  {len(cluster_rows):>5} topics "
            f"of {k} non-empty  vocab {len(vec.vocabulary_):,}")

    return n_clusters, n_members


BUILDERS = {"B0": build_b0, "B1": build_b1, "B2": build_b2}
