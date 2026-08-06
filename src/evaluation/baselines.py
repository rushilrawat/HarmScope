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

import duckdb

from src.ids import cluster_id as make_cluster_id


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


BUILDERS = {"B0": build_b0, "B1": build_b1}
