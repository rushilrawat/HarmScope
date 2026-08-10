"""Phase 5: the cluster x company x month panel everything else reads from.

docs/METHODOLOGY.md §6.2. Three decisions are made here rather than downstream,
because every statistic inherits them and none of them is recoverable later.

**The unit is a dup-group, not a complaint.** ROADMAP Phase 5 requires it and
Phase 2 is the reason: the campaign flag misses large templates that cite no
statute — a 24,507-member group scored 2 of 5 signals and went unflagged
(`ENGINEERING_NOTES.md` Phase 2). Counting complaints would let that one
template contribute 24,507 to a growth curve. Counting groups, it contributes
one per month it is active, and the flag's miss stops mattering. `n_supporting`
still carries the raw complaint count, so a signal's two numbers can be compared
— 400 complaints in 3 groups is visibly weak.

A group is counted in **every month it has a complaint**, not once at its
representative's date. A template filed steadily across a year is active across
that year, and collapsing it to its first month would move volume backwards in
time — into exactly the pre-enforcement window Phase 6 measures lead time in.

**The denominator is narrative-bearing complaints, not all complaints.** Only
22.7% of the corpus has a narrative (`DATA.md §5`) and only narratives can be
clustered, so dividing by all complaints would make every share a function of
narrative-consent rates rather than of harm. The exposure is the population the
numerator could have come from.

**A cluster's exposure is its own family's, on both sides.** A dup-group's
complaints can be filed under different products — the same template appears
under credit reporting and debt collection, which Phase 3's neighbour read found
directly — and a cluster is assigned from its group's *representative*. So 671
of 2,821 clusters (23.8%) contain complaints from more than one family. Grouping
the panel by cluster while joining exposure on the complaint's family matched
several denominator rows for those, and `any_value` picked one arbitrarily:
`share` had a randomly chosen denominator, and two runs on identical input
disagreed on 1,114 panel rows and 81 changepoint signals. The numerator is now
restricted to members in the cluster's own family, so numerator is a subset of
denominator by construction, `share` stays in [0, 1], and the result is
deterministic. Cross-family membership is not lost — it is what
`related_clusters` (METHODOLOGY §4.2) exists to represent.

**Campaign-flagged complaints are excluded** (§6.3), from numerator and
denominator alike. Dropping them from one side only would inflate or deflate
every share in a family by the campaign rate of that family, which runs from
0.01% to 30%.
"""

from __future__ import annotations

import duckdb

from src.population import EXPANDED_SQL


def build_expanded(
    con: duckdb.DuckDBPyConnection, cluster_run: str, dedup_run: str, cutoff=None
) -> int:
    """Materialize the complaint -> (group, cluster) mapping for one run pair.

    `cutoff` excludes complaints received on or after it. Filtering here rather
    than on the output is the point of EVALUATION §1.1.1: a post-cutoff
    complaint must not reach the panel, the 2x2 margins, or the campaign flag.
    """
    con.execute(EXPANDED_SQL, [cluster_run, dedup_run, dedup_run, cutoff, cutoff])
    return con.execute("SELECT count(*) FROM _expanded").fetchone()[0]


def build_panel(
    con: duckdb.DuckDBPyConnection, run_id: str, cluster_run: str, as_of,
    min_groups: int = 1,
) -> tuple[int, int]:
    """Write `cluster_timeseries` — cluster-level and per-company series.

    Cluster-level rows use the `'__ALL__'` sentinel rather than NULL, because
    `signals` does too and a NULL on one side of that join would silently drop
    exactly the cluster-level rows (schema note, and a Phase 0 regression test).

    Only months where a cluster is actually present get a row. Zero-months are
    not stored: the series are read back into a dense, zero-filled array by
    `changepoint.py`, and storing 2,821 x 140 x every company would be tens of
    millions of rows that are almost all zero.
    """
    # Scoped to the SIGNALS run. Keyed on cluster_id alone, the negative control
    # — same clusters, permuted labels — overwrote the real panel in place and
    # left the table holding shuffled data with nothing raising. Migration 006.
    con.execute("DELETE FROM cluster_timeseries WHERE run_id = ?", [run_id])

    # Exposure: distinct groups in the same family/month, and the same
    # family/company/month, over the identical population as the numerator.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _denom_family AS
        SELECT product_family, period_month, count(DISTINCT group_id) AS denom
        FROM _expanded GROUP BY 1, 2
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _denom_company AS
        SELECT product_family, company_id, period_month,
               count(DISTINCT group_id) AS denom
        FROM _expanded GROUP BY 1, 2, 3
    """)

    con.execute("""
        INSERT INTO cluster_timeseries
          (run_id, cluster_id, company_id, period_month, n, denom, share, as_of)
        SELECT ?, e.cluster_id, '__ALL__', e.period_month,
               count(DISTINCT e.group_id) AS n, any_value(d.denom) AS denom,
               count(DISTINCT e.group_id) / any_value(d.denom)::DOUBLE, ?
        FROM _expanded e
        JOIN _denom_family d USING (product_family, period_month)
        WHERE e.cluster_id IS NOT NULL AND e.product_family = e.cluster_family
        GROUP BY e.cluster_id, e.period_month
    """, [run_id, as_of])
    n_cluster = con.execute(
        "SELECT count(*) FROM cluster_timeseries WHERE run_id = ? AND company_id = '__ALL__'",
        [run_id],
    ).fetchone()[0]

    con.execute("""
        INSERT INTO cluster_timeseries
          (run_id, cluster_id, company_id, period_month, n, denom, share, as_of)
        SELECT ?, e.cluster_id, e.company_id, e.period_month,
               count(DISTINCT e.group_id), any_value(d.denom),
               count(DISTINCT e.group_id) / any_value(d.denom)::DOUBLE, ?
        FROM _expanded e
        JOIN _denom_company d USING (product_family, company_id, period_month)
        WHERE e.cluster_id IS NOT NULL AND e.company_id IS NOT NULL
          AND e.product_family = e.cluster_family
        GROUP BY e.cluster_id, e.company_id, e.period_month
        HAVING count(DISTINCT e.group_id) >= ?
    """, [run_id, as_of, min_groups])
    total = con.execute(
        "SELECT count(*) FROM cluster_timeseries WHERE run_id = ?", [run_id]
    ).fetchone()[0]
    return n_cluster, total - n_cluster


def contingency(con: duckdb.DuckDBPyConnection) -> list[tuple]:
    """The 2x2 counts per (family, company, cluster).

    `a` company-and-cluster, `b` company-not-cluster, `c` other-companies-and-
    cluster, `d` the rest — all within one product family, which is the stratum
    METHODOLOGY §6.1 defines the comparison inside.

    **The unit is a `(group, company)` pair, not a group.** A 2x2 table is only
    a test of association if every unit falls in exactly one cell, and a bare
    dup-group does not: one credit-repair template mailed to all three bureaus
    is three complaints against three companies sharing one `group_id`. Counting
    distinct groups in the cluster margin therefore counted that template once
    while the company margin counted it three times, so `c = n_cluster - a` came
    out too small and every PRR built on it was inflated. Found by the Phase 5
    negative control, which is exactly the defect it exists to find.

    `(group, company)` is also the right unit on its own terms: 24,507 identical
    complaints against one bureau are one allegation, and the same template sent
    to three bureaus is three — one per company it accuses.
    """
    return con.execute("""
        WITH unit AS (   -- one row per (group, company); the cell a unit belongs to
          SELECT DISTINCT product_family, group_id, company_id,
                 CASE WHEN product_family = cluster_family THEN cluster_id END AS cluster_id
          FROM _expanded WHERE company_id IS NOT NULL
        ),
        pair AS (
          SELECT product_family, company_id, cluster_id, count(*) AS a
          FROM unit WHERE cluster_id IS NOT NULL GROUP BY 1, 2, 3
        ),
        by_company AS (
          SELECT product_family, company_id, count(*) AS n_company
          FROM unit GROUP BY 1, 2
        ),
        by_cluster AS (
          SELECT product_family, cluster_id, count(*) AS n_cluster
          FROM unit WHERE cluster_id IS NOT NULL GROUP BY 1, 2
        ),
        by_family AS (
          SELECT product_family, count(*) AS n_family FROM unit GROUP BY 1
        )
        SELECT p.product_family, p.company_id, p.cluster_id,
               p.a,
               co.n_company - p.a                          AS b,
               cl.n_cluster - p.a                          AS c,
               f.n_family - co.n_company - cl.n_cluster + p.a AS d
        FROM pair p
        JOIN by_company co USING (product_family, company_id)
        JOIN by_cluster cl USING (product_family, cluster_id)
        JOIN by_family  f  USING (product_family)
        -- Ordered because the empirical-Bayes prior is fit by method of moments
        -- over these rows: np.mean/np.var sum in array order, so an unordered
        -- parallel scan changed alpha and beta in their last bits and moved EB05
        -- for a handful of pairs between otherwise identical runs.
        ORDER BY p.product_family, p.company_id, p.cluster_id
    """).fetchall()


def series(con: duckdb.DuckDBPyConnection, run_id: str) -> dict:
    """`{(cluster_id, company_id): [(month, n, denom, share), ...]}`, month-sorted."""
    rows = con.execute("""
        SELECT cluster_id, company_id, period_month, n, denom, share
        FROM cluster_timeseries WHERE run_id = ?
        ORDER BY cluster_id, company_id, period_month
    """, [run_id]).fetchall()
    out: dict = {}
    for cluster_id, company_id, month, n, denom, share in rows:
        out.setdefault((cluster_id, company_id), []).append((month, n, denom, share))
    return out
