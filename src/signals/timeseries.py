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

**Campaign-flagged complaints are excluded** (§6.3), from numerator and
denominator alike. Dropping them from one side only would inflate or deflate
every share in a family by the campaign rate of that family, which runs from
0.01% to 30%.
"""

from __future__ import annotations

import duckdb

# One row per narrative-bearing, non-campaign complaint, carrying the cluster its
# dup-group's representative landed in. Built once and reused: the join chain
# (member -> group -> sibling complaints) is the expensive part of the phase.
EXPANDED_SQL = """
CREATE OR REPLACE TEMP TABLE _expanded AS
WITH rep_cluster AS (
  SELECT m.complaint_id AS rep_id, m.cluster_id, cl.product_family
  FROM cluster_members m
  JOIN clusters cl USING (cluster_id)
  WHERE cl.run_id = ?
),
flagged AS (
  SELECT DISTINCT cm.complaint_id
  FROM campaign_members cm JOIN campaigns ca USING (campaign_id)
  WHERE ca.run_id = ? AND ca.flagged
)
SELECT
  c.complaint_id, c.company_id, c.period_month, c.product_family,
  d.group_id, r.cluster_id
FROM dup_groups d
JOIN complaints c USING (complaint_id)
LEFT JOIN dup_groups dr
       ON dr.run_id = d.run_id AND dr.group_id = d.group_id AND dr.is_representative
LEFT JOIN rep_cluster r ON r.rep_id = dr.complaint_id
WHERE d.run_id = ?
  AND c.complaint_id NOT IN (SELECT complaint_id FROM flagged)
"""


def build_expanded(
    con: duckdb.DuckDBPyConnection, cluster_run: str, dedup_run: str
) -> int:
    """Materialize the complaint -> (group, cluster) mapping for one run pair."""
    con.execute(EXPANDED_SQL, [cluster_run, dedup_run, dedup_run])
    return con.execute("SELECT count(*) FROM _expanded").fetchone()[0]


def build_panel(
    con: duckdb.DuckDBPyConnection, cluster_run: str, as_of, min_groups: int = 1
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
    con.execute("DELETE FROM cluster_timeseries WHERE cluster_id IN "
                "(SELECT cluster_id FROM clusters WHERE run_id = ?)", [cluster_run])

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
          (cluster_id, company_id, period_month, n, denom, share, as_of)
        SELECT e.cluster_id, '__ALL__', e.period_month,
               count(DISTINCT e.group_id) AS n, any_value(d.denom) AS denom,
               count(DISTINCT e.group_id) / any_value(d.denom)::DOUBLE, ?
        FROM _expanded e
        JOIN _denom_family d USING (product_family, period_month)
        WHERE e.cluster_id IS NOT NULL
        GROUP BY e.cluster_id, e.period_month
    """, [as_of])
    n_cluster = con.execute(
        "SELECT count(*) FROM cluster_timeseries WHERE company_id = '__ALL__'"
    ).fetchone()[0]

    con.execute("""
        INSERT INTO cluster_timeseries
          (cluster_id, company_id, period_month, n, denom, share, as_of)
        SELECT e.cluster_id, e.company_id, e.period_month,
               count(DISTINCT e.group_id), any_value(d.denom),
               count(DISTINCT e.group_id) / any_value(d.denom)::DOUBLE, ?
        FROM _expanded e
        JOIN _denom_company d USING (product_family, company_id, period_month)
        WHERE e.cluster_id IS NOT NULL AND e.company_id IS NOT NULL
        GROUP BY e.cluster_id, e.company_id, e.period_month
        HAVING count(DISTINCT e.group_id) >= ?
    """, [as_of, min_groups])
    total = con.execute("SELECT count(*) FROM cluster_timeseries").fetchone()[0]
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
          SELECT DISTINCT product_family, group_id, company_id, cluster_id
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
    """).fetchall()


def series(con: duckdb.DuckDBPyConnection, cluster_run: str) -> dict:
    """`{(cluster_id, company_id): [(month, n, denom, share), ...]}`, month-sorted."""
    rows = con.execute("""
        SELECT t.cluster_id, t.company_id, t.period_month, t.n, t.denom, t.share
        FROM cluster_timeseries t
        JOIN clusters c USING (cluster_id)
        WHERE c.run_id = ?
        ORDER BY t.cluster_id, t.company_id, t.period_month
    """, [cluster_run]).fetchall()
    out: dict = {}
    for cluster_id, company_id, month, n, denom, share in rows:
        out.setdefault((cluster_id, company_id), []).append((month, n, denom, share))
    return out
