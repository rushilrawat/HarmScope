"""Shared complaint-population semantics for signals and downstream evidence.

The expansion is detection-owned data, but not detection-package code: keeping
the SQL here lets signal construction and descriptive retrieval consume the
same dedup/campaign boundary without either package importing the other.
"""

# One row per narrative-bearing, non-campaign complaint, carrying the cluster
# assigned to its dup-group representative. Callers add their own aggregation
# or evidence-selection rules after this common boundary.
EXPANDED_SELECT_SQL = """
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
  d.group_id, r.cluster_id, r.product_family AS cluster_family
FROM dup_groups d
JOIN complaints c USING (complaint_id)
LEFT JOIN dup_groups dr
       ON dr.run_id = d.run_id AND dr.group_id = d.group_id AND dr.is_representative
LEFT JOIN rep_cluster r ON r.rep_id = dr.complaint_id
WHERE d.run_id = ?
  AND (? IS NULL OR c.date_received < ?)
  AND c.complaint_id NOT IN (SELECT complaint_id FROM flagged)
"""


EXPANDED_SQL = "CREATE OR REPLACE TEMP TABLE _expanded AS\n" + EXPANDED_SELECT_SQL
