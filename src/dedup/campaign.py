"""Tier 3: campaign detection over dup-group candidates.

docs/METHODOLOGY.md §2.2. Groups alone are not enough — a campaign may vary
phrasing enough to evade MinHash — so each candidate group is scored on six
readable signals and flagged when enough of them fire.

Deliberately **not** a supervised model, per the spec: there are not enough
labels, the features are interpretable, and an analyst has to be able to look
at a flag and see why it fired. `n_signals` is stored alongside the flag so the
reason is inspectable without recomputing anything.

Handling, not deletion (§2.3): flagged complaints stay in the database and stay
queryable. They are excluded from signal detection by default and surfaced in
the UI as a separate labelled band.
"""

from __future__ import annotations

import duckdb

from src.config import DedupConfig

# Statutory citations and formulaic legal phrasing. These are the fingerprints
# of credit-repair boilerplate rather than of a consumer describing a problem:
# a real complaint says "they keep reporting an account that isn't mine", a
# template says "under 15 U.S.C. 1681 the following accounts have violated my
# federally protected consumer rights".
BOILERPLATE_MARKERS = (
    r"15\s*u\.?\s*s\.?\s*c",           # 15 U.S.C. / 15 USC
    r"15\s*u\.?s\.?\s*code",
    r"\b1681[a-z]?\b",                 # FCRA sections
    r"\b1692[a-z]?\b",                 # FDCPA sections
    r"\bsection\s*6(0[49]|1[1-3]|23)\b",
    r"fair\s+credit\s+reporting\s+act",
    r"\bfcra\b", r"\bfdcpa\b",
    r"federally\s+protected\s+consumer\s+rights",
    r"willful\s+non[- ]?compliance",
    r"estoppel\s+by\s+silence",
    r"under\s+penalty\s+of\s+perjury",
    r"permissible\s+purpose",
    r"validation\s+of\s+(the\s+)?debt",
    r"cease\s+and\s+desist",
    r"metro\s*2\b",
    r"e[-\s]?oscar",
)
BOILERPLATE_RE = "(?i)(" + "|".join(BOILERPLATE_MARKERS) + ")"


def feature_sql(run_id: str) -> str:
    """Per-group features for every dup-group in this run.

    HHI (Herfindahl index) is `sum(share^2)`: 1.0 when every member shares one
    value, ~1/k when spread across k values. Organic harms spread across states
    and channels; campaigns concentrate (docs/METHODOLOGY.md §2.2).
    """
    return f"""
    WITH members AS (
      SELECT g.group_id, g.complaint_id, c.date_received, c.product_family,
             c.company_id, c.state, r.submitted_via, n.char_len,
             CASE WHEN regexp_matches(n.text_redacted, '{BOILERPLATE_RE}')
                  THEN 1 ELSE 0 END AS boiler
      FROM dup_groups g
      JOIN complaints      c USING (complaint_id)
      JOIN complaints_raw  r USING (complaint_id)
      JOIN narratives      n USING (complaint_id)
      WHERE g.run_id = '{run_id}'
    ),
    daily AS (           -- Fano factor needs the per-day counts, zeros included
      SELECT group_id, date_received, count(*) AS n
      FROM members GROUP BY 1, 2
    ),
    burst AS (
      SELECT d.group_id,
             CASE WHEN avg(d.n) > 0
                  THEN coalesce(var_pop(d.n), 0) / avg(d.n) END AS burstiness
      FROM daily d GROUP BY 1
    ),
    -- The three HHIs are computed by concentration_sql() and joined in Python.
    -- Inlining three more grouped subqueries here made the statement unreadable
    -- for no measurable gain; each pass is a scan of the same small join.
    _unused AS (SELECT 1)
    SELECT m.group_id,
           count(*)                                   AS n_complaints,
           min(m.date_received)                       AS first_seen,
           max(m.date_received)                       AS last_seen,
           any_value(m.product_family)                AS product_family,
           mode(m.company_id)                         AS top_company_id,
           b.burstiness,
           CASE WHEN avg(m.char_len) > 0
                THEN stddev_pop(m.char_len) / avg(m.char_len) END AS length_cv,
           avg(m.boiler)                              AS boilerplate_score
    FROM members m JOIN burst b USING (group_id)
    GROUP BY m.group_id, b.burstiness
    """


def concentration_sql(run_id: str, column: str) -> str:
    """HHI over one column, computed separately to keep the SQL readable."""
    return f"""
    WITH m AS (
      SELECT g.group_id, {column} AS v
      FROM dup_groups g
      JOIN complaints c USING (complaint_id)
      JOIN complaints_raw r USING (complaint_id)
      WHERE g.run_id = '{run_id}'
    ), per AS (
      SELECT group_id, v, count(*) AS n FROM m GROUP BY 1, 2
    ), tot AS (
      SELECT group_id, sum(n) AS t FROM per GROUP BY 1
    )
    SELECT per.group_id, sum(pow(per.n::DOUBLE / tot.t, 2)) AS hhi
    FROM per JOIN tot USING (group_id) GROUP BY 1
    """


def count_signals(row: dict, cfg: DedupConfig) -> int:
    """How many of the six signals fire. Stored so a flag is auditable."""
    checks = (
        (row.get("burstiness") or 0) > cfg.burstiness_threshold,
        (row.get("state_concentration") or 0) > cfg.state_concentration_threshold,
        (row.get("company_concentration") or 0) > cfg.company_concentration_threshold,
        (row.get("submitted_via_concentration") or 0)
        > cfg.submitted_via_concentration_threshold,
        # Templates have unnaturally LOW length variance — note the direction.
        (row.get("length_cv") if row.get("length_cv") is not None else 1.0)
        < cfg.length_cv_threshold,
        (row.get("boilerplate_score") or 0) > cfg.boilerplate_threshold,
    )
    return sum(bool(c) for c in checks)


def is_flagged(row: dict, cfg: DedupConfig) -> bool:
    """Flag a candidate as a campaign.

    Size gate first: a three-complaint group is not a mass filing whatever its
    features look like, and small groups make every concentration statistic
    degenerate (HHI over two members is at least 0.5 by construction).
    """
    if row["n_complaints"] < cfg.campaign_min_size:
        return False
    return count_signals(row, cfg) >= cfg.campaign_min_signals


def boilerplate_share(con: duckdb.DuckDBPyConnection) -> float:
    """Corpus-wide share of narratives citing statute — the baseline a group's
    `boilerplate_score` has to be read against."""
    return con.execute(
        f"SELECT avg(CASE WHEN regexp_matches(text_redacted, '{BOILERPLATE_RE}') "
        f"THEN 1.0 ELSE 0 END) FROM narratives"
    ).fetchone()[0]
