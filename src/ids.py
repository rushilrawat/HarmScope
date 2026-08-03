"""Identifier construction for artifacts that must survive repeated refits.

The backtest refits the whole clustering stage once per annual cutoff
(docs/EVALUATION.md §1.2). A cluster id that is only locally unique — say
`mortgage-3` — therefore collides across cutoffs in every table keyed on
cluster_id: `cluster_members`, `cluster_novelty`, `cluster_labels`,
`cluster_timeseries`, `signals`, `backtest_links`.

Making the id globally unique by construction fixes that once, at the point of
creation, instead of requiring every downstream table to carry a disambiguating
run column.
"""

from __future__ import annotations

# `cluster_timeseries.company_id` is NOT NULL because DuckDB enforces NOT NULL
# on primary-key columns — a NULL "all companies" marker row cannot be inserted
# at all. This sentinel carries that meaning instead.
COMPANY_TOTAL = "__ALL__"

SEP = ":"


def cluster_id(run_id: str, product_family: str, local_id: int | str) -> str:
    """`{run_id}:{product_family}:{local_id}` — unique across refits."""
    for part, name in ((run_id, "run_id"), (str(product_family), "product_family")):
        if not part:
            raise ValueError(f"{name} must be non-empty")
        if SEP in part:
            raise ValueError(f"{name} must not contain {SEP!r}: {part!r}")
    return f"{run_id}{SEP}{product_family}{SEP}{local_id}"


def parse_cluster_id(cid: str) -> tuple[str, str, str]:
    """Inverse of `cluster_id`. Returns (run_id, product_family, local_id)."""
    parts = cid.split(SEP)
    if len(parts) != 3:
        raise ValueError(f"malformed cluster_id: {cid!r}")
    return parts[0], parts[1], parts[2]
