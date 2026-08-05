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


def _scoped(run_id: str, scope: str, local_id: int | str) -> str:
    for part, name in ((run_id, "run_id"), (str(scope), "scope")):
        if not part:
            raise ValueError(f"{name} must be non-empty")
        if SEP in part:
            raise ValueError(f"{name} must not contain {SEP!r}: {part!r}")
    return f"{run_id}{SEP}{scope}{SEP}{local_id}"


def cluster_id(run_id: str, product_family: str, local_id: int | str) -> str:
    """`{run_id}:{product_family}:{local_id}` — unique across refits."""
    return _scoped(run_id, product_family, local_id)


def campaign_id(run_id: str, local_id: int | str) -> str:
    """`{run_id}:campaign:{local_id}` — unique across refits.

    Campaign features (`burstiness`, `state_concentration`, `first_seen`) are
    time-windowed, so campaigns are regenerated at every cutoff and a bare
    local id would collide the same way cluster ids would.
    """
    return _scoped(run_id, "campaign", local_id)


def parse_cluster_id(cid: str) -> tuple[str, str, str]:
    """Inverse of `cluster_id`. Returns (run_id, product_family, local_id)."""
    parts = cid.split(SEP)
    if len(parts) != 3:
        raise ValueError(f"malformed cluster_id: {cid!r}")
    return parts[0], parts[1], parts[2]


def signal_id(run_id: str, local_id: int | str) -> str:
    """`{run_id}:signal:{local_id}` — unique across refits.

    Same reason as `campaign_id`: signals are recomputed at every backtest
    cutoff from that cutoff's clusters, so eight refits would put eight
    different rows in the same slot if the id were a bare local counter.
    """
    return _scoped(run_id, "signal", local_id)
