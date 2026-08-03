"""Taxonomy crosswalk: raw CFPB labels -> stable `product_family`.

docs/DATA.md §3.4. CFPB restructured `Product` twice — 2017-04 and 2023-08 —
so a raw `Product` value is only meaningful inside its era. Clustering is
stratified by family (docs/METHODOLOGY.md §4.2) and `cluster_timeseries.denom`
counts complaints in the same family, so an unstable family assignment would
put an artificial step change into every denominator at each boundary.

Two of the changes are **splits**, not renames, so the mapping is keyed on
`(product, sub_product)` with `'*'` as the sub-product wildcard:

    Consumer Loan               -> vehicle_loan | personal_loan
    Credit card or prepaid card -> credit_card  | prepaid_card

The mapping itself is hand-curated and committed at
`data/ground_truth/taxonomy_crosswalk.csv` — it is a judgement about what
counts as the same product across a decade of schema churn, and it is a real
dataset contribution, so it lives in git rather than in a dict in this file.
"""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb

from src.config import PATHS

WILDCARD = "*"
CROSSWALK_CSV = "taxonomy_crosswalk.csv"


class CrosswalkIncomplete(RuntimeError):
    """The corpus contains labels the crosswalk does not cover."""


def crosswalk_path() -> Path:
    return PATHS.ground_truth / CROSSWALK_CSV


def load_crosswalk(con: duckdb.DuckDBPyConnection, path: Path | None = None) -> int:
    """Load the committed crosswalk CSV into `taxonomy_crosswalk`."""
    path = path or crosswalk_path()
    if not path.exists():
        raise FileNotFoundError(f"no crosswalk at {path}")

    with path.open(newline="", encoding="utf-8") as fh:
        rows = [
            (
                r["product_raw"], r["sub_product_raw"], r["product_std"],
                r["product_family"], r["era"],
            )
            for r in csv.DictReader(fh)
        ]
    if not rows:
        raise CrosswalkIncomplete(f"{path} has no rows")

    con.execute("DELETE FROM taxonomy_crosswalk")
    con.executemany(
        "INSERT INTO taxonomy_crosswalk "
        "(product_raw, sub_product_raw, product_std, product_family, era) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def uncovered_labels(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, int]]:
    """`(product, sub_product, n)` combinations the crosswalk does not resolve.

    Must be empty before `complaints` is built: `product_family` is NOT NULL,
    and defaulting an unmapped product to 'other' would silently move rows into
    a family that never sees them again.
    """
    return con.execute(
        """
        SELECT r.product, coalesce(r.sub_product, '(null)'), count(*) AS n
        FROM complaints_raw r
        LEFT JOIN taxonomy_crosswalk x
               ON x.product_raw = r.product
              AND x.sub_product_raw = coalesce(r.sub_product, '')
        LEFT JOIN taxonomy_crosswalk w
               ON w.product_raw = r.product
              AND w.sub_product_raw = ?
        WHERE x.product_family IS NULL AND w.product_family IS NULL
        GROUP BY 1, 2 ORDER BY n DESC
        """,
        [WILDCARD],
    ).fetchall()


def family_expr(alias: str = "r") -> str:
    """SQL that resolves `product_family`, exact sub-product match winning.

    Written as an expression rather than a view so the caller controls the
    scan: this runs once over 16.9M rows inside the `complaints` build.
    """
    return f"""
        coalesce(
          (SELECT x.product_family FROM taxonomy_crosswalk x
            WHERE x.product_raw = {alias}.product
              AND x.sub_product_raw = coalesce({alias}.sub_product, '')),
          (SELECT w.product_family FROM taxonomy_crosswalk w
            WHERE w.product_raw = {alias}.product
              AND w.sub_product_raw = '{WILDCARD}')
        )
    """


def product_std_expr(alias: str = "r") -> str:
    return f"""
        coalesce(
          (SELECT x.product_std FROM taxonomy_crosswalk x
            WHERE x.product_raw = {alias}.product
              AND x.sub_product_raw = coalesce({alias}.sub_product, '')),
          (SELECT w.product_std FROM taxonomy_crosswalk w
            WHERE w.product_raw = {alias}.product
              AND w.sub_product_raw = '{WILDCARD}')
        )
    """
