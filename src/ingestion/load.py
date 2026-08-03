"""CFPB bulk CSV -> `complaints_raw`.

docs/ARCHITECTURE.md §5: DuckDB `read_csv` directly, no pandas. The file is
~15 GB uncompressed; materialising it in Python is not an option and is not
necessary.

`complaints_raw` is as-downloaded and never mutated. Note what it does *not*
hold: the narrative text. Only `has_narrative` is recorded here. The narrative
goes straight from CSV through the PII sweep into `narratives.text_redacted`,
so raw consumer-written text never lands in the database file at all
(docs/DATA.md §6).

Everything is read as VARCHAR and cast explicitly. Letting DuckDB sniff types
across 10^7 rows means one malformed value in row 8,000,000 silently changes a
column's type, and the failure surfaces three stages later as a cast error on
something unrelated.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import duckdb

# CFPB header -> our column. Verified against the live CFPB export, 2026-08-03.
COLUMN_MAP: dict[str, str] = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Date sent to company": "date_sent_to_company",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Company": "company_raw",
    "Company public response": "company_public_response",
    "Company response to consumer": "company_response",
    "Timely response?": "timely_response",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Submitted via": "submitted_via",
    "Consumer complaint narrative": "has_narrative",
}

NARRATIVE_COLUMN = "Consumer complaint narrative"


class HeaderMismatch(RuntimeError):
    """The CSV does not have the columns this loader was written against."""


def _q(text: str) -> str:
    """Quote a string as a SQL literal.

    DuckDB table functions like `read_csv` cannot take bind parameters, so the
    path has to be interpolated. Doubling single quotes is the correct escape
    for a SQL string literal; this is the only untrusted-ish value that reaches
    these statements, and it is a filesystem path from our own config.
    """
    escaped = text.replace("'", "''")
    return f"'{escaped}'"


def _blank_to_null(csv_col: str) -> str:
    """CFPB writes empty cells as `''` and absent tags as the literal `None`."""
    return f"nullif(nullif(\"{csv_col}\", ''), 'None')"


def csv_header(con: duckdb.DuckDBPyConnection, csv_path: Path) -> list[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_csv({_q(str(csv_path))}, "
        f"header = true, all_varchar = true, sample_size = 1)"
    ).fetchall()
    return [r[0] for r in rows]


def check_header(con: duckdb.DuckDBPyConnection, csv_path: Path) -> list[str]:
    """Fail loudly if the export shape changed. Extra columns are fine.

    CFPB has restructured this file before (docs/DATA.md §3.4) and federal data
    availability has been volatile. A loader that silently maps four of sixteen
    columns is worse than one that refuses to run.
    """
    header = csv_header(con, csv_path)
    missing = [c for c in COLUMN_MAP if c not in header]
    if missing:
        raise HeaderMismatch(
            f"{csv_path.name} is missing expected columns: {missing}\n"
            f"found: {header}\n"
            f"Update COLUMN_MAP in this module and record the change in "
            f"docs/ENGINEERING_NOTES.md — do not paper over it."
        )
    return header


def _select_sql(csv_path: Path) -> str:
    return f"""
        SELECT
          CAST("Complaint ID" AS BIGINT)                            AS complaint_id,
          -- Bulk export uses YYYY-MM-DD; the search API uses full ISO timestamps.
          -- Taking the first ten characters handles both without a format guess.
          CAST(substr("Date received", 1, 10) AS DATE)              AS date_received,
          TRY_CAST(substr("Date sent to company", 1, 10) AS DATE)   AS date_sent_to_company,
          {_blank_to_null("Product")}                               AS product,
          {_blank_to_null("Sub-product")}                           AS sub_product,
          {_blank_to_null("Issue")}                                 AS issue,
          {_blank_to_null("Sub-issue")}                             AS sub_issue,
          {_blank_to_null("Company")}                               AS company_raw,
          {_blank_to_null("Company public response")}               AS company_public_response,
          {_blank_to_null("Company response to consumer")}          AS company_response,
          CASE lower(trim("Timely response?"))
               WHEN 'yes' THEN true WHEN 'no' THEN false END       AS timely_response,
          {_blank_to_null("State")}                                 AS state,
          {_blank_to_null("ZIP code")}                              AS zip_code,
          {_blank_to_null("Tags")}                                  AS tags,
          {_blank_to_null("Submitted via")}                         AS submitted_via,
          -- Presence only. The text itself never enters the database.
          coalesce(trim("{NARRATIVE_COLUMN}") <> '', false)         AS has_narrative
        FROM read_csv({_q(str(csv_path))}, header = true, all_varchar = true)
    """


def load_raw(
    con: duckdb.DuckDBPyConnection,
    csv_path: Path,
    validate: Callable[[duckdb.DuckDBPyConnection], None] | None = None,
) -> int:
    """Load the snapshot into `complaints_raw`. Returns the row count.

    Idempotent by clean overwrite: re-running replaces the table contents in one
    transaction, so a failure part-way leaves the previous load intact rather
    than a half-loaded table that downstream stages would happily read.

    `validate` runs after the insert and **before the commit**. If it raises,
    the load is rolled back. Committing first and asserting afterwards leaves
    the table holding data that failed its own acceptance checks while the run
    is recorded as failed — which is trap T1 with the sign flipped: downstream
    stages find rows and proceed, and nothing in the data says they shouldn't.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"no CSV at {csv_path}; run `make download` first")
    check_header(con, csv_path)

    con.execute("BEGIN")
    try:
        con.execute("DELETE FROM complaints_raw")
        con.execute(
            f"INSERT INTO complaints_raw BY NAME {_select_sql(csv_path)}"
        )
        n = con.execute("SELECT count(*) FROM complaints_raw").fetchone()[0]
        if validate is not None:
            validate(con)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    return n


def csv_row_count(con: duckdb.DuckDBPyConnection, csv_path: Path) -> int:
    """Count rows in the CSV itself, for reconciliation against the load."""
    return con.execute(
        f"SELECT count(*) FROM read_csv({_q(str(Path(csv_path)))}, "
        f"header = true, all_varchar = true)"
    ).fetchone()[0]
