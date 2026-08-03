"""`complaints_raw` -> `complaints` + `narratives`.

Two stages, split because they have different inputs:

  - `complaints` is analysis-ready metadata, built entirely in SQL from
    `complaints_raw` joined to the crosswalk and the canonical company table.
  - `narratives` is built by streaming the **CSV** a second time, because the
    raw narrative text deliberately never entered `complaints_raw`
    (docs/DATA.md §6). Text goes CSV -> PII sweep -> `text_redacted` and the
    unredacted string is never written anywhere.

`issue_std` is currently the raw `Issue` value. The issue-level crosswalk is
Phase 7's problem: B1 tracks `(company x Product x Issue x Sub-issue)` growth
and needs it, whereas Phase 1 only needs `product_family`, which is what
stratification and the `denom` exposure term key on. Recorded rather than
silently conflated.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb

from src.ingestion.load import READ_OPTS, _q
from src.normalization import taxonomy
from src.normalization.pii import PATTERN_NAMES, redact
from src.normalization.text import text_hash

BATCH = 20_000


def build_complaints(con: duckdb.DuckDBPyConnection, window_start: date) -> int:
    """Populate `complaints`. Idempotent clean overwrite inside one transaction."""
    con.execute("BEGIN")
    try:
        con.execute("DELETE FROM narratives")  # FK child
        con.execute("DELETE FROM complaints")
        con.execute(
            f"""
            INSERT INTO complaints
            SELECT
              r.complaint_id,
              r.date_received,
              date_trunc('month', r.date_received)::DATE AS period_month,
              (SELECT a.company_id FROM company_alias a
                WHERE a.alias_raw = r.company_raw)       AS company_id,
              {taxonomy.family_expr("r")}                AS product_family,
              {taxonomy.product_std_expr("r")}           AS product_std,
              r.issue                                    AS issue_std,
              r.sub_issue                                AS sub_issue_std,
              r.state, r.tags, r.company_response, r.has_narrative
            FROM complaints_raw r
            WHERE r.date_received >= ?
            """,
            [window_start],
        )
        n = con.execute("SELECT count(*) FROM complaints").fetchone()[0]
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return n


def build_narratives(
    con: duckdb.DuckDBPyConnection,
    csv_path: Path,
    window_start: date,
    run_id: str,
) -> tuple[int, dict[str, int], int]:
    """Stream narratives from the CSV, redact, and store.

    Returns `(rows, per_pattern_hits, documents_with_any_redaction)`.

    The unredacted text exists only as a local variable for the duration of one
    batch. Nothing writes it to the database, to a log, or to disk.
    """
    reader = con.cursor()
    writer = con.cursor()

    reader.execute(
        f"""
        SELECT CAST("Complaint ID" AS BIGINT) AS complaint_id,
               "Consumer complaint narrative" AS narrative
        FROM read_csv({_q(str(csv_path))}, {READ_OPTS})
        WHERE TRY_CAST("Complaint ID" AS BIGINT) IS NOT NULL
          AND trim(coalesce("Consumer complaint narrative", '')) <> ''
          AND CAST(substr("Date received", 1, 10) AS DATE) >= ?
        """,
        [window_start],
    )

    counts = dict.fromkeys(PATTERN_NAMES, 0)
    docs_redacted = 0
    total = 0

    writer.execute("BEGIN")
    try:
        while batch := reader.fetchmany(BATCH):
            payload = []
            for complaint_id, narrative in batch:
                result = redact(narrative)
                for name, hits in result.counts.items():
                    counts[name] += hits
                if result.total:
                    docs_redacted += 1
                payload.append(
                    (
                        complaint_id,
                        result.text,
                        text_hash(result.text),
                        len(result.text),
                        result.total,
                    )
                )
            writer.executemany(
                "INSERT INTO narratives (complaint_id, text_redacted, text_hash, "
                "char_len, redaction_count) VALUES (?, ?, ?, ?, ?)",
                payload,
            )
            total += len(payload)

        writer.executemany(
            "INSERT INTO redaction_stats (run_id, pattern, n_hits, n_documents) "
            "VALUES (?, ?, ?, ?)",
            [(run_id, name, hits, docs_redacted) for name, hits in counts.items()],
        )
        writer.execute("COMMIT")
    except Exception:
        writer.execute("ROLLBACK")
        raise

    return total, counts, docs_redacted


def monthly_family_volume(
    con: duckdb.DuckDBPyConnection, family: str = "credit_reporting"
) -> list[tuple[date, int]]:
    """Monthly volume for one family, for the discontinuity check.

    ROADMAP Phase 1 acceptance: the 2017 taxonomy break must be visible in raw
    `Product` volume and absent once labels are crosswalked to families.
    """
    return con.execute(
        "SELECT period_month, count(*) FROM complaints "
        "WHERE product_family = ? GROUP BY 1 ORDER BY 1",
        [family],
    ).fetchall()
