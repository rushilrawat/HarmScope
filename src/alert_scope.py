"""One canonical, run-bound definition of a fired alert scope.

This module is deliberately neutral: detection and descriptive consumers may
both import it, while it imports neither the pipeline nor ``src.llm``. It reads
the immutable Phase 5 outputs and never writes signal state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.config import CONFIG


@dataclass(frozen=True)
class CanonicalAlertScope:
    product_family: str
    company_id: str
    cluster_id: str
    eb05: float | None
    q_value: float | None
    changed: bool
    change_month: date | None
    n_supporting: int
    n_groups: int
    coherence: float
    persistence: float | None


CANONICAL_ALERT_SCOPE_SQL = """
WITH source_run AS (
  SELECT json_extract_string(params_json, '$.params.cluster_run') AS cluster_run
  FROM runs
  WHERE run_id = ? AND phase = 'signals' AND status = 'ok'
    AND json_extract_string(params_json, '$.params.cluster_run') IS NOT NULL
),
grouped_signals AS (
  SELECT cluster_id, company_id,
         max(CASE WHEN method = 'ebgm' THEN statistic END) AS eb05,
         min(CASE WHEN method = 'ebgm' THEN q_value END) AS q_value,
         max(CASE WHEN method IN ('ewma', 'pelt') THEN 1 ELSE 0 END) AS changed,
         min(CASE WHEN method IN ('ewma', 'pelt') THEN period_month END) AS change_month,
         max(n_supporting) AS n_supporting,
         max(n_supporting_groups) AS n_groups
  FROM signals
  WHERE run_id = ? AND cluster_id IS NOT NULL AND company_id IS NOT NULL
  GROUP BY cluster_id, company_id
)
SELECT c.product_family, g.company_id, g.cluster_id, g.eb05, g.q_value,
       g.changed, g.change_month, g.n_supporting, g.n_groups,
       c.coherence, c.persistence
FROM grouped_signals g
JOIN clusters c USING (cluster_id)
JOIN source_run r ON r.cluster_run = c.run_id
WHERE c.coherence >= ?
  AND g.n_groups >= ?
  AND (g.q_value <= ? OR g.changed = 1)
ORDER BY g.eb05 DESC NULLS LAST, g.n_groups DESC, g.cluster_id, g.company_id
"""


def canonical_alert_scopes(con, signals_run: str) -> tuple[CanonicalAlertScope, ...]:
    """Return every supportable/coherent q-value or changepoint alert scope."""
    if type(signals_run) is not str or not signals_run.strip():
        raise ValueError("signals_run must be a nonblank string")
    rows = con.execute(
        CANONICAL_ALERT_SCOPE_SQL,
        [
            signals_run,
            signals_run,
            CONFIG.novelty.min_coherence,
            CONFIG.signals.min_supporting_groups,
            CONFIG.signals.fdr_alpha,
        ],
    ).fetchall()
    return tuple(
        CanonicalAlertScope(
            product_family=row[0],
            company_id=row[1],
            cluster_id=row[2],
            eb05=row[3],
            q_value=row[4],
            changed=bool(row[5]),
            change_month=row[6],
            n_supporting=int(row[7]),
            n_groups=int(row[8]),
            coherence=float(row[9]),
            persistence=row[10],
        )
        for row in rows
    )


def canonical_fired_cluster_ids(con, signals_run: str) -> frozenset[str]:
    """Return clusters having at least one canonical fired company/all scope."""
    return frozenset(scope.cluster_id for scope in canonical_alert_scopes(con, signals_run))
