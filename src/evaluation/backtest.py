"""Phase 6: the point-in-time backtest harness.

docs/EVALUATION.md §1.1. For each usable enforcement action, find the most
recent annual cutoff strictly before it was filed, look only at what that
cutoff's refit produced, and ask whether anything fired for that company before
the action became public.

Two properties are the whole point and are asserted rather than assumed:

**A cutoff's refit is the only thing consulted.** Not the full-corpus run
filtered by date. `run_for_cutoff` resolves the runs by the cutoff recorded in
their own params, so a missing refit is an error rather than a silent fallback
to the run that saw everything.

**Nothing after the cutoff is read.** Signals carry `period_month`, and a lead
time is measured from the first period that fired, which is bounded by the
cutoff by construction. The anti-leakage suite checks that separately on the
stored rows, so this does not rest on the query being written correctly.

Annual cutoffs are deliberately conservative (§1.2): an action filed in December
is evaluated against a model up to twelve months stale, so every lead time here
understates the true one. Understating your own result is the right direction to
err.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import duckdb

CUTOFFS = [date(y, 1, 1) for y in range(2017, 2025)]


@dataclass(frozen=True)
class Outcome:
    action_id: str
    company_id: str
    cutoff: date
    filed_date: date
    detected: bool
    first_signal: date | None
    lead_time_days: int | None
    n_clusters_fired: int
    match_quality: str | None = None


def cutoff_for(filed: date) -> date | None:
    """The most recent annual cutoff strictly before `filed` (§1.2)."""
    prior = [c for c in CUTOFFS if c < filed]
    return prior[-1] if prior else None


def run_for_cutoff(con: duckdb.DuckDBPyConnection, phase: str, cutoff: date) -> str | None:
    """The run of `phase` produced by the refit at `cutoff`.

    Resolved from the cutoff recorded in the run's own params, never from
    recency. Returning None so the caller can report a missing refit is the
    point: falling back to the newest run would silently evaluate an action
    against a model that had seen the enforcement action itself.
    """
    row = con.execute(
        """
        SELECT run_id FROM runs
        WHERE phase = ? AND status = 'ok'
          AND json_extract_string(params_json, '$.params.cutoff') = ?
        ORDER BY started_at DESC LIMIT 1
        """,
        [phase, cutoff.isoformat()],
    ).fetchone()
    return row[0] if row else None


def evaluate(
    con: duckdb.DuckDBPyConnection,
    min_supporting_groups: int,
    fdr_alpha: float,
    strong_only: bool = False,
) -> tuple[list[Outcome], dict]:
    """One `Outcome` per usable action.

    A detection requires a signal for the action's company, from the cutoff's
    own refit, that clears the same alert criteria Phase 5 uses. Adjudication
    (§1.3) decides whether the *cluster* matches the action's harm; that is a
    separate human step and is joined in through `backtest_links` when present.
    """
    actions = con.execute(
        """
        SELECT action_id, company_id, filed_date FROM enforcement_actions
        WHERE usable AND company_id IS NOT NULL ORDER BY filed_date
        """
    ).fetchall()

    out: list[Outcome] = []
    missing: dict[date, list[str]] = {}
    for action_id, company_id, filed in actions:
        cutoff = cutoff_for(filed)
        if cutoff is None:
            continue
        signals_run = run_for_cutoff(con, "signals", cutoff)
        if signals_run is None:
            # Skipped loudly, never silently, and never by substituting another
            # run: any other run saw complaints filed after this action, so it
            # would "detect" the action using the response to it. The caller
            # reports the missing cutoffs as coverage rather than hiding the gap
            # in a denominator.
            missing.setdefault(cutoff, []).append(action_id)
            continue

        rows = con.execute(
            """
            SELECT s.cluster_id, min(s.period_month) AS first_fire
            FROM signals s
            WHERE s.run_id = ? AND s.company_id = ?
              AND s.n_supporting_groups >= ?
              AND (s.q_value <= ? OR s.method IN ('ewma', 'pelt'))
            GROUP BY s.cluster_id
            """,
            [signals_run, company_id, min_supporting_groups, fdr_alpha],
        ).fetchall()

        if strong_only and rows:
            linked = {
                r[0] for r in con.execute(
                    "SELECT cluster_id FROM backtest_links WHERE action_id = ? "
                    "AND match_quality = 'strong'", [action_id],
                ).fetchall()
            }
            rows = [r for r in rows if r[0] in linked]

        first = min((r[1] for r in rows), default=None)
        out.append(Outcome(
            action_id=action_id, company_id=company_id, cutoff=cutoff,
            filed_date=filed, detected=bool(rows), first_signal=first,
            lead_time_days=(filed - first).days if first else None,
            n_clusters_fired=len(rows),
        ))
    return out, missing


def summarise(outcomes: list[Outcome]) -> dict:
    """Detection rate, median lead time, and the denominators for both."""
    n = len(outcomes)
    hits = [o for o in outcomes if o.detected]
    leads = sorted(o.lead_time_days for o in hits if o.lead_time_days is not None)
    median = None
    if leads:
        mid = len(leads) // 2
        median = leads[mid] if len(leads) % 2 else (leads[mid - 1] + leads[mid]) / 2
    return {
        "n_actions": n,
        "n_detected": len(hits),
        "detect_rate": len(hits) / n if n else 0.0,
        "median_lead_days": median,
        "lead_p25": leads[len(leads) // 4] if leads else None,
        "lead_p75": leads[3 * len(leads) // 4] if leads else None,
    }


def write(
    con: duckdb.DuckDBPyConnection, run_id: str, system: str, outcomes: list[Outcome]
) -> int:
    """Persist to `baseline_results`, the table EVALUATION §2 compares in."""
    con.execute(
        "DELETE FROM baseline_results WHERE run_id = ? AND system = ?", [run_id, system]
    )
    con.executemany(
        "INSERT INTO baseline_results (run_id, system, action_id, cutoff, detected, "
        "first_signal, lead_time_days, match_quality) VALUES (?,?,?,?,?,?,?,?)",
        [(run_id, system, o.action_id, o.cutoff, o.detected, o.first_signal,
          o.lead_time_days, o.match_quality) for o in outcomes],
    )
    return len(outcomes)
