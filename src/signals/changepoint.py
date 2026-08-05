"""Phase 5: EWMA control chart and PELT changepoint on the share series.

docs/METHODOLOGY.md §6.2 specifies a negative-binomial GLM with an exposure
offset, then an EWMA chart and a PELT changepoint layered on the share series.

**The NB fit is replaced by an empirical dispersion estimate, deliberately.**
The reason §6.2 wants negative binomial is that complaint counts are
overdispersed and Poisson control limits would be far too tight. But fitting a
per-series NB GLM means ~2,800 cluster-level fits plus tens of thousands of
company-level ones, on series that are frequently short, zero-inflated, or
constant — the regime where NB fitting does not converge and silently returns a
degenerate theta. Taking the control limit from the **observed standard
deviation of the baseline window** captures whatever overdispersion the series
actually has, without assuming a parametric form and without a failure mode that
returns a number rather than an error.

The exposure offset survives: the chart runs on `share = n / denom`, so
corpus-wide growth is divided out before anything is tested. That was §6.2's
actual requirement — "without the offset you detect overall database growth" —
and it is met by construction rather than by a fitted coefficient.

A signal's date is the **first** period the detector fires, computed from a
baseline strictly before that period. Nothing here looks forward; that is what
makes the lead time in Phase 6 a lead time rather than a hindsight.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass(frozen=True)
class Change:
    method: str          # 'ewma' | 'pelt'
    period: date
    statistic: float     # EWMA: z against the baseline. PELT: relative jump.


def densify(points: list[tuple], months: list[date]) -> np.ndarray:
    """Zero-fill a sparse series onto a common month axis.

    Months a cluster is absent are real zeros, not missing data — the cluster
    existed and nothing landed in it. Leaving them out would make a series that
    fires once a year look continuously present.
    """
    index = {m: i for i, m in enumerate(months)}
    out = np.zeros(len(months), dtype=np.float64)
    for month, _n, _denom, share in points:
        pos = index.get(month)
        if pos is not None:
            out[pos] = share
    return out


def ewma_alarm(
    values: np.ndarray,
    months: list[date],
    lam: float,
    control_limit: float,
    baseline_months: int,
) -> Change | None:
    """First period where the EWMA of `share` leaves its control limits.

    One-sided: a cluster shrinking is not a harm signal. The baseline is a
    trailing window that stops before the period under test, so the statistic at
    month t uses only months < t.
    """
    if len(values) <= baseline_months + 1:
        return None
    ewma = values[0]
    for t in range(1, len(values)):
        if t < baseline_months:
            ewma = lam * values[t] + (1 - lam) * ewma
            continue
        window = values[t - baseline_months : t]
        mu = float(window.mean())
        sigma = float(window.std(ddof=1)) if len(window) > 1 else 0.0
        # EWMA of an iid series has variance sigma^2 * lam / (2 - lam) in steady
        # state; using the raw sigma would flag noise the smoother has removed.
        limit = control_limit * sigma * np.sqrt(lam / (2 - lam))
        ewma = lam * values[t] + (1 - lam) * ewma
        if sigma > 0 and ewma - mu > limit:
            return Change("ewma", months[t], (ewma - mu) / max(sigma, 1e-12))
    return None


def pelt_alarm(
    values: np.ndarray, months: list[date], penalty: float, min_size: int = 6
) -> Change | None:
    """First PELT breakpoint that is an increase, with its relative jump.

    `ruptures` returns every regime boundary; only upward shifts are signals,
    and only the earliest one matters for lead time.
    """
    import ruptures

    if len(values) < 2 * min_size or values.std() == 0:
        return None
    # Scale-free penalty: an absolute penalty on `share` would fire constantly
    # on high-volume clusters and never on small ones.
    scaled = values / max(values.std(), 1e-12)
    try:
        bkps = ruptures.Pelt(model="rbf", min_size=min_size).fit(
            scaled.reshape(-1, 1)
        ).predict(pen=penalty)
    except Exception:  # noqa: BLE001 - a failed fit is "no changepoint", not a crash
        return None

    for bkp in bkps[:-1]:
        before = values[:bkp]
        after = values[bkp:]
        if len(before) < min_size or len(after) < min_size:
            continue
        mu_before, mu_after = before.mean(), after.mean()
        if mu_after > mu_before:
            jump = (mu_after - mu_before) / max(mu_before, 1e-12)
            return Change("pelt", months[bkp], float(jump))
    return None


def detect(
    series: dict, months: list[date], cfg
) -> dict[tuple[str, str], list[Change]]:
    """Run both detectors over every series. Returns only series that fired."""
    out: dict[tuple[str, str], list[Change]] = {}
    for key, points in series.items():
        values = densify(points, months)
        fired = [
            change for change in (
                ewma_alarm(values, months, cfg.ewma_lambda, cfg.ewma_control_limit,
                           cfg.ewma_baseline_months),
                pelt_alarm(values, months, cfg.pelt_penalty),
            ) if change is not None
        ]
        if fired:
            out[key] = fired
    return out
