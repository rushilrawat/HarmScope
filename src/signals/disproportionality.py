"""Phase 5: PRR, ROR, empirical-Bayes shrinkage, BH correction.

docs/METHODOLOGY.md §6.1. Borrowed from pharmacovigilance, where the problem
shape is identical: spontaneous reports, no denominator, and a need to rank
company-event pairs without being fooled by small counts.

**Shrinkage is a single-gamma empirical Bayes, not DuMouchel's two-component
mixture.** §6.1 asks for "empirical-Bayes gamma-Poisson shrinkage (EBGM / EB05)
or, at minimum, `a >= 5` and the lower CI bound". The full GPS fits a five
-parameter mixture by maximum likelihood per stratum; a single gamma prior fit
by method of moments is conjugate, closed-form, has no convergence failure mode
on the sparse strata this corpus has, and does the thing that actually matters —
pulling small-`a` ratios toward the null in proportion to how little evidence
they carry. The deviation is recorded here and in `ENGINEERING_NOTES.md` Phase 5
rather than implied by the config field name.

**The ranking quantity is EB05**, the 5th percentile of the posterior, never the
point estimate. A pair with `a = 5` and PRR = 12 and a pair with `a = 900` and
PRR = 2.1 are not comparable on PRR, and ranking on it puts the noise first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Below this the normal approximation on log ROR is not worth reporting, and
# METHODOLOGY §6.1 sets the same floor for a different reason (raw PRR is
# unstable). Both point at the same guard.
EPS = 1e-12


@dataclass(frozen=True)
class Disproportionality:
    family: str
    company_id: str
    cluster_id: str
    a: int
    prr: float
    prr_low: float
    prr_high: float
    ror: float
    p_value: float
    ebgm: float
    eb05: float
    q_value: float = float("nan")


def prr_with_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """PRR and its 95% CI from the normal approximation on log PRR."""
    if a <= 0 or (a + b) <= 0 or c <= 0 or (c + d) <= 0:
        return float("nan"), float("nan"), float("nan")
    exposed = a / (a + b)
    baseline = c / (c + d)
    if baseline <= 0:
        return float("nan"), float("nan"), float("nan")
    prr = exposed / baseline
    se = math.sqrt(max(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d), EPS))
    return prr, prr * math.exp(-1.96 * se), prr * math.exp(1.96 * se)


def ror_with_p(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """ROR and a two-sided p-value from the normal approximation on log ROR.

    Normal approximation rather than Fisher's exact: with ~38,000 tests the
    exact test is the dominant cost of the phase, and `min_a` already keeps the
    approximation inside the range where it holds. The negative-control test is
    what checks that claim empirically — if the approximation were biased, a
    shuffled panel would produce alerts above the FDR rate.
    """
    if min(a, b, c, d) <= 0:
        return float("nan"), 1.0
    ror = (a / b) / (c / d)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = abs(math.log(ror)) / max(se, EPS)
    # Two-sided normal tail without scipy: erfc is in the stdlib.
    p = math.erfc(z / math.sqrt(2))
    return ror, p


def gamma_prior(observed: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """Method-of-moments gamma prior for the relative-reporting-rate.

    Fit across all pairs in a stratum, which is what makes it *empirical* Bayes:
    the shrinkage target is the observed distribution of ratios in this family,
    not an assumption imported from elsewhere. Returns `(alpha, beta)` such that
    the prior mean is the pooled ratio.
    """
    ratio = observed / np.maximum(expected, EPS)
    mean = float(np.mean(ratio))
    var = float(np.var(ratio))
    if var <= EPS or mean <= EPS:
        # No spread to learn from: a flat-ish prior that shrinks toward 1.
        return 1.0, 1.0
    beta = mean / var
    alpha = mean * beta
    # Keep the prior weakly informative; an alpha of hundreds would swamp the data.
    return float(np.clip(alpha, 0.05, 100.0)), float(np.clip(beta, 0.05, 100.0))


def eb_shrink(
    observed: np.ndarray, expected: np.ndarray, alpha: float, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean (EBGM) and 5th percentile (EB05) per pair.

    Gamma(alpha, beta) prior with Poisson likelihood is conjugate, so the
    posterior is Gamma(alpha + a, beta + E) and both quantities are closed form.
    """
    from scipy.stats import gamma as gamma_dist

    post_shape = alpha + observed
    post_rate = beta + expected
    ebgm = post_shape / post_rate
    eb05 = gamma_dist.ppf(0.05, a=post_shape, scale=1.0 / post_rate)
    return ebgm, eb05


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH step-up q-values. Applied **within product family** (§6.1).

    Within family, not globally: credit reporting contributes over half the
    tests, so a global correction would set the threshold for every other family
    according to how noisy credit reporting is.
    """
    n = len(p)
    if n == 0:
        return np.empty(0)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    # Step-up: q is monotone non-decreasing in p, enforced from the top down.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def analyse(rows: list[tuple], min_a: int) -> list[Disproportionality]:
    """Score every 2x2, shrink within family, and BH-correct within family."""
    by_family: dict[str, list[tuple]] = {}
    for family, company, cluster, a, b, c, d in rows:
        if a >= min_a and min(b, c, d) > 0:
            by_family.setdefault(family, []).append((company, cluster, a, b, c, d))

    out: list[Disproportionality] = []
    for family, pairs in by_family.items():
        a_arr = np.array([p[2] for p in pairs], dtype=np.float64)
        # Expected count under independence within the family's 2x2 margins.
        expected = np.array(
            [(p[2] + p[3]) * (p[2] + p[4]) / (p[2] + p[3] + p[4] + p[5])
             for p in pairs],
            dtype=np.float64,
        )
        alpha, beta = gamma_prior(a_arr, expected)
        ebgm, eb05 = eb_shrink(a_arr, expected, alpha, beta)

        scored = []
        for (company, cluster, a, b, c, d) in pairs:
            prr, low, high = prr_with_ci(a, b, c, d)
            ror, p = ror_with_p(a, b, c, d)
            scored.append((company, cluster, a, prr, low, high, ror, p))
        q = benjamini_hochberg(np.array([s[7] for s in scored]))

        for i, (company, cluster, a, prr, low, high, ror, p) in enumerate(scored):
            out.append(Disproportionality(
                family=family, company_id=company, cluster_id=cluster, a=int(a),
                prr=prr, prr_low=low, prr_high=high, ror=ror, p_value=p,
                ebgm=float(ebgm[i]), eb05=float(eb05[i]), q_value=float(q[i]),
            ))
    return out
