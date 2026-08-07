# Phase 5 — Signal detection

**State:** ✅ complete; negative control 0.45% against α 0.05 · **Effort:** ~1 week

## Question

Which company × harm pairs are growing abnormally, and how many of those would
fire on pure noise?

## Context

Three decisions are made in `timeseries.py` rather than downstream, because every
statistic inherits them and none is recoverable later.

**The unit is a dup-group, not a complaint.** Phase 2's campaign flag misses
large templates that cite no statute — a 24,507-member group scored 2 of 5 and
went unflagged. Counting complaints would let that template contribute 24,507 to
a growth curve. Counting groups, it contributes one per active month, and the
flag's miss stops mattering.

**The denominator is narrative-bearing complaints.** Only 22.66% of the corpus
has a narrative and only narratives can be clustered, so dividing by all
complaints would make every share a function of narrative-consent rates rather
than of harm.

**A cluster's exposure is its own family's, on both sides.** 23.8% of clusters
contain complaints from more than one family; joining exposure on the complaint's
family matched several denominator rows and `any_value` picked one arbitrarily.

## What runs

1. `build_expanded` — complaint → (dup-group, cluster) mapping, campaign-flagged
   complaints excluded from numerator and denominator alike.
2. `build_panel` — the cluster × company × month panel.
3. `contingency` — 2×2 counts per (family, company, cluster).
4. Disproportionality: PRR / ROR with empirical-Bayes shrinkage, ranked on EB05.
5. EWMA control chart + PELT changepoint on the share series.
6. Benjamini–Hochberg within product family.

## Tech

| Choice | Why this one |
|---|---|
| **Gamma-Poisson (EB) shrinkage**, ranked on **EB05** | The 5th percentile of the shrunk posterior, never the point estimate — a 3-complaint company cannot top the ranking on a ratio of small numbers. |
| **`ruptures` PELT** | Changepoint on the share series, penalty 10.0. |
| **EWMA** λ=0.2, L=3σ, 12-month baseline | Standard control chart; catches drift PELT's segmentation misses. |
| **BH within product family** | Families differ by orders of magnitude in volume; pooling would let credit reporting set the threshold for everyone. |
| **The 2×2 unit is a `(group, company)` pair** | A 2×2 is only a test of association if each unit falls in exactly one cell. One template mailed to all three bureaus is three complaints against three companies sharing one `group_id`. |

## Acceptance

> Negative control: shuffle cluster assignments, re-run detection; the
> false-alert rate should approximate α. **Mandatory.**

**Met at 0.0045 against α 0.05**, after failing first.

## Findings

**The mandatory negative control failed at a 10.7% false-alert rate and found
three bugs — one of them in the control itself.**

All three shared one signature: a quantity *defined* as "distinct groups" but
*computed* as a sum over a partition — of companies in one case, of months in
another. None is visible in a normal run. All three produce plausible,
well-formed, entirely wrong numbers. Only a null could expose them.

One of them is the 2×2 unit bug above: counting distinct groups in the cluster
margin counted a three-bureau template once while the company margin counted it
three times, so `c = n_cluster - a` came out too small and every PRR built on it
was inflated.

**The rate, not the count, is the control's verdict.** Shuffling spreads every
cluster across every company, so a permuted panel has far more series than the
real one and raw counts are not comparable — which is exactly the mistake that
would make a broken detector look fine.

**Top alerts are dominated by companies with real public enforcement histories**
— Lexington Law, Chime, Freedom Financial, Credit Karma, Coinbase. First evidence
that Phase 6 has something to find. Checked for a structural artifact at the 2017
taxonomy boundary and found none: 1.5% of changepoints.
