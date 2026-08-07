# Phase 1 — Ingestion & normalization

**State:** ✅ complete · **Effort:** ~4 days

## Question

What is actually in the CFPB corpus, and what has to be repaired before any of
it can be compared across time?

## Context

The corpus spans a decade during which CFPB restructured its own taxonomy. A
growth curve computed across that boundary measures the restructuring, not the
harm. Since the entire project is about growth over time, the crosswalk is not
cleanup — it is a precondition for every downstream number.

Only 22.66% of complaints carry a narrative, and only narratives can be
clustered. That ratio governs everything: every rate is reported over the
narrative-bearing subsample, never over all complaints.

## What runs

1. Download the bulk CSV snapshot to `data/raw/` with a manifest (URL, date,
   sha256, row count). Bulk, not API pagination.
2. Load → `complaints_raw` via DuckDB `read_csv_auto`. No pandas.
3. PII sweep → `narratives.text_redacted`, the only text used downstream.
4. Company canonicalization: normalize → fuzzy block → manual review of the top
   300 by volume.
5. Taxonomy crosswalk; assign `product_family`.

## Tech

| Choice | Why this one |
|---|---|
| **DuckDB `read_csv_auto`** | Reads a 1.41 GB gzipped CSV directly. Pandas would need the whole thing in memory for no benefit. |
| **`rapidfuzz`** for company names | Fast, but used only to *propose* — see findings. |
| **Regex PII sweep** with per-pattern counts | `redaction_stats` records counts per pattern per run, because a single runaway regex is invisible in an aggregate. |

## Acceptance

> Row counts reconcile; narrative coverage computed; top-300 mapping committed;
> the 2017 taxonomy discontinuity visible pre-crosswalk and gone post-crosswalk.

**Met**, with one finding that changed the design (below).

## Findings

**16,900,994 rows loaded.** 5,911 unkeyed rows dropped (0.035%), with
reconciliation that fails above 0.1%. 3,830,002 narratives — 22.66% coverage.

**There are two taxonomy restructurings, not one.** The spec named the 2017
boundary. The 2023 boundary turns out to be larger, and two of its changes are
*splits* rather than renames — so a crosswalk keyed on `product` alone cannot
express them. It keys on `(product, sub_product)`. 34 rules, 0 uncovered labels,
and all 12 product families have zero empty months inside their active range.

**Fuzzy matching proposes and never merges.** `token_set_ratio` scores **100** on
`CREDIT ACCEPTANCE` vs `AMERICAN CREDIT ACCEPTANCE` — two different companies.
Merging on that score would silently fuse their complaint streams, and every
company-level signal downstream would be about a company that does not exist.
Only exact normalized matches merge automatically; fuzzy output is a review
queue.
