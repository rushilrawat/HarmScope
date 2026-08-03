# Ground truth

Hand-curated, committed, and **frozen before any detection run**. Everything
else under `data/` is derived or downloaded and is gitignored.

## `enforcement_actions.csv`

CFPB public enforcement actions, **2017-01-01 → 2024-12-31**. Target ≥ 20
usable, ideally 30–40. Column reference: `docs/DATA.md` §4.

Window notes:

- The lower bound is the first backtest cutoff. An action filed before
  2017-01-01 has no annual cutoff strictly preceding it and cannot be
  evaluated (`docs/EVALUATION.md` §1.2). Keep such rows with `usable=false`
  and `exclusion_reason=pre-first-cutoff` so the exclusion is visible.
- The upper bound is CFPB's 2025 posture change. After 2024, absence of an
  enforcement action does not imply absence of harm, so 2025+ cannot be used
  as negative labels (`docs/PROJECT_SPEC.md` §5).

Selection protocol (`docs/DATA.md` §4, trap T5):

1. Select actions **before** running any detection.
2. Include actions you expect to fail. A set of only easy cases inflates results.
3. Commit and freeze. Record the commit SHA in `docs/EVALUATION.md`.
4. Adding actions after seeing results is p-hacking. CI verifies the freeze SHA
   predates the first detection run.

## `dedup_eval_pairs.csv`

300 hand-labelled narrative pairs, stratified 100 obvious duplicates / 100 hard
near-duplicates / 100 unrelated (`docs/METHODOLOGY.md` §2.4).

**Ids only — never narrative text.** This file is committed; `docs/DATA.md` §6
forbids narrative text in git. Join the text from DuckDB at evaluation time.
`tests/test_ground_truth.py` enforces this, and it cannot be undone once a
narrative lands in git history.

| Column | Meaning |
|---|---|
| `complaint_id_a`, `complaint_id_b` | the pair, ids only |
| `label` | `dup` / `not_dup` |
| `stratum` | `obvious` / `hard` / `unrelated` |
| `notes` | short adjudicator note, no narrative quotes |

## Also expected here

- `company_canonical_manual.csv` — top-300 hand review (`docs/DATA.md` §3.5)
- `taxonomy_crosswalk.csv` — old → new labels (`docs/DATA.md` §3.4)
