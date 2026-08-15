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

300 narrative pairs, stratified 100 obvious duplicates / 100 hard near-duplicates
/ 100 unrelated (`docs/METHODOLOGY.md` §2.4).

**These are not hand labels.** `label` is the exact character-5-shingle Jaccard
thresholded at 0.85, and `label_source` records that per row. It is a genuinely
independent reference for a detector that sees only a 128-permutation estimate
and an LSH bucketing, but it does not answer "is this the same filing?", which
is the judgement ROADMAP Phase 2 asks for. The frozen bar stays at 0.85 even
though `jaccard_threshold` moved to 0.88, so precision keeps measuring
over-merge against a fixed reference rather than against the detector's current
operating point.

Hand adjudication of the pairs the detector gets wrong is recorded in
`dedup_eval_adjudicated.csv` when it exists — see `ENGINEERING_NOTES.md` Phase 2.

## Not ground truth

`dedup_near_misses.csv` is **detector output**, regenerated on every dedup run,
and lives in `data/interim/`. It is the sample of LSH candidates the verifier
rejected, which recall needs as a denominator and which nothing persists
otherwise. It must never move into this directory: everything here is frozen
before detection runs, and a file that rewrites itself cannot be.

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

## `rag_eval_questions.csv`

This Phase 8D benchmark is a frozen, ID-only set of 30 synthetic analyst
questions: five each for mechanism, actors/preconditions, consumer consequence,
time sequence, company response, and unanswerable queries. Answerable rows list
every hand-marked relevant complaint ID as an ascending semicolon-separated
set; unanswerable rows contain no relevance IDs. Narrative excerpts remain in a
gitignored `data/interim/` authoring worklist and never enter this directory.

Before first freeze, a human author completes the private worklist and a human
privacy reviewer checks each synthetic question for semantic paraphrase,
recording `privacy_reviewed=yes`. Import also rejects any normalized eight-token
sequence shared with a scoped redacted narrative. That automated check detects
direct copying; it does not replace human paraphrase review.

The first committed version must record its author ID, privacy-reviewer ID, and
freeze commit here. Those fields are currently **pending** because no human has
completed the authoring worklist; an agent-generated draft is not ground truth.
After retrieval metrics have been viewed, neither questions nor relevant IDs
may be edited in place. Corrections require a new manifest version, a new freeze
commit, and preservation of results produced from the prior manifest hash.

## Also expected here

- `company_canonical_manual.csv` — top-300 hand review (`docs/DATA.md` §3.5)
- `taxonomy_crosswalk.csv` — old → new labels (`docs/DATA.md` §3.4)
