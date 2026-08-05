# ENGINEERING_NOTES

Running log. Append per phase. Decisions, failures, and things that were harder than expected.
This file is read by future-you and by anyone evaluating whether the work was real.

---

## Standing rules

1. Update the relevant doc **before** starting the next phase.
2. Every stage asserts on its own output. A stage that "finished" without valid output is a
   defect, not a pass. (See §Known traps — this failure mode has bitten before.)
3. Fix seeds. Record `run_id`, git SHA, params, input/output row counts for every run.
4. Honest numbers only.

---

## Known traps (pre-registered — check for these actively)

### T1 — Silent success
A pipeline stage completes, writes 0 rows or all-null rows, and reports success. Downstream
stages read the empty table and also "succeed." The pipeline is green and produces nothing.

**Countermeasure:** every stage ends with explicit assertions on row count, null rate, and
distribution shape. Compare output row count against a config-declared expected range. Fail
loudly. Add a `--strict` mode that is on by default in CI.

### T2 — Campaign contamination
The top signal is a credit-repair template. Because it is huge, coherent, fast-growing, and
novel-scoring, it passes every automated check.

**Countermeasure:** read the top 10 signals by hand after every detection run. Every time. The
statistics cannot tell you this and no automated test will.

### T3 — Leakage via cluster definitions
Clustering once on the full corpus and filtering signals by date. Invisible in metrics, fatal to
the result.

**Countermeasure:** `clusters.as_of` + the anti-leakage test suite. Never bypass, even for a
"quick check."

### T4 — Threshold tuning on the backtest set
Adjusting `min_cluster_size` or the novelty threshold until the backtest looks good.

**Countermeasure:** freeze `config.py` before Phase 6, or hold out 1/3 of actions. Record which
was done.

### T5 — Ground-truth expansion after seeing results
Adding enforcement actions that the system happened to catch.

**Countermeasure:** git SHA of `enforcement_actions.csv` recorded in `EVALUATION.md` and
verified to predate the first detection run, by CI.

### T6 — Company merge errors
An over-aggressive fuzzy match merges a subsidiary into a parent, or two unrelated companies
with similar names. Silently corrupts every company-level statistic.

**Countermeasure:** manual verification of the top 300. Precision over recall on merges.

---

## Log

### Phase 0 — Scaffold
_Date:_ 2026-08-03

_What was built:_ repo skeleton, `db/schema.sql` (22 tables, verified to apply),
`src/config.py` (frozen + fingerprinted), `src/db.py` (run registry),
`src/checks.py` (stage assertions), `src/ids.py`, three pure normalization
modules (`pii`, `text`, `company`), `src/pipeline.py`, `src/ingestion/download.py`,
90 tests, Makefile, ruff + GitHub Actions CI.

_Decisions:_

- **Silent success is structural, not disciplinary.** `db.run()` is a context
  manager that raises `SilentSuccess` if the stage body exits without calling
  `finish(output_rows=...)`. You cannot accidentally record a stage that
  produced nothing as a success.
- **The registry writes on its own DuckDB cursor.** Verified empirically that
  `con.cursor()` has an independent transaction context: a stage that opens a
  transaction and fails rolls back its own output but not its failure record.
  Had the runs row shared the stage transaction, the provenance table would
  only ever have logged successes — T1 wearing a disguise.
- **`config_hash` on every run.** Trap T4's countermeasure was "freeze
  `config.py` before Phase 6". A sha256 over the frozen config, recorded per
  run, makes the freeze checkable rather than promised.
- **Card detection uses a Luhn checksum.** A 13-digit account number and a
  13-digit Visa are indistinguishable by shape; the first realistic test
  narrative caught the card rule swallowing an account number. Both are
  redacted either way — Luhn decides which counter moves, which is the point of
  keeping per-pattern counts at all.
- **Requirements are split three ways** (base / dev / ml). Phase 0–1 do not need
  torch, and CI should not spend six minutes building it.

_Surprises:_

- DuckDB enforces NOT NULL on primary-key columns, so `cluster_timeseries`'s
  specified NULL "all companies" marker row could never have been inserted.
  This would have surfaced at Phase 5 — after clustering, after hours of
  embedding. Found by reading the schema against the docs before writing any
  loader. Now a regression test.
- The CFPB bulk CSV is live and healthy: 1,408,530,128 bytes compressed,
  `Last-Modified: Sun, 02 Aug 2026`, i.e. updating daily as documented. The
  `.csv.gz` variant 404s; only `.csv.zip` exists. `PROJECT_SPEC.md` §5.4
  availability check: **passed 2026-08-03**.

_Gate:_ n/a (Phase 0 is not a gate). Acceptance `make init && make test`
verified from a clean clone.

### Phase 1 — Ingestion & normalization
_Snapshot:_ 2026-08-03, sha256 `841c146eac4400dd7957403c4c2e7ddb10048a9524f58d071c12b357e199d826`,
1,409,256,676 bytes compressed / 9,048,372,819 uncompressed.

_Actual corpus size:_ 16,906,905 CSV rows → 16,900,994 loaded (75 s).
_Narrative coverage fraction:_ 0.2266 all-time, 0.2312 since 2015.
_Date range:_ 2011-12-01 .. 2026-08-03. 21 products, 178 issues, 0 duplicate IDs.

_Three things the real data forced:_

1. **`parallel = false` on `read_csv`, mandatory.** DuckDB's parallel CSV reader
   raises `NotImplementedException` on a full read of this file — narratives
   contain newlines inside quoted fields, so it cannot pick safe split points.
   Not a tuning knob; the load simply does not run without it.
2. **5,911 rows (0.035%) have no `Complaint ID`**, all recent. No primary key
   means nothing downstream can reference them, so they are dropped — but the
   loader reconciles CSV rows against loaded rows and fails above
   `Expectations.max_dropped_fraction` (0.1%), so the drop cannot go silent.
3. **DuckDB cannot read a zip member.** It reads gzip natively, so the snapshot
   is restreamed zip → gzip (1.41 GB) rather than extracted (9 GB). The zip is
   kept as the reproducibility anchor.

_Taxonomy crosswalk edge cases:_ **there are two restructurings, not one.** The
2023-08-24/25 boundary is at least as large as the 2017 one — 11.4M rows sit
under the post-2023 credit-reporting label alone. Two changes are splits rather
than renames (`Consumer Loan` → vehicle/payday; `Credit card or prepaid card` →
credit card/prepaid), so the crosswalk must key on `(Product, Sub-product)`, not
`Product` alone. Full table in `DATA.md` §3.4.

_Companies requiring manual merge:_ 8,042 raw strings → 7,991 canonical. Only
51 merged, all by **exact match after normalization** (`ONEMAIN FINANCE`,
`FLAGSTAR BANK`, `SETERUS`, …). Fuzzy similarity proposes, never merges, and
the review queue shows why: `token_set_ratio` scores **100** on
`CL` vs `MICROBILT PRBC FORMERLY CL VERIFY` and on `CREDIT ACCEPTANCE` vs
`AMERICAN CREDIT ACCEPTANCE`, and **90** on `CCS FINANCIAL SERVICES` vs
`BMW FINANCIAL SERVICES`. Auto-merging above any threshold that catches the
real duplicates also fuses unrelated companies — trap T6, with no downstream
check that would ever notice. 30 candidates written to
`data/ground_truth/company_merge_review.csv` for a human; decisions go in
`company_canonical_manual.csv`.

_Phase 1 results (2026-08-03):_ 34 crosswalk rules, **0 uncovered labels**;
16,564,967 complaints (≥ 2015-01-01); 3,830,002 narratives. Wall time 34.5 min,
dominated by the PII pass. Family volume continuity: **all 12 families have 0
empty months inside their active range**, which is the ROADMAP Phase 1
acceptance criterion — a family with a gap is the signature of a label
vanishing at a schema boundary.

_A check that was vacuous until real data ran:_ `redaction_rate_min` was 0.0,
so the redaction check would have passed with the PII sweep switched off
entirely. Measured baseline is 0.00781 mean redactions per narrative and 0.296%
of documents touched (low because CFPB already masks aggressively and this is
the *secondary* sweep). Floor raised to 0.002 and a document-fraction check
added, so drift is now actually detectable per `DATA.md` §6 item 3.

_Latent trap found:_ `apply_schema` uses `CREATE TABLE IF NOT EXISTS`, which is
idempotent but silently skips a table whose *definition* changed — the symptom
was a BinderError three stages later about a column `schema.sql` clearly
declares. `db.check_schema_drift()` now diffs the live database against a
throwaway in-memory apply of `schema.sql` and fails at bootstrap with
instructions. First migration lives at `db/migrations/001_*.sql`.

### Phase 2 — Dedup & campaign detection [GATE]
_Run:_ `1785889953195-a400c7b1`, 2026-08-04, git 7d3c7f5.

_MinHash threshold chosen and why:_ 0.88. Swept once on the labelled pairs from
the original 0.85 (`d9ead66`) and frozen since. 128 permutations give SE ≈ 0.088,
so pairs with true Jaccard just under the bar clear it roughly half the time;
0.88 buys back most of the resulting over-merge. **The labels were not
re-derived when the threshold moved** — they stay at the 0.85 exact-Jaccard
reference, so precision keeps measuring over-merge against a fixed bar instead
of against whatever the detector currently does.

_Precision / recall:_ **precision 0.9477, gate requires 0.95 — FAIL.**
tp/fp/fn/tn = 145 / 8 / 36 / 111 over 300 pairs. Recall 0.8011 overall, 0.8675
over the 151 pairs at or above the detector's own 0.88 threshold. 22 pairs were
merged despite true Jaccard below 0.88 — MinHash overestimating near the bar,
the effect `d9ead66` identified.

_What star clustering actually cost:_ **21 pairs of recall, bought with 7 fewer
false merges.** Measured directly rather than inferred: of the 163 eval pairs
carrying a verified edge in `dup_pairs`, star clustering puts **23 in different
groups** (21 of them labelled `dup`) because A was admitted to one seed's star
and B to another's. Union-find cannot do that — every verified edge lies inside
one component — so its group recall on this set would have been 166/181 =
0.9171 against star's 145/181 = 0.8011.

    union-find, threshold 0.88   precision 0.9112   recall ceiling 0.9171
    star clustering              precision 0.9477   recall         0.8011
    largest group   84,657 -> 49,457
    in groups >1000  1.8M   -> 702,032

The false-merge side reconciles with `d9ead66`'s split of "6 direct + 9
transitive-only": star removed 7 of the 9 transitive false merges and left 2.
Those 2 survive because **star groups are 2 hops wide between members, not 1** —
13 of the 153 merged pairs have no verified edge between them at all and are
together only by way of a shared seed. "Bounds group diameter at one hop" means
one hop *from the seed*; two arbitrary members are two.

An earlier version of this note claimed star clustering cost no recall, on the
grounds that group recall at threshold (0.8675) matched the pairwise recall from
the sweep (0.8674). Those are different denominators — 131/151 and something
over 181 — that agree to four decimals by coincidence. The trade is real and it
is the honest argument for the change: 21 pairs of recall is a fair price for
removing a 84,657-member chain, and unlike "free" it is checkable.

_Gate passed:_ **N.**

_Campaign-flagged fraction by family:_

    credit_reporting  30.70%   debt_relief    1.47%
    money_service     28.66%   student_loan   0.78%
    debt_collection   14.71%   vehicle_loan   0.76%
    credit_card        4.78%   personal_loan  0.16%
    bank_account       3.00%   mortgage       0.07%
                               prepaid_card   0.01%

Credit reporting is highest and mortgage is 438× lower, which is the direction
`METHODOLOGY §2.4` requires. Corpus boilerplate baseline 0.3625.

_What the flagged campaigns actually looked like on reading:_ 20 read, largest
first. **All 20 are unambiguously templated** — the criterion that would have
forced a stop-and-fix is satisfied. The largest group, 49,457 members, is one
template and not an artifact of `MAX_BUCKET_PAIRS` anchoring: every member opens
"In accordance with the Fair Credit Reporting act. The List of accounts below
has violated my federally protected consumer rights…", the canonical
credit-repair letter. Others: "estoppel by silence, Engelhardt V. Gravens",
"HI I AM SUBMITTING THIS WITHOUT ANY INFLUENCE AND THIS IS NOT A THIRD PARTY".

Two encouraging things in that set. The Cash App (22,002) and Zelle (9,971)
campaigns score `boilerplate = 0.00` — no statutory citation anywhere — and are
flagged on burstiness and length variance alone, so the feature set is not just
a credit-repair statute detector. And their burstiness is enormous (4,083 and
1,173 against a threshold of 3.0): both reference "the recent CFPB lawsuit",
i.e. filings that spike on a news event. That is what pushes `money_service` to
28.66%, second only to credit reporting.

_The finding that is not in the acceptance criteria:_ **the 20 largest unflagged
groups are also obviously templated.** A 24,507-member group whose members all
open "My credit reports are inaccurate. These inaccuracies are causing creditors
to deny me credit…" is not 24,507 consumers writing independently. Four of the
20 are the same credit-repair service's family ("I have a goal of getting a
house as soon as possible but the stuff on my credit report will really put me
in trouble"), and a 7,309-member `vehicle_loan` group is the *same* template as
the largest flagged campaign, sitting in a different family.

Every one of the twelve largest unflagged candidates scores exactly **2**
signals against a bar of 3, and the two dead signals are structural, not
threshold choices:

1. **`submitted_via_concentration` is a constant, not a weak signal.** All
   **3,830,002** narrative-bearing complaints have `submitted_via = 'Web'` —
   every one, in every family. CFPB only collects narrative consent on the web
   form, so conditioning on "has a narrative" conditions on "arrived by web".
   The corpus as a whole is 96% web and 6 channels; the population this detector
   actually sees is 100% web and 1. Family baseline HHI is exactly 1.0 in all 12
   families, the rule asks a group to exceed `1.5 × baseline`, and an HHI cannot
   exceed 1.0. It fired **0 times in 2,841** unflagged candidates and cannot
   ever fire. `METHODOLOGY §2.2` specifies six campaign features; there have
   only ever been five, so `campaign_min_signals = 3` has always been 3-of-5.
2. **`company_concentration` is backwards for tri-bureau blasts.** The family
   baseline is 0.2674 and the bar is 0.401. A template mailed to all three
   bureaus scores ≈ 0.33 — *below* the bar. The single most characteristic
   credit-repair behaviour reads as unconcentrated. The baseline is computed
   over every complaint in the family including the campaigns themselves, so
   campaign traffic sets the norm that campaigns are then measured against.

Bounded blast radius, and worth stating: `dup_groups` collapses these regardless
of the flag, and clustering consumes representatives only (`METHODOLOGY §2.3`),
so a 24,507-member unflagged template still enters Phase 4 as one document. The
campaign flag is the second layer — exclusion from signal detection — not the
first. That is why this is recorded as a defect to fix rather than as the reason
the gate failed.

_Both defects fixed, and the fix did not solve the problem next to it_
(run `1785896176289-fda4ce84`, git 36abc79): `submitted_via_concentration`
dropped in migration 003; the concentration bar now compares against
`expected_hhi(H, n) = H + (1-H)/n` instead of raw `H`.

    flagged campaigns   3,787 -> 3,376        credit_reporting share  30.70% -> 29.91%
    state_concentration fire rate among unflagged  84.7% -> 68.6%

**The large templated groups are still unflagged, all at exactly 2 signals.**
The 24,507-member group fires burstiness (37.9) and length_cv (0.011), and
cannot reach 3: it cites no statute (`boilerplate = 0.00`), its state HHI 0.071
is at chance for its size, and its company HHI 0.334 is what mailing all three
bureaus looks like. 2,429 of the 3,252 unflagged candidates sit at exactly 2.
This was predicted before the run and is recorded because it was: the two fixes
were made because each is wrong on its own terms, not because either was
expected to flag these.

_The design question that is actually left,_ and it is not a threshold to
tune: **group size is the strongest evidence of a mass filing and is not a
signal at all.** It appears only as `campaign_min_size = 20`, a floor. A group of
24,507 near-identical narratives is a mass filing by definition — `dup_groups`
membership already established the near-identity — and "does it also cite a
statute?" should not be able to veto that. Options, none picked:

- make size a signal, so a large group needs 2 of the other 4;
- make size sufficient above some bound, no signal count;
- lower `campaign_min_signals` to 2 (would flag 2,429 more candidates, most of
  them small, on weak evidence);
- leave it, and handle template inflation in Phase 5 by counting
  `n_supporting_groups` rather than complaints — the column already exists.

The last is worth weighing seriously: dedup already collapses each of these to
one representative, so nothing reaches clustering inflated. The exposure is
confined to signal detection counting complaints instead of groups.

_Determinism check (README standing rule 3):_ two independent full runs on the
same snapshot produced identical output — 1,352,065 exact pairs, 2,477,937
representatives, 41,161,012 LSH candidates, 12,935,450 verified, 1,883,062
groups, and the same 145/8/36/111 confusion matrix. The campaign fix changed
only the campaign layer, as intended.

_What the gate still needs, in order:_

1. **Hand-adjudicate the 8 false positives.** Read as template variants that the
   exact-Jaccard proxy misses because CFPB's own `XXXX` redaction runs differ in
   length, which moves character-5-shingle overlap a long way while changing
   nothing a reader would call a difference. **Not applied.** Relabelling only
   the pairs the detector merged would raise precision to 1.000 by construction,
   which is the shape of the thing trap T5 exists to prevent. The check that
   made this decidable: among the 111 pairs labelled `not_dup` and left
   unmerged, exactly **1** has 50%+ of the shorter narrative verbatim-identical
   to the other, against 3 of the 8 disputed merges and 91 of the 145 agreed
   merges. That is a lower bound, not a measurement — a shared-prefix metric
   misses any template that varies early, and it scored 5 of the 8 disputed
   merges below 50% even though reading them says otherwise. So: **the proxy is
   not shown to be broken on the negatives**, by a metric that would miss most
   templated ones. The adjudication is a human's to make and is recorded here
   unapplied either way.
2. Fix the two dead campaign signals, then re-read.
3. `data/interim/dedup_near_misses.csv` — 300 rejected LSH candidates sampled
   during the run, the only recall figure here the eval file could not have
   produced. Stratified, because the pooled 4.7% averages two different
   populations and hides where the loss is:

       est_sim in [0.85, 0.88)   12 / 50   24.0% were true duplicates
       est_sim in [0.58, 0.85)    2 / 250   0.8% were true duplicates

   The loss is entirely in the sliver just under the bar, which is what MinHash
   estimation error predicts and what raising the threshold from 0.85 to 0.88
   bought the precision with. Nothing is being lost further down.

### Phase 3 — Embedding & index
_Model, throughput, wall time:_
_Nearest-neighbour spot checks:_

### Phase 4 — Clustering & novelty [GATE]
_Params per family:_
_ARI disjoint halves:_
_Noise fraction:_
_Label-ablation AUC:_
_Could you name 15 random clusters?_
_Gate passed:_ Y / N

### Phase 5 — Signal detection
_Negative-control (shuffled labels) false-alert rate:_
_Expected under FDR α:_
_Top 20 signals, first impressions:_

### Phase 6 — Ground truth & backtest [GATE]
_Actions curated / usable / excluded:_
_enforcement_actions.csv frozen at SHA:_
_Wall time for one full-refit cutoff:_
_Anti-leakage tests passing:_
_Gate passed:_ Y / N

### Phase 7 — Baselines
_B1 implementation notes (must use identical statistical machinery):_

### Phase 8 — LLM layer
_Determinism test result:_
_Label agreement rate on 50 verified:_
_Observed LLM failure modes:_
_RAG Recall@10:_
_Cost incurred:_

### Phase 9 — Evaluation
_Headline result:_
_Did HarmScope beat B1:_
_Missed actions by failure category:_
_Top false alerts, what they actually were:_

### Phase 10 — Interface
_Shipped React or Streamlit:_
_What got cut:_

---

## Open questions

- Does clustering within `product_family` fragment cross-product harms too aggressively? The
  `related_clusters` linking may not be sufficient.
- Is annual-cutoff refit too coarse for actions filed early in a year? Consider quarterly
  cutoffs for a subset if wall time allows.
- Company public responses are optional and sparse — is the coverage high enough to display
  meaningfully in the evidence panel?

## Reversed decisions

_Record anything decided in the docs and later changed, with the reason. A doc that was never
wrong was never load-bearing._

### 2026-08-03 — Ground-truth window 2016–2024 → 2017–2024

`EVALUATION.md` §1.2 evaluates each action against the most recent annual cutoff
*strictly before* its `filed_date`, with cutoffs starting 2017-01-01. A 2016
action has no such cutoff, so the stated 2016–2024 window contained a year of
unevaluable rows. Narrowed the window rather than adding a 2016-01-01 cutoff,
because a model trained on 2015 alone would contribute near-certain misses for
reasons unrelated to method quality. Costs a year of candidates against the
≥ 20-usable requirement; leaves the gap to B1 — the actual contribution —
unchanged, since every system runs the identical harness.

### 2026-08-03 — Determinism test scoped to `signals`

`LLM_LAYER.md` §1 claimed deleting `src/llm/` leaves `signals` **and**
`backtest_results` byte-identical. But `EVALUATION.md` §1.3 shows the adjudicator
LLM labels, so LLM output reaches `backtest_results` through a human by design.
Scoped the claim to `signals` and stated adjudication as a deliberate
human-in-the-loop step. Overclaiming here would have undermined the one
architectural guarantee the project actually has.

### 2026-08-03 — No `--strict` flag

T1's countermeasure specified "a `--strict` mode that is on by default in CI".
`src/checks.py` always raises instead. A flag that is always on is a config for
a value that never changes; the behaviour is identical with less code. Add the
flag if a real need to run non-strict ever appears.

### 2026-08-03 — `run_id` is not a ULID

`ARCHITECTURE.md` §4 says ulid. `src/db.py` uses
`{epoch_ms:013d}-{8 hex chars}`, which has the two properties that are actually
load-bearing (lexicographic time-sortability, collision-freedom) with no
dependency. Swap in `python-ulid` if the canonical 26-character format is ever
needed by something external.

### 2026-08-03 — `dup_groups` and `campaigns` are run-scoped; `dup_pairs` split out

Documenting the per-cutoff refit split (above) exposed that two dedup tables
still had cutoff-independent keys: `dup_groups` was `PRIMARY KEY (complaint_id)`
while carrying a cutoff-dependent `is_representative`, and `campaigns.campaign_id`
was a bare VARCHAR while `campaigns.as_of` said it is regenerated per cutoff.
Eight refits, one slot each — the same collision `cluster_id` had.

Split along the line the refit table already draws: `dup_pairs` holds pairwise
similarity (computed once, date-independent), `dup_groups` holds connected
components and representative selection keyed `(run_id, complaint_id)`.
`campaign_id` now goes through `src/ids.py` like `cluster_id`.

Caught before Phase 2 wrote a single row. After that it would have been a
migration plus a full re-run.

### 2026-08-03 — `signals.company_id` uses the `'__ALL__'` sentinel too

`cluster_timeseries` got the sentinel; `signals` was left nullable. A join
between them on `company_id` — the natural Phase 5/6 query — would have returned
nothing for exactly the cluster-level rows. Not an error, just missing alerts.
Regression test: `test_cluster_level_signal_joins_its_timeseries_total`.

### 2026-08-03 — Embeddings and MinHash are not refit per cutoff

`EVALUATION.md` §1.1 said "rebuild the entire pipeline" per cutoff. The encoder
is a fixed pretrained checkpoint (§5 item 3) and MinHash similarity is pairwise,
so neither can leak. Both are now computed once and date-filtered; representative
selection, campaign detection, clustering, novelty, and signals are refit.
Roughly 8× off the most expensive stage with the leakage guarantee intact — and
the anti-leakage suite tests the guarantee directly, so the saving does not
depend on this reasoning being right.
