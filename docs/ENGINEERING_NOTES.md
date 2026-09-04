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

_Adjudication, and the measurement gap it exposed (2026-08-05):_

All 100 `hard` pairs were read blind under the rule pre-registered in
`METHODOLOGY §2.4.1` — shuffled, with the detector's decision, the proxy label
and `true_jaccard` withheld. **All 100 are instances of one template.** Not 82,
which is what the proxy says. The stratum was drawn from `dup_pairs`, i.e. from
pairs that had already cleared LSH banding and Jaccard verification, so every
pair in it is a genuine near-duplicate and **the stratum contains no true
negatives at all**. Its 18 `not_dup` labels are all wrong.

    against adjudicated labels   precision 1.0000 (153/153)   recall 0.7650

That 1.0000 is not the good news it looks like, and the merge audit is why.
`dedup_eval_pairs.csv` samples pairs *with a verified edge*. Star clustering
admits members to a **seed**, so two arbitrary members of a group need no edge
between them — in the 49,457-member star, 49,456 of roughly 1.2 billion
member-pairs are edges. Sampling 40 same-group pairs the way the corpus actually
holds them:

    3 of 40 have a verified edge; 37 are seed-mediated.
    The eval set can only ever sample the first kind — 8% of real merges.

So every precision figure in this section, including the 0.9477 that failed the
gate and the 1.0000 that passes it, was computed on 8% of the merge population,
and no number computed from that file could have said so. Reading 12 of the
seed-mediated merges by hand: **all 12 are the same template**, including pairs
from the 5,683- and 4,139-member groups. Star clustering's 2-hop merges hold up
where the eval set cannot look. `gate --merge-audit N` makes this repeatable.

_Gate verdict:_ **PASS**, on the criteria as restated in ROADMAP Phase 2 — with
the adjudication recorded as `model_adjudicated_blind`, not hand-labelled. See
the Reversed decisions entry; this is a changed criterion, not a satisfied one.

_What the gate still needs, in order:_

1. ~~**Hand-adjudicate the 8 false positives.**~~ Done blind across all 100 hard
   pairs, 2026-08-05; all 8 were template variants, as read. Original note kept
   below for the reasoning that made it decidable. Read as template variants that the
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
_Run:_ `1785913348…` (2026-08-05, git 8b5eab7). **Accepted on the dev model; see
the caveat at the end.**

_Model, throughput, wall time:_ `all-MiniLM-L6-v2` (384-d) on MPS.
2,477,937 distinct narratives, 3.81 GB memmap, FAISS `IndexFlatIP` with
2,477,937 vectors, 3,830,002 complaint_ids mapped and **0 unmapped**. Wall time
~131 min across an interrupt: 1,700,096 texts at 283/s, then 777,841 at 418/s
after resuming. The 1.5× difference is unexplained and most likely thermal — the
machine was cold on restart — which is worth knowing before any throughput
number here is treated as a property of the model.

_Model choice, measured rather than assumed:_

    all-MiniLM-L6-v2   386 texts/s    2.1 h for 2.48M
    bge-base-en-v1.5    41 texts/s   16.8 h for 2.48M

Also tested length-sorting the corpus to cut padding waste before accepting the
bge figure: **7%**, because sentence-transformers already sorts within each
`encode` call. Not worth scatter-writes and a harder resume, so not built.

_What gets encoded, and why it is not what METHODOLOGY §3 says:_ the unit is a
distinct `text_hash`, not a dup-group representative. §3 says "representatives
only", but representative selection is refit per cutoff while the 2026-08-03
reversed decision says embeddings are computed once and date-filtered — both
cannot be true of a representative-keyed memmap. An embedding is a pure function
of its text, so text is the honest key: a superset of every cutoff's
representatives (2,477,937 against 1,883,062 for one cutoff), never refit, and
leakage-immune for the same reason MinHash is.

_Nearest-neighbour spot checks:_ 10 narratives, 5 neighbours each,
**10/10 topically correct**. Three observations worth more than the pass:

1. **It recovers company identity through redaction.** A USAA deposit-hold
   complaint returns five USAA deposit-hold complaints; a Firstmark forbearance
   complaint returns four explicit Firstmark forbearance complaints; a Discover
   identity-theft dispute returns three Discover ones. The names survive because
   CFPB redacts inconsistently, and the encoder is reading what is left.
2. **Neighbours cross `product_family` constantly.** The same credit-repair
   template appears under `credit_reporting` and `debt_collection`; a title-loan
   complaint returns neighbours split across `personal_loan` and `vehicle_loan`.
   This is direct evidence for the open question about whether clustering within
   family fragments cross-product harms — it does, and `related_clusters`
   (METHODOLOGY §4.2) is load-bearing rather than a nicety.
3. **Cosines of 0.99 are template pairs that survived dedup.** Star clustering
   splits a template into several groups by design (Phase 2, 21 pairs of recall
   traded for 7 fewer false merges), and those splits are exactly what shows up
   at the top of a neighbour list. Phase 4 will see them as very dense regions.

_Idempotence:_ verified. A second run encodes 0 and reports `resumed at n_total`.
Resume was also exercised for real: the run was interrupted at 1,700,096 and
restarted from the checkpoint with no loss.

_Caveat — this is the dev model._ METHODOLOGY §3 names `bge-base-en-v1.5` as the
default and MiniLM for iteration. Phase 4 is a gate needing heavy iteration
(ARI across sample sizes, disjoint halves, label-ablation AUC), so it is
developed against MiniLM. **The Phase 4 gate may not be declared on these
embeddings** — bge-base is a ~16 h encode and must run before any gate number is
recorded as final.

_Two bugs the smoke test caught, both silent:_

1. `--limit 3000` encoded 3,000 rows and then mapped all 3,830,002 complaints to
   row indices computed over all 2,477,937 texts. Almost every index pointed
   past the end of a 3,000-row memmap; no error anywhere, and a downstream read
   would have returned whatever numpy found at that offset. Bounding `row_idx`
   fixes the dangling indices and converts it to a silent *drop*, so `build_map`
   returns the unmapped count and a full encode fails when it is nonzero.
2. `encode_batch` passed `batch_size=len(flat)` to `model.encode`, making a
   512-text batch one forward pass over 1,024 sequences of 512 tokens. That OOMs
   Metal on an M3 Pro, and the configured batch size is 256 — **the full run
   would have died**. GPU memory and checkpoint spacing are now separate knobs.

_A third, found by the commit security review:_ the `--limit` marker added to
`pipeline runs` read `$.limit`, but `db.run` nests its payload as
`{"config", "params"}`, so the correct path is `$.params.limit`. The marker
never fired. It passed its test because the test fabricated a flat params shape
the pipeline has never written — the test and the code were wrong about the data
in the same way, which is the failure mode a test is supposed to prevent. Now
read with `json_extract` in SQL, so there is one parser and the path is visible
in the query.

### Phase 4 — Clustering & novelty [GATE]
_Run:_ `1785937994715-d5418cb5` (2026-08-05, git 0c9e16c). **On the dev model —
see the Phase 3 caveat; the same one applies here.**

_Params per family:_ config as declared — UMAP(30, 10, min_dist 0, cosine),
HDBSCAN(min_cluster_size 50, min_samples 10, leaf), fit sample 500k,
`assign_max_distance` 0.35. `min_cluster_size` is the declared constant 50 for
every family, **not** `METHODOLOGY §4.1`'s "∝ family size". The config is what
`runs.config_hash` fingerprints, so it is the authority; the prose is
aspirational and should be corrected or implemented deliberately, not silently.

    family              reps    clusters  fit-noise  assigned  coherence
    credit_reporting   742,298      959      68.6%     96.6%     0.836
    debt_collection    329,311      425      71.5%     93.8%     0.792
    credit_card        205,327      338      64.6%     93.2%     0.784
    bank_account       192,827      293      68.5%     93.7%     0.789
    mortgage           144,577      220      67.3%     91.8%     0.772
    money_service       84,292      182      61.3%     91.8%     0.809
    vehicle_loan        57,395      118      63.3%     89.2%     0.764
    student_loan        60,957      111      68.5%     92.1%     0.779
    personal_loan       39,689       92      67.1%     84.7%     0.752
    prepaid_card        20,928       61      60.8%     90.9%     0.784
    debt_relief          5,171       19      49.0%     68.7%     0.758
    other                  290        3       9.3%     64.8%     0.747

2,821 clusters over 1,883,062 representatives, 94.1% assigned, 15,257
cross-family `related_clusters` links.

_Noise fraction:_ two numbers, because they answer different questions.
**HDBSCAN fit-noise is 49-72%** — its own judgement about the sample it saw, and
the honest measure of how much of this corpus sits in a dense region at all.
**Final unassigned is 3.4-35.2%** (94.1% assigned overall) — how much is beyond
`assign_max_distance` of every centroid. The gap between them is the design
decision recorded in `cluster/fit.py`: HDBSCAN discovers, the threshold decides
membership, and it decides identically for sampled and unsampled points so that
a cluster's size does not depend on who got drawn.

_ARI disjoint halves:_ **0.5051** (credit_reporting), **0.5411** (mortgage).
Each half fit independently, both assigning a held-out set neither had seen.

    credit_reporting   346,149 / 346,149    688 vs 693 clusters   ARI 0.5051 over 47,723
    mortgage            54,216 /  54,217    121 vs 116 clusters   ARI 0.5411 over 31,780

**What that means, stated plainly:** the *granularity* and *coverage* of the
partition are highly reproducible — 688 vs 693 clusters is under 1% apart, and
assignment fractions match to the decimal (96.1% / 96.1%) — while *which cluster
a given complaint lands in* agrees only about half the time. Leaf selection cuts
the condensed tree at its finest nodes, so boundaries between hundreds of
adjacent clusters are exactly where two samples will disagree, and ARI is
unforgiving about that.

Usable, with the claim capped accordingly: a cluster here is a region of a dense
neighbourhood, not a canonical object, and nothing downstream may treat cluster
identity as stable across refits. The architecture already assumes this —
`cluster_id` is `{run_id}:...` and clusters are refit per cutoff, so the
backtest never carries an identity between runs. `METHODOLOGY §4.3`'s remedy
(raise `min_cluster_size`, or switch to excess-of-mass for fewer coarser
clusters) is available and **not taken**: changing granularity while looking at
the gate number is how T4 happens.

_Sample-size sweep (credit_reporting; the only family with ≥ 500k):_

    size      clusters  fit noise  assigned  ARI vs largest
    100,000       254     63.4%     94.0%        0.3253
    250,000       533     65.8%     95.8%        0.4967
    500,000       937     67.6%     96.6%        1.0000  (self)

Cluster count is still climbing roughly linearly with sample size at 500k, so
the partition has **not converged** — the 500k fit is a sample of the structure,
not the structure. The 250k-vs-500k figure (0.4967) lands on the independent
half-vs-half figure (0.5051), which is the consistency check that makes both
believable.

_Label-ablation AUC:_ **0.7909 mean across 11 families, bound 0.7 — PASS.**
10 of 11 pass; `vehicle_loan` fails at 0.660.

    credit_reporting 0.9171   money_service 0.8706   debt_relief   0.8425
    credit_card      0.8014   mortgage      0.8012   student_loan  0.7964
    prepaid_card     0.7848   debt_collection 0.7695 personal_loan 0.7317
    bank_account     0.7246   vehicle_loan  0.6600 (FAIL)

**It failed first, at 0.6658, and the diagnosis is the useful part.**
`dominant_label_share` divided by *surviving* labels rather than by cluster
size, so ablation shrank numerator and denominator together. A 101-member
bank_account cluster, 99% of it the hidden issue, scored novelty **0.000**: one
member survived, that member's label was trivially 100% of survivors, and the
least-explained cluster in the family read as perfectly explained. §5 says
"fraction of members" — the whole cluster — so this was conformance, not tuning.

Verified rather than asserted, because changing a metric that just failed a gate
is precisely the T4 shape: recomputing all 2,821 production novelty scores under
the fix gives a largest difference of **0.00e+00**. Zero moved. Every complaint
carries an `issue_std` (7 NULLs corpus-wide, none in a cluster), so the two
denominators coincide outside ablation and the fix cannot have been steered
toward an outcome it does not touch.

_Could you name 15 random clusters?_ **Yes, 15/15.** Read with narratives:

- Venmo accounts frozen, funds inaccessible, no reason given (454 members)
- Vanilla gift cards drained before first use (259)
- Barclays online savings locked after verification documents (130) — flagged novel
- Navient private student loans and bankruptcy discharge (188)
- Fortiva retail financing still reporting after payoff (334)

And the negative control §5 asks for: the Convergent FCRA template scores
novelty **0.141** with 87% of members on one existing label. The score is not
simply high everywhere.

_Gate passed:_ **Y**, on the four ROADMAP criteria — ARI reported, noise
reported per family, ablation AUC above 0.7, 15 clusters nameable. The ARI is
the number to carry forward as a limitation, per §4.3's instruction to report it
in the README whatever it says.

_A finding recorded rather than fixed:_ both `§5.2` guards are miscalibrated, in
opposite directions.

    novelty_score >= 0.60   1,254 / 2,821   44.5%
    coherence     >= 0.45   2,821 / 2,821  100.0%   <- excludes nothing
    persistence   >= 0.05     305 / 2,821   10.8%   <- median persistence 0.0049
    all three (is_novel)      124 / 2,821    4.4%

The coherence floor is inert: observed coherence runs 0.74-0.93, so the guard
against calling an *incoherent* cluster novel never fires. The persistence floor
does all the filtering and is calibrated for excess-of-mass, while §4.1
specifies leaf — leaf clusters are the finest nodes in the condensed tree, so
short lifetimes are structural rather than a quality signal. Both numbers were
written in Phase 0 before any data existed. Calibrating them against the
observed distribution is legitimate; calibrating them against which clusters
they admit is not, so the distribution is recorded and the change left as a
stated decision.

### Phase 5 — Signal detection
_Run:_ `1785964943190-ac0ae219` (2026-08-05, git 780e062). Dev-model clusters —
the Phase 3 caveat still applies.

    expanded    2,965,850 non-campaign complaints
    panel         257,082 cluster-level rows, 932,134 company-level
    2x2 tests      35,500 pairs with a >= 5, of 207,566 company x cluster pairs
    changepoint    12,483 series fired of 207,883
    signals        50,104 rows      alerts 20,696 at q <= 0.05
    by method      ebgm 35,500   ewma 12,067   pelt 2,537

_Negative-control (shuffled labels) false-alert rate:_ **0.0045** — 236 alerts
of 52,033 permuted tests. **PASS.**

_Expected under FDR α:_ α is 0.05, and the realized rate is an order of
magnitude below it. That is the right direction and not a coincidence: under a
*complete* null BH does not reject α of tests, it rejects almost none — the
guarantee is on the expected proportion of false discoveries among rejections,
and with no true effects anywhere the procedure should be nearly silent. A rate
close to 0.05 would have been a weaker result than it looks.

_The control failed first, at 0.1071, and found three bugs._ Committing that
sequence because the numbers are worthless without it — every one of the three
produced plausible, well-formed, entirely wrong output, and none is visible in a
normal run.

1. **The control itself was wrong.** In production a group's cluster comes from
   its representative, so every complaint in a group shares one cluster. The
   shuffle permuted labels *per complaint*, scattering each group across many
   clusters — a null for a dataset that cannot exist. Permuted at group level
   now.
2. **The contingency margins did not partition.** A 2×2 tests association only
   if every unit is in exactly one cell. One credit-repair template mailed to
   Equifax, Experian and TransUnion is three complaints against three companies
   sharing one `group_id`: the company margin counted it three times, the
   cluster margin counted distinct groups and counted it once, so
   `c = n_cluster - a` came out too small and **every PRR was inflated**. The
   unit is now the `(group, company)` pair, which partitions and is right on its
   own terms — 24,507 identical complaints against one bureau are one
   allegation, the same template to three bureaus is three.
3. **`n_supporting_groups` summed months instead of counting groups.** The gate
   whose entire job is stopping one filing from looking like many was counting
   one filing many times: a template active for two years reported 24 supporting
   groups and cleared `min_supporting_groups = 15` unaided.

    before   5,680 / 53,015 = 0.1071   FAIL
    after      236 / 52,033 = 0.0045   PASS   (24x reduction)

All three share a signature worth remembering: **a quantity defined as "distinct
groups" but computed as a sum over a partition** — of companies in one case, of
months in another. Sums over partitions are the thing to grep for in Phase 6.

_p-value calibration under the null,_ because "few alerts" and "correct test"
are different claims:

    p <= 0.5     0.5291   1.1x uniform
    p <= 0.1     0.1433   1.4x
    p <= 0.05    0.0863   1.7x
    p <= 0.01    0.0315   3.1x
    p <= 0.001   0.0082   8.2x

Near-uniform in the bulk, mildly anti-conservative in the extreme tail. That is
the known cost of the normal approximation on log ROR rather than Fisher's exact
— chosen because Fisher on 52,000 tables is the dominant cost of the phase. BH
absorbs it (0.45% realized), and ranking is on EB05 rather than on p, so the
tail excess does not drive the ordering. Upgrade path if it ever matters:
exact tests for the pairs that clear the BH threshold, which is a few hundred
tables rather than 52,000.

_Top 20 signals, first impressions:_ the novel track's leaders are dominated by
companies with **real public enforcement histories**, which is the first
encouraging sign that Phase 6 has something to find:

    EB05 1208  credit_reporting  JOHN C HEATH ATTORNEY AT LAW (Lexington Law)
    EB05 1185  credit_reporting  CHIME FINANCIAL
    EB05  955  credit_reporting  PNC BANK
    EB05  948  credit_reporting  RADIUS GLOBAL SOLUTIONS
    EB05  740  credit_reporting  FREEDOM FINANCIAL NETWORK
    EB05  674  bank_account      COINBASE            (novel)
    EB05  669  credit_reporting  CREDIT KARMA
    EB05  710  debt_collection   PNC BANK            (novel)

Two cautions on that list, both for Phase 6 rather than now. **Lexington Law is
a credit-repair firm**, so complaints naming it sit uncomfortably close to trap
T2 — its clients are the people who file templated complaints, and the campaign
flag is known to miss unflagged templates. That cluster needs reading before any
backtest counts it as a hit. And these are **company × cluster** signals, so a
company appearing twice under different families (PNC) is two findings, not one,
and the adjudication has to treat them separately.

_A structural artifact checked for and not found:_ three of the first twelve
alerts share a changepoint of 2017-07-01, which is near the CFPB taxonomy
restructuring (`DATA.md §3.4`) and would be a systematic artifact if the
detectors were keying on it. They are not — 2017-07 accounts for 216 of 14,604
changepoints (1.5%), and the most common months are 2022-02, 2022-07 and
2021-09, which track the real credit-reporting surge. The coincidence in the top
twelve was small-sample noise.

_A feasibility change, argued on correctness first:_ changepoint now skips
series below `min_supporting_groups` instead of fitting them. §6.3 cannot turn a
changepoint on such a series into an alert, so the work was already discarded —
and PELT over ~200,000 mostly-tiny series is the dominant cost of the phase,
which Phase 6 pays eight times over for its eight cutoffs. EWMA signals fell
from 30,371 to 12,067 with no alert lost.

### Phase 6b — Adjudication (2026-08-06)

_The blocker found by starting, not by reading._ `harm_summary` held listing-page
previews: median 182 characters, 18% ending in a literal ellipsis mid-sentence.
The Citibank entry said only that Citibank "is a national bank headquartered in
New York City, New York" — no conduct at all. Adjudicating a cluster against that
is not strict or lenient, it is undefined. Enriched from the detail pages;
median 182 → 1,505 characters. Only `harm_summary` changed, asserted across all
212 rows against the seven selection-bearing fields.

_Sample:_ 8 actions drawn by seed from the 55 that **both** HarmScope and B1
detected, so the same actions can later be adjudicated for both and the
difference is paired rather than confounded.

    action                     verdict   matched cluster
    Zelle / Early Warning      strong    P2P transfer fraud, explicit Zelle scams
    Equifax 2019               strong    the 2017 breach and mishandled PII
    Fay Servicing 2024         strong    forbearance and loss-mitigation requests
    Nationstar 2020            strong    loan modification for distressed borrowers
    Portfolio Recovery 2023    strong    collecting unvalidated / not-owed debt
    Bank of America 2022       none      garnishment-notice processing — nearest
                                         cluster is consumer-side garnishment
                                         via a collector, a different mechanism
    Citibank 2023              none      ECOA national-origin discrimination —
                                         nothing among 30 candidates concerns
                                         discrimination or adverse-action notices
    Santander 2018             none      misdescribed GAP auto add-on — the
                                         vehicle_loan clusters are servicing and
                                         financing, not add-on products

    strong             5 / 8 = 62.5%   Wilson 95% CI [30.6%, 86.3%]
    detect rate        50.9% unadjudicated
    corrected          31.8%           CI [15.6%, 43.9%]

_Paired result against B1 (same 8 actions, same blinding, same standard):_

    action                     HarmScope   B1
    Zelle / Early Warning        strong    —      no P2P-specific unit; nearest
                                                  is unauthorised EFT generally
    Equifax 2019                 strong    —      no data-breach unit
    Fay Servicing 2024           strong  strong
    Nationstar 2020              strong  strong
    Portfolio Recovery 2023      strong  strong
    Bank of America 2022           —      —
    Citibank 2023                  —      —
    Santander 2018                 —      —

    strong rate     HarmScope 5/8 = 62.5%    B1 3/8 = 37.5%
    unadjudicated   HarmScope   50.9%        B1     56.2%
    corrected       HarmScope   31.8%        B1     21.1%
    discordant      HarmScope-only 2, B1-only 0, exact McNemar p = 0.50

_Extended to n = 14 (2026-08-06)._ Six more paired actions, same protocol:

    strong rate    HarmScope 8/14 = 57.1%    B1 7/14 = 50.0%
    corrected      HarmScope 29.1%           B1 28.1%
    discordant     HarmScope-only 2, B1-only 1, exact McNemar p = 1.00

**The reversal did not survive its own sample growing.** At n = 8 it was 5-3
with 2-0 discordant and B1's strong set a strict subset of HarmScope's; six more
actions made it 8-7 with 2-1, and the corrected rates are within one point. The
n = 8 result was a place it would have been very easy to stop, and stopping
there would have produced a headline the data does not support.

Sterling Jewelers is the discordant case that went the other way and is worth
naming: the order is about store cards opened without customer consent, and B1
had a unit for unsolicited prepaid cards arriving in the mail while HarmScope's
nearest cluster was identity theft — a fraudster opening an account, not a
merchant. Coarse units are not uniformly worse; they are differently wrong.

_What the adjudication does establish,_ much more strongly than any
system difference: **it halves both systems.** 57.1% and 50.0% of company-level
detections survive the question of whether the cluster describes the harm. That
correction is an order of magnitude better supported than the gap between them.

**The ordering reverses at n = 8, and the sample cannot support it.** B1's strong set is a
strict subset of HarmScope's, so the direction is at least internally consistent,
but two discordant pairs is not evidence. What the sample does show is a
mechanism: B1 leads unadjudicated because a coarse taxonomy tuple fires for a
company more readily, and loses that lead when asked whether the tuple describes
the specific harm. The two it lost are exactly the specific ones — Zelle P2P
fraud, where B1's nearest unit is unauthorised electronic transfers in general,
and the Equifax breach, which no taxonomy tuple names.

That is the contribution the project was built to test, appearing for the first
time and at a sample size that cannot confirm it. The honest next step is more
adjudication, not a louder claim.

**Roughly a third of "detections" survive the question of whether the cluster
actually corresponds to the harm.** n = 8 and the interval says so; what this
establishes is the direction and rough size of the company-level inflation, not
a headline. The three misses are informative in the same way: all three are
harms with no natural complaint-narrative expression — a bank's internal
garnishment processing, discrimination detectable only in denial rates, and an
add-on product's disclosure. Complaint text cannot contain what consumers do not
know happened to them.

_Blinding, and its limit._ Rank, statistic and decoy status are all withheld and
the presentation order is seeded. The limit is that the adjudicator is the same
process that built the system and has seen the alert list, which no amount of
worklist blinding fixes. Recorded as `claude-opus-5` in `backtest_links`, not
passed off as human labelling.

### Phase 6 — Ground truth & backtest [GATE]
_Actions curated / usable / excluded:_
_enforcement_actions.csv frozen at SHA:_
_Wall time for one full-refit cutoff:_
_Anti-leakage tests passing:_
_Gate passed:_ Y / N

### Phase 9 (partial) — Threshold sensitivity, run early because it changes Phase 7's reading

ROADMAP Phase 9 requires "sensitivity analysis on every threshold in
`config.py`". Running the one that gates an alert early, because the Phase 7
headline turns on it. `signals` already stores `n_supporting_groups`, so the
sweep is a re-query rather than eight refits.

    floor   HarmScope        B1           gap
        1   67.9%  76/112    66.1%  74    HarmScope +1.8
        5   67.9%  76/112    66.1%  74    HarmScope +1.8
       10   55.4%  62/112    58.0%  65    B1        +2.6
       15   50.9%  57/112    56.2%  63    B1        +5.3   <- frozen value
       25   46.4%  52/112    50.0%  56    B1        +3.6
       50   44.6%  50/112    44.6%  50    tie
      100   40.2%  45/112    38.4%  43    HarmScope +1.8

**The ordering changes sign three times, and the frozen value sits in the band
that favours B1.** That is not an argument for moving it — it was frozen before
the backtest and stays frozen, per trap T4 — but it is decisive for how the
Phase 7 result should be read.

_The mechanism is granularity, and it is measurable rather than speculative:_

    system      units   p25  p50  p75  p90    clears 5 / 15 / 50
    harmscope   2,010     7   15   44  163    100% / 52% / 23%
    B1            634     8   18   49  163    100% / 57% / 25%

HarmScope divides the same complaints into three times as many units, so its
support-per-unit distribution sits lower by arithmetic — median 15 against 18 —
not because its units describe harm worse. At floor 5 nothing is excluded and
the systems separate on merit; at floor 100 only large units survive in both and
they separate on merit again; in between, the floor is measuring unit size.

**An absolute support floor is not comparable across systems with different
granularity.** That is a defect in the comparison design, present since the
criteria were written in Phase 0, and it would be fixed by a floor expressed
relative to a system's own support distribution — a quantile rather than a
count. Recorded as the fix, not applied retroactively: changing the criterion
after seeing which system it favours is precisely trap T4, whatever the
justification.

Taken with the adjudication (29.1% vs 28.1%, p = 1.00), the picture is
consistent: **there is no stable difference between the systems.** The
unadjudicated +5.3 for B1 was a threshold artifact, and the n=8 reversal for
HarmScope was a sampling artifact. Both dissolved when examined.

### Phase 7 — Baselines
_B1 implementation notes (must use identical statistical machinery):_ B1 is not
a second implementation. Each `(product_family, issue_std, sub_issue_std)` tuple
is written into `clusters` under its own `run_id`, and every downstream stage —
panel, 2x2 margins, EB shrinkage, BH within family, EWMA, PELT, §6.3 criteria,
backtest — runs over it untouched. There is no parallel code path that could
diverge, which is a stronger reading of §2's "same statistical machinery" than
any refactor.

Both systems read the same `dup_groups` representatives, so the population is
identical and only the partition differs. `coherence` and `persistence` are set
to 1.0 for taxonomy units: they are HDBSCAN notions and a label is a definition
rather than a discovered density. Leaving them NULL would have let §6.3 filter
B1 out and handed HarmScope the comparison.

_First result (2026-08-06, dev model, UNADJUDICATED):_

    system      detected   rate    median lead   units at 2024 cutoff
    HarmScope     57/112   50.9%      1747 d          2,010
    B1 taxonomy   63/112   56.2%      1518 d            634

**B1 wins on detection rate.** The existing CFPB taxonomy identifies six more of
the 112 enforcement actions than the discovered clusters do. `PROJECT_SPEC` and
the README both commit to reporting this outcome if it happens, and it has
happened.

The structural reason is visible in the support distribution. HarmScope produces
2,010 units at the 2024 cutoff against B1's 634, so its units are finer and each
carries less evidence per company: median 15 supporting groups against B1's 18,
and 51.6% of HarmScope's company-level signals clear the `min_supporting_groups`
gate against B1's 57.3%. Splitting the same complaints into three times as many
units divides the support three ways, and the §6.3 floor then removes what is
left. Finer clusters are the whole point of `cluster_selection_method='leaf'`,
and this is the bill for it.

_What this result is not, yet:_ both numbers are company-level and
unadjudicated, so they measure "did anything fire for this company" rather than
"did the thing that fired correspond to this harm". That comparison is roughly
fair between the two systems — both are inflated the same way — but the gap is
small enough that adjudication could move it either direction. The median lead
times are not comparable at all for the reason recorded under Phase 6: they are
the age of a company's complaint stream, not of a harm.

Nothing here is final while Phases 3-5 run on the dev model.

_B2 and B3 — decisions recorded before either produced a detection rate
(2026-08-06):_

Everything below was fixed while B2's first cutoff was still running and B3 had
not been built. The threshold sweep two entries up is the reason for the
formality: the free parameters in a baseline are the free parameters in the
comparison, and choosing them after seeing which way they push is trap T4 no
matter how good the mechanism sounds.

**B2's topic count is HarmScope's cluster count, per family, per cutoff.**
`n_topics` sets B2's granularity directly, and the sweep already measured what
granularity does here: an absolute `min_supporting_groups` floor filters on unit
size, so a system with three times as many units clears it less often regardless
of quality. Matching HarmScope's discovered count removes the degree of freedom —
whatever the floor does to HarmScope it does to B2. This neutralizes the artifact
**between B2 and HarmScope only**. B0 has 11 units and B3 has whatever
`min_cluster_size=10` finds; granularity is not controlled across all four and
the quantile floor recorded above is still the fix.

**`min_supporting_groups` does not measure the same quantity for B3.** B3 runs
without dedup, so its groups are singletons and `n_supporting_groups ==
n_supporting`: the floor of 15 means fifteen *complaints* for B3 and fifteen
distinct *dup-groups* for every other system. Campaign exclusion is also empty
for B3, by the same construction. Two effects pull opposite ways — BERTopic's
`min_cluster_size=10` makes units finer, complaint-denominated counting with no
campaign exclusion makes support larger — and which dominates was not predicted
in advance. This is the same defect class as the threshold artifact: an absolute
count compared across systems whose units are not commensurable. Recorded before
B3's rate exists so that it reads as a caveat rather than as a discovery.
`CONFIG.signals.min_a` in the 2x2 test inherits the same problem.

**B3's "no dedup" is expressed as data, not as a flag.** A `dup_groups`
population of singletons under its own run makes the unchanged panel mean
exactly "no dedup": each complaint carries its own cluster, distinct groups equal
distinct complaints, and no `campaigns` rows exist to exclude. It is registered
under phase `dedup_identity` rather than `dedup` — `latest_run` and
`backtest.run_for_cutoff` resolve by recency within a phase, so an identity run
filed as `dedup` would be newer than the real refit at the same cutoff and would
silently shadow it for HarmScope, B0, B1 and B2.

**B3's deviations from `pip install bertopic`.** The hyperparameters are
BERTopic 0.17's constructor defaults read from its source — UMAP(15, 5,
min_dist=0.0, cosine), HDBSCAN(min_cluster_size=10, euclidean, eom) — and its
default encoder is `all-MiniLM-L6-v2`, which is the model every system in this
comparison runs on. What is not off-the-shelf: the fit is per product family
rather than global, because the panel restricts a cluster's numerator to its own
family and a global model would need a modal family per topic that silently drops
cross-family members; assignment is HarmScope's nearest-centroid rule rather than
`approximate_predict`, so the gap measures default hyperparameters instead of a
different assignment rule; and the fit sample stays at 500k, as HarmScope's does.

**B2's `norm=None` was a bug fix, found before it became a result.**
`TfidfVectorizer` l2-normalizes by default, which leaves each document carrying
about one unit of pseudo-count mass, and LDA fit on that has essentially no data.
Measured at the 2017 cutoff, credit_reporting, k=195: **5 of 195 topics ever won
an argmax and one held 99.0% of the documents.** Unchanged at max_iter 5, 20 and
50, and unchanged with `use_idf=False` — normalization was the cause, not
weighting or convergence. With `norm=None` all 195 topics populate and the
largest holds 5.4%; raw counts sit in between at 92 of 195, largest 23.7%. The
first B2 run, before the fix, produced 114 units at a cutoff where HarmScope has
1,101 — a system crippled by an implementation detail would have looked like
evidence that bag-of-words topic models cannot find harm mechanisms. `max_iter`
stays at 5: 5, 20 and 50 were indistinguishable on this diagnostic, and the
diagnostic is a topic-count property that never sees a detection rate.

_All four baselines (2026-08-07, dev model, UNADJUDICATED):_

    system                      detected   rate    median lead   units at 2024
    B1  taxonomy                  63/112   56.2%      1518 d           634
    B2  TF-IDF + LDA              58/112   51.8%      1810 d         2,010
    B3  BERTopic default          58/112   51.8%      1679 d        11,336
    HarmScope                     57/112   50.9%      1747 d         2,010
    B0  volume only               46/112   41.1%      1954 d            12

B2 and B3 tie at 58 on different actions — the per-cutoff splits differ (B2
8/3/5/11/1/7/15/8, B3 8/2/6/10/2/6/16/8), so the equal totals are coincidence
rather than the same detections.

_Granularity across all five, 2024 cutoff, floor 15:_

    system        units   p25   med   p75      max   clears floor   signals
    HarmScope     2,010     7    15    44    3,429       51.6%       29,004
    B0               12    44   113   354  168,678       96.8%        1,451
    B1              634     8    18    49   37,246       57.3%       24,273
    B2            2,010     7    16    41   12,274       54.4%       32,419
    B3           11,336     9    19    36    7,403       62.4%       55,687

**B3's support column is not comparable with the others.** It counts complaints;
every other row counts dup-groups. That is the pre-registered caveat above, and
the table shows the effect it predicted: B3 has 5.6x HarmScope's unit count yet a
*higher* median support (19 against 15), because complaint-denominated counting
with no campaign exclusion more than offsets the finer units.

**The B2 row is the result worth reading.** Unit count 2,010 against HarmScope's
2,010, median support 16 against 15, clearing the floor 54.4% against 51.6% —
matched by construction on every dimension the threshold artifact operates
through, because `harmscope_k` was fixed before B2 had a rate. On that footing a
bag-of-words topic model detects 58 actions to the embedding pipeline's 57.
EVALUATION §2 asks B2 to "test whether the embeddings buy anything"; the answer
on this evidence is that **no difference is detectable**, which is weaker than
"they do not". Pairwise on the same 112 actions: 54 detected by both, 4 B2-only,
3 HarmScope-only, exact McNemar p = 1.00. A 1-point gap is well inside what the
threshold sweep shows an arbitrary threshold can manufacture, so the honest form
is the one already used for B1 — indistinguishable, not beaten.

_Pairwise agreement, latest backtest run per system, all on 112 actions:_

    pair                    both   A-only   B-only   exact McNemar p
    B1  vs HarmScope          55       8        2         0.109
    B2  vs HarmScope          54       4        3         1.00
    B3  vs HarmScope          53       5        4         1.00
    B2  vs B3                 53       5        5         1.00
    B0  vs HarmScope          40       6       17         0.035

The only pair that separates at all is B0, which is the one comparison the
project needs to win and does. **B2 and B3 tie at 58 on different actions** —
53 shared of 58 each — so the equal totals are coincidence rather than the same
detections.

_The 2x2 floor has the same problem as the support floor._ `contingency` builds
its `a` cell from distinct `(group, company)` pairs, so for B3 — whose groups are
singletons — `min_a = 5` gates on five *complaints* where every other system is
gated on five *dup-groups*. B3's 55,687 company-level signals against
HarmScope's 29,004, and its 62.4% floor-clearing rate, are therefore partly a
floor effect rather than detection quality. This does not touch B3's 58/112,
which comes from `baseline_results`, but it is why B3's row in the granularity
table above must not be read across.

This is a much cleaner comparison than
HarmScope-vs-B1, where the systems differ threefold in granularity and the
ordering was shown to flip with the threshold.

_Cost:_ B3's 2024 cutoff took 516 minutes on credit_reporting alone. BERTopic's
`min_cluster_size=10` against HarmScope's 50 produces 8,363 clusters on a 500k
fit sample, and HDBSCAN's cost is driven by that. Recorded because a future
sensitivity sweep over B3 is not affordable at this setting without a plan.

### Phase 9 (partial) — Does the encoder choice matter? (2026-08-07)

The README has carried a "provisional, dev model" flag on the Phase 4 gate
numbers since Phase 3, with a full `bge-base` re-encode as the implied fix: 16.8
hours for the corpus plus a re-run of Phases 4-7. Rather than pay that to find
out whether it changes anything, the question was scoped down to one that a
single family can answer — **does the encoder change the partition by more than
resampling already does?**

150,000 credit_reporting representatives at the 2024 cutoff, encoded on both
models, clustered with identical config. Paired by construction: same texts, same
`fit_family`, same `assign`, same seed.

    metric                            MiniLM     bge-base
    clusters                             336          350
    fit noise                          65.4%        63.7%
    assigned at the fixed 0.65 floor   94.5%       100.0%
    nearest-centroid cosine p10/p50    0.684/0.794  0.831/0.890
    ARI vs itself, disjoint halves     0.469        0.454
    ARI across the two encoders                0.232

Two guards made this interpretable, and both mattered:

**The cross-encoder ARI is meaningless without its within-encoder baseline.**
0.232 alone looks like proof the encoder matters enormously. Against a
within-encoder disjoint-halves ARI of 0.469 on the same sample — reproducing the
0.505 already recorded for this family — the honest statement is that the encoder
disagrees about twice as much as resampling does. Real, but the same *kind* of
instability the README already documents, not a new phenomenon.

**`assign_max_distance = 0.35` is calibrated to MiniLM's geometry.** bge sits on
a visibly higher cosine scale (p10 0.831 against 0.684), so its 100% assignment
rate at a fixed 0.65 floor is the threshold moving, not coverage improving.
Reporting "bge assigns 100% against MiniLM's 94.5%" would have reproduced the
`min_supporting_groups` artifact in a new place, one entry after diagnosing it.
An absolute cosine threshold is not comparable across encoders for the same
reason an absolute support floor is not comparable across granularities, and the
quantile fix recorded for one applies to the other.

_What this does and does not settle._ It settles that cluster *identity* is
encoder-sensitive, so the provisional flag was justified and nothing downstream
may treat a cluster as the same object across encoders. It does not settle
detection rates. But the mechanism driving the Phase 7 table is granularity —
unit count and support per unit — and on that the two encoders are 4% apart (336
against 350). The encoder swap is unlikely to move the ordering, and the full
encode stays outstanding as a Phase 9 item for the whole table at once, because a
cross-system comparison requires every system on one encoder.

_Method note:_ ARI is computed by `src.cluster.stability` — the same held-out
protocol and the same noise-excluding `ari()` that produced the recorded 0.505,
because two ARIs computed differently are not comparable and that comparison is
the entire point. Validated first on a positive control: with the bge matrix
replaced by a copy of MiniLM's, cross-encoder ARI is 1.000 and within-encoder
disjoint halves is 0.499 against the recorded 0.505.

### Phase 8 — LLM layer

_State on 2026-08-11:_ **engineering built; live-provider, frozen-benchmark,
and human gates pending.** This distinction matters. Unit/integration tests with
fake clients demonstrate cache, retry, retrieval, persistence, and metric
behavior; they do not measure whether Opus labels or answers well.

_Detection boundary:_ the structural isolation suite rejects any import from
`signals`, `cluster`, `dedup`, `embed`, or `evaluation` into `src.llm` and any
query from those packages to Phase 8 tables. `pipeline.py` imports LLM handlers
inside their commands. The exact claim is: `signals` is byte-identical with
`src/llm` absent; `baseline_results` is byte-identical given the same
human-written `backtest_links`.

_What was built:_

- A single typed Anthropic transport with closed structured outputs, bounded
  retry taxonomy, model preflight, cache-token accounting, latency, and
  configuration-fingerprinted price estimates.
- Deterministic 12-medoid/8-MMR label selection, lazy fired-plus-control
  population, atomic cache/resume, per-cluster transactions, refusal/failure
  visibility, a version-bound blinded 50-label worklist, and Wilson reporting.
- Signal-consistent cluster/company evidence with exact cluster-run/dedup/model
  provenance; exact FAISS cosine search; versioned BM25 cache; configured RRF;
  top-10 complaint IDs/dates/company responses; and grounded structured answers
  whose citations are validated against returned complaint evidence.
- An answer cache and crash-durable usage outbox keyed without narrative prose.
  An interrupted or failed call cannot silently lose paid usage, and a caller's
  open transaction cannot cause a staged event to be deleted before commit.
- A strict 30-question authoring/freeze workflow; dense/BM25/fused Recall@10,
  MRR, latency, and win/tie/loss evaluation; fused-answer citation validity,
  coverage, abstention, cache/token/outcome/cost aggregation; and a run-bound,
  blinded 50-claim human groundedness workflow.
- Lazy `rag-eval` CLI actions for default full evaluation, retrieval-only,
  author, import, claims export, and claims record. A run records exact
  `manifest_sha256`, the one cluster-recorded `embed_model`, and
  `retrieval_only`; it declares 90 rows only after all requested work and stable
  rendering succeed. Retrieval-only never reaches answer/provider code.

_Network-free real-data exercise (2026-08-11):_

```text
HARMSCOPE_DATA_DIR=/Users/rushilrawat/HarmScope/data \
  .venv/bin/python -m src.pipeline rag-eval --retrieval-only

FileNotFoundError: .../data/ground_truth/rag_eval_questions.csv
exit status: 1
rag-eval rows in the real runs registry afterward: 0
```

This is the correct first failure. The command loads the exact frozen file
before `db.bootstrap()` or `db.run`, so an absent human artifact cannot create a
misleading failed benchmark run. There is consequently no manifest hash, run
ID, Recall@10/MRR, fusion loss count, or latency to record.

_Private authoring state (not ground truth):_ the ignored
`data/interim/rag_eval_authoring.csv` contains 30 blank candidates, five per
category, 15 fired/15 controls, 30 distinct clusters, and all 12 product
families. Human question/relevance/privacy fields are blank. Eighteen rows have
blank company scope. Calling this a benchmark, or deriving metrics from it,
would substitute agent selection for the required human ground truth.

_Open gates, in dependency order:_

1. **Human benchmark author/privacy review and freeze.** A person writes the 30
   synthetic analyst questions, marks every relevant complaint ID, performs
   semantic paraphrase/privacy review, and records author/reviewer/freeze
   provenance. Relevant IDs cannot be edited after metrics are viewed.
2. **Company-scope decision.** Grounded answers and `rag_answers` intentionally
   require a concrete company, but 18 candidate rows are cluster-wide. Regenerate
   those candidates or approve an explicit schema/answer amendment; do not
   silently broaden scope.
3. **`company_response` evidence decision.** The authoring worklist contains ten
   complaint excerpts, not independently sourced public responses. Five honest
   questions in this category cannot be authored from that view alone.
4. **Live provider/funding check.** On 2026-08-07, authentication and model
   preflight succeeded; the subsequent messages request returned terminal
   `400 invalid_request_error` because the organization's credit balance was
   too low. The earlier CLI token-expiry interpretation was wrong because
   authentication auto-refreshed. Phase 8D made no provider call on 2026-08-11,
   so it did not recheck the current balance. The earlier full 4,818-cluster
   estimate was roughly $205, but pricing and population must be re-estimated
   before purchase. No Task 8D call incurred provider cost.
5. **Human label and claim review.** At least 50 version-bound labels and at
   least 50 run-bound claims must be reviewed by a person. Automated/LLM judges
   cannot complete either denominator.

_Label agreement rate on 50 verified:_ **PENDING — 0 human worklists recorded;
not reported as 0%.**

_Observed live LLM failure modes:_ the 2026-08-07 messages request returned the
insufficient-credit error after successful authentication and model preflight.
No provider request or balance check ran during Phase 8D on 2026-08-11.
Structured schema/refusal/retry/cache behaviors are test-fixture observations,
not live model frequencies.

_RAG Recall@10 / MRR / fusion win-tie-loss:_ **PENDING — no frozen manifest and
no evaluation run.**

_Citation validity / coverage / abstention / groundedness:_ **PENDING — no paid
answer run; 50-claim human review not started.**

_Cost incurred by Phase 8D evaluation:_ **$0 observed.** Configuration-based
cost arithmetic is tested, but no test value is presented as an invoice or live
metric.

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

### 2026-08-05 — "Hand-label 300 pairs" → blind model adjudication

ROADMAP Phase 2 specified hand-labelled pairs. `d9ead66` flagged in advance that
the committed labels are an exact-Jaccard reference and "cannot yet stand in for
the hand-labelling ROADMAP Phase 2 requires". No human was available to label,
so the 100 `hard` pairs were adjudicated by the model instead, and the criterion
is changed rather than quietly treated as met.

What makes it worth anything: the rule was written into `METHODOLOGY §2.4.1`
**before any pair was read**, and adjudication is blind — pairs shuffled by seed,
with the detector's decision, the proxy label and `true_jaccard` all withheld.
`label_source` is `model_adjudicated_blind` on every row, so no reader can
mistake it for human judgement. A human re-reading the same 100 pairs would
supersede it; the file is structured for that.

What it does not fix: the adjudicator and the system share an author. The
independent evidence is the merge audit, which is a *measurement* rather than a
judgement — 3 of 40 same-group pairs have a verified edge — and it holds
whatever anyone thinks about any individual pair.

### 2026-08-05 — Campaign flag left as is; template inflation handled in Phase 5

Four options were written down for the large templates that score 2 of 5 signals
and go unflagged. Chose the one that adds no threshold: leave the flag, and make
Phase 5 count distinct `dup_groups` rather than raw complaints.

The reasoning is that the exposure is narrower than it looks. Dedup already
collapses each of these groups to one representative and clustering consumes
representatives only, so nothing reaches Phase 4 inflated. The only place a
24,507-member unflagged template can distort anything is a growth statistic that
counts complaints — and `signals.n_supporting_groups` was already in the schema
for exactly that. Every other option required picking a size bound while looking
at which groups it would move, which is trap T4.

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

### 2026-08-11 — Phase 8D orchestration is built; real evaluation remains gated

The `rag-eval` CLI now registers the exact manifest SHA-256, embedding model,
and retrieval-only mode under one run ID; it runs retrieval and, in full mode,
answer evaluation under that identity. Authoring, import, blinded claim export,
and reviewer-bound claim recording are also exposed through lazy handlers. The
fresh required selection passed 427 tests with one expected missing-manifest
skip; the full repository passed 659 tests with six expected real-data skips.

The only honest network-free exercise was:

```console
HARMSCOPE_DATA_DIR=/Users/rushilrawat/HarmScope/data \
  .venv/bin/python -m src.pipeline rag-eval --retrieval-only
```

It exited 1 with `FileNotFoundError` for
`data/ground_truth/rag_eval_questions.csv`, before database bootstrap or run
creation. A read-only query immediately afterward found zero `rag-eval` run
rows. No manifest hash, run ID, RAG metric, label metric, answer cost, or human
judgment therefore exists to record.

The private ignored authoring draft still needs human question authorship,
privacy review, and freeze. Eighteen rows have no concrete company scope, and
the five planned `company_response` questions lack independent public-response
evidence in the complaint-only worklist. The existing embedding artifacts use
the legacy short-model filename rather than the current full model-SHA contract
and require a validated migration or regeneration. The last live messages
request returned insufficient credit, but Phase 8D did not recheck the current
balance; live labeling and answering plus both the 50-label and 50-claim
blinded human reviews remain pending.

### 2026-08-15 — Embedding migration Fix Round 5 protocol review complete

The `migrate-embeddings` maintenance CLI is implemented with an explicit
full-model `--model`; plan mode is the default and `--execute` is required for
publication. It requires an existing DuckDB file, opens it read-only, creates no
schema or run record, and always closes its connection. It validates sidecar,
NumPy, FAISS, and mapping contracts before Darwin `fclonefileat` publishes CoW
targets from retained source descriptors. Targets are independently validated
before legacy basenames can be atomically moved, without overwrite, to
deterministic dot-prefixed retirement names. Automated execution never unlinks
those retained originals; exact digest and semantic equivalence drives crash
replay, while conflicts remain named and fail closed. Non-Darwin mutation has
no path-copy fallback. The current retrieval loader remains strictly
SHA-addressed: a legacy or retirement filename is never a runtime fallback.
Target-only state and same-inode dual aliases left by a superseded hard-link
protocol fail closed because they lack an independent retained comparison.

The Fix Round 5 implementation is locally verified, and an independent Task 2
audit found no remaining Critical or Important protocol issue. No shared-data
plan, `--execute`, provider call, or real-data access occurred during this task.
The embedding gate stays open until a separate pre-execution review authorizes
an execution record covering source/target/retirement names, distinct target
and retained-source inode/size evidence, exact bytes, validation
counts/dimension, strict-loader success, and unchanged database evidence.
Retirement deletion is a later manual operation requiring a quiescent artifact
directory.

### 2026-09-03 — MiniLM artifact migration completed and verified

After the four final Important findings were fixed with witnessed RED/GREEN
regressions, 175 migration/embed/retrieval/CLI tests and the full 912-test
repository gate passed. A fresh independent source review found no Critical or
Important issue and approved a real read-only plan plus explicitly authorized
execution. The plan reported `planned` for 2,477,937 rows × 384 dimensions and
changed no filesystem or database evidence.

The approved execute returned `migrated`; immediate replay returned
`already-migrated`. The original memmap, FAISS, and sidecar inodes
(`133525751`, `133569509`, `133526780`) are retained under deterministic
dot-prefixed retirement names. SHA targets use distinct inodes, the same device
and sizes, and exact matching per-role SHA-256 digests. All six names have link
count one; all three legacy basenames are absent. DuckDB retained inode
`133057031`, size `16013864960`, and mtime `1786116682`; MiniLM mapping
aggregates remained `(3830002 rows, 2477937 distinct row_idx, 0 min, 2477936
max, 384 min/max dim)`. The strict loader returned `(2477937, 384) float32`.
Forced-offline dense/BM25/RRF retrieval over cluster
`1786080772276-4d6745a1:bank_account:16` and company `jpmorgan-chase` returned
five complaint IDs. No provider call or narrative output occurred. Retirement
deletion remains a separate quiescent manual decision.
