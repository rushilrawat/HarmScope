# EVALUATION

The evaluation is the project. A discovery pipeline with no backtest is a topic model with a
dashboard, which is the cliché this project exists to avoid.

---

## 1. Point-in-time backtest

### 1.1 Protocol

For each enforcement action *a* with `filed_date` *d*:

1. Set cutoff `C = d`.
2. Rebuild every **date-dependent** stage using only complaints with
   `date_received < C`. Not a filter on the final signal table.
3. Ask: did any signal fire, at any period `t < C`, for a cluster matched to action *a* and
   company matched to *a*?
4. `lead_time_days = d − first_signal_date`.

### 1.1.1 What is refit per cutoff, and what is not

Refitting everything is the safe default, but two stages carry no time
dependence and refitting them costs hours per cutoff for no leakage benefit.
The split is not a judgement call — it follows from whether a stage's output
can depend on when a complaint arrived.

| Stage | Per cutoff? | Why |
|---|---|---|
| Embedding | **No — compute once** | The encoder is a fixed pretrained checkpoint, never fit on the corpus (§5 item 3). A vector for complaint *i* is identical whether or not complaint *j* exists. Date-filter the matrix; do not recompute it. |
| MinHash pair detection (`dup_pairs`) | **No — compute once** | Jaccard similarity between two narratives is pairwise and date-independent. |
| Grouping + representative selection (`dup_groups`) | **Yes** | Transitive closure is date-dependent: if A~B and B~C but B is post-cutoff, A and C are separate groups at that cutoff. And the representative must be chosen from members with `date_received < C`, or a pre-cutoff group inherits a post-cutoff exemplar. |
| Campaign detection | **Yes** | `burstiness`, `state_concentration`, `first_seen`/`last_seen` are time-windowed aggregates (`METHODOLOGY.md` §2.2). Computed over the full corpus they leak post-cutoff behaviour into a pre-cutoff flag — and campaign-flagged complaints are *excluded* from detection, so a leaked flag silently suppresses a real signal. This is the most dangerous of the five to get wrong, because its failure mode is a missing alert rather than a spurious one. |
| UMAP + HDBSCAN | **Yes** | Cluster definitions are the leakage vector trap T3 names. |
| Novelty scoring | **Yes** | Label distributions are over cluster members. |
| Timeseries + signals | **Yes** | Obviously. |

This is roughly 8× off the most expensive stage without weakening the leakage
guarantee. The anti-leakage suite (§5) tests the guarantee directly, so the
saving does not rest on this table being reasoned about correctly.

### 1.2 Why full refit

The tempting shortcut is to cluster once on all data and filter signals by date. That leaks:
cluster *definitions* would be informed by post-cutoff complaints, including complaints filed
in response to the enforcement action itself. A cluster that only exists because of
post-action complaints will "detect" the action perfectly and mean nothing.

**Cost mitigation:** full refit per action is expensive. Use **rolling annual cutoffs**
(2017-01-01, 2018-01-01, …, 2024-01-01) rather than per-action cutoffs. Each action is
evaluated against the most recent cutoff strictly before its `filed_date`. This yields 8 refits
instead of 30+, and is *conservative* — lead time is measured from a model that is up to 12
months staler than it needed to be. Understating your own lead time is the right direction to
err. Document this choice.

**Consequence for the ground-truth window.** An action filed before the first
cutoff has no cutoff strictly preceding it and therefore cannot be evaluated at
all. The ground-truth window is therefore **2017-01-01 → 2024-12-31**, not
2016–2024. Actions filed in 2016 stay in `enforcement_actions.csv` with
`usable = false` and `exclusion_reason = pre-first-cutoff`, per the `DATA.md` §4
convention of keeping excluded rows visible.

This costs a year of candidate actions against an already-tight ≥ 20-usable
requirement (`PROJECT_SPEC.md` §6), so it is a real scope reduction and is
recorded as one. It is nonetheless the right trade: every system — B0, B1, B2,
B3, HarmScope — runs the identical harness over the identical cutoffs (§2), so
narrowing the window moves absolute detection rate for all five equally and
leaves **the gap to B1 unchanged**. That gap is the entire contribution. Adding
a 2016-01-01 cutoff was the alternative; it was rejected because a model
trained on 2015 alone has one year of narratives and would contribute near-certain
misses that depress the headline for reasons unrelated to method quality.

### 1.3 Matching a cluster to an action (the adjudication step)

This is the one irreducibly manual step. Automating it with keyword matching would let the
`harm_keywords` field do the work and inflate results.

Protocol:

1. For the action's company and product family, retrieve the top 20 clusters by signal strength
   at the cutoff.
2. **Blind the adjudicator to whether each cluster fired.** Present cluster exemplar narratives
   and LLM labels only, in randomized order, with 10 decoy clusters from unrelated companies.

   Note that this step deliberately puts LLM output in front of a human whose
   decision writes `backtest_links`. That is a designed human-in-the-loop step,
   not a leak of the LLM into the detection path — but it is why the
   determinism test in `LLM_LAYER.md` §1 is scoped to `signals` and not to
   `backtest_results`. Present the labels alongside exemplar narratives, never
   instead of them, so an adjudicator can overrule a bad label.

3. Adjudicator marks `strong` / `partial` / `none` against the action's `harm_summary`.
4. Record in `backtest_links` with adjudicator ID and notes.
5. Only `strong` counts as a detection in headline metrics. Report `partial` separately.

If you are the only adjudicator, say so, and re-adjudicate a random 20% two weeks later to
report intra-rater agreement (Cohen's κ). Single-adjudicator is a limitation, not a
disqualification — hidden single-adjudicator is a credibility problem.

---

## 2. Baselines

Implement all four. The project's contribution is defined entirely by the gap to **B1**.

| ID | Baseline | Description |
|---|---|---|
| **B0** | Volume | Total complaint count per company per month; EWMA changepoint. The dumbest thing that could work. |
| **B1** | **Taxonomy** | Growth in `(company × Product × Issue × Sub-issue)` counts, same disproportionality + changepoint machinery, same FDR. **This is the baseline that matters.** |
| **B2** | TF-IDF + LDA | Classic topic model over narratives, same downstream detection. Tests whether embeddings buy anything. |
| **B3** | BERTopic default | Off-the-shelf, default params, no dedup, no novelty scoring. Tests whether the custom pipeline buys anything over `pip install bertopic`. |
| — | **HarmScope** | Full pipeline. |

**Every baseline runs through the identical backtest harness, identical cutoffs, identical
adjudication protocol.** If B1 wins, the honest headline is "the existing taxonomy is
sufficient" — which is a genuinely interesting finding and a defensible portfolio artifact.

Run B3 *without* dedup deliberately. The delta between B3 and HarmScope isolates the value of
campaign detection, which is the most novel engineering contribution here.

---

## 3. Metrics

### 3.1 Primary

| Metric | Definition |
|---|---|
| **Detection rate** | fraction of usable actions with a `strong`-matched signal before `filed_date` |
| **Median lead time** | median `lead_time_days` over detected actions |
| **Lead time distribution** | report the full distribution, not just the median — a bimodal result means something |
| **False alerts per 1,000 company-months** | signals fired that never map to any enforcement action, normalized by exposure |

Report detection rate and false-alert rate **as a pair**, always. A system that alerts on
everything detects everything.

### 3.2 Secondary

- **Precision@k** — of the top-k signals by statistic in a given period, how many are
  adjudicated as real harms (not necessarily enforced)? Requires separate adjudication; a
  smaller sample (k=20 per year) is fine.
- **Dedup precision / recall** on the 300-pair labeled set.
- **Novelty AUC** from the label-ablation test (`METHODOLOGY §5.1`).
- **Cluster stability ARI** across sample sizes and disjoint halves.
- **Calibration** — bin signals by `q_value`, plot observed match rate per bin.
- **Subgroup performance** — detection rate by product family, and separately for complaints
  tagged `Older American` / `Servicemember`.

### 3.3 Operational framing

Translate at least one metric into an analyst-facing quantity:

> "Reviewing the top 10 alerts per month would have surfaced N of M enforced harms a median of
> D days before public action, at a cost of R reviews per confirmed harm."

That sentence is what makes this a decision system rather than a model.

---

## 4. Failure analysis (required, not optional)

For every action the system **missed**, categorize the reason:

| Category | Meaning |
|---|---|
| No narratives | Company had too few narrative-bearing complaints in window |
| Not consumer-observable | Consumers could not perceive the conduct |
| Absorbed by campaign filter | Real harm was flagged as a campaign (false positive on dedup) |
| Clustered but below threshold | Cluster existed, statistic never crossed |
| Not clustered | Narratives existed but did not form a coherent cluster |
| Company mismatch | Legal entity in action ≠ company string in complaints |

Publish this table. It is more informative than the headline number and it is the part that
demonstrates engineering judgment.

Do the same for the 20 highest-statistic **false alerts**: what were they actually? Common
expected answers: unfiltered campaigns, news-driven complaint spikes, seasonal effects,
company operational changes (system migrations produce complaint bursts that are real but not
misconduct).

---

## 5. Anti-leakage checklist

Run before reporting any number. Each item is a test in `tests/evaluation/`.

- [ ] No row with `date_received >= cutoff` appears in any table used by a pre-cutoff run.
- [ ] Cluster definitions used at cutoff C were fit only on `< C` data (assert on
      `clusters.as_of`).
- [ ] Embedding model is a fixed pretrained checkpoint, not fit on the corpus.
- [ ] `enforcement_actions.csv` git SHA is recorded and predates the first detection run.
- [ ] Adjudication was blind to signal status.
- [ ] Thresholds in `config.py` were not tuned on the backtest set. If they were, hold out
      1/3 of actions as a tuning set and report on the remaining 2/3 only — and say so.
- [ ] No `harm_keywords` from the ground-truth file are used anywhere in the detection path.

Item 6 is the one most likely to be violated by accident. Freeze thresholds before backtesting,
or split.

---

## 6. Reporting template

The README results table must contain, at minimum:

```
Actions curated: 34   |   usable: 27   |   excluded: 7 (reasons in DATA.md)
Cutoffs: annual, 2017–2024
Adjudication: blind, single adjudicator, intra-rater κ = 0.__

                  Detect rate   Median lead   FA / 1k co-months
B0 volume              __%          __ d            __
B1 taxonomy            __%          __ d            __
B2 TF-IDF+LDA          __%          __ d            __
B3 BERTopic            __%          __ d            __
HarmScope              __%          __ d            __

Novelty AUC (label ablation): __
Dedup P/R @ threshold: __ / __
Cluster stability ARI (disjoint halves): __
```

If HarmScope does not beat B1, that table goes in the README unchanged, with a paragraph
explaining why. That is the version of this project worth putting on a resume.
