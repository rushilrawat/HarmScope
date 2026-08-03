# METHODOLOGY

## 1. Overview

Five sequential problems, each with a defined output and an acceptance test:

| # | Problem | Output | Gate |
|---|---|---|---|
| 1 | Are these N complaints N events or one campaign? | `dup_groups`, `campaigns` | precision/recall on 300 hand-labeled pairs |
| 2 | What harm mechanisms exist in the text? | `clusters` | stability across sample sizes (ARI) |
| 3 | Which mechanisms are *not* in the taxonomy? | `cluster_novelty` | separation on held-out labels |
| 4 | Which are growing abnormally? | `signals` | calibration + FDR control |
| 5 | Did we see it before enforcement? | `backtest_results` | see `EVALUATION.md` |

Problem 1 is the hardest and the one most likely to be skipped. Do not skip it.

---

## 2. Deduplication and campaign detection

### 2.1 Why this is the crux

The research question is "is this cluster growing?" That question is meaningless if 4,000
narratives came from one credit-repair service's template. The failure mode is not subtle: the
system will confidently report a fast-growing novel harm cluster that is actually a marketing
campaign. **Solve this before touching embeddings.**

### 2.2 Three-tier detection

**Tier 1 — exact.** `sha256` of normalized text (lowercase, collapse whitespace, strip
punctuation, replace digit runs with `#`). Group identical hashes. Fast, catches pure copies.

**Tier 2 — near-duplicate.** MinHash (`datasketch`, 128 permutations) over character 5-shingles,
LSH banded for Jaccard threshold ~0.85. Blocked by `product_family` to keep candidate sets small.
Union-find over LSH candidate pairs → `group_id`.

Tune the threshold on hand-labeled pairs. Character shingles beat word shingles here because
templates vary mainly in inserted account numbers and dates.

**Tier 3 — campaign detection.** Groups are not enough; a campaign may vary phrasing enough to
evade MinHash. Compute per candidate campaign (a dup_group, or a tight embedding neighborhood):

| Feature | Signal |
|---|---|
| `burstiness` | Fano factor of daily submission counts; campaigns are spiky |
| `state_concentration` | HHI over `state`; organic harms spread, campaigns concentrate |
| `boilerplate_score` | density of statutory citations + formulaic legal phrasing |
| `company_concentration` | HHI over `company_id` |
| `length_variance` | templates have unnaturally low length variance |
| `submitted_via_concentration` | campaigns come through one channel |

Combine into a flag with a hand-tuned threshold, validated on the labeled sample. Do not train
a supervised model here — you do not have enough labels and the features are interpretable.

### 2.3 Handling, not deletion

**Do not delete duplicates.** Collapse them:

- Each `dup_group` gets one `is_representative = TRUE` row. The representative is the
  member with the **earliest `date_received`**, chosen from members with
  `date_received < cutoff`. Two reasons: it is deterministic (no dependence on
  an embedding medoid that shifts when the group grows), and it is
  backward-looking, so a group that acquires new members after a backtest
  cutoff does not retroactively change the representative a pre-cutoff run saw.
  Ties break on the lower `complaint_id`.
- Downstream counting uses **representatives only** for cluster detection.
- The `n_supporting` on a signal reports **both** raw complaint count and distinct-group count.
  A signal backed by 400 complaints in 3 groups is weak; 400 complaints in 380 groups is strong.
- Campaign-flagged complaints are excluded from signal detection by default but remain queryable
  and are shown in the UI as a separate, labeled band.

This mirrors the MAUDE follow-up-report problem: the count is not the number of events.

### 2.4 Acceptance criteria (gate — do not proceed without)

- Hand-label 300 narrative pairs (stratified: 100 obvious dups, 100 hard near-dups, 100
  unrelated). Store as `data/ground_truth/dedup_eval_pairs.csv`.
- Report precision, recall, F1 at the chosen threshold. Target: precision ≥ 0.95 (false merges
  are worse than misses — a false merge destroys real signal).
- Report what fraction of the corpus is campaign-flagged, per product family. Sanity check: if
  credit reporting is not substantially higher than mortgage, the detector is not working.

---

## 3. Embedding

- Default: `BAAI/bge-base-en-v1.5` (768-d). Dev/iteration: `all-MiniLM-L6-v2` (384-d).
- Input: `narratives.text_redacted`, representatives only.
- Truncate at model max tokens; for long narratives, embed first + last window and mean-pool.
  Complaint narratives often state the core problem at both ends.
- Normalize to unit length (cosine == inner product; FAISS `IndexFlatIP`).
- Persist as `float32` memmap + `embedding_map` table. Checkpoint every 50k so a crash at
  2.8M does not cost the run.

**Do not fine-tune in v1.** An untuned strong general embedder is a defensible baseline; a
fine-tuned one without a supervised objective is unjustifiable. Note it as a v2 ablation.

---

## 4. Clustering

### 4.1 Pipeline

```
representatives (per product_family)
   → UMAP (n_neighbors=30, n_components=10, metric=cosine, min_dist=0.0)
   → HDBSCAN (min_cluster_size ∝ family size, min_samples=10, cluster_selection_method='leaf')
   → exemplars per cluster
   → full-corpus assignment via approximate_predict / FAISS nearest-exemplar + threshold
```

`min_dist=0.0` and `n_components=10` because UMAP here is a preprocessing step for density
clustering, not a visualization. `cluster_selection_method='leaf'` yields finer-grained, more
specific clusters — appropriate when the goal is *mechanisms*, not broad topics.

### 4.2 Stratification

Cluster **within** `product_family`, not globally. Credit reporting volume otherwise dominates.
Reconcile afterwards: clusters from different families whose centroids are within a cosine
threshold are linked as `related_clusters` (cross-product harms — e.g. the same servicing
failure appearing under both mortgage and student loan — are genuinely interesting and worth
surfacing).

### 4.3 Stability testing (gate)

Refit on independent stratified samples at 100k / 250k / 500k, and on two disjoint halves at
the chosen size. Report:

- Adjusted Rand Index between runs (on the intersection of assigned points).
- Fraction of points assigned to noise, per run.
- Cluster count as a function of sample size.

If ARI between disjoint halves is low, clusters are sampling artifacts. Either increase
`min_cluster_size`, or accept fewer, coarser clusters. **Report the ARI in the README regardless
of what it says.**

---

## 5. Novelty scoring

The question: does cluster *c* correspond to something the CFPB taxonomy already names?

For each cluster, build the distribution over `(issue_std, sub_issue_std)` tuples of its members.

| Metric | Meaning |
|---|---|
| `dominant_label_share` | fraction of members carrying the modal label tuple |
| `label_entropy` | Shannon entropy of the label distribution (normalized by log of support) |
| `normalized_mutual_info` | NMI between cluster assignment and label assignment, computed globally |

**Novelty score:**

```
novelty = w1 * (1 - dominant_label_share) + w2 * normalized_entropy
```

`w1 = w2 = 0.5` by default, constrained to sum to 1.0 (`src/config.py`,
`NoveltyConfig`). These are the one set of weights in the project that may be
tuned, and only against the **label-ablation test** in §5.1 — never against the
backtest set. The ablation test is a legitimate tuning surface because it uses
deliberately hidden `Issue` categories, not enforcement outcomes.

Trap T4 applies regardless: the config fingerprint recorded in `runs.config_hash`
changes when a weight changes, so tuning is visible in the run registry. If the
weights move after the ground-truth freeze, the fingerprint on the backtest run
will not match the one recorded at freeze time, and that discrepancy must be
explained in `ENGINEERING_NOTES.md` rather than quietly absorbed.

High novelty = the cluster's members are scattered across many existing labels, meaning no
single existing category captures it. Low novelty = the cluster is a restatement of an existing
label (useful as a sanity check that clustering works at all).

### 5.1 Validation of the novelty score (gate)

You cannot validate novelty against nothing. Use a **label-ablation test**:

1. Hold out one `Issue` value entirely — remove it from the label space used for scoring.
2. Clusters that in fact correspond to that held-out issue should score as *high novelty*.
3. Repeat across 10 held-out issues. Report AUC of `novelty_score` separating
   known-but-hidden clusters from genuinely-matched clusters.

If the score cannot recover deliberately hidden categories, it will not find real new ones.
This test is the single most important methodological validation in the project.

### 5.2 Guard against the trivial failure

A cluster with high label entropy might just be **incoherent** (bad clustering), not novel.
Require both:

- `novelty_score >= threshold`, **and**
- cluster coherence above a floor — mean intra-cluster cosine similarity, and HDBSCAN
  `persistence` above a minimum.

Incoherent-and-novel is a clustering defect. Coherent-and-novel is a finding.

---

## 6. Abnormal growth detection

Two orthogonal signals. A cluster must fire on at least one; firing on both is stronger.

### 6.1 Disproportionality (company × cluster)

Borrowed from pharmacovigilance, where it is the established method for exactly this shape of
problem (spontaneous-report signal detection without a denominator).

Build the 2×2 for company *i*, cluster *c*, within product family *f* and window *w*:

|  | in cluster c | not in c |
|---|---|---|
| company i | a | b |
| other companies | c | d |

**Proportional Reporting Ratio:**
```
PRR = [a/(a+b)] / [c/(c+d)]
```
Report PRR with a 95% CI (Gart / normal approximation on log PRR). Also compute ROR.

**Shrinkage.** Raw PRR is wildly unstable at small *a*. Apply an empirical-Bayes gamma-Poisson
shrinkage (EBGM / EB05) or, at minimum, require `a >= 5` and report the lower CI bound rather
than the point estimate. Rank by **EB05 / PRR lower bound**, never by point PRR.

**Multiple testing.** The number of (company × cluster × window) tests is large. Apply
Benjamini–Hochberg FDR at α = 0.05 within each product family. Report `q_value`, not `p_value`,
in the UI.

### 6.2 Temporal changepoint (cluster share over time)

Model the monthly count of cluster *c* with an exposure offset:

```
n_{c,t} ~ NegativeBinomial(μ_{c,t}, θ)
log μ_{c,t} = β0 + β1·t + log(denom_{c,t})
```

Negative binomial, not Poisson — complaint counts are overdispersed. `denom` = total complaints
in the same family/period (and company, for company-level series). Without the offset you detect
overall database growth, not cluster growth.

Layer two detectors:

- **EWMA control chart** on `share`, with control limits from a rolling baseline window.
  Fires on sustained shift. Cheap, interpretable, well-understood by analysts.
- **PELT changepoint** (`ruptures`) on the share series. Fires on abrupt regime change and
  gives you a changepoint *date*, which is what lead time is measured from.

Signal date = first period where the detector fires **using only data available at that period**.
This is enforced by `as_of`, not by care.

### 6.3 Alert construction

An alert requires, jointly:
- cluster coherence ≥ floor
- `novelty_score` ≥ threshold (for the novel-harm track; a parallel known-harm track is fine
  and useful, just labeled separately)
- not campaign-flagged
- `q_value` ≤ 0.05 on disproportionality **or** changepoint detected
- `n_supporting_groups` ≥ minimum (distinct dup-groups, not raw complaints)

Every threshold in that list is in `config.py` and gets a sensitivity analysis in Phase 9.

---

## 7. What could make this wrong

Write these in the README's limitations section, not buried here.

1. **Complaint volume ≠ harm volume.** Media coverage of a company drives complaints
   independent of conduct. A disproportionality spike may be an attention spike.
2. **Narrative opt-in bias** is not correctable and interacts with product type.
3. **Company size denominators are unavailable.** PRR uses complaint-share, not
   per-customer rates. A large bank has more complaints because it has more customers.
   State this every time a company is named.
4. **Enforcement `filed_date` is public disclosure, not regulator awareness.** Lead time is
   measured against the wrong event, and it is the only event available.
5. **Reverse causality on lead time.** An enforcement action generates news coverage that
   generates complaints. Only signals strictly before `filed_date` count — and even then,
   press reporting can precede the filing. Where possible, check `conduct_start` and prefer
   signals that also precede the earliest known public reporting.
6. **Survivorship in ground truth.** You only see harms CFPB chose to act on. Harms that were
   never enforced are invisible negatives, so "false alerts" may include real harms.
