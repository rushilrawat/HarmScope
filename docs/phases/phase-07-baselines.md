# Phase 7 — Baselines

**State:** ✅ complete — all four systems · **Effort:** ~1 week

## Question

Is any of this worth more than something simpler?

## Context

The project's contribution is defined *entirely* as the gap to B1. If the CFPB's
own published taxonomy gives the same lead time, HarmScope has no contribution
and the README has to say so. That commitment predates the numbers.

The design principle throughout: **a baseline is a unit definition, not a
parallel pipeline.** Each system materialises its units into `clusters` and
`cluster_members` under its own `run_id`, and every downstream stage — panel, 2×2
margins, EB shrinkage, BH within family, EWMA, PELT, alert criteria, backtest —
runs over it untouched. There is no second implementation that could diverge,
which is a stronger reading of §2's "same statistical machinery" than any
refactor could give.

## The four systems

| ID | Unit | Tests |
|---|---|---|
| **B0** | one per product family | Whether anything after Phase 1 earned its keep |
| **B1** | `(family, issue, sub_issue)` tuple | **The one that matters** — the taxonomy CFPB already publishes |
| **B2** | TF-IDF + LDA topic | Whether the embeddings buy anything |
| **B3** | BERTopic defaults, **no dedup** | Whether the custom pipeline beats `pip install bertopic` |

## Tech

| Choice | Why this one |
|---|---|
| **B1/B0 coherence and persistence = 1.0** | HDBSCAN notions a taxonomy label has no analogue for. Leaving them NULL would let §6.3's criteria silently filter B1 out and hand HarmScope the win. |
| **B2 `n_topics` = HarmScope's cluster count**, per family per cutoff | Fixed *before* B2 had a detection rate. Removes the one free parameter that sets granularity — the mechanism behind B1's apparent lead. |
| **B2 `TfidfVectorizer(norm=None)`** | See findings; the default l2 normalization destroys the model. |
| **B2 LDA online, `max_iter=5`** | Cost is linear in documents × topics and k reaches 741. Fixed on a granularity diagnostic where 5, 20 and 50 were indistinguishable — never on a detection rate. |
| **B3 hyperparameters read from BERTopic 0.17 source** | UMAP(15, 5, `min_dist=0`, cosine), HDBSCAN(`min_cluster_size=10`, euclidean, `eom`). Not recalled from memory. Its default encoder is `all-MiniLM-L6-v2` — the model every system here runs on. |
| **B3 "no dedup" as an identity `dup_groups` population** | Singleton groups, no `campaigns` rows, registered under phase `dedup_identity`. The unchanged panel then *means* "no dedup" with no flag threaded through it. |

## Acceptance

> All four produce `baseline_results` rows under the same `run_id` scheme.

**Met.** Verified directly: every system has 112 distinct actions across 8
cutoffs.

## Findings

### The table

| System | Detected | Rate | Median lead | Units at 2024 |
|---|---:|---:|---:|---:|
| B1 — CFPB taxonomy | 63 / 112 | **56.2%** | 1518 d | 634 |
| B2 — TF-IDF + LDA | 58 / 112 | 51.8% | 1810 d | 2,010 |
| B3 — BERTopic default | 58 / 112 | 51.8% | 1679 d | 11,336 |
| HarmScope | 57 / 112 | 50.9% | 1747 d | 2,010 |
| B0 — volume only | 46 / 112 | 41.1% | 1954 d | 12 |

**HarmScope finishes last of the four non-trivial systems.** Pairwise on the same
112 actions:

| Pair | both | A-only | B-only | exact McNemar *p* |
|---|---:|---:|---:|---:|
| B1 vs HarmScope | 55 | 8 | 2 | 0.109 |
| B2 vs HarmScope | 54 | 4 | 3 | 1.00 |
| B3 vs HarmScope | 53 | 5 | 4 | 1.00 |
| B2 vs B3 | 53 | 5 | 5 | 1.00 |
| **B0 vs HarmScope** | 40 | 6 | 17 | **0.035** |

The only pair that separates is B0 — the one comparison the project needs to win,
and does.

### B2 is the row that carries the most weight

It is the **only** comparison where granularity is not a confound, because
`harmscope_k` matched B2's unit count to HarmScope's before B2 had a rate. At the
2024 cutoff: 2,010 units against 2,010, median support 16 against 15, clearing
the floor 54.4% against 51.6%.

Matched on every dimension the threshold artifact operates through, **a
bag-of-words topic model detects 58 actions to the embedding pipeline's 57.**
`EVALUATION §2` asks whether the embeddings buy anything; on this evidence **no
difference is detectable**, which is weaker than "they do not".

### A bug caught before it became a result

`TfidfVectorizer` l2-normalizes by default, leaving each document with about one
unit of pseudo-count mass — and LDA fit on that has essentially no data. Measured
at credit_reporting, k=195: **5 of 195 topics ever won an argmax, one holding
99.0% of documents.** Unchanged at max_iter 5/20/50 and unchanged with
`use_idf=False`, so normalization was the cause, not weighting or convergence.

With `norm=None`, all 195 populate and the largest holds 5.4%. The first B2 run
produced 114 units where HarmScope has 1,101 — a baseline crippled by an
implementation detail would have read as evidence against bag-of-words topic
models.

### Granularity, all five systems, 2024 cutoff, floor 15

| System | Units | p25 | med | p75 | Clears floor | Signals |
|---|---:|---:|---:|---:|---:|---:|
| HarmScope | 2,010 | 7 | 15 | 44 | 51.6% | 29,004 |
| B0 | 12 | 44 | 113 | 354 | 96.8% | 1,451 |
| B1 | 634 | 8 | 18 | 49 | 57.3% | 24,273 |
| B2 | 2,010 | 7 | 16 | 41 | 54.4% | 32,419 |
| B3 | 11,336 | 9 | 19* | 36 | 62.4%* | 55,687 |

\* **B3's support column is not comparable across rows.** It counts complaints;
every other row counts dup-groups. Recorded *before* B3 ran, not discovered
after. The table shows the predicted effect: B3 has 5.6× HarmScope's unit count
yet a *higher* median support, because complaint-denominated counting with no
campaign exclusion more than offsets the finer units. `min_a` in the 2×2 test
inherits the same problem, which is part of why B3 has 55,687 company-level
signals to HarmScope's 29,004.

### Cost

B3's 2024 cutoff took **516 minutes on credit_reporting alone** —
`min_cluster_size=10` produces 8,363 clusters on a 500k fit sample and HDBSCAN's
cost follows. A sensitivity sweep over B3 is not affordable at this setting
without a plan.
