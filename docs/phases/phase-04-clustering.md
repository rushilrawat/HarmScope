# Phase 4 — Clustering & novelty **[GATE]**

**State:** ✅ gate passed; ARI 0.505 recorded as a limitation · **Effort:** ~1.5 weeks

## Question

Are there dense regions in narrative space that correspond to harm *mechanisms*,
and are any of them not already represented by the CFPB taxonomy?

## Context

"Novelty" is the project's whole claim, so it needs a definition that cannot be
satisfied by an incoherent blob. A cluster that is 40% one issue and 60% noise
looks novel by entropy and means nothing — hence the coherence and persistence
guards on top of the novelty score.

The other tension: HDBSCAN labels 54–61% of a fit sample as noise. If sampled
points kept that `-1` while unsampled points were assigned by nearest centroid,
two identical narratives would get different fates according to which was drawn.
So HDBSCAN's noise call *reports discovery quality*, and one threshold decides
membership for everyone.

## What runs

Per product family:

1. Sample 500k representatives (seeded, sorted so memmap reads stay sequential).
2. UMAP → 10 components.
3. HDBSCAN with `leaf` selection.
4. Centroids computed in the **original** space, not UMAP space.
5. Assign *every* representative by nearest centroid within
   `assign_max_distance`.
6. Novelty score against the taxonomy; coherence and persistence guards.

## Tech

| Choice | Why this one |
|---|---|
| **UMAP** `n_neighbors=30, n_components=10, min_dist=0.0` | Preprocessing for density clustering, not visualisation — hence 10 components, not 2. |
| **HDBSCAN** `min_cluster_size=50, min_samples=10`, `cluster_selection_method='leaf'` | No fixed *k*, native noise class. `leaf` gives finer clusters — mechanisms rather than topics. This is a deliberate choice whose bill arrives in Phase 7. |
| **Centroids in original space** | UMAP is fit to one sample; a distance in it is meaningless for a point the fit never saw. Cosine in the original space is where `assign_max_distance` is interpretable. |
| **Nearest-centroid assignment**, not `approximate_predict` | Would require `umap.transform` on ~1.4M unseen points — slow, and conceptually an extrapolation. |
| **`random_state` on UMAP** | Disables its parallelism and costs real wall time. Standing rule 3 requires determinism, and unreproducible clustering makes the stability numbers meaningless. |

## Acceptance

> ARI between disjoint halves reported whatever it says · noise fraction per
> family · label-ablation AUC ≥ 0.7 · name 15 random clusters.

| Criterion | Result | Verdict |
|---|---:|---|
| Label-ablation AUC ≥ 0.7 | **0.7909** | pass, 10 of 11 families |
| Disjoint-halves ARI (reported, not gated) | **0.5051** | limitation |
| Name 15 random clusters | 15 / 15 | pass |
| Noise fraction per family | 49–72% | reported |

## Findings

**The ablation failed first at 0.6658, and the cause was a conformance bug.**
`dominant_label_share` divided by *surviving* labels rather than by cluster size.
So a 101-member cluster that was 99% the hidden issue scored novelty **0.000** —
the least-explained cluster in its family, rated perfectly explained.

The spec says "fraction of *members*", so fixing it was conformance, not tuning
toward a passing number. That distinction was checked rather than asserted:
recomputing all 2,821 production novelty scores under the fix gives a largest
difference of **0.00e+00**. Zero scores moved, so the fix cannot have been tuned
toward an outcome it does not touch.

**ARI 0.505 is the number to carry forward, and it is a limitation.** Cluster
counts across two independent halves agree to within 1% (688 vs 693) and
assignment fractions match to the decimal — but *which* cluster a complaint lands
in agrees about half the time.

**A cluster here is a region of a dense neighbourhood, not a canonical object.**
Nothing downstream may treat cluster identity as stable across refits. Cluster
count was still climbing at the 500k fit sample, so the partition has not
converged.
