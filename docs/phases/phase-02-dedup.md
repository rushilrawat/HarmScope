# Phase 2 — Dedup & campaign detection **[GATE]**

**State:** ✅ gate passed, on a changed criterion · **Effort:** ~1 week

## Question

Which complaints are independent allegations, and which are one template mailed
many times?

## Context

This is the core preprocessing problem, not a nicety. A single credit-repair
template can appear tens of thousands of times. Counting those as tens of
thousands of complaints would let one filing service manufacture a growth signal
against any company it targets — and the whole project is a growth detector.

Getting this wrong does not produce a visible error. It produces a confident
signal about a harm that is one person's mail merge.

> Do not begin Phase 3 until this passes. Every downstream result is invalid
> otherwise.

## What runs

1. Exact hash grouping on normalized text.
2. MinHash + LSH for near-duplicates → `dup_pairs`.
3. **Star clustering** into `dup_groups` (not union-find — see findings).
4. Campaign features and flagging.
5. 300 eval pairs with exact-Jaccard reference labels; blind adjudication of the
   strata where detector and reference disagree.

## Tech

| Choice | Why this one |
|---|---|
| **`datasketch` MinHash + LSH** | Sub-quadratic near-dup search over 2.5M distinct texts. |
| **Star clustering**, not union-find | Bounds group diameter from a seed, so transitive chaining cannot happen by construction. |
| **Exact groups collapsed first** | Byte-identical texts are atomic; otherwise a seed could admit some members of an exact group and not others. |
| **Representative = earliest `date_received`**, ties on lower id | Deterministic and backward-looking, so a group gaining members after a cutoff cannot retroactively change what a pre-cutoff run saw. |

## Acceptance

> Precision ≥ 0.95, recall reported, plus a merge audit, campaign share per
> family, and 20 flagged + 20 unflagged groups read by hand.

**Passed, on a changed criterion.** ROADMAP asked for 300 hand-labelled pairs.
No human was available, so disagreements were adjudicated by the model, blind,
against a rule written down beforehand (`METHODOLOGY §2.4.1`). Recorded under
*Reversed decisions*, not presented as satisfying the original bar.

## Findings

**Union-find chained templates into one 84,657-member group.** Transitive
closure over near-duplicate edges merges A–B and B–C into A–C even when A and C
share nothing.

| Grouping | Precision | Recall | Largest group | In groups >1000 |
|---|---:|---:|---:|---:|
| Union-find | 0.9112 | 0.9171 | 84,657 | 1,803,654 |
| Star clustering | 0.9477 | 0.8011 | 49,457 | 702,032 |
| After adjudication | **1.0000** | 0.7650 | 49,457 | 702,032 |

**The precision number is measured on 8% of the merges.** This is the finding
that matters. The eval set samples pairs that have a *verified edge*, but star
clustering merges through a seed — in a 49,457-member star, 49,456 of roughly
1.2 billion member-pairs are edges. Sampling same-group pairs the way the corpus
actually holds them: **3 of 40 have a verified edge; 37 are seed-mediated.**

A precision figure without that denominator is a figure about a twelfth of the
merges. Twelve seed-mediated merges were read by hand; all twelve are the same
template.

**Campaign-flagged share runs credit reporting 29.9%, mortgage 0.07%** — the
required direction. All 20 flagged campaigns read as templated. So did the 20
largest *unflagged* groups, which is a known miss: a 24,507-member group scored
2 of 5 campaign signals because it cites no statute. That miss is handled in
Phase 5 by counting dup-groups rather than complaints, which makes the flag's
failure harmless.
