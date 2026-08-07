# Phase dossiers

One file per ROADMAP phase. Each carries the same six sections:

| Section | What it holds |
|---|---|
| **Question** | What the phase is actually for — the thing that would be unknown without it |
| **Context** | Why it is built this way, including the constraints that forced the design |
| **What runs** | The concrete steps, in order |
| **Tech** | Libraries, parameters, and why each was chosen over the alternative |
| **Acceptance** | The ROADMAP criterion, and whether it was met, changed, or missed |
| **Findings** | What actually happened — measured numbers, bugs found, decisions reversed |

These are navigational and explanatory. They are **not** the source of truth:

- `docs/ROADMAP.md` owns the acceptance criteria.
- `docs/ENGINEERING_NOTES.md` owns the running log and every number's provenance.
- `README.md` owns the results table.

Where a dossier and one of those disagree, the other three win and the dossier is
the defect.

## Status

| Phase | Dossier | State |
|---|---|---|
| 0 | [Scaffold](phase-00-scaffold.md) | ✅ complete |
| 1 | [Ingestion & normalization](phase-01-ingestion.md) | ✅ complete |
| 2 | [Dedup & campaign detection](phase-02-dedup.md) | ✅ **gate** passed on a changed criterion |
| 3 | [Embedding & index](phase-03-embedding.md) | ✅ complete, on the dev model |
| 4 | [Clustering & novelty](phase-04-clustering.md) | ✅ **gate** passed; ARI 0.505 recorded as a limitation |
| 5 | [Signal detection](phase-05-signals.md) | ✅ complete; negative control 0.45% vs α 0.05 |
| 6 | [Ground truth & backtest](phase-06-backtest.md) | ✅ **gate** passed |
| 7 | [Baselines](phase-07-baselines.md) | ✅ complete — all four systems |
| 8 | [LLM layer](phase-08-llm.md) | ⬜ not started |
| 9 | [Evaluation & write-up](phase-09-evaluation.md) | 🔄 two items pulled forward |
| 10 | [Interface](phase-10-interface.md) | ⬜ not started |

## The result, in one line

`B1 (56.2%) > B2 = B3 (51.8%) > HarmScope (50.9%) > B0 (41.1%)` on 112 enforcement
actions. HarmScope beats naive volume and is indistinguishable from every other
unit definition tried. See [Phase 7](phase-07-baselines.md).
