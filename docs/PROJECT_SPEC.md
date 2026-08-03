# PROJECT_SPEC

## 1. Problem

The CFPB publishes millions of consumer complaints. Each is labeled by the consumer with a
`Product / Sub-product / Issue / Sub-issue` from a fixed menu. That taxonomy is:

- **Coarse.** "Incorrect information on your report" covers dozens of distinct mechanisms.
- **Consumer-selected.** Consumers pick the closest available option, not the correct one.
- **Schema-drifting.** Categories were revised (notably a large 2017 restructuring). Time
  series across the change are not directly comparable.
- **Backward-looking.** A genuinely new harm mechanism has no category until someone adds one.

So a harm that does not fit an existing label is effectively invisible in aggregate statistics,
even when hundreds of narratives describe it precisely.

## 2. Thesis

Narratives contain harm mechanisms that the label schema does not encode. Those mechanisms can
be discovered without supervision, tracked over time, and — critically — their emergence can be
shown to *precede* public enforcement in a measurable number of historical cases.

## 3. Falsifiable research question

> Does unsupervised harm-mechanism discovery over complaint narratives produce company-level
> alerts with greater lead time before public CFPB enforcement actions than monitoring growth
> in the existing `(Product, Issue, Sub-issue)` taxonomy, at equal or lower false-alert rate?

Three outcomes, all publishable in the README:

| Outcome | Interpretation |
|---|---|
| Discovery beats taxonomy on lead time at ≤ baseline false-alert rate | Contribution confirmed |
| Discovery matches taxonomy | Method works, contribution is the pipeline not the signal — say so |
| Discovery is worse | Negative result — report it, with failure analysis |

A negative result honestly reported is a better portfolio artifact than a positive result with
leakage. This is non-negotiable.

## 4. Scope

### In scope (v1)

- Date window: **2015-01-01 → present**. Pre-2015 narrative coverage is too thin.
- Narratives only: `Consumer complaint narrative IS NOT NULL`.
- All products, but stratified analysis by product family, because credit reporting will
  otherwise dominate every cluster.
- Backtest ground truth: **CFPB public enforcement actions, 2017-01-01 → 2024-12-31.**
  The lower bound is the first annual backtest cutoff; earlier actions have no
  cutoff preceding them and are unevaluable (`EVALUATION.md` §1.2). The upper
  bound is the 2025 posture change (§5 below).

### Out of scope (v1)

- Non-CFPB complaint sources (BBB, state AG, FTC Sentinel).
- Company financial data, call transcripts, or any non-public source.
- Predicting individual complaint outcomes or company responses.
- Real-time streaming. Batch, daily refresh at most.

### Explicit deferrals (v2 candidates)

- Cross-company shared-vendor inference (detecting a common back-end service provider from
  co-occurring harm patterns). High value, high difficulty. Do not attempt before Phase 9.
- Multilingual narratives (Spanish-language complaints exist).
- Fine-tuned domain embeddings.

## 5. Critical scoping constraint: the CFPB's 2025 posture change

CFPB enforcement activity changed substantially in 2025 — investigations were closed, consent
orders terminated, and pending actions dropped or realigned. Practical consequences:

1. **Enforcement actions are only usable as ground truth through 2024.** Post-2024, absence of
   an enforcement action does not mean absence of harm. Treating 2025+ as negative labels
   would be wrong.
2. The backtest window is therefore **2017–2024** — bounded above by the 2025
   posture change and below by the first annual cutoff. State both limitations
   prominently.
3. Complaint *intake* continues, so detection on recent data is still meaningful — it is just
   unvalidatable by enforcement. Present recent signals as unvalidated candidates only.
4. Verify the current state of the database and its API before starting Phase 1. Federal data
   availability has been volatile; snapshot the bulk CSV locally on day one and treat that
   snapshot as immutable input.

## 6. Success criteria

### Must have (project is not done without these)

- [ ] ≥ 20 **usable** hand-curated enforcement actions in the ground-truth set, each mapped
      to a company present in the complaint DB. Note this is tighter than it looks: the
      2017–2024 window and the `usable = false` exclusions both prune the set, so curate
      well above 20 to land above 20.
- [ ] Template / mass-filing detection with measured precision & recall on a 300-pair
      hand-labeled sample.
- [ ] Novelty scoring that demonstrably separates known-taxonomy clusters from off-taxonomy
      clusters (validated on held-out labels).
- [ ] Point-in-time backtest with no post-cutoff information reachable by the model.
- [ ] At least 3 baselines implemented and beaten-or-not-beaten honestly.
- [ ] Every alert in the UI traceable to its supporting complaint IDs in one click.

### Should have

- [ ] LLM cluster labels with a human-verified sample (≥ 50 clusters) and reported agreement.
- [ ] Calibration analysis of the growth statistic.
- [ ] Per-product-family breakdown of performance.

### Nice to have

- [ ] Public deployed read-only instance.
- [ ] Written short paper / blog post with the negative results included.

## 7. Non-goals and hard rules

- Never render a verdict. UI language is "complaints allege", never "company did".
- Never surface a named individual. Narratives are PII-scrubbed by CFPB but residue exists —
  run a secondary PII scan and redact before display (see `docs/DATA.md §6`).
- Never let the LLM decide whether a signal fires. LLM output is descriptive metadata attached
  to a signal that statistics already produced.
- Never present an unvalidated recent signal next to backtested metrics without a visual
  distinction.

## 8. Intended users

Consumer-protection analysts, state AG staff, compliance teams, investigative journalists,
academic researchers. All of these need *evidence links*, not scores — design accordingly.
