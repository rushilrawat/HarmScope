# DATA

## 1. Source

**CFPB Consumer Complaint Database.** Public, no API key, no authentication.

- Field reference: `https://cfpb.github.io/api/ccdb/fields.html`
- API docs: `https://cfpb.github.io/ccdb5-api/documentation/`
- Bulk download + interactive search: `https://www.consumerfinance.gov/data-research/consumer-complaints/`

**Use the bulk CSV, not the API.** The API paginates and offset pagination is unreliable at
depth; cursor-based `search_after` works but pulling millions of records over HTTP is slow and
fragile. Download the full CSV once, snapshot it, treat it as immutable.

Publication rule: complaints are published after the company responds or after 15 days,
whichever comes first. The database generally updates daily.

## 2. Field reference

| Field | Type | Notes |
|---|---|---|
| `Complaint ID` | number | Primary key |
| `Date received` | date | The date CFPB received the complaint. **Use this as the time axis.** |
| `Date sent to company` | date | Usually `Date received` + small lag |
| `Product` | categorical | Consumer-identified product |
| `Sub-product` | categorical | Not all Products have Sub-products |
| `Issue` | categorical | Possible values depend on Product |
| `Sub-issue` | categorical | Depends on Product **and** Issue; not all Issues have one |
| `Consumer complaint narrative` | text | **Opt-in only.** PII-scrubbed by CFPB. The core input. |
| `Company` | categorical | Free-ish text with canonical-ish casing (`EQUIFAX, INC.`) |
| `Company public response` | text | Optional, chosen from a preset list |
| `Company response to consumer` | categorical | `Closed with explanation`, `Closed with monetary relief`, `Closed with non-monetary relief`, `Closed`, `In progress` |
| `Timely response?` | yes/no | |
| `State` | categorical | Consumer mailing address state |
| `ZIP code` | text | **Privacy-suppressed.** 5-digit published unless the consumer is in a Census ZCTA under 20,000 people *and* consented to narrative publication — then 3-digit, or nothing. This means ZIP presence correlates with narrative consent. Do not use ZIP as a feature. |
| `Tags` | text | `Older American`, `Servicemember` — used for subgroup analysis |
| `Submitted via` | categorical | Web, phone, referral, etc. |

## 3. The five quirks that will break naive analysis

### 3.1 Narrative opt-in is not random

Only complaints where the consumer consented have narratives. Consent correlates with
literacy, product type, submission channel, and third-party assistance. **Narrative-bearing
complaints are a biased subsample.** Never state a rate over narrative complaints as if it were
a rate over all complaints. Always report the narrative coverage fraction alongside any count.

### 3.2 Mass filing and templates — the single biggest threat

Credit repair organizations and complaint-filing services submit large volumes of
near-identical narratives on behalf of consumers. Untreated, these produce:

- Enormous, tight, fast-growing clusters that look exactly like an emerging harm signal.
- Company-level disproportionality that reflects who is being targeted by credit repair
  marketing, not who is harming consumers.

**Every "emerging harm" you find will be a credit repair template unless you solve this first.**
Template detection is Phase 2 and is a gate — no clustering until it passes its acceptance
criteria. See `docs/METHODOLOGY.md §2`.

Signatures of templated filings:
- Identical or near-identical narrative bodies with swapped account numbers / dates.
- Boilerplate statutory citations (FCRA section references) in unnatural density.
- Bursts of submissions from the same state in a short window.
- Very high volume concentrated on the three nationwide credit reporting agencies.

### 3.3 Credit reporting dominates volume

Complaints against the nationwide credit reporting agencies are a large fraction of the whole
database and swamp everything else. If you cluster the full corpus globally, most clusters will
be credit-reporting dispute variants.

**Mitigation:** stratify. Cluster within product family, then reconcile. Report per-family
metrics. Never report a single global cluster count as a headline.

### 3.4 Taxonomy schema drift

`Product` / `Issue` values were restructured (a major revision took effect in 2017; smaller
changes since). Consequences:

- Raw taxonomy time series have artificial discontinuities at schema boundaries.
- The baseline you are comparing against must use a **crosswalk** mapping old labels to new,
  or be evaluated only within a stable schema window. Build the crosswalk in Phase 1 and store
  it as `taxonomy_crosswalk` — this is a real dataset contribution, keep it in the repo.

### 3.5 Company name variants

Same entity under multiple strings, plus subsidiary/parent ambiguity (a bank's mortgage
servicing arm vs the bank). Enforcement actions name legal entities that may not match the
complaint-DB string at all.

**Mitigation:** a `company_canonical` table built with:
1. Normalization (case, punctuation, corporate suffixes: INC / LLC / N.A. / NATIONAL ASSOCIATION).
2. Fuzzy blocking + scoring (`rapidfuzz` token_set_ratio).
3. **Manual review of the top 300 companies by complaint volume.** These cover the vast
   majority of the corpus. Hand-verify them. Store the mapping as a checked-in CSV.

Do not automate away step 3. A wrong company merge silently corrupts every downstream signal.

## 4. Ground truth: enforcement actions

### Source

CFPB enforcement actions are published at `consumerfinance.gov/enforcement/actions/`.
There is **no clean bulk API** for these. Expect to scrape the listing and hand-curate.

### Construction protocol

Build `data/ground_truth/enforcement_actions.csv` by hand. Target **≥ 20, ideally 30–40**
actions in **2017–2024**. Per action record:

> The window starts at the first backtest cutoff, not 2016. An action filed
> before 2017-01-01 has no annual cutoff strictly preceding it and cannot be
> evaluated (`EVALUATION.md` §1.2). Curate 2016 actions if you find them, but
> mark them `usable = false, exclusion_reason = pre-first-cutoff`.

| Column | Description |
|---|---|
| `action_id` | slug |
| `filed_date` | date the public action was filed/announced — **this is the label date** |
| `company_raw` | company name as published |
| `company_canonical_id` | FK into `company_canonical`; null if no complaint-DB match |
| `product_family` | mapped to complaint-DB product family |
| `harm_summary` | 1–2 sentences, your words |
| `harm_keywords` | terms an analyst would expect in matching narratives |
| `conduct_start` | approximate start of alleged conduct, if stated in the action |
| `source_url` | |
| `usable` | bool — false if no company match or no narratives in window |

### Selection rules (avoid cherry-picking)

- Select actions **before** running any detection. Freeze the file, commit it, record the
  commit SHA in `EVALUATION.md`. Adding actions after seeing results is p-hacking.
- Include actions you expect to fail (e.g. conduct affecting few consumers, or a company with
  low complaint volume). A ground-truth set of only easy cases inflates results.
- Exclude actions where alleged conduct is not consumer-observable (e.g. internal recordkeeping
  violations) — consumers cannot complain about what they cannot see. Mark `usable = false`
  with a reason, keep the row for transparency.

### Known caveat

`filed_date` is when CFPB went public, not when CFPB *knew*. Lead time measured against
`filed_date` is therefore lead time vs. public disclosure, not vs. regulator awareness. State
this explicitly whenever you report the metric.

## 5. Volume and scale planning

| Slice | Approx. order of magnitude | Notes |
|---|---|---|
| All complaints, all time | 10⁷ | Verify actual count at ingest; log it |
| With narrative | ~10⁶–10⁷ | Compute exact ratio in Phase 1, record in this file |
| 2015+, with narrative, post-dedup | target ≤ 3×10⁶ | This is the embedding workload |

Embedding 3M narratives: ~20–40 min on a single modern GPU, ~2–5 h on CPU with batching.
Both acceptable. Clustering 3M points is **not** — see `docs/METHODOLOGY.md §4` for the
sample-then-assign strategy.

## 6. Privacy handling

CFPB scrubs PII, but scrubbing is imperfect and narratives are consumer-written.

Required before any narrative is displayed or sent to an external API:

1. Regex sweep for: SSN patterns, full 16-digit card numbers, account numbers ≥ 8 digits,
   email addresses, phone numbers, street addresses.
2. Replace with typed placeholders (`[ACCOUNT]`, `[EMAIL]`) — do not delete, the placeholder
   preserves sentence structure for embeddings.
3. Log redaction counts per run. A sudden change in redaction rate signals a schema change
   upstream.
4. Never log raw narratives at DEBUG level. Never commit narrative text to git.

**This constrains `dedup_eval_pairs.csv`**, which `ARCHITECTURE.md` §2 marks as
COMMITTED. The two rules coexist only if that file holds complaint **ids** and a
label — `complaint_id_a, complaint_id_b, label, stratum, notes` — with the
narrative text joined from DuckDB at evaluation time. `tests/test_ground_truth.py`
enforces this, because 300 hand-labelled narratives landing in git history
cannot be walked back; the only workable fix is never letting the first one land.

Company names are fine to commit. The curator's own `harm_summary` prose in
`enforcement_actions.csv` is fine — it is written by you, not by a consumer.

Names of *companies* are fine. Names of *individuals* are not — if a narrative names a bank
employee, redact it.
