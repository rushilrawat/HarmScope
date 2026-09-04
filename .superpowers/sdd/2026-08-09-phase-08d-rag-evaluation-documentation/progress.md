# SDD ledger — plan: docs/superpowers/plans/2026-08-09-phase-08d-rag-evaluation-documentation.md

Baseline: 6642764; Phase 8C accepted; 492 passed, 5 expected real-database leakage skips; Ruff/format/diff clean.

Preflight data state: the feature worktree is code-isolated and has no private populated database. The real ignored database is `/Users/rushilrawat/HarmScope/data/harmscope.duckdb` with 3,830,002 narratives, 73,122 clusters, and 24,641,664 cluster-member rows. Real-data commands must use an explicit `HARMSCOPE_DATA_DIR=/Users/rushilrawat/HarmScope/data` override; committed code remains only on `codex/phase-8-llm-layer`.

Human-ground-truth constraint: agents may implement the workflow and prepare a private draft worklist, but must not mark model/agent-authored relevance IDs or paraphrase review as human ground truth. A committed benchmark cannot be represented as human-reviewed until a person completes the review fields.

Task 1 implementer: DONE_WITH_CONCERNS — strict manifest, exact DB/privacy
validation, deterministic private export/import, tests, and freeze documentation
implemented. Real ignored draft:
`/Users/rushilrawat/HarmScope/data/interim/rag_eval_authoring.csv` (30 blank
questions/reviews; 5/category; 15 fired/15 control; 30 distinct clusters; all 12
families; zero unsafe formula-prefix cells). Human author/privacy review and the
committed ID-only manifest remain intentionally pending. Verification: 58 passed
+ 1 expected focused skip; full 515 passed + 6 expected skips; Ruff, scoped
format, and diff check clean. Implementation committed with the Task 1 files.

Task 1: fix round 1/5 — four reviewer findings reproduced and addressed:
retrieval-identical scope, canonical alert status, exact CSV row arity, and
failure-atomic durable writes. Corrected real draft independently audits to 15
canonical fired / 15 canonical controls with zero mismatches. Focused
compatibility: 129 passed + 1 expected skip. Open human-curation plan concern:
`company_response` requires public-response evidence, but the mandated worklist
contains narrative excerpts only; no code change made pending plan resolution.
Final verification: focused 129 passed + 1 expected skip; full 526 passed + 6
expected skips; Ruff, scoped format, and diff check clean. The Task 1 fix commit
contains this ledger update.
Task 1 re-review: APPROVED at 4614378 — all four counterexamples fixed; 16 targeted tests and the complete suite pass; private draft independently confirms 15 canonical fired / 15 controls, 300/300 shown IDs retrievable, zero status mismatches, and blank human fields. Engineering implementation accepted. Human manifest freeze and the `company_response` plan contradiction remain open external/design gates.

Task 2 implementer: DONE — exact dense/BM25/fused scoring from one retrieval
call; per-question atomic three-row upserts; caller-autocommit rejection;
prior-question survival; replay preservation of answer/human columns and
created-at identity; answerable-only macros; all-question lexicographic
win/tie/loss; stable ID-only loss-visible rendering with linear p95. Strict TDD:
20 expected missing-interface RED failures, then 25 GREEN; 2 relevance-invariant
RED failures, then final 27 GREEN. Verification: focused 27 passed; retrieval/
schema/isolation slice 149 passed; full 548 passed + 6 expected skips; Ruff,
scoped format, and diff check clean. Implementation commit pending review gate.

Task 2 review at 25e5387: FIXES_REQUIRED — Important: independent fused-field
validation admitted impossible fused wins because membership, component
provenance, RRF score/order, and configured result length were not tied to the
component rankings. Minor: unsafe nonblank `eval_run_id` values were persisted.

Task 2: fix round 1/5 — both findings reproduced with 14 behavioral RED
counterexamples. Fused validation now requires exact output from the shared
`retrieve.reciprocal_rank_fusion` implementation using configured RRF/top-k;
`eval_run_id` now matches `_SAFE_ID` before retrieval. Fake results also use the
real fusion function, and the report demonstrates valid fusion losses. GREEN:
new regressions 19 passed; final Task 2 selection 32 passed; compatibility
slice 163 passed; full 562 passed + 6 expected skips; Ruff, scoped format, and
diff check clean. Fix commit pending scoped re-review.

Task 2 scoped re-review after bf342fc: Important fused-provenance finding
ADDRESSED; Minor run-ID shape finding functionally addressed, but residual
Minor sequencing issue remained because invalid input executed two read-only
autocommit-probe queries before validation.

Task 2: fix round 2/5 — reproduced empty-batch, unsafe-run-ID, and duplicate-
question inputs against a zero-database spy; all three RED cases touched SQL
before their validation error. Pure batch validation now precedes the
autocommit probe. Valid explicit caller transactions still reject before
retrieval/persistence without altering caller state. GREEN: targeted 5 passed;
evaluation 73 passed; compatibility slice 166 passed; full 565 passed + 6
expected skips; Ruff, scoped format, and diff check clean. Fix commit pending
scoped re-review.
Task 2 re-review: APPROVED at 6a8bf4d — zero-I/O invalid inputs, caller-transaction safety, exact fused membership/provenance/score/order/top-k validation, rollback, replay preservation, aggregates, and loss-visible rendering all reproduced with no residual finding.

Task 3 implementer: DONE_WITH_CONCERNS — deterministic citation validity,
coverage, and abstention; typed ID-only answer/failure summary; zero-I/O batch
validation; concrete company-scope gate; all-fused-row preflight; exact typed
AnswerResult evidence validation; fused-only per-question transactions; closed
schema/citation/refusal continuation; terminal provider/preflight propagation;
replay preservation; and exact run+answer usage aggregation implemented. TDD:
17 missing-metric RED then GREEN, 16 missing-runner RED then GREEN, one
coverage-vs-validity semantic RED then GREEN, and two answerability/category
zero-I/O RED then GREEN. Verification: evaluation 109 passed; compatibility
slice 363 passed + 1 expected skip; full 601 passed + 6 expected skips; Ruff,
scoped format, and diff check clean. No live provider call. Concern: 18/30 rows
in the current private draft have blank company scopes and therefore cannot run
through the intentionally concrete-company Phase 8C answer contract; Task 3
rejects the batch before SQL/provider work. Upstream human/controller action is
required before a full paid answer evaluation. Implementation commit pending
review gate.

Task 3: fix round 1/5 — reproduced both Important reviewer findings plus the
related forged-membership counterexample. Zero-completed answer aggregates are
now unavailable (`None`) and render as deterministic `n/a`; batches with at
least one completed answer preserve genuine numeric zero. Before any provider
call, answer evaluation loads each exact retrievable cluster/company/model
corpus, pins its product family and complaint-ID membership, and delegates all
Phase 8C evidence-shape checks to the shared reviewed validator. Product
forgeries, out-of-corpus IDs, overlong evidence, invalid ordering/scores, and
invalid component ranks now fail without metric updates. GREEN: reviewer,
mixed-zero, membership, and scope selection 21 passed; evaluation 127 passed;
compatibility slice 381 passed + 1 expected skip; full 619 passed + 6 expected
skips; Ruff, scoped format, and diff check clean. Replay semantics remain as
reviewer-adjudicated; no live provider call or schema/answer/client edit.
Task 3 re-review: APPROVED at af2e4e7 — unavailable-vs-measured-zero semantics, complete Phase 8C evidence invariants, exact corpus membership preflight, replay, failure, usage, and transaction behavior all reproduced with no residual finding.

Task 4 implementer: DONE_WITH_CONCERNS — exact run/manifest/usage/cache-linked
claim recovery; deterministic 50-claim stratified sampling; cited-only blinded
and spreadsheet-safe private atomic CSV; unforgeable run/question/claim/
evidence identities; strict reviewer decisions and filename provenance; shared
Wilson reporting; fused-only atomic stale-count replacement, replay, and
rollback implemented with synthetic fixtures. TDD included six RED→GREEN waves
for missing interfaces, direct decision/gate validation, manifest-run binding,
question/source aliasing, actual excerpt-content binding, and exact lowercase
tokens. Verification: Task 4 selection 26 passed; evaluation 153 passed;
compatibility slice 321 passed; full 645 passed + 6 expected skips; Ruff,
format, and diff clean. No live provider or human review; actual groundedness
remains an external gate. Task 5 must write `params.manifest_sha256` and enforce
the completed filename suffix `.<reviewer_id>.csv`. Implementation commit
pending review gate.

Task 4: fix round 1/5 — reproduced all three Important reviewer findings plus
the related Python `bool == int` identity edge. Cited evidence now comes from a
cached exact `retrieve.load_corpus` cluster/company/recorded-model scope, so a
valid cross-company dedup-expanded nonrepresentative is admitted without a
direct `cluster_members` row while out-of-scope IDs still fail closed. Public
recording parser-equivalently validates every exact `ClaimReview` and
`ClaimEvidence` field/type before SQL, including booleans masquerading as
integers. Parsing enforces configured `data/interim` containment before opening
the file and leaks no copied private content in the failure. GREEN: reviewer
regressions 3 passed; Task 4 selection 29 passed; evaluation 156 passed;
evaluation/answer/verification/schema/isolation slice 324 passed; full 648
passed + 6 expected skips; Ruff, scoped format, and diff check clean. No
run-finished adjudication requirement, live provider call, human claim, schema,
answer, or verification code was added or changed. Fix commit pending scoped
re-review.
Task 4 re-review: APPROVED at 8e08a4b — dedup-expanded evidence, constructed-type rejection before SQL, interim-only parsing, identity, sampling, Wilson, fused-only replay/rollback, and caller-transaction behavior all reproduced with no residual finding.

Task 5 implementer: DONE_WITH_CONCERNS — lazy full/retrieval-only RAG
evaluation, author/import, claim export/record, exact manifest/run provenance,
provider isolation, connection closure, stable combined rendering, and honest
documentation are implemented. Strict TDD included missing-interface,
run-status-on-render-failure, and exact path/count output RED→GREEN waves.
Fresh verification: required selection 427 passed + 1 expected missing-human-
manifest skip; full repository 659 passed + 6 expected real-data skips; Ruff,
scoped format, and diff check clean. The only network-free real-data command
failed honestly before database bootstrap at the absent frozen manifest, and a
read-only check found zero `rag-eval` runs. No live provider, human judgment,
manifest, metric, or migration was fabricated. Human manifest/privacy freeze,
18 blank company scopes, unsupported `company_response`, legacy embedding
filename migration/regeneration, Anthropic billing/live work, and 50-label/
50-claim reviews remain open. Implementation commit is the enclosing `Complete
Phase 8 RAG evaluation` commit; independent review is pending.

Task 5: fix round 1/5 — reproduced both Important findings and the Minor
documentation error with five focused RED failures. Full evaluation now retains
one frozen scored retrieval per question, rebinds it to exact live corpus plus
same-run persisted dense/BM25/fused metrics, injects a copy of that exact fused
evidence through Phase 8C's existing retriever boundary, and rejects missing,
mutated, forged, or subset evidence before provider work. Import now performs a
complete byte/SHA-bound pure preflight before bootstrap, rejects bootstrap-time
TOCTOU, DB-validates the prepared questions, and atomically writes only prepared
ID-only rows. A second 2-failure RED→GREEN wave removed private authoring and
retrieval content from the new carriers' representations. Engineering notes
distinguish successful 2026-08-07 auth/model
preflight from the subsequent insufficient-credit messages request and state
that Phase 8D did not recheck current balance. GREEN: review regressions 5
passed; required selection 434 passed + 1 expected skip; full repository 665
passed + 6 expected skips; Ruff, scoped format, and diff check clean. No
provider/private/human action. Fix commit pending scoped re-review.

Task 5 re-review: APPROVED at 2557785 — the full CLI performs one retrieval/
query encoding per question and binds answer generation to the exact scored
fused evidence; retained-result provenance, same-run persisted metrics,
retrieval-only isolation, pure byte/SHA-bound import preflight, TOCTOU
rejection, atomic import, privacy-safe representations, and the corrected
billing timeline all reproduced with no residual finding. Fresh reviewer
verification: fix regressions 6 passed; required selection 434 passed + 1
expected missing-human-manifest skip; full repository 665 passed + 6 expected
real-data skips; Ruff, scoped format, diff, and clean-tree checks passed. Phase
8D engineering tasks are complete; live provider, frozen-manifest, embedding-
artifact, and human-review gates remain explicitly external and pending.

Phase 8 final review fix pass (uncommitted by explicit approval-layer
restriction): findings 1–6 implemented with a shared canonical alert scope,
durable label-usage outbox, immutable byte/source-bound human label review,
cluster-run embedding provenance, one-buffer manifest identities, and
descriptor-pinned interim-only private artifact I/O. Finding 7 remains an
expected-failing citation-only legal-conclusion regression plus an extractive-
answer design note; production answer behavior was not changed pending the
user's safety choice. Final network-free verification: evaluation 165 passed;
label runner 21 passed; label verification 33 passed; CLI 23 passed; answer
112 passed + 1 expected xfail; full repository 691 passed + 6 expected
real-data skips + 1 expected xfail (698 collected), exit 0. Ruff whole-repo,
13-file scoped format, identical pre/post-format AST hashes, and diff check
passed. No paid provider, private import, real DB mutation, human judgment, or
gate completion claim. Detailed report: `final-fix-report.md`.

Phase 8 residual final-review fix pass (round 2, still uncommitted by explicit
approval-layer restriction): successful signals/cluster provenance now gates
label and review work; label reports and worklist recording use immutable
export/record-time fired snapshots; DB provenance rechecks run inside the owned
review transaction; paid schema/cache failure fallbacks preserve cost and the
original error with stable, outcome-correct usage IDs; label outbox events are
flat no-follow files below a pinned cache root; private review artifacts are
direct interim-root children with configured-root rename rejection; paired
CSV/sidecar rollback covers all prior-component presence combinations; and
review text strips terminal/bidi controls before formula neutralization. Initial
residual selection: 23 RED failures among 26 cases, then GREEN. Final network-free
verification: evaluation 167 passed; label runner/outbox 30 passed;
label verification plus private-root regressions 54 passed; CLI 23 passed;
answer 112 passed + 1 intentional finding-7 xfail; full repository 723 passed +
6 expected real-data skips + 1 expected xfail (730 collected), exit 0. Ruff whole
repository, 15-file format check, and diff check passed. Finding 7 production
behavior remains paused; no paid provider, private import, real DB mutation,
human judgment, staging, or commit occurred.

Phase 8 residual final-review fix pass (round 3, uncommitted by the same
approval-layer restriction): reproduced eight publication-window failures with
real `os.replace` hooks. Private artifact and label-usage publication now retain
an exact hard-linked prior destination through replace/fsync/root revalidation,
then roll back a newly published final name from the pinned moved root on any
post-publication failure. No-old/old-target and post-publication fsync cases
leave no new private bytes or temporary/backup files; prior bytes are restored.
Paid normal-result and response-received provider-error staging now treat real
outbox root-change errors as fallback-eligible, synchronously persist exact paid
usage before cache publication, and preserve the original stage/provider error
with privacy-safe dual-failure notes. Finding 7 production behavior was not
changed. Final network-free verification: private+label runner 40 passed;
evaluation 167 passed; label verification+private 57 passed; CLI 23 passed;
answer 112 passed + 1 intentional xfail; full repository 731 passed + 6 expected
real-data skips + 1 expected xfail (738 collected), exit 0. Ruff whole repository,
15-file format check, and diff check passed. No staging or commit was attempted.

Phase 8 residual final-review fix pass (round 4, uncommitted by the same
approval-layer restriction): eight strict RED regressions covered both private
artifact and label-usage writers. Backup creation, publication, and backup
deletion now have explicit pinned-directory fsync ordering, including
pre-publication/recovery cleanup. Final backup-cleanup fsync errors fail the
operation while retaining an already root-validated new destination as the
explicit state. Temp close/unlink failures are collected by exception type only,
never mask the original error, and do not prevent remaining recovery/fsync
attempts; backup-collision double faults retain the original `FileExistsError`.
Finding 7 production behavior was not changed. Final network-free verification:
eight new regressions passed; private artifact plus label runner/outbox 48
passed; full repository 739 passed + 6 expected real-data skips + 1 intentional
finding-7 xfail (746 collected), exit 0. Ruff whole-repository, 15-file format
check with identical pre/post-format AST for the one rewrite, and diff check
passed. No staging or commit was attempted.

Phase 8 residual final-review lifecycle pass (round 5, uncommitted by the same
approval-layer restriction): six strict RED regressions showed both private and
label-usage publishers could return after a configured-root rename during
successful backup unlink, and lacked a final identity check for old/no-old
destinations. Both now retain an open no-follow descriptor to exact old bytes
through backup unlink/fsync, finally revalidate root identity, and on mismatch
restore exact bytes via a file-fsynced atomic temp replacement or restore
absence, followed by directory fsync. Tests prove empty recreated roots, no new
private bytes/event, no temp/backup names, and correct recovery fsync counts.
Finding 7 production behavior was not changed. Final network-free verification:
focused lifecycle 6 passed; private artifact plus label runner/outbox 52 passed;
full repository 743 passed + 6 expected real-data skips + 1 intentional
finding-7 xfail (750 collected), exit 0. Ruff whole-repository, 15-file format,
and diff checks passed. No staging or commit was attempted.

Phase 8 residual inode-binding pass (round 6, uncommitted by the same
approval-layer restriction): six strict RED regressions showed that reopening a
backup pathname could bind recovery to substituted bytes and that pin-to-link
destination swaps/appearances were accepted; four symlink/nonregular controls
were already green. Both writers now pin the original destination with a
descriptor-relative no-follow open before backup creation, require a regular
inode, verify the linked backup's device/inode against that original fd, and use
only the original fd for recovery. Existing-target disappearance/disagreement
restores exact old bytes before publication; absent-to-appeared races fail
closed without deleting the concurrent file. Focused selection 10 passed;
private artifact plus label runner/outbox 62 passed. Final network-free
verification: full repository 753 passed + 6 expected real-data skips + 1
intentional finding-7 xfail (760 collected), exit 0. Ruff whole-repository,
15-file format with identical ASTs for all three round-6 rewrites, and diff
checks passed. Finding 7 was not changed; no staging/commit occurred.

Phase 8 residual descriptor-teardown pass (round 7, uncommitted by the same
approval-layer restriction): two strict RED regressions showed snapshot-fd
close `OSError` masking an active root-change containment exception after safe
exact-old rollback; two successful-publication close-failure cases were already
green. Both writers now preserve the active `PrivatePathError` or
`LabelUsageOutboxError`, attach a fixed type-only cleanup note, and continue
root/parent descriptor cleanup. With no active error the close failure remains
primary and verified new state remains explicit. Focused 4 passed; private
artifact plus label runner/outbox 66 passed. Final network-free verification:
full repository 757 passed + 6 expected real-data skips + 1 intentional
finding-7 xfail (764 collected), exit 0. Ruff whole-repository, 15-file format,
and diff checks passed. Finding 7 was not changed; no staging/commit occurred.

Phase 8 extractive answer-safety Task 1 (2026-08-12, uncommitted by the same
approval-layer restriction): the approved finding-7 decision is implemented.
Strict TDD RED was 59 failed / 276 passed (335 collected): the validator had no
evidence-text boundary, citation-only prose still passed provider/evaluation
paths, rendering lacked deterministic allegation attribution, and config was
still `rag-v2`. `validate_answer` now requires exact retrieved redacted evidence,
uses NFC plus whitespace collapse and case-sensitive contiguous matching, rejects
malformed/unsafe/overlength output without truncation, and permits multi-ID
claims only when at least one cited source contains the extract. Provider,
cache, and evaluation/review-export paths propagate exact evidence; evaluation
reuses its retrievable-corpus cache. Wire answers remain joined extracts, while
local Answer and Claims sections render `Complaints allege: “<extract>”` with
declared IDs. The prompt requests verbatim excerpts without model-added framing,
claim-review identity remains bound to extract/citations/source/input identity,
and answer prompt/cache version is `rag-v3`. Final network-free verification:
focused answer/eval/CLI/config 359 passed; full repository 784 passed + 6
documented real-data skips (790 collected), exit 0; Ruff whole repository,
7-file scoped format, and diff checks passed. No finding-7 xfail remains. No live
provider, private DB, human judgment, staging, commit, or other Git metadata
mutation occurred; measured model quality and human groundedness remain pending.

Phase 8 extractive answer-safety Task 1 reviewer fix round 1/5 (2026-08-12,
uncommitted by the same approval-layer restriction): ten strict RED regressions
reproduced six residual boundaries. Evaluation now retains byte-exact live evidence
text, rejects returned text substitutions, and revalidates the complete typed answer
before metrics; typed semantic failures keep null metrics and the existing
`citation`/`schema` taxonomy. `citation_validity` is no longer callable as an
ID-only grounding check. `render_cli` revalidates against exact returned evidence,
uses canonical values, and the evidence-free synthesis helper is private. Local
attribution reversibly escapes backslashes, curly quotes, and brackets. Retrieved
IDs are positive exact integers before hash/cache/provider work; mixed object keys
remain typed schema failures with paid accounting; Unicode noncharacters are
rejected while ZWJ/ZWNJ/combining text remains supported. Targeted GREEN was 15
passed; pre-final focused answer/eval/CLI/config was 369 passed; pre-final full
repository was 794 passed + 6 documented real-data skips (800 collected). Final
counts and static checks are recorded in the Task 1 report after the required
post-documentation rerun. No live/provider/private/Git mutation occurred.

Task 1 reviewer fix round 1/5 final post-documentation verification: focused
answer/eval/CLI/config 371 passed; full repository 796 passed + 6 documented
real-data skips (802 collected), exit 0; Ruff whole repository, seven-file format,
and diff checks passed. No live provider, private DB, human judgment, staging,
commit, reset, clean, or other Git metadata mutation occurred.

Phase 8 final whole-branch fix wave (2026-08-12): strict TDD selected 26
regressions before production edits; 25 failed for the three findings and the
exact answer-usage replay control passed. Private and label publishers now copy
prior bytes into independent unlinked 0600, file-fsynced inodes, verify source
metadata/name/bytes around copy and link, and recover only from the immutable
copy. Answer usage insert-ignore now compares all 18 stored fields and retains
conflicting events. One versioned canonical renderer drives CLI and claim-review
export; review identity v2 binds its exact output hash/version without private
prose. Strict GREEN was 26/26; private/run 70/70;
answer/client/eval/CLI/isolation 391/391; final full repository 822 passed + 6
documented real-data skips (828 collected), exit 0. Ruff whole-repository,
format-check for all 18 dirty Python files, and diff checks passed. No live,
private-human, real-DB, staging, commit, reset, clean, or history work occurred.
Embedding migration operational gate completed 2026-09-03: independent review approved; real plan `planned`; execute `migrated`; replay `already-migrated`; 2,477,937 × 384 strict loader passed; target/retirement hashes and inode contract passed; DuckDB evidence unchanged; forced-offline retrieval returned five scoped complaint IDs; no provider call. Human manifest, provider billing/live runs, and 50-label/50-claim reviews remain open.
