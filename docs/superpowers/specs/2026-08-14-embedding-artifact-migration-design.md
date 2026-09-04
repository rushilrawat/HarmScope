# Embedding Artifact Migration Design

**Date:** 2026-08-14
**Status:** Fix Round 5 reviewed; real execution blocked pending separate pre-execution review
**Scope:** Phase 8 operational prerequisite only

## 1. Goal

Migrate HarmScope's completed MiniLM embedding artifacts from the historical
model-tail filenames to the collision-free SHA-256 model-identity filenames
required by the current loaders, without recomputing vectors, copying roughly
7.6 GB of data, changing the database, or adding a runtime legacy fallback.

The concrete model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Its expected model-identity digest is:

```text
e9e2c8815f70dad48afc257ffd2bcf61b63bac429fad1ba1eaefa245bb562a13
```

The migration is an explicit maintenance action. Normal embedding, clustering,
retrieval, and RAG code continues to accept only the SHA-addressed artifacts.

## 2. Motivation

The completed artifacts currently use the model tail:

```text
embeddings.all-MiniLM-L6-v2.npy
embeddings.all-MiniLM-L6-v2.progress.json
faiss.all-MiniLM-L6-v2.index
```

Phase 8 deliberately changed artifact addressing to hash the full model
identity. That avoids collisions such as two providers publishing different
models with the same tail. The strict loader is correct, but the completed
MiniLM artifacts predate it, so existing cluster runs cannot load their vectors
until the artifacts are migrated or regenerated.

Regeneration is not the first choice: it would repeat 2,477,937 encodes, consume
substantial time and compute, and could produce different floats under newer
model-library versions. A validated namespace migration preserves the exact
vectors used by the existing clusters.

## 3. Non-goals

- Do not re-encode narratives.
- Do not change `embedding_map`, cluster membership, signals, labels, caches, or
  run records.
- Do not change `CONFIG.embed.model` or reinterpret the MiniLM artifacts as BGE.
- Do not add legacy lookup behavior to any runtime loader.
- Do not copy or print complaint narratives.
- Do not silently overwrite an existing SHA-addressed artifact.
- Do not claim model quality, retrieval quality, or human evaluation completion.

## 4. Interface

Add one top-level maintenance command:

```bash
python -m src.pipeline migrate-embeddings \
  --model sentence-transformers/all-MiniLM-L6-v2
```

The command is read-only by default. It validates the complete migration and
prints the planned source and destination basenames. Mutation requires an
explicit flag:

```bash
python -m src.pipeline migrate-embeddings \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --execute
```

`--model` is required. Falling back to the configured production model would
be unsafe because the completed artifacts and successful cluster runs use the
MiniLM development model while the current production default is BGE.

The command opens HarmScope's database read-only and emits no narrative text.
Its result reports the model, row count, dimension, artifact basenames,
validation status, and whether the state is `planned`, `migrated`, or
`already-migrated`.

## 5. Artifact identities

The existing helper remains the sole source of target names:

```python
embedding_artifact_paths(PATHS.artifacts, model_name)
```

For MiniLM, the targets are:

```text
embeddings.e9e2c8815f70dad48afc257ffd2bcf61b63bac429fad1ba1eaefa245bb562a13.npy
embeddings.e9e2c8815f70dad48afc257ffd2bcf61b63bac429fad1ba1eaefa245bb562a13.progress.json
faiss.e9e2c8815f70dad48afc257ffd2bcf61b63bac429fad1ba1eaefa245bb562a13.index
```

Legacy names are derived only from a validated, path-separator-free model tail.
Every path that exists must be a direct regular-file child of the configured
artifact directory; target paths may of course be absent before publication.
Symlinks, directories, special files, and paths outside that directory fail
before publication.

## 6. Validation contract

Validation runs in both plan and execute modes. It is performed against the
legacy set before publication and against the target set before legacy names
are removed.

### 6.1 Progress sidecar

The JSON sidecar must:

- contain the expected fields and types;
- record the exact full model identity;
- have `n_done == n_total`;
- have positive row and dimension counts;
- agree with the memmap, FAISS index, and database metadata.

For the real MiniLM artifact, the expected values are 2,477,937 rows and 384
dimensions.

### 6.2 NumPy artifact

Load the `.npy` artifact read-only with memory mapping and require:

- exactly two dimensions;
- `float32` dtype;
- shape `(n_total, dim)` from the sidecar;
- only finite values;
- unit-normalized rows within the established `1e-3` tolerance.

The finite and norm checks scan in bounded chunks so validation does not load
the 3.8 GB matrix into memory.

### 6.3 FAISS artifact

Load the index read-only and require:

- exact inner-product flat-index semantics;
- `d == dim`;
- `ntotal == n_total`;
- deterministic reconstructed samples equal the corresponding memmap rows
  within float32 tolerance.

The deterministic sample includes the first and last rows plus evenly spaced
rows across the full artifact. This binds the index to the memmap without
building a second 3.8 GB index or relying only on matching dimensions.

### 6.4 Database mapping

Against a read-only DuckDB connection, require for the exact model:

- one consistent dimension equal to the sidecar dimension;
- row indices are contiguous from `0` through `n_total - 1`;
- exactly `n_total` distinct row indices;
- no complaint maps outside the artifact;
- every mapped complaint's narrative text hash resolves to the same stable
  `ORDER BY text_hash` row used by `build_map`;
- all current narratives eligible for the completed encode are mapped.

No database statement may write, bootstrap, migrate, or register a run.

## 7. Publication and recovery

The migration uses hard links inside the configured artifact directory. A hard
link gives the SHA-addressed name the exact existing inode and bytes, while
requiring negligible additional disk space.

Publication follows this order:

1. Open and pin the artifact directory, then resolve every name relative to its
   directory descriptor.
2. Validate all three legacy artifacts and the database contract.
3. Create target hard links without following symlinks and without overwrite.
4. Publish the memmap and FAISS names first; publish the sidecar last as the
   completed-artifact marker.
5. Fsync the artifact directory.
6. Validate the complete target set independently, including target/source
   inode equality.
7. Remove legacy names only after target validation succeeds.
8. Fsync the artifact directory again and perform a final target validation.

If publication fails before the complete target set validates, remove only the
target names created by this invocation, fsync the cleanup, and preserve every
legacy name. Never replace an existing target.

The implementation binds the source inode before validation and rechecks path
identity before and after publication. A concurrent source replacement or
artifact-root rename fails closed; it must never cause a different regular file
to be published under the validated identity.

If legacy cleanup is interrupted, the validated SHA-addressed set remains
authoritative. A rerun validates it and removes any remaining matching legacy
aliases. A cleanup failure is reported as `cleanup-incomplete` with a nonzero
CLI exit status rather than falsely reported as a failed or rolled-back
publication.

The operation does not need a multi-gigabyte backup: before legacy unlink, each
source and target are the same inode; after legacy unlink, the data still has
the validated target link.

## 8. Idempotent state machine

The command accepts only these states:

| Legacy set | Target set | Result |
|---|---|---|
| Complete | Absent | Validate; plan or publish |
| Complete/partial | Complete and inode-matching | Validate; remove remaining legacy aliases in execute mode |
| Absent | Complete | Validate; report `already-migrated` |
| Complete | Partial and inode-matching | Resume missing target publication |
| Missing required source and target | Any | Fail closed |
| Any conflicting target inode | Any | Fail closed |
| Any symlink/nonregular path | Any | Fail closed |

Targets that merely have the same size or bytes but a different inode are not
silently accepted. That state requires manual investigation because it could
represent an unrelated prior generation.

## 9. Testing

Use tiny temporary NumPy, FAISS, sidecar, and DuckDB fixtures. Tests must cover:

- plan mode validates and performs zero writes;
- successful no-copy migration;
- the sidecar is published last;
- a second execution is a no-op;
- resumption from each safe partial state;
- failure cleanup removes only links created by the invocation;
- legacy cleanup interruption remains recoverable;
- conflicting targets fail without mutation;
- symlink and nonregular source/target rejection;
- malformed/partial/wrong-model sidecars;
- wrong dtype, shape, finite values, and norms;
- wrong FAISS type, dimensions, count, or reconstructed vectors;
- missing, inconsistent, noncontiguous, out-of-range, or semantically wrong
  `embedding_map` rows;
- required-model CLI parsing and read-only database opening;
- no imports from the LLM layer into statistical detection modules.

Tests must establish RED behavior before production changes, then run the
focused embed/pipeline/retrieval suites and the complete repository suite.

## 10. Real migration acceptance

After code review and green tests:

1. From the isolated Phase 8 worktree, point explicitly at the shared data
   directory and run the command without `--execute`:

   ```bash
   HARMSCOPE_DATA_DIR=/Users/rushilrawat/HarmScope/data \
     .venv/bin/python -m src.pipeline migrate-embeddings \
     --model sentence-transformers/all-MiniLM-L6-v2
   ```

2. Review its counts, dimensions, source state, and planned SHA basenames.
3. Repeat the same command once with `--execute`.
4. Confirm all SHA-addressed artifacts exist and all legacy names are absent.
5. Confirm byte sizes and inodes match the pre-migration observations.
6. Load the vectors through Phase 8's strict current loader.
7. Run a network-free retrieval smoke test against an existing MiniLM cluster.
8. Confirm the DuckDB file and relevant row counts were unchanged.

The existing artifacts observed before implementation are regular files on one
filesystem, so hard-link publication is feasible:

- memmap: 3,806,111,360 bytes;
- FAISS index: 3,806,111,277 bytes;
- sidecar: 102 bytes;
- rows: 2,477,937;
- dimension: 384.

## 11. Failure and fallback

Any semantic validation failure stops before mutation and identifies only the
artifact/model contract, never complaint prose. The fallback is to investigate
or regenerate the named model explicitly; it is not to relax the loader.

If the filesystem does not support same-directory hard links, the command
fails before changing names and directs the maintainer to regeneration. It does
not silently perform a 7.6 GB copy.

## 12. Repository state

This work builds on the existing unstaged Phase 8 fix tree and must not discard,
stage, or rewrite it. The specification and implementation will remain
unstaged while the existing Git approval-layer block is active. No commit claim
will be made until Git mutation is actually permitted and verified.

## 13. Fix Round 4 atomic-rename candidate (not approved)

The hard-link publication and later legacy-cleanup protocol above is superseded
for Task 2. Adversarial review showed that macOS exposes no descriptor-bound
hard-link or conditional-unlink primitive: checking a mutable source pathname
before `link(2)`, or checking a quarantine pathname before `unlink(2)`, cannot
prove which inode the following syscall will affect. Adding permanent hard-link
anchors merely moves that unsafe bootstrap to another mutable pathname.

Execute mode therefore uses one atomic, no-overwrite rename for each logical
artifact, in memmap/index/sidecar order:

- macOS: `renameatx_np(..., RENAME_EXCL)` relative to the pinned artifact-root
  descriptor;
- Linux: `renameat2(..., RENAME_NOREPLACE)` relative to the same descriptor;
- any platform without one of those primitives: fail before namespace
  mutation.

The recovery invariant is a deterministic partition. For each artifact role,
exactly one of its legacy basename or SHA-addressed target basename exists.
Plan mode descriptor-binds and semantically validates any complete mixed
partition without mutation. Execute mode atomically moves only the legacy
members that remain, fsyncs the directory after every move, and validates the
complete target set before returning. A crash may leave any per-file partition;
the next plan/execute run validates and resumes it directly. There are no
temporary names, hidden anchors, hard links, unlink calls, rollback deletes, or
legacy-cleanup phase.

Both names present for one role fail closed even when they are hard links to the
same inode, because safely deleting either mutable name would recreate the
conditional-unlink race. Neither name present also fails closed. A target that
appears before the no-overwrite rename is never replaced. If a source is
externally replaced before the atomic move, the moved entry remains named at
the target and the post-move identity check fails; migration does not delete or
overwrite it.

The approved pre-implementation inventory recorded the real MiniLM state as a
complete three-file legacy set with all three SHA targets absent, so it was not
a rejected dual-name state. Fix Round 4 did not re-read or mutate those real
artifacts; Task 4 must confirm that inventory again before any dry-run.

This atomic move is the equivalent of a name-stable recovery anchor for the
migration's own operations: the validated entry transitions from legacy name
to target name at one filesystem linearization point, and migration never
removes its namespace name afterward. An unrelated actor that independently
unlinks or overwrites that sole target is outside this guarantee, but the
operation detects target/legacy/root changes before reporting success.

The amended final state again has exactly three SHA-addressed files, no legacy
or hidden migration names, unchanged device/inode/size/content, and link count
one. Error conversions suppress chained exception text so normal tracebacks do
not disclose stored narrative or filesystem details. This amendment requires
fresh independent approval before any real dry-run or execute; temporary-fixture
evidence alone does not clear the real-data gate.

## 14. Fix Round 4 blocker adjudication

Section 13 records the direct-rename candidate implemented for investigation;
it is **not approved as satisfying the original adversarial replacement
invariant**. Atomic rename provides a safe linearization point for the name it
moves, but it cannot create the independently pinned second name required by
the remaining review condition:

- replacement after the pre-move identity check but before the rename can move
  a semantically valid different inode into the SHA namespace; the call detects
  the mismatch, but a later replay has no durable record of the original inode;
- replacement of the SHA target immediately after the rename can reduce the
  validated inode to link count zero, so closing its retained fd destroys it.

The local macOS SDK exposes pathname-based `linkat`, `unlinkat`, and
`renameatx_np`, plus descriptor-based `fclonefileat`. It exposes no Linux-style
`AT_EMPTY_PATH` descriptor-to-hard-link operation. Consequently, the following
requirements cannot all be met with the available primitives: adversarial
uncooperative pathname replacement, exact original inode preservation,
zero-copy hard-link semantics, an independently named original through final
validation, never deleting/overwriting a replacement, deterministic crash
replay, and a final link-count-one SHA-only namespace.

Round 5 must explicitly relax at least one requirement. Viable choices are:

1. require a quiescent artifact directory/cooperative writer lock for the
   maintenance window;
2. use APFS `fclonefileat` from retained fds, accepting a different target inode
   and likely a retained deterministic retirement alias for the original;
3. retain persistent hard-link anchors, accepting that their pathname-based
   bootstrap still requires a quiescence assumption and final link count above
   one; or
4. perform a full descriptor-based copy, accepting new inodes and approximately
   7.6 GB of additional I/O/storage.

Until that adjudication, the direct-rename code is preserved only as WIP and no
real dry-run or execute is permitted. Round 5 must also wrap raw directory-fsync
errors without chaining underlying messages and adjudicate the candidate's
fail-closed rejection of same-inode dual aliases against the approved Section 8
recovery table.

## 15. Fix Round 5 approved content-equivalence protocol

This section supersedes Sections 7, 8, 10, 11, and 13 for Task 2. Section 14
remains the explanation for why the original inode-preservation contract was
impossible on Darwin.

The explicit relaxation is that exact validated **content**, not original-inode
or publication provenance, is authoritative. A different-inode regular file is
equivalent when its full SHA-256 digest is byte-identical and the combined
sidecar, NumPy, FAISS, model, and read-only DuckDB contracts all validate. This
means a byte-identical replacement may be accepted on replay. The operation no
longer claims to distinguish that replacement from the clone created by its own
earlier invocation. A digest mismatch or semantic mismatch still fails closed.

Content equivalence does not admit same-inode dual aliases. An `L+T` or `T+R`
pair that names one inode is a remnant of a superseded hard-link protocol, not a
state this descriptor-clone protocol creates. It also fails to provide two
independently mutable retained content references: an in-place write through
either name changes both. Such states fail closed without mutation. This rule
does not restore original-inode provenance; different-inode byte-identical
regular files remain equivalent after full semantic validation.

### 15.1 Darwin publication primitive

Execute mode creates each missing SHA target with macOS 10.12+
`fclonefileat(source_fd, root_fd, target_basename, 0)`. `source_fd` is the
already-open, descriptor-bound file used for digest and semantic validation;
`root_fd` is the pinned artifact-directory descriptor. The API atomically
creates a copy-on-write clone, requires an absent destination, preserves exact
bytes, and produces a different inode without a full physical copy on APFS.
The code does not infer physical allocation from `st_blocks`.

Mutation is deliberately Darwin-only. Linux and other platforms fail before
namespace mutation rather than using a pathname copy, a full copy, or a
different unreviewed clone primitive. `ENOTSUP`, `EXDEV`, `EEXIST`, and all
other clone failures are converted to path-free migration errors; no target is
removed after failure.

Targets are cloned in memmap, index, progress-sidecar order. Every directory
mutation is fsynced through a privacy-safe wrapper. Once all targets exist, the
complete SHA set is reopened with `O_NOFOLLOW`, descriptor-bound, fully
validated, digest-compared with the authoritative source set, and fsynced.

### 15.2 Retained retirement entries

After target validation, each remaining legacy basename is moved with one
descriptor-relative atomic no-overwrite rename to:

```text
.<legacy-basename>.harmscope-migration-retired
```

The automated migration never unlinks a legacy, target, retirement, rollback,
or quarantine name. If the legacy pathname is replaced before the rename, the
atomic call moves that current occupant to the retirement name; the post-move
identity/content checks fail and both the validated SHA content and replacement
remain named. If the retirement destination appears first, no-overwrite rename
fails and every entry remains named.

The three retirement entries are intentional durable safety state, not leaked
temporary files. The successful automated final namespace contains three SHA
targets plus three dot-prefixed retirement entries, with no legacy basenames.
Removing retirement entries is a separate manual operation that requires a
quiescent artifact directory and operator verification. The migration command
does not automate or authorize that cleanup.

### 15.3 Crash partitions and replay

For each artifact role, exactly these monotonic phases are accepted:

| Legacy | SHA target | Retirement | Phase |
|---|---|---|---|
| present | absent | absent | validated source; target still needs cloning |
| present | present | absent | clone published; both must be different-inode and content-equivalent |
| absent | present | present | retired; both must be different-inode and content-equivalent |

Any other per-role combination fails without mutation. Different roles may be
in different accepted phases after interruption. Plan mode validates the
descriptor-bound authoritative set plus every existing target and performs no
writes. Execute mode clones only missing targets, validates the complete target
set, then retires only remaining legacy names. A crash after any clone or
retirement therefore leaves a deterministic state that the next invocation can
validate and resume.

A target-only state is not automatically accepted because it has no retained
independent content reference. Same-inode dual aliases are likewise rejected
because they are not independent content references and can only come from the
superseded hard-link protocol or external interference. Conflicting
target/legacy/retirement bytes, symlinks, special files, recreated legacy
names, configured-root replacement, and in-place content changes all fail
closed and leave the observed names in place.

### 15.4 Final checks and execution gate

Before returning `migrated`, the implementation independently reopens and
validates the target and retirement sets, compares their full digests with the
initial descriptor-bound source digests, fsyncs regular files and the pinned
directory, checks that legacy basenames are absent, and performs configured-root
identity validation as the final filesystem check. Raw syscall, fsync, DuckDB,
and close messages are not chained into formatted migration tracebacks.

Temporary APFS fixtures prove descriptor binding after source-path replacement,
exact bytes, a different target inode, atomic no-overwrite behavior, and
copy-on-write API support. They do not prove the real artifact volume's support,
large-file duration, or operational quiescence. An independent Task 2 review
found no remaining Critical or Important protocol issue. No real plan or execute
is authorized until a separate pre-execution review and Task 4 recheck the real
pre-migration inventory.
