# Embedding Artifact Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the completed MiniLM memmap, progress sidecar, and FAISS index from model-tail filenames to full-model SHA-256 filenames without re-encoding, a full physical vector copy, or changing DuckDB.

**Architecture:** A focused `src/embed/migrate.py` module owns descriptor-bound semantic validation, content-digest identity, Darwin `fclonefileat` publication, deterministic retained retirement names, monotonic crash replay, and stable rendering. `src/pipeline.py` adds a safe-by-default maintenance CLI. Runtime loaders stay strict; only after tests and independent review does the command touch real artifacts.

**Tech Stack:** Python 3.12, NumPy memmaps, FAISS `IndexFlatIP`, DuckDB, POSIX `dir_fd` filesystem calls, pytest, Ruff.

## Global Constraints

- Real model: `sentence-transformers/all-MiniLM-L6-v2`; `--model` is mandatory.
- Target names come only from `src.embed.encode.embedding_artifact_paths`.
- Default mode is read-only; mutation requires `--execute`.
- Never re-encode, perform a full user-space/physical multi-gigabyte copy, mutate DuckDB, register a run, or change model/cluster configuration.
- Never add a runtime legacy-name fallback or print complaint narratives.
- Existing files must be direct regular children of the configured artifact directory; symlinks and special files fail closed.
- Execute mode is Darwin-only: descriptor-bound CoW clones are published without overwrite, memmap then index then sidecar, directory-fsynced, target-validated, and only then are legacy basenames atomically moved to deterministic retained names. Automated migration never unlinks them.
- Every production behavior requires witnessed RED then GREEN evidence.
- Preserve the unstaged Phase 8 tree. While the known Git approval block remains active, do not stage/commit; record suggested commit boundaries.

## File Map

- Create `src/embed/migrate.py`: validation and migration domain.
- Create `tests/test_embed_migration.py`: real tiny NumPy/FAISS/DuckDB and filesystem behavior tests.
- Modify `src/pipeline.py`: lazy CLI wiring only.
- Modify `README.md`, `docs/LLM_LAYER.md`, `docs/ENGINEERING_NOTES.md`: maintenance contract and verified status.
- Modify `.superpowers/sdd/2026-08-09-phase-08d-rag-evaluation-documentation/progress.md`: append evidence.

---

### Task 1: Artifact Identity and Semantic Validation

**Files:**
- Create: `src/embed/migrate.py`
- Create: `tests/test_embed_migration.py`
- Read: `src/embed/encode.py`, `src/embed/index.py`, `src/llm/retrieve.py:554-587`

**Interfaces:**
- Consumes: `embedding_artifact_paths(Path, str)` and a read-only `duckdb.DuckDBPyConnection`.
- Produces `ArtifactMigrationError`, `ArtifactSet`, `FileIdentity`, `ValidatedArtifacts`, `legacy_artifact_paths`, `target_artifact_paths`, and `validate_artifacts`.

- [ ] **Step 1: Write the shared real fixture and two failing happy-path tests**

```python
MODEL = "provider/example-model"


@pytest.fixture
def migration_fixture(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    con = duckdb.connect()
    con.execute("CREATE TABLE narratives (complaint_id BIGINT, text_hash VARCHAR)")
    con.execute(
        "CREATE TABLE embedding_map "
        "(complaint_id BIGINT, row_idx BIGINT, model VARCHAR, dim INTEGER)"
    )
    con.executemany(
        "INSERT INTO narratives VALUES (?, ?)",
        [(10, "hash-a"), (11, "hash-a"), (12, "hash-b")],
    )
    con.executemany(
        "INSERT INTO embedding_map VALUES (?, ?, ?, 2)",
        [(10, 0, MODEL), (11, 0, MODEL), (12, 1, MODEL)],
    )
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    legacy = migrate.legacy_artifact_paths(artifact_dir, MODEL)
    np.save(legacy.memmap, vectors)
    encode.Progress(legacy.progress, 2, 2, 2, MODEL).write()
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    faiss.write_index(index, str(legacy.index))
    yield con, artifact_dir, legacy, vectors
    con.close()


def test_names_bind_legacy_tail_and_full_model_identity(tmp_path):
    legacy = migrate.legacy_artifact_paths(tmp_path, MODEL)
    target = migrate.target_artifact_paths(tmp_path, MODEL)
    assert tuple(path.name for path in legacy.ordered()) == (
        "embeddings.example-model.npy",
        "faiss.example-model.index",
        "embeddings.example-model.progress.json",
    )
    assert target.memmap.name == (
        "embeddings.718ca0f39da16820187c663a1778f728f795ae25665fa52a44925c0dfa480378.npy"
    )


def test_validation_binds_sidecar_memmap_index_and_database(migration_fixture):
    con, _, legacy, _ = migration_fixture
    result = migrate.validate_artifacts(con, legacy, MODEL)
    assert (result.model, result.n_rows, result.dim) == (MODEL, 2, 2)
    assert result.artifacts == legacy
```

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest -q \
  tests/test_embed_migration.py::test_names_bind_legacy_tail_and_full_model_identity \
  tests/test_embed_migration.py::test_validation_binds_sidecar_memmap_index_and_database
```

Expected: collection fails only because `src.embed.migrate` is absent.

- [ ] **Step 3: Implement exact public types and names**

```python
class ArtifactMigrationError(ValueError):
    """An embedding artifact cannot be safely identified or migrated."""


@dataclass(frozen=True)
class ArtifactSet:
    memmap: Path
    index: Path
    progress: Path

    def ordered(self) -> tuple[Path, Path, Path]:
        return (self.memmap, self.index, self.progress)


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class ValidatedArtifacts:
    model: str
    n_rows: int
    dim: int
    artifacts: ArtifactSet
    identities: tuple[FileIdentity, FileIdentity, FileIdentity]
```

Validate the tail with `re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tail)`. Derive target progress from `embedding_artifact_paths(artifact_dir, model_name).memmap.with_suffix(".progress.json")`. Reject nonregular `lstat` results and bind device/inode/size/mtime before and after semantic reads.

- [ ] **Step 4: Implement minimal valid-artifact checks and verify GREEN**

Parse exact sidecar keys/types; require full model equality, positive rows/dimensions, and `n_done == n_total`. Require a 2-D float32 NumPy shape. Require FAISS `IndexFlatIP`, inner-product metric, matching dimensions/count. Implement the mapping aggregate check. Run Step 2; expect `2 passed`.

- [ ] **Step 5: Add failing corruption tests**

Use parameterized literal mutations:

```python
@pytest.mark.parametrize(
    "field,value",
    [("model", "other/model"), ("n_done", 1), ("n_total", 3), ("dim", 3)],
)
def test_validation_rejects_wrong_progress(migration_fixture, field, value):
    con, _, legacy, _ = migration_fixture
    payload = json.loads(legacy.progress.read_text())
    payload[field] = value
    legacy.progress.write_text(json.dumps(payload))
    with pytest.raises(migrate.ArtifactMigrationError):
        migrate.validate_artifacts(con, legacy, MODEL)


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    ],
)
def test_validation_rejects_bad_vector_contract(migration_fixture, vectors):
    con, _, legacy, _ = migration_fixture
    np.save(legacy.memmap, vectors)
    with pytest.raises(migrate.ArtifactMigrationError):
        migrate.validate_artifacts(con, legacy, MODEL)
```

Also add separately named tests for wrong FAISS metric/count/content, source symlink, source inode replacement during validation, deleted mapping, changed dimension, row gap, out-of-range row, extra mapping, and wrong text-hash-to-row semantics. Each test asserts no fixture narrative text appears in its exception.

- [ ] **Step 6: Verify corruption RED**

```bash
.venv/bin/python -m pytest -q tests/test_embed_migration.py -k rejects
```

Expected: every missing branch fails for the intended contract, not setup.

- [ ] **Step 7: Complete chunked vector, FAISS, and SQL validation**

Scan finite/unit norms in 16,384-row chunks using `np.isclose(norms, 1.0, atol=1e-3, rtol=0.0)`. Compare 17 evenly spaced `index.reconstruct(row_idx)` rows to the memmap using `np.allclose(reconstructed, vectors[row_idx], atol=1e-6, rtol=0.0)`.

Use only `SELECT` queries. Require one dimension, indices `0..n_total-1`, exact distinct-row count, no extra model mappings, and no unmapped narratives. Bind semantic rows with:

```sql
WITH ranked_hashes AS (
    SELECT text_hash,
           row_number() OVER (ORDER BY text_hash) - 1 AS expected_row
    FROM (SELECT DISTINCT text_hash FROM narratives)
), expected AS (
    SELECT n.complaint_id, r.expected_row
    FROM narratives n JOIN ranked_hashes r USING (text_hash)
)
SELECT
    count(*) FILTER (WHERE e.complaint_id IS NULL) AS missing,
    count(*) FILTER (
        WHERE e.complaint_id IS NOT NULL
          AND (e.row_idx != x.expected_row OR e.dim != ?)
    ) AS mismatched,
    count(DISTINCT x.expected_row) AS expected_rows
FROM expected x
LEFT JOIN embedding_map e
  ON e.complaint_id = x.complaint_id AND e.model = ?
```

- [ ] **Step 8: Verify Task 1 GREEN**

```bash
.venv/bin/python -m pytest -q tests/test_embed_migration.py tests/test_embed.py
.venv/bin/python -m ruff check src/embed/migrate.py tests/test_embed_migration.py
.venv/bin/python -m ruff format --check src/embed/migrate.py tests/test_embed_migration.py
git diff --check
```

Record suggested commit `Validate legacy embedding artifacts`; do not stage/commit while blocked.

---

### Task 2: Idempotent Hard-Link Publication and Recovery

**Files:**
- Modify: `src/embed/migrate.py`
- Modify: `tests/test_embed_migration.py`

**Interfaces:**
- Consumes all Task 1 interfaces.
- Produces `MigrationReport` and `migrate_embedding_artifacts(con, artifact_dir, model_name, *, execute=False)`.

- [ ] **Step 1: Write failing plan and execute tests**

```python
def test_plan_mode_validates_without_directory_changes(migration_fixture):
    con, artifact_dir, _, _ = migration_fixture
    before = sorted(path.name for path in artifact_dir.iterdir())
    report = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL)
    assert report.state == "planned"
    assert sorted(path.name for path in artifact_dir.iterdir()) == before
    assert all(not path.exists() for path in report.target.ordered())


def test_execute_publishes_same_inodes_then_removes_legacy(migration_fixture):
    con, artifact_dir, legacy, _ = migration_fixture
    old_inodes = tuple(path.stat().st_ino for path in legacy.ordered())
    report = migrate.migrate_embedding_artifacts(
        con, artifact_dir, MODEL, execute=True
    )
    assert report.state == "migrated"
    assert tuple(path.stat().st_ino for path in report.target.ordered()) == old_inodes
    assert all(not path.exists() for path in legacy.ordered())
```

- [ ] **Step 2: Verify RED**

Run the two tests above. Expected: fail because the migration function is absent.

- [ ] **Step 3: Implement report, root pinning, and first publication path**

`MigrationReport` contains `model`, `n_rows`, `dim`, `state`, `legacy`, and `target`; `render()` prints only those safe fields and basenames.

Open the root with `O_RDONLY | O_DIRECTORY | O_NOFOLLOW`; retain `fstat`. Publish `(memmap, index, progress)` in that order with `os.link(source_name, target_name, src_dir_fd=root_fd, dst_dir_fd=root_fd, follow_symlinks=False)`. Before and after each operation compare source and target device/inode and revalidate configured-root identity. Track names created by the invocation. On failure unlink those names in reverse order and fsync the directory before re-raising the primary exception.

After link publication, independently call `validate_artifacts` on the target. Only then unlink legacy names, fsync, revalidate the root, and validate target again.

- [ ] **Step 4: Verify initial GREEN**

Run Step 1 tests. Expected: `2 passed`.

- [ ] **Step 5: Write exact idempotency/state tests**

Add these observable arrangements and assertions:

| Test | Arrangement | Required assertion |
|---|---|---|
| `test_second_execute_is_already_migrated` | Execute once, capture target stats, execute again | state `already-migrated`; target stats unchanged |
| `test_plan_with_complete_matching_targets_keeps_legacy` | Hard-link all targets manually, keep legacy | state `planned`; all six names remain |
| `test_execute_resumes_matching_partial_targets` | Hard-link only memmap target | execution creates index/sidecar targets, removes legacy, preserves all three source inodes |
| `test_execute_finishes_interrupted_legacy_cleanup` | Hard-link all targets and remove one legacy name | execution removes remaining legacy names; target inodes unchanged |
| `test_partial_target_without_complete_legacy_fails` | Remove legacy index and create only target memmap | `ArtifactMigrationError`; directory entry set unchanged |
| `test_byte_equal_different_inode_target_fails` | Copy tiny sidecar bytes to target sidecar as a separate inode | `ArtifactMigrationError`; neither file changed |

- [ ] **Step 6: Verify state RED, implement the Design §8 state table, verify GREEN**

```bash
.venv/bin/python -m pytest -q tests/test_embed_migration.py -k \
  'already_migrated or matching_targets or resumes or interrupted or partial_target or different_inode'
```

Accept only complete legacy, matching partial publication with complete legacy, complete matching targets, or fully migrated state.

- [ ] **Step 7: Write exact race/recovery tests**

| Test | Failure injection | Required filesystem proof |
|---|---|---|
| `test_link_failure_removes_only_new_targets` | Raise `OSError` on second `os.link` | no target created by call remains; every legacy inode remains |
| `test_target_validation_failure_rolls_back_publication` | Corrupt target-validation result after links | target names removed; complete legacy set remains |
| `test_sidecar_is_published_last` | Record successful link destination names | exact order memmap, index, progress |
| `test_root_rename_during_link_rolls_back_pinned_root` | Rename root and recreate configured root inside second link | root-change error; no targets in moved or recreated root; legacy remains in moved root |
| `test_symlink_target_is_not_followed` | Target name is symlink to outside sentinel | error; sentinel bytes unchanged; legacy unchanged |
| `test_cleanup_failure_leaves_complete_targets` | Raise on second legacy unlink | report state is `cleanup-incomplete`; complete validated targets remain; no target rollback |

- [ ] **Step 8: Verify recovery RED, implement guarded recovery, verify GREEN**

Preserve primary exceptions, append only secondary exception types, continue cleanup after secondary failures, and fsync every directory-entry mutation. Run all six named tests and then all of `tests/test_embed_migration.py`.

- [ ] **Step 9: Verify Task 2 GREEN**

```bash
.venv/bin/python -m pytest -q \
  tests/test_embed_migration.py tests/test_embed.py tests/test_llm_retrieve.py
.venv/bin/python -m ruff check src/embed/migrate.py tests/test_embed_migration.py
.venv/bin/python -m ruff format --check src/embed/migrate.py tests/test_embed_migration.py
git diff --check
```

Record suggested commit `Migrate embedding artifacts without copying`; do not stage/commit while blocked.

---

### Task 3: Safe Maintenance CLI and Pre-Execution Documentation

**Files:**
- Modify: `src/pipeline.py`
- Modify: `tests/test_embed_migration.py`
- Modify: `README.md`, `docs/LLM_LAYER.md`, `docs/ENGINEERING_NOTES.md`

**Interfaces:**
- Consumes `migrate_embedding_artifacts`.
- Produces `cmd_migrate_embeddings(args)` and `migrate-embeddings --model MODEL [--execute]`.

- [ ] **Step 1: Write failing parser and handler tests**

```python
def test_cli_requires_model_and_defaults_to_plan_mode():
    parser = pipeline.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["migrate-embeddings"])
    args = parser.parse_args(
        ["migrate-embeddings", "--model", "provider/example-model"]
    )
    assert args.model == "provider/example-model"
    assert args.execute is False


def test_handler_closes_read_only_connection(monkeypatch, capsys):
    closed = []
    connection = SimpleNamespace(close=lambda: closed.append(True))
    report = SimpleNamespace(state="planned", render=lambda: "state : planned")
    monkeypatch.setattr(
        pipeline,
        "PATHS",
        SimpleNamespace(db=Path("existing.duckdb"), artifacts=Path("artifacts")),
    )
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(pipeline.db, "connect", lambda path, read_only: connection)
    monkeypatch.setattr(
        migrate,
        "migrate_embedding_artifacts",
        lambda con, root, model, execute: report,
    )
    args = SimpleNamespace(model=MODEL, execute=False)
    assert pipeline.cmd_migrate_embeddings(args) == 0
    assert closed == [True]
    assert capsys.readouterr().out == "state : planned\n"
```

Add a failure-path variant whose migration callable raises `ArtifactMigrationError`; assert connection closed. Add a bootstrap spy that raises if called. Add report rendering assertions that include model/count/dim/basenames/state and exclude a literal fixture narrative.

- [ ] **Step 2: Verify CLI RED**

```bash
.venv/bin/python -m pytest -q tests/test_embed_migration.py -k \
  'cli_requires or handler or rendering'
```

- [ ] **Step 3: Implement lazy CLI wiring**

```python
def cmd_migrate_embeddings(args: argparse.Namespace) -> int:
    from src.embed import migrate

    if not PATHS.db.is_file():
        raise SystemExit(f"database does not exist at {PATHS.db}")
    con = db.connect(PATHS.db, read_only=True)
    try:
        report = migrate.migrate_embedding_artifacts(
            con, PATHS.artifacts, args.model, execute=args.execute
        )
        print(report.render())
        return 2 if report.state == "cleanup-incomplete" else 0
    finally:
        con.close()
```

Register a top-level parser with required `--model`, boolean `--execute`, and `func=cmd_migrate_embeddings`. Keep migration import lazy. Do not use `db.bootstrap` or `db.run`.

- [ ] **Step 4: Verify CLI GREEN**

Run Step 2. Expected: every selected test passes.

- [ ] **Step 5: Add strict current-loader integration coverage**

After migrating the tiny fixture, monkeypatch `src.llm.retrieve.PATHS.artifacts`, create this corpus, and assert exact vector values:

```python
corpus = ScopedCorpus(
    cluster_id="cluster-1",
    company_id="company-1",
    embed_model=MODEL,
    rows=(),
    embed_dim=2,
    embedding_rows=2,
)
vectors = retrieve._load_default_vectors(corpus)
assert np.array_equal(vectors, expected_vectors)
```

Delete the SHA sidecar and assert the loader raises `ValueError`; legacy fallback must remain absent.

- [ ] **Step 6: Update documentation before real execution**

Document the plan/execute commands, strict fallback prohibition, validation/recovery guarantees, and status `implementation complete; real execution pending`. Keep the embedding gate open until Task 4 records real evidence.

- [ ] **Step 7: Verify Task 3 GREEN**

```bash
.venv/bin/python -m pytest -q \
  tests/test_embed_migration.py tests/test_embed.py tests/test_llm_retrieve.py tests/test_llm_cli.py
.venv/bin/python -m ruff check src/embed/migrate.py src/pipeline.py tests/test_embed_migration.py
.venv/bin/python -m ruff format --check src/embed/migrate.py src/pipeline.py tests/test_embed_migration.py
git diff --check
```

Record suggested commit `Expose validated embedding migration`; do not stage/commit while blocked.

---

### Task 4: Independent Review, Real Migration, and Acceptance

**Files:**
- Review all Task 1-3 changes against `docs/superpowers/specs/2026-08-14-embedding-artifact-migration-design.md`.
- Modify after evidence: `README.md`, `docs/LLM_LAYER.md`, `docs/ENGINEERING_NOTES.md`.
- Modify after evidence: `.superpowers/sdd/2026-08-09-phase-08d-rag-evaluation-documentation/progress.md`.

**Interfaces:**
- Consumes green Task 1-3 code and `/Users/rushilrawat/HarmScope/data`.
- Produces reviewed SHA artifacts, no legacy aliases, strict-loader evidence, unchanged-DB evidence, and accurate docs.

- [ ] **Step 1: Run the complete pre-review gate**

```bash
.venv/bin/python -m pytest -q \
  tests/test_embed_migration.py tests/test_embed.py tests/test_llm_retrieve.py tests/test_llm_cli.py
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check \
  src/embed/migrate.py src/pipeline.py tests/test_embed_migration.py
git diff --check
```

Capture counts and exit codes. Do not touch real artifacts on any failure.

- [ ] **Step 2: Obtain independent code/security approval**

The review brief includes the approved spec, this plan, exact dirty diff, and verification evidence. It must explicitly adjudicate inode/path binding, in-place races, root renames, symlink rejection, sidecar-last semantics, partial-state recovery, read-only SQL, strict-loader preservation, and privacy-safe output. Every Critical/Important finding gets a new witnessed RED/GREEN regression and another review.

- [ ] **Step 3: Capture real pre-migration evidence**

Read only: legacy sizes/device/inode/link counts, SHA-target absence, exact sidecar bytes, MiniLM mapping count/distinct/min/max/dim aggregates, and DuckDB size/mtime. Do not query narrative text.

- [ ] **Step 4: Run dry-run against real data**

```bash
HARMSCOPE_DATA_DIR=/Users/rushilrawat/HarmScope/data \
  .venv/bin/python -m src.pipeline migrate-embeddings \
  --model sentence-transformers/all-MiniLM-L6-v2
```

Expected: `planned`, 2,477,937 rows, 384 dimensions, correct basenames, no filesystem/DB change. Repeat Step 3 comparisons before execute.

- [ ] **Step 5: Execute exactly once**

```bash
HARMSCOPE_DATA_DIR=/Users/rushilrawat/HarmScope/data \
  .venv/bin/python -m src.pipeline migrate-embeddings \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --execute
```

Expected: `migrated`, with three SHA targets plus three deterministic retirement entries. On any error, inspect and rerun only the maintenance command after review; never improvise filesystem deletion or replacement.

- [ ] **Step 6: Prove idempotency and strict load**

Run execute mode again; require `already-migrated` with unchanged target and retirement stats. Construct a MiniLM `ScopedCorpus` with `embed_dim=384` and `embedding_rows=2477937`; call `_load_default_vectors` and print only shape/dtype. Require `(2477937, 384) float32`.

- [ ] **Step 7: Verify exact postconditions**

Require all SHA and deterministic retirement files, no legacy basenames, retirement device/inode/size equal to pre-migration sources, target device equal but inode different, exact source/target/retirement bytes, link count one for each entry, unchanged mapping aggregates, and unchanged DuckDB size/mtime. Any mismatch fails acceptance even if CLI exit was zero. Retirement entries remain until a separate quiescent manual cleanup; this automated task must not remove them.

- [ ] **Step 8: Run network-free retrieval smoke**

Select one successful MiniLM cluster/company using ID-only SQL, load corpus/vectors, and perform dense ranking with locally available weights. Print only scope IDs, evidence count, and complaint IDs. Never call Anthropic or render narratives. If weights are absent, report this smoke externally blocked; do not download without approval.

- [ ] **Step 9: Update operational truth**

Replace only the embedding-gate text with execution date, exact model/SHA basename, row/dim counts, dry-run/migration/idempotency/strict-loader/DB evidence, and any smoke limitation. Preserve every human, company-scope, company-response, billing, and review gate.

- [ ] **Step 10: Run final verification**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check \
  src/embed/migrate.py src/pipeline.py tests/test_embed_migration.py
git diff --check
git status --short --branch
```

Read full output and report exact counts. Confirm index empty and prior Phase 8 changes preserved. Record suggested commit `Migrate MiniLM artifacts to model identity paths`; do not stage/commit while blocked.

---

### Fix Round 4 candidate: atomic rename partition (blocked)

The Task 2 hard-link/publication/rollback/cleanup steps are superseded by the
design amendment in Section 13 of the specification. The implementation must:

1. validate a complete per-role partition in which exactly one legacy or SHA
   basename exists;
2. use descriptor-relative Darwin `renameatx_np(RENAME_EXCL)` or Linux
   `renameat2(RENAME_NOREPLACE)` for each missing SHA target, with sidecar last;
3. fsync validated regular files before mutation and the directory after every
   atomic move;
4. leave any interrupted legacy/SHA partition intact for deterministic replay,
   without rollback, cleanup, random names, hard links, or unlink calls;
5. revalidate bound content, configured-root identity, all SHA identities, and
   legacy-name absence after the final semantic read and before success;
6. suppress underlying exception messages in formatted tracebacks while
   retaining type-only close-error notes and primary-error precedence.

Focused RED/GREEN must include replacement immediately before and after the
atomic move, destination appearance, interruption after a successful move,
mixed-partition plan/resume, configured-root replacement after the final digest,
legacy-name recreation, and formatted-traceback privacy. Previous hard-link and
cleanup regressions are replaced by a proof that execute mode never calls
`link(2)` or `unlink(2)` and leaves no internal migration names.

Dual legacy/SHA names for any role are rejected even when they reference the
same inode. The approved pre-implementation inventory recorded complete legacy
artifacts and absent SHA targets, so the real state was not dual-name; Task 4
must re-confirm that fact before a dry-run because this fix round did not access
the real artifact directory.

Task 4's real commands and link-count-one postcondition remain unchanged, but
real dry-run and execute stay blocked until the amended protocol receives a
fresh independent safety review and all temporary-fixture/dependent-suite gates
are recorded green. No real-data command is authorized by this amendment.

#### Fix Round 4 stop condition

Independent adversarial analysis rejected the direct-rename candidate as an
equivalent independent anchor: a valid different-inode source replacement can
be moved into SHA state and later accepted on replay, while a target replacement
after the move can leave the validated inode unnamed. The candidate remains WIP
for Fix Round 5 comparison, but Task 2 is blocked until the user explicitly
chooses a requirement relaxation from specification Section 14. No broad green
suite or local implementation result may be presented as clearing this blocker,
and Task 4 must not access the real artifacts, even in plan mode.

### Fix Round 5 implementation: descriptor clone plus retained retirement state

This section supersedes the original Task 2 hard-link steps and the blocked Fix
Round 4 candidate. The explicit approved relaxation is content equivalence:
full digest plus complete semantic validation is authoritative; original inode
and publication provenance are not. A byte-identical different-inode
replacement can therefore be accepted on replay. This fact must be included in
the fresh pre-execution review.

- [x] Prototype Darwin `fclonefileat` only on temporary APFS fixtures. Require
  an already-open source fd, pinned destination-directory fd, exact bytes,
  different inode, atomic no-overwrite/EEXIST, and no inference from
  `st_blocks`. Treat copy-on-write behavior as the API contract, not a measured
  real-artifact result.
- [x] Add `retired_artifact_paths`, using
  `.<legacy-basename>.harmscope-migration-retired`, and include those safe
  basenames in `MigrationReport`.
- [x] Accept only the per-role phases `L`, `L+T`, and `T+R`. Mixed phases across
  memmap/index/sidecar are resumable when every present duplicate has the exact
  authoritative digest and the combined artifacts pass sidecar, NumPy, FAISS,
  model, and read-only mapping validation. Require each dual-name pair to use
  different inodes: same-inode aliases are remnants of the superseded hard-link
  protocol and do not provide independent retained content. All other
  combinations fail without mutation.
- [x] In Darwin execute mode, CoW-clone only missing targets from retained
  validated fds, in memmap/index/sidecar order. Do not add a Linux/path-copy or
  full-copy fallback. Open, descriptor-bind, digest, semantically validate, and
  fsync the complete target set before retirement.
- [x] Atomically rename only remaining legacy basenames to absent deterministic
  retirement names with no overwrite. Never call unlink for automated
  publication, rollback, retirement, or cleanup. Replacements and conflicts
  stay named and block success.
- [x] Reopen and validate the complete target and retirement sets, compare both
  to the initial source digests, fsync files and directory, require legacy
  basenames absent, and make configured-root revalidation the final filesystem
  check before return.
- [x] Cover strict RED/GREEN schedules for source replacement before clone,
  target replacement after clone, retirement replacement, target appearance,
  interruptions after clone and retirement, mixed replay partitions, final
  root replacement, same-metadata content changes, fsync traceback privacy,
  invalid presence states, pre-protocol hard-link aliases, sidecar-last
  ordering, Linux refusal, and close-error precedence.
- [x] Obtain a fresh independent Task 2 safety review of the final diff and
  evidence. The Round 5 audit found no remaining Critical or Important
  protocol issue; this does not replace the separate real-data pre-execution
  review.
- [ ] Only after that approval, return to Task 4's real inventory, plan, and
  execute gates. Confirm the real artifact volume supports `fclonefileat`
  before interpreting any successful temporary APFS probe as operational
  evidence.

The three retirement entries are intentional long-lived safety state. Later
deletion is outside this automated task and requires a quiescent operator who
first verifies exact SHA-target content. After manual removal, the migration
command intentionally rejects target-only state because the retained
independent comparison set is gone.
