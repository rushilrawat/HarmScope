"""Semantic validation for completed embedding artifacts."""

from __future__ import annotations

import json
import os
import stat
import traceback
from pathlib import Path
from types import SimpleNamespace

import duckdb
import faiss
import numpy as np
import pytest

from src import pipeline
from src.embed import encode, migrate
from src.llm import retrieve

MODEL = "provider/example-model"
FIXTURE_NARRATIVE = "this fixture narrative must never be reported"


class _FailingMappingConnection:
    """Raise a real DuckDB read error at a selected mapping query."""

    def __init__(self, connection, failure_call):
        self.connection = connection
        self.failure_call = failure_call
        self.calls = 0

    def execute(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self.failure_call:
            raise duckdb.IOException(FIXTURE_NARRATIVE)
        return self.connection.execute(*args, **kwargs)


@pytest.fixture
def migration_fixture(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE narratives (complaint_id BIGINT, text_hash VARCHAR, text_redacted VARCHAR)"
    )
    con.execute(
        "CREATE TABLE embedding_map "
        "(complaint_id BIGINT, row_idx BIGINT, model VARCHAR, dim INTEGER)"
    )
    con.executemany(
        "INSERT INTO narratives VALUES (?, ?, ?)",
        [
            (10, "hash-a", FIXTURE_NARRATIVE),
            (11, "hash-a", FIXTURE_NARRATIVE),
            (12, "hash-b", FIXTURE_NARRATIVE),
        ],
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


def _assert_rejected(con, legacy):
    with pytest.raises(migrate.ArtifactMigrationError) as excinfo:
        migrate.validate_artifacts(con, legacy, MODEL)
    assert FIXTURE_NARRATIVE not in str(excinfo.value)


def test_plan_mode_validates_without_directory_changes(migration_fixture):
    con, artifact_dir, _, _ = migration_fixture
    before = sorted(path.name for path in artifact_dir.iterdir())

    report = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL)

    assert report.state == "planned"
    assert sorted(path.name for path in artifact_dir.iterdir()) == before
    assert all(not path.exists() for path in report.target.ordered())


def test_plan_rejects_transient_validation_root_swap(migration_fixture, tmp_path, monkeypatch):
    con, artifact_dir, legacy, _ = migration_fixture
    valid_root = tmp_path / "valid-root"
    valid_root.mkdir()
    valid_legacy = migrate.legacy_artifact_paths(valid_root, MODEL)
    for source, destination in zip(legacy.ordered(), valid_legacy.ordered(), strict=True):
        destination.write_bytes(source.read_bytes())
    payload = json.loads(legacy.progress.read_text())
    payload["model"] = "other/model"
    legacy.progress.write_text(json.dumps(payload))
    before = sorted(path.name for path in artifact_dir.iterdir())
    held_root = tmp_path / "held-pinned-root"
    real_validate = migrate._validate_bound_artifacts

    def validate_during_root_swap(*args, **kwargs):
        artifact_dir.rename(held_root)
        valid_root.rename(artifact_dir)
        try:
            return real_validate(*args, **kwargs)
        finally:
            artifact_dir.rename(valid_root)
            held_root.rename(artifact_dir)

    monkeypatch.setattr(migrate, "_validate_bound_artifacts", validate_during_root_swap)

    with pytest.raises(migrate.ArtifactMigrationError, match="progress"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL)

    assert sorted(path.name for path in artifact_dir.iterdir()) == before
    assert json.loads(legacy.progress.read_text())["model"] == "other/model"


@pytest.mark.parametrize("artifact_kind", ["sidecar", "memmap", "index"])
def test_plan_rejects_transient_artifact_swap_during_semantic_read(
    migration_fixture, tmp_path, monkeypatch, artifact_kind
):
    con, artifact_dir, legacy, vectors = migration_fixture
    before = sorted(path.name for path in artifact_dir.iterdir())

    if artifact_kind == "sidecar":
        alternate = tmp_path / "valid-progress.json"
        alternate.write_bytes(legacy.progress.read_bytes())
        payload = json.loads(legacy.progress.read_text())
        payload["model"] = "other/model"
        legacy.progress.write_text(json.dumps(payload))
        held = tmp_path / "held-invalid-progress.json"
        real_read_text = Path.read_text

        def read_during_swap(path, *args, **kwargs):
            if path != legacy.progress:
                return real_read_text(path, *args, **kwargs)
            path.rename(held)
            alternate.rename(path)
            try:
                return real_read_text(path, *args, **kwargs)
            finally:
                path.rename(alternate)
                held.rename(path)

        monkeypatch.setattr(Path, "read_text", read_during_swap)
    elif artifact_kind == "memmap":
        alternate = tmp_path / "valid-vectors.npy"
        alternate.write_bytes(legacy.memmap.read_bytes())
        np.save(
            legacy.memmap,
            np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        held = tmp_path / "held-invalid-vectors.npy"
        real_load = migrate.np.load

        def load_during_swap(path, *args, **kwargs):
            if Path(path) != legacy.memmap:
                return real_load(path, *args, **kwargs)
            legacy.memmap.rename(held)
            alternate.rename(legacy.memmap)
            try:
                return real_load(path, *args, **kwargs)
            finally:
                legacy.memmap.rename(alternate)
                held.rename(legacy.memmap)

        monkeypatch.setattr(migrate.np, "load", load_during_swap)
    else:
        alternate = tmp_path / "valid.index"
        alternate.write_bytes(legacy.index.read_bytes())
        invalid_index = faiss.IndexFlatL2(2)
        invalid_index.add(vectors)
        faiss.write_index(invalid_index, str(legacy.index))
        held = tmp_path / "held-invalid.index"
        real_read_index = faiss.read_index

        def read_index_during_swap(path, *args, **kwargs):
            if Path(path) != legacy.index:
                return real_read_index(path, *args, **kwargs)
            legacy.index.rename(held)
            alternate.rename(legacy.index)
            try:
                return real_read_index(path, *args, **kwargs)
            finally:
                legacy.index.rename(alternate)
                held.rename(legacy.index)

        monkeypatch.setattr(faiss, "read_index", read_index_during_swap)

    with pytest.raises(migrate.ArtifactMigrationError):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL)

    assert sorted(path.name for path in artifact_dir.iterdir()) == before


def test_report_render_contains_only_safe_fields_and_basenames(migration_fixture):
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    report = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL)

    rendered = report.render()

    assert json.loads(rendered) == {
        "dim": 2,
        "legacy": [path.name for path in legacy.ordered()],
        "model": MODEL,
        "n_rows": 2,
        "retired": [path.name for path in retired.ordered()],
        "state": "planned",
        "target": [path.name for path in target.ordered()],
    }
    assert str(artifact_dir) not in rendered
    assert FIXTURE_NARRATIVE not in rendered


@pytest.mark.skipif(migrate.sys.platform != "darwin", reason="Darwin fclonefileat contract")
def test_descriptor_clone_uses_open_source_fd_and_never_overwrites(tmp_path):
    """A pathname swap must not change which validated bytes are cloned."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    source = artifact_dir / "source"
    target = artifact_dir / "target"
    original = (b"validated source bytes\x00" * 257) + b"tail"
    source.write_bytes(original)
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    root_fd = os.open(artifact_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        source.unlink()
        source.write_bytes(b"pathname replacement")

        migrate._clone_from_fd_no_replace(source_fd, root_fd, target.name)

        assert target.read_bytes() == original
        assert target.stat().st_ino != os.fstat(source_fd).st_ino
        with pytest.raises(FileExistsError):
            migrate._clone_from_fd_no_replace(source_fd, root_fd, target.name)
        assert target.read_bytes() == original
        assert source.read_bytes() == b"pathname replacement"
    finally:
        os.close(root_fd)
        os.close(source_fd)


@pytest.mark.skipif(migrate.sys.platform != "darwin", reason="Darwin fclonefileat contract")
def test_descriptor_clone_rejects_non_basename_destination(tmp_path):
    """Allowing a slash would let a future caller escape the pinned artifact root."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    source = artifact_dir / "source"
    source.write_bytes(b"validated source")
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    root_fd = os.open(artifact_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(migrate.ArtifactMigrationError, match="direct child"):
            migrate._clone_from_fd_no_replace(source_fd, root_fd, "../escaped")
        assert not (tmp_path / "escaped").exists()
    finally:
        os.close(root_fd)
        os.close(source_fd)


def test_execute_clones_targets_and_retains_originals_under_retirement_names(
    migration_fixture,
):
    """Returning migrated without six exact retained/target entries loses recovery state."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    original = tuple((path.stat().st_ino, path.read_bytes()) for path in legacy.ordered())

    report = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert report.state == "migrated"
    assert report.retired == retired
    assert all(not path.exists() for path in legacy.ordered())
    assert {path.name for path in artifact_dir.iterdir()} == {
        path.name for path in (*target.ordered(), *retired.ordered())
    }
    for target_path, retired_path, (source_inode, source_bytes) in zip(
        target.ordered(), retired.ordered(), original, strict=True
    ):
        assert retired_path.stat().st_ino == source_inode
        assert target_path.stat().st_ino != source_inode
        assert target_path.read_bytes() == source_bytes
        assert retired_path.read_bytes() == source_bytes


def test_linux_execute_fails_closed_without_path_copy(migration_fixture, monkeypatch):
    """Adding a pathname-copy fallback on Linux would abandon descriptor binding."""
    con, artifact_dir, _, _ = migration_fixture
    before = sorted((path.name, path.read_bytes()) for path in artifact_dir.iterdir())
    monkeypatch.setattr(migrate.sys, "platform", "linux")

    with pytest.raises(migrate.ArtifactMigrationError, match="descriptor-bound clone"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert sorted((path.name, path.read_bytes()) for path in artifact_dir.iterdir()) == before


def test_linux_execute_refuses_content_complete_clone_partition(migration_fixture, monkeypatch):
    """A replay with every target present must not bypass the Darwin-only mutation gate."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    for source, destination in zip(legacy.ordered(), target.ordered(), strict=True):
        destination.write_bytes(source.read_bytes())
    before = sorted((path.name, path.read_bytes()) for path in artifact_dir.iterdir())
    monkeypatch.setattr(migrate.sys, "platform", "linux")

    with pytest.raises(migrate.ArtifactMigrationError, match="descriptor-bound clone"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert sorted((path.name, path.read_bytes()) for path in artifact_dir.iterdir()) == before


def test_execute_resumes_content_equivalent_legacy_target_partition(migration_fixture):
    """Rejecting a byte-exact different-inode clone would make post-clone replay impossible."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    for source, destination in zip(legacy.ordered(), target.ordered(), strict=True):
        destination.write_bytes(source.read_bytes())
        assert destination.stat().st_ino != source.stat().st_ino

    report = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert report.state == "migrated"
    assert all(not path.exists() for path in legacy.ordered())
    assert all(path.exists() for path in (*target.ordered(), *retired.ordered()))


def test_pre_protocol_legacy_target_hard_links_fail_closed(migration_fixture):
    """An old hard-link L+T state is not provenance for the amended clone protocol."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    os.link(legacy.memmap, target.memmap)
    before = sorted(
        (path.name, path.stat().st_ino, path.read_bytes()) for path in artifact_dir.iterdir()
    )

    with pytest.raises(migrate.ArtifactMigrationError, match="hard-link alias"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert (
        sorted(
            (path.name, path.stat().st_ino, path.read_bytes()) for path in artifact_dir.iterdir()
        )
        == before
    )


def test_pre_protocol_target_retirement_hard_links_fail_closed(migration_fixture):
    """A fully named old hard-link state must not be accepted as completed cloning."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    for source, destination, retirement in zip(
        legacy.ordered(), target.ordered(), retired.ordered(), strict=True
    ):
        os.link(source, destination)
        source.rename(retirement)
    before = sorted(
        (path.name, path.stat().st_ino, path.read_bytes()) for path in artifact_dir.iterdir()
    )

    with pytest.raises(migrate.ArtifactMigrationError, match="hard-link alias"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert (
        sorted(
            (path.name, path.stat().st_ino, path.read_bytes()) for path in artifact_dir.iterdir()
        )
        == before
    )


def test_execute_resumes_mixed_legacy_target_retired_partition(migration_fixture):
    """A crash after any clone/retirement boundary must leave a resumable monotonic state."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    target.memmap.write_bytes(legacy.memmap.read_bytes())
    legacy.memmap.rename(retired.memmap)
    target.index.write_bytes(legacy.index.read_bytes())

    report = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert report.state == "migrated"
    assert all(not path.exists() for path in legacy.ordered())
    assert all(path.exists() for path in (*target.ordered(), *retired.ordered()))


def test_source_swap_before_clone_preserves_both_contents(migration_fixture, monkeypatch):
    """The clone must use the validated fd while a source-path replacement stays named."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    original = legacy.memmap.read_bytes()
    replacement = b"source pathname replacement"
    real_clone = getattr(migrate, "_clone_from_fd_no_replace", None)
    injected = False

    def replace_source_then_clone(source_fd, root_fd, destination):
        nonlocal injected
        if destination == target.memmap.name and not injected:
            legacy.memmap.unlink()
            legacy.memmap.write_bytes(replacement)
            injected = True
        return real_clone(source_fd, root_fd, destination)

    monkeypatch.setattr(
        migrate, "_clone_from_fd_no_replace", replace_source_then_clone, raising=False
    )

    with pytest.raises(migrate.ArtifactMigrationError):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert injected
    assert legacy.memmap.read_bytes() == replacement
    assert target.memmap.read_bytes() == original
    assert not migrate.retired_artifact_paths(artifact_dir, MODEL).memmap.exists()


def test_target_swap_after_clone_survives_and_blocks_retirement(migration_fixture, monkeypatch):
    """A target replacement must remain named and must not cause legacy retirement."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    replacement = b"target pathname replacement"
    real_clone = getattr(migrate, "_clone_from_fd_no_replace", None)
    injected = False

    def clone_then_replace_target(source_fd, root_fd, destination):
        nonlocal injected
        result = real_clone(source_fd, root_fd, destination)
        if destination == target.memmap.name and not injected:
            target.memmap.unlink()
            target.memmap.write_bytes(replacement)
            injected = True
        return result

    monkeypatch.setattr(
        migrate, "_clone_from_fd_no_replace", clone_then_replace_target, raising=False
    )

    with pytest.raises(migrate.ArtifactMigrationError):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert injected
    assert target.memmap.read_bytes() == replacement
    assert all(path.exists() for path in legacy.ordered())
    assert not migrate.retired_artifact_paths(artifact_dir, MODEL).memmap.exists()


def test_retirement_swap_survives_and_blocks_success(migration_fixture, monkeypatch):
    """A post-rename retirement replacement must stay named and prevent success."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    original = legacy.memmap.read_bytes()
    replacement = b"retirement pathname replacement"
    real_rename = migrate._rename_no_replace
    injected = False

    def retire_then_replace(root_fd, source, destination):
        nonlocal injected
        result = real_rename(root_fd, source, destination)
        if destination == retired.memmap.name and not injected:
            retired.memmap.unlink()
            retired.memmap.write_bytes(replacement)
            injected = True
        return result

    monkeypatch.setattr(migrate, "_rename_no_replace", retire_then_replace)

    with pytest.raises(migrate.ArtifactMigrationError):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert injected
    assert retired.memmap.read_bytes() == replacement
    assert target.memmap.read_bytes() == original
    assert not legacy.memmap.exists()


def test_legacy_replacement_at_retirement_linearization_is_retained(migration_fixture, monkeypatch):
    """A rename-time source replacement must be named at R while T keeps old bytes."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    original = legacy.memmap.read_bytes()
    replacement = b"legacy replacement at retirement linearization"
    replacement_path = artifact_dir / "replacement.fixture"
    replacement_path.write_bytes(replacement)
    real_rename = migrate._rename_no_replace
    injected = False

    def replace_legacy_then_retire(root_fd, source, destination):
        nonlocal injected
        if destination == retired.memmap.name and not injected:
            os.replace(replacement_path, legacy.memmap)
            injected = True
        return real_rename(root_fd, source, destination)

    monkeypatch.setattr(migrate, "_rename_no_replace", replace_legacy_then_retire)

    with pytest.raises(migrate.ArtifactMigrationError, match="retired artifact identity"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert injected
    assert target.memmap.read_bytes() == original
    assert retired.memmap.read_bytes() == replacement
    assert not legacy.memmap.exists()
    assert all(path.exists() for path in legacy.ordered()[1:])
    assert all(not path.exists() for path in retired.ordered()[1:])


def test_retirement_destination_appearance_never_overwrites_or_retires_later_roles(
    migration_fixture, monkeypatch
):
    """An R-name race must preserve L, T, and R and stop before later renames."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    conflict = b"concurrent retirement occupant"
    originals = tuple(path.read_bytes() for path in legacy.ordered())
    real_rename = migrate._rename_no_replace
    injected = False

    def create_retirement_then_attempt_rename(root_fd, source, destination):
        nonlocal injected
        if destination == retired.memmap.name and not injected:
            retired.memmap.write_bytes(conflict)
            injected = True
        return real_rename(root_fd, source, destination)

    monkeypatch.setattr(migrate, "_rename_no_replace", create_retirement_then_attempt_rename)

    with pytest.raises(migrate.ArtifactMigrationError, match="retired artifact appeared"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert injected
    assert tuple(path.read_bytes() for path in legacy.ordered()) == originals
    assert tuple(path.read_bytes() for path in target.ordered()) == originals
    assert retired.memmap.read_bytes() == conflict
    assert all(not path.exists() for path in retired.ordered()[1:])


def test_interrupt_after_clone_replays_content_partition(migration_fixture, monkeypatch):
    """A crash after clone publication must leave L+T and resume without deletion."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    real_clone = getattr(migrate, "_clone_from_fd_no_replace", None)
    interrupted = False

    def interrupt_after_first_clone(source_fd, root_fd, destination):
        nonlocal interrupted
        result = real_clone(source_fd, root_fd, destination)
        if destination == target.memmap.name and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(
        migrate, "_clone_from_fd_no_replace", interrupt_after_first_clone, raising=False
    )

    with pytest.raises(KeyboardInterrupt):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert interrupted
    assert legacy.memmap.exists() and target.memmap.exists()
    assert not retired.memmap.exists()
    monkeypatch.setattr(migrate, "_clone_from_fd_no_replace", real_clone, raising=False)

    report = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert report.state == "migrated"
    assert all(path.exists() for path in (*target.ordered(), *retired.ordered()))


def test_interrupt_after_retirement_replays_mixed_partition(migration_fixture, monkeypatch):
    """A crash after L-to-R must resume from T+R without touching retained entries."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    real_rename = migrate._rename_no_replace
    interrupted = False

    def interrupt_after_first_retirement(root_fd, source, destination):
        nonlocal interrupted
        result = real_rename(root_fd, source, destination)
        if destination == retired.memmap.name and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(migrate, "_rename_no_replace", interrupt_after_first_retirement)

    with pytest.raises(KeyboardInterrupt):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert interrupted
    assert target.memmap.exists() and retired.memmap.exists()
    assert not legacy.memmap.exists()
    assert all(path.exists() for path in target.ordered()[1:])
    monkeypatch.setattr(migrate, "_rename_no_replace", real_rename)

    report = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert report.state == "migrated"
    assert all(path.exists() for path in (*target.ordered(), *retired.ordered()))


def test_directory_fsync_error_traceback_is_privacy_safe(migration_fixture, monkeypatch):
    """Raw directory-fsync messages can disclose stored filesystem content."""
    con, artifact_dir, _, _ = migration_fixture
    real_fsync = migrate.os.fsync

    def fail_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(FIXTURE_NARRATIVE)
        return real_fsync(fd)

    monkeypatch.setattr(migrate.os, "fsync", fail_directory_fsync)

    with pytest.raises(migrate.ArtifactMigrationError, match="synchronized") as excinfo:
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    rendered = "".join(traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb))
    assert FIXTURE_NARRATIVE not in rendered


def test_regular_file_fsync_error_traceback_is_privacy_safe(migration_fixture, monkeypatch):
    """Raw regular-file fsync messages must not escape the migration boundary."""
    con, artifact_dir, _, _ = migration_fixture

    def fail_regular_fsync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(FIXTURE_NARRATIVE)
        pytest.fail("a directory fsync must not precede source-file durability")

    monkeypatch.setattr(migrate.os, "fsync", fail_regular_fsync)

    with pytest.raises(migrate.ArtifactMigrationError, match="file cannot") as excinfo:
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    rendered = "".join(traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb))
    assert FIXTURE_NARRATIVE not in rendered


def test_execute_uses_no_hard_link_or_unlink_namespace_transition(migration_fixture, monkeypatch):
    con, artifact_dir, _, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)

    monkeypatch.setattr(
        migrate.os,
        "link",
        lambda *args, **kwargs: pytest.fail("migration must not hard-link artifacts"),
    )
    monkeypatch.setattr(
        migrate.os,
        "unlink",
        lambda *args, **kwargs: pytest.fail("migration must not unlink artifacts"),
    )

    report = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert report.state == "migrated"
    assert all(path.exists() for path in (*target.ordered(), *retired.ordered()))


def test_second_execute_is_already_migrated(migration_fixture):
    con, artifact_dir, _, _ = migration_fixture
    first = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)
    before = tuple(
        (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        for path in (*first.target.ordered(), *first.retired.ordered())
        for metadata in (path.stat(),)
    )
    second = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert second.state == "already-migrated"
    after = tuple(
        (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        for path in (*second.target.ordered(), *second.retired.ordered())
        for metadata in (path.stat(),)
    )
    assert after == before


def test_fully_retired_replay_closes_interrupted_rename_durability_gap(
    migration_fixture, monkeypatch
):
    """A visible T+R partition may be replaying a rename that crashed before dir fsync."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    for source, destination, retirement in zip(
        legacy.ordered(), target.ordered(), retired.ordered(), strict=True
    ):
        destination.write_bytes(source.read_bytes())
        source.rename(retirement)
    expected_regular = {path.stat().st_ino for path in (*target.ordered(), *retired.ordered())}
    real_fsync = migrate.os.fsync
    regular_inodes = set()
    directory_fsyncs = 0

    def record_fsync(fd):
        nonlocal directory_fsyncs
        metadata = os.fstat(fd)
        if stat.S_ISREG(metadata.st_mode):
            regular_inodes.add(metadata.st_ino)
        elif stat.S_ISDIR(metadata.st_mode):
            directory_fsyncs += 1
        return real_fsync(fd)

    monkeypatch.setattr(migrate.os, "fsync", record_fsync)

    report = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert report.state == "already-migrated"
    assert expected_regular <= regular_inodes
    assert directory_fsyncs >= 1


def test_fully_migrated_rejects_transient_validation_root_swap(
    migration_fixture, tmp_path, monkeypatch
):
    con, artifact_dir, _, _ = migration_fixture
    report = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)
    target = report.target
    retired = report.retired
    valid_root = tmp_path / "valid-root"
    valid_root.mkdir()
    valid_target = migrate.target_artifact_paths(valid_root, MODEL)
    valid_retired = migrate.retired_artifact_paths(valid_root, MODEL)
    for source, destination in zip(target.ordered(), valid_target.ordered(), strict=True):
        destination.write_bytes(source.read_bytes())
    for source, destination in zip(retired.ordered(), valid_retired.ordered(), strict=True):
        destination.write_bytes(source.read_bytes())
    payload = json.loads(retired.progress.read_text())
    payload["model"] = "other/model"
    retired.progress.write_text(json.dumps(payload))
    before = sorted(path.name for path in artifact_dir.iterdir())
    held_root = tmp_path / "held-pinned-root"
    real_validate = migrate._validate_bound_artifacts

    def validate_during_root_swap(*args, **kwargs):
        artifact_dir.rename(held_root)
        valid_root.rename(artifact_dir)
        try:
            return real_validate(*args, **kwargs)
        finally:
            artifact_dir.rename(valid_root)
            held_root.rename(artifact_dir)

    monkeypatch.setattr(migrate, "_validate_bound_artifacts", validate_during_root_swap)

    with pytest.raises(migrate.ArtifactMigrationError, match="progress"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert sorted(path.name for path in artifact_dir.iterdir()) == before
    assert json.loads(retired.progress.read_text())["model"] == "other/model"


def test_migrated_rechecks_root_after_final_bound_digest(migration_fixture, tmp_path, monkeypatch):
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    moved_root = tmp_path / "moved-pinned-root"
    real_verify = migrate._verify_bound_content
    moved = False

    def move_root_after_final_digest(bound):
        nonlocal moved
        real_verify(bound)
        if (
            not moved
            and bound.artifacts == legacy
            and all(path.exists() for path in retired.ordered())
        ):
            moved = True
            artifact_dir.rename(moved_root)
            artifact_dir.mkdir()

    monkeypatch.setattr(migrate, "_verify_bound_content", move_root_after_final_digest)

    with pytest.raises(migrate.ArtifactMigrationError, match="directory identity changed"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert moved
    assert all(not path.exists() for path in (*target.ordered(), *retired.ordered()))
    assert all(
        (moved_root / path.name).exists() for path in (*target.ordered(), *retired.ordered())
    )


def test_migrated_rechecks_root_after_final_namespace_inspection(
    migration_fixture, tmp_path, monkeypatch
):
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    moved_root = tmp_path / "moved-after-migration-inspection"
    real_inspect = migrate._inspect_set
    moved = False

    def move_root_after_final_legacy_inspection(root_fd, artifacts):
        nonlocal moved
        result = real_inspect(root_fd, artifacts)
        if artifacts == legacy and not moved and all(path.exists() for path in retired.ordered()):
            moved = True
            artifact_dir.rename(moved_root)
            artifact_dir.mkdir()
        return result

    monkeypatch.setattr(migrate, "_inspect_set", move_root_after_final_legacy_inspection)

    with pytest.raises(migrate.ArtifactMigrationError, match="directory identity changed"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert moved
    assert all(not path.exists() for path in (*target.ordered(), *retired.ordered()))
    assert all(
        (moved_root / path.name).exists() for path in (*target.ordered(), *retired.ordered())
    )


def test_partial_target_without_complete_legacy_fails(migration_fixture):
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    legacy.memmap.rename(target.memmap)
    legacy.index.unlink()
    before = sorted(path.name for path in artifact_dir.iterdir())

    with pytest.raises(migrate.ArtifactMigrationError):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert sorted(path.name for path in artifact_dir.iterdir()) == before


def test_complete_target_only_state_fails_closed_without_mutation(migration_fixture):
    """SHA-only state lacks the retained independent comparison required for replay."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    for source, destination in zip(legacy.ordered(), target.ordered(), strict=True):
        destination.write_bytes(source.read_bytes())
        source.unlink()
    before = sorted((path.name, path.read_bytes()) for path in artifact_dir.iterdir())

    with pytest.raises(migrate.ArtifactMigrationError, match="clone-retirement partition"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert sorted((path.name, path.read_bytes()) for path in artifact_dir.iterdir()) == before


@pytest.mark.parametrize("invalid_phase", ["retirement-only", "legacy-retirement", "all-three"])
def test_invalid_clone_retirement_presence_states_fail_without_mutation(
    migration_fixture, invalid_phase
):
    """R-only, L+R, and L+T+R are outside the monotonic replay partition."""
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    original = legacy.memmap.read_bytes()
    if invalid_phase == "retirement-only":
        legacy.memmap.rename(retired.memmap)
    elif invalid_phase == "legacy-retirement":
        retired.memmap.write_bytes(original)
    else:
        target.memmap.write_bytes(original)
        retired.memmap.write_bytes(original)
    before = sorted((path.name, path.read_bytes()) for path in artifact_dir.iterdir())

    with pytest.raises(migrate.ArtifactMigrationError, match="clone-retirement partition"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert sorted((path.name, path.read_bytes()) for path in artifact_dir.iterdir()) == before


def test_clone_failure_leaves_resumable_partition(migration_fixture, monkeypatch):
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    real_clone = migrate._clone_from_fd_no_replace
    calls = 0

    def fail_second_clone(source_fd, root_fd, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(FIXTURE_NARRATIVE)
        return real_clone(source_fd, root_fd, destination)

    monkeypatch.setattr(migrate, "_clone_from_fd_no_replace", fail_second_clone)

    with pytest.raises(migrate.ArtifactMigrationError, match="descriptor-bound") as excinfo:
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert FIXTURE_NARRATIVE not in "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )
    assert target.memmap.exists()
    assert all(not path.exists() for path in target.ordered()[1:])
    assert all(path.exists() for path in legacy.ordered())
    assert all(not path.exists() for path in retired.ordered())

    monkeypatch.setattr(migrate, "_clone_from_fd_no_replace", real_clone)
    report = migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert report.state == "migrated"
    assert all(path.exists() for path in (*target.ordered(), *retired.ordered()))


def test_post_clone_hard_link_substitution_fails_closed(migration_fixture, monkeypatch):
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    real_clone = migrate._clone_from_fd_no_replace
    injected = False

    def substitute_target_with_legacy_alias(source_fd, root_fd, destination):
        nonlocal injected
        result = real_clone(source_fd, root_fd, destination)
        if destination == target.progress.name and not injected:
            target.memmap.unlink()
            os.link(legacy.memmap, target.memmap)
            injected = True
        return result

    monkeypatch.setattr(migrate, "_clone_from_fd_no_replace", substitute_target_with_legacy_alias)

    with pytest.raises(migrate.ArtifactMigrationError, match="hard-link alias"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert injected
    assert target.memmap.stat().st_ino == legacy.memmap.stat().st_ino
    assert all(path.exists() for path in legacy.ordered())
    assert all(not path.exists() for path in retired.ordered())


def test_destination_appearance_before_clone_is_never_overwritten(migration_fixture, monkeypatch):
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    replacement_bytes = b"replacement target"
    real_clone = migrate._clone_from_fd_no_replace
    injected = False

    def create_destination_then_clone(source_fd, root_fd, destination):
        nonlocal injected
        if destination == target.memmap.name and not injected:
            target.memmap.write_bytes(replacement_bytes)
            injected = True
        return real_clone(source_fd, root_fd, destination)

    monkeypatch.setattr(migrate, "_clone_from_fd_no_replace", create_destination_then_clone)

    with pytest.raises(migrate.ArtifactMigrationError, match="appeared during clone"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert injected
    assert target.memmap.read_bytes() == replacement_bytes
    assert all(path.exists() for path in legacy.ordered())


def test_recreated_legacy_name_after_retirement_survives_and_blocks_success(
    migration_fixture, monkeypatch
):
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    replacement_bytes = b"replacement at released legacy basename"
    real_move = migrate._rename_no_replace
    replacement_injected = False

    def recreate_legacy_after_move(root_fd, source, destination):
        nonlocal replacement_injected
        result = real_move(root_fd, source, destination)
        if destination == retired.memmap.name and not replacement_injected:
            legacy.memmap.write_bytes(replacement_bytes)
            replacement_injected = True
        return result

    monkeypatch.setattr(migrate, "_rename_no_replace", recreate_legacy_after_move)

    with pytest.raises(migrate.ArtifactMigrationError, match="retired artifact identity"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert replacement_injected
    assert legacy.memmap.read_bytes() == replacement_bytes
    assert target.memmap.exists()
    assert retired.memmap.exists()


def test_in_place_moved_artifact_mutation_is_detected(migration_fixture, monkeypatch):
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    real_fsync = migrate.os.fsync
    mutated = False

    def mutate_target_after_first_move(fd):
        nonlocal mutated
        if stat.S_ISDIR(os.fstat(fd).st_mode) and target.memmap.exists() and not mutated:
            target.memmap.write_bytes(b"in-place mutation")
            mutated = True
        return real_fsync(fd)

    monkeypatch.setattr(migrate.os, "fsync", mutate_target_after_first_move)

    with pytest.raises(migrate.ArtifactMigrationError):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert all(path.exists() for path in legacy.ordered())
    assert target.memmap.read_bytes() == b"in-place mutation"
    assert all(path.exists() for path in target.ordered()[1:])
    assert all(not path.exists() for path in retired.ordered())


def test_same_metadata_content_substitution_fails_before_success(migration_fixture, monkeypatch):
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    original = legacy.progress.read_bytes()
    assert b" " in original
    metadata = legacy.progress.stat()
    substituted = original.replace(b" ", b"\t", 1)
    real_clone = migrate._clone_from_fd_no_replace
    substituted_once = False

    def substitute_content_before_clone(source_fd, root_fd, destination):
        nonlocal substituted_once
        if destination == target.memmap.name and not substituted_once:
            legacy.progress.write_bytes(substituted)
            os.utime(
                legacy.progress,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
            )
            substituted_once = True
        return real_clone(source_fd, root_fd, destination)

    monkeypatch.setattr(migrate, "_clone_from_fd_no_replace", substitute_content_before_clone)

    with pytest.raises(migrate.ArtifactMigrationError, match="content"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert legacy.progress.stat().st_ino == metadata.st_ino
    assert legacy.progress.stat().st_size == metadata.st_size
    assert legacy.progress.stat().st_mtime_ns == metadata.st_mtime_ns
    assert legacy.progress.read_bytes() == substituted


def test_target_validation_failure_leaves_resumable_target_partition(
    migration_fixture, monkeypatch
):
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    legacy_inodes = tuple(path.stat().st_ino for path in legacy.ordered())
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    real_validate = migrate._validate_bound_artifacts
    calls = 0

    def fail_target_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise migrate.ArtifactMigrationError("injected target validation failure")
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(migrate, "_validate_bound_artifacts", fail_target_validation)

    with pytest.raises(migrate.ArtifactMigrationError, match="target validation failure"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert all(
        path.stat().st_ino != source_inode
        for path, source_inode in zip(target.ordered(), legacy_inodes, strict=True)
    )
    assert all(path.exists() for path in legacy.ordered())
    assert all(not path.exists() for path in retired.ordered())

    monkeypatch.setattr(migrate, "_validate_bound_artifacts", real_validate)
    assert (
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True).state
        == "migrated"
    )


def test_sidecar_is_published_last(migration_fixture, monkeypatch):
    con, artifact_dir, _, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    real_clone = migrate._clone_from_fd_no_replace
    real_move = migrate._rename_no_replace
    clone_destinations = []
    retirement_destinations = []

    def record_clone(source_fd, root_fd, destination):
        clone_destinations.append(destination)
        return real_clone(source_fd, root_fd, destination)

    def record_move(root_fd, source, destination):
        retirement_destinations.append(destination)
        return real_move(root_fd, source, destination)

    monkeypatch.setattr(migrate, "_clone_from_fd_no_replace", record_clone)
    monkeypatch.setattr(migrate, "_rename_no_replace", record_move)

    migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert clone_destinations == [path.name for path in target.ordered()]
    assert retirement_destinations == [path.name for path in retired.ordered()]


def test_root_rename_during_retirement_leaves_partition_in_pinned_root(
    migration_fixture, monkeypatch
):
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    moved_root = artifact_dir.with_name("moved-artifacts")
    legacy_names = [path.name for path in legacy.ordered()]
    real_move = migrate._rename_no_replace
    calls = 0

    def rename_root_during_second_move(root_fd, source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            artifact_dir.rename(moved_root)
            artifact_dir.mkdir()
        return real_move(root_fd, source, destination)

    monkeypatch.setattr(migrate, "_rename_no_replace", rename_root_during_second_move)

    with pytest.raises(migrate.ArtifactMigrationError, match="directory identity changed"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert all((moved_root / path.name).exists() for path in target.ordered())
    assert all(not path.exists() for path in target.ordered())
    assert all(not (moved_root / name).exists() for name in legacy_names[:2])
    assert (moved_root / legacy_names[2]).exists()
    assert all((moved_root / path.name).exists() for path in retired.ordered()[:2])
    assert not (moved_root / retired.progress.name).exists()


def test_symlink_target_is_not_followed(migration_fixture, tmp_path):
    con, artifact_dir, legacy, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    sentinel = tmp_path / "outside-sentinel"
    sentinel.write_bytes(b"outside bytes")
    target.memmap.symlink_to(sentinel)
    legacy_inodes = tuple(path.stat().st_ino for path in legacy.ordered())

    with pytest.raises(migrate.ArtifactMigrationError):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert sentinel.read_bytes() == b"outside bytes"
    assert tuple(path.stat().st_ino for path in legacy.ordered()) == legacy_inodes
    assert target.memmap.is_symlink()


def test_execute_fsyncs_regular_artifacts_before_success(migration_fixture, monkeypatch):
    con, artifact_dir, legacy, _ = migration_fixture
    source_inodes = {path.stat().st_ino for path in legacy.ordered()}
    real_fsync = migrate.os.fsync
    regular_inodes = set()
    directory_fsyncs = 0

    def record_fsync(fd):
        nonlocal directory_fsyncs
        metadata = os.fstat(fd)
        if stat.S_ISREG(metadata.st_mode):
            regular_inodes.add(metadata.st_ino)
        elif stat.S_ISDIR(metadata.st_mode):
            directory_fsyncs += 1
        return real_fsync(fd)

    monkeypatch.setattr(migrate.os, "fsync", record_fsync)

    migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert source_inodes <= regular_inodes
    assert directory_fsyncs > 0


def test_primary_error_survives_root_close_failure(migration_fixture, monkeypatch):
    con, artifact_dir, legacy, _ = migration_fixture
    payload = json.loads(legacy.progress.read_text())
    payload["model"] = "other/model"
    legacy.progress.write_text(json.dumps(payload))
    real_open = migrate.os.open
    real_close = migrate.os.close
    root_fd = None

    def record_root_open(path, flags, *args, **kwargs):
        nonlocal root_fd
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == artifact_dir and kwargs.get("dir_fd") is None:
            root_fd = fd
        return fd

    def fail_root_close(fd):
        if fd == root_fd:
            raise OSError(FIXTURE_NARRATIVE)
        return real_close(fd)

    monkeypatch.setattr(migrate.os, "open", record_root_open)
    monkeypatch.setattr(migrate.os, "close", fail_root_close)

    with pytest.raises(migrate.ArtifactMigrationError, match="progress") as excinfo:
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL)

    assert excinfo.value.__notes__ == ["close error: OSError"]
    assert FIXTURE_NARRATIVE not in str(excinfo.value)
    assert FIXTURE_NARRATIVE not in " ".join(excinfo.value.__notes__)
    assert FIXTURE_NARRATIVE not in "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )


def test_successful_state_surfaces_root_close_failure(migration_fixture, monkeypatch):
    con, artifact_dir, _, _ = migration_fixture
    before = sorted(path.name for path in artifact_dir.iterdir())
    real_open = migrate.os.open
    real_close = migrate.os.close
    root_fd = None

    def record_root_open(path, flags, *args, **kwargs):
        nonlocal root_fd
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == artifact_dir and kwargs.get("dir_fd") is None:
            root_fd = fd
        return fd

    def fail_root_close(fd):
        if fd == root_fd:
            raise OSError(FIXTURE_NARRATIVE)
        return real_close(fd)

    monkeypatch.setattr(migrate.os, "open", record_root_open)
    monkeypatch.setattr(migrate.os, "close", fail_root_close)

    with pytest.raises(migrate.ArtifactMigrationError, match="close") as excinfo:
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL)

    assert sorted(path.name for path in artifact_dir.iterdir()) == before
    assert FIXTURE_NARRATIVE not in str(excinfo.value)
    assert FIXTURE_NARRATIVE not in "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )


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


@pytest.mark.parametrize("failure_call", [2, 3, 4, 5])
def test_validation_wraps_every_late_mapping_read_error(migration_fixture, failure_call):
    con, _, legacy, _ = migration_fixture
    failing_con = _FailingMappingConnection(con, failure_call)

    _assert_rejected(failing_con, legacy)


def test_mapping_read_error_traceback_does_not_disclose_stored_narrative(migration_fixture):
    con, _, legacy, _ = migration_fixture
    failing_con = _FailingMappingConnection(con, 2)

    with pytest.raises(migrate.ArtifactMigrationError) as excinfo:
        migrate.validate_artifacts(failing_con, legacy, MODEL)

    rendered = "".join(traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb))
    assert "embedding mapping cannot be read" in rendered
    assert FIXTURE_NARRATIVE not in rendered


def test_validation_errors_do_not_disclose_stored_narrative(migration_fixture):
    con, _, legacy, _ = migration_fixture
    stored_narrative = con.execute(
        "SELECT text_redacted FROM narratives WHERE complaint_id = 10"
    ).fetchone()[0]
    assert stored_narrative == FIXTURE_NARRATIVE
    con.execute("DELETE FROM embedding_map WHERE complaint_id = 10 AND model = ?", [MODEL])

    _assert_rejected(con, legacy)


@pytest.mark.parametrize(
    "field,value",
    [("model", "other/model"), ("n_done", 1), ("n_total", 3), ("dim", 3)],
)
def test_validation_rejects_wrong_progress(migration_fixture, field, value):
    con, _, legacy, _ = migration_fixture
    payload = json.loads(legacy.progress.read_text())
    payload[field] = value
    legacy.progress.write_text(json.dumps(payload))

    _assert_rejected(con, legacy)


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

    _assert_rejected(con, legacy)


def test_validation_rejects_unexpected_or_badly_typed_progress_field(migration_fixture):
    con, _, legacy, _ = migration_fixture
    payload = json.loads(legacy.progress.read_text())
    payload["unexpected"] = True
    legacy.progress.write_text(json.dumps(payload))

    _assert_rejected(con, legacy)


def test_validation_rejects_wrong_faiss_metric(migration_fixture):
    con, _, legacy, vectors = migration_fixture
    index = faiss.IndexFlatL2(2)
    index.add(vectors)
    faiss.write_index(index, str(legacy.index))

    _assert_rejected(con, legacy)


def test_validation_rejects_wrong_faiss_count(migration_fixture):
    con, _, legacy, vectors = migration_fixture
    index = faiss.IndexFlatIP(2)
    index.add(vectors[:1])
    faiss.write_index(index, str(legacy.index))

    _assert_rejected(con, legacy)


def test_validation_rejects_wrong_faiss_content(migration_fixture):
    con, _, legacy, vectors = migration_fixture
    index = faiss.IndexFlatIP(2)
    index.add(vectors[::-1].copy())
    faiss.write_index(index, str(legacy.index))

    _assert_rejected(con, legacy)


def test_validation_rejects_source_symlink(migration_fixture):
    con, artifact_dir, legacy, _ = migration_fixture
    replacement = artifact_dir / "replacement.npy"
    np.save(replacement, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    legacy.memmap.unlink()
    legacy.memmap.symlink_to(replacement.name)

    _assert_rejected(con, legacy)


def test_validation_rejects_source_inode_replacement_during_validation(
    migration_fixture, monkeypatch
):
    con, artifact_dir, legacy, vectors = migration_fixture
    real_read_index = faiss.read_index

    def replace_source_then_read(path):
        replacement = artifact_dir / "replacement.npy"
        np.save(replacement, vectors)
        os.replace(replacement, legacy.memmap)
        return real_read_index(path)

    monkeypatch.setattr(faiss, "read_index", replace_source_then_read)

    _assert_rejected(con, legacy)


def test_validation_rejects_deleted_mapping(migration_fixture):
    con, _, legacy, _ = migration_fixture
    con.execute("DELETE FROM embedding_map WHERE complaint_id = 10 AND model = ?", [MODEL])

    _assert_rejected(con, legacy)


def test_validation_rejects_changed_mapping_dimension(migration_fixture):
    con, _, legacy, _ = migration_fixture
    con.execute("UPDATE embedding_map SET dim = 3 WHERE complaint_id = 12 AND model = ?", [MODEL])

    _assert_rejected(con, legacy)


def test_validation_rejects_mapping_row_gap(migration_fixture):
    con, _, legacy, _ = migration_fixture
    con.execute(
        "UPDATE embedding_map SET row_idx = 0 WHERE complaint_id = 12 AND model = ?", [MODEL]
    )

    _assert_rejected(con, legacy)


def test_validation_rejects_out_of_range_mapping_row(migration_fixture):
    con, _, legacy, _ = migration_fixture
    con.execute(
        "UPDATE embedding_map SET row_idx = 2 WHERE complaint_id = 12 AND model = ?", [MODEL]
    )

    _assert_rejected(con, legacy)


def test_validation_rejects_extra_mapping(migration_fixture):
    con, _, legacy, _ = migration_fixture
    con.execute("INSERT INTO embedding_map VALUES (99, 0, ?, 2)", [MODEL])

    _assert_rejected(con, legacy)


def test_target_mutation_during_final_retired_validation_fails_closed(
    migration_fixture, monkeypatch
):
    con, artifact_dir, _, _ = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    retired = migrate.retired_artifact_paths(artifact_dir, MODEL)
    real_validate = migrate._validate_bound_artifacts
    retired_validations = 0
    substituted = b""

    def mutate_after_retired_validation(connection, bound, model_name):
        nonlocal retired_validations, substituted
        result = real_validate(connection, bound, model_name)
        if bound.artifacts == retired:
            retired_validations += 1
            if retired_validations == 2:
                metadata = target.progress.stat()
                original = target.progress.read_bytes()
                assert b" " in original
                substituted = original.replace(b" ", b"\t", 1)
                target.progress.write_bytes(substituted)
                os.utime(
                    target.progress,
                    ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
                )
        return result

    monkeypatch.setattr(migrate, "_validate_bound_artifacts", mutate_after_retired_validation)

    with pytest.raises(migrate.ArtifactMigrationError, match="content"):
        migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)

    assert retired_validations == 2
    assert target.progress.read_bytes() == substituted


def test_validation_rejects_wrong_text_hash_to_row_semantics(migration_fixture):
    con, _, legacy, _ = migration_fixture
    con.execute(
        "UPDATE embedding_map SET row_idx = CASE complaint_id "
        "WHEN 10 THEN 1 WHEN 11 THEN 1 WHEN 12 THEN 0 END WHERE model = ?",
        [MODEL],
    )

    _assert_rejected(con, legacy)


def test_cli_requires_model_and_defaults_to_plan_mode():
    """Removing explicit model selection could target a different artifact set."""
    parser = pipeline.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["migrate-embeddings"])

    args = parser.parse_args(["migrate-embeddings", "--model", MODEL])

    assert args.model == MODEL
    assert args.execute is False


def test_execute_help_describes_clone_retirement_protocol(capsys):
    parser = pipeline.build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["migrate-embeddings", "--execute", "--help"])

    assert excinfo.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "copy-on-write clones" in help_text
    assert "exact content on different inodes" in help_text
    assert "retain legacy artifacts under deterministic retirement names for replay" in help_text
    assert "hard-link artifacts" not in help_text
    assert "remove legacy aliases" not in help_text


def test_handler_uses_existing_read_only_database_closes_and_renders(monkeypatch, tmp_path, capsys):
    """Changing the handler to bootstrap/write or leak a connection is unsafe maintenance I/O."""
    database = tmp_path / "existing.duckdb"
    database.touch()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    connection = SimpleNamespace(close=lambda: closed.append(True))
    closed: list[bool] = []
    calls: list[tuple[Path, bool]] = []
    report = SimpleNamespace(state="planned", render=lambda: "state : planned")

    monkeypatch.setattr(pipeline, "PATHS", SimpleNamespace(db=database, artifacts=artifacts))
    monkeypatch.setattr(
        pipeline.db,
        "connect",
        lambda path, read_only: calls.append((path, read_only)) or connection,
    )
    monkeypatch.setattr(
        pipeline.db,
        "bootstrap",
        lambda: pytest.fail("migration handler must not bootstrap the database"),
    )
    monkeypatch.setattr(
        migrate,
        "migrate_embedding_artifacts",
        lambda con, root, model, *, execute: report,
    )

    assert pipeline.cmd_migrate_embeddings(SimpleNamespace(model=MODEL, execute=False)) == 0
    assert calls == [(database, True)]
    assert closed == [True]
    assert capsys.readouterr().out == "state : planned\n"


def test_handler_closes_read_only_connection_when_migration_rejects(monkeypatch, tmp_path):
    """A validation failure must not leave the maintenance database connection open."""
    database = tmp_path / "existing.duckdb"
    database.touch()
    closed: list[bool] = []
    connection = SimpleNamespace(close=lambda: closed.append(True))

    monkeypatch.setattr(
        pipeline,
        "PATHS",
        SimpleNamespace(db=database, artifacts=tmp_path / "artifacts"),
    )
    monkeypatch.setattr(pipeline.db, "connect", lambda path, read_only: connection)
    monkeypatch.setattr(
        pipeline.db,
        "bootstrap",
        lambda: pytest.fail("migration handler must not bootstrap the database"),
    )

    def reject_migration(con, root, model, *, execute):
        raise migrate.ArtifactMigrationError("fixture validation failure")

    monkeypatch.setattr(migrate, "migrate_embedding_artifacts", reject_migration)

    with pytest.raises(migrate.ArtifactMigrationError, match="fixture validation failure"):
        pipeline.cmd_migrate_embeddings(SimpleNamespace(model=MODEL, execute=False))

    assert closed == [True]


def test_handler_primary_error_survives_connection_close_failure(monkeypatch, tmp_path):
    database = tmp_path / "existing.duckdb"
    database.touch()

    def fail_close():
        raise RuntimeError(FIXTURE_NARRATIVE)

    connection = SimpleNamespace(close=fail_close)
    monkeypatch.setattr(
        pipeline,
        "PATHS",
        SimpleNamespace(db=database, artifacts=tmp_path / "artifacts"),
    )
    monkeypatch.setattr(pipeline.db, "connect", lambda path, read_only: connection)

    def reject_migration(con, root, model, *, execute):
        raise migrate.ArtifactMigrationError("fixture validation failure")

    monkeypatch.setattr(migrate, "migrate_embedding_artifacts", reject_migration)

    with pytest.raises(
        migrate.ArtifactMigrationError, match="fixture validation failure"
    ) as excinfo:
        pipeline.cmd_migrate_embeddings(SimpleNamespace(model=MODEL, execute=False))

    assert excinfo.value.__notes__ == ["close error: RuntimeError"]
    assert FIXTURE_NARRATIVE not in "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )


def test_handler_success_surfaces_private_connection_close_failure(monkeypatch, tmp_path):
    database = tmp_path / "existing.duckdb"
    database.touch()

    def fail_close():
        raise RuntimeError(FIXTURE_NARRATIVE)

    connection = SimpleNamespace(close=fail_close)
    report = SimpleNamespace(state="planned", render=lambda: "state : planned")
    monkeypatch.setattr(
        pipeline,
        "PATHS",
        SimpleNamespace(db=database, artifacts=tmp_path / "artifacts"),
    )
    monkeypatch.setattr(pipeline.db, "connect", lambda path, read_only: connection)
    monkeypatch.setattr(
        migrate,
        "migrate_embedding_artifacts",
        lambda con, root, model, *, execute: report,
    )

    with pytest.raises(
        migrate.ArtifactMigrationError, match="database connection close failed"
    ) as excinfo:
        pipeline.cmd_migrate_embeddings(SimpleNamespace(model=MODEL, execute=False))

    assert excinfo.value.__suppress_context__ is True
    assert FIXTURE_NARRATIVE not in "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )


def test_handler_refuses_to_create_a_missing_database(monkeypatch, tmp_path):
    """A missing database must fail before a connection can create it."""
    missing_database = tmp_path / "missing.duckdb"
    monkeypatch.setattr(
        pipeline,
        "PATHS",
        SimpleNamespace(db=missing_database, artifacts=tmp_path / "artifacts"),
    )
    monkeypatch.setattr(
        pipeline.db,
        "connect",
        lambda *args, **kwargs: pytest.fail("missing database must not be opened"),
    )

    with pytest.raises(SystemExit, match="database does not exist"):
        pipeline.cmd_migrate_embeddings(SimpleNamespace(model=MODEL, execute=False))


def test_current_loader_uses_only_sha_artifacts_after_migration(migration_fixture, monkeypatch):
    """Reintroducing a legacy sidecar cannot make the strict current loader accept it."""
    con, artifact_dir, legacy, expected_vectors = migration_fixture
    target = migrate.target_artifact_paths(artifact_dir, MODEL)
    migrate.migrate_embedding_artifacts(con, artifact_dir, MODEL, execute=True)
    np.save(legacy.memmap, expected_vectors)
    encode.Progress(legacy.progress, 2, 2, 2, MODEL).write()
    target.progress.unlink()
    monkeypatch.setattr(retrieve, "PATHS", SimpleNamespace(artifacts=artifact_dir))
    corpus = retrieve.ScopedCorpus(
        cluster_id="cluster-1",
        company_id="company-1",
        embed_model=MODEL,
        rows=(),
        embed_dim=2,
        embedding_rows=2,
    )

    target.progress.write_text(json.dumps({"invalid": "sidecar"}))
    with pytest.raises(ValueError, match="invalid embedding artifact metadata"):
        retrieve._load_default_vectors(corpus)

    encode.Progress(target.progress, 2, 2, 2, MODEL).write()
    vectors = retrieve._load_default_vectors(corpus)
    assert np.array_equal(vectors, expected_vectors)

    target.progress.unlink()
    with pytest.raises(ValueError, match="missing embedding artifact metadata"):
        retrieve._load_default_vectors(corpus)
