"""Descriptor-bound private artifact path regressions."""

from __future__ import annotations

import stat
from contextlib import suppress

import pytest

from src import private_artifacts


@pytest.mark.parametrize("operation", ["write", "read"])
def test_private_artifact_rejects_configured_root_rename_before_io(
    tmp_path,
    monkeypatch,
    operation,
):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    original = b"existing private bytes\n"
    path.write_bytes(original)
    moved = tmp_path / "outside-moved-interim"
    original_verify = private_artifacts._verify_root
    calls = 0

    def swap_root(parent_fd, root_fd, root_name):
        nonlocal calls
        calls += 1
        if calls == 2:
            root.rename(moved)
            root.mkdir()
        return original_verify(parent_fd, root_fd, root_name)

    monkeypatch.setattr(private_artifacts, "_verify_root", swap_root)

    with pytest.raises(private_artifacts.PrivatePathError, match="interim changed"):
        if operation == "write":
            private_artifacts.atomic_write_bytes(path, root, b"new private bytes\n")
        else:
            private_artifacts.read_private_bytes(path, root)

    assert (moved / path.name).read_bytes() == original
    assert not (root / path.name).exists()
    assert list(moved.glob(f".{path.name}.*.tmp")) == []
    assert list(root.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.parametrize("previous", [None, b"exact old private bytes\n"])
def test_atomic_write_rolls_back_publication_when_root_moves_inside_replace(
    tmp_path,
    monkeypatch,
    previous,
):
    """A rename during publication cannot leave the newly written private bytes."""
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    if previous is not None:
        path.write_bytes(previous)
    moved = tmp_path / "outside-moved-interim"
    original_replace = private_artifacts.os.replace
    swapped = False

    def swap_then_replace(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            root.rename(moved)
            root.mkdir()
        return original_replace(*args, **kwargs)

    monkeypatch.setattr(private_artifacts.os, "replace", swap_then_replace)

    with pytest.raises(private_artifacts.PrivatePathError, match="interim changed"):
        private_artifacts.atomic_write_bytes(path, root, b"new private bytes\n")

    assert ((moved / path.name).read_bytes() if (moved / path.name).exists() else None) == previous
    assert not (root / path.name).exists()
    assert sorted(item.name for item in moved.iterdir()) == ([path.name] if previous else [])
    assert list(root.iterdir()) == []


def test_atomic_write_rolls_back_after_post_publication_fsync_failure(
    tmp_path,
    monkeypatch,
):
    """A durability error after replace restores the exact previous destination."""
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    previous = b"exact old private bytes\n"
    path.write_bytes(previous)
    original_replace = private_artifacts.os.replace
    original_fsync = private_artifacts.os.fsync
    published = False
    failed = False

    def observe_replace(*args, **kwargs):
        nonlocal published
        result = original_replace(*args, **kwargs)
        published = True
        return result

    def fail_first_post_publication_fsync(descriptor):
        nonlocal failed
        if published and not failed:
            failed = True
            raise OSError("post-publication fsync failed")
        return original_fsync(descriptor)

    monkeypatch.setattr(private_artifacts.os, "replace", observe_replace)
    monkeypatch.setattr(private_artifacts.os, "fsync", fail_first_post_publication_fsync)

    with pytest.raises(OSError, match="post-publication fsync failed"):
        private_artifacts.atomic_write_bytes(path, root, b"new private bytes\n")

    assert path.read_bytes() == previous
    assert sorted(item.name for item in root.iterdir()) == [path.name]


def test_atomic_write_fsyncs_each_backup_namespace_change_in_order(tmp_path, monkeypatch):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    path.write_bytes(b"old\n")
    operations: list[str] = []
    original_link = private_artifacts.os.link
    original_replace = private_artifacts.os.replace
    original_unlink = private_artifacts.os.unlink
    original_fsync = private_artifacts.os.fsync
    original_verify = private_artifacts._verify_root

    def observe_link(*args, **kwargs):
        operations.append("backup-link")
        return original_link(*args, **kwargs)

    def observe_replace(*args, **kwargs):
        operations.append("publish-replace")
        return original_replace(*args, **kwargs)

    def observe_unlink(name, *args, **kwargs):
        if str(name).endswith(".bak"):
            operations.append("backup-unlink")
        return original_unlink(name, *args, **kwargs)

    def observe_fsync(descriptor):
        if stat.S_ISDIR(private_artifacts.os.fstat(descriptor).st_mode):
            operations.append("directory-fsync")
        return original_fsync(descriptor)

    def observe_verify(*args, **kwargs):
        if operations[-2:] == ["backup-unlink", "directory-fsync"]:
            operations.append("final-verify")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(private_artifacts.os, "link", observe_link)
    monkeypatch.setattr(private_artifacts.os, "replace", observe_replace)
    monkeypatch.setattr(private_artifacts.os, "unlink", observe_unlink)
    monkeypatch.setattr(private_artifacts.os, "fsync", observe_fsync)
    monkeypatch.setattr(private_artifacts, "_verify_root", observe_verify)

    private_artifacts.atomic_write_bytes(path, root, b"new\n")

    assert operations == [
        "directory-fsync",
        "backup-link",
        "directory-fsync",
        "publish-replace",
        "directory-fsync",
        "backup-unlink",
        "directory-fsync",
        "final-verify",
    ]
    assert path.read_bytes() == b"new\n"


def test_atomic_write_fsyncs_prepublication_backup_cleanup(tmp_path, monkeypatch):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    path.write_bytes(b"old\n")
    operations: list[str] = []
    original_link = private_artifacts.os.link
    original_unlink = private_artifacts.os.unlink
    original_fsync = private_artifacts.os.fsync

    def observe_link(*args, **kwargs):
        operations.append("backup-link")
        return original_link(*args, **kwargs)

    def fail_replace(*_args, **_kwargs):
        operations.append("publish-replace")
        raise OSError("publication failed")

    def observe_unlink(name, *args, **kwargs):
        if str(name).endswith(".bak"):
            operations.append("backup-unlink")
        return original_unlink(name, *args, **kwargs)

    def observe_fsync(descriptor):
        if stat.S_ISDIR(private_artifacts.os.fstat(descriptor).st_mode):
            operations.append("directory-fsync")
        return original_fsync(descriptor)

    monkeypatch.setattr(private_artifacts.os, "link", observe_link)
    monkeypatch.setattr(private_artifacts.os, "replace", fail_replace)
    monkeypatch.setattr(private_artifacts.os, "unlink", observe_unlink)
    monkeypatch.setattr(private_artifacts.os, "fsync", observe_fsync)

    with pytest.raises(OSError, match="publication failed"):
        private_artifacts.atomic_write_bytes(path, root, b"new\n")

    assert operations == [
        "directory-fsync",
        "backup-link",
        "directory-fsync",
        "publish-replace",
        "backup-unlink",
        "directory-fsync",
    ]
    assert path.read_bytes() == b"old\n"
    assert sorted(item.name for item in root.iterdir()) == [path.name]


def test_atomic_write_reports_final_backup_cleanup_fsync_failure_consistently(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    path.write_bytes(b"old\n")
    original_unlink = private_artifacts.os.unlink
    original_fsync = private_artifacts.os.fsync
    backup_unlinked = False
    failed = False

    def observe_unlink(name, *args, **kwargs):
        nonlocal backup_unlinked
        result = original_unlink(name, *args, **kwargs)
        if str(name).endswith(".bak"):
            backup_unlinked = True
        return result

    def fail_cleanup_fsync(descriptor):
        nonlocal failed
        if (
            backup_unlinked
            and not failed
            and stat.S_ISDIR(private_artifacts.os.fstat(descriptor).st_mode)
        ):
            failed = True
            raise OSError("final cleanup fsync failed")
        return original_fsync(descriptor)

    monkeypatch.setattr(private_artifacts.os, "unlink", observe_unlink)
    monkeypatch.setattr(private_artifacts.os, "fsync", fail_cleanup_fsync)

    with pytest.raises(OSError, match="final cleanup fsync failed"):
        private_artifacts.atomic_write_bytes(path, root, b"new\n")

    assert path.read_bytes() == b"new\n"
    assert sorted(item.name for item in root.iterdir()) == [path.name]


def test_atomic_write_preserves_backup_collision_when_temp_cleanup_also_fails(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    path.write_bytes(b"old\n")
    nonce = "fixed-nonce"
    backup = root / f".{path.name}.{nonce}.bak"
    backup.write_bytes(b"preexisting collision\n")
    original_unlink = private_artifacts.os.unlink
    original_fsync = private_artifacts.os.fsync
    directory_fsyncs = 0

    def fail_temp_unlink(name, *args, **kwargs):
        if str(name).endswith(".tmp"):
            raise OSError("secret prose and /private/path must not leak")
        return original_unlink(name, *args, **kwargs)

    def observe_fsync(descriptor):
        nonlocal directory_fsyncs
        if stat.S_ISDIR(private_artifacts.os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
        return original_fsync(descriptor)

    monkeypatch.setattr(private_artifacts.secrets, "token_hex", lambda _size: nonce)
    monkeypatch.setattr(private_artifacts.os, "unlink", fail_temp_unlink)
    monkeypatch.setattr(private_artifacts.os, "fsync", observe_fsync)

    with pytest.raises(FileExistsError) as raised:
        private_artifacts.atomic_write_bytes(path, root, b"new secret prose\n")

    notes = getattr(raised.value, "__notes__", [])
    assert notes == ["private artifact publication recovery also failed: OSError"]
    assert "secret prose" not in notes[0]
    assert "/private/path" not in notes[0]
    assert directory_fsyncs == 2
    assert path.read_bytes() == b"old\n"
    assert backup.read_bytes() == b"preexisting collision\n"
    assert len(list(root.glob(f".{path.name}.{nonce}.tmp"))) == 1


def test_atomic_write_rolls_back_when_root_moves_during_backup_unlink(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    previous = b"exact old private bytes\n"
    path.write_bytes(previous)
    moved = tmp_path / "outside-moved-interim"
    original_unlink = private_artifacts.os.unlink
    original_fsync = private_artifacts.os.fsync
    swapped = False
    post_swap_directory_fsyncs = 0

    def unlink_then_swap_root(name, *args, **kwargs):
        nonlocal swapped
        result = original_unlink(name, *args, **kwargs)
        if str(name).endswith(".bak") and not swapped:
            swapped = True
            root.rename(moved)
            root.mkdir()
        return result

    def observe_fsync(descriptor):
        nonlocal post_swap_directory_fsyncs
        if swapped and stat.S_ISDIR(private_artifacts.os.fstat(descriptor).st_mode):
            post_swap_directory_fsyncs += 1
        return original_fsync(descriptor)

    monkeypatch.setattr(private_artifacts.os, "unlink", unlink_then_swap_root)
    monkeypatch.setattr(private_artifacts.os, "fsync", observe_fsync)

    with pytest.raises(private_artifacts.PrivatePathError, match="interim changed"):
        private_artifacts.atomic_write_bytes(path, root, b"new private bytes\n")

    assert list(root.iterdir()) == []
    assert (moved / path.name).read_bytes() == previous
    assert sorted(item.name for item in moved.iterdir()) == [path.name]
    assert post_swap_directory_fsyncs == 2


def test_atomic_write_final_verify_removes_new_file_after_no_old_target_root_swap(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    moved = tmp_path / "outside-moved-interim"
    original_replace = private_artifacts.os.replace
    original_fsync = private_artifacts.os.fsync
    original_verify = private_artifacts._verify_root
    published = False
    swapped = False
    recovery_directory_fsyncs = 0

    def observe_replace(*args, **kwargs):
        nonlocal published
        result = original_replace(*args, **kwargs)
        published = True
        return result

    def verify_then_swap_root(*args, **kwargs):
        nonlocal swapped
        result = original_verify(*args, **kwargs)
        if published and not swapped:
            swapped = True
            root.rename(moved)
            root.mkdir()
        return result

    def observe_fsync(descriptor):
        nonlocal recovery_directory_fsyncs
        if swapped and stat.S_ISDIR(private_artifacts.os.fstat(descriptor).st_mode):
            recovery_directory_fsyncs += 1
        return original_fsync(descriptor)

    monkeypatch.setattr(private_artifacts.os, "replace", observe_replace)
    monkeypatch.setattr(private_artifacts.os, "fsync", observe_fsync)
    monkeypatch.setattr(private_artifacts, "_verify_root", verify_then_swap_root)

    with pytest.raises(private_artifacts.PrivatePathError, match="interim changed"):
        private_artifacts.atomic_write_bytes(path, root, b"new private bytes\n")

    assert list(root.iterdir()) == []
    assert list(moved.iterdir()) == []
    assert recovery_directory_fsyncs == 1


def test_atomic_write_recovery_uses_original_inode_not_replaced_backup_name(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    previous = b"exact original private bytes\n"
    path.write_bytes(previous)
    moved = tmp_path / "outside-moved-interim"
    original_open = private_artifacts.os.open
    original_unlink = private_artifacts.os.unlink
    substituted = False
    swapped = False

    def substitute_backup_before_open(name, flags, *args, **kwargs):
        nonlocal substituted
        if str(name).endswith(".bak") and not substituted:
            substituted = True
            original_unlink(name, dir_fd=kwargs["dir_fd"])
            substitute_fd = original_open(
                name,
                private_artifacts.os.O_WRONLY
                | private_artifacts.os.O_CREAT
                | private_artifacts.os.O_EXCL,
                0o600,
                dir_fd=kwargs["dir_fd"],
            )
            private_artifacts.os.write(substitute_fd, b"substitute private bytes\n")
            private_artifacts.os.close(substitute_fd)
        return original_open(name, flags, *args, **kwargs)

    def unlink_then_swap_root(name, *args, **kwargs):
        nonlocal swapped
        result = original_unlink(name, *args, **kwargs)
        if str(name).endswith(".bak") and not swapped:
            swapped = True
            root.rename(moved)
            root.mkdir()
        return result

    monkeypatch.setattr(private_artifacts.os, "open", substitute_backup_before_open)
    monkeypatch.setattr(private_artifacts.os, "unlink", unlink_then_swap_root)

    with pytest.raises(private_artifacts.PrivatePathError, match="interim changed"):
        private_artifacts.atomic_write_bytes(path, root, b"new private bytes\n")

    assert list(root.iterdir()) == []
    assert (moved / path.name).read_bytes() == previous
    assert sorted(item.name for item in moved.iterdir()) == [path.name]


def test_atomic_write_rejects_destination_swap_between_pin_and_backup_link(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    previous = b"exact original private bytes\n"
    path.write_bytes(previous)
    original_open = private_artifacts.os.open
    original_link = private_artifacts.os.link
    original_pinned = False

    def observe_original_open(name, flags, *args, **kwargs):
        nonlocal original_pinned
        result = original_open(name, flags, *args, **kwargs)
        if (
            name == path.name
            and flags & private_artifacts.os.O_ACCMODE == private_artifacts.os.O_RDONLY
        ):
            original_pinned = True
        return result

    def swap_then_link(*args, **kwargs):
        if original_pinned:
            substitute = root / "substitute"
            substitute.write_bytes(b"substitute private bytes\n")
            private_artifacts.os.replace(substitute, path)
        return original_link(*args, **kwargs)

    monkeypatch.setattr(private_artifacts.os, "open", observe_original_open)
    monkeypatch.setattr(private_artifacts.os, "link", swap_then_link)

    with pytest.raises(private_artifacts.PrivatePathError, match="changed during publication"):
        private_artifacts.atomic_write_bytes(path, root, b"new private bytes\n")

    assert path.read_bytes() == previous
    assert sorted(item.name for item in root.iterdir()) == [path.name]


def test_atomic_write_rejects_in_place_mutation_before_backup_link_and_restores_snapshot(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    previous = b"exact original private bytes\n"
    path.write_bytes(previous)
    original_link = private_artifacts.os.link
    mutated = False

    def mutate_then_link(*args, **kwargs):
        nonlocal mutated
        if not mutated:
            mutated = True
            with path.open("r+b") as source:
                source.truncate(0)
                source.write(b"in-place mutated private bytes\n")
                source.flush()
                private_artifacts.os.fsync(source.fileno())
        return original_link(*args, **kwargs)

    monkeypatch.setattr(private_artifacts.os, "link", mutate_then_link)

    with pytest.raises(private_artifacts.PrivatePathError, match="changed during publication"):
        private_artifacts.atomic_write_bytes(path, root, b"new private bytes\n")

    assert path.read_bytes() == previous
    assert sorted(item.name for item in root.iterdir()) == [path.name]


def test_atomic_write_root_failure_restores_independent_snapshot_after_linked_inode_mutation(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    previous = b"exact original private bytes\n"
    path.write_bytes(previous)
    moved = tmp_path / "outside-moved-interim"
    original_replace = private_artifacts.os.replace
    swapped = False

    def mutate_move_then_publish(source, destination, *args, **kwargs):
        nonlocal swapped
        if not swapped and str(source).endswith(".tmp") and destination == path.name:
            swapped = True
            with path.open("r+b") as old_destination:
                old_destination.truncate(0)
                old_destination.write(b"in-place mutated private bytes\n")
                old_destination.flush()
                private_artifacts.os.fsync(old_destination.fileno())
            root.rename(moved)
            root.mkdir()
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(private_artifacts.os, "replace", mutate_move_then_publish)

    with pytest.raises(private_artifacts.PrivatePathError, match="interim changed"):
        private_artifacts.atomic_write_bytes(path, root, b"new private bytes\n")

    assert list(root.iterdir()) == []
    assert (moved / path.name).read_bytes() == previous
    assert sorted(item.name for item in moved.iterdir()) == [path.name]


def test_atomic_write_rejects_destination_appearance_after_absence_was_pinned(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    original_open = private_artifacts.os.open
    original_link = private_artifacts.os.link
    absence_pinned = False

    def observe_original_absence(name, flags, *args, **kwargs):
        nonlocal absence_pinned
        try:
            return original_open(name, flags, *args, **kwargs)
        except FileNotFoundError:
            if name == path.name:
                absence_pinned = True
            raise

    def create_then_link(*args, **kwargs):
        if absence_pinned:
            path.write_bytes(b"concurrently created private bytes\n")
        return original_link(*args, **kwargs)

    monkeypatch.setattr(private_artifacts.os, "open", observe_original_absence)
    monkeypatch.setattr(private_artifacts.os, "link", create_then_link)

    with pytest.raises(private_artifacts.PrivatePathError, match="changed during publication"):
        private_artifacts.atomic_write_bytes(path, root, b"new private bytes\n")

    assert path.read_bytes() == b"concurrently created private bytes\n"
    assert sorted(item.name for item in root.iterdir()) == [path.name]


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_atomic_write_rejects_nonregular_old_destination(tmp_path, kind):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    if kind == "symlink":
        outside = tmp_path / "outside.txt"
        outside.write_bytes(b"outside private bytes\n")
        path.symlink_to(outside)
    else:
        path.mkdir()

    with pytest.raises(private_artifacts.PrivatePathError, match="symlink|regular file"):
        private_artifacts.atomic_write_bytes(path, root, b"new private bytes\n")

    assert not list(root.glob(f".{path.name}.*"))


def test_atomic_write_snapshot_close_failure_does_not_mask_root_change(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    previous = b"exact original private bytes\n"
    path.write_bytes(previous)
    moved = tmp_path / "outside-moved-interim"
    original_open = private_artifacts.os.open
    original_close = private_artifacts.os.close
    original_unlink = private_artifacts.os.unlink
    descriptors: dict[str, int] = {}
    swapped = False
    close_failed = False

    def observe_open(name, flags, *args, **kwargs):
        descriptor = original_open(name, flags, *args, **kwargs)
        if name == root.name:
            descriptors["root"] = descriptor
        elif (
            name == path.name
            and flags & private_artifacts.os.O_ACCMODE == private_artifacts.os.O_RDONLY
        ):
            descriptors["snapshot"] = descriptor
        return descriptor

    def unlink_then_swap_root(name, *args, **kwargs):
        nonlocal swapped
        result = original_unlink(name, *args, **kwargs)
        if str(name).endswith(".bak") and not swapped:
            swapped = True
            root.rename(moved)
            root.mkdir()
        return result

    def fail_snapshot_close(descriptor):
        nonlocal close_failed
        if descriptor == descriptors.get("snapshot") and not close_failed:
            close_failed = True
            raise OSError("secret prose and /private/path must not leak")
        return original_close(descriptor)

    monkeypatch.setattr(private_artifacts.os, "open", observe_open)
    monkeypatch.setattr(private_artifacts.os, "unlink", unlink_then_swap_root)
    monkeypatch.setattr(private_artifacts.os, "close", fail_snapshot_close)

    try:
        with pytest.raises(private_artifacts.PrivatePathError, match="interim changed") as raised:
            private_artifacts.atomic_write_bytes(path, root, b"new private bytes\n")

        notes = getattr(raised.value, "__notes__", [])
        assert notes == ["private artifact descriptor cleanup also failed: OSError"]
        assert "secret prose" not in notes[0]
        assert "/private/path" not in notes[0]
        assert list(root.iterdir()) == []
        assert (moved / path.name).read_bytes() == previous
        assert sorted(item.name for item in moved.iterdir()) == [path.name]
        with pytest.raises(OSError):
            private_artifacts.os.fstat(descriptors["root"])
    finally:
        with suppress(OSError):
            original_close(descriptors["snapshot"])


def test_atomic_write_surfaces_snapshot_close_failure_after_verified_success(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "interim"
    root.mkdir()
    path = root / "review.csv"
    path.write_bytes(b"old private bytes\n")
    original_open = private_artifacts.os.open
    original_close = private_artifacts.os.close
    descriptors: dict[str, int] = {}
    close_failed = False

    def observe_open(name, flags, *args, **kwargs):
        descriptor = original_open(name, flags, *args, **kwargs)
        if name == root.name:
            descriptors["root"] = descriptor
        elif (
            name == path.name
            and flags & private_artifacts.os.O_ACCMODE == private_artifacts.os.O_RDONLY
        ):
            descriptors["snapshot"] = descriptor
        return descriptor

    def fail_snapshot_close(descriptor):
        nonlocal close_failed
        if descriptor == descriptors.get("snapshot") and not close_failed:
            close_failed = True
            raise OSError("snapshot close failed")
        return original_close(descriptor)

    monkeypatch.setattr(private_artifacts.os, "open", observe_open)
    monkeypatch.setattr(private_artifacts.os, "close", fail_snapshot_close)

    try:
        with pytest.raises(OSError, match="snapshot close failed"):
            private_artifacts.atomic_write_bytes(path, root, b"verified new private bytes\n")

        assert path.read_bytes() == b"verified new private bytes\n"
        assert sorted(item.name for item in root.iterdir()) == [path.name]
        with pytest.raises(OSError):
            private_artifacts.os.fstat(descriptors["root"])
    finally:
        with suppress(OSError):
            original_close(descriptors["snapshot"])
