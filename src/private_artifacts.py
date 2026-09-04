"""Race-safe filesystem boundaries for private reviewer artifacts.

Private files must be direct children of the configured root. All opens and
replacements are descriptor-relative; symlink redirects are rejected, and the
pinned root is revalidated from its parent before and after publication. A
post-publication failure rolls back the new name on the pinned root and restores
an exact hard-linked prior destination when one existed.
"""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path


class PrivatePathError(ValueError):
    """A private artifact escaped or changed its configured directory scope."""


_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def read_stable_bytes(path: Path) -> bytes:
    """Read one path once and reject replacement or mutation during the read."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    parent = path.parent.resolve(strict=True)
    parent_fd = os.open(parent, _DIRECTORY_FLAGS)
    file_fd: int | None = None
    try:
        file_fd = os.open(path.name, os.O_RDONLY | _FILE_NOFOLLOW, dir_fd=parent_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise PrivatePathError("artifact path must name a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if path.parent.resolve(strict=True) != parent:
            raise PrivatePathError("artifact parent changed during read")
        identities = [
            (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
            for value in (before, after, current)
        ]
        if len(set(identities)) != 1:
            raise PrivatePathError("artifact changed during read")
        return b"".join(chunks)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def _parts(path: Path, root: Path) -> tuple[Path, str]:
    if not isinstance(path, Path) or not isinstance(root, Path):
        raise TypeError("private artifact path and root must be Paths")
    requested = Path(os.path.abspath(path))
    configured = Path(os.path.abspath(root))
    try:
        relative = requested.relative_to(configured)
    except ValueError:
        raise PrivatePathError("private artifact must stay under configured data/interim") from None
    parts = relative.parts
    if len(parts) != 1 or parts[0] in {"", ".", ".."}:
        raise PrivatePathError(
            "private artifacts must be direct file children of configured data/interim; "
            "nested paths are not allowed"
        )
    try:
        canonical_root = configured.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PrivatePathError("configured data/interim does not exist") from exc
    if canonical_root != configured:
        raise PrivatePathError("configured data/interim must not contain symlink components")
    return canonical_root, parts[0]


def _open_root(path: Path, root: Path) -> tuple[Path, str, int, int]:
    canonical_root, name = _parts(path, root)
    parent_fd = os.open(canonical_root.parent, _DIRECTORY_FLAGS)
    root_fd: int | None = None
    try:
        root_fd = os.open(canonical_root.name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        _verify_root(parent_fd, root_fd, canonical_root.name)
        return canonical_root, name, parent_fd, root_fd
    except BaseException:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)
        raise


def _verify_root(parent_fd: int, root_fd: int, root_name: str) -> None:
    try:
        current = os.stat(root_name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise PrivatePathError("configured data/interim changed during access") from None
    pinned = os.fstat(root_fd)
    if not stat.S_ISDIR(current.st_mode) or (pinned.st_dev, pinned.st_ino) != (
        current.st_dev,
        current.st_ino,
    ):
        raise PrivatePathError("configured data/interim changed during access")


def _reject_final_symlink(root_fd: int, name: str, *, must_exist: bool) -> None:
    try:
        target = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        if must_exist:
            raise
        return
    if stat.S_ISLNK(target.st_mode):
        raise PrivatePathError("private artifact path must not be a symlink")
    if not stat.S_ISREG(target.st_mode):
        raise PrivatePathError("private artifact path must name a regular file")


def _close_root(parent_fd: int, root_fd: int) -> None:
    try:
        os.close(root_fd)
    finally:
        os.close(parent_fd)


def canonical_private_path(
    path: Path,
    root: Path,
    *,
    must_exist: bool = False,
    create_parents: bool = False,
) -> Path:
    """Return a canonical direct-child path below a pinned configured root."""
    del create_parents
    canonical_root, name, parent_fd, root_fd = _open_root(path, root)
    try:
        _verify_root(parent_fd, root_fd, canonical_root.name)
        _reject_final_symlink(root_fd, name, must_exist=must_exist)
        return canonical_root / name
    finally:
        _close_root(parent_fd, root_fd)


def read_private_bytes(path: Path, root: Path) -> tuple[Path, bytes]:
    """Read one stable regular-file identity from a private directory."""
    canonical_root, name, parent_fd, root_fd = _open_root(path, root)
    file_fd: int | None = None
    try:
        _verify_root(parent_fd, root_fd, canonical_root.name)
        file_fd = os.open(name, os.O_RDONLY | _FILE_NOFOLLOW, dir_fd=root_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise PrivatePathError("private artifact path must name a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        _verify_root(parent_fd, root_fd, canonical_root.name)
        current = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        identity_current = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if identity_before != identity_after or identity_after != identity_current:
            raise PrivatePathError("private artifact changed during read")
        return canonical_root / name, b"".join(chunks)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        _close_root(parent_fd, root_fd)


def _replace_from_snapshot(root_fd: int, snapshot_fd: int, temporary_name: str, name: str) -> None:
    recovery_fd: int | None = None
    try:
        os.lseek(snapshot_fd, 0, os.SEEK_SET)
        recovery_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        while chunk := os.read(snapshot_fd, 1024 * 1024):
            view = memoryview(chunk)
            while view:
                written = os.write(recovery_fd, view)
                view = view[written:]
        os.fsync(recovery_fd)
        os.close(recovery_fd)
        recovery_fd = None
        os.replace(
            temporary_name,
            name,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
    finally:
        if recovery_fd is not None:
            os.close(recovery_fd)


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _same_file_bytes(first_fd: int, second_fd: int) -> bool:
    os.lseek(first_fd, 0, os.SEEK_SET)
    os.lseek(second_fd, 0, os.SEEK_SET)
    while True:
        first = os.read(first_fd, 1024 * 1024)
        second = os.read(second_fd, 1024 * 1024)
        if first != second:
            return False
        if not first:
            return True


def atomic_write_bytes(path: Path, root: Path, data: bytes) -> Path:
    """Durably replace one scoped file without following path symlinks."""
    if type(data) is not bytes:
        raise TypeError("atomic artifact data must be bytes")
    canonical_root, name, parent_fd, root_fd = _open_root(path, root)
    nonce = secrets.token_hex(8)
    temporary_name = f".{name}.{nonce}.tmp"
    backup_name = f".{name}.{nonce}.bak"
    snapshot_name = f".{name}.{nonce}.snapshot"
    temporary_fd: int | None = None
    old_source_fd: int | None = None
    old_snapshot_fd: int | None = None
    old_source_identity: tuple[int, ...] | None = None
    snapshot_named = False
    snapshot_ready = False
    backup_created = False
    published = False
    publication_verified = False
    destination_changed = False
    operation_error: BaseException | None = None
    try:
        _verify_root(parent_fd, root_fd, canonical_root.name)
        _reject_final_symlink(root_fd, name, must_exist=False)
        try:
            old_source_fd = os.open(name, os.O_RDONLY | _FILE_NOFOLLOW, dir_fd=root_fd)
        except FileNotFoundError:
            old_source_fd = None
        except OSError:
            raise PrivatePathError(
                "private artifact destination changed during publication"
            ) from None
        if old_source_fd is not None:
            source_before = os.fstat(old_source_fd)
            try:
                named_before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except OSError:
                raise PrivatePathError(
                    "private artifact destination changed during publication"
                ) from None
            if not stat.S_ISREG(source_before.st_mode):
                raise PrivatePathError("private artifact path must name a regular file")
            old_source_identity = _file_identity(source_before)
            if _file_identity(named_before) != old_source_identity:
                raise PrivatePathError("private artifact destination changed during publication")
            old_snapshot_fd = os.open(
                snapshot_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW,
                0o600,
                dir_fd=root_fd,
            )
            snapshot_named = True
            os.fchmod(old_snapshot_fd, 0o600)
            while chunk := os.read(old_source_fd, 1024 * 1024):
                view = memoryview(chunk)
                while view:
                    written = os.write(old_snapshot_fd, view)
                    view = view[written:]
            os.fsync(old_snapshot_fd)
            snapshot_stat = os.fstat(old_snapshot_fd)
            try:
                source_after = os.fstat(old_source_fd)
                named_after = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except OSError:
                raise PrivatePathError(
                    "private artifact destination changed during publication"
                ) from None
            if (
                not stat.S_ISREG(snapshot_stat.st_mode)
                or stat.S_IMODE(snapshot_stat.st_mode) != 0o600
                or (snapshot_stat.st_dev, snapshot_stat.st_ino)
                == (source_before.st_dev, source_before.st_ino)
                or _file_identity(source_after) != old_source_identity
                or _file_identity(named_after) != old_source_identity
            ):
                raise PrivatePathError("private artifact destination changed during publication")
            snapshot_ready = True
            os.unlink(snapshot_name, dir_fd=root_fd)
            snapshot_named = False
            os.fsync(root_fd)
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        view = memoryview(data)
        while view:
            written = os.write(temporary_fd, view)
            view = view[written:]
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        _verify_root(parent_fd, root_fd, canonical_root.name)
        if old_source_fd is None:
            try:
                os.link(
                    name,
                    backup_name,
                    src_dir_fd=root_fd,
                    dst_dir_fd=root_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                backup_created = True
                os.fsync(root_fd)
                raise PrivatePathError("private artifact destination changed during publication")
        else:
            try:
                source_before_link = os.fstat(old_source_fd)
                named_before_link = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except OSError:
                destination_changed = True
                raise PrivatePathError(
                    "private artifact destination changed during publication"
                ) from None
            if (
                _file_identity(source_before_link) != old_source_identity
                or _file_identity(named_before_link) != old_source_identity
            ):
                destination_changed = True
                raise PrivatePathError("private artifact destination changed during publication")
            try:
                os.link(
                    name,
                    backup_name,
                    src_dir_fd=root_fd,
                    dst_dir_fd=root_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                destination_changed = True
                raise PrivatePathError(
                    "private artifact destination changed during publication"
                ) from None
            backup_created = True
            os.fsync(root_fd)
            try:
                backup = os.stat(backup_name, dir_fd=root_fd, follow_symlinks=False)
            except OSError:
                destination_changed = True
                raise PrivatePathError(
                    "private artifact destination changed during publication"
                ) from None
            source_after_link = os.fstat(old_source_fd)
            named_after_link = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(backup.st_mode)
                or _file_identity(backup) != _file_identity(source_after_link)
                or _file_identity(named_after_link) != _file_identity(source_after_link)
                or old_snapshot_fd is None
                or not _same_file_bytes(old_snapshot_fd, old_source_fd)
            ):
                destination_changed = True
                raise PrivatePathError("private artifact destination changed during publication")
        os.replace(
            temporary_name,
            name,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
        published = True
        os.fsync(root_fd)
        _verify_root(parent_fd, root_fd, canonical_root.name)
        publication_verified = True
        if backup_created:
            os.unlink(backup_name, dir_fd=root_fd)
            backup_created = False
            os.fsync(root_fd)
        try:
            _verify_root(parent_fd, root_fd, canonical_root.name)
        except BaseException:
            publication_verified = False
            raise
        return canonical_root / name
    except BaseException as write_error:
        operation_error = write_error
        recovery_errors: list[str] = []
        if temporary_fd is not None:
            try:
                os.close(temporary_fd)
            except Exception as recovery_error:
                recovery_errors.append(type(recovery_error).__name__)
        try:
            os.unlink(temporary_name, dir_fd=root_fd)
        except FileNotFoundError:
            pass
        except Exception as recovery_error:
            recovery_errors.append(type(recovery_error).__name__)
        if destination_changed or (published and not publication_verified):
            try:
                if snapshot_ready and old_snapshot_fd is not None:
                    _replace_from_snapshot(root_fd, old_snapshot_fd, temporary_name, name)
                elif old_source_fd is None and published:
                    os.unlink(name, dir_fd=root_fd)
            except Exception as recovery_error:
                recovery_errors.append(type(recovery_error).__name__)
        if backup_created:
            try:
                os.unlink(backup_name, dir_fd=root_fd)
                backup_created = False
            except Exception as recovery_error:
                recovery_errors.append(type(recovery_error).__name__)
        if snapshot_named:
            try:
                os.unlink(snapshot_name, dir_fd=root_fd)
                snapshot_named = False
            except FileNotFoundError:
                snapshot_named = False
            except Exception as recovery_error:
                recovery_errors.append(type(recovery_error).__name__)
        try:
            os.fsync(root_fd)
        except Exception as recovery_error:
            recovery_errors.append(type(recovery_error).__name__)
        if recovery_errors:
            write_error.add_note(
                "private artifact publication recovery also failed: " + ", ".join(recovery_errors)
            )
        raise
    finally:
        cleanup_error: Exception | None = None
        try:
            for descriptor in (old_source_fd, old_snapshot_fd):
                if descriptor is None:
                    continue
                try:
                    os.close(descriptor)
                except Exception as error:
                    if operation_error is None:
                        if cleanup_error is None:
                            cleanup_error = error
                        else:
                            cleanup_error.add_note(
                                "private artifact descriptor cleanup also failed: "
                                + type(error).__name__
                            )
                    else:
                        operation_error.add_note(
                            "private artifact descriptor cleanup also failed: "
                            + type(error).__name__
                        )
        finally:
            _close_root(parent_fd, root_fd)
        if cleanup_error is not None:
            raise cleanup_error


def unlink_private_file(path: Path, root: Path, *, missing_ok: bool = False) -> None:
    """Durably unlink one direct private regular file without following symlinks."""
    canonical_root, name, parent_fd, root_fd = _open_root(path, root)
    try:
        _verify_root(parent_fd, root_fd, canonical_root.name)
        try:
            _reject_final_symlink(root_fd, name, must_exist=True)
            os.unlink(name, dir_fd=root_fd)
        except FileNotFoundError:
            if not missing_ok:
                raise
            return
        os.fsync(root_fd)
        _verify_root(parent_fd, root_fd, canonical_root.name)
    finally:
        _close_root(parent_fd, root_fd)
