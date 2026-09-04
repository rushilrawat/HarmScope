"""Validate legacy embedding artifacts before a namespace migration."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

from src.embed.encode import embedding_artifact_paths


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


@dataclass(frozen=True)
class MigrationReport:
    model: str
    n_rows: int
    dim: int
    state: str
    legacy: ArtifactSet
    target: ArtifactSet
    retired: ArtifactSet

    def render(self) -> str:
        """Render only migration metadata and safe artifact basenames."""
        return json.dumps(
            {
                "model": self.model,
                "n_rows": self.n_rows,
                "dim": self.dim,
                "state": self.state,
                "legacy": [path.name for path in self.legacy.ordered()],
                "target": [path.name for path in self.target.ordered()],
                "retired": [path.name for path in self.retired.ordered()],
            },
            sort_keys=True,
        )


@dataclass(frozen=True)
class _BoundArtifacts:
    artifacts: ArtifactSet
    fds: tuple[int, int, int]
    identities: tuple[FileIdentity, FileIdentity, FileIdentity]
    digests: tuple[str, str, str]


def legacy_artifact_paths(artifact_dir: Path, model_name: str) -> ArtifactSet:
    """Return the historical model-tail filenames for one full model identity."""
    if not isinstance(model_name, str):
        raise ArtifactMigrationError("model identity must be a string")
    tail = model_name.rsplit("/", 1)[-1]
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tail) is None:
        raise ArtifactMigrationError("model tail is not safe for legacy artifact names")
    artifact_dir = Path(artifact_dir)
    return ArtifactSet(
        memmap=artifact_dir / f"embeddings.{tail}.npy",
        index=artifact_dir / f"faiss.{tail}.index",
        progress=artifact_dir / f"embeddings.{tail}.progress.json",
    )


def target_artifact_paths(artifact_dir: Path, model_name: str) -> ArtifactSet:
    """Return SHA-addressed paths derived solely from the existing helper."""
    paths = embedding_artifact_paths(Path(artifact_dir), model_name)
    return ArtifactSet(
        memmap=paths.memmap,
        index=paths.index,
        progress=paths.memmap.with_suffix(".progress.json"),
    )


def retired_artifact_paths(artifact_dir: Path, model_name: str) -> ArtifactSet:
    """Return deterministic retained names for the validated legacy entries."""
    legacy = legacy_artifact_paths(artifact_dir, model_name)
    retired = tuple(
        path.with_name(f".{path.name}.harmscope-migration-retired") for path in legacy.ordered()
    )
    return ArtifactSet(memmap=retired[0], index=retired[1], progress=retired[2])


def _identity(path: Path, artifact_dir: Path) -> FileIdentity:
    if path.parent != artifact_dir or path.name != Path(path.name).name:
        raise ArtifactMigrationError("artifact must be a direct child of the artifact directory")
    try:
        metadata = path.lstat()
    except OSError:
        raise ArtifactMigrationError("artifact is unavailable") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactMigrationError("artifact must be a regular file")
    return FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )


def _identities(artifacts: ArtifactSet, artifact_dir: Path):
    return tuple(_identity(path, artifact_dir) for path in artifacts.ordered())


def _parse_progress(raw: bytes, model_name: str) -> tuple[int, int]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ArtifactMigrationError("invalid embedding progress sidecar") from None
    expected_keys = {"n_done", "n_total", "dim", "model"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ArtifactMigrationError("progress sidecar has an invalid schema")
    if type(payload["model"]) is not str or any(
        type(payload[field]) is not int for field in ("n_done", "n_total", "dim")
    ):
        raise ArtifactMigrationError("progress sidecar has invalid field types")
    if payload["model"] != model_name:
        raise ArtifactMigrationError("progress sidecar model does not match requested model")
    if payload["n_total"] <= 0 or payload["dim"] <= 0:
        raise ArtifactMigrationError("progress sidecar must describe positive dimensions")
    if payload["n_done"] != payload["n_total"]:
        raise ArtifactMigrationError("progress sidecar does not describe a complete encode")
    return payload["n_total"], payload["dim"]


def _read_progress(path: Path, model_name: str) -> tuple[int, int]:
    try:
        raw = path.read_text(encoding="utf-8").encode("utf-8")
    except (OSError, UnicodeDecodeError):
        raise ArtifactMigrationError("invalid embedding progress sidecar") from None
    return _parse_progress(raw, model_name)


def _load_vectors(path: Path, n_rows: int, dim: int) -> np.ndarray:
    try:
        vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        raise ArtifactMigrationError("invalid embedding vector artifact") from None
    if vectors.ndim != 2 or vectors.dtype != np.dtype(np.float32):
        raise ArtifactMigrationError("embedding vectors must be a two-dimensional float32 array")
    if vectors.shape != (n_rows, dim):
        raise ArtifactMigrationError("embedding vector shape does not match progress sidecar")
    for start in range(0, n_rows, 16_384):
        chunk = vectors[start : start + 16_384]
        if not np.isfinite(chunk).all():
            raise ArtifactMigrationError("embedding vectors must contain only finite values")
        norms = np.linalg.norm(chunk, axis=1)
        if not np.isclose(norms, 1.0, atol=1e-3, rtol=0.0).all():
            raise ArtifactMigrationError("embedding vectors must be unit normalized")
    return vectors


def _validate_index(path: Path, vectors: np.ndarray, n_rows: int, dim: int) -> None:
    import faiss

    try:
        index = faiss.read_index(str(path))
    except (OSError, RuntimeError):
        raise ArtifactMigrationError("invalid FAISS index") from None
    if type(index) is not faiss.IndexFlatIP or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ArtifactMigrationError("FAISS index must be an inner-product IndexFlatIP")
    if index.d != dim or index.ntotal != n_rows:
        raise ArtifactMigrationError("FAISS index dimensions do not match embedding vectors")
    for row_idx in np.linspace(0, n_rows - 1, num=17, dtype=np.int64):
        try:
            reconstructed = index.reconstruct(int(row_idx))
        except RuntimeError:
            raise ArtifactMigrationError("FAISS index cannot reconstruct vector rows") from None
        if not np.allclose(reconstructed, vectors[row_idx], atol=1e-6, rtol=0.0):
            raise ArtifactMigrationError("FAISS index content does not match embedding vectors")


def _validate_mapping(
    con: duckdb.DuckDBPyConnection, model_name: str, n_rows: int, dim: int
) -> None:
    try:
        n_dimensions, min_row, max_row, distinct_rows, mapping_rows = con.execute(
            "SELECT count(DISTINCT dim), min(row_idx), max(row_idx), "
            "count(DISTINCT row_idx), count(*) FROM embedding_map WHERE model = ?",
            [model_name],
        ).fetchone()
        actual_dim = con.execute(
            "SELECT min(dim) FROM embedding_map WHERE model = ?", [model_name]
        ).fetchone()[0]
        narrative_rows = con.execute("SELECT count(*) FROM narratives").fetchone()[0]
        extra_mappings = con.execute(
            "SELECT count(*) FROM embedding_map e "
            "LEFT JOIN narratives n ON n.complaint_id = e.complaint_id "
            "WHERE e.model = ? AND n.complaint_id IS NULL",
            [model_name],
        ).fetchone()[0]
        missing, mismatched, expected_rows = con.execute(
            """
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
            """,
            [dim, model_name],
        ).fetchone()
    except duckdb.Error:
        raise ArtifactMigrationError("embedding mapping cannot be read") from None
    if (n_dimensions, min_row, max_row, distinct_rows) != (1, 0, n_rows - 1, n_rows):
        raise ArtifactMigrationError("embedding mapping does not cover contiguous vector rows")
    if actual_dim != dim:
        raise ArtifactMigrationError("embedding mapping dimension does not match vectors")
    if mapping_rows != narrative_rows:
        raise ArtifactMigrationError("embedding mapping does not cover each narrative exactly once")
    if extra_mappings:
        raise ArtifactMigrationError("embedding mapping contains complaints without narratives")
    if (missing, mismatched, expected_rows) != (0, 0, n_rows):
        raise ArtifactMigrationError("embedding mapping does not match stable text-hash rows")


def validate_artifacts(
    con: duckdb.DuckDBPyConnection, artifacts: ArtifactSet, model_name: str
) -> ValidatedArtifacts:
    """Validate one completed artifact set and its read-only database binding."""
    artifact_dir = artifacts.memmap.parent
    before = _identities(artifacts, artifact_dir)
    n_rows, dim = _read_progress(artifacts.progress, model_name)
    vectors = _load_vectors(artifacts.memmap, n_rows, dim)
    _validate_index(artifacts.index, vectors, n_rows, dim)
    _validate_mapping(con, model_name, n_rows, dim)
    after = _identities(artifacts, artifact_dir)
    if after != before:
        raise ArtifactMigrationError("artifact identity changed during validation")
    return ValidatedArtifacts(model_name, n_rows, dim, artifacts, before)


def _open_artifact_root(artifact_dir: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        root_fd = os.open(artifact_dir, flags)
    except OSError:
        raise ArtifactMigrationError("artifact directory cannot be safely opened") from None
    return root_fd, os.fstat(root_fd)


def _revalidate_root(artifact_dir: Path, root_fd: int, root_identity: os.stat_result) -> None:
    try:
        current_fd = os.fstat(root_fd)
        configured = os.stat(artifact_dir, follow_symlinks=False)
    except OSError:
        raise ArtifactMigrationError("artifact directory identity changed") from None
    expected = (root_identity.st_dev, root_identity.st_ino)
    if (
        not stat.S_ISDIR(configured.st_mode)
        or (current_fd.st_dev, current_fd.st_ino) != expected
        or (configured.st_dev, configured.st_ino) != expected
    ):
        raise ArtifactMigrationError("artifact directory identity changed")


def _fd_identity(root_fd: int, name: str) -> FileIdentity:
    try:
        metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except OSError:
        raise ArtifactMigrationError("artifact is unavailable") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactMigrationError("artifact must be a regular file")
    return FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )


def _fd_optional_identity(root_fd: int, name: str) -> FileIdentity | None:
    try:
        metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return None
        raise ArtifactMigrationError("artifact cannot be inspected") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactMigrationError("artifact must be a regular file")
    return FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )


def _fstat_identity(fd: int) -> FileIdentity:
    try:
        metadata = os.fstat(fd)
    except OSError:
        raise ArtifactMigrationError("artifact descriptor cannot be inspected") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactMigrationError("artifact descriptor must reference a regular file")
    return FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )


def _digest_fd(fd: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        try:
            chunk = os.pread(fd, 1024 * 1024, offset)
        except OSError:
            raise ArtifactMigrationError("artifact content cannot be read") from None
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def _read_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while True:
        try:
            chunk = os.pread(fd, 64 * 1024, offset)
        except OSError:
            raise ArtifactMigrationError("artifact content cannot be read") from None
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        offset += len(chunk)


def _fd_path(fd: int) -> Path:
    proc_path = Path("/proc/self/fd") / str(fd)
    if proc_path.exists():
        return proc_path
    return Path("/dev/fd") / str(fd)


def _open_bound_artifacts(
    root_fd: int,
    artifacts: ArtifactSet,
    expected: tuple[FileIdentity | None, FileIdentity | None, FileIdentity | None],
) -> _BoundArtifacts:
    opened: list[int] = []
    try:
        for path, expected_identity in zip(artifacts.ordered(), expected, strict=True):
            if expected_identity is None:
                raise ArtifactMigrationError("complete artifact set is required")
            try:
                fd = os.open(
                    path.name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=root_fd,
                )
            except OSError:
                raise ArtifactMigrationError("artifact cannot be safely opened") from None
            opened.append(fd)
            if _fstat_identity(fd) != expected_identity:
                raise ArtifactMigrationError("artifact identity changed while opening")
        fds = (opened[0], opened[1], opened[2])
        identities = tuple(_fstat_identity(fd) for fd in fds)
        digests = tuple(_digest_fd(fd) for fd in fds)
        return _BoundArtifacts(artifacts, fds, identities, digests)
    except BaseException:
        for fd in reversed(opened):
            with suppress(OSError):
                os.close(fd)
        raise


def _verify_bound_content(bound: _BoundArtifacts) -> None:
    if tuple(_fstat_identity(fd) for fd in bound.fds) != bound.identities:
        raise ArtifactMigrationError("artifact identity changed while bound")
    if tuple(_digest_fd(fd) for fd in bound.fds) != bound.digests:
        raise ArtifactMigrationError("artifact content changed while bound")


def _validate_bound_artifacts(
    con: duckdb.DuckDBPyConnection,
    bound: _BoundArtifacts,
    model_name: str,
) -> ValidatedArtifacts:
    _verify_bound_content(bound)
    n_rows, dim = _parse_progress(_read_fd(bound.fds[2]), model_name)
    try:
        with os.fdopen(os.dup(bound.fds[0]), "rb") as stream:
            stream.seek(0)
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
            else:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
            offset = stream.tell()
        vectors = np.memmap(
            _fd_path(bound.fds[0]),
            dtype=dtype,
            mode="r",
            offset=offset,
            shape=shape,
            order="F" if fortran_order else "C",
        )
    except (OSError, ValueError):
        raise ArtifactMigrationError("invalid embedding vector artifact") from None
    if vectors.ndim != 2 or vectors.dtype != np.dtype(np.float32):
        raise ArtifactMigrationError("embedding vectors must be a two-dimensional float32 array")
    if vectors.shape != (n_rows, dim):
        raise ArtifactMigrationError("embedding vector shape does not match progress sidecar")
    for start in range(0, n_rows, 16_384):
        chunk = vectors[start : start + 16_384]
        if not np.isfinite(chunk).all():
            raise ArtifactMigrationError("embedding vectors must contain only finite values")
        norms = np.linalg.norm(chunk, axis=1)
        if not np.isclose(norms, 1.0, atol=1e-3, rtol=0.0).all():
            raise ArtifactMigrationError("embedding vectors must be unit normalized")
    os.lseek(bound.fds[1], 0, os.SEEK_SET)
    _validate_index(_fd_path(bound.fds[1]), vectors, n_rows, dim)
    _validate_mapping(con, model_name, n_rows, dim)
    _verify_bound_content(bound)
    return ValidatedArtifacts(model_name, n_rows, dim, bound.artifacts, bound.identities)


def _fsync_bound(bound: _BoundArtifacts) -> None:
    for fd in bound.fds:
        try:
            os.fsync(fd)
        except OSError:
            raise ArtifactMigrationError("artifact file cannot be synchronized") from None


def _fsync_directory(root_fd: int) -> None:
    try:
        os.fsync(root_fd)
    except OSError:
        raise ArtifactMigrationError("artifact directory cannot be synchronized") from None


def _inspect_set(
    root_fd: int, artifacts: ArtifactSet
) -> tuple[FileIdentity | None, FileIdentity | None, FileIdentity | None]:
    return tuple(  # type: ignore[return-value]
        _fd_optional_identity(root_fd, path.name) for path in artifacts.ordered()
    )


def _migration_report(
    validated: ValidatedArtifacts,
    state: str,
    legacy: ArtifactSet,
    target: ArtifactSet,
    retired: ArtifactSet,
) -> MigrationReport:
    return MigrationReport(
        model=validated.model,
        n_rows=validated.n_rows,
        dim=validated.dim,
        state=state,
        legacy=legacy,
        target=target,
        retired=retired,
    )


def _clone_from_fd_no_replace(source_fd: int, root_fd: int, destination: str) -> None:
    """Create one atomic CoW clone from a retained source descriptor on Darwin."""
    if (
        not destination
        or destination in {".", ".."}
        or "\x00" in destination
        or Path(destination).name != destination
    ):
        raise ArtifactMigrationError("clone destination must be a direct child basename")
    if sys.platform != "darwin":
        raise ArtifactMigrationError(
            "descriptor-bound clone publication is unavailable on this platform"
        )
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        clone = libc.fclonefileat
    except (AttributeError, OSError):
        raise ArtifactMigrationError(
            "descriptor-bound clone publication is unavailable on this platform"
        ) from None
    clone.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint32,
    ]
    clone.restype = ctypes.c_int
    result = clone(source_fd, root_fd, os.fsencode(destination), 0)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error))
    raise OSError(error, os.strerror(error))


def _rename_no_replace(root_fd: int, source: str, destination: str) -> None:
    """Atomically move one direct child without overwriting the destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        try:
            rename = libc.renameatx_np
        except AttributeError:
            raise ArtifactMigrationError("atomic no-replace rename is unavailable") from None
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(root_fd, source_bytes, root_fd, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError:
            raise ArtifactMigrationError("atomic no-replace rename is unavailable") from None
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(root_fd, source_bytes, root_fd, destination_bytes, 0x00000001)
    else:
        raise ArtifactMigrationError("atomic no-replace rename is unavailable")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error))
    if error == errno.ENOENT:
        raise FileNotFoundError(error, os.strerror(error))
    raise OSError(error, os.strerror(error))


def _partition_artifacts(
    legacy: ArtifactSet,
    target: ArtifactSet,
    retired: ArtifactSet,
    source_identities: tuple[FileIdentity | None, FileIdentity | None, FileIdentity | None],
    target_identities: tuple[FileIdentity | None, FileIdentity | None, FileIdentity | None],
    retired_identities: tuple[FileIdentity | None, FileIdentity | None, FileIdentity | None],
) -> tuple[
    ArtifactSet,
    tuple[FileIdentity | None, FileIdentity | None, FileIdentity | None],
    tuple[str, str, str],
]:
    paths: list[Path] = []
    identities: list[FileIdentity] = []
    phases: list[str] = []
    for (
        source_path,
        _target_path,
        retired_path,
        source_identity,
        target_identity,
        retired_identity,
    ) in zip(
        legacy.ordered(),
        target.ordered(),
        retired.ordered(),
        source_identities,
        target_identities,
        retired_identities,
        strict=True,
    ):
        if source_identity is not None and target_identity is None and retired_identity is None:
            paths.append(source_path)
            identities.append(source_identity)
            phases.append("legacy")
        elif (
            source_identity is not None and target_identity is not None and retired_identity is None
        ):
            if (source_identity.device, source_identity.inode) == (
                target_identity.device,
                target_identity.inode,
            ):
                raise ArtifactMigrationError(
                    "pre-protocol hard-link alias is not a valid clone-retirement state"
                )
            paths.append(source_path)
            identities.append(source_identity)
            phases.append("cloned")
        elif (
            source_identity is None and target_identity is not None and retired_identity is not None
        ):
            if (target_identity.device, target_identity.inode) == (
                retired_identity.device,
                retired_identity.inode,
            ):
                raise ArtifactMigrationError(
                    "pre-protocol hard-link alias is not a valid clone-retirement state"
                )
            paths.append(retired_path)
            identities.append(retired_identity)
            phases.append("retired")
        else:
            raise ArtifactMigrationError("artifact state is not a clone-retirement partition")
    artifacts = ArtifactSet(memmap=paths[0], index=paths[1], progress=paths[2])
    return (
        artifacts,
        (identities[0], identities[1], identities[2]),
        (
            phases[0],
            phases[1],
            phases[2],
        ),
    )


def migrate_embedding_artifacts(
    con: duckdb.DuckDBPyConnection,
    artifact_dir: Path,
    model_name: str,
    *,
    execute: bool = False,
) -> MigrationReport:
    """Validate and optionally publish legacy artifacts under SHA-addressed names."""
    artifact_dir = Path(artifact_dir)
    legacy = legacy_artifact_paths(artifact_dir, model_name)
    target = target_artifact_paths(artifact_dir, model_name)
    retired = retired_artifact_paths(artifact_dir, model_name)
    root_fd, root_identity = _open_artifact_root(artifact_dir)
    opened_bounds: list[_BoundArtifacts] = []
    active_error: BaseException | None = None
    try:
        _revalidate_root(artifact_dir, root_fd, root_identity)
        source_identities = _inspect_set(root_fd, legacy)
        target_identities = _inspect_set(root_fd, target)
        retired_identities = _inspect_set(root_fd, retired)
        authoritative_paths, authoritative_identities, phases = _partition_artifacts(
            legacy,
            target,
            retired,
            source_identities,
            target_identities,
            retired_identities,
        )
        authoritative = _open_bound_artifacts(
            root_fd,
            authoritative_paths,
            authoritative_identities,
        )
        opened_bounds.append(authoritative)
        validated = _validate_bound_artifacts(con, authoritative, model_name)

        existing_target_bound: _BoundArtifacts | None = None
        if any(identity is not None for identity in target_identities):
            candidate_paths = tuple(
                target_path if target_identity is not None else source_path
                for target_path, source_path, target_identity in zip(
                    target.ordered(),
                    authoritative_paths.ordered(),
                    target_identities,
                    strict=True,
                )
            )
            candidate_identities = tuple(
                target_identity if target_identity is not None else source_identity
                for target_identity, source_identity in zip(
                    target_identities,
                    authoritative.identities,
                    strict=True,
                )
            )
            candidate = ArtifactSet(
                memmap=candidate_paths[0],
                index=candidate_paths[1],
                progress=candidate_paths[2],
            )
            existing_target_bound = _open_bound_artifacts(
                root_fd,
                candidate,
                candidate_identities,
            )
            opened_bounds.append(existing_target_bound)
            _validate_bound_artifacts(con, existing_target_bound, model_name)
            if existing_target_bound.digests != authoritative.digests:
                raise ArtifactMigrationError(
                    "existing target content does not match validated source"
                )

        _revalidate_root(artifact_dir, root_fd, root_identity)
        if (
            _inspect_set(root_fd, legacy) != source_identities
            or _inspect_set(root_fd, target) != target_identities
            or _inspect_set(root_fd, retired) != retired_identities
        ):
            raise ArtifactMigrationError("artifact identity changed during validation")
        fully_migrated = all(phase == "retired" for phase in phases)
        if not execute:
            state = "already-migrated" if fully_migrated else "planned"
            if (
                _inspect_set(root_fd, legacy) != source_identities
                or _inspect_set(root_fd, target) != target_identities
                or _inspect_set(root_fd, retired) != retired_identities
            ):
                raise ArtifactMigrationError("artifact identity changed before plan return")
            _revalidate_root(artifact_dir, root_fd, root_identity)
            return _migration_report(validated, state, legacy, target, retired)

        _verify_bound_content(authoritative)
        if fully_migrated:
            if existing_target_bound is None:
                raise ArtifactMigrationError("complete target set is required for replay")
            _fsync_bound(authoritative)
            _fsync_bound(existing_target_bound)
            _fsync_directory(root_fd)
            _verify_bound_content(authoritative)
            _verify_bound_content(existing_target_bound)
            if (
                _inspect_set(root_fd, legacy) != source_identities
                or _inspect_set(root_fd, target) != target_identities
                or _inspect_set(root_fd, retired) != retired_identities
            ):
                raise ArtifactMigrationError("artifact identity changed before replay return")
            _revalidate_root(artifact_dir, root_fd, root_identity)
            return _migration_report(validated, "already-migrated", legacy, target, retired)

        if sys.platform != "darwin":
            raise ArtifactMigrationError(
                "descriptor-bound clone publication is unavailable on this platform"
            )
        _fsync_bound(authoritative)
        _verify_bound_content(authoritative)
        for role, (_source_path, target_path, phase, source_fd) in enumerate(
            zip(
                legacy.ordered(),
                target.ordered(),
                phases,
                authoritative.fds,
                strict=True,
            )
        ):
            if phase != "legacy":
                continue
            _revalidate_root(artifact_dir, root_fd, root_identity)
            try:
                _clone_from_fd_no_replace(source_fd, root_fd, target_path.name)
            except FileExistsError:
                raise ArtifactMigrationError(
                    "target artifact appeared during clone publication"
                ) from None
            except OSError:
                raise ArtifactMigrationError("descriptor-bound artifact clone failed") from None
            _fsync_directory(root_fd)
            _revalidate_root(artifact_dir, root_fd, root_identity)
            if _fd_optional_identity(root_fd, target_path.name) is None:
                raise ArtifactMigrationError("cloned target artifact is unavailable")
            if role == 2 and any(
                _fd_optional_identity(root_fd, earlier.name) is None
                for earlier in target.ordered()[:2]
            ):
                raise ArtifactMigrationError("progress target was published before data targets")

        published_identities = _inspect_set(root_fd, target)
        if any(identity is None for identity in published_identities):
            raise ArtifactMigrationError("complete cloned target set is required")
        target_bound = _open_bound_artifacts(root_fd, target, published_identities)
        opened_bounds.append(target_bound)
        if any(
            (source_identity.device, source_identity.inode)
            == (target_identity.device, target_identity.inode)
            for source_identity, target_identity in zip(
                authoritative.identities,
                target_bound.identities,
                strict=True,
            )
        ):
            raise ArtifactMigrationError("target artifact is a hard-link alias of validated source")
        _validate_bound_artifacts(con, target_bound, model_name)
        if target_bound.digests != authoritative.digests:
            raise ArtifactMigrationError("cloned target content does not match validated source")
        _fsync_bound(target_bound)
        _fsync_directory(root_fd)
        _verify_bound_content(authoritative)
        _verify_bound_content(target_bound)

        expected_legacy = tuple(
            identity if phase != "retired" else None
            for identity, phase in zip(authoritative.identities, phases, strict=True)
        )
        expected_retired = tuple(
            identity if phase == "retired" else None
            for identity, phase in zip(authoritative.identities, phases, strict=True)
        )
        if _inspect_set(root_fd, legacy) != expected_legacy:
            raise ArtifactMigrationError("legacy artifact identity changed before retirement")
        if _inspect_set(root_fd, target) != target_bound.identities:
            raise ArtifactMigrationError("target artifact identity changed before retirement")
        if _inspect_set(root_fd, retired) != expected_retired:
            raise ArtifactMigrationError("retired artifact identity changed before retirement")

        for source_path, target_path, retired_path, source_identity, phase in zip(
            legacy.ordered(),
            target.ordered(),
            retired.ordered(),
            authoritative.identities,
            phases,
            strict=True,
        ):
            if phase == "retired":
                continue
            _revalidate_root(artifact_dir, root_fd, root_identity)
            try:
                _rename_no_replace(root_fd, source_path.name, retired_path.name)
            except FileExistsError:
                raise ArtifactMigrationError(
                    "retired artifact appeared during legacy retirement"
                ) from None
            except OSError:
                raise ArtifactMigrationError("atomic legacy retirement failed") from None
            _fsync_directory(root_fd)
            moved_retired = _fd_identity(root_fd, retired_path.name)
            remaining_source = _fd_optional_identity(root_fd, source_path.name)
            if moved_retired != source_identity or remaining_source is not None:
                raise ArtifactMigrationError(
                    "retired artifact identity does not match validated source"
                )
            if (
                _fd_identity(root_fd, target_path.name)
                != target_bound.identities[target.ordered().index(target_path)]
            ):
                raise ArtifactMigrationError("target artifact identity changed during retirement")
            _revalidate_root(artifact_dir, root_fd, root_identity)

        final_retired_identities = _inspect_set(root_fd, retired)
        if final_retired_identities != authoritative.identities:
            raise ArtifactMigrationError("retired artifacts do not match validated sources")
        retired_bound = _open_bound_artifacts(root_fd, retired, final_retired_identities)
        opened_bounds.append(retired_bound)
        _validate_bound_artifacts(con, retired_bound, model_name)
        if retired_bound.digests != authoritative.digests:
            raise ArtifactMigrationError("retired artifact content changed")
        _fsync_bound(retired_bound)

        _validate_bound_artifacts(con, retired_bound, model_name)
        final_validation = _validate_bound_artifacts(con, target_bound, model_name)
        terminal_retired_digests = tuple(_digest_fd(fd) for fd in retired_bound.fds)
        terminal_target_digests = tuple(_digest_fd(fd) for fd in target_bound.fds)
        if terminal_target_digests != terminal_retired_digests:
            raise ArtifactMigrationError("terminal target and retired content does not match")
        _verify_bound_content(authoritative)
        _fsync_directory(root_fd)
        _revalidate_root(artifact_dir, root_fd, root_identity)
        final_target_identities = _inspect_set(root_fd, target)
        if final_target_identities != target_bound.identities:
            raise ArtifactMigrationError("target identity changed after publication")
        final_retired_identities = _inspect_set(root_fd, retired)
        if final_retired_identities != retired_bound.identities:
            raise ArtifactMigrationError("retired identity changed after publication")
        if any(
            (target_identity.device, target_identity.inode)
            == (retired_identity.device, retired_identity.inode)
            for target_identity, retired_identity in zip(
                target_bound.identities,
                retired_bound.identities,
                strict=True,
            )
        ):
            raise ArtifactMigrationError("target and retired artifacts must use different inodes")
        if any(identity is not None for identity in _inspect_set(root_fd, legacy)):
            raise ArtifactMigrationError("legacy identity changed after retirement")
        _revalidate_root(artifact_dir, root_fd, root_identity)
        return _migration_report(final_validation, "migrated", legacy, target, retired)
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        close_failure: ArtifactMigrationError | None = None
        for bound in reversed(opened_bounds):
            for fd in reversed(bound.fds):
                try:
                    os.close(fd)
                except BaseException as exc:
                    if active_error is not None:
                        active_error.add_note(f"close error: {type(exc).__name__}")
                    elif close_failure is None:
                        close_failure = ArtifactMigrationError(
                            "artifact file descriptor close failed"
                        )
                        close_failure.__suppress_context__ = True
        try:
            os.close(root_fd)
        except BaseException as exc:
            if active_error is not None:
                active_error.add_note(f"close error: {type(exc).__name__}")
            elif close_failure is not None:
                close_failure.add_note(f"close error: {type(exc).__name__}")
            else:
                close_failure = ArtifactMigrationError("artifact directory close failed")
                close_failure.__suppress_context__ = True
        if active_error is None and close_failure is not None:
            raise close_failure
