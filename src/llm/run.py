"""Phase 8: the labelling job.

docs/LLM_LAYER.md §2.4 — label **lazily**. Labelling every cluster in every
backtest refit is a large avoidable bill and produces labels nobody reads, so the
default population is the clusters that actually fired a signal, plus a seeded
random control sample so §2.5's verification is not drawn only from alerts.

This module is the only writer in `src/llm/`; retrieval has a separate read-only
evidence boundary. This writer touches only the downstream `cluster_labels` and
`llm_usage` tables, neither of which a detection stage reads. The §1 determinism
contract is a property of that:
`tests/test_llm.py` asserts no detection module can import this package at all.
"""

from __future__ import annotations

import json
import math
import os
import random
import stat
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src import db
from src.alert_scope import canonical_fired_cluster_ids
from src.config import CONFIG, PATHS
from src.llm import label as label_mod
from src.llm import select as select_mod
from src.llm.client import AnthropicModelClient, ModelCallError, ModelCallResult


@dataclass
class LabelRunStats:
    labelled: int = 0
    cached: int = 0
    refused: int = 0
    failed: int = 0
    skipped: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_seconds: float = 0.0


class ReviewedLabelChangeError(RuntimeError):
    """A reviewed cluster cannot be rebound to a different model input."""


class LabelTransactionError(RuntimeError):
    """Label persistence requires an autocommit DuckDB connection."""


class LabelUsageOutboxError(ValueError):
    """A durable label-usage event is corrupt or incompatible."""


@dataclass(frozen=True)
class UsageRecord:
    run_id: str | None
    cluster_id: str
    input_hash: str
    cache_status: str
    attempts: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    latency_seconds: float
    estimated_cost_usd: float
    outcome: str
    error_category: str | None = None
    usage_id: str = ""
    model: str = ""
    prompt_version: str = ""
    created_at: datetime | None = None


def record_usage(con, usage: UsageRecord) -> None:
    """Persist one privacy-safe accounting event for one label target."""
    con.execute(
        "INSERT INTO llm_usage "
        "(usage_id, run_id, operation, cluster_id, model, prompt_version, "
        "input_hash, cache_status, attempts, input_tokens, output_tokens, "
        "cache_read_input_tokens, cache_creation_input_tokens, latency_seconds, "
        "estimated_cost_usd, outcome, error_category, created_at) "
        "VALUES (?, ?, 'label', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (usage_id) DO NOTHING",
        [
            usage.usage_id,
            usage.run_id,
            usage.cluster_id,
            usage.model,
            usage.prompt_version,
            usage.input_hash,
            usage.cache_status,
            usage.attempts,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
            usage.latency_seconds,
            usage.estimated_cost_usd,
            usage.outcome,
            usage.error_category,
            usage.created_at,
        ],
    )
    expected = (
        usage.run_id,
        "label",
        usage.cluster_id,
        usage.model,
        usage.prompt_version,
        usage.input_hash,
        usage.cache_status,
        usage.attempts,
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_input_tokens,
        usage.cache_creation_input_tokens,
        usage.latency_seconds,
        usage.estimated_cost_usd,
        usage.outcome,
        usage.error_category,
        usage.created_at,
    )
    stored = con.execute(
        "SELECT run_id, operation, cluster_id, model, prompt_version, input_hash, "
        "cache_status, attempts, input_tokens, output_tokens, cache_read_input_tokens, "
        "cache_creation_input_tokens, latency_seconds, estimated_cost_usd, outcome, "
        "error_category, created_at FROM llm_usage WHERE usage_id = ?",
        [usage.usage_id],
    ).fetchone()
    if stored is None or stored[:14] != expected[:14] or stored[16] != expected[16]:
        raise LabelUsageOutboxError("label usage_id conflicts with different accounting data")
    if stored[14:16] == expected[14:16]:
        return
    desired_cache_failure = expected[14:16] == ("failed", "cache_write")
    stored_cache_failure = stored[14:16] == ("failed", "cache_write")
    if desired_cache_failure and stored[14] in {"ok", "refused"} and stored[15] is None:
        con.execute(
            "UPDATE llm_usage SET outcome = 'failed', error_category = 'cache_write' "
            "WHERE usage_id = ?",
            [usage.usage_id],
        )
        corrected = con.execute(
            "SELECT outcome, error_category FROM llm_usage WHERE usage_id = ?",
            [usage.usage_id],
        ).fetchone()
        if corrected != ("failed", "cache_write"):
            raise LabelUsageOutboxError("label usage cache-failure correction was not durable")
        return
    if stored_cache_failure and expected[14] in {"ok", "refused"}:
        # A stale pre-cache outbox event may survive a crash after the durable
        # cache-write failure correction. It must never downgrade that outcome.
        return
    raise LabelUsageOutboxError("label usage_id conflicts with a different outcome")


def population(con, cluster_run: str, signals_run: str, control_n: int):
    """Clusters that fired, plus a seeded random control sample of those that did not."""
    fired_ids = canonical_fired_cluster_ids(con, signals_run)
    rows = con.execute(
        """
        SELECT c.cluster_id, c.product_family, c.n_members, c.centroid_idx,
               coalesce(n.dominant_label, '') AS dominant_label
        FROM clusters c
        LEFT JOIN cluster_novelty n USING (cluster_id)
        WHERE c.run_id = ? AND c.n_members >= ?
        ORDER BY c.cluster_id
        """,
        [cluster_run, CONFIG.llm.min_cluster_size_for_label],
    ).fetchall()
    rows = [(*row, row[0] in fired_ids) for row in rows]
    fired = [r for r in rows if r[5]]
    quiet = [r for r in rows if not r[5]]
    rng = random.Random(CONFIG.seed)  # noqa: S311 - sampling, not cryptography
    control = rng.sample(quiet, min(control_n, len(quiet)))
    return fired + control, len(fired), len(control)


def _validate_run_provenance(con, cluster_run: str, signals_run: str) -> None:
    """Prove the signals run scored the exact requested cluster refit."""
    row = con.execute(
        "SELECT json_extract_string(params_json, '$.params.cluster_run') "
        "FROM runs WHERE run_id = ? AND phase = 'signals' AND status = 'ok'",
        [signals_run],
    ).fetchone()
    if row is None or not row[0]:
        raise ValueError(
            f"signals run {signals_run!r} is not a successful signals run with cluster provenance"
        )
    if row[0] != cluster_run:
        raise ValueError(
            f"signals run {signals_run!r} records cluster_run {row[0]!r}; requested {cluster_run!r}"
        )


def embedding_model_for_cluster_run(
    con,
    cluster_run: str,
    requested_model: str | None = None,
) -> str:
    """Return the exact successful cluster-run model and reject mismatches."""
    row = con.execute(
        "SELECT json_extract_string(params_json, '$.params.model') "
        "FROM runs WHERE run_id = ? AND phase = 'cluster' AND status = 'ok'",
        [cluster_run],
    ).fetchone()
    if row is None or type(row[0]) is not str or not row[0].strip():
        raise ValueError(f"cluster run {cluster_run!r} does not record an embedding model")
    recorded = row[0]
    if requested_model is not None and requested_model != recorded:
        raise ValueError(
            f"cluster run {cluster_run} records embedding model {recorded!r}; "
            f"requested {requested_model!r}"
        )
    return recorded


def _require_autocommit(con) -> None:
    first_id = con.execute("SELECT current_transaction_id()").fetchone()[0]
    second_id = con.execute("SELECT current_transaction_id()").fetchone()[0]
    if first_id == second_id:
        raise LabelTransactionError("label runner requires an autocommit connection")


_OUTBOX_VERSION = 1
_OUTBOX_FIELDS = frozenset(
    {
        "version",
        "usage_id",
        "run_id",
        "cluster_id",
        "model",
        "prompt_version",
        "input_hash",
        "cache_status",
        "attempts",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "latency_seconds",
        "estimated_cost_usd",
        "outcome",
        "error_category",
        "created_at",
    }
)


_OUTBOX_PREFIX = "label-usage-"
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _verify_cache_root(parent_fd: int, root_fd: int, root_name: str) -> None:
    try:
        current = os.stat(root_name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise LabelUsageOutboxError("label cache root changed during usage accounting") from None
    pinned = os.fstat(root_fd)
    if not os.path.samestat(current, pinned):
        raise LabelUsageOutboxError("label cache root changed during usage accounting")


def _open_cache_root(cache_dir: Path, *, create: bool) -> tuple[Path, int, int]:
    configured = Path(os.path.abspath(cache_dir))
    if configured.name in {"", ".", ".."}:
        raise LabelUsageOutboxError("label cache root is invalid")
    parent = configured.parent
    try:
        canonical_parent = parent.resolve(strict=True)
    except FileNotFoundError:
        raise LabelUsageOutboxError("label cache parent does not exist") from None
    if canonical_parent != parent:
        raise LabelUsageOutboxError("label cache root must not contain symlink components")
    parent_fd = os.open(parent, _DIRECTORY_FLAGS)
    root_fd: int | None = None
    try:
        try:
            root_fd = os.open(configured.name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        except FileNotFoundError:
            if not create:
                raise
            os.mkdir(configured.name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            root_fd = os.open(configured.name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        except OSError:
            raise LabelUsageOutboxError("label cache root is not a trusted directory") from None
        _verify_cache_root(parent_fd, root_fd, configured.name)
        return configured, parent_fd, root_fd
    except BaseException:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)
        raise


def _close_cache_root(parent_fd: int, root_fd: int) -> None:
    try:
        os.close(root_fd)
    finally:
        os.close(parent_fd)


def _event_filename(usage_id: str) -> str:
    if (
        type(usage_id) is not str
        or not usage_id
        or usage_id.startswith(".")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in usage_id
        )
    ):
        raise LabelUsageOutboxError("label usage_id is not safe for durable accounting")
    return f"{_OUTBOX_PREFIX}{usage_id}.json"


def _usage_payload(usage: UsageRecord) -> dict[str, object]:
    return {
        "version": _OUTBOX_VERSION,
        "usage_id": usage.usage_id,
        "run_id": usage.run_id,
        "cluster_id": usage.cluster_id,
        "model": usage.model,
        "prompt_version": usage.prompt_version,
        "input_hash": usage.input_hash,
        "cache_status": usage.cache_status,
        "attempts": usage.attempts,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        "latency_seconds": usage.latency_seconds,
        "estimated_cost_usd": usage.estimated_cost_usd,
        "outcome": usage.outcome,
        "error_category": usage.error_category,
        "created_at": usage.created_at.isoformat() if usage.created_at is not None else None,
    }


def _replace_usage_from_snapshot(
    root_fd: int,
    snapshot_fd: int,
    temporary_name: str,
    target_name: str,
) -> None:
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
            target_name,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
    finally:
        if recovery_fd is not None:
            os.close(recovery_fd)


def _usage_file_identity(value: os.stat_result) -> tuple[int, ...]:
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


def _same_usage_file_bytes(first_fd: int, second_fd: int) -> bool:
    os.lseek(first_fd, 0, os.SEEK_SET)
    os.lseek(second_fd, 0, os.SEEK_SET)
    while True:
        first = os.read(first_fd, 1024 * 1024)
        second = os.read(second_fd, 1024 * 1024)
        if first != second:
            return False
        if not first:
            return True


def _stage_usage_direct(usage: UsageRecord, cache_dir: Path) -> Path:
    configured, parent_fd, root_fd = _open_cache_root(cache_dir, create=True)
    target_name = _event_filename(usage.usage_id)
    nonce = db.new_run_id()
    temporary_name = f".{target_name}.{nonce}.tmp"
    backup_name = f".{target_name}.{nonce}.bak"
    snapshot_name = f".{target_name}.{nonce}.snapshot"
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
        _verify_cache_root(parent_fd, root_fd, configured.name)
        try:
            current = os.stat(target_name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and not stat.S_ISREG(current.st_mode):
            raise LabelUsageOutboxError("label usage outbox event is not a regular file")
        try:
            old_source_fd = os.open(
                target_name,
                os.O_RDONLY | _FILE_NOFOLLOW,
                dir_fd=root_fd,
            )
        except FileNotFoundError:
            old_source_fd = None
        except OSError:
            raise LabelUsageOutboxError(
                "label usage outbox event changed during publication"
            ) from None
        if old_source_fd is not None:
            source_before = os.fstat(old_source_fd)
            try:
                named_before = os.stat(target_name, dir_fd=root_fd, follow_symlinks=False)
            except OSError:
                raise LabelUsageOutboxError(
                    "label usage outbox event changed during publication"
                ) from None
            if not stat.S_ISREG(source_before.st_mode):
                raise LabelUsageOutboxError("label usage outbox event is not a regular file")
            old_source_identity = _usage_file_identity(source_before)
            if _usage_file_identity(named_before) != old_source_identity:
                raise LabelUsageOutboxError("label usage outbox event changed during publication")
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
                named_after = os.stat(target_name, dir_fd=root_fd, follow_symlinks=False)
            except OSError:
                raise LabelUsageOutboxError(
                    "label usage outbox event changed during publication"
                ) from None
            if (
                not stat.S_ISREG(snapshot_stat.st_mode)
                or stat.S_IMODE(snapshot_stat.st_mode) != 0o600
                or (snapshot_stat.st_dev, snapshot_stat.st_ino)
                == (source_before.st_dev, source_before.st_ino)
                or _usage_file_identity(source_after) != old_source_identity
                or _usage_file_identity(named_after) != old_source_identity
            ):
                raise LabelUsageOutboxError("label usage outbox event changed during publication")
            snapshot_ready = True
            os.unlink(snapshot_name, dir_fd=root_fd)
            snapshot_named = False
            os.fsync(root_fd)
        data = json.dumps(_usage_payload(usage), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
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
        _verify_cache_root(parent_fd, root_fd, configured.name)
        if old_source_fd is None:
            try:
                os.link(
                    target_name,
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
                raise LabelUsageOutboxError("label usage outbox event changed during publication")
        else:
            try:
                source_before_link = os.fstat(old_source_fd)
                named_before_link = os.stat(
                    target_name,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
            except OSError:
                destination_changed = True
                raise LabelUsageOutboxError(
                    "label usage outbox event changed during publication"
                ) from None
            if (
                _usage_file_identity(source_before_link) != old_source_identity
                or _usage_file_identity(named_before_link) != old_source_identity
            ):
                destination_changed = True
                raise LabelUsageOutboxError("label usage outbox event changed during publication")
            try:
                os.link(
                    target_name,
                    backup_name,
                    src_dir_fd=root_fd,
                    dst_dir_fd=root_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                destination_changed = True
                raise LabelUsageOutboxError(
                    "label usage outbox event changed during publication"
                ) from None
            backup_created = True
            os.fsync(root_fd)
            try:
                backup = os.stat(backup_name, dir_fd=root_fd, follow_symlinks=False)
            except OSError:
                destination_changed = True
                raise LabelUsageOutboxError(
                    "label usage outbox event changed during publication"
                ) from None
            source_after_link = os.fstat(old_source_fd)
            named_after_link = os.stat(
                target_name,
                dir_fd=root_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(backup.st_mode)
                or _usage_file_identity(backup) != _usage_file_identity(source_after_link)
                or _usage_file_identity(named_after_link) != _usage_file_identity(source_after_link)
                or old_snapshot_fd is None
                or not _same_usage_file_bytes(old_snapshot_fd, old_source_fd)
            ):
                destination_changed = True
                raise LabelUsageOutboxError("label usage outbox event changed during publication")
        os.replace(
            temporary_name,
            target_name,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
        published = True
        os.fsync(root_fd)
        _verify_cache_root(parent_fd, root_fd, configured.name)
        publication_verified = True
        if backup_created:
            os.unlink(backup_name, dir_fd=root_fd)
            backup_created = False
            os.fsync(root_fd)
        try:
            _verify_cache_root(parent_fd, root_fd, configured.name)
        except BaseException:
            publication_verified = False
            raise
        return configured / target_name
    except BaseException as stage_error:
        operation_error = stage_error
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
                    _replace_usage_from_snapshot(
                        root_fd,
                        old_snapshot_fd,
                        temporary_name,
                        target_name,
                    )
                elif old_source_fd is None and published:
                    os.unlink(target_name, dir_fd=root_fd)
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
            stage_error.add_note(
                "label usage publication recovery also failed: " + ", ".join(recovery_errors)
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
                                "label usage descriptor cleanup also failed: "
                                + type(error).__name__
                            )
                    else:
                        operation_error.add_note(
                            "label usage descriptor cleanup also failed: " + type(error).__name__
                        )
        finally:
            _close_cache_root(parent_fd, root_fd)
        if cleanup_error is not None:
            raise cleanup_error


def _stage_usage(usage: UsageRecord, cache_dir: Path) -> Path:
    return _stage_usage_direct(usage, cache_dir)


def _remove_staged_usage(path: Path) -> None:
    configured, parent_fd, root_fd = _open_cache_root(path.parent, create=False)
    try:
        if configured / path.name != path:
            raise LabelUsageOutboxError("label usage outbox event path is invalid")
        try:
            current = os.stat(path.name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISREG(current.st_mode):
            raise LabelUsageOutboxError("label usage outbox event is not a regular file")
        _verify_cache_root(parent_fd, root_fd, configured.name)
        os.unlink(path.name, dir_fd=root_fd)
        os.fsync(root_fd)
        _verify_cache_root(parent_fd, root_fd, configured.name)
    finally:
        _close_cache_root(parent_fd, root_fd)


def _nonblank(payload: dict[str, object], field: str) -> str:
    value = payload[field]
    if type(value) is not str or not value.strip():
        raise LabelUsageOutboxError(f"label usage outbox {field} must be nonblank")
    return value


def _nonnegative_int(payload: dict[str, object], field: str) -> int:
    value = payload[field]
    if type(value) is not int or value < 0:
        raise LabelUsageOutboxError(f"label usage outbox {field} must be a non-negative integer")
    return value


def _nonnegative_float(payload: dict[str, object], field: str) -> float:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabelUsageOutboxError(f"label usage outbox {field} must be finite")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise LabelUsageOutboxError(f"label usage outbox {field} must be non-negative")
    return normalized


def _usage_from_payload(payload: object) -> UsageRecord:
    if type(payload) is not dict or set(payload) != _OUTBOX_FIELDS:
        raise LabelUsageOutboxError("label usage outbox fields do not match version 1")
    if payload["version"] != _OUTBOX_VERSION:
        raise LabelUsageOutboxError("label usage outbox version is incompatible")
    run_id = payload["run_id"]
    error_category = payload["error_category"]
    if run_id is not None and (type(run_id) is not str or not run_id.strip()):
        raise LabelUsageOutboxError("label usage outbox run_id must be null or nonblank")
    if error_category is not None and (
        type(error_category) is not str or not error_category.strip()
    ):
        raise LabelUsageOutboxError("label usage outbox error_category must be null or nonblank")
    cache_status = _nonblank(payload, "cache_status")
    outcome = _nonblank(payload, "outcome")
    if cache_status not in {"hit", "miss", "bypass"}:
        raise LabelUsageOutboxError("label usage outbox cache_status is invalid")
    if outcome not in {"ok", "refused", "failed", "skipped"}:
        raise LabelUsageOutboxError("label usage outbox outcome is invalid")
    try:
        created_at = datetime.fromisoformat(_nonblank(payload, "created_at"))
    except ValueError as exc:
        raise LabelUsageOutboxError("label usage outbox created_at is invalid") from exc
    return UsageRecord(
        run_id=run_id,
        cluster_id=_nonblank(payload, "cluster_id"),
        input_hash=_nonblank(payload, "input_hash"),
        cache_status=cache_status,
        attempts=_nonnegative_int(payload, "attempts"),
        input_tokens=_nonnegative_int(payload, "input_tokens"),
        output_tokens=_nonnegative_int(payload, "output_tokens"),
        cache_read_input_tokens=_nonnegative_int(payload, "cache_read_input_tokens"),
        cache_creation_input_tokens=_nonnegative_int(payload, "cache_creation_input_tokens"),
        latency_seconds=_nonnegative_float(payload, "latency_seconds"),
        estimated_cost_usd=_nonnegative_float(payload, "estimated_cost_usd"),
        outcome=outcome,
        error_category=error_category,
        usage_id=_nonblank(payload, "usage_id"),
        model=_nonblank(payload, "model"),
        prompt_version=_nonblank(payload, "prompt_version"),
        created_at=created_at,
    )


def drain_usage_outbox(con, cache_dir: Path) -> int:
    """Idempotently persist staged paid events before cache/provider work."""
    _require_autocommit(con)
    try:
        configured, parent_fd, root_fd = _open_cache_root(cache_dir, create=False)
    except FileNotFoundError:
        return 0
    try:
        drained = 0
        names = sorted(
            name
            for name in os.listdir(root_fd)
            if name.startswith(_OUTBOX_PREFIX) and name.endswith(".json")
        )
        for name in names:
            file_fd: int | None = None
            try:
                _verify_cache_root(parent_fd, root_fd, configured.name)
                try:
                    file_fd = os.open(name, os.O_RDONLY | _FILE_NOFOLLOW, dir_fd=root_fd)
                except OSError as exc:
                    raise LabelUsageOutboxError(
                        f"cannot read label usage outbox event {name}"
                    ) from exc
                before = os.fstat(file_fd)
                if not stat.S_ISREG(before.st_mode):
                    raise LabelUsageOutboxError("label usage outbox event is not a regular file")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(file_fd, 64 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                after = os.fstat(file_fd)
                current = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ) or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                    current.st_dev,
                    current.st_ino,
                    current.st_size,
                    current.st_mtime_ns,
                ):
                    raise LabelUsageOutboxError("label usage outbox event changed during read")
                try:
                    payload = json.loads(b"".join(chunks).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise LabelUsageOutboxError(
                        f"cannot read label usage outbox event {name}"
                    ) from exc
                usage = _usage_from_payload(payload)
                if name != _event_filename(usage.usage_id):
                    raise LabelUsageOutboxError(
                        "label usage outbox filename does not match usage_id"
                    )
                record_usage(con, usage)
                _verify_cache_root(parent_fd, root_fd, configured.name)
                os.unlink(name, dir_fd=root_fd)
                os.fsync(root_fd)
                drained += 1
            finally:
                if file_fd is not None:
                    os.close(file_fd)
        return drained
    finally:
        _close_cache_root(parent_fd, root_fd)


def narratives_for(con, vectors, cluster_id: str, medoid_idx, model: str):
    """The k narratives §2.1 selects, and the complaint ids they came from."""
    import numpy as np

    rows = con.execute(
        """
        SELECT m.complaint_id, e.row_idx, n.text_redacted
        FROM cluster_members m
        JOIN embedding_map e USING (complaint_id)
        JOIN narratives n USING (complaint_id)
        WHERE m.cluster_id = ? AND e.model = ?
        ORDER BY m.complaint_id
        """,
        [cluster_id, model],
    ).fetchnumpy()
    if not len(rows["complaint_id"]):
        return [], []

    text_by_id = dict(zip(rows["complaint_id"], rows["text_redacted"], strict=True))
    picked = select_mod.select_for_label(
        vectors,
        rows["row_idx"].astype(np.int64),
        rows["complaint_id"].astype(np.int64),
        medoid_idx,
        CONFIG.llm.label_sample_k,
        CONFIG.llm.label_medoid_k,
    )
    return picked, [text_by_id[i] for i in picked]


def write_label(con, cluster_id: str, key: str, payload: dict) -> None:
    con.execute(
        "INSERT INTO cluster_labels (cluster_id, harm_mechanism, actors, "
        "preconditions, consumer_impact, distinct_from_taxonomy, rationale, "
        "confidence, is_likely_template, model, prompt_version, input_hash, "
        "generated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT (cluster_id) DO UPDATE SET "
        "harm_mechanism = excluded.harm_mechanism, actors = excluded.actors, "
        "preconditions = excluded.preconditions, consumer_impact = excluded.consumer_impact, "
        "distinct_from_taxonomy = excluded.distinct_from_taxonomy, "
        "rationale = excluded.rationale, confidence = excluded.confidence, "
        "is_likely_template = excluded.is_likely_template, model = excluded.model, "
        "prompt_version = excluded.prompt_version, input_hash = excluded.input_hash, "
        "generated_at = excluded.generated_at",
        [
            cluster_id,
            payload.get("harm_mechanism"),
            ", ".join(payload.get("actors") or []),
            payload.get("preconditions"),
            payload.get("consumer_impact"),
            payload.get("distinct_from_taxonomy"),
            payload.get("distinctness_rationale"),
            payload.get("confidence"),
            payload.get("is_likely_template"),
            CONFIG.llm.model,
            CONFIG.llm.prompt_version,
            key,
            datetime.now(),
        ],
    )


def _usage_from_result(
    run_id: str | None,
    cluster_id: str,
    key: str,
    cache_status: str,
    outcome: str,
    result: ModelCallResult | None = None,
    error_category: str | None = None,
    attempts: int = 0,
) -> UsageRecord:
    common = {
        "usage_id": db.new_run_id(),
        "model": CONFIG.llm.model,
        "prompt_version": CONFIG.llm.prompt_version,
        "created_at": datetime.now(),
    }
    if result is None:
        return UsageRecord(
            run_id,
            cluster_id,
            key,
            cache_status,
            attempts,
            0,
            0,
            0,
            0,
            0.0,
            0.0,
            outcome,
            error_category,
            **common,
        )
    return UsageRecord(
        run_id,
        cluster_id,
        key,
        cache_status,
        result.attempts,
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.cache_read_input_tokens,
        result.usage.cache_creation_input_tokens,
        result.latency_seconds,
        result.estimated_cost_usd,
        outcome,
        error_category,
        **common,
    )


def _usage_from_error(
    run_id: str | None,
    cluster_id: str,
    key: str,
    cache_status: str,
    error: ModelCallError,
) -> UsageRecord:
    return UsageRecord(
        run_id,
        cluster_id,
        key,
        cache_status,
        error.attempts,
        error.usage.input_tokens,
        error.usage.output_tokens,
        error.usage.cache_read_input_tokens,
        error.usage.cache_creation_input_tokens,
        error.latency_seconds,
        error.estimated_cost_usd,
        "failed",
        error.category,
        usage_id=db.new_run_id(),
        model=CONFIG.llm.model,
        prompt_version=CONFIG.llm.prompt_version,
        created_at=datetime.now(),
    )


def _add_result_totals(stats: LabelRunStats, result: ModelCallResult) -> None:
    stats.input_tokens += result.usage.input_tokens
    stats.output_tokens += result.usage.output_tokens
    stats.estimated_cost_usd += result.estimated_cost_usd
    stats.latency_seconds += result.latency_seconds


def _add_error_totals(stats: LabelRunStats, error: ModelCallError) -> None:
    stats.input_tokens += error.usage.input_tokens
    stats.output_tokens += error.usage.output_tokens
    stats.estimated_cost_usd += error.estimated_cost_usd
    stats.latency_seconds += error.latency_seconds


def _existing_label(con, cluster_id: str) -> tuple[str | None, bool] | None:
    row = con.execute(
        "SELECT l.input_hash, EXISTS ("
        "SELECT 1 FROM label_verifications v WHERE v.cluster_id = l.cluster_id"
        ") FROM cluster_labels l WHERE l.cluster_id = ?",
        [cluster_id],
    ).fetchone()
    return None if row is None else (row[0], bool(row[1]))


def _persist_target(
    con, cluster_id: str, key: str, payload: dict | None, usage: UsageRecord
) -> None:
    """Commit the label and its accounting row together, or neither."""
    con.execute("BEGIN TRANSACTION")
    try:
        if payload is not None and not payload.get("refused"):
            write_label(con, cluster_id, key, payload)
        record_usage(con, usage)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


def run(
    con,
    cluster_run: str,
    signals_run: str,
    control_n: int,
    limit: int | None,
    embed_model: str,
    log=print,
    *,
    client=None,
    vectors=None,
    cache_dir=None,
    run_id: str | None = None,
) -> LabelRunStats:
    """Label the lazy population with resumable cache and per-target accounting."""
    _require_autocommit(con)
    _validate_run_provenance(con, cluster_run, signals_run)
    embedding_model_for_cluster_run(con, cluster_run, embed_model)

    cache = Path(cache_dir or PATHS.llm_cache)
    drain_usage_outbox(con, cache)

    import numpy as np

    if vectors is None:
        from src.embed.encode import embedding_artifact_paths

        memmap = embedding_artifact_paths(PATHS.artifacts, embed_model).memmap
        vectors = np.load(memmap, mmap_mode="r")
    targets, n_fired, n_control = population(con, cluster_run, signals_run, control_n)
    if limit:
        targets = targets[:limit]
    log(
        f"population : {len(targets):,} clusters "
        f"({n_fired:,} fired, {n_control:,} control) "
        f"at n_members >= {CONFIG.llm.min_cluster_size_for_label}"
    )
    log(f"model      : {CONFIG.llm.model}   prompt {CONFIG.llm.prompt_version}")

    stats = LabelRunStats()
    preflight_done = False
    for cluster_id, family, _n, medoid_idx, dominant, _fired in targets:
        ids, texts = narratives_for(con, vectors, cluster_id, medoid_idx, embed_model)
        key = label_mod.input_hash(
            CONFIG.llm.prompt_version,
            CONFIG.llm.model,
            ids,
            embed_model,
        )
        if not texts:
            stats.skipped += 1
            _persist_target(
                con,
                cluster_id,
                key,
                None,
                _usage_from_result(run_id, cluster_id, key, "bypass", "skipped"),
            )
            continue

        existing = _existing_label(con, cluster_id)
        if existing is not None and existing[0] == key:
            stats.cached += 1
            _persist_target(
                con,
                cluster_id,
                key,
                None,
                _usage_from_result(run_id, cluster_id, key, "hit", "ok"),
            )
            stats.labelled += 1
            continue
        if existing is not None and existing[1]:
            raise ReviewedLabelChangeError(
                f"reviewed cluster {cluster_id!r} already has input_hash "
                f"{existing[0]!r}; refusing changed input_hash {key!r}"
            )

        payload = label_mod.cached(cache, key)
        if payload is None:
            if client is None:
                client = AnthropicModelClient()
            if not preflight_done:
                try:
                    client.preflight(CONFIG.llm.model)
                except ModelCallError as error:
                    stats.failed += 1
                    _add_error_totals(stats, error)
                    _persist_target(
                        con,
                        cluster_id,
                        key,
                        None,
                        _usage_from_error(run_id, cluster_id, key, "miss", error),
                    )
                    if not error.retryable:
                        raise
                    continue
                preflight_done = True
            try:
                result = client.call_json(
                    model=CONFIG.llm.model,
                    system=label_mod.SYSTEM,
                    prompt=label_mod.build_prompt(
                        texts,
                        [d for d in [dominant] if d],
                        CONFIG.llm.max_narrative_chars,
                    ),
                    schema=label_mod.LABEL_SCHEMA,
                    max_tokens=2000,
                )
            except ModelCallError as error:
                stats.failed += 1
                _add_error_totals(stats, error)
                usage = _usage_from_error(run_id, cluster_id, key, "miss", error)
                staged_path: Path | None = None
                if error.response_received:
                    try:
                        staged_path = _stage_usage(usage, cache)
                    except (OSError, LabelUsageOutboxError) as stage_error:
                        try:
                            _persist_target(con, cluster_id, key, None, usage)
                        except Exception as persistence_error:
                            error.add_note(
                                "paid usage outbox and database persistence also failed: "
                                f"{type(stage_error).__name__}, "
                                f"{type(persistence_error).__name__}"
                            )
                        else:
                            error.add_note(
                                "paid usage outbox staging failed after durable database insert: "
                                f"{type(stage_error).__name__}"
                            )
                        raise error from stage_error
                try:
                    _persist_target(con, cluster_id, key, None, usage)
                except Exception as persistence_error:
                    error.add_note(
                        f"paid usage persistence also failed: {type(persistence_error).__name__}"
                    )
                    raise error from persistence_error
                if staged_path is not None:
                    with suppress(OSError):
                        _remove_staged_usage(staged_path)
                if not error.retryable and not error.response_received:
                    raise
                continue
            _add_result_totals(stats, result)
            try:
                payload = label_mod._validate_cached_payload(result.payload)
            except label_mod.LabelSchemaError as schema_error:
                stats.failed += 1
                usage = _usage_from_result(
                    run_id,
                    cluster_id,
                    key,
                    "miss",
                    "failed",
                    result,
                    "schema",
                )
                try:
                    staged_path = _stage_usage(usage, cache)
                except Exception as stage_error:
                    try:
                        _persist_target(con, cluster_id, key, None, usage)
                    except Exception as persistence_error:
                        schema_error.add_note(
                            "paid usage outbox staging and database fallback failed: "
                            f"{type(stage_error).__name__}, "
                            f"{type(persistence_error).__name__}"
                        )
                    else:
                        schema_error.add_note(
                            "paid usage outbox staging failed after durable database fallback: "
                            f"{type(stage_error).__name__}"
                        )
                    raise schema_error from stage_error
                try:
                    _persist_target(con, cluster_id, key, None, usage)
                except Exception as persistence_error:
                    schema_error.add_note(
                        f"paid usage persistence also failed: {type(persistence_error).__name__}"
                    )
                    raise schema_error from persistence_error
                with suppress(OSError):
                    _remove_staged_usage(staged_path)
                continue

            usage = _usage_from_result(
                run_id,
                cluster_id,
                key,
                "miss",
                "refused" if payload.get("refused") else "ok",
                result,
            )
            try:
                staged_path = _stage_usage(usage, cache)
            except (OSError, LabelUsageOutboxError) as stage_error:
                try:
                    _persist_target(
                        con,
                        cluster_id,
                        key,
                        None if payload.get("refused") else payload,
                        usage,
                    )
                except Exception as persistence_error:
                    stage_error.add_note(
                        "paid usage database persistence also failed: "
                        f"{type(persistence_error).__name__}"
                    )
                raise
            try:
                label_mod.write_cache(cache, key, payload)
            except Exception as cache_error:
                stats.failed += 1
                failed_usage = UsageRecord(
                    **{
                        **usage.__dict__,
                        "outcome": "failed",
                        "error_category": "cache_write",
                    }
                )
                try:
                    corrected_path = _stage_usage(failed_usage, cache)
                except Exception as stage_error:
                    try:
                        _persist_target(con, cluster_id, key, None, failed_usage)
                    except Exception as persistence_error:
                        try:
                            _stage_usage_direct(failed_usage, cache)
                        except Exception as repair_error:
                            cache_error.add_note(
                                "paid cache-failure usage repair also failed: "
                                f"{type(repair_error).__name__}"
                            )
                        cache_error.add_note(
                            "paid cache-failure usage restaging and database fallback failed: "
                            f"{type(stage_error).__name__}, "
                            f"{type(persistence_error).__name__}"
                        )
                    else:
                        with suppress(OSError, LabelUsageOutboxError):
                            _remove_staged_usage(staged_path)
                else:
                    try:
                        _persist_target(con, cluster_id, key, None, failed_usage)
                    except Exception as persistence_error:
                        cache_error.add_note(
                            "paid cache-failure database persistence failed: "
                            f"{type(persistence_error).__name__}"
                        )
                    else:
                        with suppress(OSError, LabelUsageOutboxError):
                            _remove_staged_usage(corrected_path)
                raise cache_error
        else:
            stats.cached += 1
            usage = _usage_from_result(
                run_id,
                cluster_id,
                key,
                "hit",
                "refused" if payload.get("refused") else "ok",
            )

        if payload.get("refused"):
            stats.refused += 1
            try:
                _persist_target(con, cluster_id, key, None, usage)
            except Exception:
                raise
            if usage.input_tokens or usage.output_tokens:
                with suppress(OSError):
                    _remove_staged_usage(staged_path)
            continue
        _persist_target(con, cluster_id, key, payload, usage)
        if usage.input_tokens or usage.output_tokens:
            with suppress(OSError):
                _remove_staged_usage(staged_path)
        stats.labelled += 1
        if stats.labelled % 25 == 0:
            log(f"  {stats.labelled:,} labelled ({family})")
    return stats
