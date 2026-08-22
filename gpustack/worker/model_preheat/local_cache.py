import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from uuid import uuid4

from filelock import SoftFileLock, Timeout

from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    decode_path,
)
from gpustack.worker.model_preheat.manifest import (
    MAX_MANIFEST_BYTES,
    ManifestFile,
    ModelPreheatManifest,
    parse_artifact_manifest,
)


class LocalCacheState(str, Enum):
    VALID = "valid"
    CANDIDATE = "candidate"
    MISSING = "missing"
    CONFLICT = "conflict"
    ERROR = "error"


class LocalCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalCacheInspection:
    state: LocalCacheState
    total_size: int = 0
    manifest: ModelPreheatManifest | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class LocalCachePublishResult:
    state: LocalCacheState
    published: bool
    target_dir: Path
    error_code: str | None = None


def trusted_manifest_path(cache_dir: str | Path, cache_key: str) -> Path:
    _validate_cache_key(cache_key)
    return Path(cache_dir) / ".gpustack-manifests" / f"{cache_key}.json"


def model_lock_path(cache_dir: str | Path, target_dir: str | Path) -> Path:
    cache_root = Path(cache_dir).resolve()
    target = Path(target_dir).resolve()
    _require_descendant(cache_root, target)
    return target.parent / f"{target.name}.lock"


def create_staging_dir(
    cache_dir: str | Path, task_id: int | str, attempt: int | str
) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_root = cache_root.resolve()
    task_component = _safe_component(task_id)
    attempt_component = _safe_component(attempt)
    staging = cache_root / ".preheat" / task_component / attempt_component
    if os.path.lexists(staging):
        if not _is_real_directory(staging):
            raise LocalCacheError("local_cache_staging_conflict")
        _require_descendant(cache_root / ".preheat", staging.resolve())
        _require_same_device(cache_root, staging)
        return staging
    try:
        staging.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise LocalCacheError("local_cache_staging_create_failed") from exc
    _require_same_device(cache_root, staging)
    return staging


def write_trusted_manifest(
    cache_dir: str | Path, cache_key: str, manifest: ModelPreheatManifest
) -> Path:
    path = trusted_manifest_path(cache_dir, cache_key)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with SoftFileLock(f"{path}.lock", timeout=0):
            existing = _read_trusted_manifest(cache_dir, cache_key)
            if existing is not None:
                if existing != manifest:
                    raise LocalCacheError("local_manifest_conflict")
                return path
            with temporary.open("xb") as file:
                file.write(manifest.to_json_bytes())
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
    except Timeout as exc:
        raise LocalCacheError("local_manifest_lock_unavailable") from exc
    except OSError as exc:
        raise LocalCacheError("local_manifest_write_failed") from exc
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
    return path


def replace_trusted_manifest(
    cache_dir: str | Path,
    cache_key: str,
    expected: ModelPreheatManifest,
    replacement: ModelPreheatManifest,
) -> Path:
    path = trusted_manifest_path(cache_dir, cache_key)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with SoftFileLock(f"{path}.lock", timeout=0):
            existing = _read_trusted_manifest(cache_dir, cache_key)
            if existing == replacement:
                return path
            if existing != expected:
                raise LocalCacheError("local_manifest_conflict")
            with temporary.open("xb") as file:
                file.write(replacement.to_json_bytes())
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
    except Timeout as exc:
        raise LocalCacheError("local_manifest_lock_unavailable") from exc
    except OSError as exc:
        raise LocalCacheError("local_manifest_write_failed") from exc
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
    return path


def inspect_local_cache(
    cache_dir: str | Path,
    target_dir: str | Path,
    cache_key: str,
    reference_manifest: ModelPreheatManifest | None = None,
    *,
    cancel_callback=None,
) -> LocalCacheInspection:
    target = Path(target_dir)
    try:
        _require_descendant(Path(cache_dir).resolve(), target.resolve(strict=False))
    except LocalCacheError as exc:
        return LocalCacheInspection(LocalCacheState.ERROR, error_code=str(exc))
    if not os.path.lexists(target):
        return LocalCacheInspection(LocalCacheState.MISSING)
    if not _is_real_directory(target):
        if target.is_symlink() or not target.is_file():
            return LocalCacheInspection(
                LocalCacheState.ERROR, error_code="local_cache_scan_failed"
            )
        return LocalCacheInspection(
            LocalCacheState.CONFLICT, error_code="local_cache_conflict"
        )

    try:
        local_manifest = _read_trusted_manifest(cache_dir, cache_key)
    except LocalCacheError:
        return LocalCacheInspection(
            LocalCacheState.ERROR, error_code="local_manifest_invalid"
        )

    trusted_manifest = reference_manifest or local_manifest
    if trusted_manifest is None:
        return LocalCacheInspection(LocalCacheState.CANDIDATE)
    if (
        reference_manifest is not None
        and local_manifest is not None
        and local_manifest != reference_manifest
    ):
        return LocalCacheInspection(
            LocalCacheState.CONFLICT, error_code="local_cache_conflict"
        )

    try:
        verification = _verify_directory(
            target,
            trusted_manifest,
            allow_extra=True,
            cancel_callback=cancel_callback,
        )
    except (LocalCacheError, OSError):
        return LocalCacheInspection(
            LocalCacheState.ERROR, error_code="local_cache_scan_failed"
        )
    return LocalCacheInspection(
        state=verification.state,
        total_size=(
            trusted_manifest.total_size
            if verification.state == LocalCacheState.VALID
            else 0
        ),
        manifest=(
            trusted_manifest if verification.state == LocalCacheState.VALID else None
        ),
        error_code=verification.error_code,
    )


def publish_staging(
    cache_dir: str | Path,
    target_dir: str | Path,
    cache_key: str,
    staging_dir: str | Path,
    manifest: ModelPreheatManifest,
    *,
    replace_conflicting: bool = False,
    cancel_callback=None,
) -> LocalCachePublishResult:
    target = Path(target_dir)
    staging = Path(staging_dir)
    try:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_root = cache_root.resolve()
        _require_descendant(cache_root, target.resolve(strict=False))
        _require_descendant(cache_root / ".preheat", staging.resolve(strict=False))
        if staging.exists():
            _require_same_device(cache_root, staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        _require_same_device(cache_root, target.parent)
    except LocalCacheError as exc:
        return LocalCachePublishResult(LocalCacheState.ERROR, False, target, str(exc))
    except OSError:
        return LocalCachePublishResult(
            LocalCacheState.ERROR,
            False,
            target,
            "local_cache_publish_failed",
        )

    try:
        with SoftFileLock(str(model_lock_path(cache_root, target)), timeout=0):
            if os.path.lexists(target):
                inspection = inspect_local_cache(
                    cache_root,
                    target,
                    cache_key,
                    manifest,
                    cancel_callback=cancel_callback,
                )
                if inspection.state != LocalCacheState.VALID:
                    if replace_conflicting and inspection.state in {
                        LocalCacheState.CONFLICT,
                        LocalCacheState.MISSING,
                    }:
                        return _replace_conflicting_target(
                            cache_root,
                            target,
                            cache_key,
                            staging,
                            manifest,
                            cancel_callback,
                        )
                    if inspection.state == LocalCacheState.ERROR:
                        return LocalCachePublishResult(
                            LocalCacheState.ERROR,
                            False,
                            target,
                            inspection.error_code or "local_cache_scan_failed",
                        )
                    return LocalCachePublishResult(
                        LocalCacheState.CONFLICT,
                        False,
                        target,
                        "local_cache_conflict",
                    )
                if staging.exists():
                    _run_cancel_callback(cancel_callback)
                    try:
                        shutil.rmtree(staging)
                    except OSError:
                        return LocalCachePublishResult(
                            LocalCacheState.ERROR,
                            False,
                            target,
                            "local_cache_staging_cleanup_failed",
                        )
                _run_cancel_callback(cancel_callback)
                try:
                    write_trusted_manifest(cache_root, cache_key, manifest)
                except LocalCacheError as exc:
                    return _manifest_publish_error(target, str(exc))
                return LocalCachePublishResult(LocalCacheState.VALID, False, target)

            if not _is_real_directory(staging):
                return LocalCachePublishResult(
                    LocalCacheState.ERROR,
                    False,
                    target,
                    "local_cache_staging_missing",
                )
            try:
                staging_verification = _verify_directory(
                    staging, manifest, cancel_callback=cancel_callback
                )
            except (LocalCacheError, OSError):
                staging_verification = LocalCacheInspection(
                    LocalCacheState.ERROR, error_code="local_cache_scan_failed"
                )
            if staging_verification.state != LocalCacheState.VALID:
                return LocalCachePublishResult(
                    LocalCacheState.ERROR,
                    False,
                    target,
                    "local_cache_staging_invalid",
                )
            try:
                _ensure_trusted_manifest_compatible(cache_root, cache_key, manifest)
            except LocalCacheError as exc:
                return _manifest_publish_error(target, str(exc))
            try:
                _run_cancel_callback(cancel_callback)
                os.replace(staging, target)
            except OSError:
                return LocalCachePublishResult(
                    LocalCacheState.ERROR,
                    False,
                    target,
                    "local_cache_publish_failed",
                )
            try:
                _run_cancel_callback(cancel_callback)
            except Exception:
                try:
                    os.replace(target, staging)
                except OSError:
                    return LocalCachePublishResult(
                        LocalCacheState.ERROR,
                        False,
                        target,
                        "local_cache_publish_rollback_failed",
                    )
                raise
            try:
                write_trusted_manifest(cache_root, cache_key, manifest)
            except LocalCacheError as exc:
                return _manifest_publish_error(target, str(exc))
            return LocalCachePublishResult(LocalCacheState.VALID, True, target)
    except Timeout:
        return LocalCachePublishResult(
            LocalCacheState.ERROR,
            False,
            target,
            "local_cache_lock_unavailable",
        )
    except OSError:
        return LocalCachePublishResult(
            LocalCacheState.ERROR,
            False,
            target,
            "local_cache_publish_failed",
        )


def _replace_conflicting_target(
    cache_root: Path,
    target: Path,
    cache_key: str,
    staging: Path,
    manifest: ModelPreheatManifest,
    cancel_callback,
) -> LocalCachePublishResult:
    if not _is_real_directory(staging):
        return LocalCachePublishResult(
            LocalCacheState.ERROR,
            False,
            target,
            "local_cache_staging_missing",
        )
    try:
        verification = _verify_directory(
            staging,
            manifest,
            allow_extra=False,
            cancel_callback=cancel_callback,
        )
    except (LocalCacheError, OSError):
        verification = LocalCacheInspection(
            LocalCacheState.ERROR, error_code="local_cache_scan_failed"
        )
    if verification.state != LocalCacheState.VALID:
        return LocalCachePublishResult(
            LocalCacheState.ERROR,
            False,
            target,
            "local_cache_staging_invalid",
        )

    try:
        source_snapshot = _scan_real_directory(
            Path(os.path.abspath(target)), cancel_callback=cancel_callback
        )
        merged = _merge_unselected_files(
            target,
            staging,
            manifest,
            cancel_callback=cancel_callback,
        )
    except LocalCacheError as exc:
        return LocalCachePublishResult(
            LocalCacheState.ERROR,
            False,
            target,
            str(exc),
        )
    try:
        merged_verification = _verify_directory(
            staging,
            manifest,
            allow_extra=True,
            cancel_callback=cancel_callback,
        )
    except (LocalCacheError, OSError):
        merged_verification = LocalCacheInspection(
            LocalCacheState.ERROR, error_code="local_cache_scan_failed"
        )
    except Exception:
        _cleanup_merged_files(staging, *merged)
        raise
    if merged_verification.state != LocalCacheState.VALID:
        _cleanup_merged_files(staging, *merged)
        return LocalCachePublishResult(
            LocalCacheState.ERROR,
            False,
            target,
            "local_cache_staging_invalid",
        )

    try:
        _run_cancel_callback(cancel_callback)
        current_snapshot = _scan_real_directory(
            Path(os.path.abspath(target)), cancel_callback=cancel_callback
        )
        if not _tree_snapshots_match(source_snapshot, current_snapshot):
            raise LocalCacheError("local_cache_extra_source_changed")
    except LocalCacheError as exc:
        _cleanup_merged_files(staging, *merged)
        return LocalCachePublishResult(
            LocalCacheState.ERROR,
            False,
            target,
            str(exc),
        )
    except Exception:
        _cleanup_merged_files(staging, *merged)
        raise

    backup = target.with_name(f".{target.name}.preheat-backup-{uuid4().hex}")
    target_replaced = False
    try:
        os.replace(target, backup)
        try:
            os.replace(staging, target)
            target_replaced = True
        except OSError:
            os.replace(backup, target)
            _cleanup_merged_files(staging, *merged)
            raise
        try:
            _overwrite_trusted_manifest(cache_root, cache_key, manifest)
        except LocalCacheError as exc:
            try:
                os.replace(target, staging)
                target_replaced = False
                os.replace(backup, target)
                _cleanup_merged_files(staging, *merged)
            except OSError:
                return LocalCachePublishResult(
                    LocalCacheState.ERROR,
                    False,
                    target,
                    "local_cache_publish_rollback_failed",
                )
            return _manifest_publish_error(target, str(exc))
        _remove_replaced_path(backup)
        return LocalCachePublishResult(LocalCacheState.VALID, True, target)
    except OSError:
        if not target_replaced and backup.exists() and not target.exists():
            try:
                os.replace(backup, target)
                _cleanup_merged_files(staging, *merged)
            except OSError:
                return LocalCachePublishResult(
                    LocalCacheState.ERROR,
                    False,
                    target,
                    "local_cache_publish_rollback_failed",
                )
        if staging.exists():
            _cleanup_merged_files(staging, *merged)
        return LocalCachePublishResult(
            LocalCacheState.ERROR,
            False,
            target,
            "local_cache_publish_failed",
        )


def _merge_unselected_files(
    source_dir: Path,
    staging_dir: Path,
    manifest: ModelPreheatManifest,
    *,
    cancel_callback=None,
):
    source_root = Path(os.path.abspath(source_dir))
    staging_root = Path(os.path.abspath(staging_dir))
    if not _is_real_directory(source_root) or not _is_real_directory(staging_root):
        raise LocalCacheError("local_cache_unsafe_extra_file")
    expected_paths = {decode_path(file.path) for file in manifest.files}
    directory_stats = {source_root: source_root.lstat()}
    extra_files = []

    for path in source_root.rglob("*"):
        _run_cancel_callback(cancel_callback)
        source_stat = path.lstat()
        if stat.S_ISLNK(source_stat.st_mode):
            raise LocalCacheError("local_cache_unsafe_extra_file")
        resolved = path.resolve(strict=True)
        _require_descendant(source_root, resolved)
        if stat.S_ISDIR(source_stat.st_mode):
            directory_stats[resolved] = source_stat
            continue
        if not stat.S_ISREG(source_stat.st_mode):
            raise LocalCacheError("local_cache_unsafe_extra_file")
        relative = resolved.relative_to(source_root).as_posix()
        if relative not in expected_paths:
            extra_files.append((relative, resolved, source_stat))

    copied_files = []
    created_directories = []
    try:
        for relative, source, source_stat in sorted(extra_files):
            _run_cancel_callback(cancel_callback)
            destination = staging_root / relative
            _require_descendant(staging_root, destination)
            _ensure_safe_parent_directories(
                staging_root, destination.parent, created_directories
            )
            if os.path.lexists(destination):
                raise LocalCacheError("local_cache_extra_path_conflict")
            try:
                _copy_stable_file(
                    source,
                    source_stat,
                    destination,
                    cancel_callback=cancel_callback,
                )
            except Exception:
                try:
                    destination.unlink()
                except OSError:
                    pass
                raise
            copied_files.append(destination)
        for directory, expected_stat in directory_stats.items():
            _run_cancel_callback(cancel_callback)
            if not _same_file_identity(directory.lstat(), expected_stat):
                raise LocalCacheError("local_cache_extra_source_changed")
    except Exception:
        _cleanup_merged_files(staging_root, copied_files, created_directories)
        raise
    return copied_files, created_directories


def _ensure_safe_parent_directories(root, parent, created_directories):
    relative = parent.relative_to(root)
    current = root
    for component in relative.parts:
        current = current / component
        if os.path.lexists(current):
            current_stat = current.lstat()
            if not stat.S_ISDIR(current_stat.st_mode):
                raise LocalCacheError("local_cache_extra_path_conflict")
            continue
        current.mkdir()
        created_directories.append(current)


def _copy_stable_file(source, expected_stat, destination, *, cancel_callback=None):
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if not _same_file_identity(before, expected_stat):
            raise LocalCacheError("local_cache_extra_source_changed")
        with (
            os.fdopen(descriptor, "rb", closefd=False) as input_file,
            destination.open("xb") as output_file,
        ):
            while True:
                _run_cancel_callback(cancel_callback)
                chunk = input_file.read(1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)
            output_file.flush()
        after = os.fstat(descriptor)
        if not _same_file_identity(after, before) or not _same_file_identity(
            source.lstat(), before
        ):
            raise LocalCacheError("local_cache_extra_source_changed")
    finally:
        os.close(descriptor)


def _same_file_identity(first, second):
    return all(
        getattr(first, field) == getattr(second, field)
        for field in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


def _run_cancel_callback(cancel_callback):
    if cancel_callback is not None:
        cancel_callback()


def _cleanup_merged_files(staging_root, copied_files, created_directories):
    del staging_root
    for path in reversed(copied_files):
        try:
            path.unlink()
        except OSError:
            pass
    for path in reversed(created_directories):
        try:
            path.rmdir()
        except OSError:
            pass


def _overwrite_trusted_manifest(
    cache_dir: str | Path, cache_key: str, manifest: ModelPreheatManifest
) -> Path:
    path = trusted_manifest_path(cache_dir, cache_key)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with SoftFileLock(f"{path}.lock", timeout=0):
            with temporary.open("xb") as file:
                file.write(manifest.to_json_bytes())
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
    except Timeout as exc:
        raise LocalCacheError("local_manifest_lock_unavailable") from exc
    except OSError as exc:
        raise LocalCacheError("local_manifest_write_failed") from exc
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
    return path


def _remove_replaced_path(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError:
        pass


def _ensure_trusted_manifest_compatible(
    cache_dir: str | Path, cache_key: str, manifest: ModelPreheatManifest
):
    existing = _read_trusted_manifest(cache_dir, cache_key)
    if existing is not None and existing != manifest:
        raise LocalCacheError("local_manifest_conflict")


def _manifest_publish_error(target: Path, error_code: str) -> LocalCachePublishResult:
    return LocalCachePublishResult(
        (
            LocalCacheState.CONFLICT
            if error_code == "local_manifest_conflict"
            else LocalCacheState.ERROR
        ),
        False,
        target,
        error_code,
    )


def _read_trusted_manifest(
    cache_dir: str | Path, cache_key: str
) -> ModelPreheatManifest | None:
    path = trusted_manifest_path(cache_dir, cache_key)
    if not path.exists():
        return None
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise LocalCacheError("local_manifest_invalid")
        with path.open("rb") as file:
            raw = file.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise LocalCacheError("local_manifest_invalid")
        payload = json.loads(raw.decode("utf-8"))
        return _manifest_from_payload(payload)
    except (
        KeyError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        raise LocalCacheError("local_manifest_invalid") from exc


def _manifest_from_payload(payload: dict) -> ModelPreheatManifest:
    return parse_artifact_manifest(payload)


def _verify_directory(
    root_dir: Path,
    manifest: ModelPreheatManifest,
    *,
    allow_extra: bool = False,
    cancel_callback=None,
) -> LocalCacheInspection:
    try:
        root = Path(os.path.abspath(root_dir))
        actual_paths, directory_stats, file_stats = _scan_real_directory(
            root, cancel_callback=cancel_callback
        )
    except (LocalCacheError, OSError):
        return LocalCacheInspection(
            LocalCacheState.ERROR, error_code="local_cache_scan_failed"
        )
    expected_paths = set()
    for file in manifest.files:
        path = _manifest_file_path(root, file)
        relative_path = path.relative_to(root).as_posix()
        expected_paths.add(relative_path)
        expected_stat = file_stats.get(relative_path)
        if expected_stat is None:
            return LocalCacheInspection(LocalCacheState.MISSING)
        if (
            expected_stat.st_size != file.size
            or _sha256_file(path, expected_stat, cancel_callback=cancel_callback)
            != file.sha256
        ):
            return LocalCacheInspection(
                LocalCacheState.CONFLICT, error_code="local_cache_conflict"
            )

    try:
        final_paths, final_directories, final_files = _scan_real_directory(
            root, cancel_callback=cancel_callback
        )
        if (
            final_paths != actual_paths
            or not _identities_match(directory_stats, final_directories)
            or not _identities_match(file_stats, final_files)
        ):
            raise LocalCacheError("local_cache_source_changed")
    except (LocalCacheError, OSError):
        return LocalCacheInspection(
            LocalCacheState.ERROR, error_code="local_cache_scan_failed"
        )

    paths_match = (
        expected_paths.issubset(actual_paths)
        if allow_extra
        else actual_paths == expected_paths
    )
    if not paths_match:
        return LocalCacheInspection(
            LocalCacheState.CONFLICT, error_code="local_cache_conflict"
        )
    return LocalCacheInspection(LocalCacheState.VALID)


def _manifest_file_path(root: Path, file: ManifestFile) -> Path:
    path = Path(os.path.abspath(root / decode_path(file.path)))
    _require_descendant(root, path)
    return path


def _scan_real_directory(root: Path, *, cancel_callback=None):
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise LocalCacheError("local_cache_unsafe_root")

    actual_paths = set()
    directory_stats = {"": root_stat}
    file_stats = {}
    for path in root.rglob("*"):
        _run_cancel_callback(cancel_callback)
        path_stat = path.lstat()
        relative_path = path.relative_to(root).as_posix()
        if stat.S_ISDIR(path_stat.st_mode):
            directory_stats[relative_path] = path_stat
            continue
        if not stat.S_ISREG(path_stat.st_mode):
            raise LocalCacheError("local_cache_unsafe_file")
        actual_paths.add(relative_path)
        file_stats[relative_path] = path_stat
    return actual_paths, directory_stats, file_stats


def _is_real_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def _identities_match(before, after) -> bool:
    return before.keys() == after.keys() and all(
        _same_file_identity(expected, after[path]) for path, expected in before.items()
    )


def _tree_snapshots_match(before, after) -> bool:
    before_paths, before_directories, before_files = before
    after_paths, after_directories, after_files = after
    return (
        before_paths == after_paths
        and _identities_match(before_directories, after_directories)
        and _identities_match(before_files, after_files)
    )


def _sha256_file(path: Path, expected_stat, *, cancel_callback=None) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not _same_file_identity(
            before, expected_stat
        ):
            raise LocalCacheError("local_cache_source_changed")
        with os.fdopen(descriptor, "rb", closefd=False) as file:
            while True:
                _run_cancel_callback(cancel_callback)
                chunk = file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        after = os.fstat(descriptor)
        current = path.lstat()
        if not _same_file_identity(before, after) or not _same_file_identity(
            current, before
        ):
            raise LocalCacheError("local_cache_source_changed")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _require_same_device(first: Path, second: Path):
    if os.stat(first).st_dev != os.stat(second).st_dev:
        raise LocalCacheError("local_cache_staging_cross_device")


def _require_descendant(root: Path, path: Path):
    if root != path and root not in path.parents:
        raise LocalCacheError("local_cache_path_escape")


def _safe_component(value: int | str) -> str:
    component = str(value)
    if component in {"", ".", ".."} or "/" in component or "\\" in component:
        raise LocalCacheError("local_cache_invalid_staging_component")
    return component


def _validate_cache_key(cache_key: str):
    if (
        not isinstance(cache_key, str)
        or cache_key in {"", ".", ".."}
        or "/" in cache_key
        or "\\" in cache_key
    ):
        raise LocalCacheError("local_cache_invalid_cache_key")
