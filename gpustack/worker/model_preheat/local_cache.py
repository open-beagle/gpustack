import hashlib
import json
import os
import shutil
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
    if staging.exists():
        if not staging.is_dir():
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
) -> LocalCacheInspection:
    target = Path(target_dir)
    try:
        _require_descendant(Path(cache_dir).resolve(), target.resolve(strict=False))
    except LocalCacheError as exc:
        return LocalCacheInspection(LocalCacheState.ERROR, error_code=str(exc))
    if not target.exists():
        return LocalCacheInspection(LocalCacheState.MISSING)
    if not target.is_dir():
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
        verification = _verify_directory(target, trusted_manifest)
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
            if target.exists():
                inspection = inspect_local_cache(
                    cache_root, target, cache_key, manifest
                )
                if inspection.state != LocalCacheState.VALID:
                    if (
                        replace_conflicting
                        and inspection.state == LocalCacheState.CONFLICT
                    ):
                        return _replace_conflicting_target(
                            cache_root,
                            target,
                            cache_key,
                            staging,
                            manifest,
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
                    try:
                        shutil.rmtree(staging)
                    except OSError:
                        return LocalCachePublishResult(
                            LocalCacheState.ERROR,
                            False,
                            target,
                            "local_cache_staging_cleanup_failed",
                        )
                try:
                    write_trusted_manifest(cache_root, cache_key, manifest)
                except LocalCacheError as exc:
                    return _manifest_publish_error(target, str(exc))
                return LocalCachePublishResult(LocalCacheState.VALID, False, target)

            if not staging.is_dir():
                return LocalCachePublishResult(
                    LocalCacheState.ERROR,
                    False,
                    target,
                    "local_cache_staging_missing",
                )
            try:
                staging_verification = _verify_directory(staging, manifest)
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
                os.replace(staging, target)
            except OSError:
                return LocalCachePublishResult(
                    LocalCacheState.ERROR,
                    False,
                    target,
                    "local_cache_publish_failed",
                )
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
) -> LocalCachePublishResult:
    if not staging.is_dir():
        return LocalCachePublishResult(
            LocalCacheState.ERROR,
            False,
            target,
            "local_cache_staging_missing",
        )
    try:
        verification = _verify_directory(staging, manifest)
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

    backup = target.with_name(f".{target.name}.preheat-backup-{uuid4().hex}")
    target_replaced = False
    try:
        os.replace(target, backup)
        try:
            os.replace(staging, target)
            target_replaced = True
        except OSError:
            os.replace(backup, target)
            raise
        try:
            _overwrite_trusted_manifest(cache_root, cache_key, manifest)
        except LocalCacheError as exc:
            try:
                os.replace(target, staging)
                target_replaced = False
                os.replace(backup, target)
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
            except OSError:
                return LocalCachePublishResult(
                    LocalCacheState.ERROR,
                    False,
                    target,
                    "local_cache_publish_rollback_failed",
                )
        return LocalCachePublishResult(
            LocalCacheState.ERROR,
            False,
            target,
            "local_cache_publish_failed",
        )


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
    if not isinstance(payload, dict) or not isinstance(payload.get("identity"), dict):
        raise ValueError("invalid_manifest")
    identity_payload = payload["identity"]
    identity = ModelPreheatIdentity(
        source=identity_payload["source"],
        model_id=decode_path(identity_payload["model_id"]),
        revision=decode_path(identity_payload["revision"]),
        file_patterns=tuple(
            decode_path(pattern) for pattern in identity_payload["file_patterns"]
        ),
    )
    files = tuple(
        ManifestFile(path=file["path"], size=file["size"], sha256=file["sha256"])
        for file in payload["files"]
    )
    manifest = ModelPreheatManifest(
        identity=identity,
        files=files,
        cache_key=payload["cache_key"],
        selection_digest=payload["selection_digest"],
        generation_id=payload["generation_id"],
        exclude_patterns=tuple(
            decode_path(pattern) for pattern in payload.get("exclude_patterns", [])
        ),
        requested_revision=decode_path(
            payload.get("requested_revision", identity.revision_path)
        ),
        schema_version=payload.get("schema_version", 1),
    )
    if payload != manifest.to_dict():
        raise ValueError("invalid_manifest")
    return manifest


def _verify_directory(
    root_dir: Path, manifest: ModelPreheatManifest
) -> LocalCacheInspection:
    root = root_dir.resolve()
    expected_paths = set()
    for file in manifest.files:
        path = _manifest_file_path(root, file)
        expected_paths.add(path.relative_to(root).as_posix())
        if not path.exists() or not path.is_file():
            return LocalCacheInspection(LocalCacheState.MISSING)
        if path.stat().st_size != file.size or _sha256_file(path) != file.sha256:
            return LocalCacheInspection(
                LocalCacheState.CONFLICT, error_code="local_cache_conflict"
            )

    try:
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
    except OSError:
        return LocalCacheInspection(
            LocalCacheState.ERROR, error_code="local_cache_scan_failed"
        )
    if actual_paths != expected_paths:
        return LocalCacheInspection(
            LocalCacheState.CONFLICT, error_code="local_cache_conflict"
        )
    return LocalCacheInspection(LocalCacheState.VALID)


def _manifest_file_path(root: Path, file: ManifestFile) -> Path:
    path = (root / decode_path(file.path)).resolve()
    _require_descendant(root, path)
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
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
