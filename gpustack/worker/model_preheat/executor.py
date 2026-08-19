import hashlib
import io
import os
import secrets
import shutil
import socket
import ssl
import stat
import time
import asyncio
import threading
from dataclasses import dataclass
from glob import has_magic
from pathlib import Path
from urllib.parse import urlparse

from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    decode_path,
    encode_path,
)
from gpustack.worker.model_preheat.local_cache import (
    LocalCacheError,
    LocalCacheState,
    create_staging_dir,
    inspect_local_cache,
    publish_staging,
    replace_trusted_manifest,
)
from gpustack.worker.model_preheat.manifest import build_model_preheat_manifest
from gpustack.worker.model_preheat.s3_client import (
    ModelPreheatCanceled,
    ModelPreheatS3Client,
    ModelPreheatS3Conflict,
    ModelPreheatS3ManifestError,
    ReadyGenerationConflict,
)


@dataclass(frozen=True)
class TrustedLocalCandidate:
    source: str
    root: Path
    paths: tuple[Path, ...]
    repository_complete: bool = False


@dataclass(frozen=True)
class SeedExecutionRequest:
    cache_dir: Path
    target_dir: Path
    cache_key: str
    task_id: int
    attempt: int
    identity: ModelPreheatIdentity
    selection_digest: str
    generation_id: str
    exclude_patterns: tuple[str, ...] | list[str]
    bucket: str
    prefix: str
    requested_revision: str | None = None
    bandwidth_limit_mbps: int | None = None
    resumable_cursor: dict | None = None
    trusted_local_candidate: TrustedLocalCandidate | None = None


@dataclass(frozen=True)
class TargetExecutionRequest:
    cache_dir: Path
    target_dir: Path
    cache_key: str
    task_id: int
    attempt: int
    identity: ModelPreheatIdentity
    selection_digest: str
    generation_id: str
    exclude_patterns: tuple[str, ...] | list[str]
    bucket: str
    prefix: str
    requested_revision: str | None = None
    bandwidth_limit_mbps: int | None = None
    resumable_cursor: dict | None = None
    trusted_local_candidate: TrustedLocalCandidate | None = None


def execute_seed_preheat(
    request: SeedExecutionRequest,
    s3_client,
    *,
    download_to_staging=None,
    cancel_check=None,
    progress_callback=None,
) -> dict:
    staging = None
    try:
        reference_manifest = s3_client.read_ready_manifest(
            request.bucket,
            request.prefix,
            request.identity,
            cache_key=request.cache_key,
            selection_digest=request.selection_digest,
        )
        inspection = inspect_local_cache(
            request.cache_dir,
            request.target_dir,
            request.cache_key,
            reference_manifest,
        )
        if inspection.state == LocalCacheState.VALID:
            manifest = inspection.manifest
            replaced_local_manifest = None
            if (
                reference_manifest is None
                and manifest.generation_id != request.generation_id
            ):
                replaced_local_manifest = manifest
                manifest = build_model_preheat_manifest(
                    request.target_dir,
                    request.identity,
                    cache_key=request.cache_key,
                    selection_digest=request.selection_digest,
                    generation_id=request.generation_id,
                    exclude_patterns=request.exclude_patterns,
                    requested_revision=request.requested_revision,
                )
            publish_result = s3_client.publish_generation(
                request.bucket,
                request.prefix,
                manifest,
                request.target_dir,
                cancel_check=cancel_check,
                bandwidth_limit_mbps=request.bandwidth_limit_mbps,
            )
            if replaced_local_manifest is not None:
                replace_trusted_manifest(
                    request.cache_dir,
                    request.cache_key,
                    replaced_local_manifest,
                    manifest,
                )
            return _ready_result(
                request,
                s3_client,
                manifest,
                inspection.state,
                publish_result.uploaded,
                publish_result.skipped,
                0,
            )

        if reference_manifest is not None:
            return execute_target_preheat(
                TargetExecutionRequest(
                    cache_dir=request.cache_dir,
                    target_dir=request.target_dir,
                    cache_key=request.cache_key,
                    task_id=request.task_id,
                    attempt=request.attempt,
                    identity=request.identity,
                    requested_revision=request.requested_revision,
                    selection_digest=request.selection_digest,
                    generation_id=request.generation_id,
                    exclude_patterns=request.exclude_patterns,
                    bucket=request.bucket,
                    prefix=request.prefix,
                    bandwidth_limit_mbps=request.bandwidth_limit_mbps,
                    resumable_cursor=request.resumable_cursor,
                    trusted_local_candidate=request.trusted_local_candidate,
                ),
                s3_client,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )

        staging = create_staging_dir(
            request.cache_dir, request.task_id, request.attempt
        )
        trusted_candidate_staged = _stage_trusted_local_candidate(
            request.trusted_local_candidate,
            staging,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )
        trusted_candidate_manifest = None
        if trusted_candidate_staged:
            trusted_candidate_manifest = _build_trusted_candidate_manifest(
                request,
                staging,
                request.generation_id,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
            if trusted_candidate_manifest is None:
                _discard_staging(staging)
                staging.mkdir(parents=True, exist_ok=True)
                trusted_candidate_staged = False
            else:
                _retain_manifest_files(
                    staging, trusted_candidate_manifest, cancel_check=cancel_check
                )
        if not trusted_candidate_staged:
            _reuse_previous_seed_staging(request, staging, cancel_check=cancel_check)
        _raise_if_cancelled(cancel_check)
        if not trusted_candidate_staged:
            if download_to_staging is None:
                from gpustack.worker.downloaders import (
                    download_resolved_revision_to_staging,
                )

                download_to_staging = download_resolved_revision_to_staging
            download_options = {"exclude_patterns": request.exclude_patterns}
            if download_to_staging.__module__ == "gpustack.worker.downloaders":
                download_options.update(
                    cancel_check=cancel_check,
                    progress_callback=progress_callback,
                )
            download_to_staging(request.identity, staging, **download_options)
        _raise_if_cancelled(cancel_check)
        _remove_excluded_files(
            staging, request.exclude_patterns, cancel_check=cancel_check
        )
        manifest = trusted_candidate_manifest or build_model_preheat_manifest(
            staging,
            request.identity,
            cache_key=request.cache_key,
            selection_digest=request.selection_digest,
            generation_id=request.generation_id,
            exclude_patterns=request.exclude_patterns,
            requested_revision=request.requested_revision,
            cancel_callback=lambda: _raise_if_cancelled(cancel_check),
            progress_callback=progress_callback,
        )
        if reference_manifest is not None and manifest != reference_manifest:
            return _error_result("checksum_mismatch", staging)
        publish_result = s3_client.publish_generation(
            request.bucket,
            request.prefix,
            manifest,
            staging,
            cancel_check=cancel_check,
            bandwidth_limit_mbps=request.bandwidth_limit_mbps,
        )
        local_result = publish_staging(
            request.cache_dir,
            request.target_dir,
            request.cache_key,
            staging,
            manifest,
        )
        if local_result.state != LocalCacheState.VALID:
            return _error_result(
                local_result.error_code or "local_cache_conflict",
                staging,
                local_cache_state=local_result.state,
            )
        return _ready_result(
            request,
            s3_client,
            manifest,
            local_result.state,
            publish_result.uploaded,
            publish_result.skipped,
            0 if trusted_candidate_staged else len(manifest.files),
        )
    except ModelPreheatCanceled:
        return _error_result("canceled", staging)
    except ModelPreheatS3ManifestError:
        return _error_result("s3_manifest_invalid", staging)
    except ReadyGenerationConflict:
        return _error_result("ready_generation_conflict", staging)
    except ModelPreheatS3Conflict:
        return _error_result("s3_object_conflict", staging)
    except LocalCacheError as exc:
        return _error_result(str(exc), staging)
    except OSError:
        return _error_result("worker_execution_failed", staging)
    except ValueError:
        return _error_result("validation_error", staging)
    except Exception:
        return _error_result("worker_execution_failed", staging)


def execute_target_preheat(
    request: TargetExecutionRequest,
    s3_client,
    *,
    cancel_check=None,
    progress_callback=None,
) -> dict:
    staging = None
    completed_files = []
    try:
        manifest = s3_client.read_ready_manifest(
            request.bucket,
            request.prefix,
            request.identity,
            cache_key=request.cache_key,
            selection_digest=request.selection_digest,
        )
        if manifest is None:
            return _error_result("s3_ready_not_found")
        trusted_staging = create_staging_dir(
            request.cache_dir, request.task_id, request.attempt
        )
        if _stage_trusted_local_candidate(
            request.trusted_local_candidate,
            trusted_staging,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        ):
            candidate_manifest = _build_trusted_candidate_manifest(
                request,
                trusted_staging,
                manifest.generation_id,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
            if candidate_manifest is not None:
                _retain_manifest_files(
                    trusted_staging, candidate_manifest, cancel_check=cancel_check
                )
            if candidate_manifest == manifest:
                local_result = publish_staging(
                    request.cache_dir,
                    request.target_dir,
                    request.cache_key,
                    trusted_staging,
                    manifest,
                    replace_conflicting=True,
                    cancel_callback=lambda: _raise_if_cancelled(cancel_check),
                )
                if local_result.state == LocalCacheState.VALID:
                    return _ready_result(
                        request,
                        s3_client,
                        manifest,
                        local_result.state,
                        0,
                        len(manifest.files),
                        0,
                    )
            _discard_staging(trusted_staging)
        inspection = inspect_local_cache(
            request.cache_dir,
            request.target_dir,
            request.cache_key,
            manifest,
        )
        if inspection.state == LocalCacheState.VALID:
            return _ready_result(
                request,
                s3_client,
                manifest,
                inspection.state,
                0,
                len(manifest.files),
                0,
            )

        staging = create_staging_dir(
            request.cache_dir, request.task_id, request.attempt
        )
        _reuse_previous_staging_files(
            request, staging, manifest, cancel_check=cancel_check
        )
        downloaded = 0
        skipped = 0
        downloaded_size = 0
        for file in manifest.files:
            _raise_if_cancelled(cancel_check)
            target = _staging_manifest_path(staging, file.path)
            if _file_matches(target, file.size, file.sha256, cancel_check):
                skipped += 1
                completed_files.append(file.path)
                downloaded_size += file.size
                _report_progress(
                    progress_callback,
                    completed_files,
                    downloaded_size,
                    manifest.total_size,
                )
                continue
            for _ in range(3):
                _raise_if_cancelled(cancel_check)
                download_options = {}
                if request.bandwidth_limit_mbps is not None:
                    download_options["bandwidth_limit_mbps"] = (
                        request.bandwidth_limit_mbps
                    )
                s3_client.download_generation_file(
                    request.bucket,
                    request.prefix,
                    manifest,
                    file,
                    target,
                    **download_options,
                )
                if _file_matches(target, file.size, file.sha256, cancel_check):
                    downloaded += 1
                    completed_files.append(file.path)
                    downloaded_size += file.size
                    _report_progress(
                        progress_callback,
                        completed_files,
                        downloaded_size,
                        manifest.total_size,
                    )
                    break
            else:
                return _error_result(
                    "checksum_mismatch",
                    staging,
                    completed_files,
                )

        _raise_if_cancelled(cancel_check)
        local_result = publish_staging(
            request.cache_dir,
            request.target_dir,
            request.cache_key,
            staging,
            manifest,
            replace_conflicting=True,
            cancel_callback=lambda: _raise_if_cancelled(cancel_check),
        )
        if local_result.state != LocalCacheState.VALID:
            return _error_result(
                local_result.error_code or "local_cache_conflict",
                staging,
                completed_files,
                local_result.state,
            )
        return _ready_result(
            request,
            s3_client,
            manifest,
            local_result.state,
            0,
            skipped,
            downloaded,
        )
    except ModelPreheatCanceled:
        return _error_result("canceled", staging, completed_files)
    except ModelPreheatS3ManifestError:
        return _error_result("s3_manifest_invalid", staging, completed_files)
    except ModelPreheatS3Conflict:
        return _error_result("s3_object_conflict", staging, completed_files)
    except LocalCacheError as exc:
        return _error_result(str(exc), staging, completed_files)
    except OSError:
        return _error_result("worker_execution_failed", staging, completed_files)
    except ValueError:
        return _error_result("validation_error", staging, completed_files)
    except Exception:
        return _error_result("worker_execution_failed", staging, completed_files)


def _report_progress(callback, completed_files, downloaded_size, total_size):
    if callback is not None:
        callback(tuple(completed_files), downloaded_size, total_size)


def _stage_trusted_local_candidate(
    candidate,
    staging: Path,
    *,
    cancel_check=None,
    progress_callback=None,
) -> bool:
    if candidate is None:
        return False
    try:
        root_input = Path(candidate.root)
        if not root_input.is_absolute() or root_input.is_symlink():
            return False
        root = root_input.resolve(strict=True)
        if not root.is_dir():
            return False
        root_stat = root.stat()
        files = {}
        for raw_path in candidate.paths:
            _raise_if_cancelled(cancel_check)
            path_input = Path(raw_path)
            if not path_input.is_absolute() or path_input.is_symlink():
                return False
            path = path_input.resolve(strict=True)
            if path != root and root not in path.parents:
                return False
            candidates = path.rglob("*") if path.is_dir() else (path,)
            for source in candidates:
                _raise_if_cancelled(cancel_check)
                source_stat = source.lstat()
                if stat.S_ISLNK(source_stat.st_mode):
                    return False
                if stat.S_ISDIR(source_stat.st_mode):
                    continue
                if not stat.S_ISREG(source_stat.st_mode):
                    return False
                resolved = source.resolve(strict=True)
                if root not in resolved.parents:
                    return False
                files[resolved.relative_to(root)] = (resolved, source_stat)
        if not files:
            return False
        _discard_staging(staging)
        staging.mkdir(parents=True, exist_ok=False)
        staging_root = staging.resolve()
        completed_files = []
        copied_size = 0
        total_size = sum(source_stat.st_size for _, source_stat in files.values())
        for relative, (source, source_stat) in sorted(
            files.items(), key=lambda item: str(item[0])
        ):
            _raise_if_cancelled(cancel_check)
            target = (staging_root / relative).resolve(strict=False)
            if staging_root not in target.parents:
                raise LocalCacheError("local_cache_path_escape")
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_stable_snapshot(source, source_stat, target, cancel_check)
            completed_files.append(relative.as_posix())
            copied_size += source_stat.st_size
            _report_progress(
                progress_callback, completed_files, copied_size, total_size
            )
        if not _same_file_identity(root.stat(), root_stat):
            raise LocalCacheError("trusted_local_candidate_changed")
        return True
    except (LocalCacheError, OSError, ValueError):
        _discard_staging(staging)
        staging.mkdir(parents=True, exist_ok=True)
        return False


def _discard_staging(staging: Path) -> None:
    if staging.exists():
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        else:
            staging.unlink()


def _copy_stable_snapshot(source, expected_stat, target, cancel_check):
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if not _same_file_identity(before, expected_stat):
            raise LocalCacheError("trusted_local_candidate_changed")
        with (
            os.fdopen(descriptor, "rb", closefd=False) as input_file,
            target.open("xb") as output_file,
        ):
            while True:
                _raise_if_cancelled(cancel_check)
                chunk = input_file.read(1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)
            output_file.flush()
        after = os.fstat(descriptor)
        path_after = source.lstat()
        if not _same_file_identity(after, before) or not _same_file_identity(
            path_after, before
        ):
            raise LocalCacheError("trusted_local_candidate_changed")
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


def _build_trusted_candidate_manifest(
    request,
    staging,
    generation_id,
    *,
    cancel_check=None,
    progress_callback=None,
):
    try:
        manifest = build_model_preheat_manifest(
            staging,
            request.identity,
            cache_key=request.cache_key,
            selection_digest=request.selection_digest,
            generation_id=generation_id,
            exclude_patterns=request.exclude_patterns,
            requested_revision=request.requested_revision,
            cancel_callback=lambda: _raise_if_cancelled(cancel_check),
            progress_callback=progress_callback,
        )
        if not _trusted_selection_complete(
            manifest, request.identity, request.trusted_local_candidate
        ):
            return None
        return manifest
    except (OSError, ValueError):
        return None


def _trusted_selection_complete(manifest, identity, candidate) -> bool:
    selected_paths = [decode_path(file.path) for file in manifest.files]
    if not selected_paths:
        return False
    if candidate is not None and candidate.repository_complete:
        return True
    patterns = [decode_path(pattern) for pattern in identity.file_patterns]
    if not patterns or any(has_magic(pattern) for pattern in patterns):
        return False
    return all(pattern in selected_paths for pattern in patterns)


def _retain_manifest_files(staging: Path, manifest, *, cancel_check=None) -> None:
    expected = {decode_path(file.path) for file in manifest.files}
    root = staging.resolve()
    for path in sorted(root.rglob("*"), reverse=True):
        _raise_if_cancelled(cancel_check)
        if path.is_symlink():
            raise LocalCacheError("local_cache_path_escape")
        if path.is_file() and path.relative_to(root).as_posix() not in expected:
            path.unlink()
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def _ready_result(
    request,
    s3_client,
    manifest,
    local_cache_state,
    uploaded,
    skipped,
    downloaded,
) -> dict:
    return {
        "state": "ready",
        "manifest_digest": manifest.digest,
        "ready_path": s3_client.ready_object(request.prefix, manifest),
        "manifest_path": s3_client.manifest_object(request.prefix, manifest),
        "generation_id": manifest.generation_id,
        "local_cache_state": local_cache_state.value,
        "uploaded": uploaded,
        "skipped": skipped,
        "downloaded": downloaded,
        "total_size": manifest.total_size,
    }


def _error_result(
    error_code,
    staging_dir=None,
    completed_files=None,
    local_cache_state=LocalCacheState.ERROR,
) -> dict:
    return {
        "state": "error",
        "error_code": error_code,
        "local_cache_state": local_cache_state.value,
        "cursor": {
            "completed_files": list(completed_files or [])[:1024],
            "staging_exists": bool(staging_dir and Path(staging_dir).exists()),
        },
    }


def _raise_if_cancelled(cancel_check):
    if cancel_check is not None and cancel_check():
        raise ModelPreheatCanceled("canceled")


def _staging_manifest_path(staging_dir: Path, encoded_path: str) -> Path:
    root = staging_dir.resolve()
    path = (root / decode_path(encoded_path)).resolve()
    if root not in path.parents:
        raise LocalCacheError("local_cache_path_escape")
    return path


def _file_matches(
    path: Path, size: int, expected_sha256: str, cancel_check=None
) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            _raise_if_cancelled(cancel_check)
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def _reuse_previous_staging_files(
    request, staging: Path, manifest, *, cancel_check=None
) -> None:
    task_root = Path(request.cache_dir) / ".preheat" / str(request.task_id)
    if not task_root.is_dir():
        return
    try:
        current_attempt = int(request.attempt)
    except (TypeError, ValueError):
        return
    previous = sorted(
        (
            path
            for path in task_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and path.name.isdigit()
            and int(path.name) < current_attempt
        ),
        key=lambda path: int(path.name),
        reverse=True,
    )
    cursor_files = _cursor_completed_files(request)
    for file in manifest.files:
        _raise_if_cancelled(cancel_check)
        if cursor_files is not None and file.path not in cursor_files:
            continue
        target = _staging_manifest_path(staging, file.path)
        if _file_matches(target, file.size, file.sha256, cancel_check):
            continue
        for old_staging in previous:
            try:
                source = _staging_manifest_path(old_staging, file.path)
                matches = _file_matches(source, file.size, file.sha256, cancel_check)
            except (LocalCacheError, OSError):
                continue
            if not matches:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                _copy_stable_snapshot(source, source.lstat(), target, cancel_check)
            except (LocalCacheError, OSError):
                target.unlink(missing_ok=True)
                continue
            break


def _reuse_previous_seed_staging(request, staging: Path, *, cancel_check=None) -> None:
    task_root = Path(request.cache_dir) / ".preheat" / str(request.task_id)
    try:
        current_attempt = int(request.attempt)
    except (TypeError, ValueError):
        return
    if not task_root.is_dir():
        return
    previous = sorted(
        (
            path
            for path in task_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and path.name.isdigit()
            and int(path.name) < current_attempt
        ),
        key=lambda path: int(path.name),
        reverse=True,
    )
    if not previous:
        return
    resolved_task_root = task_root.resolve()
    source_root = previous[0].resolve()
    if resolved_task_root not in source_root.parents:
        return
    target_root = staging.resolve()
    cursor_files = _cursor_completed_files(request)
    for source in source_root.rglob("*"):
        _raise_if_cancelled(cancel_check)
        try:
            source_stat = source.lstat()
        except OSError:
            continue
        if not stat.S_ISREG(source_stat.st_mode) or source.is_symlink():
            continue
        resolved_source = source.resolve()
        if source_root not in resolved_source.parents:
            continue
        relative = source.relative_to(source_root)
        if (
            cursor_files is not None
            and encode_path(relative.as_posix()) not in cursor_files
        ):
            continue
        target = (target_root / relative).resolve(strict=False)
        if target_root not in target.parents or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            _copy_stable_snapshot(resolved_source, source_stat, target, cancel_check)
        except (LocalCacheError, OSError):
            target.unlink(missing_ok=True)
            continue


def _cursor_completed_files(request):
    cursor = getattr(request, "resumable_cursor", None)
    if cursor is None:
        return None
    return set(cursor.get("completed_files", []))


def _remove_excluded_files(
    staging_dir: Path, exclude_patterns, *, cancel_check=None
) -> None:
    root = staging_dir.resolve()
    decoded_patterns = tuple(decode_path(pattern) for pattern in exclude_patterns)
    for path in root.rglob("*"):
        _raise_if_cancelled(cancel_check)
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if any(Path(relative_path).match(pattern) for pattern in decoded_patterns):
            path.unlink()


def build_preheat_role_handlers(cache_dir: str | Path) -> dict:
    async def seed_handler(payload, context):
        return await _execute_payload(payload, context, cache_dir, seed=True)

    async def target_handler(payload, context):
        return await _execute_payload(payload, context, cache_dir, seed=False)

    return {"seed": seed_handler, "distribute": target_handler}


async def _execute_payload(
    payload, context, cache_dir: str | Path, *, seed: bool
) -> dict:
    from gpustack.worker.downloaders import preheat_model_target_dir

    task = payload.task
    patterns = [decode_path(pattern) for pattern in task.get("include_patterns") or []]
    exclude_patterns = [
        decode_path(pattern) for pattern in (task.get("exclude_patterns") or [])
    ]
    identity = ModelPreheatIdentity(
        source=task["source"],
        model_id=task["model_id"],
        revision=task["resolved_revision"],
        file_patterns=patterns,
    )
    profile = payload.profile
    endpoint = urlparse(profile.endpoint)
    client = ModelPreheatS3Client.from_minio(
        endpoint=profile.endpoint,
        access_key=profile.access_key,
        secret_key=profile.secret_key,
        secure=endpoint.scheme == "https" and profile.tls_enabled,
        tls_verify=profile.tls_verify,
        region=profile.region or None,
        use_virtual_hosted_style=profile.use_virtual_hosted_style,
    )
    request_fields = {
        "cache_dir": Path(cache_dir),
        "target_dir": preheat_model_target_dir(cache_dir, identity),
        "cache_key": task["cache_key"],
        "task_id": task.get("id", payload.worker_task_id),
        "attempt": payload.attempt,
        "identity": identity,
        "requested_revision": task.get("requested_revision"),
        "selection_digest": task["selection_digest"],
        "generation_id": task["generation_id"],
        "exclude_patterns": exclude_patterns,
        "bucket": profile.bucket,
        "prefix": profile.prefix,
        "bandwidth_limit_mbps": task.get("bandwidth_limit_mbps"),
        "resumable_cursor": getattr(payload, "resumable_cursor", None),
        "trusted_local_candidate": (
            TrustedLocalCandidate(
                source=payload.trusted_local_candidate.source,
                root=Path(payload.trusted_local_candidate.root),
                paths=tuple(
                    Path(path) for path in payload.trusted_local_candidate.paths
                ),
                repository_complete=payload.trusted_local_candidate.repository_complete,
            )
            if getattr(payload, "trusted_local_candidate", None) is not None
            else None
        ),
    }
    cancel_event = threading.Event()
    execution = execute_seed_preheat if seed else execute_target_preheat
    request_type = SeedExecutionRequest if seed else TargetExecutionRequest
    loop = asyncio.get_running_loop()

    def report_progress(completed_files, downloaded_size, total_size):
        progress = min(
            99,
            (downloaded_size * 100 / total_size) if total_size else 0,
        )
        future = asyncio.run_coroutine_threadsafe(
            context.progress(
                progress,
                downloaded_size=downloaded_size,
                total_size=total_size,
                resumable_cursor={
                    "completed_files": [
                        encode_path(path) for path in list(completed_files)[-1024:]
                    ],
                    "staging_exists": True,
                },
                state_message="downloading" if not seed else "uploading",
            ),
            loop,
        )
        future.result()

    worker = asyncio.create_task(
        asyncio.to_thread(
            execution,
            request_type(**request_fields),
            client,
            cancel_check=cancel_event.is_set,
            progress_callback=report_progress,
        )
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel_event.set()
        await asyncio.shield(worker)
        raise


def execute_connectivity_check(
    client,
    bucket: str,
    prefix: str,
    check_id: int,
    worker_uuid: str,
    *,
    endpoint: str | None = None,
    tls_verify: bool = True,
    check_network: bool = True,
    network_probe=None,
) -> dict:
    started = time.monotonic()
    if check_network and endpoint:
        failure = (network_probe or _probe_network)(endpoint, tls_verify)
        if failure is not None:
            return _failure(failure[0], failure[1], started)

    probe_name = _probe_name(prefix, check_id, worker_uuid)
    payload = secrets.token_bytes(32)
    payload_digest = hashlib.sha256(payload).hexdigest()
    object_may_exist = False
    result = None
    try:
        try:
            list(client.list_objects(bucket, prefix=probe_name.rsplit("/", 1)[0]))
        except Exception as exc:
            code = (
                "s3_authentication_failed"
                if _is_auth_failure(exc)
                else "s3_list_failed"
            )
            return _failure(
                "auth" if code.endswith("authentication_failed") else "list",
                code,
                started,
            )

        try:
            object_may_exist = True
            client.put_object(
                bucket,
                probe_name,
                io.BytesIO(payload),
                len(payload),
                content_type="application/octet-stream",
                metadata={"sha256": payload_digest},
            )
        except Exception:
            result = _failure("write", "s3_write_failed", started)
            return result

        try:
            response = client.get_object(bucket, probe_name)
            try:
                received = response.read()
            finally:
                _close_response(response)
            if hashlib.sha256(received).hexdigest() != payload_digest:
                result = _failure("read", "s3_read_content_mismatch", started)
                return result
        except Exception:
            result = _failure("read", "s3_read_failed", started)
            return result

        result = {
            "state": "ready",
            "readable": True,
            "writable": True,
            "deletable": True,
            "cleanup_failed": False,
            "latency_ms": _latency_ms(started),
        }
        return result
    finally:
        if object_may_exist:
            try:
                client.remove_object(bucket, probe_name)
            except Exception:
                if result is not None:
                    if result["state"] == "ready":
                        result.update(_failure("delete", "s3_delete_failed", started))
                else:
                    result = _failure("delete", "s3_delete_failed", started)
                result["cleanup_failed"] = True


def execute_profile_connectivity_check(
    profile: dict,
    check_id: int,
    worker_uuid: str,
    client_factory,
) -> dict:
    started = time.monotonic()
    endpoint = profile["endpoint"]
    parsed = urlparse(endpoint)
    try:
        client = client_factory(
            endpoint=parsed.netloc,
            access_key=profile["access_key"],
            secret_key=profile["secret_key"],
            secure=parsed.scheme == "https" and profile.get("tls_enabled", True),
            region=profile.get("region") or None,
        )
    except Exception:
        return _failure("client", "s3_client_initialization_failed", started)
    return execute_connectivity_check(
        client,
        profile["bucket"],
        profile.get("prefix", ""),
        check_id,
        worker_uuid,
        endpoint=endpoint,
        tls_verify=profile.get("tls_verify", True),
    )


def _probe_network(endpoint: str, tls_verify: bool):
    parsed = urlparse(endpoint)
    host = parsed.hostname
    if not host:
        return "dns", "dns_resolution_failed"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return "dns", "dns_resolution_failed"
    try:
        connection = socket.create_connection((host, port), timeout=10)
    except OSError:
        return "tcp", "tcp_connection_failed"
    try:
        if parsed.scheme == "https":
            context = (
                ssl.create_default_context()
                if tls_verify
                else ssl._create_unverified_context()
            )
            with context.wrap_socket(connection, server_hostname=host):
                pass
        else:
            pass
    except ssl.SSLCertVerificationError:
        return "tls", "tls_certificate_verify_failed"
    except ssl.SSLError:
        return "tls", "tls_handshake_failed"
    except (TimeoutError, ConnectionResetError, OSError):
        return "tls", "tls_handshake_failed"
    finally:
        connection.close()
    return None


def _probe_name(prefix: str, check_id: int, worker_uuid: str) -> str:
    clean_prefix = prefix.strip("/")
    parts = [
        part
        for part in (
            clean_prefix,
            "_healthchecks",
            str(check_id),
            worker_uuid,
            f"probe-{secrets.token_hex(16)}",
        )
        if part
    ]
    return "/".join(parts)


def _failure(stage: str, code: str, started: float) -> dict:
    return {
        "state": "error",
        "readable": False,
        "writable": False,
        "deletable": False,
        "cleanup_failed": False,
        "failed_stage": stage,
        "error_code": code,
        "latency_ms": _latency_ms(started),
    }


def _latency_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


def _close_response(response):
    close = getattr(response, "close", None)
    release_conn = getattr(response, "release_conn", None)
    if callable(close):
        close()
    if callable(release_conn):
        release_conn()


def _is_auth_failure(exc: Exception) -> bool:
    code = str(getattr(exc, "code", "")).lower()
    message = str(exc).lower()
    return code in {
        "accessdenied",
        "invalidaccesskeyid",
        "signaturedoesnotmatch",
    } or any(
        token in message for token in ("access denied", "invalid access", "signature")
    )
