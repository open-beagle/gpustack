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
from pathlib import Path
from urllib.parse import urlparse

from gpustack.worker.model_preheat.identity import ModelPreheatIdentity, decode_path
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


def execute_seed_preheat(
    request: SeedExecutionRequest,
    s3_client,
    *,
    download_to_staging=None,
    cancel_check=None,
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
                ),
                s3_client,
                cancel_check=cancel_check,
            )

        staging = create_staging_dir(
            request.cache_dir, request.task_id, request.attempt
        )
        _reuse_previous_seed_staging(request, staging)
        _raise_if_cancelled(cancel_check)
        if download_to_staging is None:
            from gpustack.worker.downloaders import (
                download_resolved_revision_to_staging,
            )

            download_to_staging = download_resolved_revision_to_staging
        download_to_staging(
            request.identity,
            staging,
            exclude_patterns=request.exclude_patterns,
        )
        _raise_if_cancelled(cancel_check)
        _remove_excluded_files(staging, request.exclude_patterns)
        manifest = build_model_preheat_manifest(
            staging,
            request.identity,
            cache_key=request.cache_key,
            selection_digest=request.selection_digest,
            generation_id=request.generation_id,
            exclude_patterns=request.exclude_patterns,
            requested_revision=request.requested_revision,
        )
        if reference_manifest is not None and manifest != reference_manifest:
            return _error_result("checksum_mismatch", staging)
        publish_result = s3_client.publish_generation(
            request.bucket,
            request.prefix,
            manifest,
            staging,
            cancel_check=cancel_check,
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
            len(manifest.files),
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
        _reuse_previous_staging_files(request, staging, manifest)
        downloaded = 0
        skipped = 0
        for file in manifest.files:
            _raise_if_cancelled(cancel_check)
            target = _staging_manifest_path(staging, file.path)
            if _file_matches(target, file.size, file.sha256):
                skipped += 1
                completed_files.append(file.path)
                continue
            for _ in range(3):
                _raise_if_cancelled(cancel_check)
                s3_client.download_generation_file(
                    request.bucket, request.prefix, manifest, file, target
                )
                if _file_matches(target, file.size, file.sha256):
                    downloaded += 1
                    completed_files.append(file.path)
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


def _file_matches(path: Path, size: int, expected_sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def _reuse_previous_staging_files(request, staging: Path, manifest) -> None:
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
    for file in manifest.files:
        target = _staging_manifest_path(staging, file.path)
        if _file_matches(target, file.size, file.sha256):
            continue
        for old_staging in previous:
            try:
                source = _staging_manifest_path(old_staging, file.path)
                matches = _file_matches(source, file.size, file.sha256)
            except (LocalCacheError, OSError):
                continue
            if not matches:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            break


def _reuse_previous_seed_staging(request, staging: Path) -> None:
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
    for source in source_root.rglob("*"):
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
        target = (target_root / relative).resolve(strict=False)
        if target_root not in target.parents or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(resolved_source, target)
        except OSError:
            try:
                shutil.copy2(resolved_source, target)
            except OSError:
                continue


def _remove_excluded_files(staging_dir: Path, exclude_patterns) -> None:
    root = staging_dir.resolve()
    decoded_patterns = tuple(decode_path(pattern) for pattern in exclude_patterns)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if any(Path(relative_path).match(pattern) for pattern in decoded_patterns):
            path.unlink()


def build_preheat_role_handlers(cache_dir: str | Path) -> dict:
    async def seed_handler(payload, context):
        return await _execute_payload(payload, cache_dir, seed=True)

    async def target_handler(payload, context):
        return await _execute_payload(payload, cache_dir, seed=False)

    return {"seed": seed_handler, "distribute": target_handler}


async def _execute_payload(payload, cache_dir: str | Path, *, seed: bool) -> dict:
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
    }
    cancel_event = threading.Event()
    execution = execute_seed_preheat if seed else execute_target_preheat
    request_type = SeedExecutionRequest if seed else TargetExecutionRequest
    worker = asyncio.create_task(
        asyncio.to_thread(
            execution,
            request_type(**request_fields),
            client,
            cancel_check=cancel_event.is_set,
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
