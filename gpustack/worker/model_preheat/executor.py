import asyncio
import hashlib
import io
import os
import secrets
import shutil
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    decode_path,
    encode_path,
)
from gpustack.worker.model_preheat.local_cache import create_staging_dir
from gpustack.worker.model_preheat.ollama_artifact import install_ollama_artifact
from gpustack.worker.model_preheat.manifest import (
    ModelPreheatManifestError,
    build_model_preheat_manifest,
)
from gpustack.worker.model_preheat.s3_client import (
    ModelPreheatCanceled,
    ModelPreheatS3Client,
    ModelPreheatS3Conflict,
    ModelPreheatS3ManifestError,
)


@dataclass(frozen=True)
class TrustedLocalCandidate:
    source: str
    root: Path
    paths: tuple[Path, ...]
    repository_complete: bool = False


def _raise_if_cancelled(cancel_check):
    if cancel_check is not None and cancel_check():
        raise ModelPreheatCanceled("canceled")


def _staging_manifest_path(staging_dir: Path, encoded_path: str) -> Path:
    root = staging_dir.resolve()
    path = (root / decode_path(encoded_path)).resolve()
    if root not in path.parents:
        raise ValueError("local_cache_path_escape")
    return path


# 统一 Artifact 执行契约。预热与普通下载共用不可变 Artifact Manifest。
@dataclass(frozen=True)
class SeedExecutionRequest:
    cache_dir: Path
    target_dir: Path
    task_id: int
    attempt: int
    request_digest: str
    identity: ModelPreheatIdentity
    exclude_patterns: tuple[str, ...] | list[str]
    bucket: str
    prefix: str
    source_fallback_enabled: bool
    artifact_id: str | None = None
    manifest_path: str | None = None
    bandwidth_limit_mbps: int | None = None
    resumable_cursor: dict | None = None
    trusted_local_candidate: TrustedLocalCandidate | None = None


@dataclass(frozen=True)
class TargetExecutionRequest:
    cache_dir: Path
    target_dir: Path
    task_id: int
    attempt: int
    request_digest: str
    identity: ModelPreheatIdentity
    exclude_patterns: tuple[str, ...] | list[str]
    bucket: str
    prefix: str
    artifact_id: str
    manifest_path: str
    bandwidth_limit_mbps: int | None = None
    resumable_cursor: dict | None = None
    trusted_local_candidate: TrustedLocalCandidate | None = None


def execute_seed_preheat(
    request: SeedExecutionRequest,
    s3_client,
    *,
    download_to_staging=None,
    source_token=None,
    cancel_check=None,
    progress_callback=None,
):
    if request.request_digest != request.identity.request_digest:
        return _error_result("request_digest_mismatch")
    if request.artifact_id is not None:
        if request.manifest_path is None:
            return _error_result("s3_manifest_invalid")
        if request.trusted_local_candidate is not None:
            staging = create_staging_dir(
                request.cache_dir, request.task_id, request.attempt
            )
            try:
                _clear_directory(staging)
                _copy_candidate(
                    request.trusted_local_candidate,
                    staging,
                    cancel_check=cancel_check,
                )
                manifest = build_model_preheat_manifest(
                    staging,
                    request.identity,
                    exclude_patterns=request.exclude_patterns,
                )
                if manifest.artifact_id != request.artifact_id:
                    return _error_result("local_manifest_conflict")
                _install_staging(staging, request.target_dir, manifest)
                _write_local_artifact_marker(request.cache_dir, manifest)
                return _ready_result(
                    request,
                    s3_client,
                    manifest,
                    transfer_source="current_node",
                    uploaded=0,
                    skipped=len(manifest.files) + 1,
                    downloaded=0,
                )
            except (ModelPreheatManifestError, ModelPreheatS3Conflict):
                return _error_result("local_manifest_conflict")
        target_request = TargetExecutionRequest(
            cache_dir=request.cache_dir,
            target_dir=request.target_dir,
            task_id=request.task_id,
            attempt=request.attempt,
            request_digest=request.request_digest,
            identity=request.identity,
            exclude_patterns=request.exclude_patterns,
            bucket=request.bucket,
            prefix=request.prefix,
            artifact_id=request.artifact_id,
            manifest_path=request.manifest_path,
            bandwidth_limit_mbps=request.bandwidth_limit_mbps,
            resumable_cursor=request.resumable_cursor,
            trusted_local_candidate=request.trusted_local_candidate,
        )
        return execute_target_preheat(
            target_request,
            s3_client,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )

    staging = create_staging_dir(request.cache_dir, request.task_id, request.attempt)
    transfer_source = request.identity.source
    try:
        _clear_directory(staging)
        candidate = request.trusted_local_candidate
        if candidate is not None:
            _copy_candidate(candidate, staging, cancel_check=cancel_check)
            transfer_source = "current_node"
        else:
            if not request.source_fallback_enabled:
                return _error_result("model_artifact_not_found")
            downloader = download_to_staging
            if downloader is None:
                from gpustack.worker.downloaders import (
                    download_resolved_revision_to_staging,
                )

                downloader = download_resolved_revision_to_staging
            downloader(
                request.identity,
                staging,
                token=source_token,
                exclude_patterns=request.exclude_patterns,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
        manifest = build_model_preheat_manifest(
            staging,
            request.identity,
            exclude_patterns=request.exclude_patterns,
            cancel_callback=(
                (lambda: _raise_if_cancelled(cancel_check))
                if cancel_check is not None
                else None
            ),
        )
        published = s3_client.publish_artifact(
            request.bucket,
            request.prefix,
            manifest,
            staging,
            cancel_check=cancel_check,
            bandwidth_limit_mbps=request.bandwidth_limit_mbps,
        )
        _install_staging(staging, request.target_dir, manifest)
        _write_local_artifact_marker(request.cache_dir, manifest)
        return _ready_result(
            request,
            s3_client,
            manifest,
            transfer_source=transfer_source,
            uploaded=published.uploaded,
            skipped=published.skipped,
            downloaded=0,
        )
    except ModelPreheatCanceled:
        return _error_result("canceled")
    except ModelPreheatManifestError:
        return _error_result("local_manifest_invalid")
    except ModelPreheatS3ManifestError:
        return _error_result("s3_manifest_invalid")
    except ModelPreheatS3Conflict as exc:
        return _error_result(_safe_s3_error(exc))
    except (OSError, ValueError):
        return _error_result("worker_execution_failed")


def execute_target_preheat(
    request: TargetExecutionRequest,
    s3_client,
    *,
    cancel_check=None,
    progress_callback=None,
):
    if request.request_digest != request.identity.request_digest:
        return _error_result("request_digest_mismatch")
    try:
        manifest = s3_client.read_artifact_manifest_path(
            request.bucket, request.manifest_path
        )
        if manifest is None or not _manifest_matches_request(manifest, request):
            return _error_result("s3_manifest_invalid")
        target = Path(request.target_dir)
        target.mkdir(parents=True, exist_ok=True)
        if request.identity.source == "ollama_library":
            install_ollama_artifact(
                s3_client,
                manifest,
                bucket=request.bucket,
                prefix=request.prefix,
                target_root=target,
                model_id=decode_path(request.identity.model_path),
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
            _write_local_artifact_marker(request.cache_dir, manifest)
            return _ready_result(
                request,
                s3_client,
                manifest,
                transfer_source="s3",
                uploaded=0,
                skipped=0,
                downloaded=1,
            )
        downloaded = 0
        completed = []
        completed_size = 0
        for file in manifest.files:
            _raise_if_cancelled(cancel_check)
            destination = _staging_manifest_path(target, file.path)
            if not _local_file_matches(destination, file.size, file.sha256):
                s3_client.download_artifact_file(
                    request.bucket,
                    request.prefix,
                    manifest,
                    file,
                    destination,
                )
                downloaded += 1
            completed.append(decode_path(file.path))
            completed_size += file.size
            if progress_callback is not None:
                progress_callback(completed, completed_size, manifest.total_size)
        _write_local_artifact_marker(request.cache_dir, manifest)
        return _ready_result(
            request,
            s3_client,
            manifest,
            transfer_source="s3",
            uploaded=0,
            skipped=0,
            downloaded=downloaded,
        )
    except ModelPreheatCanceled:
        return _error_result("canceled")
    except ModelPreheatS3ManifestError:
        return _error_result("s3_manifest_invalid")
    except ModelPreheatS3Conflict as exc:
        return _error_result(_safe_s3_error(exc))
    except (OSError, ValueError):
        return _error_result("worker_execution_failed")


def _manifest_matches_request(manifest, request) -> bool:
    return (
        manifest.artifact_id == request.artifact_id
        and manifest.identity.source == request.identity.source
        and manifest.identity.model_path == request.identity.model_path
        and manifest.identity.revision_path == request.identity.revision_path
        and tuple(manifest.identity.file_patterns)
        == tuple(request.identity.file_patterns)
        and tuple(manifest.exclude_patterns)
        == tuple(sorted(encode_path(value) for value in request.exclude_patterns))
    )


def _ready_result(
    request,
    client,
    manifest,
    *,
    transfer_source,
    uploaded,
    skipped,
    downloaded,
):
    manifest_bytes = manifest.to_artifact_json_bytes()
    return {
        "state": "ready",
        "request_digest": request.request_digest,
        "artifact_id": manifest.artifact_id,
        "manifest_digest": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_path": client.artifact_manifest_object(request.prefix, manifest),
        "file_count": len(manifest.files),
        "total_size": manifest.total_size,
        "local_cache_state": "valid",
        "transfer_source": transfer_source,
        "uploaded": uploaded,
        "skipped": skipped,
        "downloaded": downloaded,
    }


def _error_result(error_code, *args, **kwargs):
    return {"state": "error", "error_code": error_code}


def _safe_s3_error(exc):
    code = str(exc)
    allowed = {
        "checksum_mismatch",
        "object_content_conflict",
        "artifact_manifest_conflict",
        "local_file_content_mismatch",
    }
    return code if code in allowed else "s3_object_conflict"


def _clear_directory(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _copy_candidate(candidate, staging, *, cancel_check=None):
    root = Path(candidate.root).resolve()
    if not root.is_dir():
        raise OSError("trusted_candidate_missing")
    for path in root.rglob("*"):
        _raise_if_cancelled(cancel_check)
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        destination = (staging / relative).resolve()
        if staging.resolve() not in destination.parents:
            raise OSError("trusted_candidate_path_escape")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _install_staging(staging, target, manifest):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for file in manifest.files:
        source = _staging_manifest_path(staging, file.path)
        destination = _staging_manifest_path(target, file.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.preheat")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)


def _local_file_matches(path, size, sha256):
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == sha256


def _write_local_artifact_marker(cache_dir, manifest):
    directory = Path(cache_dir) / ".gpustack-manifests"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{manifest.artifact_id}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_bytes(manifest.to_artifact_json_bytes())
    os.replace(temporary, target)


def build_preheat_role_handlers(
    cache_dir: str | Path, huggingface_token: str | None = None
) -> dict:
    async def seed_handler(payload, context):
        return await _execute_payload(
            payload,
            context,
            cache_dir,
            seed=True,
            huggingface_token=huggingface_token,
        )

    async def target_handler(payload, context):
        return await _execute_payload(payload, context, cache_dir, seed=False)

    return {"seed": seed_handler, "distribute": target_handler}


async def _execute_payload(
    payload,
    context,
    cache_dir: str | Path,
    *,
    seed: bool,
    huggingface_token: str | None = None,
) -> dict:
    from gpustack.worker.downloaders import preheat_model_target_dir

    task = payload.task
    patterns = list(task.get("include_patterns") or [])
    exclude_patterns = list(task.get("exclude_patterns") or [])
    identity = ModelPreheatIdentity(
        source=task["source"],
        model_id=task["model_id"],
        revision=task["resolved_revision"],
        file_patterns=patterns,
        requested_revision=task.get("requested_revision"),
        exclude_patterns=exclude_patterns,
    )
    profile = payload.profile
    client = ModelPreheatS3Client.from_minio(
        endpoint=profile.endpoint,
        access_key=profile.access_key,
        secret_key=profile.secret_key,
        secure=bool(profile.tls_enabled),
        tls_verify=profile.tls_verify,
        region=profile.region or None,
        use_virtual_hosted_style=profile.use_virtual_hosted_style,
    )
    request_fields = {
        "cache_dir": Path(cache_dir),
        "target_dir": preheat_model_target_dir(cache_dir, identity),
        "task_id": task.get("id", payload.worker_task_id),
        "attempt": payload.attempt,
        "request_digest": task["request_digest"],
        "identity": identity,
        "exclude_patterns": exclude_patterns,
        "bucket": profile.bucket,
        "prefix": profile.prefix,
        "artifact_id": task.get("artifact_id"),
        "manifest_path": task.get("s3_manifest_path"),
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
    if seed:
        request_fields["source_fallback_enabled"] = bool(
            profile.source_fallback_enabled
        )
        if request_fields["artifact_id"] is None:
            request_fields.pop("manifest_path")
    elif (
        request_fields["artifact_id"] is None or request_fields["manifest_path"] is None
    ):
        raise RuntimeError("artifact_not_bound")
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

    execution_kwargs = {
        "cancel_check": cancel_event.is_set,
        "progress_callback": report_progress,
    }
    if seed:
        execution_kwargs["source_token"] = huggingface_token
    worker = asyncio.create_task(
        asyncio.to_thread(
            execution,
            request_type(**request_fields),
            client,
            **execution_kwargs,
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
    tls_enabled = bool(profile.get("tls_enabled", True))
    effective_endpoint = _endpoint_with_tls_enabled(endpoint, tls_enabled)
    try:
        client = client_factory(
            endpoint=endpoint,
            access_key=profile["access_key"],
            secret_key=profile["secret_key"],
            secure=tls_enabled,
            tls_verify=profile.get("tls_verify", True),
            region=profile.get("region") or None,
            use_virtual_hosted_style=profile.get("use_virtual_hosted_style", True),
        )
    except Exception:
        return _failure("client", "s3_client_initialization_failed", started)
    return execute_connectivity_check(
        client,
        profile["bucket"],
        profile.get("prefix", ""),
        check_id,
        worker_uuid,
        endpoint=effective_endpoint,
        tls_verify=profile.get("tls_verify", True),
    )


def _endpoint_with_tls_enabled(endpoint: str, tls_enabled: bool) -> str:
    """使用显式 TLS 开关生成与 MinIO 客户端一致的网络探测地址。"""
    parsed = urlparse(endpoint)
    return parsed._replace(scheme="https" if tls_enabled else "http").geturl()


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
