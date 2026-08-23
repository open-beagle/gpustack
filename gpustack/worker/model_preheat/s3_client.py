import io
import hashlib
import json
import ssl
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

from minio import Minio
from minio.datatypes import Part
from minio.helpers import (
    MAX_MULTIPART_COUNT,
    MAX_MULTIPART_OBJECT_SIZE,
    MAX_PART_SIZE,
    md5sum_hash,
)
import urllib3

from gpustack.worker.model_preheat.identity import decode_path

from gpustack.worker.model_preheat.manifest import (
    MAX_MANIFEST_BYTES,
    ManifestFile,
    ModelPreheatManifest,
    ModelPreheatManifestError,
    parse_artifact_manifest,
)


ARTIFACT_MANIFEST_OBJECT_NAME = "manifest.json"
CONDITIONAL_SINGLE_PUT_MAX_SIZE = 64 * 1024 * 1024
CONDITIONAL_MULTIPART_PART_SIZE = 5 * 1024 * 1024
RATE_LIMIT_CHUNK_SIZE = 64 * 1024


class ReadyGenerationConflict(RuntimeError):
    pass


class ModelPreheatS3Conflict(RuntimeError):
    pass


class ModelPreheatS3ManifestError(RuntimeError):
    pass


class ModelPreheatCanceled(RuntimeError):
    pass


class ModelPreheatS3ManifestConflict(RuntimeError):
    """统一 Artifact 的 Manifest 已存在且内容不一致。"""

    pass


class _BandwidthLimiter:
    def __init__(self, bandwidth_limit_mbps: int | None):
        self._bytes_per_second = (
            bandwidth_limit_mbps * 1_000_000 / 8
            if bandwidth_limit_mbps is not None
            else None
        )
        self._started_at = time.monotonic()
        self._transferred = 0

    @property
    def enabled(self) -> bool:
        return self._bytes_per_second is not None

    def consume(self, size: int):
        if self._bytes_per_second is None or size <= 0:
            return
        self._transferred += size
        target_elapsed = self._transferred / self._bytes_per_second
        delay = target_elapsed - (time.monotonic() - self._started_at)
        if delay > 0:
            time.sleep(delay)


class _RateLimitedReader:
    def __init__(self, source, limiter, expected_length, cancel_check=None):
        self._source = source
        self._limiter = limiter
        self._expected_length = expected_length
        self._transferred = 0
        self._cancel_check = cancel_check

    def read(self, size=-1):
        if self._cancel_check is not None and self._cancel_check():
            raise ModelPreheatCanceled("canceled")
        remaining = self._expected_length - self._transferred
        if remaining <= 0:
            return b""
        if size < 0 or size > remaining:
            size = remaining
        if self._limiter.enabled:
            size = min(size, RATE_LIMIT_CHUNK_SIZE)
        chunk = self._source.read(size)
        self._limiter.consume(len(chunk))
        self._transferred += len(chunk)
        return chunk

    def validate_complete(self):
        if self._transferred != self._expected_length or self._source.read(1):
            raise ModelPreheatS3Conflict("conditional_upload_source_size_mismatch")


class _BoundedReader:
    def __init__(self, source, length, cancel_check=None):
        self._source = source
        self._remaining = length
        self._cancel_check = cancel_check

    def read(self, size=-1):
        if self._remaining <= 0:
            return b""
        if self._cancel_check is not None and self._cancel_check():
            raise ModelPreheatCanceled("canceled")
        if size < 0 or size > self._remaining:
            size = self._remaining
        chunk = self._source.read(size)
        if not isinstance(chunk, bytes) or not chunk:
            raise ModelPreheatS3Conflict("conditional_upload_source_size_mismatch")
        self._remaining -= len(chunk)
        return chunk

    def validate_complete(self):
        if self._remaining != 0:
            raise ModelPreheatS3Conflict("conditional_upload_source_size_mismatch")


def _read_exact(source, size, cancel_check=None):
    chunks = []
    remaining = size
    while remaining:
        if cancel_check is not None and cancel_check():
            raise ModelPreheatCanceled("canceled")
        chunk = source.read(remaining)
        if not isinstance(chunk, bytes) or not chunk:
            raise ModelPreheatS3Conflict("conditional_upload_source_size_mismatch")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass(frozen=True)
class PublishResult:
    uploaded: int
    skipped: int
    ready_written: bool
    ready_digest: str
    generation_prefix: str


class _QuietUnverifiedHTTPSConnectionPool(urllib3.HTTPSConnectionPool):
    """仅供显式 tls_verify=False 的 S3 client 使用，不改全局 warning 过滤器。"""

    def _validate_conn(self, conn):
        urllib3.HTTPConnectionPool._validate_conn(self, conn)
        if conn.is_closed:
            conn.connect()


class ModelPreheatS3Client:
    def __init__(
        self,
        client,
        *,
        cancel_cleanup_attempts: int = 3,
        cancel_cleanup_sleep=time.sleep,
        cancel_cleanup_backoff: float = 0.1,
    ):
        if cancel_cleanup_attempts < 1:
            raise ValueError("invalid_cancel_cleanup_attempts")
        self._client = client
        self._cancel_cleanup_attempts = cancel_cleanup_attempts
        self._cancel_cleanup_sleep = cancel_cleanup_sleep
        self._cancel_cleanup_backoff = cancel_cleanup_backoff

    @classmethod
    def from_minio(
        cls,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
        tls_verify: bool = True,
        region: str | None = None,
        use_virtual_hosted_style: bool = True,
    ):
        parsed = urlparse(endpoint)
        if parsed.scheme in ("http", "https"):
            endpoint = parsed.netloc
        else:
            endpoint = endpoint.rstrip("/")
        http_client = None
        if secure and not tls_verify:
            http_client = urllib3.PoolManager(
                cert_reqs=ssl.CERT_NONE,
                assert_hostname=False,
            )
            http_client.pool_classes_by_scheme = {
                **http_client.pool_classes_by_scheme,
                "https": _QuietUnverifiedHTTPSConnectionPool,
            }
        minio_client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            region=region,
            http_client=http_client,
        )
        if use_virtual_hosted_style:
            minio_client.enable_virtual_style_endpoint()
        else:
            minio_client.disable_virtual_style_endpoint()
        return cls(minio_client)

    def list_objects(self, *args, **kwargs):
        """透传 SDK 的对象枚举，供连通性检查使用。"""
        return self._client.list_objects(*args, **kwargs)

    def put_object(self, *args, **kwargs):
        """透传 SDK 的对象写入，供连通性检查使用。"""
        return self._client.put_object(*args, **kwargs)

    def get_object(self, *args, **kwargs):
        """透传 SDK 的对象读取，供连通性检查使用。"""
        return self._client.get_object(*args, **kwargs)

    def remove_object(self, *args, **kwargs):
        """透传 SDK 的对象删除，供连通性检查使用。"""
        return self._client.remove_object(*args, **kwargs)

    def artifact_manifest_object(
        self,
        profile_prefix: str,
        manifest: ModelPreheatManifest,
    ) -> str:
        """统一 Artifact 的 Manifest 对象 Key。

        固定为 `<profile_prefix>/<source>/<organization>/<model>/<artifact_id>/manifest.json`。
        profile_prefix 必须显式传入并安全透传到 Artifact 前缀。
        """
        return self._join_object_name(
            manifest.artifact_prefix(profile_prefix),
            ARTIFACT_MANIFEST_OBJECT_NAME,
        )

    def artifact_file_object(
        self,
        profile_prefix: str,
        manifest: ModelPreheatManifest,
        file: ManifestFile,
    ) -> str:
        """统一 Artifact 的文件对象 Key：`.../<artifact_id>/files/<relative_path>`。"""
        return self._join_object_name(
            manifest.artifact_prefix(profile_prefix),
            "files",
            file.path,
        )

    def read_artifact_manifest(
        self,
        bucket_name: str,
        profile_prefix: str,
        manifest: ModelPreheatManifest,
    ) -> ModelPreheatManifest | None:
        """读取并严格校验统一 Artifact Manifest；不存在时返回 None。

        旧协议的 Manifest 字段集合与统一 Artifact 不同，
        会被 `parse_artifact_manifest` 拒绝，不视为有效 Artifact。
        """
        object_name = self.artifact_manifest_object(profile_prefix, manifest)
        try:
            manifest_bytes = self._read_object_bytes(bucket_name, object_name)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelPreheatS3ManifestError("s3_manifest_invalid") from exc
        if manifest_bytes is None:
            return None
        try:
            payload = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelPreheatS3ManifestError("s3_manifest_invalid") from exc
        stored = parse_artifact_manifest(payload)
        if stored.artifact_id != manifest.artifact_id:
            # 对象 Key 由调用方 artifact_id 推导，存储内容却指向
            # 不同身份，视为内容冲突而不是格式非法。
            raise ModelPreheatS3ManifestConflict("artifact_manifest_conflict")
        return stored

    def read_artifact_manifest_path(
        self, bucket_name: str, manifest_path: str
    ) -> ModelPreheatManifest | None:
        """按 Server 库存固定的对象 Key 读取并严格解析 Manifest。"""
        self._validate_object_name(manifest_path)
        try:
            manifest_bytes = self._read_object_bytes(bucket_name, manifest_path)
            if manifest_bytes is None:
                return None
            return parse_artifact_manifest(json.loads(manifest_bytes.decode("utf-8")))
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ModelPreheatManifestError,
        ) as exc:
            raise ModelPreheatS3ManifestError("s3_manifest_invalid") from exc

    def download_artifact_file(
        self,
        bucket_name: str,
        profile_prefix: str,
        manifest: ModelPreheatManifest,
        file: ManifestFile,
        target: str | Path,
    ) -> None:
        """下载单个 Artifact 文件，并以大小和 SHA-256 校验后原子替换。"""
        import hashlib
        import os

        destination = Path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.part")
        object_name = self.artifact_file_object(profile_prefix, manifest, file)
        digest = hashlib.sha256()
        size = 0
        try:
            response = self._client.get_object(bucket_name, object_name)
            try:
                with temporary.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            finally:
                self._close_response(response)
            if size != file.size or digest.hexdigest() != file.sha256:
                raise ModelPreheatS3Conflict("checksum_mismatch")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def publish_artifact(
        self,
        bucket_name: str,
        profile_prefix: str,
        manifest: ModelPreheatManifest,
        root_dir: str | Path,
        *,
        cancel_check=None,
        bandwidth_limit_mbps: int | None = None,
    ) -> PublishResult:
        """按统一 Artifact 协议增量发布文件，最后写入 Manifest。

        本地文件在发布前流式重算 SHA-256。S3 对象仅通过 HEAD 返回的
        大小和 SHA-256 元数据判断是否可跳过；不一致或缺少可信摘要时
        使用普通 PutObject 覆盖。这样既避免回读和重复上传流量，也兼容
        不支持条件 PUT 的私有 S3。
        profile_prefix 必须显式传入并安全透传到所有对象 Key。
        """
        root = Path(root_dir).resolve()
        uploaded = 0
        skipped = 0
        publish_attempt = uuid.uuid4().hex
        self._raise_if_cancelled(cancel_check)

        for file in manifest.files:
            self._raise_if_cancelled(cancel_check)
            local_path = self._local_manifest_path(root, file)
            local_sha256 = self._sha256_file_bytes(
                local_path,
                cancel_check=cancel_check,
                bandwidth_limit_mbps=bandwidth_limit_mbps,
            )
            if local_sha256 != file.sha256:
                raise ModelPreheatS3Conflict("local_file_content_mismatch")
            object_name = self.artifact_file_object(profile_prefix, manifest, file)
            if self._remote_object_matches(
                bucket_name, object_name, file.size, file.sha256
            ):
                skipped += 1
                continue
            self._put_file(
                bucket_name,
                object_name,
                local_path,
                metadata={
                    "sha256": file.sha256,
                    "model-artifact-id": manifest.artifact_id,
                    "model-artifact-publish-attempt": publish_attempt,
                },
                cancel_check=cancel_check,
                bandwidth_limit_mbps=bandwidth_limit_mbps,
            )
            self._raise_if_cancelled(cancel_check)
            uploaded += 1

        self._raise_if_cancelled(cancel_check)
        manifest_bytes = manifest.to_artifact_json_bytes()
        manifest_sha256 = self._sha256_bytes(manifest_bytes)
        manifest_object = self.artifact_manifest_object(profile_prefix, manifest)
        if self._remote_object_matches(
            bucket_name, manifest_object, len(manifest_bytes), manifest_sha256
        ):
            return PublishResult(
                uploaded=uploaded,
                skipped=skipped + 1,
                ready_written=False,
                ready_digest=manifest.artifact_id,
                generation_prefix=manifest.artifact_prefix(profile_prefix),
            )
        self._put_bytes(
            bucket_name,
            manifest_object,
            manifest_bytes,
            content_type="application/json",
            metadata={
                "sha256": manifest_sha256,
                "model-artifact-id": manifest.artifact_id,
                "model-artifact-publish-attempt": publish_attempt,
            },
        )
        self._raise_if_cancelled(cancel_check)
        return PublishResult(
            uploaded=uploaded + 1,
            skipped=skipped,
            ready_written=True,
            ready_digest=manifest.artifact_id,
            generation_prefix=manifest.artifact_prefix(profile_prefix),
        )

    @staticmethod
    def _join_object_name(*segments: str) -> str:
        clean_segments = []
        for segment in segments:
            if segment == "":
                continue
            clean_segment = segment.strip("/")
            if clean_segment == "":
                continue
            ModelPreheatS3Client._validate_encoded_object_name(clean_segment)
            clean_segments.append(clean_segment)
        return "/".join(clean_segments)

    @staticmethod
    def _validate_encoded_object_name(object_name: str):
        if object_name.startswith("/") or "\\" in object_name:
            raise ValueError("invalid_s3_object_name")
        if any(ord(char) < 32 or ord(char) == 127 for char in object_name):
            raise ValueError("invalid_s3_object_name")
        segments = object_name.split("/")
        if any(segment in ("", ".", "..") for segment in segments):
            raise ValueError("invalid_s3_object_name")

    @staticmethod
    def _local_manifest_path(root: Path, file: ManifestFile) -> Path:
        local_path = (root / decode_path(file.path)).resolve()
        if root not in local_path.parents and local_path != root:
            raise ValueError("manifest_path_escape")
        return local_path

    @staticmethod
    def _validate_object_name(object_name: str) -> None:
        if (
            not isinstance(object_name, str)
            or not object_name
            or object_name.startswith("/")
            or "\\" in object_name
            or any(part in {"", ".", ".."} for part in object_name.split("/"))
            or any(ord(char) < 32 or ord(char) == 127 for char in object_name)
        ):
            raise ValueError("invalid_s3_object_name")

    def _sha256_file_bytes(
        self,
        path: Path,
        *,
        cancel_check=None,
        bandwidth_limit_mbps: int | None = None,
    ) -> str:
        """流式读取本地文件并计算内容 SHA-256（不信任 Manifest 声明值）。"""
        import hashlib

        digest = hashlib.sha256()
        limiter = _BandwidthLimiter(bandwidth_limit_mbps)
        chunk_size = RATE_LIMIT_CHUNK_SIZE if limiter.enabled else 1024 * 1024
        with path.open("rb") as source:
            while True:
                if cancel_check is not None and cancel_check():
                    raise ModelPreheatCanceled("canceled")
                chunk = source.read(chunk_size)
                if not chunk:
                    return digest.hexdigest()
                limiter.consume(len(chunk))
                digest.update(chunk)

    def _read_object_bytes(
        self, bucket_name: str, object_name: str, *, max_bytes=MAX_MANIFEST_BYTES
    ) -> bytes | None:
        try:
            response = self._client.get_object(bucket_name, object_name)
        except Exception as exc:
            if self._is_not_found(exc):
                return None
            raise
        try:
            chunks = []
            total = 0
            while True:
                chunk = response.read(min(1024 * 1024, max_bytes + 1 - total))
                if not chunk:
                    return b"".join(chunks)
                total += len(chunk)
                if total > max_bytes:
                    raise ModelPreheatS3ManifestError("s3_manifest_invalid")
                chunks.append(chunk)
        finally:
            self._close_response(response)

    def _remote_object_matches(
        self,
        bucket_name: str,
        object_name: str,
        size: int,
        sha256: str,
        *,
        cancel_check=None,
        bandwidth_limit_mbps: int | None = None,
    ) -> bool:
        """通过 HEAD 的大小和受管 SHA-256 元数据判断是否可跳过上传。"""
        try:
            stat = self._client.stat_object(bucket_name, object_name)
        except Exception as exc:
            if self._is_not_found(exc):
                return False
            raise
        del cancel_check, bandwidth_limit_mbps
        metadata = self._normalized_metadata(getattr(stat, "metadata", {}) or {})
        return getattr(stat, "size", None) == size and metadata.get("sha256") == sha256

    def _stream_sha256(
        self,
        bucket_name: str,
        object_name: str,
        *,
        cancel_check=None,
        bandwidth_limit_mbps: int | None = None,
    ) -> str:
        """流式读取对象并计算内容 SHA-256（不信任任何 metadata）。"""
        import hashlib

        digest = hashlib.sha256()
        limiter = _BandwidthLimiter(bandwidth_limit_mbps)
        try:
            response = self._client.get_object(bucket_name, object_name)
            chunk_size = RATE_LIMIT_CHUNK_SIZE if limiter.enabled else 1024 * 1024
            while True:
                if cancel_check is not None and cancel_check():
                    raise ModelPreheatCanceled("canceled")
                chunk = response.read(chunk_size)
                if not chunk:
                    return digest.hexdigest()
                limiter.consume(len(chunk))
                digest.update(chunk)
        finally:
            self._close_response(response)

    def _put_file_if_absent(
        self,
        bucket_name: str,
        object_name: str,
        path: Path,
        metadata: dict[str, str],
        cancel_check=None,
        bandwidth_limit_mbps: int | None = None,
    ) -> bool:
        with path.open("rb") as source:
            return self._put_stream_if_absent(
                bucket_name,
                object_name,
                source,
                path.stat().st_size,
                "application/octet-stream",
                metadata,
                cancel_check=cancel_check,
                bandwidth_limit_mbps=bandwidth_limit_mbps,
            )

    def _put_file(
        self,
        bucket_name: str,
        object_name: str,
        path: Path,
        metadata: dict[str, str],
        cancel_check=None,
        bandwidth_limit_mbps: int | None = None,
    ) -> None:
        with path.open("rb") as source:
            self._put_stream(
                bucket_name,
                object_name,
                source,
                path.stat().st_size,
                "application/octet-stream",
                metadata,
                cancel_check=cancel_check,
                bandwidth_limit_mbps=bandwidth_limit_mbps,
            )

    def _put_bytes(
        self,
        bucket_name: str,
        object_name: str,
        payload: bytes,
        *,
        content_type: str,
        metadata: dict[str, str],
    ) -> None:
        self._put_stream(
            bucket_name,
            object_name,
            io.BytesIO(payload),
            len(payload),
            content_type,
            metadata,
            resumable=False,
        )

    def _put_stream(
        self,
        bucket_name,
        object_name,
        source,
        length,
        content_type,
        metadata,
        cancel_check=None,
        bandwidth_limit_mbps: int | None = None,
        resumable: bool = True,
    ) -> None:
        if resumable and length > CONDITIONAL_SINGLE_PUT_MAX_SIZE:
            try:
                if self._resumable_multipart_put(
                    bucket_name,
                    object_name,
                    source,
                    length,
                    content_type,
                    metadata,
                    cancel_check=cancel_check,
                    bandwidth_limit_mbps=bandwidth_limit_mbps,
                ):
                    return
            except Exception as exc:
                if not self._is_multipart_resume_unsupported(exc):
                    raise
                self._rewind_upload_source(source)
        source = _RateLimitedReader(
            source,
            _BandwidthLimiter(bandwidth_limit_mbps),
            length,
            cancel_check=cancel_check,
        )
        self._client.put_object(
            bucket_name,
            object_name,
            source,
            length,
            content_type=content_type,
            metadata=metadata,
        )
        source.validate_complete()

    @staticmethod
    def _rewind_upload_source(source) -> None:
        seek = getattr(source, "seek", None)
        if not callable(seek):
            raise ModelPreheatS3Conflict("upload_source_not_rewindable")
        seek(0)

    def _resumable_multipart_put(
        self,
        bucket_name,
        object_name,
        source,
        length,
        content_type,
        metadata,
        cancel_check=None,
        bandwidth_limit_mbps: int | None = None,
    ) -> bool:
        methods = self._multipart_resume_methods()
        if methods is None:
            return False
        list_uploads, list_parts, create, upload_part, complete = methods
        upload_id = self._get_or_create_resumable_upload(
            list_uploads,
            create,
            bucket_name,
            object_name,
            content_type,
            metadata,
        )
        if upload_id is None:
            return False
        completed_parts = self._listed_multipart_parts(
            list_parts,
            bucket_name,
            object_name,
            upload_id,
        )
        parts = self._upload_resumable_parts(
            upload_part,
            bucket_name,
            object_name,
            upload_id,
            source,
            length,
            completed_parts,
            cancel_check,
            bandwidth_limit_mbps,
        )
        self._raise_if_cancelled(cancel_check)
        complete(bucket_name, object_name, upload_id, parts)
        return True

    def _multipart_resume_methods(self):
        list_uploads = getattr(self._client, "_list_multipart_uploads", None)
        list_parts = getattr(self._client, "_list_parts", None)
        create = getattr(self._client, "_create_multipart_upload", None)
        upload_part = getattr(self._client, "_upload_part", None)
        complete = getattr(self._client, "_complete_multipart_upload", None)
        methods = (list_uploads, list_parts, create, upload_part, complete)
        return methods if all(callable(method) for method in methods) else None

    def _get_or_create_resumable_upload(
        self,
        list_uploads,
        create,
        bucket_name,
        object_name,
        content_type,
        metadata,
    ):
        try:
            uploads = list_uploads(
                bucket_name,
                prefix=object_name,
                max_uploads=1000,
            ).uploads
        except Exception as exc:
            if self._is_multipart_resume_unsupported(exc):
                return None
            raise
        matching_uploads = [
            upload
            for upload in uploads
            if upload.object_name == object_name and upload.upload_id
        ]
        if matching_uploads:
            return max(
                matching_uploads,
                key=lambda upload: (
                    upload.initiated_time.timestamp()
                    if upload.initiated_time is not None
                    else float("-inf")
                ),
            ).upload_id

        headers = {"Content-Type": content_type}
        headers.update({f"x-amz-meta-{key}": value for key, value in metadata.items()})
        try:
            return create(bucket_name, object_name, headers)
        except Exception as exc:
            if self._is_multipart_resume_unsupported(exc):
                return None
            raise

    def _upload_resumable_parts(
        self,
        upload_part,
        bucket_name,
        object_name,
        upload_id,
        source,
        length,
        completed_parts,
        cancel_check,
        bandwidth_limit_mbps,
    ) -> list[Part]:
        part_size = max(
            CONDITIONAL_MULTIPART_PART_SIZE,
            (length + MAX_MULTIPART_COUNT - 1) // MAX_MULTIPART_COUNT,
        )
        limiter = _BandwidthLimiter(bandwidth_limit_mbps)
        parts = []
        remaining = length
        part_number = 1
        while remaining:
            self._raise_if_cancelled(cancel_check)
            expected = min(part_size, remaining)
            data = _read_exact(source, expected, cancel_check)
            existing = completed_parts.get(part_number)
            local_etag = hashlib.md5(data, usedforsecurity=False).hexdigest()
            if (
                existing is not None
                and existing.size == expected
                and existing.etag.strip('"') == local_etag
            ):
                etag = existing.etag.strip('"')
            else:
                etag = upload_part(
                    bucket_name,
                    object_name,
                    data,
                    None,
                    upload_id,
                    part_number,
                )
                limiter.consume(len(data))
            parts.append(Part(part_number, etag))
            remaining -= expected
            part_number += 1
        if source.read(1):
            raise ModelPreheatS3Conflict("upload_source_size_mismatch")
        return parts

    @staticmethod
    def _listed_multipart_parts(
        list_parts,
        bucket_name,
        object_name,
        upload_id,
    ) -> dict[int, Part]:
        parts = {}
        marker = None
        while True:
            result = list_parts(
                bucket_name,
                object_name,
                upload_id,
                max_parts=1000,
                part_number_marker=marker,
            )
            parts.update({part.part_number: part for part in result.parts})
            if not result.is_truncated:
                return parts
            marker = result.next_part_number_marker

    @staticmethod
    def _is_multipart_resume_unsupported(exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        response = getattr(exc, "response", None)
        status = getattr(exc, "status", None) or getattr(response, "status", None)
        return code in {"NotImplemented", "MethodNotAllowed"} or status in {
            405,
            501,
            "405",
            "501",
        }

    def _put_bytes_if_absent(
        self,
        bucket_name: str,
        object_name: str,
        payload: bytes,
        *,
        content_type: str,
        metadata: dict[str, str],
    ) -> bool:
        return self._put_stream_if_absent(
            bucket_name,
            object_name,
            io.BytesIO(payload),
            len(payload),
            content_type,
            metadata,
        )

    def _put_stream_if_absent(
        self,
        bucket_name,
        object_name,
        source,
        length,
        content_type,
        metadata,
        cancel_check=None,
        bandwidth_limit_mbps: int | None = None,
    ) -> bool:
        source = _RateLimitedReader(
            source, _BandwidthLimiter(bandwidth_limit_mbps), length
        )
        if length > CONDITIONAL_SINGLE_PUT_MAX_SIZE:
            return self._conditional_multipart_put(
                bucket_name,
                object_name,
                source,
                length,
                content_type,
                metadata,
                cancel_check=cancel_check,
                stream_parts=bandwidth_limit_mbps is not None,
            )
        put_if_absent = getattr(self._client, "put_object_if_absent", None)
        if callable(put_if_absent):
            created = bool(
                put_if_absent(
                    bucket_name,
                    object_name,
                    source,
                    length,
                    content_type=content_type,
                    metadata=metadata,
                )
            )
            if created:
                source.validate_complete()
            return created
        presign = getattr(self._client, "presigned_put_object", None)
        http = getattr(self._client, "_http", None)
        if not callable(presign) or http is None:
            raise ModelPreheatS3Conflict("conditional_create_unsupported")
        headers = {
            "Content-Length": str(length),
            "Content-Type": content_type,
            "If-None-Match": "*",
        }
        headers.update({f"x-amz-meta-{key}": value for key, value in metadata.items()})
        response = None
        try:
            url = presign(
                bucket_name,
                object_name,
                expires=timedelta(minutes=5),
            )
            response = http.urlopen(
                "PUT",
                url,
                body=source,
                headers=headers,
                preload_content=True,
            )
            if response.status == 412:
                return False
            if response.status not in {200, 204}:
                raise ModelPreheatS3Conflict(
                    f"conditional_upload_failed:{response.status}"
                )
            source.validate_complete()
            return True
        finally:
            if response is not None:
                response.release_conn()

    def _conditional_multipart_put(
        self,
        bucket_name,
        object_name,
        source,
        length,
        content_type,
        metadata,
        cancel_check=None,
        stream_parts=False,
    ) -> bool:
        if length > MAX_MULTIPART_OBJECT_SIZE:
            raise ModelPreheatS3Conflict("conditional_upload_too_large")
        create = getattr(self._client, "_create_multipart_upload", None)
        upload_part = getattr(self._client, "_upload_part", None)
        abort = getattr(self._client, "_abort_multipart_upload", None)
        execute = getattr(self._client, "_execute", None)
        required_methods = (create, abort, execute)
        if not stream_parts:
            required_methods += (upload_part,)
        if not all(callable(method) for method in required_methods):
            raise ModelPreheatS3Conflict("conditional_multipart_unsupported")

        part_size = max(
            CONDITIONAL_MULTIPART_PART_SIZE,
            (length + MAX_MULTIPART_COUNT - 1) // MAX_MULTIPART_COUNT,
        )
        if part_size > MAX_PART_SIZE:
            raise ModelPreheatS3Conflict("conditional_upload_too_large")
        create_headers = {"Content-Type": content_type}
        create_headers.update(
            {f"x-amz-meta-{key}": value for key, value in metadata.items()}
        )
        self._raise_if_cancelled(cancel_check)
        upload_id = create(bucket_name, object_name, create_headers)
        completed = False
        try:
            parts = []
            remaining = length
            part_number = 1
            while remaining:
                self._raise_if_cancelled(cancel_check)
                expected = min(part_size, remaining)
                if stream_parts:
                    etag = self._upload_streaming_part(
                        bucket_name,
                        object_name,
                        source,
                        expected,
                        upload_id,
                        part_number,
                        cancel_check,
                    )
                else:
                    data = _read_exact(source, expected, cancel_check)
                    etag = upload_part(
                        bucket_name,
                        object_name,
                        data,
                        None,
                        upload_id,
                        part_number,
                    )
                parts.append(Part(part_number, etag))
                remaining -= expected
                part_number += 1
                self._raise_if_cancelled(cancel_check)

            source.validate_complete()

            body = self._complete_multipart_body(parts)
            headers = {
                "Content-MD5": md5sum_hash(body),
                "Content-Type": "application/xml",
                "If-None-Match": "*",
            }
            try:
                execute(
                    "POST",
                    bucket_name,
                    object_name,
                    body=body,
                    headers=headers,
                    query_params={"uploadId": upload_id},
                    no_body_trace=True,
                )
            except Exception as exc:
                if self._is_precondition_failed(exc):
                    return False
                raise
            self._raise_if_cancelled(cancel_check)
            completed = True
            return True
        finally:
            if not completed:
                try:
                    abort(bucket_name, object_name, upload_id)
                except Exception:
                    pass

    def _upload_streaming_part(
        self,
        bucket_name,
        object_name,
        source,
        length,
        upload_id,
        part_number,
        cancel_check,
    ):
        presign = getattr(self._client, "get_presigned_url", None)
        http = getattr(self._client, "_http", None)
        if not callable(presign) or http is None:
            raise ModelPreheatS3Conflict("conditional_multipart_stream_unsupported")
        body = _BoundedReader(source, length, cancel_check)
        response = None
        try:
            url = presign(
                "PUT",
                bucket_name,
                object_name,
                expires=timedelta(minutes=5),
                extra_query_params={
                    "uploadId": upload_id,
                    "partNumber": str(part_number),
                },
            )
            response = http.urlopen(
                "PUT",
                url,
                body=body,
                headers={"Content-Length": str(length)},
                preload_content=True,
            )
            if response.status not in {200, 204}:
                raise ModelPreheatS3Conflict(
                    f"conditional_multipart_part_failed:{response.status}"
                )
            body.validate_complete()
            etag = response.headers.get("etag") or response.headers.get("ETag")
            if not etag:
                raise ModelPreheatS3Conflict("conditional_multipart_etag_missing")
            return etag.strip('"')
        finally:
            if response is not None:
                response.release_conn()

    @staticmethod
    def _complete_multipart_body(parts: list[Part]) -> bytes:
        root = ET.Element("CompleteMultipartUpload")
        for part in parts:
            element = ET.SubElement(root, "Part")
            ET.SubElement(element, "PartNumber").text = str(part.part_number)
            ET.SubElement(element, "ETag").text = f'"{part.etag}"'
        return ET.tostring(root, encoding="utf-8")

    @staticmethod
    def _close_response(response):
        close = getattr(response, "close", None)
        release_conn = getattr(response, "release_conn", None)
        if callable(close):
            close()
        if callable(release_conn):
            release_conn()

    @staticmethod
    def _raise_if_cancelled(cancel_check):
        if cancel_check is not None and cancel_check():
            raise ModelPreheatCanceled("canceled")

    @staticmethod
    def _normalized_metadata(metadata: dict) -> dict[str, str]:
        normalized = {}
        for key, value in metadata.items():
            key = key.lower()
            if key.startswith("x-amz-meta-"):
                key = key[len("x-amz-meta-") :]
            normalized[key] = value
        return normalized

    @staticmethod
    def _is_not_found(exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        return isinstance(exc, FileNotFoundError) or code in {
            "NoSuchKey",
            "NoSuchBucket",
            "NoSuchObject",
        }

    @staticmethod
    def _is_precondition_failed(exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        response = getattr(exc, "response", None)
        status = getattr(exc, "status", None) or getattr(response, "status", None)
        if code in {"PreconditionFailed", "ObjectAlreadyExists", "AlreadyExists"}:
            return True
        if code == "412" or status == 412 or status == "412":
            return True
        return "already exists" in str(exc).lower()

    @staticmethod
    def _sha256_bytes(payload: bytes) -> str:
        import hashlib

        return hashlib.sha256(payload).hexdigest()
