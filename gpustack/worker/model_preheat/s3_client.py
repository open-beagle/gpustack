import io
import json
import ssl
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
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

from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    decode_path,
    encode_path,
)
from gpustack.worker.model_preheat.manifest import (
    MAX_MANIFEST_BYTES,
    ManifestFile,
    ModelPreheatManifest,
)


GENERATION_MANIFEST_OBJECT_NAME = ".gpustack-manifest.json"
MAX_READY_BYTES = 64 * 1024
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
    def __init__(self, source, limiter, expected_length):
        self._source = source
        self._limiter = limiter
        self._expected_length = expected_length
        self._transferred = 0

    def read(self, size=-1):
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


class CancelCleanupResult(str, Enum):
    REMOVED = "removed"
    ABSENT = "absent"
    NOT_OWNED = "not_owned"


@dataclass(frozen=True)
class PublishResult:
    uploaded: int
    skipped: int
    ready_written: bool
    ready_digest: str
    generation_prefix: str


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
            secure = parsed.scheme == "https" and secure
        else:
            endpoint = endpoint.rstrip("/")
        http_client = None
        if secure and not tls_verify:
            http_client = urllib3.PoolManager(
                cert_reqs=ssl.CERT_NONE,
                assert_hostname=False,
            )
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

    def ready_object(self, prefix: str, manifest: ModelPreheatManifest) -> str:
        return self._ready_object(prefix, manifest.identity, manifest.selection_digest)

    def generation_prefix(self, prefix: str, manifest: ModelPreheatManifest) -> str:
        return self._join_object_name(
            self._selection_prefix(
                prefix, manifest.identity, manifest.selection_digest
            ),
            "generations",
            manifest.generation_id,
        )

    def generation_file_object(
        self,
        prefix: str,
        manifest: ModelPreheatManifest,
        file: ManifestFile,
    ) -> str:
        return self._join_object_name(
            self.generation_prefix(prefix, manifest),
            "files",
            file.path,
        )

    def manifest_object(self, prefix: str, manifest: ModelPreheatManifest) -> str:
        return self._join_object_name(
            self.generation_prefix(prefix, manifest),
            GENERATION_MANIFEST_OBJECT_NAME,
        )

    def publish_generation(
        self,
        bucket_name: str,
        prefix: str,
        manifest: ModelPreheatManifest,
        root_dir: str | Path,
        *,
        cancel_check=None,
        bandwidth_limit_mbps: int | None = None,
    ) -> PublishResult:
        root = Path(root_dir).resolve()
        uploaded = 0
        skipped = 0
        publish_attempt = uuid.uuid4().hex
        written_manifest = None
        written_ready = None
        self._raise_if_cancelled(cancel_check)
        try:
            existing_ready = self.read_ready_manifest(
                bucket_name,
                prefix,
                manifest.identity,
                cache_key=manifest.cache_key,
                selection_digest=manifest.selection_digest,
            )
            if existing_ready is not None and existing_ready != manifest:
                raise ReadyGenerationConflict("ready_generation_conflict")

            for file in manifest.files:
                self._raise_if_cancelled(cancel_check)
                local_path = self._local_manifest_path(root, file)
                object_name = self.generation_file_object(prefix, manifest, file)
                if self._object_matches(
                    bucket_name, object_name, file.size, file.sha256
                ):
                    skipped += 1
                    continue
                written = self._put_file_if_absent(
                    bucket_name,
                    object_name,
                    local_path,
                    metadata={
                        "sha256": file.sha256,
                        "model-preheat-digest": manifest.digest,
                        "model-preheat-publish-attempt": publish_attempt,
                    },
                    cancel_check=cancel_check,
                    bandwidth_limit_mbps=bandwidth_limit_mbps,
                )
                self._raise_if_cancelled(cancel_check)
                if not written:
                    if self._object_matches(
                        bucket_name, object_name, file.size, file.sha256
                    ):
                        skipped += 1
                        continue
                    raise ModelPreheatS3Conflict("object_content_conflict")
                uploaded += 1

            for file in manifest.files:
                object_name = self.generation_file_object(prefix, manifest, file)
                if not self._object_matches(
                    bucket_name, object_name, file.size, file.sha256
                ):
                    raise ModelPreheatS3Conflict("object_content_conflict")

            self._raise_if_cancelled(cancel_check)
            manifest_bytes = manifest.to_json_bytes()
            manifest_sha256 = self._sha256_bytes(manifest_bytes)
            manifest_object = self.manifest_object(prefix, manifest)
            if self._object_matches(
                bucket_name, manifest_object, len(manifest_bytes), manifest_sha256
            ):
                skipped += 1
            else:
                written_manifest = self._put_bytes_if_absent(
                    bucket_name,
                    manifest_object,
                    manifest_bytes,
                    content_type="application/json",
                    metadata={
                        "sha256": manifest_sha256,
                        "model-preheat-digest": manifest.digest,
                        "model-preheat-publish-attempt": publish_attempt,
                    },
                )
                self._raise_if_cancelled(cancel_check)
                if not written_manifest:
                    if not self._object_matches(
                        bucket_name,
                        manifest_object,
                        len(manifest_bytes),
                        manifest_sha256,
                    ):
                        raise ModelPreheatS3Conflict("object_content_conflict")
                    skipped += 1
                else:
                    uploaded += 1

            if not self._object_matches(
                bucket_name, manifest_object, len(manifest_bytes), manifest_sha256
            ):
                raise ModelPreheatS3Conflict("object_content_conflict")
            self._raise_if_cancelled(cancel_check)
            ready_object = self.ready_object(prefix, manifest)
            existing_ready = self.read_ready_manifest(
                bucket_name,
                prefix,
                manifest.identity,
                cache_key=manifest.cache_key,
                selection_digest=manifest.selection_digest,
            )
            ready_payload = self._ready_payload(prefix, manifest, manifest_sha256)
            if existing_ready is not None:
                if existing_ready != manifest:
                    raise ReadyGenerationConflict("ready_generation_conflict")
                ready_written = False
            else:
                ready_written = self._put_ready_if_absent(
                    bucket_name,
                    ready_object,
                    ready_payload,
                    metadata={
                        "model-preheat-digest": manifest.digest,
                        "model-preheat-publish-attempt": publish_attempt,
                        "sha256": self._sha256_bytes(ready_payload),
                    },
                    ready_payload=ready_payload,
                )
                written_ready = ready_written
                self._raise_if_cancelled(cancel_check)

            return PublishResult(
                uploaded=uploaded,
                skipped=skipped,
                ready_written=ready_written,
                ready_digest=manifest.digest,
                generation_prefix=self.generation_prefix(prefix, manifest),
            )
        except ModelPreheatCanceled:
            cleanup_objects = []
            if written_ready:
                cleanup_objects.append((ready_object, ready_payload))
            if written_manifest:
                cleanup_objects.append((manifest_object, manifest_bytes))
            cleanup_succeeded = True
            for object_name, payload in cleanup_objects:
                if not self._retry_remove_if_owned(
                    bucket_name, object_name, payload, publish_attempt
                ):
                    cleanup_succeeded = False
            if not cleanup_succeeded:
                raise ModelPreheatS3Conflict("cancel_cleanup_failed") from None
            raise

    def head_object(self, bucket_name: str, object_name: str) -> dict | None:
        try:
            stat = self._client.stat_object(bucket_name, object_name)
        except Exception as exc:
            if self._is_not_found(exc):
                return None
            raise
        metadata = self._normalized_metadata(getattr(stat, "metadata", {}) or {})
        return {
            "size": getattr(stat, "size", None),
            "sha256": metadata.get("sha256"),
        }

    def list_objects(self, bucket_name: str, prefix: str) -> list[str]:
        return sorted(self.iter_objects(bucket_name, prefix))

    def iter_objects(self, bucket_name: str, prefix: str, *, max_objects=None):
        objects = self._client.list_objects(bucket_name, prefix=prefix, recursive=True)
        count = 0
        for item in objects:
            object_name = getattr(item, "object_name", None)
            if object_name is None:
                continue
            count += 1
            if max_objects is not None and count > max_objects:
                raise ModelPreheatS3ManifestError("s3_inventory_object_limit")
            yield object_name

    def remove_object(self, bucket_name: str, object_name: str):
        try:
            self._client.remove_object(bucket_name, object_name)
        except Exception as exc:
            if not self._is_not_found(exc):
                raise

    def stream_object(
        self,
        bucket_name: str,
        object_name: str,
        chunk_size: int = 1024 * 1024,
        bandwidth_limit_mbps: int | None = None,
    ):
        response = self._client.get_object(bucket_name, object_name)
        limiter = _BandwidthLimiter(bandwidth_limit_mbps)
        if limiter.enabled:
            chunk_size = min(chunk_size, RATE_LIMIT_CHUNK_SIZE)
        try:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    return
                limiter.consume(len(chunk))
                yield chunk
        finally:
            self._close_response(response)

    def download_generation_file(
        self,
        bucket_name: str,
        prefix: str,
        manifest: ModelPreheatManifest,
        file: ManifestFile,
        target_path: str | Path,
        *,
        bandwidth_limit_mbps: int | None = None,
    ):
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        object_name = self.generation_file_object(prefix, manifest, file)
        with target.open("wb") as output:
            for chunk in self.stream_object(
                bucket_name,
                object_name,
                bandwidth_limit_mbps=bandwidth_limit_mbps,
            ):
                output.write(chunk)

    def read_ready(
        self,
        bucket_name: str,
        prefix: str,
        identity,
        selection_digest: str,
    ) -> dict | None:
        return self._read_ready(
            bucket_name,
            self._ready_object(prefix, identity, selection_digest),
        )

    def read_ready_manifest(
        self,
        bucket_name: str,
        prefix: str,
        identity,
        *,
        cache_key: str,
        selection_digest: str,
    ) -> ModelPreheatManifest | None:
        try:
            ready = self.read_ready(bucket_name, prefix, identity, selection_digest)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelPreheatS3ManifestError("s3_manifest_invalid") from exc
        if ready is None:
            return None
        if (
            not isinstance(ready, dict)
            or ready.get("identity_digest") != identity.digest
        ):
            raise ModelPreheatS3ManifestError("s3_manifest_invalid")
        manifest_name = ready.get("manifest_object")
        ready_generation_id = ready.get("generation_id")
        if (
            not isinstance(manifest_name, str)
            or not self._is_sha256(ready.get("manifest_sha256"))
            or not self._is_safe_identifier(ready_generation_id)
            or not self._is_safe_identifier(ready.get("cache_key"))
            or not self._is_safe_identifier(ready.get("selection_digest"))
        ):
            raise ModelPreheatS3ManifestError("s3_manifest_invalid")
        expected_generation_prefix = self._join_object_name(
            self._selection_prefix(prefix, identity, selection_digest),
            "generations",
            ready_generation_id,
        )
        expected_manifest_name = self._join_object_name(
            expected_generation_prefix, GENERATION_MANIFEST_OBJECT_NAME
        )
        if (
            ready.get("generation_prefix") != expected_generation_prefix
            or manifest_name != expected_manifest_name
        ):
            raise ModelPreheatS3ManifestError("s3_manifest_invalid")
        try:
            manifest_bytes = self._read_object_bytes(bucket_name, manifest_name)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelPreheatS3ManifestError("s3_manifest_invalid") from exc
        if manifest_bytes is None or self._sha256_bytes(manifest_bytes) != ready.get(
            "manifest_sha256"
        ):
            raise ModelPreheatS3ManifestError("s3_manifest_invalid")
        try:
            manifest = self._manifest_from_payload(
                json.loads(manifest_bytes.decode("utf-8"))
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelPreheatS3ManifestError("s3_manifest_invalid") from exc
        if (
            manifest.identity != identity
            or ready.get("digest") != manifest.digest
            or ready.get("cache_key") != manifest.cache_key
            or ready.get("selection_digest") != manifest.selection_digest
            or ready_generation_id != manifest.generation_id
            or ready.get("generation_prefix")
            != self.generation_prefix(prefix, manifest)
            or manifest_name != self.manifest_object(prefix, manifest)
            or ready.get("files") != len(manifest.files)
            or ready.get("total_size") != manifest.total_size
            or ready.get("schema_version") != 1
        ):
            raise ModelPreheatS3ManifestError("s3_manifest_invalid")
        if (
            manifest.cache_key != cache_key
            or manifest.selection_digest != selection_digest
        ):
            raise ModelPreheatS3ManifestError("s3_manifest_invalid")
        return manifest

    @staticmethod
    def _encoded_prefix(prefix: str) -> str:
        stripped = prefix.strip("/")
        if stripped == "":
            return ""
        return encode_path(stripped)

    def _selection_prefix(self, prefix: str, identity, selection_digest: str) -> str:
        if not self._is_safe_identifier(selection_digest):
            raise ModelPreheatS3ManifestError("s3_manifest_invalid")
        return self._join_object_name(
            self._encoded_prefix(prefix),
            "model-cache",
            "v1",
            identity.source,
            identity.model_path,
            identity.revision_path,
            selection_digest,
        )

    def _ready_object(self, prefix: str, identity, selection_digest: str) -> str:
        return self._join_object_name(
            self._selection_prefix(prefix, identity, selection_digest),
            "ready.json",
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

    def _read_ready(self, bucket_name: str, object_name: str) -> dict | None:
        return self._read_json_object(bucket_name, object_name)

    def _read_json_object(self, bucket_name: str, object_name: str) -> dict | None:
        payload = self._read_object_bytes(
            bucket_name, object_name, max_bytes=MAX_READY_BYTES
        )
        if payload is None:
            return None
        return json.loads(payload.decode("utf-8"))

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

    def _object_matches(
        self,
        bucket_name: str,
        object_name: str,
        size: int,
        sha256: str,
    ) -> bool:
        try:
            stat = self._client.stat_object(bucket_name, object_name)
        except Exception as exc:
            if self._is_not_found(exc):
                return False
            raise
        metadata = self._normalized_metadata(getattr(stat, "metadata", {}) or {})
        return getattr(stat, "size", None) == size and metadata.get("sha256") == sha256

    def _manifest_from_payload(self, payload: dict) -> ModelPreheatManifest:
        try:
            identity_payload = payload["identity"]
            identity = ModelPreheatIdentity(
                source=identity_payload["source"],
                model_id=decode_path(identity_payload["model_id"]),
                revision=decode_path(identity_payload["revision"]),
                file_patterns=tuple(
                    decode_path(pattern)
                    for pattern in identity_payload["file_patterns"]
                ),
            )
            manifest = ModelPreheatManifest(
                identity=identity,
                files=tuple(
                    ManifestFile(
                        path=file["path"],
                        size=file["size"],
                        sha256=file["sha256"],
                    )
                    for file in payload["files"]
                ),
                cache_key=payload["cache_key"],
                selection_digest=payload["selection_digest"],
                generation_id=payload["generation_id"],
                exclude_patterns=tuple(
                    decode_path(pattern)
                    for pattern in payload.get("exclude_patterns", [])
                ),
                requested_revision=decode_path(
                    payload.get("requested_revision", identity.revision_path)
                ),
                schema_version=payload.get("schema_version", 1),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelPreheatS3ManifestError("s3_manifest_invalid") from exc
        if payload != manifest.to_dict():
            raise ModelPreheatS3ManifestError("s3_manifest_invalid")
        return manifest

    def _ensure_no_conflicting_object(self, bucket_name: str, object_name: str):
        try:
            self._client.stat_object(bucket_name, object_name)
        except Exception as exc:
            if self._is_not_found(exc):
                return
            raise
        raise ModelPreheatS3Conflict(f"object_content_conflict:{object_name}")

    def _put_json_bytes(
        self,
        bucket_name: str,
        object_name: str,
        payload: bytes,
        metadata: dict[str, str] | None = None,
    ):
        self._client.put_object(
            bucket_name,
            object_name,
            io.BytesIO(payload),
            len(payload),
            content_type="application/json",
            metadata=metadata or {},
        )

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

    def _remove_if_owned(
        self, bucket_name: str, object_name: str, payload: bytes, attempt: str
    ) -> CancelCleanupResult:
        try:
            stat = self._client.stat_object(bucket_name, object_name)
        except Exception as exc:
            if self._is_not_found(exc):
                return CancelCleanupResult.ABSENT
            raise
        metadata = self._normalized_metadata(getattr(stat, "metadata", {}) or {})
        if (
            getattr(stat, "size", None) != len(payload)
            or metadata.get("sha256") != self._sha256_bytes(payload)
            or metadata.get("model-preheat-publish-attempt") != attempt
            or self._read_object_bytes(bucket_name, object_name) != payload
        ):
            return CancelCleanupResult.NOT_OWNED
        self._client.remove_object(bucket_name, object_name)
        if not self._cleanup_object_absent(bucket_name, object_name):
            raise ModelPreheatS3Conflict("cancel_cleanup_not_confirmed")
        return CancelCleanupResult.REMOVED

    def _retry_remove_if_owned(
        self, bucket_name: str, object_name: str, payload: bytes, attempt: str
    ) -> bool:
        for retry in range(self._cancel_cleanup_attempts):
            try:
                self._remove_if_owned(bucket_name, object_name, payload, attempt)
                return True
            except Exception:
                if retry + 1 == self._cancel_cleanup_attempts:
                    return False
                self._cancel_cleanup_sleep(self._cancel_cleanup_backoff * (2**retry))
        return False

    def _cleanup_object_absent(self, bucket_name: str, object_name: str) -> bool:
        try:
            self._client.stat_object(bucket_name, object_name)
            return False
        except Exception as exc:
            if not self._is_not_found(exc):
                raise
        return self._read_object_bytes(bucket_name, object_name) is None

    def _put_ready_if_absent(
        self,
        bucket_name: str,
        object_name: str,
        payload: bytes,
        metadata: dict[str, str],
        ready_payload: bytes,
    ) -> bool:
        written = self._put_stream_if_absent(
            bucket_name,
            object_name,
            io.BytesIO(payload),
            len(payload),
            "application/json",
            metadata,
        )
        if written:
            return True
        existing_ready = self._read_ready(bucket_name, object_name)
        if existing_ready == json.loads(ready_payload.decode("utf-8")):
            return False
        raise ReadyGenerationConflict("ready_generation_conflict")

    def _ready_payload(
        self,
        prefix: str,
        manifest: ModelPreheatManifest,
        manifest_sha256: str,
    ) -> bytes:
        payload = {
            "digest": manifest.digest,
            "cache_key": manifest.cache_key,
            "files": len(manifest.files),
            "generation_prefix": self.generation_prefix(prefix, manifest),
            "generation_id": manifest.generation_id,
            "identity_digest": manifest.identity.digest,
            "manifest_object": self.manifest_object(prefix, manifest),
            "manifest_sha256": manifest_sha256,
            "selection_digest": manifest.selection_digest,
            "schema_version": 1,
            "total_size": manifest.total_size,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

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

    @staticmethod
    def _is_sha256(value) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
        )

    @staticmethod
    def _is_safe_identifier(value) -> bool:
        return (
            isinstance(value, str)
            and 0 < len(value) <= 256
            and "/" not in value
            and "\\" not in value
            and all(ord(char) >= 32 and ord(char) != 127 for char in value)
        )
