import io
import json
from dataclasses import dataclass
from pathlib import Path

from minio import Minio

from gpustack.worker.model_preheat.identity import decode_path, encode_path
from gpustack.worker.model_preheat.manifest import ManifestFile, ModelPreheatManifest


GENERATION_MANIFEST_OBJECT_NAME = ".gpustack-manifest.json"


class ReadyGenerationConflict(RuntimeError):
    pass


class ModelPreheatS3Conflict(RuntimeError):
    pass


@dataclass(frozen=True)
class PublishResult:
    uploaded: int
    skipped: int
    ready_written: bool
    ready_digest: str
    generation_prefix: str


class ModelPreheatS3Client:
    def __init__(self, client):
        self._client = client

    @classmethod
    def from_minio(
        cls,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
        region: str | None = None,
    ):
        return cls(
            Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
                region=region,
            )
        )

    def ready_object(self, prefix: str, identity) -> str:
        return self._join_object_name(
            self._encoded_prefix(prefix),
            identity.storage_prefix,
            "ready.json",
        )

    def generation_prefix(self, prefix: str, manifest: ModelPreheatManifest) -> str:
        return self._join_object_name(
            self._encoded_prefix(prefix),
            manifest.identity.storage_prefix,
            "generations",
            manifest.digest,
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
    ) -> PublishResult:
        root = Path(root_dir).resolve()
        uploaded = 0
        skipped = 0

        existing_ready = self._read_ready(
            bucket_name, self.ready_object(prefix, manifest.identity)
        )
        if (
            existing_ready is not None
            and existing_ready.get("digest") != manifest.digest
        ):
            raise ReadyGenerationConflict("ready_generation_conflict")

        for file in manifest.files:
            local_path = self._local_manifest_path(root, file)
            object_name = self.generation_file_object(prefix, manifest, file)
            if self._object_matches(bucket_name, object_name, file.size, file.sha256):
                skipped += 1
                continue
            self._ensure_no_conflicting_object(bucket_name, object_name)
            self._client.fput_object(
                bucket_name,
                object_name,
                str(local_path),
                metadata={
                    "sha256": file.sha256,
                    "model-preheat-digest": manifest.digest,
                },
            )
            uploaded += 1

        manifest_bytes = manifest.to_json_bytes()
        manifest_sha256 = self._sha256_bytes(manifest_bytes)
        manifest_object = self.manifest_object(prefix, manifest)
        if self._object_matches(
            bucket_name,
            manifest_object,
            len(manifest_bytes),
            manifest_sha256,
        ):
            skipped += 1
        else:
            self._ensure_no_conflicting_object(bucket_name, manifest_object)
            self._put_json_bytes(
                bucket_name,
                manifest_object,
                manifest_bytes,
                metadata={
                    "sha256": manifest_sha256,
                    "model-preheat-digest": manifest.digest,
                },
            )
            uploaded += 1

        ready_object = self.ready_object(prefix, manifest.identity)
        existing_ready = self._read_ready(bucket_name, ready_object)
        if existing_ready is not None:
            if existing_ready.get("digest") != manifest.digest:
                raise ReadyGenerationConflict("ready_generation_conflict")
            ready_written = False
        else:
            ready_payload = self._ready_payload(prefix, manifest)
            ready_written = self._put_ready_if_absent(
                bucket_name,
                ready_object,
                ready_payload,
                metadata={"model-preheat-digest": manifest.digest},
                manifest_digest=manifest.digest,
            )

        return PublishResult(
            uploaded=uploaded,
            skipped=skipped,
            ready_written=ready_written,
            ready_digest=manifest.digest,
            generation_prefix=self.generation_prefix(prefix, manifest),
        )

    @staticmethod
    def _encoded_prefix(prefix: str) -> str:
        stripped = prefix.strip("/")
        if stripped == "":
            return ""
        return encode_path(stripped)

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
        try:
            response = self._client.get_object(bucket_name, object_name)
        except Exception as exc:
            if self._is_not_found(exc):
                return None
            raise
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            close = getattr(response, "close", None)
            release_conn = getattr(response, "release_conn", None)
            if close:
                close()
            if release_conn:
                release_conn()

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

    def _put_ready_if_absent(
        self,
        bucket_name: str,
        object_name: str,
        payload: bytes,
        metadata: dict[str, str],
        manifest_digest: str,
    ) -> bool:
        put_if_absent = getattr(self._client, "put_object_if_absent", None)
        if callable(put_if_absent):
            written = put_if_absent(
                bucket_name,
                object_name,
                io.BytesIO(payload),
                len(payload),
                content_type="application/json",
                metadata=metadata,
            )
            if written:
                return True
            existing_ready = self._read_ready(bucket_name, object_name)
            if (
                existing_ready is not None
                and existing_ready.get("digest") == manifest_digest
            ):
                return False
            raise ReadyGenerationConflict("ready_generation_conflict")

        execute = getattr(self._client, "_execute", None)
        if callable(execute):
            headers = {
                "Content-Type": "application/json",
                "If-None-Match": "*",
            }
            for key, value in metadata.items():
                headers[f"x-amz-meta-{key}"] = value
            try:
                execute(
                    "PUT",
                    bucket_name,
                    object_name,
                    body=payload,
                    headers=headers,
                    no_body_trace=True,
                )
                return True
            except Exception as exc:
                if not self._is_precondition_failed(exc):
                    raise
                existing_ready = self._read_ready(bucket_name, object_name)
                if (
                    existing_ready is not None
                    and existing_ready.get("digest") == manifest_digest
                ):
                    return False
            raise ReadyGenerationConflict("ready_generation_conflict")

        raise ReadyGenerationConflict("ready_generation_conflict")

    def _ready_payload(self, prefix: str, manifest: ModelPreheatManifest) -> bytes:
        payload = {
            "digest": manifest.digest,
            "files": len(manifest.files),
            "generation_prefix": self.generation_prefix(prefix, manifest),
            "identity_digest": manifest.identity.digest,
            "manifest_object": self.manifest_object(prefix, manifest),
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
