import json
from dataclasses import dataclass, field

import pytest

from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
from gpustack.worker.model_preheat.manifest import (
    ManifestFile,
    ModelPreheatManifest,
    build_model_preheat_manifest,
)
from gpustack.worker.model_preheat.s3_client import (
    ModelPreheatS3Client,
    ReadyGenerationConflict,
)


@dataclass
class StoredObject:
    data: bytes
    metadata: dict[str, str] = field(default_factory=dict)
    etag: str = '"multipart-etag-2"'

    @property
    def size(self) -> int:
        return len(self.data)

    def read(self, length: int | None = None) -> bytes:
        if length is None:
            return self.data
        return self.data[:length]

    def close(self):
        pass

    def release_conn(self):
        pass


class InMemoryMinio:
    def __init__(self):
        self.objects: dict[tuple[str, str], StoredObject] = {}
        self.puts: list[str] = []

    def stat_object(self, bucket_name, object_name):
        try:
            return self.objects[(bucket_name, object_name)]
        except KeyError as exc:
            raise FileNotFoundError(object_name) from exc

    def get_object(self, bucket_name, object_name):
        return self.stat_object(bucket_name, object_name)

    def fput_object(self, bucket_name, object_name, file_path, metadata=None):
        self.puts.append(object_name)
        with open(file_path, "rb") as file:
            self.objects[(bucket_name, object_name)] = StoredObject(
                file.read(),
                metadata=metadata or {},
            )

    def put_object(
        self,
        bucket_name,
        object_name,
        data,
        length,
        content_type=None,
        metadata=None,
    ):
        del content_type
        self.puts.append(object_name)
        payload = data.read(length)
        self.objects[(bucket_name, object_name)] = StoredObject(
            payload,
            metadata=metadata or {},
        )

    def put_object_if_absent(
        self,
        bucket_name,
        object_name,
        data,
        length,
        content_type=None,
        metadata=None,
    ):
        if (bucket_name, object_name) in self.objects:
            return False
        self.put_object(
            bucket_name,
            object_name,
            data,
            length,
            content_type=content_type,
            metadata=metadata,
        )
        return True


class _S3CodeError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class BasicMinio:
    def __init__(self):
        self.objects: dict[tuple[str, str], StoredObject] = {}
        self.puts: list[str] = []

    def stat_object(self, bucket_name, object_name):
        try:
            return self.objects[(bucket_name, object_name)]
        except KeyError as exc:
            raise FileNotFoundError(object_name) from exc

    def get_object(self, bucket_name, object_name):
        return self.stat_object(bucket_name, object_name)

    def fput_object(self, bucket_name, object_name, file_path, metadata=None):
        self.puts.append(object_name)
        with open(file_path, "rb") as file:
            self.objects[(bucket_name, object_name)] = StoredObject(
                file.read(),
                metadata=metadata or {},
            )

    def put_object(
        self,
        bucket_name,
        object_name,
        data,
        length,
        content_type=None,
        metadata=None,
    ):
        del content_type
        self.puts.append(object_name)
        payload = data.read(length)
        self.objects[(bucket_name, object_name)] = StoredObject(
            payload,
            metadata=metadata or {},
        )


class ExecuteConditionalMinio(BasicMinio):
    def __init__(self):
        super().__init__()
        self.execute_calls: list[dict] = []
        self.concurrent_ready_digest: str | None = None

    def _execute(
        self,
        method,
        bucket_name=None,
        object_name=None,
        body=None,
        headers=None,
        query_params=None,
        preload_content=True,
        no_body_trace=False,
    ):
        del query_params, preload_content, no_body_trace
        headers = headers or {}
        self.execute_calls.append(
            {
                "method": method,
                "bucket_name": bucket_name,
                "object_name": object_name,
                "body": body,
                "headers": headers,
            }
        )
        if self.concurrent_ready_digest is not None:
            self.objects[(bucket_name, object_name)] = StoredObject(
                json.dumps({"digest": self.concurrent_ready_digest}).encode("utf-8")
            )
        if (
            headers.get("If-None-Match") == "*"
            and (bucket_name, object_name) in self.objects
        ):
            raise _S3CodeError("PreconditionFailed")
        assert isinstance(body, bytes)
        self.objects[(bucket_name, object_name)] = StoredObject(
            body,
            metadata={
                key[len("x-amz-meta-") :]: value
                for key, value in headers.items()
                if key.startswith("x-amz-meta-")
            },
        )


class NoConditionalMinio(BasicMinio):
    pass


def _manifest(tmp_path, content: bytes = b"weights"):
    (tmp_path / "model.bin").write_bytes(content)
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="main",
        file_patterns=["model.bin"],
    )
    return build_model_preheat_manifest(tmp_path, identity)


def test_duplicate_upload_skip_uses_size_and_sha256_metadata_not_etag(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    file_object = client.generation_file_object("preheat", manifest, manifest.files[0])
    minio.objects[("bucket", file_object)] = StoredObject(
        b"weights",
        metadata={"sha256": manifest.files[0].sha256},
        etag='"not-a-content-sha-2"',
    )

    result = client.publish_generation("bucket", "preheat", manifest, tmp_path)

    assert result.skipped == 1
    assert file_object not in minio.puts


def test_generation_manifest_uses_protocol_object_name(tmp_path):
    manifest = _manifest(tmp_path)
    client = ModelPreheatS3Client(InMemoryMinio())

    assert client.manifest_object("preheat", manifest).endswith(
        "/.gpustack-manifest.json"
    )


def test_publish_repairs_missing_ready_after_generation_was_uploaded(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)

    first = client.publish_generation("bucket", "preheat", manifest, tmp_path)
    ready_object = client.ready_object("preheat", manifest.identity)
    del minio.objects[("bucket", ready_object)]

    second = client.publish_generation("bucket", "preheat", manifest, tmp_path)

    ready = json.loads(minio.objects[("bucket", ready_object)].data)
    assert first.ready_written is True
    assert second.ready_written is True
    assert second.skipped == 2
    assert ready["digest"] == manifest.digest


def test_same_ready_digest_is_idempotent_success(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)

    client.publish_generation("bucket", "preheat", manifest, tmp_path)
    result = client.publish_generation("bucket", "preheat", manifest, tmp_path)

    assert result.ready_written is False
    assert result.ready_digest == manifest.digest


def test_ready_conflict_rejects_different_digest(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    ready_object = client.ready_object("preheat", manifest.identity)
    minio.objects[("bucket", ready_object)] = StoredObject(
        json.dumps({"digest": "different"}).encode("utf-8")
    )

    with pytest.raises(ReadyGenerationConflict, match="ready_generation_conflict"):
        client.publish_generation("bucket", "preheat", manifest, tmp_path)


def test_ready_conflict_detected_after_generation_upload(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    ready_object = client.ready_object("preheat", manifest.identity)
    file_object = client.generation_file_object("preheat", manifest, manifest.files[0])

    def inject_conflict(bucket_name, object_name, file_path, metadata=None):
        InMemoryMinio.fput_object(minio, bucket_name, object_name, file_path, metadata)
        if object_name == file_object:
            minio.objects[(bucket_name, ready_object)] = StoredObject(
                json.dumps({"digest": "other-generation"}).encode("utf-8")
            )

    minio.fput_object = inject_conflict

    with pytest.raises(ReadyGenerationConflict, match="ready_generation_conflict"):
        client.publish_generation("bucket", "preheat", manifest, tmp_path)


def test_ready_final_write_does_not_overwrite_concurrent_digest(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    ready_object = client.ready_object("preheat", manifest.identity)

    def inject_conflict_before_conditional_put(
        bucket_name,
        object_name,
        data,
        length,
        content_type=None,
        metadata=None,
    ):
        if object_name == ready_object:
            minio.objects[(bucket_name, object_name)] = StoredObject(
                json.dumps({"digest": "other-generation"}).encode("utf-8")
            )
        return InMemoryMinio.put_object_if_absent(
            minio,
            bucket_name,
            object_name,
            data,
            length,
            content_type=content_type,
            metadata=metadata,
        )

    minio.put_object_if_absent = inject_conflict_before_conditional_put

    with pytest.raises(ReadyGenerationConflict, match="ready_generation_conflict"):
        client.publish_generation("bucket", "preheat", manifest, tmp_path)

    ready = json.loads(minio.objects[("bucket", ready_object)].data)
    assert ready["digest"] == "other-generation"


def test_ready_write_uses_minio_execute_condition_and_keeps_concurrent_digest(tmp_path):
    manifest = _manifest(tmp_path)
    minio = ExecuteConditionalMinio()
    minio.concurrent_ready_digest = "other-generation"
    client = ModelPreheatS3Client(minio)
    ready_object = client.ready_object("preheat", manifest.identity)

    with pytest.raises(ReadyGenerationConflict, match="ready_generation_conflict"):
        client.publish_generation("bucket", "preheat", manifest, tmp_path)

    ready = json.loads(minio.objects[("bucket", ready_object)].data)
    assert ready["digest"] == "other-generation"
    assert minio.execute_calls == [
        {
            "method": "PUT",
            "bucket_name": "bucket",
            "object_name": ready_object,
            "body": minio.execute_calls[0]["body"],
            "headers": {
                "Content-Type": "application/json",
                "If-None-Match": "*",
                "x-amz-meta-model-preheat-digest": manifest.digest,
            },
        }
    ]
    assert json.loads(minio.execute_calls[0]["body"])["digest"] == manifest.digest


def test_ready_write_without_conditional_capability_does_not_put_ready(tmp_path):
    manifest = _manifest(tmp_path)
    minio = NoConditionalMinio()
    client = ModelPreheatS3Client(minio)
    ready_object = client.ready_object("preheat", manifest.identity)

    with pytest.raises(ReadyGenerationConflict, match="ready_generation_conflict"):
        client.publish_generation("bucket", "preheat", manifest, tmp_path)

    assert ("bucket", ready_object) not in minio.objects
    assert ready_object not in minio.puts


def test_publish_rechecks_manifest_paths_stay_under_root(tmp_path):
    (tmp_path / "safe.bin").write_bytes(b"safe")
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="main",
        file_patterns=["safe.bin"],
    )
    unsafe_file = object.__new__(ManifestFile)
    object.__setattr__(unsafe_file, "path", "%2E%2E/secret.bin")
    object.__setattr__(unsafe_file, "size", 6)
    object.__setattr__(unsafe_file, "sha256", "0" * 64)
    manifest = object.__new__(ModelPreheatManifest)
    object.__setattr__(manifest, "identity", identity)
    object.__setattr__(manifest, "files", (unsafe_file,))
    object.__setattr__(manifest, "schema_version", 1)

    with pytest.raises(ValueError, match="manifest_path_escape"):
        ModelPreheatS3Client(InMemoryMinio()).publish_generation(
            "bucket",
            "preheat",
            manifest,
            tmp_path,
        )
