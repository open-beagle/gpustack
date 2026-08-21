import json
import io
import hashlib
import ssl
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from minio import Minio

from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    ModelPreheatIdentityError,
)
from gpustack.worker.model_preheat.manifest import (
    ManifestFile,
    ModelPreheatManifest,
    build_model_preheat_manifest,
)
from gpustack.worker.model_preheat.s3_client import (
    ModelPreheatCanceled,
    ModelPreheatS3ManifestConflict,
    ModelPreheatS3Client,
    ModelPreheatS3Conflict,
    ModelPreheatS3ManifestError,
)


@dataclass
class StoredObject:
    data: bytes
    metadata: dict[str, str] = field(default_factory=dict)
    etag: str = '"multipart-etag-2"'

    @property
    def size(self) -> int:
        return len(self.data)


class ObjectResponse(io.BytesIO):
    def release_conn(self):
        pass


class InMemoryMinio:
    """内存对象存储；实现 `put_object_if_absent` 用于条件写路径。

    该存储用于验证条件写“已存在→返回 False（等价 412）→重读收敛”的
    控制流；它不是对真实 MinIO 行为的冒充分配，条件写的底层 412
    语义由 `ExecuteConditionalMinio`/`MultipartConditionalMinio` 覆盖。
    """

    def __init__(self):
        self.objects: dict[tuple[str, str], StoredObject] = {}
        self.puts: list[str] = []

    def stat_object(self, bucket_name, object_name):
        try:
            return self.objects[(bucket_name, object_name)]
        except KeyError as exc:
            raise FileNotFoundError(object_name) from exc

    def get_object(self, bucket_name, object_name):
        return ObjectResponse(self.stat_object(bucket_name, object_name).data)

    def remove_object(self, bucket_name, object_name):
        self.objects.pop((bucket_name, object_name), None)

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
        chunks = []
        remaining = length
        while remaining:
            chunk = data.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
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
        return ObjectResponse(self.stat_object(bucket_name, object_name).data)

    def remove_object(self, bucket_name, object_name):
        self.objects.pop((bucket_name, object_name), None)

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
    """通过 presigned + `If-None-Match` 头模拟真实条件写（含 412）。"""

    def __init__(self):
        super().__init__()
        self.execute_calls: list[dict] = []
        self._http = self

    def presigned_put_object(self, bucket_name, object_name, expires):
        assert expires.total_seconds() == 300
        return f"s3://{bucket_name}/{object_name}"

    def urlopen(self, method, url, body=None, headers=None, **kwargs):
        del kwargs
        bucket_name, object_name = url.removeprefix("s3://").split("/", 1)
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
        if (
            headers.get("If-None-Match") == "*"
            and (bucket_name, object_name) in self.objects
        ):
            return SimpleNamespace(status=412, release_conn=lambda: None)
        chunks = []
        while True:
            chunk = body.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        self.objects[(bucket_name, object_name)] = StoredObject(
            payload,
            metadata={
                key[len("x-amz-meta-") :]: value
                for key, value in headers.items()
                if key.startswith("x-amz-meta-")
            },
        )
        return SimpleNamespace(status=200, release_conn=lambda: None)


class NoConditionalMinio(BasicMinio):
    pass


class SdkLikeConditionalMinio:
    def __init__(self):
        self.calls = []
        self._http = self

    def presigned_put_object(self, bucket_name, object_name, expires):
        assert expires.total_seconds() == 300
        return f"https://s3.test/{bucket_name}/{object_name}?signed=true"

    def urlopen(self, method, url, body=None, headers=None, **kwargs):
        del kwargs
        chunks = []
        while True:
            chunk = body.read(2 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        hashlib.sha256(payload).hexdigest()
        self.calls.append((method, url, payload, headers, chunks))
        return SimpleNamespace(status=200, release_conn=lambda: None)


class MultipartConditionalMinio:
    def __init__(self, cancel_event=None, fail_complete=False):
        self.cancel_event = cancel_event
        self.fail_complete = fail_complete
        self.created = []
        self.parts = []
        self.completed = []
        self.aborted = []

    def _create_multipart_upload(self, bucket_name, object_name, headers):
        self.created.append((bucket_name, object_name, headers))
        return "upload-id"

    def _upload_part(
        self, bucket_name, object_name, data, headers, upload_id, part_number
    ):
        assert isinstance(data, bytes)
        self.parts.append((part_number, len(data)))
        if self.cancel_event is not None:
            self.cancel_event.set()
        return f"etag-{part_number}"

    def _execute(
        self,
        method,
        bucket_name,
        object_name,
        body=None,
        headers=None,
        query_params=None,
        **kwargs,
    ):
        del kwargs
        self.completed.append(
            (method, bucket_name, object_name, body, headers, query_params)
        )
        if self.fail_complete:
            raise _S3CodeError("PreconditionFailed")

    def _abort_multipart_upload(self, bucket_name, object_name, upload_id):
        self.aborted.append((bucket_name, object_name, upload_id))


class StreamingMultipartConditionalMinio(MultipartConditionalMinio):
    def __init__(self, clock):
        super().__init__()
        self._http = self
        self.clock = clock
        self.sent_chunks = []

    def _upload_part(self, *args, **kwargs):
        raise AssertionError("限速 multipart 不应把完整 part 交给 _upload_part")

    def get_presigned_url(
        self,
        method,
        bucket_name,
        object_name,
        *,
        expires,
        extra_query_params,
    ):
        assert method == "PUT"
        assert expires.total_seconds() == 300
        return (
            f"s3://{bucket_name}/{object_name}"
            f"?uploadId={extra_query_params['uploadId']}"
            f"&partNumber={extra_query_params['partNumber']}"
        )

    def urlopen(self, method, url, body, headers, **kwargs):
        del kwargs
        assert method == "PUT"
        assert int(headers["Content-Length"]) > 0
        while True:
            chunk = body.read(2 * 1024 * 1024)
            if not chunk:
                break
            self.sent_chunks.append((self.clock[0], len(chunk), url))
        part_number = url.rsplit("partNumber=", 1)[1]
        return SimpleNamespace(
            status=200,
            headers={"etag": f'"etag-{part_number}"'},
            release_conn=lambda: None,
        )


def _artifact_manifest(
    tmp_path, revision: str = "8f73c6a91b", content: bytes = b"weights"
):
    (tmp_path / "model.bin").write_bytes(content)
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision=revision,
        file_patterns=["model.bin"],
        requested_revision="master",
    )
    return build_model_preheat_manifest(tmp_path, identity)


PREFIX = "model-storage"


# ---------------------------------------------------------------------------
# 底层条件写（与 generation/ready 无关，直接调用 `_put_stream_if_absent`）
# ---------------------------------------------------------------------------


def test_small_conditional_put_streams_body_compatible_with_real_sdk_headers():
    minio = SdkLikeConditionalMinio()
    client = ModelPreheatS3Client(minio)

    written = client._put_stream_if_absent(
        "bucket",
        "object",
        io.BytesIO(b"payload"),
        7,
        "application/octet-stream",
        {"sha256": "0" * 64},
    )

    assert written is True
    assert minio.calls[0][2] == b"payload"
    assert minio.calls[0][3]["If-None-Match"] == "*"


def test_rate_limited_conditional_put_yields_small_chunks_to_transport(monkeypatch):
    from gpustack.worker.model_preheat import s3_client as s3_client_module

    sleeps = []
    monkeypatch.setattr(s3_client_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(s3_client_module.time, "sleep", sleeps.append)
    minio = SdkLikeConditionalMinio()
    payload = b"x" * (2 * 1024 * 1024)

    written = ModelPreheatS3Client(minio)._put_stream_if_absent(
        "bucket",
        "object",
        io.BytesIO(payload),
        len(payload),
        "application/octet-stream",
        {"sha256": "0" * 64},
        bandwidth_limit_mbps=8,
    )

    assert written is True
    assert b"".join(minio.calls[0][4]) == payload
    assert max(map(len, minio.calls[0][4])) <= 64 * 1024
    assert len(sleeps) == len(minio.calls[0][4])


def test_large_conditional_put_uses_multipart_and_conditions_complete(monkeypatch):
    monkeypatch.setattr(
        "gpustack.worker.model_preheat.s3_client.CONDITIONAL_SINGLE_PUT_MAX_SIZE",
        1024,
    )
    monkeypatch.setattr(
        "gpustack.worker.model_preheat.s3_client.CONDITIONAL_MULTIPART_PART_SIZE",
        5 * 1024 * 1024,
    )
    payload = b"x" * (5 * 1024 * 1024 + 1)
    minio = MultipartConditionalMinio()
    client = ModelPreheatS3Client(minio)

    written = client._put_stream_if_absent(
        "bucket",
        "large-object",
        io.BytesIO(payload),
        len(payload),
        "application/octet-stream",
        {"sha256": "0" * 64},
    )

    assert written is True
    assert minio.parts == [(1, 5 * 1024 * 1024), (2, 1)]
    complete = minio.completed[0]
    assert complete[0] == "POST"
    assert complete[4]["If-None-Match"] == "*"
    assert complete[5] == {"uploadId": "upload-id"}
    assert minio.aborted == []


def test_rate_limited_multipart_streams_small_chunks_through_http_transport(
    monkeypatch,
):
    from gpustack.worker.model_preheat import s3_client as s3_client_module

    monkeypatch.setattr(
        "gpustack.worker.model_preheat.s3_client.CONDITIONAL_SINGLE_PUT_MAX_SIZE",
        1024,
    )
    clock = [0.0]
    monkeypatch.setattr(s3_client_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        s3_client_module.time,
        "sleep",
        lambda delay: clock.__setitem__(0, clock[0] + delay),
    )
    payload = b"x" * (5 * 1024 * 1024 + 1)
    minio = StreamingMultipartConditionalMinio(clock)

    written = ModelPreheatS3Client(minio)._put_stream_if_absent(
        "bucket",
        "large-object",
        io.BytesIO(payload),
        len(payload),
        "application/octet-stream",
        {"sha256": "0" * 64},
        bandwidth_limit_mbps=8,
    )

    assert written is True
    assert sum(chunk[1] for chunk in minio.sent_chunks) == len(payload)
    assert max(chunk[1] for chunk in minio.sent_chunks) <= 64 * 1024
    assert minio.sent_chunks[0][0] == pytest.approx(0.065536)
    assert minio.sent_chunks[-1][0] == pytest.approx(5.242881)
    assert minio.parts == []
    assert minio.completed[0][4]["If-None-Match"] == "*"
    assert minio.aborted == []


def test_rate_limited_multipart_uses_real_minio_presigned_url_shape(monkeypatch):
    from gpustack.worker.model_preheat import s3_client as s3_client_module

    monkeypatch.setattr(
        "gpustack.worker.model_preheat.s3_client.CONDITIONAL_SINGLE_PUT_MAX_SIZE",
        1024,
    )
    clock = [0.0]
    monkeypatch.setattr(s3_client_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        s3_client_module.time,
        "sleep",
        lambda delay: clock.__setitem__(0, clock[0] + delay),
    )
    transport = StreamingMultipartConditionalMinio(clock)
    sdk = Minio(
        "localhost:9000",
        access_key="access-key",
        secret_key="secret-key",
        secure=False,
        region="us-east-1",
    )
    monkeypatch.setattr(sdk, "_http", transport)
    monkeypatch.setattr(sdk, "_create_multipart_upload", lambda *args: "upload-id")
    monkeypatch.setattr(
        sdk, "_abort_multipart_upload", transport._abort_multipart_upload
    )
    monkeypatch.setattr(sdk, "_execute", transport._execute)
    payload = b"x" * (5 * 1024 * 1024 + 1)

    written = ModelPreheatS3Client(sdk)._put_stream_if_absent(
        "bucket",
        "large-object",
        io.BytesIO(payload),
        len(payload),
        "application/octet-stream",
        {"sha256": "0" * 64},
        bandwidth_limit_mbps=8,
    )

    assert written is True
    assert sum(chunk[1] for chunk in transport.sent_chunks) == len(payload)
    assert all("uploadId=upload-id" in chunk[2] for chunk in transport.sent_chunks)
    assert all("X-Amz-Signature=" in chunk[2] for chunk in transport.sent_chunks)
    assert transport.completed[0][4]["If-None-Match"] == "*"


def test_rate_limited_multipart_cancel_during_http_stream_aborts(monkeypatch):
    monkeypatch.setattr(
        "gpustack.worker.model_preheat.s3_client.CONDITIONAL_SINGLE_PUT_MAX_SIZE",
        1024,
    )
    canceled = threading.Event()
    clock = [0.0]

    class CancelingTransport(StreamingMultipartConditionalMinio):
        def urlopen(self, method, url, body, headers, **kwargs):
            del kwargs
            chunk = body.read(2 * 1024 * 1024)
            self.sent_chunks.append((self.clock[0], len(chunk), url))
            canceled.set()
            body.read(2 * 1024 * 1024)
            raise AssertionError("取消后不应继续读取")

    minio = CancelingTransport(clock)

    with pytest.raises(ModelPreheatCanceled, match="canceled"):
        ModelPreheatS3Client(minio)._put_stream_if_absent(
            "bucket",
            "large-object",
            io.BytesIO(b"x" * (5 * 1024 * 1024 + 1)),
            5 * 1024 * 1024 + 1,
            "application/octet-stream",
            {"sha256": "0" * 64},
            cancel_check=canceled.is_set,
            bandwidth_limit_mbps=8,
        )

    assert len(minio.sent_chunks) == 1
    assert minio.sent_chunks[0][1] <= 64 * 1024
    assert minio.aborted == [("bucket", "large-object", "upload-id")]


def test_multipart_cancel_aborts_upload(monkeypatch):
    monkeypatch.setattr(
        "gpustack.worker.model_preheat.s3_client.CONDITIONAL_SINGLE_PUT_MAX_SIZE", 1
    )
    canceled = threading.Event()
    minio = MultipartConditionalMinio(cancel_event=canceled)
    client = ModelPreheatS3Client(minio)

    with pytest.raises(ModelPreheatCanceled, match="canceled"):
        client._put_stream_if_absent(
            "bucket",
            "large-object",
            io.BytesIO(b"payload"),
            7,
            "application/octet-stream",
            {"sha256": "0" * 64},
            cancel_check=canceled.is_set,
        )

    assert minio.completed == []
    assert minio.aborted == [("bucket", "large-object", "upload-id")]


def test_multipart_complete_precondition_failure_aborts_and_reports_existing(
    monkeypatch,
):
    monkeypatch.setattr(
        "gpustack.worker.model_preheat.s3_client.CONDITIONAL_SINGLE_PUT_MAX_SIZE", 1
    )
    minio = MultipartConditionalMinio(fail_complete=True)
    client = ModelPreheatS3Client(minio)

    written = client._put_stream_if_absent(
        "bucket",
        "large-object",
        io.BytesIO(b"payload"),
        7,
        "application/octet-stream",
        {"sha256": "0" * 64},
    )

    assert written is False
    assert minio.completed[0][4]["If-None-Match"] == "*"
    assert minio.aborted == [("bucket", "large-object", "upload-id")]


def test_over_5_gib_never_uses_single_put(monkeypatch):
    class UnreadableSparseStream:
        def read(self, size=-1):
            raise AssertionError("不得走单 PUT 或实际读取 5 GiB")

    client = ModelPreheatS3Client(SdkLikeConditionalMinio())
    calls = []

    def multipart(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(client, "_conditional_multipart_put", multipart)
    size = 5 * 1024**3 + 1

    assert client._put_stream_if_absent(
        "bucket",
        "huge-object",
        UnreadableSparseStream(),
        size,
        "application/octet-stream",
        {},
    )
    assert calls[0][0][3] == size


def test_minio_tls_verify_false_uses_unverified_transport(monkeypatch):
    captured = {}

    class FakeMinio:
        def __init__(self, endpoint, **kwargs):
            captured["endpoint"] = endpoint
            captured.update(kwargs)

        def enable_virtual_style_endpoint(self):
            captured["virtual"] = True

        def disable_virtual_style_endpoint(self):
            captured["virtual"] = False

    monkeypatch.setattr("gpustack.worker.model_preheat.s3_client.Minio", FakeMinio)

    ModelPreheatS3Client.from_minio(
        "s3.example.invalid",
        "access-key",
        "secret-key",
        secure=True,
        tls_verify=False,
    )

    assert captured["secure"] is True
    assert captured["http_client"].connection_pool_kw["cert_reqs"] == ssl.CERT_NONE
    assert captured["virtual"] is True


def test_minio_can_disable_virtual_hosted_style(monkeypatch):
    captured = {}

    class FakeMinio:
        def __init__(self, endpoint, **kwargs):
            captured["endpoint"] = endpoint

        def enable_virtual_style_endpoint(self):
            captured["virtual"] = True

        def disable_virtual_style_endpoint(self):
            captured["virtual"] = False

    monkeypatch.setattr("gpustack.worker.model_preheat.s3_client.Minio", FakeMinio)

    ModelPreheatS3Client.from_minio(
        "http://s3.example.invalid/",
        "access-key",
        "secret-key",
        secure=True,
        use_virtual_hosted_style=False,
    )

    assert captured == {"endpoint": "s3.example.invalid", "virtual": False}


def test_local_manifest_path_rejects_escape_from_root(tmp_path):
    (tmp_path / "safe.bin").write_bytes(b"safe")
    unsafe_file = object.__new__(ManifestFile)
    object.__setattr__(unsafe_file, "path", "%2E%2E/secret.bin")
    object.__setattr__(unsafe_file, "size", 6)
    object.__setattr__(unsafe_file, "sha256", "0" * 64)

    with pytest.raises(ValueError, match="manifest_path_escape"):
        ModelPreheatS3Client(InMemoryMinio())._local_manifest_path(
            tmp_path, unsafe_file
        )


# ---------------------------------------------------------------------------
# 统一 Artifact 发布协议（manifest.json + files/，无 ready.json）
# ---------------------------------------------------------------------------


def test_artifact_object_paths_follow_unified_layout(tmp_path):
    manifest = _artifact_manifest(tmp_path)
    client = ModelPreheatS3Client(InMemoryMinio())

    manifest_object = client.artifact_manifest_object(PREFIX, manifest)
    file_object = client.artifact_file_object(PREFIX, manifest, manifest.files[0])

    assert manifest_object == (
        f"model-storage/modelscope/org/model/{manifest.artifact_id}/manifest.json"
    )
    assert file_object == (
        f"model-storage/modelscope/org/model/{manifest.artifact_id}/files/model.bin"
    )
    # 不含 ready.json、generation 或协议版本目录。
    assert "ready.json" not in manifest_object
    assert "ready.json" not in file_object
    assert "generation" not in manifest_object
    assert "/v1/" not in file_object


@pytest.mark.parametrize("prefix", ["", "storage", "team/a/b"])
def test_artifact_object_paths_transmit_profile_prefix(tmp_path, prefix):
    manifest = _artifact_manifest(tmp_path)
    client = ModelPreheatS3Client(InMemoryMinio())
    file_object = client.artifact_file_object(prefix, manifest, manifest.files[0])
    if prefix:
        expected_prefix = (
            f"{prefix}/modelscope/org/model/{manifest.artifact_id}/files/model.bin"
        )
    else:
        expected_prefix = f"modelscope/org/model/{manifest.artifact_id}/files/model.bin"
    assert file_object == expected_prefix
    manifest_object = client.artifact_manifest_object(prefix, manifest)
    assert manifest_object.endswith(
        f"modelscope/org/model/{manifest.artifact_id}/manifest.json"
    )


@pytest.mark.parametrize("bad_prefix", ["a/../b", "..", "a//b", "pre\\fix"])
def test_artifact_object_paths_reject_unsafe_prefix(tmp_path, bad_prefix):
    manifest = _artifact_manifest(tmp_path)
    client = ModelPreheatS3Client(InMemoryMinio())

    with pytest.raises(ModelPreheatIdentityError):
        client.artifact_file_object(bad_prefix, manifest, manifest.files[0])
    with pytest.raises(ModelPreheatIdentityError):
        client.artifact_manifest_object(bad_prefix, manifest)


def test_artifact_publish_writes_files_then_manifest_and_no_ready(tmp_path):
    manifest = _artifact_manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)

    result = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    manifest_object = client.artifact_manifest_object(PREFIX, manifest)
    file_object = client.artifact_file_object(PREFIX, manifest, manifest.files[0])
    assert result.uploaded == 2
    assert result.skipped == 0
    assert result.ready_digest == manifest.artifact_id
    assert minio.objects[("bucket", file_object)].data == b"weights"
    stored = json.loads(minio.objects[("bucket", manifest_object)].data)
    assert stored["artifact_id"] == manifest.artifact_id
    assert set(stored) == {
        "schema_version",
        "artifact_id",
        "source",
        "model_id",
        "resolved_revision",
        "include_patterns",
        "exclude_patterns",
        "file_count",
        "total_size",
        "files",
    }
    assert all(not name.endswith("ready.json") for name in minio.puts)


def test_artifact_publish_is_idempotent_when_manifest_matches(tmp_path):
    manifest = _artifact_manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)

    first = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)
    second = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    assert first.uploaded == 2
    assert first.ready_written is True
    assert second.uploaded == 0
    assert second.ready_written is False
    file_object = client.artifact_file_object(PREFIX, manifest, manifest.files[0])
    # 幂等重放不重复上传文件。
    assert minio.puts.count(file_object) == 1


def test_artifact_publish_fails_when_manifest_content_conflicts(tmp_path):
    manifest = _artifact_manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    manifest_object = client.artifact_manifest_object(PREFIX, manifest)

    # 同一 artifact_id 前缀下存在语义不同的 Manifest（内容冲突）。
    conflicting = json.loads(manifest.to_artifact_json_bytes().decode("utf-8"))
    conflicting["files"][0]["sha256"] = "1" * 64
    minio.objects[("bucket", manifest_object)] = StoredObject(
        json.dumps(conflicting, sort_keys=True).encode("utf-8")
    )

    with pytest.raises(
        ModelPreheatS3ManifestConflict, match="artifact_manifest_conflict"
    ):
        client.publish_artifact("bucket", PREFIX, manifest, tmp_path)


def test_artifact_publish_repairs_missing_files_and_keeps_existing(tmp_path):
    manifest = _artifact_manifest(tmp_path, content=b"weights-v2")
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)

    # 已有文件缺失（模拟部分文件补传），先发布一次再删除文件对象。
    client.publish_artifact("bucket", PREFIX, manifest, tmp_path)
    file_object = client.artifact_file_object(PREFIX, manifest, manifest.files[0])
    del minio.objects[("bucket", file_object)]

    result = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    assert result.uploaded == 1
    assert result.skipped == 1  # Manifest 一致，跳过。
    assert minio.objects[("bucket", file_object)].data == b"weights-v2"


def test_artifact_publish_skips_only_content_verified_files(tmp_path):
    manifest = _artifact_manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    file_object = client.artifact_file_object(PREFIX, manifest, manifest.files[0])
    # 正确内容 + 正确大小 + 元数据；跳过只允许在流式重算后确认一致。
    minio.objects[("bucket", file_object)] = StoredObject(
        b"weights",
        metadata={"sha256": manifest.files[0].sha256},
        etag='"not-a-content-sha-2"',
    )

    result = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    # 摘要一致的文件跳过，只发布 Manifest。
    assert result.skipped == 1
    assert result.uploaded == 1
    assert file_object not in minio.puts


def test_artifact_publish_detects_same_size_tampered_remote_content(tmp_path):
    """同尺寸篡改：大小与元数据 sha256 都对，但远端内容不同 → 必须失败。"""
    manifest = _artifact_manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    file_object = client.artifact_file_object(PREFIX, manifest, manifest.files[0])
    # 内容被篡改为同长度 b"tampered"，元数据仍声明原始 sha256。
    minio.objects[("bucket", file_object)] = StoredObject(
        b"tampered",
        metadata={"sha256": manifest.files[0].sha256},
    )

    with pytest.raises(ModelPreheatS3Conflict, match="object_content_conflict"):
        client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    # 绝不覆盖被篡改的远端对象。
    assert minio.objects[("bucket", file_object)].data == b"tampered"


def test_artifact_publish_detects_local_file_content_mismatch(tmp_path):
    """本地文件内容与 Manifest 声明 sha256 不一致 → 发布前失败。"""
    manifest = _artifact_manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    # 篡改本地文件，使其 sha256 与 Manifest 不一致。
    (tmp_path / "model.bin").write_bytes(b"different-local-content")

    with pytest.raises(ModelPreheatS3Conflict, match="local_file_content_mismatch"):
        client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    # 任何对象都不得写入。
    assert minio.objects == {}


def test_artifact_publish_detects_file_content_conflict(tmp_path):
    manifest = _artifact_manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    file_object = client.artifact_file_object(PREFIX, manifest, manifest.files[0])
    original = minio.put_object_if_absent

    def inject_conflict(bucket_name, object_name, *args, **kwargs):
        if object_name == file_object:
            minio.objects[(bucket_name, object_name)] = StoredObject(
                b"different" * 8,
                metadata={"sha256": "0" * 64},
            )
        return original(bucket_name, object_name, *args, **kwargs)

    minio.put_object_if_absent = inject_conflict

    with pytest.raises(ModelPreheatS3Conflict, match="object_content_conflict"):
        client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    assert minio.objects[("bucket", file_object)].data == b"different" * 8


def test_artifact_conditional_manifest_write_returns_false_on_412(tmp_path):
    """412 PreconditionFailed：条件写竞争失败时返回 False 而不是异常。"""
    manifest = _artifact_manifest(tmp_path)
    minio = ExecuteConditionalMinio()
    client = ModelPreheatS3Client(minio)
    manifest_object = client.artifact_manifest_object(PREFIX, manifest)
    minio.objects[("bucket", manifest_object)] = StoredObject(b"{}")

    written = client._put_bytes_if_absent(
        "bucket",
        manifest_object,
        manifest.to_artifact_json_bytes(),
        content_type="application/json",
        metadata={"sha256": "0" * 64},
    )

    assert written is False
    manifest_call = minio.execute_calls[-1]
    assert manifest_call["headers"]["If-None-Match"] == "*"


def test_artifact_publish_manifest_without_conditional_capability_fails_closed(
    tmp_path,
):
    """SDK 不支持条件写时拒绝发布 Manifest，避免覆盖并发 Artifact。"""
    manifest = _artifact_manifest(tmp_path)
    minio = NoConditionalMinio()
    client = ModelPreheatS3Client(minio)
    manifest_object = client.artifact_manifest_object(PREFIX, manifest)

    with pytest.raises(ModelPreheatS3Conflict, match="conditional_create_unsupported"):
        client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    assert ("bucket", manifest_object) not in minio.objects


def test_artifact_publish_reads_invalid_manifest_and_fails_closed(tmp_path):
    """路径上存在非法 Manifest 时绝不覆盖，按冲突失败闭合。"""
    manifest = _artifact_manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    manifest_object = client.artifact_manifest_object(PREFIX, manifest)
    minio.objects[("bucket", manifest_object)] = StoredObject(b"not json")

    with pytest.raises(
        ModelPreheatS3ManifestConflict, match="artifact_manifest_conflict"
    ):
        client.publish_artifact("bucket", PREFIX, manifest, tmp_path)


def test_concurrent_artifact_publishers_converge_to_one_manifest(tmp_path):
    """Manifest 缺失→条件写返回 False（412）→重读到同一内容→收敛。

    走真实小对象条件写（presigned + If-None-Match，已存在返回 412→False）：
    在 Manifest 条件写 PUT 到达前的那一刻，模拟并发发布者已写入相同内容，
    本次 PUT 触发真实 412，随后必须重读一致并收敛，而不是覆盖或误判冲突。
    """
    manifest = _artifact_manifest(tmp_path)
    minio = ExecuteConditionalMinio()
    client = ModelPreheatS3Client(minio)
    manifest_object = client.artifact_manifest_object(PREFIX, manifest)
    seeded = {"done": False}
    original_urlopen = minio.urlopen

    def racing_urlopen(method, url, body=None, headers=None, **kwargs):
        bucket_name, object_name = url.removeprefix("s3://").split("/", 1)
        if (
            not seeded["done"]
            and object_name == manifest_object
            and (headers or {}).get("If-None-Match") == "*"
            and (bucket_name, object_name) not in minio.objects
        ):
            # 并发发布者已先一步写入相同的 Manifest：触发真实 412。
            seeded["done"] = True
            minio.objects[(bucket_name, object_name)] = StoredObject(
                manifest.to_artifact_json_bytes(),
                metadata={"sha256": "0" * 64},
            )
        return original_urlopen(method, url, body=body, headers=headers, **kwargs)

    minio.urlopen = racing_urlopen

    result = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    stored = json.loads(minio.objects[("bucket", manifest_object)].data)
    assert stored["artifact_id"] == manifest.artifact_id
    # 条件写 412 后重读一致 → 收敛为幂等成功。
    assert result.ready_written is False
    # 最终 Manifest 内容与发布方语义一致。
    assert (
        client.read_artifact_manifest("bucket", PREFIX, manifest).to_artifact_dict()
        == manifest.to_artifact_dict()
    )


def test_concurrent_artifact_publishers_conflict_when_manifest_differs(tmp_path):
    """条件写 412 但重读到不同 artifact_id → 冲突失败闭合。"""
    manifest = _artifact_manifest(tmp_path)
    minio = ExecuteConditionalMinio()
    client = ModelPreheatS3Client(minio)
    manifest_object = client.artifact_manifest_object(PREFIX, manifest)
    seeded = {"done": False}

    conflicting = json.loads(manifest.to_artifact_json_bytes().decode("utf-8"))
    conflicting["files"][0]["sha256"] = "1" * 64
    conflicting_bytes = json.dumps(conflicting, sort_keys=True).encode("utf-8")
    original_urlopen = minio.urlopen

    def racing_urlopen(method, url, body=None, headers=None, **kwargs):
        bucket_name, object_name = url.removeprefix("s3://").split("/", 1)
        if (
            not seeded["done"]
            and object_name == manifest_object
            and (headers or {}).get("If-None-Match") == "*"
            and (bucket_name, object_name) not in minio.objects
        ):
            # 并发发布者先写入不同内容的 Manifest：触发真实 412。
            seeded["done"] = True
            minio.objects[(bucket_name, object_name)] = StoredObject(
                conflicting_bytes, metadata={"sha256": "0" * 64}
            )
        return original_urlopen(method, url, body=body, headers=headers, **kwargs)

    minio.urlopen = racing_urlopen

    with pytest.raises(
        ModelPreheatS3ManifestConflict, match="artifact_manifest_conflict"
    ):
        client.publish_artifact("bucket", PREFIX, manifest, tmp_path)
    # 并发写入的冲突 Manifest 未被覆盖。
    assert minio.objects[("bucket", manifest_object)].data == conflicting_bytes
