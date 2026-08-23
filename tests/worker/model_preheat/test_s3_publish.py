import json
import io
import hashlib
import ssl
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from minio import Minio
from minio.datatypes import Part
import urllib3

from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    ModelPreheatIdentityError,
)
from gpustack.worker.model_preheat.manifest import (
    ManifestFile,
    build_model_preheat_manifest,
)
from gpustack.worker.model_preheat.s3_client import (
    ModelPreheatCanceled,
    ModelPreheatS3Client,
    ModelPreheatS3Conflict,
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


class ResumableMultipartMinio(BasicMinio):
    def __init__(self):
        super().__init__()
        self.uploads = {}
        self.part_uploads = []
        self.failed_once = False

    def _list_multipart_uploads(self, bucket_name, prefix=None, **kwargs):
        del kwargs
        uploads = [
            SimpleNamespace(
                object_name=upload["object_name"],
                upload_id=upload_id,
                initiated_time=None,
            )
            for upload_id, upload in self.uploads.items()
            if upload["bucket_name"] == bucket_name
            and upload["object_name"].startswith(prefix or "")
        ]
        return SimpleNamespace(uploads=uploads, is_truncated=False)

    def _create_multipart_upload(self, bucket_name, object_name, headers):
        upload_id = f"upload-{len(self.uploads) + 1}"
        self.uploads[upload_id] = {
            "bucket_name": bucket_name,
            "object_name": object_name,
            "headers": headers,
            "parts": {},
        }
        return upload_id

    def _list_parts(self, bucket_name, object_name, upload_id, **kwargs):
        del bucket_name, object_name, kwargs
        parts = [
            Part(number, hashlib.md5(data).hexdigest(), size=len(data))
            for number, data in sorted(self.uploads[upload_id]["parts"].items())
        ]
        return SimpleNamespace(
            parts=parts,
            is_truncated=False,
            next_part_number_marker=None,
        )

    def _upload_part(
        self, bucket_name, object_name, data, headers, upload_id, part_number
    ):
        del bucket_name, object_name, headers
        if part_number == 2 and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("simulated_part_failure")
        self.part_uploads.append(part_number)
        self.uploads[upload_id]["parts"][part_number] = data
        return hashlib.md5(data).hexdigest()

    def _complete_multipart_upload(
        self, bucket_name, object_name, upload_id, parts, ssec=None
    ):
        del parts, ssec
        upload = self.uploads.pop(upload_id)
        payload = b"".join(
            upload["parts"][number] for number in sorted(upload["parts"])
        )
        metadata = {
            key[len("x-amz-meta-") :]: value
            for key, value in upload["headers"].items()
            if key.startswith("x-amz-meta-")
        }
        self.puts.append(object_name)
        self.objects[(bucket_name, object_name)] = StoredObject(
            payload,
            metadata=metadata,
        )


class UnsupportedResumeMinio(ResumableMultipartMinio):
    def _list_parts(self, bucket_name, object_name, upload_id, **kwargs):
        del bucket_name, object_name, upload_id, kwargs
        raise _S3CodeError("NotImplemented")


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
    assert (
        captured["http_client"].pool_classes_by_scheme["https"].__name__
        == "_QuietUnverifiedHTTPSConnectionPool"
    )
    assert (
        urllib3.PoolManager().pool_classes_by_scheme["https"]
        is urllib3.HTTPSConnectionPool
    )
    assert captured["virtual"] is True


@pytest.mark.parametrize(
    ("endpoint", "secure"),
    [
        ("http://s3.example.invalid", True),
        ("https://s3.example.invalid", False),
    ],
)
def test_minio_explicit_secure_overrides_endpoint_scheme(monkeypatch, endpoint, secure):
    captured = {}

    class FakeMinio:
        def __init__(self, endpoint, **kwargs):
            captured["endpoint"] = endpoint
            captured["secure"] = kwargs["secure"]

        def enable_virtual_style_endpoint(self):
            pass

        def disable_virtual_style_endpoint(self):
            pass

    monkeypatch.setattr("gpustack.worker.model_preheat.s3_client.Minio", FakeMinio)

    ModelPreheatS3Client.from_minio(
        endpoint,
        "access-key",
        "secret-key",
        secure=secure,
    )

    assert captured == {
        "endpoint": "s3.example.invalid",
        "secure": secure,
    }


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


def test_artifact_publish_overwrites_manifest_when_digest_differs(tmp_path):
    manifest = _artifact_manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    manifest_object = client.artifact_manifest_object(PREFIX, manifest)

    conflicting = json.loads(manifest.to_artifact_json_bytes().decode("utf-8"))
    conflicting["files"][0]["sha256"] = "1" * 64
    minio.objects[("bucket", manifest_object)] = StoredObject(
        json.dumps(conflicting, sort_keys=True).encode("utf-8")
    )

    result = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    assert result.ready_written is True
    assert minio.objects[("bucket", manifest_object)].data == (
        manifest.to_artifact_json_bytes()
    )


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
    # 大小和受管摘要元数据一致时，仅通过 HEAD 即可跳过。
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


def test_artifact_publish_uploads_only_changed_files_in_multi_file_model(tmp_path):
    (tmp_path / "model-a.bin").write_bytes(b"weights-a")
    (tmp_path / "model-b.bin").write_bytes(b"weights-b")
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a91b",
        file_patterns=["*.bin"],
        requested_revision="master",
    )
    manifest = build_model_preheat_manifest(tmp_path, identity)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    client.publish_artifact("bucket", PREFIX, manifest, tmp_path)
    first_object = client.artifact_file_object(PREFIX, manifest, manifest.files[0])
    second_object = client.artifact_file_object(PREFIX, manifest, manifest.files[1])
    minio.objects[("bucket", second_object)].metadata["sha256"] = "0" * 64
    minio.puts.clear()

    result = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    assert result.uploaded == 1
    assert result.skipped == 2
    assert first_object not in minio.puts
    assert minio.puts == [second_object]


def test_artifact_publish_resumes_incomplete_large_file_parts(tmp_path, monkeypatch):
    from gpustack.worker.model_preheat import s3_client as s3_client_module

    monkeypatch.setattr(s3_client_module, "CONDITIONAL_SINGLE_PUT_MAX_SIZE", 4)
    monkeypatch.setattr(s3_client_module, "CONDITIONAL_MULTIPART_PART_SIZE", 4)
    content = b"abcdefghijkl"
    manifest = _artifact_manifest(tmp_path, content=content)
    minio = ResumableMultipartMinio()
    client = ModelPreheatS3Client(minio)

    with pytest.raises(RuntimeError, match="simulated_part_failure"):
        client.publish_artifact("bucket", PREFIX, manifest, tmp_path)
    assert minio.part_uploads == [1]

    result = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    file_object = client.artifact_file_object(PREFIX, manifest, manifest.files[0])
    assert result.ready_written is True
    assert minio.part_uploads == [1, 2, 3]
    assert minio.objects[("bucket", file_object)].data == content


def test_artifact_publish_falls_back_when_list_parts_is_unsupported(
    tmp_path, monkeypatch
):
    from gpustack.worker.model_preheat import s3_client as s3_client_module

    monkeypatch.setattr(s3_client_module, "CONDITIONAL_SINGLE_PUT_MAX_SIZE", 4)
    content = b"abcdefghijkl"
    manifest = _artifact_manifest(tmp_path, content=content)
    minio = UnsupportedResumeMinio()
    client = ModelPreheatS3Client(minio)

    result = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    file_object = client.artifact_file_object(PREFIX, manifest, manifest.files[0])
    assert result.ready_written is True
    assert minio.objects[("bucket", file_object)].data == content


def test_resumed_parts_only_limit_actual_upload_traffic(monkeypatch):
    from gpustack.worker.model_preheat import s3_client as s3_client_module

    monkeypatch.setattr(s3_client_module, "CONDITIONAL_MULTIPART_PART_SIZE", 4)
    consumed = []
    limiter = SimpleNamespace(consume=consumed.append)
    monkeypatch.setattr(s3_client_module, "_BandwidthLimiter", lambda _: limiter)
    content = b"abcdefghijkl"
    minio = ResumableMultipartMinio()
    minio.failed_once = True
    upload_id = minio._create_multipart_upload("bucket", "object", {})
    minio.uploads[upload_id]["parts"][1] = content[:4]

    handled = ModelPreheatS3Client(minio)._resumable_multipart_put(
        "bucket",
        "object",
        io.BytesIO(content),
        len(content),
        "application/octet-stream",
        {"sha256": hashlib.sha256(content).hexdigest()},
        bandwidth_limit_mbps=1,
    )

    assert handled is True
    assert consumed == [4, 4]


def test_artifact_publish_skips_same_size_object_with_matching_digest_metadata(
    tmp_path,
):
    manifest = _artifact_manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    file_object = client.artifact_file_object(PREFIX, manifest, manifest.files[0])
    minio.objects[("bucket", file_object)] = StoredObject(
        b"corrupt",
        metadata={"sha256": manifest.files[0].sha256},
    )

    result = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    assert result.skipped == 1
    assert minio.objects[("bucket", file_object)].data == b"corrupt"
    assert file_object not in minio.puts


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


def test_artifact_publish_overwrites_file_when_digest_metadata_differs(tmp_path):
    manifest = _artifact_manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    file_object = client.artifact_file_object(PREFIX, manifest, manifest.files[0])
    minio.objects[("bucket", file_object)] = StoredObject(
        b"different" * 8,
        metadata={"sha256": "0" * 64},
    )

    result = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    assert result.uploaded == 2
    assert minio.objects[("bucket", file_object)].data == b"weights"


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


def test_artifact_publish_uses_standard_put_without_conditional_requests(tmp_path):
    manifest = _artifact_manifest(tmp_path)
    minio = ExecuteConditionalMinio()
    client = ModelPreheatS3Client(minio)

    result = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    assert result.ready_written is True
    assert minio.execute_calls == []
    file_object = client.artifact_file_object(PREFIX, manifest, manifest.files[0])
    manifest_object = client.artifact_manifest_object(PREFIX, manifest)
    assert minio.objects[("bucket", file_object)].data == b"weights"
    assert minio.objects[("bucket", manifest_object)].data == (
        manifest.to_artifact_json_bytes()
    )


def test_artifact_publish_supports_client_without_conditional_capability(tmp_path):
    manifest = _artifact_manifest(tmp_path)
    minio = NoConditionalMinio()
    client = ModelPreheatS3Client(minio)
    manifest_object = client.artifact_manifest_object(PREFIX, manifest)

    result = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    assert result.ready_written is True
    assert ("bucket", manifest_object) in minio.objects


def test_artifact_publish_overwrites_invalid_manifest(tmp_path):
    manifest = _artifact_manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    manifest_object = client.artifact_manifest_object(PREFIX, manifest)
    minio.objects[("bucket", manifest_object)] = StoredObject(b"not json")

    result = client.publish_artifact("bucket", PREFIX, manifest, tmp_path)

    assert result.ready_written is True
    assert minio.objects[("bucket", manifest_object)].data == (
        manifest.to_artifact_json_bytes()
    )
