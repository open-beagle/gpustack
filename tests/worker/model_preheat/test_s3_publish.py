import json
import io
import hashlib
import ssl
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from minio import Minio

from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
from gpustack.worker.model_preheat.manifest import (
    ManifestFile,
    ModelPreheatManifest,
    build_model_preheat_manifest,
)
from gpustack.worker.model_preheat.s3_client import (
    ModelPreheatCanceled,
    ModelPreheatS3Client,
    ModelPreheatS3Conflict,
    ModelPreheatS3ManifestError,
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


class ObjectResponse(io.BytesIO):
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
    def __init__(self):
        super().__init__()
        self.execute_calls: list[dict] = []
        self.concurrent_ready_digest: str | None = None
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
        if self.concurrent_ready_digest is not None and object_name.endswith(
            "ready.json"
        ):
            self.objects[(bucket_name, object_name)] = StoredObject(
                json.dumps({"digest": self.concurrent_ready_digest}).encode("utf-8")
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


def _manifest(tmp_path, content: bytes = b"weights"):
    (tmp_path / "model.bin").write_bytes(content)
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="main",
        file_patterns=["model.bin"],
    )
    return build_model_preheat_manifest(
        tmp_path,
        identity,
        cache_key="cache-key",
        selection_digest="selection-digest",
        generation_id="generation-id",
    )


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


def test_publish_generation_honors_optional_upload_bandwidth_limit(
    tmp_path, monkeypatch
):
    from gpustack.worker.model_preheat import s3_client as s3_client_module

    sleeps = []
    monkeypatch.setattr(s3_client_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(s3_client_module.time, "sleep", sleeps.append)
    manifest = _manifest(tmp_path, b"x" * (2 * 1024 * 1024))

    ModelPreheatS3Client(InMemoryMinio()).publish_generation(
        "models",
        "preheat",
        manifest,
        tmp_path,
        bandwidth_limit_mbps=8,
    )

    assert len(sleeps) == 32
    assert sleeps[-1] == pytest.approx(2.097152)


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
    assert client.ready_object("preheat", manifest).startswith(
        "preheat/model-cache/v1/"
    )


def test_generation_file_conditional_create_never_overwrites_concurrent_object(
    tmp_path,
):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    file_object = client.generation_file_object("preheat", manifest, manifest.files[0])
    original = minio.put_object_if_absent

    def inject_conflict(bucket_name, object_name, *args, **kwargs):
        if object_name == file_object:
            minio.objects[(bucket_name, object_name)] = StoredObject(
                b"different",
                metadata={"sha256": "0" * 64},
            )
        return original(bucket_name, object_name, *args, **kwargs)

    minio.put_object_if_absent = inject_conflict

    with pytest.raises(ModelPreheatS3Conflict, match="object_content_conflict"):
        client.publish_generation("bucket", "preheat", manifest, tmp_path)

    assert minio.objects[("bucket", file_object)].data == b"different"


def test_publish_rechecks_every_generation_file_before_manifest(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    file_object = client.generation_file_object("preheat", manifest, manifest.files[0])
    original = minio.put_object_if_absent

    def corrupt_after_upload(bucket_name, object_name, *args, **kwargs):
        written = original(bucket_name, object_name, *args, **kwargs)
        if object_name == file_object and written:
            minio.objects[(bucket_name, object_name)] = StoredObject(
                b"corrupt", metadata={"sha256": "0" * 64}
            )
        return written

    minio.put_object_if_absent = corrupt_after_upload

    with pytest.raises(ModelPreheatS3Conflict, match="object_content_conflict"):
        client.publish_generation("bucket", "preheat", manifest, tmp_path)

    assert not any(
        name.endswith(".gpustack-manifest.json") for _, name in minio.objects
    )


def test_cancel_during_manifest_put_removes_only_attempt_manifest(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    canceled = threading.Event()
    original = minio.put_object_if_absent

    def cancel_after_put(bucket_name, object_name, *args, **kwargs):
        written = original(bucket_name, object_name, *args, **kwargs)
        if object_name.endswith(".gpustack-manifest.json"):
            canceled.set()
        return written

    minio.put_object_if_absent = cancel_after_put

    with pytest.raises(ModelPreheatCanceled, match="canceled"):
        client.publish_generation(
            "bucket", "preheat", manifest, tmp_path, cancel_check=canceled.is_set
        )

    assert not any(
        name.endswith(".gpustack-manifest.json") for _, name in minio.objects
    )
    assert not any(name.endswith("ready.json") for _, name in minio.objects)
    assert any("/files/" in name for _, name in minio.objects)


def test_cancel_cleanup_retries_remove_failures_then_confirms_absence(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    canceled = threading.Event()
    original_put = minio.put_object_if_absent
    original_remove = minio.remove_object
    remove_calls = []
    sleeps = []

    def cancel_after_manifest(bucket_name, object_name, *args, **kwargs):
        written = original_put(bucket_name, object_name, *args, **kwargs)
        if object_name.endswith(".gpustack-manifest.json"):
            canceled.set()
        return written

    def flaky_remove(bucket_name, object_name):
        remove_calls.append(object_name)
        if len(remove_calls) < 3:
            raise OSError("temporary unavailable")
        original_remove(bucket_name, object_name)

    minio.put_object_if_absent = cancel_after_manifest
    minio.remove_object = flaky_remove
    client = ModelPreheatS3Client(
        minio,
        cancel_cleanup_attempts=3,
        cancel_cleanup_sleep=sleeps.append,
        cancel_cleanup_backoff=0.01,
    )

    with pytest.raises(ModelPreheatCanceled, match="canceled"):
        client.publish_generation(
            "bucket", "preheat", manifest, tmp_path, cancel_check=canceled.is_set
        )

    assert len(remove_calls) == 3
    assert sleeps == [0.01, 0.02]
    assert not any(
        name.endswith(".gpustack-manifest.json") for _, name in minio.objects
    )


def test_cancel_cleanup_persistent_failure_is_conflict_not_canceled(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    canceled = threading.Event()
    original_put = minio.put_object_if_absent

    def cancel_after_manifest(bucket_name, object_name, *args, **kwargs):
        written = original_put(bucket_name, object_name, *args, **kwargs)
        if object_name.endswith(".gpustack-manifest.json"):
            canceled.set()
        return written

    minio.put_object_if_absent = cancel_after_manifest
    minio.remove_object = lambda *args: (_ for _ in ()).throw(
        OSError("persistent unavailable")
    )
    client = ModelPreheatS3Client(
        minio,
        cancel_cleanup_attempts=3,
        cancel_cleanup_sleep=lambda delay: None,
    )

    with pytest.raises(ModelPreheatS3Conflict, match="cancel_cleanup_failed"):
        client.publish_generation(
            "bucket", "preheat", manifest, tmp_path, cancel_check=canceled.is_set
        )

    assert any(name.endswith(".gpustack-manifest.json") for _, name in minio.objects)


def test_ready_paths_are_isolated_by_selection_digest_for_different_excludes(
    tmp_path,
):
    first = _manifest(tmp_path)
    second = build_model_preheat_manifest(
        tmp_path,
        first.identity,
        cache_key=first.cache_key,
        selection_digest="different-selection-digest",
        generation_id="different-generation-id",
        exclude_patterns=["*.tmp"],
    )
    client = ModelPreheatS3Client(InMemoryMinio())

    assert client.ready_object("preheat", first) != client.ready_object(
        "preheat", second
    )
    assert "/selection-digest/ready.json" in client.ready_object("preheat", first)
    assert "/different-selection-digest/ready.json" in client.ready_object(
        "preheat", second
    )


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


def test_ready_manifest_rejects_oversized_stream(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    client.publish_generation("bucket", "preheat", manifest, tmp_path)
    ready_object = client.ready_object("preheat", manifest)
    manifest_object = client.manifest_object("preheat", manifest)
    oversized = b"{" + b"x" * (4 * 1024 * 1024)
    ready = json.loads(minio.objects[("bucket", ready_object)].data)
    ready["manifest_sha256"] = hashlib.sha256(oversized).hexdigest()
    minio.objects[("bucket", ready_object)] = StoredObject(
        json.dumps(ready).encode("utf-8")
    )
    minio.objects[("bucket", manifest_object)] = StoredObject(oversized)

    with pytest.raises(ModelPreheatS3ManifestError, match="s3_manifest_invalid"):
        client.read_ready_manifest(
            "bucket",
            "preheat",
            manifest.identity,
            cache_key=manifest.cache_key,
            selection_digest=manifest.selection_digest,
        )


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


def test_ready_manifest_rejects_tampered_manifest_sha256(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    client.publish_generation("bucket", "preheat", manifest, tmp_path)
    ready_object = client.ready_object("preheat", manifest)
    ready = json.loads(minio.objects[("bucket", ready_object)].data)
    ready["manifest_sha256"] = "0" * 64
    minio.objects[("bucket", ready_object)] = StoredObject(
        json.dumps(ready).encode("utf-8")
    )

    with pytest.raises(ModelPreheatS3ManifestError, match="s3_manifest_invalid"):
        client.read_ready_manifest(
            "bucket",
            "preheat",
            manifest.identity,
            cache_key=manifest.cache_key,
            selection_digest=manifest.selection_digest,
        )


def test_ready_manifest_rejects_tampered_manifest_content(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    client.publish_generation("bucket", "preheat", manifest, tmp_path)
    manifest_object = client.manifest_object("preheat", manifest)
    minio.objects[("bucket", manifest_object)] = StoredObject(b'{"files":[]}')

    with pytest.raises(ModelPreheatS3ManifestError, match="s3_manifest_invalid"):
        client.read_ready_manifest(
            "bucket",
            "preheat",
            manifest.identity,
            cache_key=manifest.cache_key,
            selection_digest=manifest.selection_digest,
        )


def test_publish_repairs_missing_ready_after_generation_was_uploaded(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)

    first = client.publish_generation("bucket", "preheat", manifest, tmp_path)
    ready_object = client.ready_object("preheat", manifest)
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
    ready_object = client.ready_object("preheat", manifest)
    minio.objects[("bucket", ready_object)] = StoredObject(
        json.dumps({"digest": "different"}).encode("utf-8")
    )

    with pytest.raises(ModelPreheatS3ManifestError, match="s3_manifest_invalid"):
        client.publish_generation("bucket", "preheat", manifest, tmp_path)


def test_ready_conflict_detected_after_generation_upload(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    ready_object = client.ready_object("preheat", manifest)
    file_object = client.generation_file_object("preheat", manifest, manifest.files[0])

    original = minio.put_object_if_absent

    def inject_conflict(bucket_name, object_name, *args, **kwargs):
        written = original(bucket_name, object_name, *args, **kwargs)
        if object_name == file_object and written:
            minio.objects[(bucket_name, ready_object)] = StoredObject(
                json.dumps({"digest": "other-generation"}).encode("utf-8")
            )
        return written

    minio.put_object_if_absent = inject_conflict

    with pytest.raises(ModelPreheatS3ManifestError, match="s3_manifest_invalid"):
        client.publish_generation("bucket", "preheat", manifest, tmp_path)


def test_ready_final_write_does_not_overwrite_concurrent_digest(tmp_path):
    manifest = _manifest(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    ready_object = client.ready_object("preheat", manifest)

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


def test_ready_write_uses_streaming_condition_and_keeps_concurrent_digest(tmp_path):
    manifest = _manifest(tmp_path)
    minio = ExecuteConditionalMinio()
    minio.concurrent_ready_digest = "other-generation"
    client = ModelPreheatS3Client(minio)
    ready_object = client.ready_object("preheat", manifest)

    with pytest.raises(ReadyGenerationConflict, match="ready_generation_conflict"):
        client.publish_generation("bucket", "preheat", manifest, tmp_path)

    ready = json.loads(minio.objects[("bucket", ready_object)].data)
    assert ready["digest"] == "other-generation"
    ready_call = minio.execute_calls[-1]
    assert ready_call["method"] == "PUT"
    assert ready_call["bucket_name"] == "bucket"
    assert ready_call["object_name"] == ready_object
    assert ready_call["headers"]["If-None-Match"] == "*"
    assert ready_call["headers"]["x-amz-meta-model-preheat-digest"] == manifest.digest


def test_ready_write_without_conditional_capability_does_not_put_ready(tmp_path):
    manifest = _manifest(tmp_path)
    minio = NoConditionalMinio()
    client = ModelPreheatS3Client(minio)
    ready_object = client.ready_object("preheat", manifest)

    with pytest.raises(ModelPreheatS3Conflict, match="conditional_create_unsupported"):
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
    object.__setattr__(manifest, "cache_key", "cache-key")
    object.__setattr__(manifest, "selection_digest", "selection-digest")
    object.__setattr__(manifest, "generation_id", "generation-id")
    object.__setattr__(manifest, "exclude_patterns", ())
    object.__setattr__(manifest, "schema_version", 1)

    with pytest.raises(ValueError, match="manifest_path_escape"):
        ModelPreheatS3Client(InMemoryMinio()).publish_generation(
            "bucket",
            "preheat",
            manifest,
            tmp_path,
        )
