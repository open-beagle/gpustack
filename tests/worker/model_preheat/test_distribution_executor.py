from dataclasses import replace

import pytest

from gpustack.worker.model_preheat.executor import (
    SeedExecutionRequest,
    TargetExecutionRequest,
    execute_seed_preheat,
    execute_target_preheat,
)
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
from gpustack.worker.model_preheat.local_cache import inspect_local_cache
from gpustack.worker.model_preheat.s3_client import ModelPreheatS3Client
from tests.worker.model_preheat.test_seed_executor import (
    InMemoryMinio,
    StoredObject,
    _write_model,
)


def _identity():
    return ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="resolved-commit",
        file_patterns=["config.json", "weights/model.bin"],
    )


def _seed_request(tmp_path):
    return SeedExecutionRequest(
        cache_dir=tmp_path / "seed-cache",
        target_dir=tmp_path / "seed-cache" / "model_scope" / "org" / "model",
        cache_key="cache-key",
        task_id=8,
        attempt=1,
        identity=_identity(),
        selection_digest="selection-digest",
        generation_id="parent-generation-id",
        exclude_patterns=[],
        bucket="models",
        prefix="preheat",
    )


def _target_request(tmp_path):
    return TargetExecutionRequest(
        cache_dir=tmp_path / "target-cache",
        target_dir=tmp_path / "target-cache" / "model_scope" / "org" / "model",
        cache_key="cache-key",
        task_id=8,
        attempt=1,
        identity=_identity(),
        selection_digest="selection-digest",
        generation_id="parent-generation-id",
        exclude_patterns=[],
        bucket="models",
        prefix="preheat",
    )


def _published_client(tmp_path):
    minio = InMemoryMinio()
    seed_request = _seed_request(tmp_path)
    execute_seed_preheat(
        seed_request,
        ModelPreheatS3Client(minio),
        download_to_staging=lambda identity, staging, **kwargs: _write_model(staging),
    )
    return minio


def test_target_downloads_ready_generation_and_publishes_atomically(tmp_path):
    minio = _published_client(tmp_path)
    request = _target_request(tmp_path)

    result = execute_target_preheat(request, ModelPreheatS3Client(minio))

    inspection = inspect_local_cache(
        request.cache_dir, request.target_dir, request.cache_key
    )
    assert result["state"] == "ready"
    assert result["downloaded"] == 2
    assert inspection.state == "valid"


def test_target_reports_resumable_cursor_after_each_real_file_boundary(tmp_path):
    minio = _published_client(tmp_path)
    request = _target_request(tmp_path)
    cursors = []

    result = execute_target_preheat(
        request,
        ModelPreheatS3Client(minio),
        progress_callback=lambda completed, downloaded_size, total_size: cursors.append(
            (list(completed), downloaded_size, total_size)
        ),
    )

    assert result["state"] == "ready"
    assert [cursor[0] for cursor in cursors] == [
        ["config.json"],
        ["config.json", "weights/model.bin"],
    ]
    assert cursors[-1][1] == cursors[-1][2]


def test_s3_download_honors_optional_bandwidth_limit(tmp_path, monkeypatch):
    from gpustack.worker.model_preheat import s3_client as s3_client_module

    minio = InMemoryMinio()
    minio.objects[("models", "large.bin")] = StoredObject(b"x" * (2 * 1024 * 1024))
    sleeps = []
    monkeypatch.setattr(s3_client_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(s3_client_module.time, "sleep", sleeps.append)

    chunks = list(
        ModelPreheatS3Client(minio).stream_object(
            "models",
            "large.bin",
            chunk_size=1024 * 1024,
            bandwidth_limit_mbps=8,
        )
    )

    assert sum(map(len, chunks)) == 2 * 1024 * 1024
    assert len(chunks) == 32
    assert max(map(len, chunks)) <= 64 * 1024
    assert len(sleeps) == 32
    assert sleeps[-1] == pytest.approx(2.097152)


def test_seed_delegation_preserves_bandwidth_limit_and_resumable_cursor(
    tmp_path, monkeypatch
):
    request = replace(
        _seed_request(tmp_path),
        bandwidth_limit_mbps=8,
        resumable_cursor={"completed_files": ["config.json"]},
    )
    delegated = []

    class ReadyClient:
        def read_ready_manifest(self, *args, **kwargs):
            return object()

    def capture(target_request, *args, **kwargs):
        delegated.append(target_request)
        return {"state": "ready"}

    monkeypatch.setattr(
        "gpustack.worker.model_preheat.executor.execute_target_preheat", capture
    )

    assert execute_seed_preheat(request, ReadyClient())["state"] == "ready"
    assert delegated[0].bandwidth_limit_mbps == 8
    assert delegated[0].resumable_cursor == {"completed_files": ["config.json"]}


def test_target_reuses_partial_file_and_retries_checksum_mismatch(tmp_path):
    minio = _published_client(tmp_path)
    request = _target_request(tmp_path)
    staging = request.cache_dir / ".preheat" / "8" / "1"
    _write_model(staging)
    (staging / "weights" / "model.bin").unlink()

    result = execute_target_preheat(request, ModelPreheatS3Client(minio))

    assert result["state"] == "ready"
    assert result["downloaded"] == 1
    assert result["skipped"] == 1


def test_target_resume_reuses_only_files_confirmed_by_persisted_cursor(tmp_path):
    minio = _published_client(tmp_path)
    request = replace(
        _target_request(tmp_path),
        attempt=2,
        resumable_cursor={
            "completed_files": ["config.json"],
            "staging_exists": True,
        },
    )
    previous = request.cache_dir / ".preheat" / str(request.task_id) / "1"
    _write_model(previous)

    result = execute_target_preheat(request, ModelPreheatS3Client(minio))

    assert result["state"] == "ready"
    assert result["skipped"] == 1
    assert result["downloaded"] == 1


def test_target_resume_reuses_encoded_space_and_unicode_paths(tmp_path):
    minio = _published_client(tmp_path)
    manifest = ModelPreheatS3Client(minio).read_ready_manifest(
        "models",
        "preheat",
        _identity(),
        cache_key="cache-key",
        selection_digest="selection-digest",
    )
    renamed_files = [
        replace(manifest.files[0], path="weights/model%207b.bin"),
        replace(
            manifest.files[1],
            path="%E8%B5%84%E6%96%99/%E6%A8%A1%E5%9E%8B.bin",
        ),
    ]
    manifest = replace(manifest, files=renamed_files)
    request = replace(
        _target_request(tmp_path),
        attempt=2,
        resumable_cursor={
            "completed_files": [file.path for file in manifest.files],
            "staging_exists": True,
        },
    )
    previous = request.cache_dir / ".preheat" / str(request.task_id) / "1"
    (previous / "weights").mkdir(parents=True)
    (previous / "weights" / "model 7b.bin").write_bytes(b"config")
    (previous / "资料").mkdir(parents=True)
    (previous / "资料" / "模型.bin").write_bytes(b"weights")
    client = ModelPreheatS3Client(minio)
    downloaded = []

    client.read_ready_manifest = lambda *args, **kwargs: manifest
    client.download_generation_file = lambda *args, **kwargs: downloaded.append(args)
    result = execute_target_preheat(request, client)

    assert result["state"] == "ready"
    assert result["skipped"] == 2
    assert result["downloaded"] == 0
    assert downloaded == []


def test_target_new_attempt_reuses_only_verified_previous_staging_files(
    tmp_path, monkeypatch
):
    minio = _published_client(tmp_path)
    request = TargetExecutionRequest(
        **{**_target_request(tmp_path).__dict__, "attempt": 2}
    )
    previous = request.cache_dir / ".preheat" / "8" / "1"
    _write_model(previous)
    (previous / "weights" / "model.bin").write_bytes(b"corrupt")
    client = ModelPreheatS3Client(minio)
    downloaded = []
    original = client.download_generation_file

    def track_download(bucket, prefix, manifest, file, target):
        downloaded.append(file.path)
        return original(bucket, prefix, manifest, file, target)

    monkeypatch.setattr(client, "download_generation_file", track_download)
    monkeypatch.setattr(
        "gpustack.worker.model_preheat.executor.os.link",
        lambda *args: (_ for _ in ()).throw(OSError()),
    )

    result = execute_target_preheat(request, client)

    assert result["state"] == "ready"
    assert downloaded == ["weights/model.bin"]
    assert result["skipped"] == 1


def test_target_checksum_mismatch_and_cancellation_keep_staging(tmp_path, monkeypatch):
    minio = _published_client(tmp_path)
    request = _target_request(tmp_path)
    client = ModelPreheatS3Client(minio)

    def corrupt_download(bucket, prefix, manifest, file, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"bad")

    monkeypatch.setattr(client, "download_generation_file", corrupt_download)
    mismatch = execute_target_preheat(request, client)

    canceled_request = TargetExecutionRequest(**{**request.__dict__, "attempt": 2})
    canceled = execute_target_preheat(
        canceled_request, ModelPreheatS3Client(minio), cancel_check=lambda: True
    )

    assert mismatch["error_code"] == "checksum_mismatch"
    assert canceled["error_code"] == "canceled"
    assert (canceled_request.cache_dir / ".preheat" / "8" / "2").exists()


def test_embedded_target_with_valid_local_manifest_does_not_download(
    tmp_path, monkeypatch
):
    minio = _published_client(tmp_path)
    request = _seed_request(tmp_path)
    client = ModelPreheatS3Client(minio)

    def unexpected_download(*args, **kwargs):
        raise AssertionError("embedded worker should not download from S3")

    monkeypatch.setattr(client, "download_generation_file", unexpected_download)
    result = execute_target_preheat(TargetExecutionRequest(**request.__dict__), client)

    assert result["state"] == "ready"
    assert result["downloaded"] == 0


def test_target_rejects_manifest_with_more_than_1024_files(tmp_path):
    minio = _published_client(tmp_path)
    request = _target_request(tmp_path)
    client = ModelPreheatS3Client(minio)
    ready_object = client._ready_object(
        request.prefix, request.identity, request.selection_digest
    )
    ready = json.loads(minio.objects[(request.bucket, ready_object)].data)
    manifest_object = ready["manifest_object"]
    payload = json.loads(minio.objects[(request.bucket, manifest_object)].data)
    template = payload["files"][0]
    payload["files"] = [
        {**template, "path": f"files/{index}.bin"} for index in range(1025)
    ]
    oversized = json.dumps(payload).encode("utf-8")
    ready["manifest_sha256"] = hashlib.sha256(oversized).hexdigest()
    minio.objects[(request.bucket, manifest_object)] = StoredObject(oversized)
    minio.objects[(request.bucket, ready_object)] = StoredObject(
        json.dumps(ready).encode("utf-8")
    )

    result = execute_target_preheat(request, client)

    assert result["state"] == "error"
    assert result["error_code"] == "s3_manifest_invalid"
    assert not request.target_dir.exists()


import hashlib
import json
