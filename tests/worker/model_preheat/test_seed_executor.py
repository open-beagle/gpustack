import asyncio
import hashlib
import json
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpustack.worker.model_preheat.executor import (
    SeedExecutionRequest,
    TargetExecutionRequest,
    build_preheat_role_handlers,
    execute_seed_preheat,
    execute_target_preheat,
)
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
from gpustack.worker.model_preheat.local_cache import (
    inspect_local_cache,
    write_trusted_manifest,
)
from gpustack.worker.model_preheat.manifest import build_model_preheat_manifest
from gpustack.worker.model_preheat.s3_client import (
    ModelPreheatCanceled,
    ModelPreheatS3Client,
)
from gpustack.worker import downloaders


@dataclass
class StoredObject:
    data: bytes
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def size(self):
        return len(self.data)


class Response:
    def __init__(self, data):
        self._data = data
        self._offset = 0

    def read(self, length=None):
        if length is None:
            result = self._data[self._offset :]
            self._offset = len(self._data)
            return result
        result = self._data[self._offset : self._offset + length]
        self._offset += len(result)
        return result

    def close(self):
        pass

    def release_conn(self):
        pass


class InMemoryMinio:
    def __init__(self):
        self.objects = {}
        self.uploads = []

    def stat_object(self, bucket, name):
        try:
            return self.objects[(bucket, name)]
        except KeyError as exc:
            raise FileNotFoundError(name) from exc

    def get_object(self, bucket, name):
        return Response(self.stat_object(bucket, name).data)

    def fput_object(self, bucket, name, file_path, metadata=None):
        self.uploads.append(name)
        with open(file_path, "rb") as file:
            self.objects[(bucket, name)] = StoredObject(file.read(), metadata or {})

    def put_object(self, bucket, name, data, length, content_type=None, metadata=None):
        del content_type
        self.uploads.append(name)
        self.objects[(bucket, name)] = StoredObject(data.read(length), metadata or {})

    def put_object_if_absent(
        self, bucket, name, data, length, content_type=None, metadata=None
    ):
        if (bucket, name) in self.objects:
            return False
        self.put_object(bucket, name, data, length, content_type, metadata)
        return True

    def list_objects(self, bucket, prefix, recursive=True):
        del recursive
        return [
            type("Object", (), {"object_name": name})
            for object_bucket, name in self.objects
            if object_bucket == bucket and name.startswith(prefix)
        ]


class InterruptedTransferMinio(InMemoryMinio):
    def __init__(self):
        super().__init__()
        self.file_puts = 0
        self.interrupt_put = False
        self.interrupt_get = False

    def put_object_if_absent(
        self, bucket, name, data, length, content_type=None, metadata=None
    ):
        if "/files/" in name:
            self.file_puts += 1
            if self.interrupt_put and self.file_puts == 2:
                data.read(max(1, length // 2))
                self.interrupt_put = False
                raise OSError("injected upload interruption")
        return super().put_object_if_absent(
            bucket, name, data, length, content_type, metadata
        )

    def get_object(self, bucket, name):
        response = super().get_object(bucket, name)
        if not self.interrupt_get or "/files/" not in name:
            return response
        self.interrupt_get = False
        return InterruptedResponse(response._data)


class InterruptedResponse(Response):
    def __init__(self, data):
        super().__init__(data)
        self._interrupted = False

    def read(self, length=None):
        if self._interrupted:
            raise OSError("injected download interruption")
        self._interrupted = True
        chunk_size = max(1, len(self._data) // 2)
        return super().read(chunk_size)


def _identity():
    return ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="resolved-commit",
        file_patterns=["config.json", "weights/model.bin"],
    )


def _write_model(root):
    (root / "weights").mkdir(parents=True, exist_ok=True)
    (root / "weights" / "model.bin").write_bytes(b"weights")
    (root / "config.json").write_bytes(b"config")


def _request(tmp_path):
    return SeedExecutionRequest(
        cache_dir=tmp_path / "cache",
        target_dir=tmp_path / "cache" / "model_scope" / "org" / "model",
        cache_key="cache-key",
        task_id=8,
        attempt=1,
        identity=_identity(),
        selection_digest="selection-digest",
        generation_id="parent-generation-id",
        exclude_patterns=["*.tmp"],
        bucket="models",
        prefix="preheat",
    )


def test_seed_reuses_trusted_local_cache_and_publishes_ready(tmp_path):
    request = _request(tmp_path)
    _write_model(request.target_dir)
    manifest = build_model_preheat_manifest(
        request.target_dir,
        request.identity,
        cache_key=request.cache_key,
        selection_digest=request.selection_digest,
        generation_id=request.generation_id,
        exclude_patterns=request.exclude_patterns,
    )
    write_trusted_manifest(request.cache_dir, request.cache_key, manifest)
    minio = InMemoryMinio()

    result = execute_seed_preheat(request, ModelPreheatS3Client(minio))

    assert result["state"] == "ready"
    assert result["local_cache_state"] == "valid"
    assert result["manifest_digest"] == manifest.digest
    assert result["generation_id"] == "parent-generation-id"
    assert result["ready_path"].endswith("ready.json")
    assert "access" not in json.dumps(result)


def test_seed_rebuilds_current_generation_from_valid_old_local_sidecar(
    tmp_path, monkeypatch
):
    request = _request(tmp_path)
    _write_model(request.target_dir)
    old_manifest = build_model_preheat_manifest(
        request.target_dir,
        request.identity,
        cache_key=request.cache_key,
        selection_digest=request.selection_digest,
        generation_id="old-generation-id",
        exclude_patterns=request.exclude_patterns,
    )
    write_trusted_manifest(request.cache_dir, request.cache_key, old_manifest)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)

    result = execute_seed_preheat(request, client)
    published = client.read_ready_manifest(
        request.bucket,
        request.prefix,
        request.identity,
        cache_key=request.cache_key,
        selection_digest=request.selection_digest,
    )

    assert result["state"] == "ready"
    assert result["generation_id"] == request.generation_id
    assert published.generation_id == request.generation_id

    monkeypatch.setattr(
        client,
        "download_generation_file",
        lambda *args, **kwargs: pytest.fail("embedded target 不应下载 S3 文件"),
    )

    target_result = execute_target_preheat(
        TargetExecutionRequest(**request.__dict__),
        client,
        cancel_check=None,
    )
    inspection = inspect_local_cache(
        request.cache_dir, request.target_dir, request.cache_key, published
    )

    assert target_result["state"] == "ready"
    assert target_result["downloaded"] == 0
    assert inspection.state == "valid"
    assert inspection.manifest == published


def test_seed_downloads_resolved_identity_to_staging_and_skips_existing_objects(
    tmp_path,
):
    request = _request(tmp_path)
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    seen = []

    def downloader(identity, staging_dir, **kwargs):
        seen.append((identity.revision, staging_dir, kwargs))
        _write_model(staging_dir)

    first = execute_seed_preheat(request, client, download_to_staging=downloader)
    second_request = SeedExecutionRequest(**{**request.__dict__, "attempt": 2})
    second = execute_seed_preheat(
        second_request, client, download_to_staging=downloader
    )

    assert seen[0][0] == "resolved-commit"
    assert first["uploaded"] == 3
    assert second["skipped"] >= 3
    assert second["state"] == "ready"


def test_seed_new_attempt_prefills_previous_partial_before_hub_resume(tmp_path):
    request = SeedExecutionRequest(**{**_request(tmp_path).__dict__, "attempt": 2})
    previous = request.cache_dir / ".preheat" / "8" / "1"
    (previous / ".cache" / "huggingface").mkdir(parents=True)
    (previous / ".cache" / "huggingface" / "resume.json").write_text("resume")
    (previous / "config.json").write_bytes(b"config")
    outside = tmp_path / "outside-secret"
    outside.write_text("secret")
    (previous / "escape").symlink_to(outside)
    (previous / "special").parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(previous / "special")
    seen = []

    def resume_downloader(identity, staging, **kwargs):
        assert not (staging / "escape").exists()
        assert not (staging / "special").exists()
        seen.append(
            (
                identity.revision,
                (staging / "config.json").read_bytes(),
                (staging / ".cache" / "huggingface" / "resume.json").read_text(),
            )
        )
        shutil.rmtree(staging / ".cache")
        _write_model(staging)

    result = execute_seed_preheat(
        request,
        ModelPreheatS3Client(InMemoryMinio()),
        download_to_staging=resume_downloader,
    )

    assert seen == [("resolved-commit", b"config", "resume")]
    assert result["state"] == "ready"


def test_seed_upload_interruption_is_reclaimed_without_publishing_partial_ready(
    tmp_path,
):
    request = _request(tmp_path)
    minio = InterruptedTransferMinio()
    minio.interrupt_put = True
    client = ModelPreheatS3Client(minio)

    def initial_download(identity, staging, exclude_patterns):
        del identity, exclude_patterns
        _write_model(staging)

    interrupted = execute_seed_preheat(
        request,
        client,
        download_to_staging=initial_download,
    )

    assert interrupted["state"] == "error"
    assert interrupted["error_code"] == "worker_execution_failed"
    assert interrupted["cursor"]["staging_exists"] is True
    assert len([name for _, name in minio.objects if "/files/" in name]) == 1
    assert not any(
        name.endswith(".gpustack-manifest.json") for _, name in minio.objects
    )
    assert not any(name.endswith("ready.json") for _, name in minio.objects)

    resumed_request = SeedExecutionRequest(**{**request.__dict__, "attempt": 2})

    def resumed_download(identity, staging, exclude_patterns):
        del identity, exclude_patterns
        assert (staging / "config.json").read_bytes() == b"config"
        assert (staging / "weights" / "model.bin").read_bytes() == b"weights"

    resumed = execute_seed_preheat(
        resumed_request,
        client,
        download_to_staging=resumed_download,
    )

    assert resumed["state"] == "ready"
    assert any(name.endswith(".gpustack-manifest.json") for _, name in minio.objects)
    assert any(name.endswith("ready.json") for _, name in minio.objects)


def test_s3_download_stream_interruption_keeps_staging_for_next_attempt(tmp_path):
    source_request = _request(tmp_path)
    _write_model(source_request.target_dir)
    source_manifest = build_model_preheat_manifest(
        source_request.target_dir,
        source_request.identity,
        cache_key=source_request.cache_key,
        selection_digest=source_request.selection_digest,
        generation_id=source_request.generation_id,
        exclude_patterns=source_request.exclude_patterns,
    )
    write_trusted_manifest(
        source_request.cache_dir, source_request.cache_key, source_manifest
    )
    minio = InterruptedTransferMinio()
    client = ModelPreheatS3Client(minio)
    assert execute_seed_preheat(source_request, client)["state"] == "ready"

    target_request = TargetExecutionRequest(
        **{
            **source_request.__dict__,
            "cache_dir": tmp_path / "target-cache",
            "target_dir": tmp_path / "target-cache" / "model",
            "task_id": 9,
        }
    )
    minio.interrupt_get = True

    interrupted = execute_target_preheat(target_request, client)

    assert interrupted["state"] == "error"
    assert interrupted["error_code"] == "worker_execution_failed"
    assert interrupted["cursor"]["staging_exists"] is True
    assert not target_request.target_dir.exists()

    resumed_request = TargetExecutionRequest(
        **{**target_request.__dict__, "attempt": 2}
    )
    resumed = execute_target_preheat(resumed_request, client)

    assert resumed["state"] == "ready"
    assert (
        inspect_local_cache(
            resumed_request.cache_dir,
            resumed_request.target_dir,
            resumed_request.cache_key,
            source_manifest,
        ).state.value
        == "valid"
    )


def test_seed_reports_s3_conflict_when_cancel_cleanup_never_succeeds(tmp_path):
    request = _request(tmp_path)
    minio = InMemoryMinio()
    canceled = threading.Event()
    original_put = minio.put_object_if_absent

    def cancel_after_manifest(bucket_name, object_name, *args, **kwargs):
        written = original_put(bucket_name, object_name, *args, **kwargs)
        if object_name.endswith(".gpustack-manifest.json"):
            canceled.set()
        return written

    minio.put_object_if_absent = cancel_after_manifest
    minio.remove_object = lambda *args: (_ for _ in ()).throw(OSError())
    client = ModelPreheatS3Client(
        minio,
        cancel_cleanup_attempts=2,
        cancel_cleanup_sleep=lambda delay: None,
    )

    result = execute_seed_preheat(
        request,
        client,
        download_to_staging=lambda identity, staging, **kwargs: _write_model(staging),
        cancel_check=canceled.is_set,
    )

    assert result["state"] == "error"
    assert result["error_code"] == "s3_object_conflict"


def test_seed_with_existing_ready_downloads_generation_instead_of_hub(tmp_path):
    source_request = SeedExecutionRequest(
        **{
            **_request(tmp_path / "source").__dict__,
            "generation_id": "historical-generation-id",
        }
    )
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    source_result = execute_seed_preheat(
        source_request,
        client,
        download_to_staging=lambda identity, staging, **kwargs: _write_model(staging),
    )
    request = SeedExecutionRequest(
        **{
            **_request(tmp_path / "target").__dict__,
            "task_id": 9,
            "generation_id": "new-parent-generation-id",
        }
    )
    ready_object = source_result["ready_path"]
    original_ready = minio.objects[("models", ready_object)].data

    result = execute_seed_preheat(
        request,
        client,
        download_to_staging=lambda *_: pytest.fail("不得重新从 Hub 下载"),
    )

    assert result["state"] == "ready"
    assert result["downloaded"] == 2
    assert result["generation_id"] == "historical-generation-id"
    assert minio.objects[("models", ready_object)].data == original_ready
    assert (request.target_dir / "weights" / "model.bin").read_bytes() == b"weights"


def test_seed_cancellation_never_writes_generation_manifest_or_ready(tmp_path):
    request = _request(tmp_path)
    minio = InMemoryMinio()

    result = execute_seed_preheat(
        request,
        ModelPreheatS3Client(minio),
        download_to_staging=lambda identity, staging, **kwargs: _write_model(staging),
        cancel_check=lambda: True,
    )

    assert result["state"] == "error"
    assert result["error_code"] == "canceled"
    assert not any(
        name.endswith(".gpustack-manifest.json") for _, name in minio.objects
    )
    assert not any(name.endswith("ready.json") for _, name in minio.objects)


def test_resolved_revision_download_uses_public_hubs_and_explicit_staging(
    tmp_path, monkeypatch
):
    calls = []

    def huggingface(**kwargs):
        calls.append(("huggingface", kwargs))

    def modelscope(**kwargs):
        calls.append(("modelscope", kwargs))

    monkeypatch.setattr(downloaders, "snapshot_download", huggingface)
    monkeypatch.setattr(downloaders, "modelscope_snapshot_download", modelscope)
    huggingface_identity = ModelPreheatIdentity(
        source="huggingface",
        model_id="org/model",
        revision="sha-123",
        file_patterns=["weights/*.bin"],
    )

    downloaders.download_resolved_revision_to_staging(
        huggingface_identity, tmp_path / "hf", exclude_patterns=["*.tmp"]
    )
    downloaders.download_resolved_revision_to_staging(
        _identity(), tmp_path / "ms", exclude_patterns=["*.tmp"]
    )

    assert calls[0][1]["revision"] == "sha-123"
    assert calls[0][1]["local_dir"] == str(tmp_path / "hf")
    assert calls[0][1]["ignore_patterns"] == ["*.tmp"]
    assert calls[1][1]["revision"] == "resolved-commit"
    assert calls[1][1]["local_dir"] == str(tmp_path / "ms")
    assert calls[1][1]["ignore_patterns"] == ["*.tmp"]


def test_empty_include_passes_none_to_hub_downloaders(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(
        downloaders,
        "snapshot_download",
        lambda **kwargs: calls.append(("huggingface", kwargs)),
    )
    monkeypatch.setattr(
        downloaders,
        "modelscope_snapshot_download",
        lambda **kwargs: calls.append(("modelscope", kwargs)),
    )
    for source in ("huggingface", "modelscope"):
        identity = ModelPreheatIdentity(
            source=source,
            model_id="org/model",
            revision="resolved-commit",
            file_patterns=[],
        )
        downloaders.download_resolved_revision_to_staging(
            identity,
            tmp_path / source,
            exclude_patterns=["*.tmp"],
        )

    assert [kwargs["allow_patterns"] for _, kwargs in calls] == [None, None]


def test_modelscope_download_stops_at_file_boundary_and_reports_cursor(
    tmp_path, monkeypatch
):
    class FakeModelScopeHubApi:
        def list_repo_files(self, model_id, repo_type, *, revision, recursive):
            assert model_id == "org/model"
            assert repo_type == "model"
            assert revision == "resolved-commit"
            assert recursive is True
            return [
                SimpleNamespace(path="config.json", size=2),
                SimpleNamespace(path="weights/model.bin", size=3),
            ]

    downloaded = []
    canceled = False

    def download_file(*, model_id, file_path, revision, local_dir):
        assert model_id == "org/model"
        assert revision == "resolved-commit"
        target = Path(local_dir) / file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        downloaded.append(file_path)

    def report_progress(completed, downloaded_size, total_size):
        nonlocal canceled
        assert list(completed) == ["config.json"]
        assert (downloaded_size, total_size) == (2, 5)
        canceled = True

    monkeypatch.setattr(downloaders, "ModelScopeHubApi", FakeModelScopeHubApi)
    monkeypatch.setattr(downloaders, "model_file_download", download_file)

    with pytest.raises(ModelPreheatCanceled):
        downloaders.download_resolved_revision_to_staging(
            _identity(),
            tmp_path / "staging",
            cancel_check=lambda: canceled,
            progress_callback=report_progress,
        )
    assert downloaded == ["config.json"]


def test_modelscope_hub_runtime_dependency_is_declared_and_locked():
    declared = set()
    in_runtime_dependencies = False
    for line in (
        (Path.cwd() / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    ):
        stripped = line.strip()
        if stripped.startswith("["):
            in_runtime_dependencies = stripped == "[tool.poetry.dependencies]"
            continue
        if (
            in_runtime_dependencies
            and stripped
            and not stripped.startswith("#")
            and "=" in stripped
        ):
            declared.add(stripped.split("=", 1)[0].strip())

    locked = set()
    in_package = False
    for line in (Path.cwd() / "poetry.lock").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[[package]]":
            in_package = True
            continue
        if stripped.startswith("["):
            in_package = False
            continue
        if in_package and stripped.startswith("name = "):
            locked.add(stripped.removeprefix("name = ").strip('"'))

    assert "modelscope-hub" in declared
    assert "modelscope-hub" in locked


def test_handler_cancellation_waits_for_thread_and_never_publishes_ready(
    tmp_path, monkeypatch
):
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    started = threading.Event()

    def slow_download(identity, staging_dir, **kwargs):
        del identity, kwargs
        _write_model(staging_dir)
        started.set()
        threading.Event().wait(0.05)

    monkeypatch.setattr(
        "gpustack.worker.model_preheat.s3_client.ModelPreheatS3Client.from_minio",
        classmethod(lambda cls, **kwargs: client),
    )
    monkeypatch.setattr(
        downloaders,
        "download_resolved_revision_to_staging",
        slow_download,
    )
    payload = SimpleNamespace(
        worker_task_id=8,
        attempt=1,
        task={
            "id": 8,
            "source": "modelscope",
            "model_id": "org/model",
            "resolved_revision": "resolved-commit",
            "include_patterns": ["config.json", "weights/model.bin"],
            "exclude_patterns": [],
            "cache_key": "cache-key",
            "selection_digest": "selection-digest",
            "generation_id": "parent-generation-id",
        },
        profile=SimpleNamespace(
            endpoint="https://s3.example.invalid",
            access_key="access-key",
            secret_key="secret-key",
            tls_enabled=True,
            tls_verify=True,
            region="",
            bucket="models",
            prefix="preheat",
            use_virtual_hosted_style=False,
        ),
    )
    handler = build_preheat_role_handlers(tmp_path / "cache")["seed"]

    async def run():
        running = asyncio.create_task(handler(payload, None))
        await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

    asyncio.run(run())

    assert not any(
        name.endswith(".gpustack-manifest.json") for _, name in minio.objects
    )
    assert not any(name.endswith("ready.json") for _, name in minio.objects)
