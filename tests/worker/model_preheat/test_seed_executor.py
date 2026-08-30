import hashlib
import json
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from unittest.mock import patch

import pytest
from modelscope.hub.errors import NotExistError as ModelScopeNotExistError

from gpustack.worker.model_preheat.executor import (
    SeedExecutionRequest,
    TargetExecutionRequest,
    TrustedLocalCandidate,
    execute_seed_preheat,
    execute_target_preheat,
)
from gpustack.worker.downloaders import download_resolved_revision_to_staging
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
from gpustack.worker.model_preheat.manifest import build_model_preheat_manifest
from gpustack.worker.model_preheat.s3_client import ModelPreheatS3Client
from gpustack.routes.model_preheat_worker_tasks import _validated_preheat_result


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
        self.downloads = []
        self._lock = threading.Lock()

    def stat_object(self, bucket, name):
        try:
            return self.objects[(bucket, name)]
        except KeyError as exc:
            raise FileNotFoundError(name) from exc

    def get_object(self, bucket, name):
        self.downloads.append(name)
        return Response(self.stat_object(bucket, name).data)

    def fput_object(self, bucket, name, file_path, metadata=None):
        with open(file_path, "rb") as file:
            data = file.read()
        with self._lock:
            self.uploads.append(name)
            self.objects[(bucket, name)] = StoredObject(data, metadata or {})

    def put_object(self, bucket, name, data, length, content_type=None, metadata=None):
        del content_type
        payload = data.read(length)
        with self._lock:
            self.uploads.append(name)
            self.objects[(bucket, name)] = StoredObject(payload, metadata or {})

    def put_object_if_absent(
        self, bucket, name, data, length, content_type=None, metadata=None
    ):
        payload = data.read(length)
        with self._lock:
            if (bucket, name) in self.objects:
                return False
            self.uploads.append(name)
            self.objects[(bucket, name)] = StoredObject(payload, metadata or {})
            return True

    def list_objects(self, bucket, prefix, recursive=True):
        del recursive
        return [
            type("Object", (), {"object_name": name})
            for object_bucket, name in self.objects
            if object_bucket == bucket and name.startswith(prefix)
        ]


def _identity(*, requested_revision="master", patterns=None):
    return ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="resolved-commit",
        requested_revision=requested_revision,
        file_patterns=patterns or ["config.json", "weights/model.bin"],
    )


def _write_model(root):
    (root / "weights").mkdir(parents=True, exist_ok=True)
    (root / "weights" / "model.bin").write_bytes(b"weights")
    (root / "config.json").write_bytes(b"config")


def _seed_request(tmp_path, **changes):
    request = SeedExecutionRequest(
        cache_dir=tmp_path / "cache",
        target_dir=tmp_path / "cache" / "model_scope" / "org" / "model",
        task_id=8,
        attempt=1,
        request_digest=_identity().request_digest,
        identity=_identity(),
        exclude_patterns=(),
        bucket="models",
        prefix="model-storage",
        source_fallback_enabled=True,
    )
    return replace(request, **changes)


def _target_request(tmp_path, manifest, client):
    return TargetExecutionRequest(
        cache_dir=tmp_path / "target-cache",
        target_dir=tmp_path / "target-cache" / "model_scope" / "org" / "model",
        task_id=9,
        attempt=1,
        request_digest=_identity().request_digest,
        identity=_identity(),
        exclude_patterns=(),
        bucket="models",
        prefix="model-storage",
        artifact_id=manifest.artifact_id,
        manifest_path=client.artifact_manifest_object("model-storage", manifest),
    )


def test_ollama_pending_seed_keeps_auth_cache_outside_artifact_and_returns_revision(
    tmp_path,
):
    identity = ModelPreheatIdentity(
        source="ollama_library",
        model_id="llama3:latest",
        revision="ollama-pending",
        requested_revision="latest",
        file_patterns=(),
    )
    request = _seed_request(
        tmp_path,
        identity=identity,
        request_digest=identity.request_digest,
        exclude_patterns=(),
        install_local=False,
    )
    private_cache = request.cache_dir / ".ollama-auth"

    def download(_identity, staging, **kwargs):
        assert kwargs["private_cache_dir"] == private_cache
        private_cache.mkdir(parents=True)
        (private_cache / "id_ed25519").write_bytes(b"private")
        (private_cache / "id_ed25519.pub").write_bytes(b"public")
        (staging / "llama3_latest").write_bytes(b"model")

    minio = InMemoryMinio()
    result = execute_seed_preheat(
        request,
        ModelPreheatS3Client(minio),
        download_to_staging=download,
    )

    assert result["state"] == "ready"
    assert result["resolved_revision"].startswith("local-snapshot-")
    assert (
        _validated_preheat_result(result)["resolved_revision"]
        == result["resolved_revision"]
    )
    manifest = next(
        payload.data
        for (bucket, name), payload in minio.objects.items()
        if bucket == "models" and name.endswith("/manifest.json")
    )
    files = json.loads(manifest)["files"]
    assert [item["path"] for item in files] == ["llama3_latest"]
    assert all("id_ed25519" not in name for name in minio.uploads)


def test_seed_reuses_trusted_local_and_publishes_unified_artifact(tmp_path):
    root = tmp_path / "trusted"
    _write_model(root)
    request = _seed_request(
        tmp_path,
        trusted_local_candidate=TrustedLocalCandidate(
            source="model_file",
            root=root,
            paths=(root,),
            repository_complete=True,
        ),
    )
    minio = InMemoryMinio()
    hub_calls = []

    result = execute_seed_preheat(
        request,
        ModelPreheatS3Client(minio),
        download_to_staging=lambda *args, **kwargs: hub_calls.append(True),
    )

    assert result["state"] == "ready"
    assert result["transfer_source"] == "current_node"
    assert len(result["artifact_id"]) == 64
    assert len(result["manifest_digest"]) == 64
    assert result["manifest_path"].endswith(f"/{result['artifact_id']}/manifest.json")
    assert hub_calls == []
    assert not any(
        "generations" in name or name.endswith("ready.json") for name in minio.uploads
    )


def test_seed_falls_back_to_hub_then_must_publish(tmp_path):
    request = _seed_request(tmp_path)
    minio = InMemoryMinio()
    calls = []

    def download(identity, staging, **kwargs):
        calls.append((identity.revision, kwargs))
        _write_model(staging)

    result = execute_seed_preheat(
        request,
        ModelPreheatS3Client(minio),
        download_to_staging=download,
    )

    assert calls and calls[0][0] == "resolved-commit"
    assert result["transfer_source"] == "modelscope"
    assert ("models", result["manifest_path"]) in minio.objects


def test_seed_hub_fallback_receives_token_and_exclude_patterns(tmp_path):
    identity = _identity(patterns=["**"])
    request = _seed_request(
        tmp_path,
        identity=identity,
        request_digest=identity.request_digest,
        exclude_patterns=("private/**",),
    )
    calls = []

    def download(identity, staging, **kwargs):
        calls.append(kwargs)
        _write_model(staging)

    result = execute_seed_preheat(
        request,
        ModelPreheatS3Client(InMemoryMinio()),
        download_to_staging=download,
        source_token="hf-secret-token",
    )

    assert result["state"] == "ready"
    assert calls == [
        {
            "token": "hf-secret-token",
            "exclude_patterns": ("private/**",),
            "cancel_check": None,
            "progress_callback": None,
        }
    ]
    assert "hf-secret-token" not in json.dumps(result)


def test_modelscope_filelist_revision_downloads_requested_upstream_revision(tmp_path):
    files = [
        type(
            "File",
            (),
            {"path": "model.bin", "size": 10, "blob_id": "b" * 64},
        )()
    ]
    from gpustack.server.model_preheat_revision import modelscope_filelist_revision

    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision=modelscope_filelist_revision(files),
        requested_revision="release",
        file_patterns=(),
    )
    with (
        patch(
            "gpustack.worker.downloaders.modelscope_snapshot_download"
        ) as snapshot_download,
        patch("gpustack.worker.downloaders.ModelScopeHubApi") as hub_api,
    ):
        hub_api.return_value.list_repo_files.return_value = files
        download_resolved_revision_to_staging(identity, tmp_path / "staging")

    assert snapshot_download.call_args.kwargs["revision"] == "release"


def test_modelscope_preheat_progress_download_skips_missing_git_metadata(tmp_path):
    from gpustack.server.model_preheat_revision import modelscope_filelist_revision

    files = [
        type(
            "File",
            (),
            {"path": ".gitattributes", "size": 1, "blob_id": "a" * 64},
        )(),
        type(
            "File",
            (),
            {"path": "nested/.gitignore", "size": 1, "blob_id": "c" * 64},
        )(),
        type(
            "File",
            (),
            {"path": "model.bin", "size": 10, "blob_id": "b" * 64},
        )(),
    ]
    patterns = (".gitattributes", "nested/.gitignore", "model.bin")
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="Qwen/Qwen3-Reranker-0.6B",
        revision=modelscope_filelist_revision(files, include_patterns=patterns),
        requested_revision="master",
        file_patterns=patterns,
    )
    attempted = []
    progress = []

    def download_file(*, file_path, local_dir, **kwargs):
        del kwargs
        attempted.append(file_path)
        if file_path in {".gitattributes", "nested/.gitignore"}:
            raise ModelScopeNotExistError(f"{file_path} not exist")
        path = Path(local_dir) / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
        return str(path)

    with (
        patch("gpustack.worker.downloaders.ModelScopeHubApi") as hub_api,
        patch(
            "gpustack.worker.downloaders.model_file_download",
            side_effect=download_file,
        ),
    ):
        hub_api.return_value.list_repo_files.return_value = files
        download_resolved_revision_to_staging(
            identity,
            tmp_path / "staging",
            progress_callback=lambda completed, downloaded_size, total_size: progress.append(
                (completed, downloaded_size, total_size)
            ),
        )

    assert attempted == [".gitattributes", "model.bin", "nested/.gitignore"]
    assert not (tmp_path / "staging" / ".gitattributes").exists()
    assert not (tmp_path / "staging" / "nested" / ".gitignore").exists()
    assert (tmp_path / "staging" / "model.bin").is_file()
    assert progress == [(("model.bin",), 10, 12)]


def test_modelscope_preheat_progress_download_ignores_default_git_metadata(tmp_path):
    from gpustack.server.model_preheat_revision import modelscope_filelist_revision

    files = [
        type(
            "File",
            (),
            {"path": ".gitattributes", "size": 1, "blob_id": "a" * 64},
        )(),
        type(
            "File",
            (),
            {"path": "nested/.gitignore", "size": 1, "blob_id": "c" * 64},
        )(),
        type(
            "File",
            (),
            {"path": "model.bin", "size": 10, "blob_id": "b" * 64},
        )(),
    ]
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="Qwen/Qwen3-Reranker-0.6B",
        revision=modelscope_filelist_revision(files),
        requested_revision="master",
        file_patterns=(),
    )
    attempted = []
    progress = []

    def download_file(*, file_path, local_dir, **kwargs):
        del kwargs
        attempted.append(file_path)
        path = Path(local_dir) / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
        return str(path)

    with (
        patch("gpustack.worker.downloaders.ModelScopeHubApi") as hub_api,
        patch(
            "gpustack.worker.downloaders.model_file_download",
            side_effect=download_file,
        ),
    ):
        hub_api.return_value.list_repo_files.return_value = files
        download_resolved_revision_to_staging(
            identity,
            tmp_path / "staging",
            progress_callback=lambda completed, downloaded_size, total_size: progress.append(
                (completed, downloaded_size, total_size)
            ),
        )

    assert attempted == ["model.bin"]
    assert not (tmp_path / "staging" / ".gitattributes").exists()
    assert not (tmp_path / "staging" / "nested" / ".gitignore").exists()
    assert (tmp_path / "staging" / "model.bin").is_file()
    assert progress == [(("model.bin",), 10, 10)]


def test_modelscope_preheat_progress_download_keeps_existing_git_metadata(tmp_path):
    files = [
        type(
            "File",
            (),
            {"path": ".gitattributes", "size": 1, "blob_id": "a" * 64},
        )()
    ]
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="Qwen/Qwen3-Reranker-0.6B",
        revision="master",
        requested_revision="master",
        file_patterns=(".gitattributes",),
    )
    downloaded = []

    def download_file(*, file_path, local_dir, **kwargs):
        del kwargs
        downloaded.append(file_path)
        path = Path(local_dir) / file_path
        path.write_bytes(b"metadata")
        return str(path)

    with (
        patch("gpustack.worker.downloaders.ModelScopeHubApi") as hub_api,
        patch(
            "gpustack.worker.downloaders.model_file_download",
            side_effect=download_file,
        ),
    ):
        hub_api.return_value.list_repo_files.return_value = files
        download_resolved_revision_to_staging(
            identity,
            tmp_path / "staging",
            progress_callback=lambda *args: None,
        )

    assert downloaded == [".gitattributes"]
    assert (tmp_path / "staging" / ".gitattributes").read_bytes() == b"metadata"


def test_modelscope_preheat_progress_download_fails_for_missing_model_file(tmp_path):
    files = [
        type(
            "File",
            (),
            {"path": "model.bin", "size": 10, "blob_id": "b" * 64},
        )()
    ]
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="Qwen/Qwen3-Reranker-0.6B",
        revision="master",
        requested_revision="master",
        file_patterns=(),
    )

    with (
        patch("gpustack.worker.downloaders.ModelScopeHubApi") as hub_api,
        patch(
            "gpustack.worker.downloaders.model_file_download",
            side_effect=ModelScopeNotExistError("model.bin not exist"),
        ),
        pytest.raises(ModelScopeNotExistError, match="model.bin not exist"),
    ):
        hub_api.return_value.list_repo_files.return_value = files
        download_resolved_revision_to_staging(
            identity,
            tmp_path / "staging",
            progress_callback=lambda *args: None,
        )


def test_modelscope_filelist_revision_change_rejects_download(tmp_path):
    from gpustack.server.model_preheat_revision import modelscope_filelist_revision

    before = [
        type(
            "File",
            (),
            {"path": "model.bin", "size": 10, "blob_id": "a" * 64},
        )()
    ]
    after = [
        type(
            "File",
            (),
            {"path": "model.bin", "size": 10, "blob_id": "b" * 64},
        )()
    ]
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision=modelscope_filelist_revision(before),
        requested_revision="release",
        file_patterns=(),
    )
    with (
        patch("gpustack.worker.downloaders.modelscope_snapshot_download"),
        patch("gpustack.worker.downloaders.ModelScopeHubApi") as hub_api,
        pytest.raises(ValueError, match="modelscope_revision_changed"),
    ):
        hub_api.return_value.list_repo_files.side_effect = [before, after]
        download_resolved_revision_to_staging(identity, tmp_path / "staging")


def test_download_preserves_raw_special_character_exclude_patterns(tmp_path):
    identity = ModelPreheatIdentity(
        source="huggingface",
        model_id="org/model",
        revision="a" * 40,
        file_patterns=("**",),
    )
    exclude_patterns = ("private/%20/*.bin", "中文/模型 文件.bin")
    with patch("gpustack.worker.downloaders.snapshot_download") as snapshot_download:
        download_resolved_revision_to_staging(
            identity,
            tmp_path / "staging",
            exclude_patterns=exclude_patterns,
        )

    assert snapshot_download.call_args.kwargs["ignore_patterns"] == list(
        exclude_patterns
    )


def test_seed_disabled_fallback_never_calls_hub(tmp_path):
    request = _seed_request(tmp_path, source_fallback_enabled=False)
    calls = []

    result = execute_seed_preheat(
        request,
        ModelPreheatS3Client(InMemoryMinio()),
        download_to_staging=lambda *args, **kwargs: calls.append(True),
    )

    assert result == {"state": "error", "error_code": "model_artifact_not_found"}
    assert calls == []


def test_bound_artifact_seed_reuses_trusted_local_without_s3_file_download(tmp_path):
    source = tmp_path / "source"
    _write_model(source)
    manifest = build_model_preheat_manifest(source, _identity())
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    request = _seed_request(
        tmp_path,
        artifact_id=manifest.artifact_id,
        manifest_path=client.artifact_manifest_object("model-storage", manifest),
        trusted_local_candidate=TrustedLocalCandidate(
            source="model_file",
            root=source,
            paths=(source,),
            repository_complete=True,
        ),
    )

    result = execute_seed_preheat(request, client)

    assert result["state"] == "ready"
    assert result["transfer_source"] == "current_node"
    assert minio.downloads == []


def test_target_downloads_only_missing_or_invalid_files(tmp_path):
    source = tmp_path / "source"
    _write_model(source)
    manifest = build_model_preheat_manifest(source, _identity())
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    client.publish_artifact("models", "model-storage", manifest, source)
    request = _target_request(tmp_path, manifest, client)
    request.target_dir.mkdir(parents=True)
    (request.target_dir / "config.json").write_bytes(b"config")
    (request.target_dir / "weights").mkdir()
    (request.target_dir / "weights" / "model.bin").write_bytes(b"stale")
    minio.downloads.clear()

    result = execute_target_preheat(request, client)

    file_downloads = [name for name in minio.downloads if "/files/" in name]
    assert result["state"] == "ready"
    assert result["transfer_source"] == "s3"
    assert result["downloaded"] == 1
    assert len(file_downloads) == 1
    assert file_downloads[0].endswith("/files/weights/model.bin")


def test_target_rejects_manifest_for_different_file_selection(tmp_path):
    source = tmp_path / "source"
    _write_model(source)
    full = build_model_preheat_manifest(source, _identity(patterns=["**"]))
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    client.publish_artifact("models", "model-storage", full, source)
    request = _target_request(tmp_path, full, client)

    result = execute_target_preheat(request, client)

    assert result == {"state": "error", "error_code": "s3_manifest_invalid"}


def test_target_execution_failure_returns_safe_error_details(tmp_path):
    source = tmp_path / "source"
    _write_model(source)
    manifest = build_model_preheat_manifest(source, _identity())
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    client.publish_artifact("models", "model-storage", manifest, source)

    def fail_download(*args, **kwargs):
        del args, kwargs
        raise OSError(
            "failed token=secret-value at https://access:secret@s3.example.com/object"
        )

    client.download_artifact_file = fail_download
    result = execute_target_preheat(_target_request(tmp_path, manifest, client), client)

    assert result["state"] == "error"
    assert result["error_code"] == "worker_execution_failed"
    assert result["error_type"] == "OSError"
    assert "secret-value" not in result["error_message"]
    assert "access:secret" not in result["error_message"]
    assert "token=" not in result["error_message"]
    assert "[redacted]" in result["error_message"]


def test_requested_revision_does_not_change_artifact_manifest(tmp_path):
    root = tmp_path / "source"
    _write_model(root)
    moving = build_model_preheat_manifest(root, _identity(requested_revision="master"))
    immutable = build_model_preheat_manifest(
        root, _identity(requested_revision="resolved-commit")
    )

    assert moving.artifact_id == immutable.artifact_id
    assert moving.to_artifact_json_bytes() == immutable.to_artifact_json_bytes()
    assert "requested_revision" not in json.loads(moving.to_artifact_json_bytes())


def test_two_publishers_converge_to_one_manifest(tmp_path):
    root = tmp_path / "source"
    _write_model(root)
    manifest = build_model_preheat_manifest(root, _identity())
    minio = InMemoryMinio()
    client = ModelPreheatS3Client(minio)
    results = []

    def publish():
        results.append(
            client.publish_artifact("models", "model-storage", manifest, root)
        )

    threads = [threading.Thread(target=publish) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    manifest_names = [name for name in minio.uploads if name.endswith("/manifest.json")]
    assert len(results) == 2
    assert len(manifest_names) == 1
    assert hashlib.sha256(minio.objects[("models", manifest_names[0])].data).hexdigest()
