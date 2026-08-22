import hashlib
import json
import threading
from dataclasses import dataclass, field, replace

from gpustack.worker.model_preheat.executor import (
    SeedExecutionRequest,
    TargetExecutionRequest,
    TrustedLocalCandidate,
    execute_seed_preheat,
    execute_target_preheat,
)
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
from gpustack.worker.model_preheat.manifest import build_model_preheat_manifest
from gpustack.worker.model_preheat.s3_client import ModelPreheatS3Client


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
    client.publish_artifact("models", "model-storage", manifest, source)
    minio.downloads.clear()
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
    assert not any("/files/" in name for name in minio.downloads)


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
