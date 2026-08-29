import asyncio
from dataclasses import replace
from types import SimpleNamespace

from gpustack.schemas.model_preheats import ModelPreheatWorkerTaskRoleEnum
from gpustack.worker.model_preheat import executor as preheat_executor
from gpustack.worker.model_preheat.executor import (
    SeedExecutionRequest,
    TargetExecutionRequest,
    build_preheat_role_handlers,
    execute_seed_preheat,
    execute_target_preheat,
)
from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    ollama_model_filename,
)
from gpustack.worker.model_preheat.s3_client import ModelPreheatS3Client
from tests.worker.model_preheat.test_seed_executor import InMemoryMinio, _write_model


def _identity():
    return ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="resolved-commit",
        file_patterns=["config.json", "weights/model.bin"],
    )


def _seed_request(tmp_path):
    identity = _identity()
    return SeedExecutionRequest(
        cache_dir=tmp_path / "seed-cache",
        target_dir=tmp_path / "seed-cache" / "model_scope" / "org" / "model",
        task_id=8,
        attempt=1,
        request_digest=identity.request_digest,
        identity=identity,
        exclude_patterns=[],
        bucket="models",
        prefix="preheat",
        source_fallback_enabled=True,
    )


def _published(tmp_path):
    minio = InMemoryMinio()
    request = _seed_request(tmp_path)
    result = execute_seed_preheat(
        request,
        ModelPreheatS3Client(minio),
        download_to_staging=lambda _identity, staging, **_kwargs: _write_model(staging),
    )
    assert result["state"] == "ready"
    return minio, result


def _target_request(tmp_path, published):
    identity = _identity()
    return TargetExecutionRequest(
        cache_dir=tmp_path / "target-cache",
        target_dir=tmp_path / "target-cache" / "model_scope" / "org" / "model",
        task_id=9,
        attempt=1,
        request_digest=identity.request_digest,
        identity=identity,
        exclude_patterns=[],
        bucket="models",
        prefix="preheat",
        artifact_id=published["artifact_id"],
        manifest_path=published["manifest_path"],
    )


def test_distribution_handler_constructs_target_request_without_seed_only_fields(
    tmp_path, monkeypatch
):
    identity = _identity()
    captured = {}

    def execute(request, client, **kwargs):
        captured["request"] = request
        captured["client"] = client
        captured["kwargs"] = kwargs
        return {"state": "ready"}

    monkeypatch.setattr(preheat_executor, "execute_target_preheat", execute)
    payload = SimpleNamespace(
        worker_task_id=9,
        attempt=1,
        resumable_cursor=None,
        trusted_local_candidate=None,
        task={
            "id": 8,
            "source": identity.source,
            "model_id": identity.model_id,
            "resolved_revision": identity.revision,
            "requested_revision": None,
            "include_patterns": list(identity.file_patterns),
            "exclude_patterns": [],
            "request_digest": identity.request_digest,
            "artifact_id": "a" * 64,
            "s3_manifest_path": "preheat/artifact/manifest.json",
            "delivery_mode": "s3_and_workers",
        },
        profile=SimpleNamespace(
            endpoint="https://s3.example.com",
            access_key="access-key",
            secret_key="secret-key",
            tls_enabled=True,
            tls_verify=True,
            region="",
            use_virtual_hosted_style=True,
            bucket="models",
            prefix="preheat",
            source_fallback_enabled=True,
        ),
    )
    context = SimpleNamespace(progress=None)
    handler = build_preheat_role_handlers(tmp_path)[
        ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE
    ]

    result = asyncio.run(handler(payload, context))

    assert result == {"state": "ready"}
    assert isinstance(captured["request"], TargetExecutionRequest)


def test_target_downloads_artifact_and_installs_directory(tmp_path):
    minio, published = _published(tmp_path)
    request = _target_request(tmp_path, published)

    result = execute_target_preheat(request, ModelPreheatS3Client(minio))

    assert result["state"] == "ready"
    assert result["downloaded"] == 2
    assert (request.target_dir / "config.json").read_bytes() == b"config"
    assert (request.target_dir / "weights" / "model.bin").read_bytes() == b"weights"


def test_target_skips_files_with_matching_artifact_sha256(tmp_path):
    minio, published = _published(tmp_path)
    request = _target_request(tmp_path, published)
    client = ModelPreheatS3Client(minio)

    assert execute_target_preheat(request, client)["downloaded"] == 2
    repeated = execute_target_preheat(replace(request, attempt=2), client)

    assert repeated["state"] == "ready"
    assert repeated["downloaded"] == 0


def test_target_redownloads_only_checksum_mismatch(tmp_path):
    minio, published = _published(tmp_path)
    request = _target_request(tmp_path, published)
    client = ModelPreheatS3Client(minio)
    execute_target_preheat(request, client)
    (request.target_dir / "weights" / "model.bin").write_bytes(b"corrupt")

    result = execute_target_preheat(replace(request, attempt=2), client)

    assert result["state"] == "ready"
    assert result["downloaded"] == 1
    assert (request.target_dir / "weights" / "model.bin").read_bytes() == b"weights"


def test_target_rejects_fixed_artifact_mismatch_without_fallback(tmp_path):
    minio, published = _published(tmp_path)
    request = replace(_target_request(tmp_path, published), artifact_id="f" * 64)

    result = execute_target_preheat(request, ModelPreheatS3Client(minio))

    assert result == {"state": "error", "error_code": "s3_manifest_invalid"}


def test_target_installs_discovered_ollama_artifact_as_single_file(tmp_path):
    identity = ModelPreheatIdentity(
        source="ollama_library",
        model_id="llama3:latest",
        revision="sha256:immutable",
        file_patterns=(),
    )
    seed_request = replace(
        _seed_request(tmp_path),
        identity=identity,
        request_digest=identity.request_digest,
        target_dir=tmp_path / "seed-cache" / "ollama",
    )
    filename = ollama_model_filename(identity.model_path)
    minio = InMemoryMinio()
    published = execute_seed_preheat(
        seed_request,
        ModelPreheatS3Client(minio),
        download_to_staging=lambda _identity, staging, **_kwargs: (
            staging / filename
        ).write_bytes(b"ollama-model"),
    )
    request = TargetExecutionRequest(
        cache_dir=tmp_path / "target-cache",
        target_dir=tmp_path / "target-cache" / "ollama",
        task_id=9,
        attempt=1,
        request_digest=identity.request_digest,
        identity=identity,
        exclude_patterns=[],
        bucket="models",
        prefix="preheat",
        artifact_id=published["artifact_id"],
        manifest_path=published["manifest_path"],
    )

    result = execute_target_preheat(request, ModelPreheatS3Client(minio))

    assert result["state"] == "ready"
    assert (request.target_dir / filename).read_bytes() == b"ollama-model"
