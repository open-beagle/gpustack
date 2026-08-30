import asyncio
import json
from types import SimpleNamespace

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.auth import get_admin_user
from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
    generate_model_preheat_credential_key,
)
from gpustack.routes import model_preheat_worker_tasks
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatConnectivityCheckStateEnum,
    ModelPreheatS3ConnectivityCheck,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
)
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_session
from gpustack.server.model_preheat_worker_identity import (
    get_model_preheat_worker_identity,
)
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
from gpustack.worker.model_preheat.executor import (
    SeedExecutionRequest,
    execute_seed_preheat,
)


API_PREFIX = "/v1/model-preheat-worker-tasks"


def _test_app(tmp_path):
    key = generate_model_preheat_credential_key()
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'worker-tasks.db'}")

    async def create_tables():
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)

    asyncio.run(create_tables())
    app = FastAPI()
    app.state.server_config = SimpleNamespace(
        model_preheat_credential_key=key,
        model_preheat_credential_key_version="v1",
        model_preheat_credential_old_keys=None,
    )

    async def session_override():
        async with AsyncSession(engine, expire_on_commit=True) as session:
            yield session

    async def admin_override():
        return User(id=1, username="admin", is_admin=True, hashed_password="")

    async def identity_override():
        return SimpleNamespace(worker_id=1, worker_uuid="worker-uuid")

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_admin_user] = admin_override
    app.dependency_overrides[get_model_preheat_worker_identity] = identity_override
    router = APIRouter(dependencies=[Depends(get_admin_user)])
    router.include_router(model_preheat_worker_tasks.router, prefix=API_PREFIX)
    app.include_router(router)
    exceptions.register_handlers(app)
    return app, engine, key


async def _seed(
    engine,
    key,
    *,
    artifact_id=None,
    source="modelscope",
    model_id="Qwen/Test",
    resolved_revision="a" * 40,
    role=ModelPreheatWorkerTaskRoleEnum.SEED,
):
    cipher = ModelPreheatCredentialCipher(key, "v1")
    identity = ModelPreheatIdentity(
        source=source,
        model_id=model_id,
        revision=resolved_revision,
        requested_revision="master",
        file_patterns=(),
    )
    profile_payload = {
        "endpoint": "https://s3.example.com",
        "bucket": "models",
        "prefix": "model-storage",
        "tls_enabled": True,
        "tls_verify": True,
        "region": "",
        "use_virtual_hosted_style": False,
        "source_fallback_enabled": True,
        "access_key_encrypted": cipher.encrypt("access-plain"),
        "secret_key_encrypted": cipher.encrypt("secret-plain"),
    }
    async with AsyncSession(engine) as session:
        worker = Worker(
            id=1,
            name="worker-a",
            hostname="worker-a",
            ip="127.0.0.1",
            port=10150,
            worker_uuid="worker-uuid",
            state=WorkerStateEnum.READY,
            model_storage_protocol_version=1,
        )
        profile = ModelPreheatS3Profile(
            name="storage",
            endpoint=profile_payload["endpoint"],
            bucket=profile_payload["bucket"],
            prefix=profile_payload["prefix"],
            access_key_encrypted=profile_payload["access_key_encrypted"],
            secret_key_encrypted=profile_payload["secret_key_encrypted"],
            encryption_key_version="v1",
            config_version=3,
        )
        session.add_all([worker, profile])
        await session.flush()
        task = ModelPreheatTask(
            source=source,
            model_id=model_id,
            requested_revision="master",
            resolved_revision=resolved_revision,
            include_patterns=[],
            exclude_patterns=[],
            selection_digest="b" * 64,
            request_identity={
                "source": identity.source,
                "model_id": identity.model_path,
                "requested_revision": identity.requested_revision_path,
                "include_patterns": [],
                "exclude_patterns": [],
            },
            request_digest=identity.request_digest,
            artifact_id=artifact_id,
            seed_worker_uuid=worker.worker_uuid,
            seed_worker_id=worker.id,
            target_scope=ModelPreheatTargetScopeEnum.SEED_WORKER,
            target_worker_uuids=[worker.worker_uuid],
            target_worker_snapshot=[],
            s3_profile_id=profile.id,
            s3_profile_config_version=profile.config_version,
            s3_profile_snapshot_encrypted=cipher.encrypt(json.dumps(profile_payload)),
            encryption_key_version="v1",
            s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
        )
        session.add(task)
        await session.flush()
        child = ModelPreheatWorkerTask(
            task_id=task.id,
            worker_uuid=worker.worker_uuid,
            worker_id=worker.id,
            role=role,
        )
        session.add(child)
        await session.flush()
        child_id = child.id
        task_id = task.id
        await session.commit()
        return child_id, task_id, identity.request_digest


def _claim(client, child_id):
    response = client.post(
        f"{API_PREFIX}/{child_id}/claim",
        json={"worker_uuid": "worker-uuid", "worker_id": 1},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _ready_result(request_digest, artifact_id="c" * 64):
    return {
        "state": "ready",
        "request_digest": request_digest,
        "artifact_id": artifact_id,
        "manifest_digest": "d" * 64,
        "manifest_path": f"model-storage/modelscope/Qwen/Test/{artifact_id}/manifest.json",
        "file_count": 2,
        "total_size": 10,
        "local_cache_state": "valid",
        "transfer_source": "modelscope",
        "uploaded": 2,
        "skipped": 0,
        "downloaded": 0,
        "resolved_revision": "a" * 40,
    }


async def _seed_connectivity_task(engine, key):
    cipher = ModelPreheatCredentialCipher(key, "v1")
    async with AsyncSession(engine) as session:
        worker = Worker(
            id=1,
            name="worker-a",
            hostname="worker-a",
            ip="127.0.0.1",
            port=10150,
            worker_uuid="worker-uuid",
            state=WorkerStateEnum.READY,
            model_storage_protocol_version=1,
        )
        profile = ModelPreheatS3Profile(
            name="storage",
            endpoint="https://s3.example.com",
            bucket="models",
            access_key_encrypted=cipher.encrypt("access-plain"),
            secret_key_encrypted=cipher.encrypt("secret-plain"),
            encryption_key_version="v1",
        )
        session.add_all([worker, profile])
        await session.flush()
        check = ModelPreheatS3ConnectivityCheck(
            profile_id=profile.id,
            profile_config_version=profile.config_version,
            scope_key="connectivity-scope",
            active_key="connectivity-active-scope",
            state=ModelPreheatConnectivityCheckStateEnum.RUNNING,
            target_worker_uuids=[worker.worker_uuid],
        )
        session.add(check)
        await session.flush()
        worker_task = ModelPreheatWorkerTask(
            connectivity_check_id=check.id,
            worker_uuid=worker.worker_uuid,
            worker_id=worker.id,
            role=ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
        )
        session.add(worker_task)
        await session.flush()
        worker_task_id = worker_task.id
        await session.commit()
        return worker_task_id


def _connectivity_ready_result():
    return {
        "state": "ready",
        "readable": True,
        "writable": True,
        "deletable": True,
        "cleanup_failed": False,
        "latency_ms": 1,
    }


def _connectivity_error_result():
    return {
        "state": "error",
        "error_code": "network_timeout",
        "failed_stage": "tcp",
    }


def test_claim_and_payload_keep_credentials_private(tmp_path):
    app, engine, key = _test_app(tmp_path)
    child_id, task_id, _ = asyncio.run(_seed(engine, key))

    async def maintain_after_task_was_frozen():
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            profile = await session.get(ModelPreheatS3Profile, parent.s3_profile_id)
            profile.lifecycle_state = "maintenance"
            session.add(profile)
            await session.commit()

    asyncio.run(maintain_after_task_was_frozen())
    with TestClient(app) as client:
        claim = _claim(client, child_id)
        payload = client.get(
            f"{API_PREFIX}/{child_id}/execution-payload",
            headers={
                "X-Worker-UUID": "worker-uuid",
                "X-Worker-ID": "1",
                "X-Task-Attempt": str(claim["attempt"]),
                "X-Lease-Token": claim["lease_token"],
            },
        )
        public = client.get(f"{API_PREFIX}/{child_id}")

    assert payload.status_code == 200
    assert payload.headers["cache-control"] == "no-store"
    assert payload.json()["profile"]["access_key"] == "access-plain"
    assert "access-plain" not in public.text
    assert claim["lease_token"] not in public.text

    async def used_at():
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            profile = await session.get(ModelPreheatS3Profile, parent.s3_profile_id)
            return profile.ever_used_at

    assert asyncio.run(used_at()) is not None
    asyncio.run(engine.dispose())


def test_connectivity_payload_normalizes_nullable_profile_region(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_task_id = asyncio.run(_seed_connectivity_task(engine, key))

    async def clear_region():
        async with AsyncSession(engine) as session:
            profile = (await session.exec(select(ModelPreheatS3Profile))).one()
            profile.region = None
            session.add(profile)
            await session.commit()

    asyncio.run(clear_region())
    with TestClient(app) as client:
        claim = _claim(client, worker_task_id)
        payload = client.get(
            f"{API_PREFIX}/{worker_task_id}/execution-payload",
            headers={
                "X-Worker-UUID": "worker-uuid",
                "X-Worker-ID": "1",
                "X-Task-Attempt": str(claim["attempt"]),
                "X-Lease-Token": claim["lease_token"],
            },
        )

    assert payload.status_code == 200, payload.text
    assert payload.json()["profile"]["region"] == ""
    asyncio.run(engine.dispose())


def test_claim_rejects_worker_without_storage_protocol(tmp_path):
    app, engine, key = _test_app(tmp_path)
    child_id, _, _ = asyncio.run(_seed(engine, key))

    async def downgrade_protocol():
        async with AsyncSession(engine) as session:
            worker = await session.get(Worker, 1)
            worker.model_storage_protocol_version = 0
            session.add(worker)
            await session.commit()

    asyncio.run(downgrade_protocol())
    with TestClient(app) as client:
        response = client.post(
            f"{API_PREFIX}/{child_id}/claim",
            json={"worker_uuid": "worker-uuid", "worker_id": 1},
        )

    assert response.status_code == 409
    assert response.json()["message"] == "model_storage_protocol_mismatch"
    asyncio.run(engine.dispose())


def test_seed_complete_cas_binds_artifact_and_inventory(tmp_path):
    app, engine, key = _test_app(tmp_path)
    child_id, task_id, request_digest = asyncio.run(_seed(engine, key))
    artifact_id = "c" * 64
    with TestClient(app) as client:
        claim = _claim(client, child_id)
        completed = client.post(
            f"{API_PREFIX}/{child_id}/complete",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": 1,
                "attempt": claim["attempt"],
                "lease_token": claim["lease_token"],
                "result": _ready_result(request_digest, artifact_id),
            },
        )

    async def inspect():
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            artifacts = (await session.exec(select(ModelPreheatArtifact))).all()
            return parent, artifacts

    parent, artifacts = asyncio.run(inspect())
    assert completed.status_code == 200, completed.text
    assert parent.artifact_id == artifact_id
    assert parent.transfer_source == "modelscope"
    assert len(artifacts) == 1
    assert artifacts[0].profile_config_version == 3
    asyncio.run(engine.dispose())


def test_distribution_complete_registers_ready_model_file(tmp_path):
    app, engine, key = _test_app(tmp_path)
    artifact_id = "c" * 64
    child_id, _task_id, request_digest = asyncio.run(
        _seed(
            engine,
            key,
            artifact_id=artifact_id,
            model_id="Qwen/Qwen-7B-Chat-Int8",
            role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
        )
    )
    result = {
        **_ready_result(request_digest, artifact_id),
        "transfer_source": "s3",
        "local_dir": "/var/lib/gpustack/cache/model_scope/Qwen/Qwen-7B-Chat-Int8",
        "resolved_paths": [
            "/var/lib/gpustack/cache/model_scope/Qwen/Qwen-7B-Chat-Int8"
        ],
    }
    with TestClient(app) as client:
        claim = _claim(client, child_id)
        completed = client.post(
            f"{API_PREFIX}/{child_id}/complete",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": 1,
                "attempt": claim["attempt"],
                "lease_token": claim["lease_token"],
                "result": result,
            },
        )

    async def inspect():
        async with AsyncSession(engine) as session:
            rows = (await session.exec(select(ModelFile))).all()
            return rows

    model_files = asyncio.run(inspect())
    assert completed.status_code == 200, completed.text
    assert len(model_files) == 1
    model_file = model_files[0]
    assert model_file.state == ModelFileStateEnum.READY
    assert model_file.worker_id == 1
    assert model_file.model_scope_model_id == "Qwen/Qwen-7B-Chat-Int8"
    assert model_file.resolved_revision == "a" * 40
    assert model_file.resolved_paths == [
        "/var/lib/gpustack/cache/model_scope/Qwen/Qwen-7B-Chat-Int8"
    ]
    asyncio.run(engine.dispose())


@pytest.mark.parametrize(
    "result_patch",
    [
        {"resolved_paths": None},
        {
            "local_dir": "/var/lib/gpustack/cache/model_scope/Qwen/Qwen-7B-Chat-Int8",
            "resolved_paths": ["/tmp/outside/model.bin"],
        },
        {
            "local_dir": "/var/lib/gpustack/cache/model_scope/Qwen/Qwen-7B-Chat-Int8",
            "resolved_paths": [
                "/var/lib/gpustack/cache/model_scope/Qwen/Qwen-7B-Chat-Int8/../outside/model.bin"
            ],
        },
        {
            "local_dir": "relative/cache",
            "resolved_paths": ["relative/cache/model.bin"],
        },
    ],
)
def test_distribution_complete_rejects_untrusted_model_paths(tmp_path, result_patch):
    app, engine, key = _test_app(tmp_path)
    artifact_id = "c" * 64
    child_id, _task_id, request_digest = asyncio.run(
        _seed(
            engine,
            key,
            artifact_id=artifact_id,
            model_id="Qwen/Qwen-7B-Chat-Int8",
            role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
        )
    )
    result = {
        **_ready_result(request_digest, artifact_id),
        "transfer_source": "s3",
        "local_dir": "/var/lib/gpustack/cache/model_scope/Qwen/Qwen-7B-Chat-Int8",
        "resolved_paths": [
            "/var/lib/gpustack/cache/model_scope/Qwen/Qwen-7B-Chat-Int8"
        ],
        **result_patch,
    }
    if result["resolved_paths"] is None:
        result.pop("resolved_paths")
    with TestClient(app) as client:
        claim = _claim(client, child_id)
        completed = client.post(
            f"{API_PREFIX}/{child_id}/complete",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": 1,
                "attempt": claim["attempt"],
                "lease_token": claim["lease_token"],
                "result": result,
            },
        )

    async def count_model_files():
        async with AsyncSession(engine) as session:
            return len((await session.exec(select(ModelFile))).all())

    assert completed.status_code == 422, completed.text
    assert completed.json()["message"] == "invalid_preheat_result"
    assert asyncio.run(count_model_files()) == 0
    asyncio.run(engine.dispose())


def test_distribution_complete_accepts_windows_absolute_model_paths(tmp_path):
    app, engine, key = _test_app(tmp_path)
    artifact_id = "c" * 64
    child_id, _task_id, request_digest = asyncio.run(
        _seed(
            engine,
            key,
            artifact_id=artifact_id,
            model_id="Qwen/Qwen-7B-Chat-Int8",
            role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
        )
    )
    result = {
        **_ready_result(request_digest, artifact_id),
        "transfer_source": "s3",
        "local_dir": "C:\\gpustack\\cache\\model_scope\\Qwen\\Qwen-7B-Chat-Int8",
        "resolved_paths": [
            "C:\\gpustack\\cache\\model_scope\\Qwen\\Qwen-7B-Chat-Int8\\model.safetensors"
        ],
    }
    with TestClient(app) as client:
        claim = _claim(client, child_id)
        completed = client.post(
            f"{API_PREFIX}/{child_id}/complete",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": 1,
                "attempt": claim["attempt"],
                "lease_token": claim["lease_token"],
                "result": result,
            },
        )

    async def inspect():
        async with AsyncSession(engine) as session:
            rows = (await session.exec(select(ModelFile))).all()
            return rows

    model_files = asyncio.run(inspect())
    assert completed.status_code == 200, completed.text
    assert len(model_files) == 1
    assert model_files[0].local_dir == result["local_dir"]
    assert model_files[0].resolved_paths == result["resolved_paths"]
    asyncio.run(engine.dispose())


@pytest.mark.parametrize(
    ("source", "model_id", "revision", "filename"),
    [
        ("huggingface", "org/model", "b" * 40, "config.json"),
        ("modelscope", "Qwen/Test", "a" * 40, "config.json"),
        ("ollama_library", "llama3:latest", "sha256:" + "c" * 64, "llama3_latest"),
    ],
)
def test_seed_complete_accepts_real_executor_ready_result(
    tmp_path, source, model_id, revision, filename
):
    app, engine, key = _test_app(tmp_path)
    child_id, task_id, request_digest = asyncio.run(
        _seed(
            engine,
            key,
            source=source,
            model_id=model_id,
            resolved_revision=revision,
        )
    )
    identity = ModelPreheatIdentity(
        source=source,
        model_id=model_id,
        revision=revision,
        requested_revision="master",
        file_patterns=(),
    )

    class FakeS3:
        def publish_artifact(self, bucket, prefix, manifest, staging, **kwargs):
            del bucket, prefix, staging, kwargs
            self.manifest = manifest
            return SimpleNamespace(uploaded=len(manifest.files) + 1, skipped=0)

        def artifact_manifest_object(self, prefix, manifest):
            return f"{prefix}/{manifest.artifact_id}/manifest.json"

    request = SeedExecutionRequest(
        cache_dir=tmp_path / "cache",
        target_dir=tmp_path / "target",
        task_id=task_id,
        attempt=1,
        request_digest=request_digest,
        identity=identity,
        exclude_patterns=(),
        bucket="models",
        prefix="model-storage",
        source_fallback_enabled=True,
    )
    result = execute_seed_preheat(
        request,
        FakeS3(),
        download_to_staging=lambda _identity, staging, **kwargs: (
            staging / filename
        ).write_bytes(b"config"),
    )
    assert result["state"] == "ready"
    assert result["resolved_revision"] == revision

    with TestClient(app) as client:
        claim = _claim(client, child_id)
        completed = client.post(
            f"{API_PREFIX}/{child_id}/complete",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": 1,
                "attempt": claim["attempt"],
                "lease_token": claim["lease_token"],
                "result": result,
            },
        )

    async def inspect():
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            artifact = (await session.exec(select(ModelPreheatArtifact))).one()
            return parent, artifact

    parent, artifact = asyncio.run(inspect())
    assert completed.status_code == 200, completed.text
    assert parent.resolved_revision == revision
    assert artifact.resolved_revision == revision
    asyncio.run(engine.dispose())


def test_pending_ollama_complete_binds_only_actual_local_snapshot(tmp_path):
    app, engine, key = _test_app(tmp_path)
    child_id, task_id, request_digest = asyncio.run(
        _seed(
            engine,
            key,
            source="ollama_library",
            model_id="llama3:latest",
            resolved_revision="ollama-pending",
        )
    )
    identity = ModelPreheatIdentity(
        source="ollama_library",
        model_id="llama3:latest",
        revision="ollama-pending",
        requested_revision="master",
        file_patterns=(),
    )

    class FakeS3:
        def publish_artifact(self, bucket, prefix, manifest, staging, **kwargs):
            del bucket, prefix, staging, kwargs
            return SimpleNamespace(uploaded=len(manifest.files) + 1, skipped=0)

        def artifact_manifest_object(self, prefix, manifest):
            return f"{prefix}/{manifest.artifact_id}/manifest.json"

    request = SeedExecutionRequest(
        cache_dir=tmp_path / "cache",
        target_dir=tmp_path / "target",
        task_id=task_id,
        attempt=1,
        request_digest=request_digest,
        identity=identity,
        exclude_patterns=(),
        bucket="models",
        prefix="model-storage",
        source_fallback_enabled=True,
    )
    result = execute_seed_preheat(
        request,
        FakeS3(),
        download_to_staging=lambda _identity, staging, **kwargs: (
            staging / "llama3_latest"
        ).write_bytes(b"ollama"),
    )
    assert result["resolved_revision"].startswith("local-snapshot-")

    with TestClient(app) as client:
        claim = _claim(client, child_id)
        completed = client.post(
            f"{API_PREFIX}/{child_id}/complete",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": 1,
                "attempt": claim["attempt"],
                "lease_token": claim["lease_token"],
                "result": result,
            },
        )

    async def inspect():
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            artifact = (await session.exec(select(ModelPreheatArtifact))).one()
            return parent, artifact

    parent, artifact = asyncio.run(inspect())
    assert completed.status_code == 200, completed.text
    assert parent.resolved_revision == result["resolved_revision"]
    assert artifact.resolved_revision == result["resolved_revision"]
    asyncio.run(engine.dispose())


@pytest.mark.parametrize(
    "resolved_revision",
    [
        "local-snapshot-x",
        "local-snapshot-" + "A" * 64,
        "local-snapshot-" + "a" * 63,
        1,
        [],
        None,
    ],
)
def test_pending_ollama_complete_rejects_malformed_snapshot_without_binding(
    tmp_path, resolved_revision
):
    app, engine, key = _test_app(tmp_path)
    child_id, task_id, request_digest = asyncio.run(
        _seed(
            engine,
            key,
            source="ollama_library",
            model_id="llama3:latest",
            resolved_revision="ollama-pending",
        )
    )
    result = _ready_result(request_digest)
    result["transfer_source"] = "ollama_library"
    result["resolved_revision"] = resolved_revision
    with TestClient(app) as client:
        claim = _claim(client, child_id)
        completed = client.post(
            f"{API_PREFIX}/{child_id}/complete",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": 1,
                "attempt": claim["attempt"],
                "lease_token": claim["lease_token"],
                "result": result,
            },
        )

    async def inspect():
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            artifacts = (await session.exec(select(ModelPreheatArtifact))).all()
            return parent.artifact_id, artifacts

    artifact_id, artifacts = asyncio.run(inspect())
    assert completed.status_code == 422, completed.text
    assert completed.json()["message"] == "invalid_preheat_result"
    assert artifact_id is None
    assert artifacts == []
    asyncio.run(engine.dispose())


def test_seed_complete_rejects_wrong_request_digest(tmp_path):
    app, engine, key = _test_app(tmp_path)
    child_id, task_id, request_digest = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claim = _claim(client, child_id)
        result = _ready_result(request_digest)
        result["request_digest"] = "f" * 64
        completed = client.post(
            f"{API_PREFIX}/{child_id}/complete",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": 1,
                "attempt": claim["attempt"],
                "lease_token": claim["lease_token"],
                "result": result,
            },
        )

    async def artifact_id():
        async with AsyncSession(engine) as session:
            return (await session.get(ModelPreheatTask, task_id)).artifact_id

    assert completed.status_code == 409
    assert asyncio.run(artifact_id()) is None
    asyncio.run(engine.dispose())


@pytest.mark.parametrize("operation", ["complete", "fail"])
def test_connectivity_terminal_updates_remain_serializable_after_aggregation(
    tmp_path, operation
):
    app, engine, key = _test_app(tmp_path)
    worker_task_id = asyncio.run(_seed_connectivity_task(engine, key))
    with TestClient(app) as client:
        claim = _claim(client, worker_task_id)
        payload = {
            "worker_uuid": "worker-uuid",
            "worker_id": 1,
            "attempt": claim["attempt"],
            "lease_token": claim["lease_token"],
        }
        if operation == "complete":
            payload["result"] = _connectivity_ready_result()
        else:
            payload["error_code"] = "network_timeout"
            payload["result"] = _connectivity_error_result()
        first = client.post(f"{API_PREFIX}/{worker_task_id}/{operation}", json=payload)
        second = client.post(f"{API_PREFIX}/{worker_task_id}/{operation}", json=payload)

    expected_state = "ready" if operation == "complete" else "error"
    for response in (first, second):
        assert response.status_code == 200, response.text
        assert response.json()["state"] == expected_state
        assert "access-plain" not in response.text

    async def connectivity_did_not_use_storage():
        async with AsyncSession(engine) as session:
            profile = (await session.exec(select(ModelPreheatS3Profile))).one()
            return profile.ever_used_at

    assert asyncio.run(connectivity_did_not_use_storage()) is None
    asyncio.run(engine.dispose())
