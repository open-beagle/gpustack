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


async def _seed(engine, key, *, artifact_id=None):
    cipher = ModelPreheatCredentialCipher(key, "v1")
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="Qwen/Test",
        revision="a" * 40,
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
            source="modelscope",
            model_id="Qwen/Test",
            requested_revision="master",
            resolved_revision="a" * 40,
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
            role=ModelPreheatWorkerTaskRoleEnum.SEED,
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
