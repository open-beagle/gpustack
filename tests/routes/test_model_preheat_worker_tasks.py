import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.auth import get_admin_user
from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
    generate_model_preheat_credential_key,
)
from gpustack.routes import model_preheat_worker_tasks
from gpustack.schemas.model_preheats import (
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
)
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_session


API_PREFIX = "/v1/model-preheat-worker-tasks"


def _test_app(tmp_path):
    key = generate_model_preheat_credential_key()
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'worker-tasks.db'}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False, "timeout": 30},
    )

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
        async with AsyncSession(engine) as session:
            yield session

    async def admin_override():
        return User(id=1, username="admin", is_admin=True, hashed_password="")

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_admin_user] = admin_override
    router = APIRouter(dependencies=[Depends(get_admin_user)])
    router.include_router(model_preheat_worker_tasks.router, prefix=API_PREFIX)
    app.include_router(router)
    exceptions.register_handlers(app)
    return app, engine, key


async def _seed(engine, key):
    cipher = ModelPreheatCredentialCipher(key, "v1")
    snapshot = cipher.encrypt(
        json.dumps(
            {
                "endpoint": "https://s3.example.com",
                "bucket": "models",
                "prefix": "cache",
                "tls_enabled": True,
                "tls_verify": True,
                "region": "cn-test-1",
                "use_virtual_hosted_style": False,
                "access_key_encrypted": cipher.encrypt("access-plain"),
                "secret_key_encrypted": cipher.encrypt("secret-plain"),
            }
        )
    )
    async with AsyncSession(engine) as session:
        worker = Worker(
            name="worker-a",
            hostname="worker-a",
            ip="127.0.0.1",
            port=10150,
            worker_uuid="worker-uuid",
            state=WorkerStateEnum.READY,
        )
        session.add(worker)
        await session.flush()
        task = ModelPreheatTask(
            source="modelscope",
            model_id="Qwen/Test",
            resolved_revision="commit-1",
            include_patterns=[],
            exclude_patterns=[],
            selection_digest="selection",
            cache_key="cache-key",
            generation_id="generation-1",
            seed_worker_uuid=worker.worker_uuid,
            seed_worker_id=worker.id,
            target_scope=ModelPreheatTargetScopeEnum.SEED_WORKER,
            target_worker_uuids=[worker.worker_uuid],
            target_worker_snapshot=[],
            s3_profile_id=1,
            s3_profile_config_version=1,
            s3_profile_snapshot_encrypted=snapshot,
            encryption_key_version="v1",
            s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
        )
        session.add(task)
        await session.flush()
        worker_task = ModelPreheatWorkerTask(
            task_id=task.id,
            worker_uuid=worker.worker_uuid,
            worker_id=worker.id,
            role=ModelPreheatWorkerTaskRoleEnum.SEED,
        )
        session.add(worker_task)
        await session.commit()
        await session.refresh(worker)
        await session.refresh(worker_task)
        return worker.id, worker_task.id


def _claim(client, task_id, worker_id):
    return client.post(
        f"{API_PREFIX}/{task_id}/claim",
        json={"worker_uuid": "worker-uuid", "worker_id": worker_id},
    )


def test_cas_claim_allows_only_one_concurrent_winner(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))

    def claim_once():
        with TestClient(app) as client:
            return _claim(client, task_id, worker_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: claim_once(), range(2)))

    assert sorted(response.status_code for response in responses) == [200, 409]
    winner = next(response for response in responses if response.status_code == 200)
    assert winner.json()["attempt"] == 1
    assert winner.json()["lease_token"]
    assert "lease_token" not in responses[1 - responses.index(winner)].text
    asyncio.run(engine.dispose())


def test_lease_guards_reject_stale_attempt_token_and_expiry(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, task_id, worker_id).json()
        lease = {
            "worker_uuid": "worker-uuid",
            "worker_id": worker_id,
            "attempt": claimed["attempt"],
            "lease_token": claimed["lease_token"],
        }
        stale_attempt = client.post(
            f"{API_PREFIX}/{task_id}/complete",
            json={**lease, "attempt": 0, "result": {}},
        )
        wrong_token = client.patch(
            f"{API_PREFIX}/{task_id}/progress",
            json={**lease, "lease_token": "wrong", "progress": 10},
        )

        async def expire():
            async with AsyncSession(engine) as session:
                worker_task = await session.get(ModelPreheatWorkerTask, task_id)
                worker_task.lease_expires_at = datetime.now(timezone.utc) - timedelta(
                    seconds=1
                )
                session.add(worker_task)
                await session.commit()

        asyncio.run(expire())
        expired = client.post(f"{API_PREFIX}/{task_id}/heartbeat", json=lease)

    assert stale_attempt.status_code == 409
    assert stale_attempt.json()["message"] == "stale_attempt"
    assert wrong_token.status_code == 409
    assert wrong_token.json()["message"] == "invalid_lease_token"
    assert expired.status_code == 409
    assert expired.json()["message"] == "lease_expired"
    asyncio.run(engine.dispose())


def test_same_uuid_reregistration_rejects_old_process_result(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, task_id, worker_id).json()
        lease = {
            "worker_uuid": "worker-uuid",
            "worker_id": worker_id,
            "attempt": claimed["attempt"],
            "lease_token": claimed["lease_token"],
        }

        async def reregister():
            async with AsyncSession(engine) as session:
                session.add(
                    Worker(
                        name="worker-a-new",
                        hostname="worker-a-new",
                        ip="127.0.0.2",
                        port=10150,
                        worker_uuid="worker-uuid",
                        state=WorkerStateEnum.READY,
                    )
                )
                await session.commit()

        asyncio.run(reregister())
        rejected = client.post(
            f"{API_PREFIX}/{task_id}/complete", json={**lease, "result": {}}
        )

    assert rejected.status_code == 409
    assert rejected.json()["message"] == "stale_worker_registration"
    asyncio.run(engine.dispose())


def test_expired_lease_takeover_invalidates_previous_attempt(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        first = _claim(client, task_id, worker_id).json()

        async def expire():
            async with AsyncSession(engine) as session:
                worker_task = await session.get(ModelPreheatWorkerTask, task_id)
                worker_task.lease_expires_at = datetime.now(timezone.utc) - timedelta(
                    seconds=1
                )
                session.add(worker_task)
                await session.commit()

        asyncio.run(expire())
        second_response = _claim(client, task_id, worker_id)
        second = second_response.json()
        stale = client.post(
            f"{API_PREFIX}/{task_id}/complete",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": worker_id,
                "attempt": first["attempt"],
                "lease_token": first["lease_token"],
                "result": {},
            },
        )

    assert second_response.status_code == 200
    assert second["attempt"] == first["attempt"] + 1
    assert second["lease_token"] != first["lease_token"]
    assert stale.status_code == 409
    assert stale.json()["message"] == "stale_attempt"
    asyncio.run(engine.dispose())


def test_registration_generation_is_part_of_claim_cas(tmp_path, monkeypatch):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    original = model_preheat_worker_tasks._validate_current_registration

    async def reregister_after_validation(session, worker_uuid, claimed_worker_id):
        current = await original(session, worker_uuid, claimed_worker_id)
        session.add(
            Worker(
                name="worker-raced",
                hostname="worker-raced",
                ip="127.0.0.3",
                port=10150,
                worker_uuid=worker_uuid,
                state=WorkerStateEnum.READY,
            )
        )
        await session.flush()
        return current

    monkeypatch.setattr(
        model_preheat_worker_tasks,
        "_validate_current_registration",
        reregister_after_validation,
    )
    with TestClient(app) as client:
        response = _claim(client, task_id, worker_id)

    assert response.status_code == 409
    assert response.json()["message"] == "task_not_claimable"
    asyncio.run(engine.dispose())


def test_duplicate_complete_is_idempotent(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, task_id, worker_id).json()
    body = {
        "worker_uuid": "worker-uuid",
        "worker_id": worker_id,
        "attempt": claimed["attempt"],
        "lease_token": claimed["lease_token"],
        "result": {},
    }

    def complete_once():
        with TestClient(app) as client:
            return client.post(f"{API_PREFIX}/{task_id}/complete", json=body)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: complete_once(), range(2)))

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["state"] == "ready"
    asyncio.run(engine.dispose())


def test_execution_payload_is_claim_bound_no_store_and_public_data_is_sanitized(
    tmp_path,
):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        public = client.get(API_PREFIX, params={"worker_uuid": "worker-uuid"})
        claimed = _claim(client, task_id, worker_id).json()
        headers = {
            "X-Worker-UUID": "worker-uuid",
            "X-Worker-ID": str(worker_id),
            "X-Task-Attempt": str(claimed["attempt"]),
            "X-Lease-Token": claimed["lease_token"],
        }
        payload = client.get(
            f"{API_PREFIX}/{task_id}/execution-payload", headers=headers
        )
        rejected = client.get(
            f"{API_PREFIX}/{task_id}/execution-payload",
            headers={**headers, "X-Lease-Token": "wrong"},
        )

    assert public.status_code == 200
    assert "access-plain" not in public.text
    assert "secret-plain" not in public.text
    assert "lease_token" not in public.text
    assert payload.status_code == 200, payload.text
    assert payload.headers["cache-control"] == "no-store"
    assert payload.json()["profile"]["access_key"] == "access-plain"
    assert payload.json()["profile"]["secret_key"] == "secret-plain"
    assert "s3_profile_snapshot_encrypted" not in payload.json()["task"]
    assert rejected.status_code == 409
    asyncio.run(engine.dispose())


def test_list_can_filter_reconciliation_to_active_states(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        pending = client.get(
            API_PREFIX,
            params=[
                ("worker_uuid", "worker-uuid"),
                ("state", "pending"),
                ("state", "running"),
            ],
        )
        claimed = _claim(client, task_id, worker_id).json()
        completed = client.post(
            f"{API_PREFIX}/{task_id}/complete",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": worker_id,
                "attempt": claimed["attempt"],
                "lease_token": claimed["lease_token"],
                "result": {},
            },
        )
        active_after_complete = client.get(
            API_PREFIX,
            params=[
                ("worker_uuid", "worker-uuid"),
                ("state", "pending"),
                ("state", "running"),
            ],
        )

    assert pending.status_code == 200
    assert [item["id"] for item in pending.json()["items"]] == [task_id]
    assert completed.status_code == 200
    assert active_after_complete.status_code == 200
    assert active_after_complete.json()["items"] == []
    asyncio.run(engine.dispose())


def test_sensitive_worker_result_is_rejected_without_persistence(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, task_id, worker_id).json()
        response = client.post(
            f"{API_PREFIX}/{task_id}/complete",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": worker_id,
                "attempt": claimed["attempt"],
                "lease_token": claimed["lease_token"],
                "result": {"nested": {"secret_key": "secret-plain"}},
            },
        )

    async def persisted_cursor():
        async with AsyncSession(engine) as session:
            return (await session.get(ModelPreheatWorkerTask, task_id)).resumable_cursor

    assert response.status_code == 422
    assert "secret-plain" not in response.text
    assert asyncio.run(persisted_cursor()) is None
    asyncio.run(engine.dispose())
