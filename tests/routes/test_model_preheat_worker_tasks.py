import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.exceptions import HTTPException
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
GENERATION_ID = "preheat-00000000-0000-4000-8000-000000000001"


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
            generation_id=GENERATION_ID,
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


def _ready_result():
    return {
        "state": "ready",
        "manifest_digest": "a" * 64,
        "ready_path": "model-cache/v1/source/model/revision/selection/ready.json",
        "manifest_path": "model-cache/v1/source/model/revision/selection/generations/g/.gpustack-manifest.json",
        "generation_id": GENERATION_ID,
        "local_cache_state": "valid",
        "uploaded": 1,
        "skipped": 2,
        "downloaded": 0,
        "total_size": 3,
    }


def test_seed_result_and_cursor_use_strict_safe_fields_only():
    result = {
        "state": "ready",
        "manifest_digest": "a" * 64,
        "ready_path": "ready.json",
        "manifest_path": "manifest.json",
        "generation_id": GENERATION_ID,
        "local_cache_state": "valid",
        "uploaded": 1,
        "skipped": 2,
        "downloaded": 0,
        "total_size": 3,
        "cursor": {"completed_files": ["config.json"], "staging_exists": False},
    }

    sanitized = model_preheat_worker_tasks._validated_result(
        ModelPreheatWorkerTaskRoleEnum.SEED, result
    )
    assert "ready_path" not in sanitized
    assert "manifest_path" not in sanitized
    assert "cursor" not in sanitized
    assert model_preheat_worker_tasks._validated_cursor(
        ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
        {"completed_files": ["config.json"], "staging_exists": True},
    ) == {"staging_exists": True}
    with pytest.raises(HTTPException, match="invalid_preheat_result"):
        model_preheat_worker_tasks._validated_result(
            ModelPreheatWorkerTaskRoleEnum.SEED,
            {**result, "access_key": "plain"},
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
    first_body = {
        "worker_uuid": "worker-uuid",
        "worker_id": worker_id,
        "attempt": claimed["attempt"],
        "lease_token": claimed["lease_token"],
        "result": _ready_result(),
    }
    with TestClient(app) as client:
        first = client.post(f"{API_PREFIX}/{task_id}/complete", json=first_body)
        second = client.post(
            f"{API_PREFIX}/{task_id}/complete",
            json={**first_body, "result": {}},
        )

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
                "result": _ready_result(),
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


def test_preheat_result_rejects_oversized_or_unsafe_paths():
    valid = {
        "state": "ready",
        "manifest_digest": "a" * 64,
        "ready_path": "cache/modelscope/org/model/revision/ready.json",
        "manifest_path": "cache/modelscope/org/model/revision/generations/g/.gpustack-manifest.json",
        "generation_id": GENERATION_ID,
        "local_cache_state": "valid",
        "uploaded": 0,
        "skipped": 1,
        "downloaded": 0,
        "total_size": 1,
    }

    with pytest.raises(HTTPException, match="invalid_preheat_result"):
        model_preheat_worker_tasks._validated_preheat_result(
            {**valid, "ready_path": "x" * 4097}
        )
    with pytest.raises(HTTPException, match="invalid_preheat_result"):
        model_preheat_worker_tasks._validated_preheat_result(
            {**valid, "manifest_path": "cache/../secret"}
        )


def test_running_preheat_complete_rejects_empty_result(tmp_path):
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
                "result": {},
            },
        )

    assert response.status_code == 422
    asyncio.run(engine.dispose())


def test_progress_persists_validated_cursor_and_safe_state_message(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, task_id, worker_id).json()
        response = client.patch(
            f"{API_PREFIX}/{task_id}/progress",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": worker_id,
                "attempt": claimed["attempt"],
                "lease_token": claimed["lease_token"],
                "progress": 50,
                "resumable_cursor": {
                    "completed_files": ["config.json"],
                    "staging_exists": True,
                },
                "state_message": "downloading",
            },
        )

    async def persisted():
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatWorkerTask, task_id)
            return task.resumable_cursor, task.state_message

    assert response.status_code == 200
    assert asyncio.run(persisted()) == (
        {"staging_exists": True},
        "downloading",
    )
    asyncio.run(engine.dispose())


@pytest.mark.parametrize(
    "message",
    ["token=plain-secret", "AKIAIOSFODNN7EXAMPLE", "x" * 257, "line\nbreak"],
)
def test_progress_rejects_sensitive_or_oversized_state_message(tmp_path, message):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, task_id, worker_id).json()
        response = client.patch(
            f"{API_PREFIX}/{task_id}/progress",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": worker_id,
                "attempt": claimed["attempt"],
                "lease_token": claimed["lease_token"],
                "progress": 50,
                "state_message": message,
            },
        )

    async def persisted_message():
        async with AsyncSession(engine) as session:
            return (await session.get(ModelPreheatWorkerTask, task_id)).state_message

    assert response.status_code == 422
    assert "plain-secret" not in response.text
    assert asyncio.run(persisted_message()) is None
    asyncio.run(engine.dispose())


def test_preheat_result_rejects_generation_id_credential_disguise():
    with pytest.raises(HTTPException, match="invalid_preheat_result"):
        model_preheat_worker_tasks._validated_preheat_result(
            {
                **_ready_result(),
                "generation_id": "preheat-123e4567-e89b-12d3-a456-token00000000",
            }
        )


def test_generation_id_credential_disguise_is_not_persisted(tmp_path):
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
                "result": {
                    **_ready_result(),
                    "generation_id": "preheat-123e4567-e89b-12d3-a456-AKIA00000000",
                },
            },
        )

    async def persisted_cursor():
        async with AsyncSession(engine) as session:
            return (await session.get(ModelPreheatWorkerTask, task_id)).resumable_cursor

    assert response.status_code == 422
    assert "AKIA" not in response.text
    assert asyncio.run(persisted_cursor()) is None
    asyncio.run(engine.dispose())


def test_error_cursor_validates_but_does_not_persist_completed_file_paths(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, task_id, worker_id).json()
        response = client.post(
            f"{API_PREFIX}/{task_id}/fail",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": worker_id,
                "attempt": claimed["attempt"],
                "lease_token": claimed["lease_token"],
                "error_code": "checksum_mismatch",
                "result": {
                    "state": "error",
                    "error_code": "checksum_mismatch",
                    "local_cache_state": "error",
                    "cursor": {
                        "completed_files": [
                            "access/AKIAIOSFODNN7EXAMPLE",
                            "secret/plain-secret",
                        ],
                        "staging_exists": True,
                    },
                },
            },
        )

    async def persisted_cursor():
        async with AsyncSession(engine) as session:
            return (await session.get(ModelPreheatWorkerTask, task_id)).resumable_cursor

    cursor = asyncio.run(persisted_cursor())
    assert response.status_code == 200, response.text
    assert cursor["cursor"] == {"staging_exists": True}
    assert "AKIA" not in json.dumps(cursor)
    assert "plain-secret" not in json.dumps(cursor)
    asyncio.run(engine.dispose())


def test_ready_result_paths_are_not_persisted_even_when_they_look_like_credentials(
    tmp_path,
):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, task_id, worker_id).json()
        result = {
            **_ready_result(),
            "ready_path": "token/plain-secret/ready.json",
            "manifest_path": "credential/plain-secret/manifest.json",
        }
        response = client.post(
            f"{API_PREFIX}/{task_id}/complete",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": worker_id,
                "attempt": claimed["attempt"],
                "lease_token": claimed["lease_token"],
                "result": result,
            },
        )

    async def persisted():
        async with AsyncSession(engine) as session:
            return (await session.get(ModelPreheatWorkerTask, task_id)).resumable_cursor

    cursor = asyncio.run(persisted())
    assert response.status_code == 200
    assert "ready_path" not in cursor
    assert "manifest_path" not in cursor
    assert "plain-secret" not in json.dumps(cursor)
    asyncio.run(engine.dispose())
