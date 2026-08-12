import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.exceptions import HTTPException
from gpustack.api.auth import get_admin_user
from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
    generate_model_preheat_credential_key,
)
from gpustack.routes import model_preheats, model_preheat_worker_tasks
from gpustack.schemas.model_preheats import (
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatCachedModel,
    ModelPreheatDesiredStateEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatPublicationMarker,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskLease,
    ModelPreheatWorkerTaskProgress,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_session
from gpustack.server.model_preheat_worker_identity import (
    ModelPreheatWorkerPrincipal,
    get_model_preheat_worker_identity,
    issue_model_preheat_worker_credential,
)


API_PREFIX = "/v1/model-preheat-worker-tasks"
GENERATION_ID = "preheat-00000000-0000-4000-8000-000000000001"


def _test_app(tmp_path, *, secure_identity=False):
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
    if not secure_identity:

        async def identity_override():
            return SimpleNamespace(worker_id=1, worker_uuid="worker-uuid")

        app.dependency_overrides[get_model_preheat_worker_identity] = identity_override
    router = APIRouter(dependencies=[Depends(get_admin_user)])
    router.include_router(model_preheat_worker_tasks.router, prefix=API_PREFIX)
    router.include_router(model_preheats.router, prefix="/v1/model-preheats")
    app.include_router(router)
    exceptions.register_handlers(app)
    return app, engine, key


def test_worker_identity_blocks_impersonation_and_rotates_on_reregistration(tmp_path):
    app, engine, key = _test_app(tmp_path, secure_identity=True)

    async def seed_identities():
        first_worker_id, first_task_id = await _seed(engine, key)
        async with AsyncSession(engine) as session:
            first_token = await issue_model_preheat_worker_credential(
                session, first_worker_id, "worker-uuid"
            )
            other = Worker(
                name="worker-b",
                hostname="worker-b",
                ip="127.0.0.2",
                port=10150,
                worker_uuid="worker-b-uuid",
                state=WorkerStateEnum.READY,
            )
            session.add(other)
            await session.flush()
            other_task = ModelPreheatWorkerTask(
                worker_uuid=other.worker_uuid,
                worker_id=other.id,
                role=ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
            )
            session.add(other_task)
            await session.commit()
            await session.refresh(other)
            await session.refresh(other_task)
            other_id = other.id
            other_task_id = other_task.id
            other_token = await issue_model_preheat_worker_credential(
                session, other_id, "worker-b-uuid"
            )
            replacement = Worker(
                name="worker-a-replacement",
                hostname="worker-a",
                ip="127.0.0.3",
                port=10150,
                worker_uuid="worker-uuid",
                state=WorkerStateEnum.READY,
            )
            session.add(replacement)
            await session.commit()
            await session.refresh(replacement)
            replacement_id = replacement.id
            replacement_token = await issue_model_preheat_worker_credential(
                session, replacement_id, "worker-uuid"
            )
            return (
                first_worker_id,
                first_task_id,
                first_token,
                other_id,
                other_task_id,
                other_token,
                replacement_id,
                replacement_token,
            )

    (
        first_worker_id,
        first_task_id,
        first_token,
        other_id,
        other_task_id,
        other_token,
        replacement_id,
        replacement_token,
    ) = asyncio.run(seed_identities())
    with TestClient(app) as client:
        shared_admin = client.get(
            API_PREFIX, headers={"Authorization": "Bearer shared-admin-token"}
        )
        old_registration = client.get(
            API_PREFIX, headers={"X-GPUStack-Worker-Credential": first_token}
        )
        replacement = client.get(
            API_PREFIX,
            params={"worker_uuid": "worker-b-uuid", "worker_id": other_id},
            headers={"X-GPUStack-Worker-Credential": replacement_token},
        )
        forged_claim = client.post(
            f"{API_PREFIX}/{other_task_id}/claim",
            json={"worker_uuid": "worker-b-uuid", "worker_id": other_id},
            headers={"X-GPUStack-Worker-Credential": replacement_token},
        )
        own = client.get(
            API_PREFIX, headers={"X-GPUStack-Worker-Credential": other_token}
        )

    assert shared_admin.status_code == 401
    assert old_registration.status_code == 401
    assert replacement.status_code == 200
    assert replacement.json()["items"] == []
    assert forged_claim.status_code == 409
    assert [item["id"] for item in own.json()["items"]] == [other_task_id]
    assert first_worker_id != replacement_id
    assert first_token not in own.text
    assert other_token not in own.text
    assert replacement_token not in replacement.text
    asyncio.run(engine.dispose())


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
        session.add(
            ModelPreheatPublicationMarker(
                profile_id=task.s3_profile_id,
                selection_key=task.cache_key,
                generation_id=task.generation_id,
                task_id=task.id,
                parent_attempt=task.attempt,
                profile_config_version=task.s3_profile_config_version,
            )
        )
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
    ) == {"completed_files": ["config.json"], "staging_exists": True}
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


def test_seed_cannot_be_claimed_before_publication_marker_exists(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))

    async def remove_marker():
        async with AsyncSession(engine) as session:
            marker = (await session.exec(select(ModelPreheatPublicationMarker))).one()
            await session.delete(marker)
            await session.commit()

    asyncio.run(remove_marker())
    with TestClient(app) as client:
        response = _claim(client, task_id, worker_id)
    assert response.status_code == 409
    assert response.json()["message"] == "publication_marker_required"
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


def test_terminal_replay_validates_identity_attempt_and_token_before_parent_state(
    tmp_path,
):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, task_id, worker_id).json()
        body = {
            "worker_uuid": "worker-uuid",
            "worker_id": worker_id,
            "attempt": claimed["attempt"],
            "lease_token": claimed["lease_token"],
            "result": _ready_result(),
        }
        first = client.post(f"{API_PREFIX}/{task_id}/complete", json=body)

        async def cancel_parent():
            async with AsyncSession(engine) as session:
                worker_task = await session.get(ModelPreheatWorkerTask, task_id)
                parent = await session.get(ModelPreheatTask, worker_task.task_id)
                parent.execution_state = ModelPreheatExecutionStateEnum.CANCELED
                session.add(parent)
                await session.commit()

        asyncio.run(cancel_parent())
        wrong_token = client.post(
            f"{API_PREFIX}/{task_id}/complete",
            json={**body, "lease_token": "wrong", "result": {}},
        )
        replay = client.post(
            f"{API_PREFIX}/{task_id}/complete", json={**body, "result": {}}
        )

    assert first.status_code == 200
    assert wrong_token.status_code == 409
    assert wrong_token.json()["message"] == "invalid_lease_token"
    assert replay.status_code == 200
    assert replay.json()["state"] == "ready"
    asyncio.run(engine.dispose())


def test_execution_payload_is_claim_bound_no_store_and_public_data_is_sanitized(
    tmp_path,
):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))

    async def seed_cursor():
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatWorkerTask, task_id)
            task.resumable_cursor = {
                "completed_files": ["weights/model%207b.bin"],
                "staging_exists": True,
            }
            session.add(task)
            await session.commit()

    asyncio.run(seed_cursor())
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
    assert payload.json()["resumable_cursor"] == {
        "completed_files": ["weights/model%207b.bin"],
        "staging_exists": True,
    }
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


def test_seed_complete_does_not_trust_worker_result_as_valid_inventory(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_id, task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, task_id, worker_id).json()
        body = {
            "worker_uuid": "worker-uuid",
            "worker_id": worker_id,
            "attempt": claimed["attempt"],
            "lease_token": claimed["lease_token"],
            "result": _ready_result(),
        }
        assert (
            client.post(f"{API_PREFIX}/{task_id}/complete", json=body).status_code
            == 200
        )
        assert (
            client.post(f"{API_PREFIX}/{task_id}/complete", json=body).status_code
            == 200
        )

    async def inventory():
        async with AsyncSession(engine) as session:
            return (await session.exec(select(ModelPreheatCachedModel))).all()

    assert asyncio.run(inventory()) == []
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
                    "completed_files": [
                        "config.json",
                        "weights/model%207b.bin",
                        "%E8%B5%84%E6%96%99/%E6%A8%A1%E5%9E%8B.bin",
                    ],
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
        {
            "completed_files": [
                "config.json",
                "weights/model%207b.bin",
                "%E8%B5%84%E6%96%99/%E6%A8%A1%E5%9E%8B.bin",
            ],
            "staging_exists": True,
        },
        "downloading",
    )
    asyncio.run(engine.dispose())


def test_paused_worker_uses_active_lease_to_persist_cursor_and_confirm_pause(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_id, worker_task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, worker_task_id, worker_id).json()

        async def request_pause():
            async with AsyncSession(engine) as session:
                worker_task = await session.get(ModelPreheatWorkerTask, worker_task_id)
                parent = await session.get(ModelPreheatTask, worker_task.task_id)
                parent.desired_state = ModelPreheatDesiredStateEnum.PAUSED
                worker_task.state_message = "pause_requested"
                session.add(parent)
                session.add(worker_task)
                await session.commit()

        asyncio.run(request_pause())
        heartbeat = client.post(
            f"{API_PREFIX}/{worker_task_id}/heartbeat",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": worker_id,
                "attempt": claimed["attempt"],
                "lease_token": claimed["lease_token"],
            },
        )
        payload = client.get(
            f"{API_PREFIX}/{worker_task_id}/execution-payload",
            headers={
                "X-Worker-UUID": "worker-uuid",
                "X-Worker-ID": str(worker_id),
                "X-Task-Attempt": str(claimed["attempt"]),
                "X-Lease-Token": claimed["lease_token"],
            },
        )
        boundary = client.patch(
            f"{API_PREFIX}/{worker_task_id}/progress",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": worker_id,
                "attempt": claimed["attempt"],
                "lease_token": claimed["lease_token"],
                "progress": 50,
                "resumable_cursor": {
                    "completed_files": ["weights/model%207b.bin"],
                    "staging_exists": True,
                },
                "state_message": "downloading",
            },
        )
        response = client.patch(
            f"{API_PREFIX}/{worker_task_id}/progress",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": worker_id,
                "attempt": claimed["attempt"],
                "lease_token": claimed["lease_token"],
                "progress": 50,
                "resumable_cursor": {
                    "completed_files": ["weights/model%207b.bin"],
                    "staging_exists": True,
                },
                "state_message": "paused",
            },
        )

    async def persisted():
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatWorkerTask, worker_task_id)
            parent = await session.get(ModelPreheatTask, task.task_id)
            return (
                task.state,
                task.resumable_cursor,
                task.lease_owner,
                task.lease_token_hash,
                parent.execution_state,
            )

    assert heartbeat.status_code == 409, heartbeat.text
    assert heartbeat.json()["message"] == "parent_not_running"
    assert payload.status_code == 409, payload.text
    assert payload.json()["message"] == "parent_not_running"
    assert boundary.status_code == 200, boundary.text
    assert response.status_code == 200, response.text
    assert asyncio.run(persisted()) == (
        ModelPreheatWorkerTaskStateEnum.PAUSED,
        {
            "completed_files": ["weights/model%207b.bin"],
            "staging_exists": True,
        },
        None,
        None,
        ModelPreheatExecutionStateEnum.PAUSED,
    )
    asyncio.run(engine.dispose())


def test_pause_confirmation_updates_parent_before_child(tmp_path):
    app, engine, key = _test_app(tmp_path)
    worker_id, worker_task_id = asyncio.run(_seed(engine, key))

    async def request_pause():
        async with AsyncSession(engine) as session:
            child = await session.get(ModelPreheatWorkerTask, worker_task_id)
            parent = await session.get(ModelPreheatTask, child.task_id)
            parent.desired_state = ModelPreheatDesiredStateEnum.PAUSED
            parent.paused_from_state = parent.execution_state
            child.state_message = "pause_requested"
            session.add(parent)
            session.add(child)
            await session.commit()

    update_tables = []

    def record_update_order(
        connection, cursor, statement, parameters, context, executemany
    ):
        del connection, cursor, parameters, context, executemany
        normalized = " ".join(statement.lower().split())
        if normalized.startswith("update model_preheat_"):
            update_tables.append(normalized.split()[1].strip('"`'))

    with TestClient(app) as client:
        claimed_response = _claim(client, worker_task_id, worker_id)
        assert claimed_response.status_code == 200, claimed_response.text
        claimed = claimed_response.json()
        asyncio.run(request_pause())
        event.listen(engine.sync_engine, "before_cursor_execute", record_update_order)
        try:
            confirmation = client.patch(
                f"{API_PREFIX}/{worker_task_id}/progress",
                json={
                    "worker_uuid": "worker-uuid",
                    "worker_id": worker_id,
                    "attempt": claimed["attempt"],
                    "lease_token": claimed["lease_token"],
                    "progress": 50,
                    "state_message": "paused",
                },
            )
        finally:
            event.remove(
                engine.sync_engine, "before_cursor_execute", record_update_order
            )

    assert confirmation.status_code == 200, confirmation.text
    assert update_tables == [
        "model_preheat_tasks",
        "model_preheat_worker_tasks",
        "model_preheat_tasks",
    ]
    asyncio.run(engine.dispose())


def test_resume_route_recovers_pause_pending_child_and_rejects_late_confirmation(
    tmp_path,
):
    app, engine, key = _test_app(tmp_path)
    worker_id, worker_task_id = asyncio.run(_seed(engine, key))

    async def request_schedule_pause():
        async with AsyncSession(engine) as session:
            child = await session.get(ModelPreheatWorkerTask, worker_task_id)
            parent = await session.get(ModelPreheatTask, child.task_id)
            parent.desired_state = ModelPreheatDesiredStateEnum.PAUSED
            parent.paused_from_state = parent.execution_state
            child.state_message = "pause_requested"
            session.add(parent)
            session.add(child)
            parent_id = parent.id
            await session.commit()
            return parent_id

    update_tables = []

    def record_update_order(
        connection, cursor, statement, parameters, context, executemany
    ):
        del connection, cursor, parameters, context, executemany
        normalized = " ".join(statement.lower().split())
        if normalized.startswith("update model_preheat_"):
            update_tables.append(normalized.split()[1].strip('"`'))

    with TestClient(app) as client:
        claimed_response = _claim(client, worker_task_id, worker_id)
        assert claimed_response.status_code == 200, claimed_response.text
        claimed = claimed_response.json()
        parent_id = asyncio.run(request_schedule_pause())

        event.listen(engine.sync_engine, "before_cursor_execute", record_update_order)
        try:
            resumed = client.post(f"/v1/model-preheats/{parent_id}/resume")
        finally:
            event.remove(
                engine.sync_engine, "before_cursor_execute", record_update_order
            )

        async def state_after_resume():
            async with AsyncSession(engine) as session:
                child = await session.get(ModelPreheatWorkerTask, worker_task_id)
                return (
                    child.state,
                    child.state_message,
                    child.lease_owner,
                    child.lease_token_hash,
                )

        after_resume = asyncio.run(state_after_resume())
        late_confirmation = client.patch(
            f"{API_PREFIX}/{worker_task_id}/progress",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": worker_id,
                "attempt": claimed["attempt"],
                "lease_token": claimed["lease_token"],
                "progress": 50,
                "state_message": "paused",
            },
        )
        reclaimed_response = _claim(client, worker_task_id, worker_id)

    async def final_state():
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, parent_id)
            child = await session.get(ModelPreheatWorkerTask, worker_task_id)
            return (
                parent.desired_state,
                parent.execution_state,
                child.state,
                child.state_message,
            )

    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["desired_state"] == "running"
    assert update_tables[:2] == [
        "model_preheat_tasks",
        "model_preheat_worker_tasks",
    ]
    assert after_resume == (
        ModelPreheatWorkerTaskStateEnum.PENDING,
        None,
        None,
        None,
    )
    assert late_confirmation.status_code == 409, late_confirmation.text
    assert reclaimed_response.status_code == 200, reclaimed_response.text
    assert reclaimed_response.json()["lease_token"] != claimed["lease_token"]
    assert asyncio.run(final_state()) == (
        ModelPreheatDesiredStateEnum.RUNNING,
        ModelPreheatExecutionStateEnum.PENDING,
        ModelPreheatWorkerTaskStateEnum.RUNNING,
        None,
    )
    asyncio.run(engine.dispose())


def test_progress_database_update_preserves_concurrent_pause_request(
    tmp_path, monkeypatch
):
    app, engine, key = _test_app(tmp_path)
    worker_id, worker_task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, worker_task_id, worker_id).json()

    async def run():
        validated = asyncio.Event()
        release_progress = asyncio.Event()
        original_validate = model_preheat_worker_tasks._validate_active_lease

        async def validate_then_wait(*args, **kwargs):
            worker_task = await original_validate(*args, **kwargs)
            validated.set()
            await release_progress.wait()
            return worker_task

        monkeypatch.setattr(
            model_preheat_worker_tasks,
            "_validate_active_lease",
            validate_then_wait,
        )
        identity = ModelPreheatWorkerPrincipal(
            worker_id=worker_id,
            worker_uuid="worker-uuid",
            credential_id=1,
            token_version=1,
        )
        progress = ModelPreheatWorkerTaskProgress(
            worker_uuid="worker-uuid",
            worker_id=worker_id,
            attempt=claimed["attempt"],
            lease_token=claimed["lease_token"],
            progress=45,
            state_message="downloading",
        )
        async with AsyncSession(engine) as stale_session:
            progress_update = asyncio.create_task(
                model_preheat_worker_tasks.update_model_preheat_worker_task_progress(
                    stale_session,
                    worker_task_id,
                    progress,
                    identity,
                )
            )
            await asyncio.wait_for(validated.wait(), timeout=1)
            async with AsyncSession(engine) as controller_session:
                worker_task = await controller_session.get(
                    ModelPreheatWorkerTask, worker_task_id
                )
                parent = await controller_session.get(
                    ModelPreheatTask, worker_task.task_id
                )
                parent.desired_state = ModelPreheatDesiredStateEnum.PAUSED
                worker_task.state_message = "pause_requested"
                controller_session.add(parent)
                controller_session.add(worker_task)
                await controller_session.commit()
            release_progress.set()
            await progress_update

        async with AsyncSession(engine) as session:
            persisted = await session.get(ModelPreheatWorkerTask, worker_task_id)
            return persisted.progress, persisted.state_message

    assert asyncio.run(run()) == (45, "pause_requested")
    asyncio.run(engine.dispose())


def test_heartbeat_database_cas_rejects_concurrent_pause_without_renewal(
    tmp_path, monkeypatch
):
    app, engine, key = _test_app(tmp_path)
    worker_id, worker_task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, worker_task_id, worker_id).json()

    async def run():
        validated = asyncio.Event()
        release_heartbeat = asyncio.Event()
        original_validate = model_preheat_worker_tasks._validate_active_lease

        async def validate_then_wait(*args, **kwargs):
            worker_task = await original_validate(*args, **kwargs)
            validated.set()
            await release_heartbeat.wait()
            return worker_task

        monkeypatch.setattr(
            model_preheat_worker_tasks,
            "_validate_active_lease",
            validate_then_wait,
        )
        identity = ModelPreheatWorkerPrincipal(
            worker_id=worker_id,
            worker_uuid="worker-uuid",
            credential_id=1,
            token_version=1,
        )
        lease = ModelPreheatWorkerTaskLease(
            worker_uuid="worker-uuid",
            worker_id=worker_id,
            attempt=claimed["attempt"],
            lease_token=claimed["lease_token"],
        )
        async with AsyncSession(engine) as stale_session:
            heartbeat_update = asyncio.create_task(
                model_preheat_worker_tasks.heartbeat_model_preheat_worker_task(
                    stale_session,
                    worker_task_id,
                    lease,
                    identity,
                )
            )
            await asyncio.wait_for(validated.wait(), timeout=1)
            async with AsyncSession(engine) as controller_session:
                worker_task = await controller_session.get(
                    ModelPreheatWorkerTask, worker_task_id
                )
                original_expiry = worker_task.lease_expires_at
                parent = await controller_session.get(
                    ModelPreheatTask, worker_task.task_id
                )
                parent.desired_state = ModelPreheatDesiredStateEnum.PAUSED
                worker_task.state_message = "pause_requested"
                controller_session.add(parent)
                controller_session.add(worker_task)
                await controller_session.commit()
            release_heartbeat.set()
            with pytest.raises(HTTPException) as conflict:
                await heartbeat_update

        async with AsyncSession(engine) as session:
            persisted = await session.get(ModelPreheatWorkerTask, worker_task_id)
            return (
                conflict.value.status_code,
                conflict.value.message,
                persisted.lease_expires_at == original_expiry,
                persisted.state_message,
            )

    assert asyncio.run(run()) == (409, "lease_lost", True, "pause_requested")
    asyncio.run(engine.dispose())


@pytest.mark.parametrize(
    ("parent_execution_state", "state_message"),
    [
        (ModelPreheatExecutionStateEnum.PENDING, "pause_requested"),
        (ModelPreheatExecutionStateEnum.PAUSED, "downloading"),
    ],
    ids=["pause-ack-pending", "parent-paused"],
)
def test_expired_child_of_pausing_parent_cannot_be_reclaimed_or_renewed(
    tmp_path, parent_execution_state, state_message
):
    app, engine, key = _test_app(tmp_path)
    worker_id, worker_task_id = asyncio.run(_seed(engine, key))
    with TestClient(app) as client:
        claimed = _claim(client, worker_task_id, worker_id).json()

        async def expire_during_pause():
            async with AsyncSession(engine) as session:
                worker_task = await session.get(ModelPreheatWorkerTask, worker_task_id)
                parent = await session.get(ModelPreheatTask, worker_task.task_id)
                parent.desired_state = ModelPreheatDesiredStateEnum.PAUSED
                parent.execution_state = parent_execution_state
                worker_task.state_message = state_message
                worker_task.lease_expires_at = datetime.now(timezone.utc) - timedelta(
                    seconds=1
                )
                session.add(parent)
                session.add(worker_task)
                await session.commit()

        asyncio.run(expire_during_pause())
        reclaimed = _claim(client, worker_task_id, worker_id)
        candidate = reclaimed.json() if reclaimed.status_code == 200 else claimed
        heartbeat = client.post(
            f"{API_PREFIX}/{worker_task_id}/heartbeat",
            json={
                "worker_uuid": "worker-uuid",
                "worker_id": worker_id,
                "attempt": candidate["attempt"],
                "lease_token": candidate["lease_token"],
            },
        )

    async def persisted():
        async with AsyncSession(engine) as session:
            worker_task = await session.get(ModelPreheatWorkerTask, worker_task_id)
            return worker_task.attempt, worker_task.state_message

    assert reclaimed.status_code == 409, reclaimed.text
    assert reclaimed.json()["message"] == "task_not_claimable"
    assert heartbeat.status_code == 409, heartbeat.text
    assert asyncio.run(persisted()) == (claimed["attempt"], state_message)
    asyncio.run(engine.dispose())


def test_parent_pauses_only_after_every_child_acknowledges(tmp_path):
    app, engine, key = _test_app(tmp_path)
    del app
    first_worker_id, first_worker_task_id = asyncio.run(_seed(engine, key))

    async def run():
        first_token = "first-lease"
        second_token = "second-lease"
        async with AsyncSession(engine) as session:
            first_child = await session.get(
                ModelPreheatWorkerTask, first_worker_task_id
            )
            parent = await session.get(ModelPreheatTask, first_child.task_id)
            second_worker = Worker(
                name="worker-b",
                hostname="worker-b",
                ip="127.0.0.2",
                port=10150,
                worker_uuid="worker-b-uuid",
                state=WorkerStateEnum.READY,
            )
            session.add(second_worker)
            await session.flush()
            second_child = ModelPreheatWorkerTask(
                task_id=parent.id,
                parent_attempt=parent.attempt,
                worker_uuid=second_worker.worker_uuid,
                worker_id=second_worker.id,
                role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                state=ModelPreheatWorkerTaskStateEnum.RUNNING,
                attempt=1,
                lease_owner=second_worker.worker_uuid,
                lease_token_hash=model_preheat_worker_tasks._hash_token(second_token),
                lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
                state_message="pause_requested",
            )
            first_child.state = ModelPreheatWorkerTaskStateEnum.RUNNING
            first_child.attempt = 1
            first_child.lease_owner = first_child.worker_uuid
            first_child.lease_token_hash = model_preheat_worker_tasks._hash_token(
                first_token
            )
            first_child.lease_expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=60
            )
            first_child.state_message = "pause_requested"
            parent.desired_state = ModelPreheatDesiredStateEnum.PAUSED
            session.add(first_child)
            session.add(second_child)
            session.add(parent)
            await session.commit()
            await session.refresh(second_worker)
            await session.refresh(second_child)
            second_worker_id = second_worker.id
            second_worker_task_id = second_child.id

        async def confirm(worker_task_id, worker_id, worker_uuid, token):
            identity = ModelPreheatWorkerPrincipal(
                worker_id=worker_id,
                worker_uuid=worker_uuid,
                credential_id=1,
                token_version=1,
            )
            async with AsyncSession(engine) as session:
                await model_preheat_worker_tasks.update_model_preheat_worker_task_progress(
                    session,
                    worker_task_id,
                    ModelPreheatWorkerTaskProgress(
                        worker_uuid=worker_uuid,
                        worker_id=worker_id,
                        attempt=1,
                        lease_token=token,
                        progress=50,
                        state_message="paused",
                    ),
                    identity,
                )

        await confirm(
            first_worker_task_id,
            first_worker_id,
            "worker-uuid",
            first_token,
        )
        async with AsyncSession(engine) as session:
            parent = (await session.exec(select(ModelPreheatTask))).one()
            after_first = parent.execution_state

        await confirm(
            second_worker_task_id,
            second_worker_id,
            "worker-b-uuid",
            second_token,
        )
        async with AsyncSession(engine) as session:
            parent = (await session.exec(select(ModelPreheatTask))).one()
            children = (
                await session.exec(
                    select(ModelPreheatWorkerTask).order_by(ModelPreheatWorkerTask.id)
                )
            ).all()
            return (
                after_first,
                parent.execution_state,
                [child.state for child in children],
            )

    assert asyncio.run(run()) == (
        ModelPreheatExecutionStateEnum.PENDING,
        ModelPreheatExecutionStateEnum.PAUSED,
        [
            ModelPreheatWorkerTaskStateEnum.PAUSED,
            ModelPreheatWorkerTaskStateEnum.PAUSED,
        ],
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


def test_error_cursor_persists_canonical_paths_without_exposing_them(tmp_path):
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
                            "weights/model%207b.bin",
                            "%E8%B5%84%E6%96%99/%E6%A8%A1%E5%9E%8B.bin",
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
    assert cursor["cursor"] == {
        "completed_files": [
            "weights/model%207b.bin",
            "%E8%B5%84%E6%96%99/%E6%A8%A1%E5%9E%8B.bin",
        ],
        "staging_exists": True,
    }
    assert "completed_files" not in response.text
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
