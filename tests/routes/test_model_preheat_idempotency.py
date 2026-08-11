import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.auth import get_admin_user
from gpustack.model_preheat_credentials import generate_model_preheat_credential_key
from gpustack.routes import model_preheats
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3ConnectivityStateEnum,
    ModelPreheatS3Profile,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatConnectivityCheckStateEnum,
    ModelPreheatIdempotencyRecord,
    ModelPreheatS3ConnectivityCheck,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_session
from gpustack.server.model_preheat_idempotency import canonical_request_hash


API_PREFIX = "/v1/model-preheats"


async def _create_tables(engine):
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)


async def _drop_tables(engine):
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.drop_all)


async def _seed(session):
    profile = ModelPreheatS3Profile(
        name="profile",
        endpoint="https://s3.example.com",
        bucket="models",
        access_key_encrypted={"ciphertext": "encrypted"},
        secret_key_encrypted={"ciphertext": "encrypted"},
        encryption_key_version="v1",
        connectivity_state=ModelPreheatS3ConnectivityStateEnum.AVAILABLE,
    )
    worker = Worker(
        name="worker-a",
        hostname="worker-a",
        ip="127.0.0.1",
        port=10150,
        worker_uuid="worker-a-uuid",
        state=WorkerStateEnum.READY,
    )
    session.add(profile)
    session.add(worker)
    await session.commit()
    await session.refresh(profile)
    await session.refresh(worker)
    worker_id = worker.id
    worker_uuid = worker.worker_uuid
    checked_at = datetime.now(timezone.utc)
    check = ModelPreheatS3ConnectivityCheck(
        profile_id=profile.id,
        profile_config_version=profile.config_version,
        state=ModelPreheatConnectivityCheckStateEnum.AVAILABLE,
        target_worker_uuids=[worker_uuid],
        finished_at=checked_at,
    )
    session.add(check)
    await session.commit()
    await session.refresh(check)
    session.add(
        ModelPreheatWorkerTask(
            connectivity_check_id=check.id,
            worker_uuid=worker_uuid,
            worker_id=worker_id,
            role=ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
            state=ModelPreheatWorkerTaskStateEnum.READY,
        )
    )
    profile.last_connectivity_check_id = check.id
    profile.last_connectivity_checked_at = checked_at
    session.add(profile)
    await session.commit()
    await session.refresh(profile)
    await session.refresh(worker)
    return profile, worker


def _test_app(tmp_path):
    db_path = tmp_path / "idempotency.db"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
    asyncio.run(_create_tables(engine))

    app = FastAPI()
    app.state.server_config = SimpleNamespace(
        model_preheat_credential_key=generate_model_preheat_credential_key(),
        model_preheat_credential_key_version="v1",
        model_preheat_credential_old_keys=None,
        huggingface_token=None,
    )
    app.state.model_preheat_revision_resolver = (
        lambda source, model_id, revision, token=None: revision
    )

    async def session_override():
        async with AsyncSession(engine) as session:
            yield session

    async def admin_user_override():
        return User(id=1, username="admin", is_admin=True, hashed_password="")

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_admin_user] = admin_user_override
    admin_router = APIRouter(dependencies=[Depends(get_admin_user)])
    admin_router.include_router(model_preheats.router, prefix="/model-preheats")
    app.include_router(admin_router, prefix="/v1")
    exceptions.register_handlers(app)
    return app, engine


def payload(profile_id, worker_id, **overrides):
    result = {
        "source": "modelscope",
        "model_id": "Qwen/Qwen-Image-2512",
        "revision": "commit-123",
        "include_patterns": ["config.json"],
        "exclude_patterns": [],
        "target_scope": "selected_workers",
        "target_worker_ids": [worker_id],
        "s3_profile_id": profile_id,
        "s3_backfill_policy": "when_missing",
    }
    result.update(overrides)
    return result


def test_same_idempotency_key_and_body_replays_original_task(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, worker = asyncio.run(seed())
    with TestClient(app) as client:
        request = payload(profile.id, worker.id)
        first = client.post(
            API_PREFIX, json=request, headers={"Idempotency-Key": "key-1"}
        )
        second = client.post(
            API_PREFIX, json=request, headers={"Idempotency-Key": "key-1"}
        )

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["id"] == second.json()["id"]


def test_same_idempotency_key_with_different_body_returns_conflict(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, worker = asyncio.run(seed())
    with TestClient(app) as client:
        first = client.post(
            API_PREFIX,
            json=payload(profile.id, worker.id),
            headers={"Idempotency-Key": "key-1"},
        )
        second = client.post(
            API_PREFIX,
            json=payload(profile.id, worker.id, revision="commit-456"),
            headers={"Idempotency-Key": "key-1"},
        )

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert first.status_code == 200, first.text
    assert second.status_code == 409
    assert second.json()["reason"] == "idempotency_key_reused"


def test_operation_lock_collision_persists_idempotency_replay_record(
    tmp_path, monkeypatch
):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, worker = asyncio.run(seed())
    original_active_task_for_operation = model_preheats._active_task_for_operation
    calls = 0

    async def race_active_task_for_operation(session, operation_key):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return await original_active_task_for_operation(session, operation_key)

    with TestClient(app) as client:
        initial = client.post(API_PREFIX, json=payload(profile.id, worker.id))
        monkeypatch.setattr(
            model_preheats, "_active_task_for_operation", race_active_task_for_operation
        )
        replay = client.post(
            API_PREFIX,
            json=payload(profile.id, worker.id, keep_new_workers_in_sync=True),
            headers={"Idempotency-Key": "race-key"},
        )
        conflicting_replay = client.post(
            API_PREFIX,
            json=payload(profile.id, worker.id),
            headers={"Idempotency-Key": "race-key"},
        )

    async def stored_record():
        async with AsyncSession(engine) as session:
            return (
                await session.exec(
                    select(ModelPreheatIdempotencyRecord).where(
                        ModelPreheatIdempotencyRecord.idempotency_key == "race-key"
                    )
                )
            ).one()

    record = asyncio.run(stored_record())
    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert initial.status_code == 200, initial.text
    assert replay.status_code == 200, replay.text
    assert replay.json()["id"] == initial.json()["id"]
    assert replay.json()["deduplicated"] is True
    assert record.user_id == 1
    assert record.operation == model_preheats.CREATE_OPERATION
    assert record.idempotency_key == "race-key"
    assert record.request_hash == canonical_request_hash(
        payload(profile.id, worker.id, keep_new_workers_in_sync=True)
    )
    assert record.resource_id == initial.json()["id"]
    assert conflicting_replay.status_code == 409
    assert conflicting_replay.json()["reason"] == "idempotency_key_reused"
