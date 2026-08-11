import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import UniqueConstraint, text
from sqlalchemy.exc import IntegrityError
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
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatDesiredStateEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatIdempotencyRecord,
    ModelPreheatS3ConnectivityCheck,
    ModelPreheatTask,
    ModelPreheatTaskLock,
    ModelPreheatTargetScopeEnum,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.users import User
from gpustack.schemas.workers import (
    GPUDeviceInfo,
    Worker,
    WorkerStateEnum,
    WorkerStatus,
)
from gpustack.server.db import get_session
from gpustack.server.model_preheat_connectivity import aggregate_connectivity_check


API_PREFIX = "/v1/model-preheats"


async def _create_tables(engine):
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)


async def _drop_tables(engine):
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.drop_all)


def _test_app(tmp_path):
    db_path = tmp_path / "preheats.db"
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


async def _seed(session, profile_state=ModelPreheatS3ConnectivityStateEnum.AVAILABLE):
    profile = ModelPreheatS3Profile(
        name="profile",
        endpoint="https://s3.example.com",
        bucket="models",
        access_key_encrypted={"ciphertext": "encrypted"},
        secret_key_encrypted={"ciphertext": "encrypted"},
        encryption_key_version="v1",
        connectivity_state=profile_state,
    )
    workers = [
        Worker(
            name="worker-z",
            hostname="worker-z",
            ip="127.0.0.1",
            port=10150,
            worker_uuid="z-uuid",
            state=WorkerStateEnum.READY,
        ),
        Worker(
            name="worker-a",
            hostname="worker-a",
            ip="127.0.0.2",
            port=10150,
            worker_uuid="a-uuid",
            state=WorkerStateEnum.READY,
        ),
    ]
    session.add(profile)
    session.add_all(workers)
    await session.commit()
    await session.refresh(profile)
    for worker in workers:
        await session.refresh(worker)
    worker_snapshot = [
        {"id": worker.id, "worker_uuid": worker.worker_uuid} for worker in workers
    ]
    if profile_state == ModelPreheatS3ConnectivityStateEnum.AVAILABLE:
        checked_at = datetime.now(timezone.utc)
        check = ModelPreheatS3ConnectivityCheck(
            profile_id=profile.id,
            profile_config_version=profile.config_version,
            state=ModelPreheatConnectivityCheckStateEnum.AVAILABLE,
            target_worker_uuids=[worker["worker_uuid"] for worker in worker_snapshot],
            finished_at=checked_at,
        )
        session.add(check)
        await session.commit()
        await session.refresh(check)
        session.add_all(
            [
                ModelPreheatWorkerTask(
                    connectivity_check_id=check.id,
                    worker_uuid=worker["worker_uuid"],
                    worker_id=worker["id"],
                    role=ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
                    state=ModelPreheatWorkerTaskStateEnum.READY,
                )
                for worker in worker_snapshot
            ]
        )
        profile.last_connectivity_check_id = check.id
        profile.last_connectivity_checked_at = checked_at
        session.add(profile)
        await session.commit()
        await session.refresh(profile)
        for worker in workers:
            await session.refresh(worker)
    return profile, workers


def payload(profile_id, worker_ids, **overrides):
    result = {
        "source": "modelscope",
        "model_id": "Qwen/Qwen-Image-2512",
        "revision": "commit-123",
        "include_patterns": ["weights/*.safetensors", "config.json"],
        "exclude_patterns": [],
        "target_scope": "selected_workers",
        "target_worker_ids": worker_ids,
        "s3_profile_id": profile_id,
        "s3_backfill_policy": "when_missing",
    }
    result.update(overrides)
    return result


def test_operation_lock_deduplicates_without_idempotency_key_and_snapshots_sorted_targets(
    tmp_path,
):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    with TestClient(app) as client:
        first = client.post(
            API_PREFIX, json=payload(profile.id, [workers[0].id, workers[1].id])
        )
        second = client.post(
            API_PREFIX, json=payload(profile.id, [workers[1].id, workers[0].id])
        )

    async def get_created_task():
        async with AsyncSession(engine) as session:
            return await session.get(ModelPreheatTask, first.json()["id"])

    created_task = asyncio.run(get_created_task())

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["id"] == second.json()["id"]
    assert second.json()["deduplicated"] is True
    assert first.json()["target_worker_uuids"] == ["a-uuid", "z-uuid"]
    assert [item["worker_uuid"] for item in first.json()["target_worker_snapshot"]] == [
        "a-uuid",
        "z-uuid",
    ]
    assert created_task.seed_worker_uuid == "a-uuid"
    assert created_task.created_by_user_id == 1


def test_creation_resolves_requested_revision_before_persisting(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    calls = []

    def resolver(source, model_id, revision, token=None):
        calls.append((source, model_id, revision, token))
        return "a" * 40

    app.state.model_preheat_revision_resolver = resolver
    with TestClient(app) as client:
        response = client.post(
            API_PREFIX,
            json=payload(
                profile.id,
                [workers[0].id],
                source="huggingface",
                revision="release-branch",
            ),
        )

    assert response.status_code == 200, response.text
    assert response.json()["requested_revision"] == "release-branch"
    assert response.json()["resolved_revision"] == "a" * 40
    assert calls == [("huggingface", "Qwen/Qwen-Image-2512", "release-branch", None)]
    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())


def test_creation_revision_resolution_failure_is_stable_and_sanitized(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    app.state.model_preheat_revision_resolver = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("token=plain-secret upstream detail"))
    with TestClient(app) as client:
        response = client.post(
            API_PREFIX,
            json=payload(profile.id, [workers[0].id]),
        )

    assert response.status_code == 422
    assert response.json()["message"] == "remote_revision_resolution_failed"
    assert "plain-secret" not in response.text
    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())


def test_expired_operation_lock_keeps_non_terminal_task_idempotent(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    request = payload(profile.id, [worker.id for worker in workers])
    with TestClient(app) as client:
        first = client.post(API_PREFIX, json=request)

        async def expire_lock():
            async with AsyncSession(engine) as session:
                lock = (await session.exec(select(ModelPreheatTaskLock))).one()
                lock.lease_expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
                session.add(lock)
                await session.commit()

        asyncio.run(expire_lock())
        second = client.post(API_PREFIX, json=request)

    async def locks():
        async with AsyncSession(engine) as session:
            return (await session.exec(select(ModelPreheatTaskLock))).all()

    stored_locks = asyncio.run(locks())
    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["id"] == second.json()["id"]
    assert second.json()["deduplicated"] is True
    assert len(stored_locks) == 1
    assert stored_locks[0].task_id == first.json()["id"]


def test_cancel_route_is_idempotent_and_releases_operation_lock(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    with TestClient(app) as client:
        created = client.post(
            API_PREFIX, json=payload(profile.id, [worker.id for worker in workers])
        )

    with TestClient(app) as client:
        finalize = client.post(
            f"{API_PREFIX}/{created.json()['id']}/finalize",
            json={"execution_state": "ready"},
        )
        canceled = client.post(f"{API_PREFIX}/{created.json()['id']}/cancel")
        repeated = client.post(f"{API_PREFIX}/{created.json()['id']}/cancel")
        replacement = client.post(
            API_PREFIX, json=payload(profile.id, [worker.id for worker in workers])
        )

    async def canceled_task_and_locks():
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatTask, created.json()["id"])
            locks = (await session.exec(select(ModelPreheatTaskLock))).all()
            return task, locks

    task, locks = asyncio.run(canceled_task_and_locks())
    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert created.status_code == 200, created.text
    assert finalize.status_code == 404
    assert canceled.status_code == 200, canceled.text
    assert repeated.status_code == 200, repeated.text
    assert canceled.json()["desired_state"] == "canceled"
    assert canceled.json()["execution_state"] == "canceled"
    assert repeated.json()["execution_state"] == "canceled"
    assert task.finished_at is not None
    assert all(lock.task_id != task.id for lock in locks)
    assert replacement.status_code == 200, replacement.text
    assert replacement.json()["id"] != created.json()["id"]


def test_cancel_does_not_overwrite_terminal_task(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    with TestClient(app) as client:
        created = client.post(
            API_PREFIX, json=payload(profile.id, [worker.id for worker in workers])
        )

        async def mark_error():
            async with AsyncSession(engine) as session:
                task = await session.get(ModelPreheatTask, created.json()["id"])
                task.execution_state = ModelPreheatExecutionStateEnum.ERROR
                task.finished_at = datetime.now(timezone.utc)
                session.add(task)
                await session.commit()

        asyncio.run(mark_error())
        canceled = client.post(f"{API_PREFIX}/{created.json()['id']}/cancel")

    async def terminal_task_and_locks():
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatTask, created.json()["id"])
            locks = (await session.exec(select(ModelPreheatTaskLock))).all()
            return task, locks

    task, locks = asyncio.run(terminal_task_and_locks())

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert created.status_code == 200, created.text
    assert canceled.status_code == 200, canceled.text
    assert canceled.json()["desired_state"] == "running"
    assert canceled.json()["execution_state"] == "error"
    assert task.desired_state == ModelPreheatDesiredStateEnum.RUNNING
    assert task.execution_state == ModelPreheatExecutionStateEnum.ERROR
    assert all(lock.task_id != task.id for lock in locks)


def test_selected_workers_uses_explicit_online_seed_worker(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    with TestClient(app) as client:
        response = client.post(
            API_PREFIX,
            json=payload(
                profile.id,
                [worker.id for worker in workers],
                seed_worker_id=workers[0].id,
            ),
        )

    async def get_task_and_worker_tasks():
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatTask, response.json()["id"])
            worker_tasks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.task_id == task.id
                    )
                )
            ).all()
            return task, worker_tasks

    task, worker_tasks = asyncio.run(get_task_and_worker_tasks())
    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 200, response.text
    assert task.seed_worker_id == workers[0].id
    assert task.seed_worker_uuid == workers[0].worker_uuid
    assert worker_tasks == []


def test_seed_worker_scope_targets_only_seed_and_creates_no_distribute_task(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    seed_worker = workers[1]
    with TestClient(app) as client:
        response = client.post(
            API_PREFIX,
            json=payload(
                profile.id,
                [],
                target_scope="seed_worker",
                seed_worker_id=seed_worker.id,
            ),
        )

    async def get_worker_tasks():
        async with AsyncSession(engine) as session:
            return (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.task_id == response.json()["id"]
                    )
                )
            ).all()

    worker_tasks = asyncio.run(get_worker_tasks())
    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 200, response.text
    assert response.json()["target_worker_uuids"] == [seed_worker.worker_uuid]
    assert worker_tasks == []


def test_same_gpu_model_scope_snapshots_ready_workers_with_matching_gpu_names(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            profile, workers = await _seed(session)
            workers[0].status = WorkerStatus(
                gpu_devices=[GPUDeviceInfo(name="NVIDIA A100")]
            )
            workers[1].status = WorkerStatus(
                gpu_devices=[GPUDeviceInfo(name="  nvidia   a100  ")]
            )
            other = Worker(
                name="worker-other",
                hostname="worker-other",
                ip="127.0.0.3",
                port=10150,
                worker_uuid="other-uuid",
                state=WorkerStateEnum.READY,
                status=WorkerStatus(gpu_devices=[GPUDeviceInfo(name="NVIDIA H100")]),
            )
            session.add(other)
            other_uuid = other.worker_uuid
            await session.commit()
            await session.refresh(other)
            other_id = other.id
            await session.refresh(profile)
            check = await session.get(
                ModelPreheatS3ConnectivityCheck, profile.last_connectivity_check_id
            )
            check.target_worker_uuids = [
                *check.target_worker_uuids,
                other_uuid,
            ]
            session.add(check)
            session.add(
                ModelPreheatWorkerTask(
                    connectivity_check_id=check.id,
                    worker_uuid=other_uuid,
                    worker_id=other_id,
                    role=ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
                    state=ModelPreheatWorkerTaskStateEnum.READY,
                )
            )
            await session.commit()
            await session.refresh(profile)
            for worker in workers:
                await session.refresh(worker)
            await session.refresh(other)
            return profile, workers, other

    profile, workers, other = asyncio.run(seed())
    with TestClient(app) as client:
        response = client.post(
            API_PREFIX,
            json=payload(
                profile.id,
                [],
                target_scope="same_gpu_model",
                seed_worker_id=workers[0].id,
            ),
        )

    async def get_task_and_worker_tasks():
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatTask, response.json()["id"])
            worker_tasks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.task_id == task.id
                    )
                )
            ).all()
            return task, worker_tasks

    task, worker_tasks = asyncio.run(get_task_and_worker_tasks())
    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 200, response.text
    assert task.target_gpu_names == ["nvidia a100"]
    assert task.target_worker_uuids == ["a-uuid", "z-uuid"]
    assert other.worker_uuid not in task.target_worker_uuids
    assert worker_tasks == []


def test_same_gpu_model_scope_requires_identifiable_seed_gpu(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    with TestClient(app) as client:
        response = client.post(
            API_PREFIX,
            json=payload(
                profile.id,
                [],
                target_scope="same_gpu_model",
                seed_worker_id=workers[0].id,
            ),
        )

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 422
    assert response.json()["message"] == "seed_worker_gpu_required"


@pytest.mark.parametrize("target_scope", ["seed_worker", "same_gpu_model"])
def test_seed_scopes_require_seed_worker_id(tmp_path, target_scope):
    app, engine = _test_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            API_PREFIX,
            json=payload(1, [], target_scope=target_scope),
        )

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 422
    assert "seed_worker_id_required" in response.json()["message"]


def test_selected_workers_rejects_seed_outside_online_target_scope(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    with TestClient(app) as client:
        outside_target = client.post(
            API_PREFIX,
            json=payload(
                profile.id,
                [workers[0].id],
                seed_worker_id=workers[1].id,
            ),
        )
        offline_seed = client.post(
            API_PREFIX,
            json=payload(profile.id, [workers[0].id], seed_worker_id=999),
        )

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert outside_target.status_code == 422
    assert outside_target.json()["message"] == "seed_worker_not_in_target_scope"
    assert offline_seed.status_code == 422
    assert offline_seed.json()["message"] == "seed_worker_not_online"


def test_creation_resolves_and_persists_default_revision(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    app.state.model_preheat_revision_resolver = (
        lambda source, model_id, revision, token=None: "c" * 40
    )
    with TestClient(app) as client:
        response = client.post(
            API_PREFIX,
            json=payload(profile.id, [workers[0].id], revision=None),
        )

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 200, response.text
    assert response.json()["requested_revision"] is None
    assert response.json()["resolved_revision"] == "c" * 40


def test_sqlite_enforces_model_preheat_unique_constraints(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    with TestClient(app) as client:
        created = client.post(
            API_PREFIX, json=payload(profile.id, [worker.id for worker in workers])
        )

    async def assert_unique_constraints():
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatTask, created.json()["id"])
            lock = (await session.exec(select(ModelPreheatTaskLock))).one()
            task_id = task.id
            parent_attempt = task.attempt
            operation_key = lock.operation_key
            profile_id = profile.id
            profile_config_version = profile.config_version
            connectivity_worker_uuid = workers[0].worker_uuid
            session.add(
                ModelPreheatWorkerTask(
                    task_id=task_id,
                    parent_attempt=parent_attempt,
                    worker_uuid=workers[0].worker_uuid,
                    worker_id=workers[0].id,
                    role=ModelPreheatWorkerTaskRoleEnum.SEED,
                )
            )
            await session.commit()
            worker_task = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.task_id == task_id
                    )
                )
            ).first()
            worker_uuid = worker_task.worker_uuid
            worker_role = worker_task.role

            record = ModelPreheatIdempotencyRecord(
                user_id=1,
                operation="model_preheats.create",
                idempotency_key="sqlite-idempotency-key",
                request_hash="request-hash",
                resource_type="model_preheat_task",
                resource_id=task_id,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
            session.add(record)
            await session.commit()

            session.add(
                ModelPreheatIdempotencyRecord(
                    user_id=1,
                    operation="model_preheats.create",
                    idempotency_key="sqlite-idempotency-key",
                    request_hash="another-request-hash",
                    resource_type="model_preheat_task",
                    resource_id=task_id,
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
            else:
                raise AssertionError(
                    "idempotency record unique constraint was not enforced"
                )

            session.add(
                ModelPreheatTaskLock(
                    operation_key=operation_key,
                    task_id=task_id,
                    lease_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
            else:
                raise AssertionError(
                    "operation lock unique constraint was not enforced"
                )

            session.add(
                ModelPreheatWorkerTask(
                    task_id=task_id,
                    parent_attempt=parent_attempt,
                    worker_uuid=worker_uuid,
                    role=worker_role,
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
            else:
                raise AssertionError(
                    "task worker role unique constraint was not enforced"
                )

            session.add(
                ModelPreheatWorkerTask(
                    task_id=task_id,
                    parent_attempt=parent_attempt + 1,
                    worker_uuid=worker_uuid,
                    role=worker_role,
                )
            )
            await session.commit()

            check = ModelPreheatS3ConnectivityCheck(
                profile_id=profile_id,
                profile_config_version=profile_config_version,
                target_worker_uuids=[connectivity_worker_uuid],
            )
            session.add(check)
            await session.commit()
            await session.refresh(check)
            check_id = check.id
            worker_task = ModelPreheatWorkerTask(
                connectivity_check_id=check_id,
                worker_uuid=connectivity_worker_uuid,
                role=ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
            )
            session.add(worker_task)
            await session.commit()

            session.add(
                ModelPreheatWorkerTask(
                    connectivity_check_id=check_id,
                    worker_uuid=connectivity_worker_uuid,
                    role=ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
            else:
                raise AssertionError(
                    "connectivity check worker role unique constraint was not enforced"
                )

    asyncio.run(assert_unique_constraints())
    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert created.status_code == 200, created.text


def test_schema_and_migration_use_regular_unique_constraints_for_postgresql():
    migration = Path(
        "gpustack/migrations/versions/2026_08_10_1000-f6a7b8c9d0e1_add_model_preheat_core.py"
    ).read_text()

    assert "postgresql_where" not in migration
    assert "CREATE UNIQUE INDEX" not in migration
    for table, name in (
        (ModelPreheatTaskLock.__table__, "uix_preheat_operation"),
        (ModelPreheatIdempotencyRecord.__table__, "uix_preheat_idempotency"),
        (
            ModelPreheatWorkerTask.__table__,
            "uix_preheat_task_attempt_worker_role",
        ),
        (ModelPreheatWorkerTask.__table__, "uix_preheat_check_worker_role"),
    ):
        assert any(
            isinstance(constraint, UniqueConstraint) and constraint.name == name
            for constraint in table.constraints
        )


def test_parent_attempt_uses_successor_portable_migration():
    core = Path(
        "gpustack/migrations/versions/2026_08_10_1000-f6a7b8c9d0e1_add_model_preheat_core.py"
    ).read_text()
    successor = Path(
        "gpustack/migrations/versions/2026_08_11_1100-a7b8c9d0e1f2_add_preheat_parent_attempt.py"
    ).read_text()

    assert "paused_from_state" not in core
    assert "parent_attempt" not in core
    assert 'down_revision: Union[str, None] = "f6a7b8c9d0e1"' in successor
    assert 'batch_alter_table("model_preheat_tasks")' in successor
    assert 'batch_alter_table("model_preheat_worker_tasks")' in successor
    assert "UPDATE model_preheat_worker_tasks" in successor
    assert "postgresql_where" not in successor


def test_postgresql_enforces_model_preheat_unique_constraints():
    postgres_url = os.getenv("GPUSTACK_TEST_POSTGRES_URL") or os.getenv(
        "TEST_POSTGRES_URL"
    )
    if not postgres_url:
        pytest.skip("未设置 GPUSTACK_TEST_POSTGRES_URL 或 TEST_POSTGRES_URL")
    if postgres_url.startswith("postgresql://"):
        postgres_url = postgres_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    async def assert_unique_constraints():
        schema = f"model_preheat_constraints_{uuid4().hex}"
        engine = create_async_engine(postgres_url, poolclass=NullPool)
        tables = [
            User.__table__,
            ModelPreheatS3Profile.__table__,
            ModelPreheatTask.__table__,
            ModelPreheatS3ConnectivityCheck.__table__,
            ModelPreheatWorkerTask.__table__,
            ModelPreheatIdempotencyRecord.__table__,
            ModelPreheatTaskLock.__table__,
        ]
        translated_engine = engine.execution_options(
            schema_translate_map={None: schema}
        )
        try:
            async with engine.begin() as connection:
                await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
                translated_connection = connection.execution_options(
                    schema_translate_map={None: schema}
                )
                await translated_connection.run_sync(
                    lambda sync_connection: SQLModel.metadata.create_all(
                        sync_connection, tables=tables
                    )
                )

            async with AsyncSession(translated_engine) as session:
                user = User(
                    username="postgres-admin", is_admin=True, hashed_password="hashed"
                )
                profile = ModelPreheatS3Profile(
                    name="postgres-profile",
                    endpoint="https://s3.example.com",
                    bucket="models",
                    access_key_encrypted={"ciphertext": "encrypted"},
                    secret_key_encrypted={"ciphertext": "encrypted"},
                    encryption_key_version="v1",
                )
                session.add_all([user, profile])
                await session.flush()
                task = ModelPreheatTask(
                    source="modelscope",
                    model_id="Qwen/Qwen-Image-2512",
                    resolved_revision="commit-123",
                    include_patterns=[],
                    exclude_patterns=[],
                    selection_digest="digest",
                    cache_key="cache-key",
                    generation_id="generation",
                    target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                    target_worker_uuids=["worker-uuid"],
                    target_worker_snapshot=[],
                    s3_profile_id=profile.id,
                    s3_profile_config_version=profile.config_version,
                    s3_profile_snapshot_encrypted={"ciphertext": "encrypted"},
                    encryption_key_version="v1",
                    s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
                )
                check = ModelPreheatS3ConnectivityCheck(
                    profile_id=profile.id,
                    profile_config_version=profile.config_version,
                    target_worker_uuids=["worker-uuid"],
                )
                session.add_all([task, check])
                await session.flush()
                task_id = task.id
                check_id = check.id
                session.add_all(
                    [
                        ModelPreheatTaskLock(
                            operation_key="operation-key",
                            task_id=task_id,
                            lease_expires_at=datetime.now(timezone.utc)
                            + timedelta(hours=1),
                        ),
                        ModelPreheatIdempotencyRecord(
                            user_id=user.id,
                            operation="model_preheats.create",
                            idempotency_key="postgres-idempotency-key",
                            request_hash="request-hash",
                            resource_type="model_preheat_task",
                            resource_id=task_id,
                            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                        ),
                        ModelPreheatWorkerTask(
                            task_id=task_id,
                            worker_uuid="worker-uuid",
                            role=ModelPreheatWorkerTaskRoleEnum.SEED,
                        ),
                        ModelPreheatWorkerTask(
                            connectivity_check_id=check_id,
                            worker_uuid="worker-uuid",
                            role=ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
                        ),
                    ]
                )
                await session.commit()

                duplicate_rows = [
                    ModelPreheatTaskLock(
                        operation_key="operation-key",
                        task_id=task_id,
                        lease_expires_at=datetime.now(timezone.utc)
                        + timedelta(hours=1),
                    ),
                    ModelPreheatIdempotencyRecord(
                        user_id=user.id,
                        operation="model_preheats.create",
                        idempotency_key="postgres-idempotency-key",
                        request_hash="another-request-hash",
                        resource_type="model_preheat_task",
                        resource_id=task_id,
                        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                    ),
                    ModelPreheatWorkerTask(
                        task_id=task_id,
                        worker_uuid="worker-uuid",
                        role=ModelPreheatWorkerTaskRoleEnum.SEED,
                    ),
                    ModelPreheatWorkerTask(
                        connectivity_check_id=check_id,
                        worker_uuid="worker-uuid",
                        role=ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
                    ),
                ]
                for duplicate in duplicate_rows:
                    session.add(duplicate)
                    with pytest.raises(IntegrityError):
                        await session.commit()
                    await session.rollback()
        finally:
            async with engine.begin() as connection:
                await connection.execute(
                    text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                )
            await engine.dispose()

    asyncio.run(assert_unique_constraints())


def test_task_target_snapshot_does_not_change_when_worker_is_deleted(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    with TestClient(app) as client:
        created = client.post(
            API_PREFIX, json=payload(profile.id, [worker.id for worker in workers])
        )

        async def delete_worker():
            async with AsyncSession(engine) as session:
                worker = await session.get(Worker, workers[0].id)
                await session.delete(worker)
                await session.commit()

        asyncio.run(delete_worker())
        detail = client.get(f"{API_PREFIX}/{created.json()['id']}")

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert created.status_code == 200, created.text
    assert detail.status_code == 200, detail.text
    assert detail.json()["target_worker_uuids"] == ["a-uuid", "z-uuid"]
    assert detail.json()["target_worker_snapshot"] == [
        {
            "worker_uuid": "a-uuid",
            "worker_id": workers[1].id,
            "worker_name": "worker-a",
        },
        {
            "worker_uuid": "z-uuid",
            "worker_id": workers[0].id,
            "worker_name": "worker-z",
        },
    ]


def test_creation_rejects_profile_that_is_not_available_on_all_workers(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session, ModelPreheatS3ConnectivityStateEnum.PARTIAL)

    profile, workers = asyncio.run(seed())
    with TestClient(app) as client:
        response = client.post(API_PREFIX, json=payload(profile.id, [workers[0].id]))

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 422
    assert response.json()["message"] == "s3_unavailable_on_workers"


def test_creation_starts_and_reuses_targeted_check_for_expired_profile(tmp_path):
    app, engine = _test_app(tmp_path)
    app.state.server_config.model_preheat_connectivity_ttl_seconds = 30

    async def seed():
        async with AsyncSession(engine) as session:
            profile, workers = await _seed(session)
            expired_at = datetime.now(timezone.utc) - timedelta(minutes=2)
            check = await session.get(
                ModelPreheatS3ConnectivityCheck,
                profile.last_connectivity_check_id,
            )
            check.finished_at = expired_at
            profile.last_connectivity_checked_at = expired_at
            profile_id = profile.id
            worker_id = workers[0].id
            worker_uuid = workers[0].worker_uuid
            session.add_all([check, profile])
            await session.commit()
            return profile_id, worker_id, worker_uuid

    profile_id, worker_id, worker_uuid = asyncio.run(seed())
    request_payload = payload(profile_id, [worker_id])
    with TestClient(app) as client:
        first = client.post(API_PREFIX, json=request_payload)
        second = client.post(API_PREFIX, json=request_payload)

    async def stored_checks_and_tasks():
        async with AsyncSession(engine) as session:
            checks = (
                await session.exec(
                    select(ModelPreheatS3ConnectivityCheck).order_by(
                        ModelPreheatS3ConnectivityCheck.id
                    )
                )
            ).all()
            tasks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.connectivity_check_id == checks[-1].id
                    )
                )
            ).all()
            return checks, tasks

    checks, tasks = asyncio.run(stored_checks_and_tasks())
    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert first.status_code == 422, first.text
    assert second.status_code == 422, second.text
    assert first.json()["message"] == second.json()["message"]
    assert first.json()["message"].startswith(
        "s3_unavailable_on_workers: connectivity_check_id="
    )
    quick_check_id = int(first.json()["message"].rsplit("=", 1)[1])
    assert len(checks) == 2
    assert checks[-1].id == quick_check_id
    assert checks[-1].target_worker_uuids == [worker_uuid]
    assert [(task.worker_uuid, task.worker_id) for task in tasks] == [
        (worker_uuid, worker_id)
    ]


def test_creation_reuses_fresh_target_result_after_ttl_quick_check(tmp_path):
    app, engine = _test_app(tmp_path)
    app.state.server_config.model_preheat_connectivity_ttl_seconds = 30

    async def seed():
        async with AsyncSession(engine) as session:
            profile, workers = await _seed(session)
            expired_at = datetime.now(timezone.utc) - timedelta(minutes=2)
            check = await session.get(
                ModelPreheatS3ConnectivityCheck,
                profile.last_connectivity_check_id,
            )
            check.finished_at = expired_at
            profile.last_connectivity_checked_at = expired_at
            profile_id = profile.id
            worker_id = workers[0].id
            session.add_all([check, profile])
            await session.commit()
            return profile_id, worker_id

    profile_id, worker_id = asyncio.run(seed())
    request_payload = payload(profile_id, [worker_id])
    with TestClient(app) as client:
        blocked = client.post(API_PREFIX, json=request_payload)
        quick_check_id = int(blocked.json()["message"].rsplit("=", 1)[1])

        async def finish_quick_check():
            async with AsyncSession(engine) as session:
                task = (
                    await session.exec(
                        select(ModelPreheatWorkerTask).where(
                            ModelPreheatWorkerTask.connectivity_check_id
                            == quick_check_id
                        )
                    )
                ).one()
                task.state = ModelPreheatWorkerTaskStateEnum.READY
                session.add(task)
                await session.commit()
                await aggregate_connectivity_check(session, quick_check_id)

        asyncio.run(finish_quick_check())
        retried = client.post(API_PREFIX, json=request_payload)

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert blocked.status_code == 422, blocked.text
    assert retried.status_code == 200, retried.text


def test_creation_uses_latest_results_across_incremental_checks(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed_incremental_result():
        async with AsyncSession(engine) as session:
            profile, workers = await _seed(session)
            checked_at = datetime.now(timezone.utc)
            incremental_check = ModelPreheatS3ConnectivityCheck(
                profile_id=profile.id,
                profile_config_version=profile.config_version,
                state=ModelPreheatConnectivityCheckStateEnum.AVAILABLE,
                target_worker_uuids=[workers[1].worker_uuid],
                finished_at=checked_at,
            )
            session.add(incremental_check)
            await session.flush()
            session.add(
                ModelPreheatWorkerTask(
                    connectivity_check_id=incremental_check.id,
                    worker_uuid=workers[1].worker_uuid,
                    worker_id=workers[1].id,
                    role=ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
                    state=ModelPreheatWorkerTaskStateEnum.READY,
                )
            )
            profile.last_connectivity_check_id = incremental_check.id
            profile.last_connectivity_checked_at = checked_at
            profile_id = profile.id
            target_worker_id = workers[0].id
            session.add(profile)
            await session.commit()
            return profile_id, target_worker_id

    profile_id, target_worker_id = asyncio.run(seed_incremental_result())
    with TestClient(app) as client:
        response = client.post(API_PREFIX, json=payload(profile_id, [target_worker_id]))

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 200, response.text


def test_creation_rejects_old_ready_registration_after_same_uuid_reregisters(
    tmp_path,
):
    app, engine = _test_app(tmp_path)

    async def seed_reregistered_worker():
        async with AsyncSession(engine) as session:
            profile, workers = await _seed(session)
            old_worker = workers[0]
            profile_id = profile.id
            old_worker_id = old_worker.id
            session.add(
                Worker(
                    name="worker-z-new",
                    hostname="worker-z-new",
                    ip="127.0.0.9",
                    port=10150,
                    worker_uuid=old_worker.worker_uuid,
                    state=WorkerStateEnum.NOT_READY,
                )
            )
            await session.commit()
            return profile_id, old_worker_id

    profile_id, old_worker_id = asyncio.run(seed_reregistered_worker())
    with TestClient(app) as client:
        response = client.post(API_PREFIX, json=payload(profile_id, [old_worker_id]))

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 422, response.text
    assert response.json()["message"] == "target_workers_not_online"


def test_creation_rejects_when_no_workers_are_online(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            profile, workers = await _seed(session)
            for worker in workers:
                worker.state = WorkerStateEnum.NOT_READY
                session.add(worker)
            profile_id = profile.id
            seed_worker_id = workers[0].id
            await session.commit()
            return profile_id, seed_worker_id

    profile_id, seed_worker_id = asyncio.run(seed())
    with TestClient(app) as client:
        response = client.post(
            API_PREFIX,
            json=payload(
                profile_id,
                [],
                target_scope="seed_worker",
                seed_worker_id=seed_worker_id,
            ),
        )

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 422
    assert response.json()["message"] == "no_online_workers"


def test_creation_rejects_selected_worker_that_is_not_online(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            profile, workers = await _seed(session)
            profile_id = profile.id
            offline_worker_id = workers[0].id
            workers[0].state = WorkerStateEnum.NOT_READY
            session.add(workers[0])
            await session.commit()
            return profile_id, offline_worker_id

    profile_id, offline_worker_id = asyncio.run(seed())
    with TestClient(app) as client:
        response = client.post(
            API_PREFIX, json=payload(profile_id, [offline_worker_id])
        )

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 422
    assert response.json()["message"] == "target_workers_not_online"


def test_creation_rejects_check_that_does_not_cover_new_online_worker(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            profile, workers = await _seed(session)
            profile_id = profile.id
            worker_ids = [worker.id for worker in workers]
            new_worker = Worker(
                name="worker-new",
                hostname="worker-new",
                ip="127.0.0.3",
                port=10150,
                worker_uuid="new-uuid",
                state=WorkerStateEnum.READY,
            )
            session.add(new_worker)
            await session.commit()
            return profile_id, worker_ids

    profile_id, worker_ids = asyncio.run(seed())
    with TestClient(app) as client:
        response = client.post(API_PREFIX, json=payload(profile_id, [worker_ids[0]]))

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 422
    assert response.json()["message"] == "s3_unavailable_on_workers"


def test_creation_rejects_check_worker_without_ready_result(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            profile, workers = await _seed(session)
            profile_id = profile.id
            worker_id = workers[0].id
            worker_uuid = workers[0].worker_uuid
            check_task = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.connectivity_check_id
                        == profile.last_connectivity_check_id,
                        ModelPreheatWorkerTask.worker_uuid == worker_uuid,
                    )
                )
            ).one()
            check_task.state = ModelPreheatWorkerTaskStateEnum.ERROR
            session.add(check_task)
            await session.commit()
            return profile_id, worker_id

    profile_id, worker_id = asyncio.run(seed())
    with TestClient(app) as client:
        response = client.post(API_PREFIX, json=payload(profile_id, [worker_id]))

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 422
    assert response.json()["message"] == "s3_unavailable_on_workers"


def test_creation_rejects_available_profile_without_connectivity_check(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            profile, workers = await _seed(session)
            profile_id = profile.id
            worker_id = workers[0].id
            profile.last_connectivity_check_id = None
            session.add(profile)
            await session.commit()
            return profile_id, worker_id

    profile_id, worker_id = asyncio.run(seed())
    with TestClient(app) as client:
        response = client.post(API_PREFIX, json=payload(profile_id, [worker_id]))

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 422
    assert response.json()["message"] == "s3_unavailable_on_workers"


def test_creation_rejects_connectivity_check_for_old_profile_version(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            profile, workers = await _seed(session)
            profile_id = profile.id
            worker_id = workers[0].id
            profile.config_version += 1
            session.add(profile)
            await session.commit()
            return profile_id, worker_id

    profile_id, worker_id = asyncio.run(seed())
    with TestClient(app) as client:
        response = client.post(API_PREFIX, json=payload(profile_id, [worker_id]))

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 422
    assert response.json()["message"] == "s3_unavailable_on_workers"


def test_get_and_list_return_public_tasks_without_profile_snapshot(tmp_path):
    app, engine = _test_app(tmp_path)

    async def seed():
        async with AsyncSession(engine) as session:
            return await _seed(session)

    profile, workers = asyncio.run(seed())
    with TestClient(app) as client:
        created = client.post(
            API_PREFIX, json=payload(profile.id, [worker.id for worker in workers])
        )
        detail = client.get(f"{API_PREFIX}/{created.json()['id']}")
        listing = client.get(API_PREFIX)

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert created.status_code == 200, created.text
    assert detail.status_code == 200, detail.text
    assert listing.status_code == 200, listing.text
    assert listing.json()["pagination"]["total"] == 1
    assert "s3_profile_snapshot_encrypted" not in detail.json()
    assert "encryption_key_version" not in detail.json()
