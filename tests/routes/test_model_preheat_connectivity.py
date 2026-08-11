import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
import httpx
from sqlalchemy import func
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.auth import get_admin_user
from gpustack.model_preheat_credentials import generate_model_preheat_credential_key
from gpustack.routes import model_preheat_s3_profiles
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3ConnectivityStateEnum,
    ModelPreheatS3Profile,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatS3ConnectivityCheck,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_session
from gpustack.server.model_preheat_connectivity import (
    aggregate_connectivity_check,
    create_or_reuse_connectivity_check,
)


def test_profile_create_and_manual_check_are_idempotent(tmp_path):
    async def tables(engine, action):
        async with engine.begin() as connection:
            await connection.run_sync(action)

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'routes.db'}", poolclass=NullPool
    )
    asyncio.run(tables(engine, SQLModel.metadata.create_all))
    app = FastAPI()
    app.state.server_config = SimpleNamespace(
        model_preheat_credential_key=generate_model_preheat_credential_key(),
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
    router.include_router(
        model_preheat_s3_profiles.router, prefix="/model-preheat-s3-profiles"
    )
    app.include_router(router, prefix="/v1")
    exceptions.register_handlers(app)

    async def seed_worker():
        async with AsyncSession(engine) as session:
            session.add(
                Worker(
                    name="embedded",
                    hostname="embedded",
                    ip="127.0.0.1",
                    port=10150,
                    worker_uuid="embedded-uuid",
                    state=WorkerStateEnum.READY,
                )
            )
            await session.commit()

    asyncio.run(seed_worker())
    payload = {
        "name": "profile",
        "endpoint": "https://s3.example.com",
        "bucket": "models",
        "access_key": "plain-access-key",
        "secret_key": "plain-secret-key",
    }
    try:
        with TestClient(app) as client:
            created = client.post("/v1/model-preheat-s3-profiles", json=payload)
            assert created.status_code == 200, created.text
            profile = created.json()
            assert profile["connectivity_state"] == "checking"
            assert profile["last_connectivity_check_id"] is not None

            url = f"/v1/model-preheat-s3-profiles/{profile['id']}/connectivity-checks"
            first = client.post(url, headers={"Idempotency-Key": "retry-key"})
            second = client.post(url, headers={"Idempotency-Key": "retry-key"})
            assert first.status_code == 200
            assert second.status_code == 200
            assert first.json()["id"] == second.json()["id"]
            original_check_id = profile["last_connectivity_check_id"]

            unchanged = client.patch(
                f"/v1/model-preheat-s3-profiles/{profile['id']}",
                json={"description": "renamed"},
            )
            assert unchanged.status_code == 200
            assert unchanged.json()["last_connectivity_check_id"] == original_check_id

            updated = client.patch(
                f"/v1/model-preheat-s3-profiles/{profile['id']}",
                json={"endpoint": "https://s3-updated.example.com"},
            )
            assert updated.status_code == 200
            assert updated.json()["connectivity_state"] == "checking"
            assert updated.json()["last_connectivity_check_id"] != original_check_id
            current_check_id = updated.json()["last_connectivity_check_id"]

            conflict = client.post(url, headers={"Idempotency-Key": "retry-key"})
            assert conflict.status_code == 409
            assert conflict.json()["message"] == "idempotency_key_reused"
            detail = client.get(f"{url}/{first.json()['id']}")
            assert detail.status_code == 200
            assert detail.json()["workers"][0]["worker_uuid"] == "embedded-uuid"
            assert "plain-access-key" not in detail.text
            assert "plain-secret-key" not in detail.text

            async def expire_connectivity_result():
                async with AsyncSession(engine) as session:
                    check = await session.get(
                        ModelPreheatS3ConnectivityCheck, current_check_id
                    )
                    profile_record = await session.get(
                        ModelPreheatS3Profile, profile["id"]
                    )
                    task = (
                        await session.exec(
                            select(ModelPreheatWorkerTask).where(
                                ModelPreheatWorkerTask.connectivity_check_id == check.id
                            )
                        )
                    ).one()
                    task.state = ModelPreheatWorkerTaskStateEnum.READY
                    expired_at = datetime.now(timezone.utc) - timedelta(minutes=11)
                    check.finished_at = expired_at
                    profile_record.connectivity_state = (
                        ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                    )
                    profile_record.last_connectivity_checked_at = expired_at
                    session.add_all([check, profile_record, task])
                    await session.commit()

            asyncio.run(expire_connectivity_result())
            refreshed_detail = client.get(f"{url}/{current_check_id}")
            assert refreshed_detail.status_code == 200

            async def stored_profile_state():
                async with AsyncSession(engine) as session:
                    stored = await session.get(ModelPreheatS3Profile, profile["id"])
                    return stored.connectivity_state

            assert (
                asyncio.run(stored_profile_state())
                == ModelPreheatS3ConnectivityStateEnum.STALE
            )
    finally:
        asyncio.run(tables(engine, SQLModel.metadata.drop_all))
        asyncio.run(engine.dispose())


def test_manual_check_idempotency_survives_fast_terminal_race(tmp_path, monkeypatch):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'fast-terminal-route.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        app = FastAPI()
        app.state.server_config = SimpleNamespace(
            model_preheat_credential_key=generate_model_preheat_credential_key(),
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
        router.include_router(
            model_preheat_s3_profiles.router,
            prefix="/model-preheat-s3-profiles",
        )
        app.include_router(router, prefix="/v1")
        exceptions.register_handlers(app)

        try:
            async with AsyncSession(engine) as session:
                session.add(
                    Worker(
                        name="embedded",
                        hostname="embedded",
                        ip="127.0.0.1",
                        port=10150,
                        worker_uuid="embedded-uuid",
                        state=WorkerStateEnum.READY,
                    )
                )
                await session.commit()

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                created = await client.post(
                    "/v1/model-preheat-s3-profiles",
                    json={
                        "name": "profile",
                        "endpoint": "https://s3.example.com",
                        "bucket": "models",
                        "access_key": "plain-access-key",
                        "secret_key": "plain-secret-key",
                    },
                )
                assert created.status_code == 200, created.text
                profile = created.json()

                original_create = (
                    model_preheat_s3_profiles.create_or_reuse_connectivity_check
                )
                first_terminal = asyncio.Event()
                both_created = asyncio.Event()
                invocation_lock = asyncio.Lock()
                invocation_count = 0
                created_count = 0

                async def create_then_finish(*args, **kwargs):
                    nonlocal invocation_count, created_count
                    async with invocation_lock:
                        invocation_count += 1
                        invocation = invocation_count
                    if invocation == 2:
                        await asyncio.wait_for(first_terminal.wait(), timeout=5)
                    check = await original_create(*args, **kwargs)
                    async with AsyncSession(engine) as terminal_session:
                        task = (
                            await terminal_session.exec(
                                select(ModelPreheatWorkerTask).where(
                                    ModelPreheatWorkerTask.connectivity_check_id
                                    == check.id
                                )
                            )
                        ).one()
                        task.state = ModelPreheatWorkerTaskStateEnum.READY
                        terminal_session.add(task)
                        check_id = check.id
                        await terminal_session.commit()
                        await aggregate_connectivity_check(terminal_session, check_id)
                    if invocation == 1:
                        first_terminal.set()
                    async with invocation_lock:
                        created_count += 1
                        if created_count == 2:
                            both_created.set()
                    await asyncio.wait_for(both_created.wait(), timeout=5)
                    return check

                monkeypatch.setattr(
                    model_preheat_s3_profiles,
                    "create_or_reuse_connectivity_check",
                    create_then_finish,
                )
                url = (
                    "/v1/model-preheat-s3-profiles/"
                    f"{profile['id']}/connectivity-checks"
                )
                first, second = await asyncio.gather(
                    client.post(url, headers={"Idempotency-Key": "fast-key"}),
                    client.post(url, headers={"Idempotency-Key": "fast-key"}),
                )

            assert first.status_code == 200, first.text
            assert second.status_code == 200, second.text
            assert first.json()["id"] == second.json()["id"]
            async with AsyncSession(engine) as session:
                check_count = await session.scalar(
                    select(func.count()).select_from(ModelPreheatS3ConnectivityCheck)
                )
                assert check_count == 1
        finally:
            async with engine.begin() as connection:
                await connection.run_sync(SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_reused_check_pointer_cas_miss_refreshes_profile_for_route_serialization(
    tmp_path,
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'pointer-cas-route.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as setup_session:
                profile = ModelPreheatS3Profile(
                    name="profile",
                    endpoint="https://s3.example.com",
                    bucket="models",
                    access_key_encrypted={"ciphertext": "encrypted"},
                    secret_key_encrypted={"ciphertext": "encrypted"},
                    encryption_key_version="v1",
                )
                worker = Worker(
                    name="worker",
                    hostname="worker",
                    ip="127.0.0.1",
                    port=10150,
                    worker_uuid="worker-uuid",
                    state=WorkerStateEnum.READY,
                )
                setup_session.add_all([profile, worker])
                await setup_session.flush()
                profile_id = profile.id
                await setup_session.commit()

            async with AsyncSession(engine) as stale_session:
                stale_profile = await stale_session.get(
                    ModelPreheatS3Profile, profile_id
                )
                assert stale_profile.last_connectivity_check_id is None

                async with AsyncSession(engine) as winner_session:
                    winner_profile = await winner_session.get(
                        ModelPreheatS3Profile, profile_id
                    )
                    winner = await create_or_reuse_connectivity_check(
                        winner_session, winner_profile
                    )

                reused = await create_or_reuse_connectivity_check(
                    stale_session, stale_profile
                )
                public = model_preheat_s3_profiles._to_public(stale_profile)

                assert reused.id == winner.id
                assert public.last_connectivity_check_id == winner.id
        finally:
            async with engine.begin() as connection:
                await connection.run_sync(SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_profile_patch_races_old_connectivity_matrix_aggregation(tmp_path, monkeypatch):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'profile-patch-race.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        app = FastAPI()
        app.state.server_config = SimpleNamespace(
            model_preheat_credential_key=generate_model_preheat_credential_key(),
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
        router.include_router(
            model_preheat_s3_profiles.router,
            prefix="/model-preheat-s3-profiles",
        )
        app.include_router(router, prefix="/v1")
        exceptions.register_handlers(app)

        try:
            async with AsyncSession(engine) as session:
                session.add(
                    Worker(
                        name="worker",
                        hostname="worker",
                        ip="127.0.0.1",
                        port=10150,
                        worker_uuid="worker-uuid",
                        state=WorkerStateEnum.READY,
                    )
                )
                await session.commit()

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                created = await client.post(
                    "/v1/model-preheat-s3-profiles",
                    json={
                        "name": "profile",
                        "endpoint": "https://s3.example.com",
                        "bucket": "models",
                        "access_key": "plain-access-key",
                        "secret_key": "plain-secret-key",
                    },
                )
                assert created.status_code == 200, created.text
                profile = created.json()
                old_check_id = profile["last_connectivity_check_id"]

                async with AsyncSession(engine) as session:
                    old_task = (
                        await session.exec(
                            select(ModelPreheatWorkerTask).where(
                                ModelPreheatWorkerTask.connectivity_check_id
                                == old_check_id
                            )
                        )
                    ).one()
                    old_task.state = ModelPreheatWorkerTaskStateEnum.READY
                    old_task.resumable_cursor = {
                        "state": "ready",
                        "readable": True,
                        "writable": True,
                        "deletable": True,
                        "cleanup_failed": False,
                        "latency_ms": 12,
                    }
                    session.add(old_task)
                    await session.commit()

                aggregation_started = asyncio.Event()
                release_aggregation = asyncio.Event()
                original_aggregate = (
                    model_preheat_s3_profiles.aggregate_connectivity_check
                )

                async def aggregate_with_barrier(session, check_id):
                    aggregation_started.set()
                    await asyncio.wait_for(release_aggregation.wait(), timeout=5)
                    return await original_aggregate(session, check_id)

                monkeypatch.setattr(
                    model_preheat_s3_profiles,
                    "aggregate_connectivity_check",
                    aggregate_with_barrier,
                )
                old_detail_request = asyncio.create_task(
                    client.get(
                        "/v1/model-preheat-s3-profiles/"
                        f"{profile['id']}/connectivity-checks/{old_check_id}"
                    )
                )
                await asyncio.wait_for(aggregation_started.wait(), timeout=5)
                try:
                    patched = await client.patch(
                        f"/v1/model-preheat-s3-profiles/{profile['id']}",
                        json={"endpoint": "https://s3-new.example.com"},
                    )
                finally:
                    release_aggregation.set()
                old_detail = await old_detail_request
                current = await client.get(
                    f"/v1/model-preheat-s3-profiles/{profile['id']}"
                )

            assert patched.status_code == 200, patched.text
            assert patched.json()["config_version"] == 2
            assert patched.json()["connectivity_state"] == "checking"
            assert patched.json()["last_connectivity_check_id"] != old_check_id
            assert old_detail.status_code == 200, old_detail.text
            assert old_detail.json()["profile_config_version"] == 1
            assert old_detail.json()["state"] == "available"
            assert old_detail.json()["workers"][0]["readable"] is True
            assert current.status_code == 200, current.text
            assert current.json()["config_version"] == 2
            assert current.json()["connectivity_state"] == "checking"
            assert (
                current.json()["last_connectivity_check_id"]
                == patched.json()["last_connectivity_check_id"]
            )
        finally:
            async with engine.begin() as connection:
                await connection.run_sync(SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())
