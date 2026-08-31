import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import Response
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import ForbiddenException, UnauthorizedException

from gpustack.routes.workers import (
    _authorize_worker_registration,
    _issue_preheat_credential,
    bootstrap_model_preheat_worker_credential,
    create_worker,
    get_workers,
)
from gpustack.schemas.common import ListParams, PaginatedList, Pagination
from gpustack.schemas.model_preheats import (
    ModelPreheatWorkerIdentity,
    ModelPreheatWorkerPendingCredential,
)
from gpustack.schemas.workers import (
    CPUInfo,
    GPUCoreInfo,
    GPUDeviceInfo,
    MemoryInfo,
    Worker,
    WorkerCreate,
    WorkerStateEnum,
    WorkerStatus,
)
from gpustack.schemas.users import User
import gpustack.server.model_preheat_worker_identity as worker_credential_identity
from gpustack.server.model_preheat_worker_identity import (
    WORKER_CREDENTIAL_TTL,
    get_model_preheat_worker_identity,
    issue_model_preheat_worker_credential,
    validate_model_preheat_worker_credential,
    validate_model_preheat_worker_registration_credential,
)


def test_get_workers_clears_not_ready_allocated_resources():
    worker = Worker(
        id=1,
        name="worker-a",
        hostname="worker-a",
        ip="10.0.0.1",
        port=10150,
        state=WorkerStateEnum.NOT_READY,
        labels={},
        system_reserved=None,
        status=WorkerStatus(
            cpu=CPUInfo(total=16, utilization_rate=0),
            memory=MemoryInfo(total=1024, used=0, allocated=512),
            gpu_devices=[
                GPUDeviceInfo(
                    index=0,
                    name="GPU-0",
                    core=GPUCoreInfo(total=100, utilization_rate=0),
                    memory=MemoryInfo(total=1024, used=0, allocated=900),
                )
            ],
        ),
        unreachable=False,
        heartbeat_time=datetime.now(timezone.utc),
        worker_uuid="worker-a-uuid",
    )
    page = PaginatedList[Worker](
        items=[worker],
        pagination=Pagination(page=1, perPage=100, total=1, totalPage=1),
    )

    with patch(
        "gpustack.routes.workers.Worker.paginated_by_query",
        new=AsyncMock(return_value=page),
    ):
        result = asyncio.run(
            get_workers(
                engine=AsyncMock(),
                session=AsyncMock(),
                params=ListParams(page=1, perPage=100, watch=False),
            )
        )

    normalized = result.items[0]
    assert normalized.status.memory.allocated == 0
    assert normalized.status.gpu_devices[0].memory.allocated == 0


def test_worker_registration_credential_is_one_time_system_header_only():
    worker = SimpleNamespace(id=9, worker_uuid="worker-uuid")
    system_request = SimpleNamespace(
        state=SimpleNamespace(user=SimpleNamespace(username="system/worker/10.0.0.1"))
    )
    admin_request = SimpleNamespace(
        state=SimpleNamespace(user=SimpleNamespace(username="admin"))
    )
    system_response = Response()
    admin_response = Response()

    with patch(
        "gpustack.routes.workers.issue_model_preheat_worker_credential",
        new=AsyncMock(return_value="mpw_9_one-time-secret"),
    ) as issue:
        asyncio.run(
            _issue_preheat_credential(
                system_request, system_response, AsyncMock(), worker, True
            )
        )
        asyncio.run(
            _issue_preheat_credential(
                admin_request, admin_response, AsyncMock(), worker, True
            )
        )

    assert system_response.headers["X-GPUStack-Worker-Credential"].startswith("mpw_9_")
    assert system_response.headers["cache-control"] == "no-store"
    assert "X-GPUStack-Worker-Credential" not in admin_response.headers
    issue.assert_awaited_once()


def test_new_worker_creation_issues_credential_and_returns_loaded_worker(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'new-worker.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: SQLModel.metadata.create_all(
                    sync_connection,
                    tables=[
                        Worker.__table__,
                        ModelPreheatWorkerIdentity.__table__,
                        ModelPreheatWorkerPendingCredential.__table__,
                    ],
                )
            )
        request = SimpleNamespace(
            state=SimpleNamespace(
                user=SimpleNamespace(username="system/worker/10.0.0.4")
            )
        )
        response = Response()
        async with AsyncSession(engine) as session:
            worker = await create_worker(
                request=request,
                response=response,
                session=session,
                worker_in=WorkerCreate(
                    name="new-worker",
                    hostname="new-worker",
                    ip="127.0.0.1",
                    port=10150,
                    worker_uuid="new-worker-uuid",
                    system_reserved=None,
                    status=None,
                ),
                upgrade_proof="a" * 43,
            )
            result = worker.id, worker.worker_uuid
        await engine.dispose()
        return result, response.headers

    (worker_id, worker_uuid), headers = asyncio.run(run())
    assert worker_id is not None
    assert worker_uuid == "new-worker-uuid"
    assert headers["X-GPUStack-Worker-Credential"].startswith("mpw_")
    assert headers["cache-control"] == "no-store"


def test_new_worker_creation_response_loss_recovers_only_with_same_proof(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'new-worker-retry.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: SQLModel.metadata.create_all(
                    sync_connection,
                    tables=[
                        Worker.__table__,
                        ModelPreheatWorkerIdentity.__table__,
                        ModelPreheatWorkerPendingCredential.__table__,
                    ],
                )
            )
        request = SimpleNamespace(
            state=SimpleNamespace(
                user=SimpleNamespace(username="system/worker/10.0.0.4")
            )
        )
        worker_in = WorkerCreate(
            name="retry-worker",
            hostname="retry-worker",
            ip="127.0.0.1",
            port=10150,
            worker_uuid="retry-worker-uuid",
            system_reserved=None,
            status=None,
        )
        proof = "a" * 43
        async with AsyncSession(engine) as session:
            await create_worker(
                request=request,
                response=Response(),
                session=session,
                worker_in=worker_in,
                upgrade_proof=proof,
            )
            retry_response = Response()
            retry = await create_worker(
                request=request,
                response=retry_response,
                session=session,
                worker_in=worker_in,
                upgrade_proof=proof,
            )
            with pytest.raises(UnauthorizedException):
                await create_worker(
                    request=request,
                    response=Response(),
                    session=session,
                    worker_in=worker_in,
                    upgrade_proof="b" * 43,
                )
            credential = retry_response.headers["X-GPUStack-Worker-Credential"]
            retry_worker_uuid = retry.worker_uuid
            principal = await get_model_preheat_worker_identity(
                request=SimpleNamespace(state=SimpleNamespace()),
                session=session,
                credential=credential,
            )
        await engine.dispose()
        return retry_worker_uuid, principal

    worker_uuid, principal = asyncio.run(run())
    assert worker_uuid == "retry-worker-uuid"
    assert principal.worker_uuid == "retry-worker-uuid"


def test_confirmed_credential_renews_inside_window(tmp_path, monkeypatch):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'credential-renew.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: SQLModel.metadata.create_all(
                    sync_connection,
                    tables=[
                        Worker.__table__,
                        ModelPreheatWorkerIdentity.__table__,
                        ModelPreheatWorkerPendingCredential.__table__,
                    ],
                )
            )
        async with AsyncSession(engine) as session:
            worker = Worker(
                name="renew-worker",
                hostname="renew-worker",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="renew-worker-uuid",
            )
            session.add(worker)
            await session.commit()
            await session.refresh(worker)
            worker_id = worker.id
            worker_uuid = worker.worker_uuid
            token = await issue_model_preheat_worker_credential(
                session, worker_id, worker_uuid
            )
            now = datetime.now(timezone.utc)
            monkeypatch.setattr(worker_credential_identity, "_utcnow", lambda: now)
            await get_model_preheat_worker_identity(
                request=SimpleNamespace(state=SimpleNamespace()),
                session=session,
                credential=token,
            )
            identity = (
                await session.exec(
                    select(ModelPreheatWorkerIdentity).where(
                        ModelPreheatWorkerIdentity.worker_id == worker_id
                    )
                )
            ).one()
            identity.expires_at = (
                now
                + worker_credential_identity.WORKER_CREDENTIAL_RENEW_WINDOW
                - timedelta(seconds=1)
            )
            session.add(identity)
            await session.commit()
            await get_model_preheat_worker_identity(
                request=SimpleNamespace(state=SimpleNamespace()),
                session=session,
                credential=token,
            )
            await session.refresh(identity)
            return now, identity.expires_at
        await engine.dispose()

    now, expires_at = asyncio.run(run())
    assert expires_at == now + WORKER_CREDENTIAL_TTL


def test_concurrent_confirmed_credential_renewal_keeps_both_requests_valid(
    tmp_path, monkeypatch
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'credential-renew-race.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: SQLModel.metadata.create_all(
                    sync_connection,
                    tables=[
                        Worker.__table__,
                        ModelPreheatWorkerIdentity.__table__,
                        ModelPreheatWorkerPendingCredential.__table__,
                    ],
                )
            )
        async with AsyncSession(engine) as session:
            worker = Worker(
                name="renew-race-worker",
                hostname="renew-race-worker",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="renew-race-worker-uuid",
            )
            session.add(worker)
            await session.commit()
            await session.refresh(worker)
            worker_id, worker_uuid = worker.id, worker.worker_uuid
            token = await issue_model_preheat_worker_credential(
                session, worker_id, worker_uuid
            )
            await get_model_preheat_worker_identity(
                request=SimpleNamespace(state=SimpleNamespace()),
                session=session,
                credential=token,
            )
            identity = (
                await session.exec(
                    select(ModelPreheatWorkerIdentity).where(
                        ModelPreheatWorkerIdentity.worker_id == worker_id
                    )
                )
            ).one()
            now = datetime.now(timezone.utc)
            identity.expires_at = (
                now
                + worker_credential_identity.WORKER_CREDENTIAL_RENEW_WINDOW
                - timedelta(seconds=1)
            )
            session.add(identity)
            await session.commit()
        monkeypatch.setattr(worker_credential_identity, "_utcnow", lambda: now)
        start = asyncio.Event()

        async def renew():
            async with AsyncSession(engine) as session:
                await start.wait()
                principal = await get_model_preheat_worker_identity(
                    request=SimpleNamespace(state=SimpleNamespace()),
                    session=session,
                    credential=token,
                )
                return principal.worker_uuid

        first, second = asyncio.create_task(renew()), asyncio.create_task(renew())
        await asyncio.sleep(0)
        start.set()
        result = await asyncio.gather(first, second)
        await engine.dispose()
        return result

    assert asyncio.run(run()) == ["renew-race-worker-uuid", "renew-race-worker-uuid"]


def test_existing_worker_uuid_cannot_be_rebound_with_shared_token_only():
    request = SimpleNamespace(
        state=SimpleNamespace(user=SimpleNamespace(username="system/worker/10.0.0.2"))
    )
    with (
        patch(
            "gpustack.routes.workers.worker_uuid_has_credential",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "gpustack.routes.workers.validate_model_preheat_worker_registration_credential",
            new=AsyncMock(return_value=None),
        ),
    ):
        with pytest.raises(UnauthorizedException) as error:
            asyncio.run(
                _authorize_worker_registration(
                    request, AsyncMock(), "existing-uuid", None
                )
            )
    assert error.value.message == "Invalid worker registration credentials"


def test_initial_worker_credential_response_loss_recovers_via_admin_bootstrap(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'initial-worker-bootstrap.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: SQLModel.metadata.create_all(
                    sync_connection,
                    tables=[
                        Worker.__table__,
                        ModelPreheatWorkerIdentity.__table__,
                        ModelPreheatWorkerPendingCredential.__table__,
                    ],
                )
            )
        system_request = SimpleNamespace(
            state=SimpleNamespace(
                user=SimpleNamespace(username="system/worker/10.0.0.9")
            )
        )
        admin = User(
            id=1,
            username="admin",
            is_admin=True,
            hashed_password="unused",
        )
        async with AsyncSession(engine) as session:
            worker = Worker(
                name="initial-worker",
                hostname="initial-worker",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="initial-worker-uuid",
            )
            session.add(worker)
            await session.commit()
            await session.refresh(worker)
            worker_id = worker.id
            worker_uuid = worker.worker_uuid
            await issue_model_preheat_worker_credential(session, worker_id, worker_uuid)
            with pytest.raises(UnauthorizedException):
                await _authorize_worker_registration(
                    system_request, session, worker_uuid, None
                )
            bootstrap = await bootstrap_model_preheat_worker_credential(
                response=Response(),
                session=session,
                current_user=admin,
                id=worker_id,
            )
            recovered = await validate_model_preheat_worker_credential(
                session, bootstrap.credential, worker_uuid
            )
        await engine.dispose()
        return recovered

    assert asyncio.run(run()) is not None


def test_upgraded_worker_bootstrap_recovery_credential_only_allows_registration(
    tmp_path,
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'worker-bootstrap.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: SQLModel.metadata.create_all(
                    sync_connection,
                    tables=[
                        Worker.__table__,
                        ModelPreheatWorkerIdentity.__table__,
                        ModelPreheatWorkerPendingCredential.__table__,
                    ],
                )
            )
        system_request = SimpleNamespace(
            state=SimpleNamespace(
                user=SimpleNamespace(username="system/worker/10.0.0.2")
            )
        )
        admin = User(
            id=1,
            username="admin",
            is_admin=True,
            hashed_password="unused",
        )
        async with AsyncSession(engine) as session:
            worker = Worker(
                name="upgraded-worker",
                hostname="upgraded-worker",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="existing-uuid",
                state=WorkerStateEnum.READY,
            )
            session.add(worker)
            await session.commit()
            await session.refresh(worker)
            worker_id = worker.id
            worker_uuid = worker.worker_uuid

            with pytest.raises(UnauthorizedException):
                await _authorize_worker_registration(
                    system_request, session, worker_uuid, None
                )

            response = Response()
            bootstrap = await bootstrap_model_preheat_worker_credential(
                response=response,
                session=session,
                current_user=admin,
                id=worker_id,
            )
            assert response.headers["cache-control"] == "no-store"
            assert bootstrap.credential.startswith("mpw_")
            assert bootstrap.worker_id == worker_id
            await get_model_preheat_worker_identity(
                request=SimpleNamespace(state=SimpleNamespace()),
                session=session,
                credential=bootstrap.credential,
            )

            await _authorize_worker_registration(
                system_request,
                session,
                worker_uuid,
                bootstrap.credential,
            )
            await session.refresh(worker)
            registration_response = Response()
            await _issue_preheat_credential(
                system_request,
                registration_response,
                session,
                worker,
                True,
            )
            rotated = registration_response.headers["X-GPUStack-Worker-Credential"]
            old_identity = await validate_model_preheat_worker_credential(
                session, bootstrap.credential, worker_uuid
            )
            recovery_identity = (
                await validate_model_preheat_worker_registration_credential(
                    session, bootstrap.credential, worker_uuid
                )
            )
            new_identity = await validate_model_preheat_worker_credential(
                session, rotated, worker_uuid
            )
            # 丢失轮换响应后，旧凭据只能重试注册，不能访问 Worker 任务/payload。
            await _authorize_worker_registration(
                system_request, session, worker_uuid, bootstrap.credential
            )
            retry_response = Response()
            await _issue_preheat_credential(
                system_request,
                retry_response,
                session,
                worker,
                True,
            )
            retry_rotated = retry_response.headers["X-GPUStack-Worker-Credential"]
            recovery_after_repeated_loss = (
                await validate_model_preheat_worker_registration_credential(
                    session, bootstrap.credential, worker_uuid
                )
            )
            principal_request = SimpleNamespace(state=SimpleNamespace())
            await get_model_preheat_worker_identity(
                request=principal_request,
                session=session,
                credential=retry_rotated,
            )
            old_after_confirmation = (
                await validate_model_preheat_worker_registration_credential(
                    session, bootstrap.credential, worker_uuid
                )
            )
        await engine.dispose()
        return (
            old_identity,
            recovery_identity,
            new_identity,
            recovery_after_repeated_loss,
            old_after_confirmation,
            registration_response.headers,
        )

    (
        old,
        recovery,
        new,
        recovery_after_repeated_loss,
        old_after_confirmation,
        headers,
    ) = asyncio.run(run())
    assert old is None
    assert recovery is not None
    assert new is not None
    assert recovery_after_repeated_loss is not None
    assert old_after_confirmation is None
    assert headers["X-GPUStack-Worker-Credential"].startswith("mpw_")


def test_system_worker_cannot_call_admin_credential_bootstrap():
    system_user = User(
        username="system/worker/10.0.0.3",
        is_admin=True,
        hashed_password="unused",
    )
    with pytest.raises(ForbiddenException):
        asyncio.run(
            bootstrap_model_preheat_worker_credential(
                response=Response(),
                session=AsyncMock(),
                current_user=system_user,
                id=1,
            )
        )


def test_recovery_credential_expires_but_pending_candidates_remain_safe(
    tmp_path, monkeypatch
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'worker-recovery-ttl.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: SQLModel.metadata.create_all(
                    sync_connection,
                    tables=[
                        Worker.__table__,
                        ModelPreheatWorkerIdentity.__table__,
                        ModelPreheatWorkerPendingCredential.__table__,
                    ],
                )
            )
        async with AsyncSession(engine) as session:
            worker = Worker(
                name="ttl-worker",
                hostname="ttl-worker",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="ttl-worker-uuid",
            )
            session.add(worker)
            await session.commit()
            await session.refresh(worker)
            worker_id = worker.id
            worker_uuid = worker.worker_uuid
            first = await issue_model_preheat_worker_credential(
                session, worker_id, worker_uuid
            )
            await get_model_preheat_worker_identity(
                request=SimpleNamespace(state=SimpleNamespace()),
                session=session,
                credential=first,
            )
            second = await issue_model_preheat_worker_credential(
                session, worker_id, worker_uuid
            )
            return engine, worker_uuid, first, second

    engine, worker_uuid, first, second = asyncio.run(run())
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(
        "gpustack.server.model_preheat_worker_identity._utcnow",
        lambda: now + WORKER_CREDENTIAL_TTL - timedelta(seconds=1),
    )

    async def within_ttl():
        async with AsyncSession(engine) as session:
            recovery = await validate_model_preheat_worker_registration_credential(
                session, first, worker_uuid
            )
            normal = await validate_model_preheat_worker_credential(
                session, first, worker_uuid
            )
            return recovery, normal

    recovery, normal = asyncio.run(within_ttl())
    assert recovery is not None
    assert normal is None
    monkeypatch.setattr(
        "gpustack.server.model_preheat_worker_identity._utcnow",
        lambda: now + timedelta(days=30),
    )

    async def expired():
        async with AsyncSession(engine) as session:
            recovery = await validate_model_preheat_worker_registration_credential(
                session, first, worker_uuid
            )
            pending = await validate_model_preheat_worker_credential(
                session, second, worker_uuid
            )
            return recovery, pending

    recovery, pending = asyncio.run(expired())
    assert recovery is None
    assert pending is None
    asyncio.run(engine.dispose())


def test_concurrent_rotation_candidates_confirm_once(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'worker-candidates.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: SQLModel.metadata.create_all(
                    sync_connection,
                    tables=[
                        Worker.__table__,
                        ModelPreheatWorkerIdentity.__table__,
                        ModelPreheatWorkerPendingCredential.__table__,
                    ],
                )
            )
        async with AsyncSession(engine) as setup_session:
            worker = Worker(
                name="candidate-worker",
                hostname="candidate-worker",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="candidate-worker-uuid",
            )
            setup_session.add(worker)
            await setup_session.commit()
            await setup_session.refresh(worker)
            worker_id = worker.id
            worker_uuid = worker.worker_uuid
        async with AsyncSession(engine) as first_session:
            first = await issue_model_preheat_worker_credential(
                first_session, worker_id, worker_uuid
            )
        async with AsyncSession(engine) as second_session:
            second = await issue_model_preheat_worker_credential(
                second_session, worker_id, worker_uuid
            )
        async with AsyncSession(engine) as validation_session:
            first_pending = await validate_model_preheat_worker_credential(
                validation_session, first, worker_uuid
            )
            second_pending = await validate_model_preheat_worker_credential(
                validation_session, second, worker_uuid
            )
            request = SimpleNamespace(state=SimpleNamespace())
            await get_model_preheat_worker_identity(
                request=request,
                session=validation_session,
                credential=second,
            )
            first_after_confirmation = await validate_model_preheat_worker_credential(
                validation_session, first, worker_uuid
            )
        await engine.dispose()
        return first_pending, second_pending, first_after_confirmation

    first_pending, second_pending, first_after_confirmation = asyncio.run(run())
    assert first_pending is not None
    assert second_pending is not None
    assert first_after_confirmation is None


def test_stale_pending_confirmation_cannot_replace_new_generation(
    tmp_path, monkeypatch
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'worker-stale-candidate.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: SQLModel.metadata.create_all(
                    sync_connection,
                    tables=[
                        Worker.__table__,
                        ModelPreheatWorkerIdentity.__table__,
                        ModelPreheatWorkerPendingCredential.__table__,
                    ],
                )
            )
        async with AsyncSession(engine) as setup_session:
            worker = Worker(
                name="stale-candidate-worker",
                hostname="stale-candidate-worker",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="stale-candidate-worker-uuid",
            )
            setup_session.add(worker)
            await setup_session.commit()
            await setup_session.refresh(worker)
            worker_id = worker.id
            worker_uuid = worker.worker_uuid
            confirmed = await issue_model_preheat_worker_credential(
                setup_session, worker_id, worker_uuid
            )
            await get_model_preheat_worker_identity(
                request=SimpleNamespace(state=SimpleNamespace()),
                session=setup_session,
                credential=confirmed,
            )
            stale = await issue_model_preheat_worker_credential(
                setup_session, worker_id, worker_uuid
            )

        pending_loaded = asyncio.Event()
        continue_confirmation = asyncio.Event()
        original_pending_credential = worker_credential_identity._pending_credential

        async def pause_after_pending_read(session, credential_id, credential, now):
            pending = await original_pending_credential(
                session, credential_id, credential, now
            )
            if credential == stale:
                # SQLite 的读事务会阻塞独立 session 提交；真实服务在此边界
                # 可被其他 Server 更新，因此释放本测试会话的读取快照后再继续。
                session.expunge(pending)
                await session.rollback()
                pending_loaded.set()
                await continue_confirmation.wait()
            return pending

        monkeypatch.setattr(
            worker_credential_identity,
            "_pending_credential",
            pause_after_pending_read,
        )
        async with AsyncSession(engine) as stale_session:
            stale_confirmation = asyncio.create_task(
                get_model_preheat_worker_identity(
                    request=SimpleNamespace(state=SimpleNamespace()),
                    session=stale_session,
                    credential=stale,
                )
            )
            await pending_loaded.wait()
            async with AsyncSession(engine) as issue_session:
                current = await issue_model_preheat_worker_credential(
                    issue_session, worker_id, worker_uuid
                )
            async with AsyncSession(engine) as observer_session:
                recovery_before_stale_confirmation = (
                    await validate_model_preheat_worker_registration_credential(
                        observer_session, confirmed, worker_uuid
                    )
                )
                current_before_stale_confirmation = (
                    await validate_model_preheat_worker_credential(
                        observer_session, current, worker_uuid
                    )
                )
            continue_confirmation.set()
            with pytest.raises(UnauthorizedException):
                await stale_confirmation

        async with AsyncSession(engine) as validation_session:
            current_after_stale_confirmation = (
                await validate_model_preheat_worker_credential(
                    validation_session, current, worker_uuid
                )
            )
            recovery_after_stale_confirmation = (
                await validate_model_preheat_worker_registration_credential(
                    validation_session, confirmed, worker_uuid
                )
            )
            await get_model_preheat_worker_identity(
                request=SimpleNamespace(state=SimpleNamespace()),
                session=validation_session,
                credential=current,
            )
            before_admin_reset = await issue_model_preheat_worker_credential(
                validation_session, worker_id, worker_uuid
            )
            after_admin_reset = await issue_model_preheat_worker_credential(
                validation_session,
                worker_id,
                worker_uuid,
                reset_pending=True,
            )
            with pytest.raises(UnauthorizedException):
                await get_model_preheat_worker_identity(
                    request=SimpleNamespace(state=SimpleNamespace()),
                    session=validation_session,
                    credential=before_admin_reset,
                )
            reset_principal = await get_model_preheat_worker_identity(
                request=SimpleNamespace(state=SimpleNamespace()),
                session=validation_session,
                credential=after_admin_reset,
            )
        await engine.dispose()
        return (
            recovery_before_stale_confirmation,
            current_before_stale_confirmation,
            current_after_stale_confirmation,
            recovery_after_stale_confirmation,
            reset_principal,
        )

    (
        recovery_before_stale_confirmation,
        current_before_stale_confirmation,
        current_after_stale_confirmation,
        recovery_after_stale_confirmation,
        reset_principal,
    ) = asyncio.run(run())
    assert recovery_before_stale_confirmation is not None
    assert current_before_stale_confirmation is not None
    assert current_after_stale_confirmation is not None
    assert recovery_after_stale_confirmation is not None
    assert reset_principal is not None
