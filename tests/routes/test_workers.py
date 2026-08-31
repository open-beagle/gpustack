import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import Response
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel
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
from gpustack.schemas.model_preheats import ModelPreheatWorkerIdentity
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
from gpustack.server.model_preheat_worker_identity import (
    get_model_preheat_worker_identity,
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
                    tables=[Worker.__table__, ModelPreheatWorkerIdentity.__table__],
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
            )
            result = worker.id, worker.worker_uuid
        await engine.dispose()
        return result, response.headers

    (worker_id, worker_uuid), headers = asyncio.run(run())
    assert worker_id is not None
    assert worker_uuid == "new-worker-uuid"
    assert headers["X-GPUStack-Worker-Credential"].startswith("mpw_")
    assert headers["cache-control"] == "no-store"


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
                    tables=[Worker.__table__, ModelPreheatWorkerIdentity.__table__],
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
