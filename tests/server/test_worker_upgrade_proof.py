import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import Response
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import UnauthorizedException
from gpustack.routes.workers import _authorize_worker_registration, update_worker
from gpustack.schemas.model_preheats import (
    ModelPreheatWorkerIdentity,
    ModelPreheatWorkerPendingCredential,
)
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker, WorkerUpdate
from gpustack.server.model_preheat_worker_identity import (
    WORKER_UPGRADE_PROOF_TTL,
    get_model_preheat_worker_identity,
    issue_model_preheat_worker_credential,
    validate_model_preheat_worker_credential,
)


def _system_request():
    return SimpleNamespace(
        state=SimpleNamespace(
            user=User(
                username="system/worker/10.0.0.8",
                is_admin=False,
                hashed_password="unused",
            )
        )
    )


def _proof(seed: str) -> str:
    return (seed * 43)[:43]


async def _create_worker_identity(session, *, bootstrap_required=True, window=None):
    worker = Worker(
        name="legacy-worker",
        hostname="legacy-host",
        ip="10.0.0.8",
        port=10150,
        worker_uuid="legacy-worker-uuid",
    )
    session.add(worker)
    await session.flush()
    identity = ModelPreheatWorkerIdentity(
        worker_id=worker.id,
        worker_uuid=worker.worker_uuid,
        bootstrap_required=bootstrap_required,
        upgrade_proof_window_started_at=(
            window if window is not None else datetime.now(timezone.utc)
        ),
    )
    session.add(identity)
    worker_id = worker.id
    worker_uuid = worker.worker_uuid
    await session.commit()
    return worker_id, worker_uuid


async def _worker_update(session, worker_id):
    worker = await session.get(Worker, worker_id)
    return worker, WorkerUpdate.model_validate(worker.model_dump())


async def _create_engine(tmp_path, name):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / name}", poolclass=NullPool
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
    return engine


def test_legacy_worker_upgrade_proof_cannot_claim_existing_identity(tmp_path):
    async def run():
        engine = await _create_engine(tmp_path, "proof-success.db")
        proof = _proof("a")
        async with AsyncSession(engine) as session:
            worker_id, worker_uuid = await _create_worker_identity(session)
            worker, worker_update = await _worker_update(session, worker_id)
            with pytest.raises(UnauthorizedException):
                await update_worker(
                    request=_system_request(),
                    response=Response(),
                    session=session,
                    id=worker_id,
                    worker_in=worker_update,
                    rotate_preheat_credential=True,
                    worker_credential=None,
                    upgrade_proof=proof,
                )
            identity = (
                await session.exec(
                    select(ModelPreheatWorkerIdentity).where(
                        ModelPreheatWorkerIdentity.worker_id == worker_id
                    )
                )
            ).one()
            worker, worker_update = await _worker_update(session, worker_id)
            with pytest.raises(UnauthorizedException):
                await _authorize_worker_registration(
                    _system_request(),
                    session,
                    worker_uuid,
                    None,
                    worker=worker,
                    worker_update=worker_update,
                    upgrade_proof=proof,
                )
        await engine.dispose()
        return identity

    identity = asyncio.run(run())
    assert identity.bootstrap_required is True
    assert identity.upgrade_proof_hash is None


@pytest.mark.parametrize("credential_kind", ["current", "pending", "recovery"])
def test_registration_rejects_worker_uuid_mutation_for_all_credentials(
    tmp_path, credential_kind
):
    async def run():
        engine = await _create_engine(tmp_path, f"uuid-{credential_kind}.db")
        async with AsyncSession(engine) as session:
            worker_id, worker_uuid = await _create_worker_identity(
                session, bootstrap_required=False
            )
            current = await issue_model_preheat_worker_credential(
                session, worker_id, worker_uuid
            )
            await get_model_preheat_worker_identity(
                request=SimpleNamespace(state=SimpleNamespace()),
                session=session,
                credential=current,
            )
            pending = await issue_model_preheat_worker_credential(
                session, worker_id, worker_uuid
            )
            credential = {"current": pending, "pending": pending, "recovery": current}[
                credential_kind
            ]
            if credential_kind == "current":
                await get_model_preheat_worker_identity(
                    request=SimpleNamespace(state=SimpleNamespace()),
                    session=session,
                    credential=pending,
                )
                credential = pending
            worker, worker_update = await _worker_update(session, worker_id)
            worker_update.worker_uuid = "attacker-worker-uuid"
            with pytest.raises(UnauthorizedException):
                await update_worker(
                    request=_system_request(),
                    response=Response(),
                    session=session,
                    id=worker_id,
                    worker_in=worker_update,
                    rotate_preheat_credential=True,
                    worker_credential=credential,
                )
            return await session.get(Worker, worker_id)
        await engine.dispose()

    worker = asyncio.run(run())
    assert worker.worker_uuid == "legacy-worker-uuid"


def test_legacy_upgrade_proofs_cannot_retry_or_bind(tmp_path):
    async def run():
        engine = await _create_engine(tmp_path, "proof-retry.db")
        first_proof = _proof("b")
        second_proof = _proof("c")
        async with AsyncSession(engine) as initial_session:
            worker_id, worker_uuid = await _create_worker_identity(initial_session)
            worker, worker_update = await _worker_update(initial_session, worker_id)
            with pytest.raises(UnauthorizedException):
                await _authorize_worker_registration(
                    _system_request(),
                    initial_session,
                    worker_uuid,
                    None,
                    worker=worker,
                    worker_update=worker_update,
                    upgrade_proof=first_proof,
                )
        async with AsyncSession(engine) as competing_session:
            worker, worker_update = await _worker_update(competing_session, worker_id)
            with pytest.raises(UnauthorizedException):
                await _authorize_worker_registration(
                    _system_request(),
                    competing_session,
                    worker_uuid,
                    None,
                    worker=worker,
                    worker_update=worker_update,
                    upgrade_proof=second_proof,
                )
        async with AsyncSession(engine) as retry_session:
            worker, worker_update = await _worker_update(retry_session, worker_id)
            with pytest.raises(UnauthorizedException):
                await _authorize_worker_registration(
                    _system_request(),
                    retry_session,
                    worker_uuid,
                    None,
                    worker=worker,
                    worker_update=worker_update,
                    upgrade_proof=first_proof,
                )
        await engine.dispose()

    asyncio.run(run())


def test_different_legacy_upgrade_proofs_are_both_rejected(tmp_path):
    async def run():
        engine = await _create_engine(tmp_path, "proof-cas.db")
        async with AsyncSession(engine) as setup_session:
            worker_id, worker_uuid = await _create_worker_identity(setup_session)
        start = asyncio.Event()

        async def attempt(proof):
            async with AsyncSession(engine) as session:
                worker, worker_update = await _worker_update(session, worker_id)
                await start.wait()
                try:
                    await _authorize_worker_registration(
                        _system_request(),
                        session,
                        worker_uuid,
                        None,
                        worker=worker,
                        worker_update=worker_update,
                        upgrade_proof=proof,
                    )
                except UnauthorizedException:
                    await session.rollback()
                    return False
                await session.commit()
                return True

        first = asyncio.create_task(attempt(_proof("e")))
        second = asyncio.create_task(attempt(_proof("f")))
        await asyncio.sleep(0)
        start.set()
        bound = await asyncio.gather(first, second)
        await engine.dispose()
        return bound

    assert sorted(asyncio.run(run())) == [False, False]


def test_upgrade_proof_rejects_different_worker_uuid_without_state_change(tmp_path):
    async def run():
        engine = await _create_engine(tmp_path, "proof-wrong-uuid.db")
        async with AsyncSession(engine) as session:
            worker_id, worker_uuid = await _create_worker_identity(session)
            worker, worker_update = await _worker_update(session, worker_id)
            worker_update.worker_uuid = "different-worker-uuid"
            with pytest.raises(UnauthorizedException):
                await update_worker(
                    request=_system_request(),
                    response=Response(),
                    session=session,
                    id=worker_id,
                    worker_in=worker_update,
                    rotate_preheat_credential=True,
                    worker_credential=None,
                    upgrade_proof=_proof("g"),
                )
            worker_after = await session.get(Worker, worker_id)
            identity_after = (
                await session.exec(
                    select(ModelPreheatWorkerIdentity).where(
                        ModelPreheatWorkerIdentity.worker_id == worker_id
                    )
                )
            ).one()
            pending_after = (
                await session.exec(
                    select(ModelPreheatWorkerPendingCredential).where(
                        ModelPreheatWorkerPendingCredential.identity_id
                        == identity_after.id
                    )
                )
            ).all()
        await engine.dispose()
        return worker_after, identity_after, pending_after

    worker_after, identity_after, pending_after = asyncio.run(run())
    assert worker_after.worker_uuid == "legacy-worker-uuid"
    assert identity_after.bootstrap_required is True
    assert identity_after.upgrade_proof_hash is None
    assert pending_after == []


@pytest.mark.parametrize(
    "bootstrap_required,window,hostname,ip",
    [
        (
            True,
            datetime.now(timezone.utc)
            - WORKER_UPGRADE_PROOF_TTL
            - timedelta(seconds=1),
            "legacy-host",
            "10.0.0.8",
        ),
        (False, datetime.now(timezone.utc), "legacy-host", "10.0.0.8"),
        (True, datetime.now(timezone.utc), "changed-host", "10.0.0.8"),
        (True, datetime.now(timezone.utc), "legacy-host", "10.0.0.9"),
    ],
)
def test_upgrade_proof_rejects_outside_legacy_window_or_stable_identity(
    tmp_path, bootstrap_required, window, hostname, ip
):
    async def run():
        engine = await _create_engine(tmp_path, "proof-rejected.db")
        async with AsyncSession(engine) as session:
            worker_id, worker_uuid = await _create_worker_identity(
                session,
                bootstrap_required=bootstrap_required,
                window=window,
            )
            worker, worker_update = await _worker_update(session, worker_id)
            worker_update.hostname = hostname
            worker_update.ip = ip
            with pytest.raises(UnauthorizedException):
                await _authorize_worker_registration(
                    _system_request(),
                    session,
                    worker_uuid,
                    None,
                    worker=worker,
                    worker_update=worker_update,
                    upgrade_proof=_proof("d"),
                )
        await engine.dispose()

    asyncio.run(run())
