import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.policies.base import Allocatable, ModelInstanceScheduleCandidate
from gpustack.schemas.models import ComputedResourceClaim, PlacementStrategyEnum
from gpustack.schemas.scheduler import SchedulingAttemptEvent, SchedulingOutcome
from gpustack.scheduler.scheduler import _build_scheduling_event
from gpustack.schemas.workers import (
    GPUDeviceInfo,
    MemoryInfo,
    SystemReserved,
    Worker,
    WorkerStateEnum,
    WorkerStatus,
)
from gpustack.scheduler.aggregation import filter_by_aggregation_rate


async def build_scheduling_event(session: AsyncSession, workload_id: str):
    return await _build_scheduling_event(
        session=session,
        model=SimpleNamespace(
            id=1,
            meta={"model_deploy_id": workload_id},
            placement_strategy=PlacementStrategyEnum.BINPACK,
            replicas=1,
        ),
        model_instance=SimpleNamespace(),
        policy=SimpleNamespace(enabled=True, runtime_revision=1),
        candidates=[],
        selected=None,
        projected_loads={},
        outcome=SchedulingOutcome.FAILED,
        reason_code="no_candidate",
        reason="No candidate available.",
        started_at=time.monotonic(),
    )


@pytest.mark.asyncio
async def test_scheduling_event_first_attempt_starts_at_one():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                SQLModel.metadata.create_all, tables=[SchedulingAttemptEvent.__table__]
            )
        async with AsyncSession(engine) as session:
            event = await build_scheduling_event(session, "workload-first")

        assert event.attempt_no == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_scheduling_event_increments_existing_attempt():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                SQLModel.metadata.create_all, tables=[SchedulingAttemptEvent.__table__]
            )
        async with AsyncSession(engine) as session:
            existing = await build_scheduling_event(session, "workload-existing")
            existing_event_id = existing.event_id
            session.add(existing)
            await session.commit()
            stored_outcome = (
                await session.exec(
                    text(
                        "SELECT outcome FROM scheduling_attempt_events "
                        "WHERE event_id = :event_id"
                    ),
                    params={"event_id": existing_event_id},
                )
            ).scalar_one()

            event = await build_scheduling_event(session, "workload-existing")

        assert stored_outcome == "failed"
        assert event.attempt_no == 2
    finally:
        await engine.dispose()


def worker(worker_id: int, name: str) -> Worker:
    return Worker(
        id=worker_id,
        name=name,
        hostname=name,
        ip=f"10.0.0.{worker_id}",
        port=10150,
        worker_uuid=f"worker-{worker_id}",
        state=WorkerStateEnum.READY,
        system_reserved=SystemReserved(ram=0, vram=0),
        status=WorkerStatus(
            memory=MemoryInfo(total=1000, used=0),
            gpu_devices=[GPUDeviceInfo(index=0, memory=MemoryInfo(total=100, used=0))],
        ),
    )


@pytest.mark.asyncio
async def test_aggregation_rate_filters_projected_gpu_load():
    dense = worker(1, "dense")
    overloaded = worker(2, "overloaded")
    claim = ComputedResourceClaim(ram=0, vram={0: 25})
    candidates = [
        ModelInstanceScheduleCandidate(dense, [0], claim),
        ModelInstanceScheduleCandidate(overloaded, [0], claim),
    ]

    async def allocatable(_engine, selected_worker):
        available = 50 if selected_worker.id == dense.id else 10
        return Allocatable(ram=1000, vram={0: available})

    with patch(
        "gpustack.scheduler.aggregation.get_worker_allocatable_resource",
        new=AsyncMock(side_effect=allocatable),
    ):
        accepted, loads = await filter_by_aggregation_rate(
            AsyncMock(), candidates, [dense, overloaded], 80
        )

    assert accepted == [candidates[0]]
    assert loads[id(candidates[0])] == 75
    assert loads[id(candidates[1])] == 115


@pytest.mark.asyncio
async def test_aggregation_rate_100_keeps_candidate_at_exact_capacity():
    selected_worker = worker(1, "full")
    candidate = ModelInstanceScheduleCandidate(
        selected_worker,
        [0],
        ComputedResourceClaim(ram=0, vram={0: 25}),
    )
    with patch(
        "gpustack.scheduler.aggregation.get_worker_allocatable_resource",
        new=AsyncMock(return_value=Allocatable(ram=1000, vram={0: 25})),
    ):
        accepted, loads = await filter_by_aggregation_rate(
            AsyncMock(), [candidate], [selected_worker], 100
        )

    assert accepted == [candidate]
    assert loads[id(candidate)] == 100
