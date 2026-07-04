import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from gpustack.routes.workers import get_workers
from gpustack.schemas.common import ListParams, PaginatedList, Pagination
from gpustack.schemas.workers import (
    CPUInfo,
    GPUCoreInfo,
    GPUDeviceInfo,
    MemoryInfo,
    Worker,
    WorkerStateEnum,
    WorkerStatus,
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
