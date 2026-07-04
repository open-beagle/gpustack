from datetime import datetime, timedelta, timezone

from gpustack.schemas.workers import (
    GPUDeviceInfo,
    MemoryInfo,
    Worker,
    WorkerStateEnum,
    WorkerStatus,
)


def test_compute_state_clears_allocated_resources_when_worker_is_not_ready():
    worker = Worker(
        id=1,
        name="worker-a",
        hostname="worker-a",
        ip="10.0.0.1",
        port=10150,
        state=WorkerStateEnum.READY,
        labels={},
        system_reserved=None,
        status=WorkerStatus(
            memory=MemoryInfo(total=1024, used=0, allocated=512),
            gpu_devices=[
                GPUDeviceInfo(
                    index=0,
                    name="GPU-0",
                    memory=MemoryInfo(total=1024, used=0, allocated=900),
                )
            ],
        ),
        unreachable=False,
        heartbeat_time=datetime.now(timezone.utc) - timedelta(seconds=120),
        worker_uuid="worker-a-uuid",
    )

    worker.compute_state(worker_offline_timeout=60)

    assert worker.state == WorkerStateEnum.NOT_READY
    assert worker.status.memory.allocated == 0
    assert worker.status.gpu_devices[0].memory.allocated == 0
