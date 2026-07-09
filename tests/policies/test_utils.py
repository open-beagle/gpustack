from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpustack.policies.utils import get_worker_allocatable_resource
from tests.fixtures.workers.fixtures import linux_nvidia_8_3090_24gx8


GiB = 1024**3


@pytest.mark.asyncio
async def test_worker_allocatable_vram_uses_live_gpu_memory_when_higher():
    worker = linux_nvidia_8_3090_24gx8(reserved=False)
    worker.status.gpu_devices = worker.status.gpu_devices[:1]
    gpu = worker.status.gpu_devices[0]
    gpu.memory.used = 10 * GiB
    total = gpu.memory.total

    with patch("gpustack.policies.utils.get_worker_model_instances", return_value=[]):
        allocatable = await get_worker_allocatable_resource(None, worker)

    assert allocatable.vram[gpu.index] == total - 10 * GiB


@pytest.mark.asyncio
async def test_worker_allocatable_vram_keeps_allocated_claim_when_higher_than_live_used():
    worker = linux_nvidia_8_3090_24gx8(reserved=False)
    worker.status.gpu_devices = worker.status.gpu_devices[:1]
    gpu = worker.status.gpu_devices[0]
    gpu.memory.used = 1 * GiB
    total = gpu.memory.total
    vram_claim = 20 * GiB
    model_instance = SimpleNamespace(
        worker_id=worker.id,
        computed_resource_claim=SimpleNamespace(ram=0, vram={gpu.index: vram_claim}),
        gpu_indexes=[gpu.index],
        distributed_servers=None,
    )

    with patch(
        "gpustack.policies.utils.get_worker_model_instances",
        return_value=[model_instance],
    ):
        allocatable = await get_worker_allocatable_resource(None, worker)

    assert allocatable.vram[gpu.index] == total - vram_claim
