from unittest.mock import AsyncMock, patch

import pytest

from gpustack.policies.base import Allocatable, ModelInstanceScheduleCandidate
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.schemas.models import (
    ComputedResourceClaim,
    ModelInstanceSubordinateWorker,
    PlacementStrategyEnum,
)
from tests.fixtures.workers.fixtures import (
    linux_nvidia_2_4080_16gx2,
    linux_nvidia_5_a100_80gx2,
)
from tests.utils.model import new_model


def candidate(worker, gpu_indexes):
    return ModelInstanceScheduleCandidate(
        worker=worker,
        gpu_indexes=gpu_indexes,
        computed_resource_claim=ComputedResourceClaim(ram=1, vram={0: 1, 1: 1}),
    )


@pytest.mark.asyncio
async def test_spread_keeps_only_unallocated_gpu_candidates():
    worker = linux_nvidia_2_4080_16gx2()
    worker.status.gpu_devices[0].memory.allocated = 0
    worker.status.gpu_devices[1].memory.allocated = 2 * 1024**3
    scorer = PlacementScorer(
        new_model(1, "spread", placement_strategy=PlacementStrategyEnum.SPREAD)
    )
    scorer._get_worker_model_instance_count = AsyncMock(return_value={})

    result = await scorer.score([candidate(worker, [0]), candidate(worker, [1])])

    assert [item.gpu_indexes for item in result] == [[0]]


@pytest.mark.asyncio
async def test_spread_returns_no_gpu_candidate_when_all_are_allocated():
    worker = linux_nvidia_2_4080_16gx2()
    for device in worker.status.gpu_devices:
        device.memory.allocated = 1
    scorer = PlacementScorer(
        new_model(1, "spread", placement_strategy=PlacementStrategyEnum.SPREAD)
    )
    scorer._get_worker_model_instance_count = AsyncMock(return_value={})

    result = await scorer.score([candidate(worker, [0]), candidate(worker, [1])])

    assert result == []


@pytest.mark.asyncio
async def test_spread_rejects_gpu_with_unknown_allocation():
    worker = linux_nvidia_2_4080_16gx2()
    worker.status.gpu_devices[0].memory.allocated = None
    scorer = PlacementScorer(
        new_model(1, "spread", placement_strategy=PlacementStrategyEnum.SPREAD)
    )
    scorer._get_worker_model_instance_count = AsyncMock(return_value={})

    result = await scorer.score([candidate(worker, [0])])

    assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize("subordinate_allocated,expected_count", [(0, 1), (1, 0)])
async def test_spread_checks_subordinate_worker_gpus(
    subordinate_allocated, expected_count
):
    main_worker = linux_nvidia_2_4080_16gx2()
    subordinate_worker = linux_nvidia_5_a100_80gx2()
    main_worker.status.gpu_devices[0].memory.allocated = 0
    subordinate_worker.status.gpu_devices[0].memory.allocated = subordinate_allocated
    distributed_candidate = candidate(main_worker, [0])
    distributed_candidate.subordinate_workers = [
        ModelInstanceSubordinateWorker(
            worker_id=subordinate_worker.id,
            gpu_indexes=[0],
            computed_resource_claim=ComputedResourceClaim(vram={0: 1}),
        )
    ]
    scorer = PlacementScorer(
        new_model(1, "spread", placement_strategy=PlacementStrategyEnum.SPREAD)
    )
    scorer._get_worker_model_instance_count = AsyncMock(return_value={})

    with (
        patch(
            "gpustack.policies.scorers.placement_scorer.AsyncSession",
        ) as session,
        patch(
            "gpustack.policies.scorers.placement_scorer.Worker.all",
            AsyncMock(return_value=[main_worker, subordinate_worker]),
        ),
    ):
        session.return_value.__aenter__ = AsyncMock()
        session.return_value.__aexit__ = AsyncMock(return_value=None)
        result = await scorer.score([distributed_candidate])

    assert len(result) == expected_count


@pytest.mark.asyncio
async def test_spread_keeps_cpu_only_candidates():
    worker = linux_nvidia_2_4080_16gx2()
    scorer = PlacementScorer(
        new_model(1, "spread", placement_strategy=PlacementStrategyEnum.SPREAD)
    )
    scorer._get_worker_model_instance_count = AsyncMock(return_value={})

    result = await scorer.score([candidate(worker, [])])

    assert len(result) == 1
    assert result[0].gpu_indexes == []


@pytest.mark.asyncio
async def test_binpack_keeps_allocated_gpu_candidates():
    worker = linux_nvidia_2_4080_16gx2()
    worker.status.gpu_devices[0].memory.allocated = 1
    scorer = PlacementScorer(
        new_model(1, "binpack", placement_strategy=PlacementStrategyEnum.BINPACK)
    )
    allocatable = Allocatable(ram=1024, vram={0: 1024})

    with patch(
        "gpustack.policies.scorers.placement_scorer.get_worker_allocatable_resource",
        AsyncMock(return_value=allocatable),
    ):
        result = await scorer.score([candidate(worker, [0])])

    assert len(result) == 1
    assert result[0].gpu_indexes == [0]
