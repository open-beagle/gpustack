from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

import pytest

from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.scheduler.scheduler import (
    discover_model_placement,
    evaluate_model_placement,
    scheduling_failure_reason_code,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    PlacementStrategyEnum,
)
from gpustack.schemas.scheduler import PlacementEvaluationReplicaGroup
from gpustack.schemas.workers import (
    GPUDeviceInfo,
    MemoryInfo,
    Worker,
    WorkerStateEnum,
    WorkerStatus,
)
from tests.utils.model import new_model


def placement_worker():
    return Worker(
        id=1,
        name="worker-a",
        hostname="worker-a",
        ip="10.0.0.1",
        port=10150,
        worker_uuid="worker-a",
        state=WorkerStateEnum.READY,
        status=WorkerStatus(
            memory=MemoryInfo(total=64_000, allocated=0),
            gpu_devices=[
                GPUDeviceInfo(
                    index=0,
                    memory=MemoryInfo(total=24_000, allocated=0),
                )
            ],
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("independent", "expected_fit"),
    [(False, [True, False]), (True, [True, True])],
)
async def test_placement_evaluation_simulates_previous_replica_reservation(
    independent, expected_fit
):
    model = new_model(
        10,
        "model-a",
        placement_strategy=PlacementStrategyEnum.SPREAD,
    )
    worker = placement_worker()

    async def find_candidate(_config, scheduling_model, workers, **_kwargs):
        current = workers[0]
        candidate = ModelInstanceScheduleCandidate(
            worker=current,
            gpu_indexes=[0],
            computed_resource_claim=ComputedResourceClaim(
                ram=1_000, vram={0: 2_000}
            ),
        )
        assert scheduling_model.gpu_selector.gpu_ids == ["worker-a:cuda:0"]
        if current.status.gpu_devices[0].memory.allocated:
            return None, ["GPU is not idle"], [candidate], {}
        return candidate, [], [candidate], {}

    with (
        patch.object(Worker, "all", AsyncMock(return_value=[worker])),
        patch(
            "gpustack.scheduler.scheduler.SchedulerPolicy.one_by_field",
            AsyncMock(return_value=SimpleNamespace(enabled=True, aggregation_rate=80)),
        ),
        patch(
            "gpustack.scheduler.scheduler.find_candidate_detailed",
            side_effect=find_candidate,
        ),
    ):
        result = await evaluate_model_placement(
            object(),
            object(),
            model,
            [
                PlacementEvaluationReplicaGroup(gpu_ids=["worker-a:cuda:0"]),
                PlacementEvaluationReplicaGroup(gpu_ids=["worker-a:cuda:0"]),
            ],
            independent=independent,
        )

    assert [item.fit for item in result.results] == expected_fit
    assert result.fit is all(expected_fit)
    if not independent:
        assert result.results[1].reason_code == "spread_requires_idle_gpu"


def test_spread_failure_with_resource_candidate_has_specific_reason_code():
    model = new_model(
        10,
        "model-a",
        placement_strategy=PlacementStrategyEnum.SPREAD,
    )
    candidate = ModelInstanceScheduleCandidate(
        worker=placement_worker(),
        gpu_indexes=[0],
        computed_resource_claim=ComputedResourceClaim(vram={0: 2_000}),
    )

    assert (
        scheduling_failure_reason_code(model, [], [candidate])
        == "spread_requires_idle_gpu"
    )


@pytest.mark.asyncio
async def test_placement_discovery_returns_all_post_policy_candidates():
    model = new_model(
        10,
        "model-a",
        placement_strategy=PlacementStrategyEnum.SPREAD,
    )
    worker = placement_worker()
    candidates = [
        ModelInstanceScheduleCandidate(
            worker=worker,
            gpu_indexes=[index],
            computed_resource_claim=ComputedResourceClaim(vram={index: 2_000}),
            score=90 - index,
        )
        for index in (0, 1)
    ]

    with (
        patch.object(Worker, "all", AsyncMock(return_value=[worker])),
        patch(
            "gpustack.scheduler.scheduler.SchedulerPolicy.one_by_field",
            AsyncMock(return_value=SimpleNamespace(enabled=True, aggregation_rate=80)),
        ),
        patch(
            "gpustack.scheduler.scheduler.find_candidate_options_detailed",
            AsyncMock(return_value=(candidates, [], candidates, {})),
        ),
    ):
        result = await discover_model_placement(object(), object(), model)

    assert result.fit is True
    assert len(result.results[0].candidate_targets) == 2
    assert result.results[0].candidate_targets[1]["gpu_indexes"] == [1]
