import asyncio
from typing import List
import pytest
from gpustack.policies.base import ModelInstanceScore

from gpustack.schemas.models import (
    ComputedResourceClaim,
    GPUSelector,
    ModelPlacementOverride,
    PlacementOverrideReplicaGroup,
    ModelInstanceStateEnum,
)
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.controllers import find_scale_down_candidates, sync_replicas
from tests.fixtures.workers.fixtures import (
    linux_nvidia_19_4090_24gx2,
    linux_nvidia_2_4080_16gx2,
    linux_cpu_1,
)

from unittest.mock import patch, AsyncMock

from tests.utils.model import new_model, new_model_instance


@pytest.mark.asyncio
async def test_find_scale_down_candidates():
    w1 = linux_nvidia_19_4090_24gx2()
    w1.state = WorkerStateEnum.NOT_READY
    workers = [
        w1,
        linux_nvidia_2_4080_16gx2(),
        linux_cpu_1(),
    ]

    m = new_model(1, "test", 3, "llama3:70b")
    mis = [
        new_model_instance(
            1,
            "test-1",
            1,
            4,
            ModelInstanceStateEnum.RUNNING,
            [0, 1],
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=81,
                total_layers=81,
                ram=455165112,
                vram={0: 22912443392, 1: 22911897600},
            ),
        ),
        new_model_instance(
            2,
            "test-2",
            1,
            3,
            ModelInstanceStateEnum.RUNNING,
            [0, 1],
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=60,
                total_layers=81,
                ram=1093245112,
                vram={0: 16900820992, 1: 16900820992},
            ),
        ),
        new_model_instance(
            3,
            "test-3",
            1,
            6,
            ModelInstanceStateEnum.RUNNING,
            None,
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=0,
                total_layers=81,
                ram=3106511032,
            ),
        ),
    ]

    with (
        patch('sqlmodel.ext.asyncio.session.AsyncSession', AsyncMock()),
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            return_value=mis,
        ),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
    ):

        candidates = await find_scale_down_candidates(mis, m)

        expected_candidates = [
            {
                "worker_id": 4,
                "instacnce_id": 1,
                "gpu_indexes": [0, 1],
                "score": 9.538995598356342,
            },
            {
                "worker_id": 6,
                "instacnce_id": 3,
                "score": 90.1308159326069,
            },
            {
                "worker_id": 3,
                "instacnce_id": 2,
                "score": 97.3594505895714,
            },
        ]

        compare_candidates(candidates, expected_candidates)


def compare_candidates(candidates: List[ModelInstanceScore], expected_candidates):
    for i, expected in enumerate(expected_candidates):
        candidate = candidates[i]
        instance = candidate.model_instance

        if "worker_id" in expected:
            assert instance.worker_id == expected["worker_id"]

        if "instance_id" in expected:
            assert instance.id == expected["instance_id"]

        if "score" in expected:
            assert str(candidate.score)[:5] == str(expected["score"])[:5]


def test_sync_replicas_puts_placement_override_on_new_instances_only():
    model = new_model(1, "test", 2, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    created_instances = []

    async def mock_create(instance):
        created_instances.append(instance)
        return instance

    placement_override = ModelPlacementOverride(
        replica_groups=[
            PlacementOverrideReplicaGroup(
                gpu_selector=GPUSelector(gpu_ids=["host4090:cuda:0"])
            ),
            PlacementOverrideReplicaGroup(
                gpu_selector=GPUSelector(
                    gpu_ids=["host4080:cuda:0", "host4080:cuda:1"]
                )
            ),
        ]
    )

    with (
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            return_value=[],
        ),
        patch(
            'gpustack.server.services.ModelInstanceService.create',
            side_effect=mock_create,
        ),
    ):
        asyncio.run(
            sync_replicas(
                AsyncMock(),
                model,
                AsyncMock(),
                placement_override=placement_override,
            )
        )

    assert model.gpu_selector is None
    assert len(created_instances) == 2
    assert created_instances[0].placement_override.gpu_selector.gpu_ids == [
        "host4090:cuda:0"
    ]
    assert created_instances[1].placement_override.gpu_selector.gpu_ids == [
        "host4080:cuda:0",
        "host4080:cuda:1",
    ]


def test_sync_replicas_interprets_replica_groups_as_new_instances_only():
    model = new_model(1, "test", 3, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    existing_instances = [
        new_model_instance(1, "test-existing-1", model.id),
    ]
    created_instances = []

    async def mock_create(instance):
        created_instances.append(instance)
        return instance

    placement_override = ModelPlacementOverride(
        replica_groups=[
            PlacementOverrideReplicaGroup(
                gpu_selector=GPUSelector(gpu_ids=["host4090:cuda:0"])
            ),
            PlacementOverrideReplicaGroup(
                gpu_selector=GPUSelector(gpu_ids=["host4080:cuda:1"])
            ),
        ]
    )

    with (
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            return_value=existing_instances,
        ),
        patch(
            'gpustack.server.services.ModelInstanceService.create',
            side_effect=mock_create,
        ),
    ):
        asyncio.run(
            sync_replicas(
                AsyncMock(),
                model,
                AsyncMock(),
                placement_override=placement_override,
            )
        )

    assert len(created_instances) == 2
    assert created_instances[0].placement_override.gpu_selector.gpu_ids == [
        "host4090:cuda:0"
    ]
    assert created_instances[1].placement_override.gpu_selector.gpu_ids == [
        "host4080:cuda:1"
    ]


def test_sync_replicas_accepts_new_replica_groups_name():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    created_instances = []

    async def mock_create(instance):
        created_instances.append(instance)
        return instance

    placement_override = ModelPlacementOverride(
        new_replica_groups=[
            PlacementOverrideReplicaGroup(
                gpu_selector=GPUSelector(gpu_ids=["host4090:cuda:0"])
            )
        ]
    )

    with (
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            return_value=[],
        ),
        patch(
            'gpustack.server.services.ModelInstanceService.create',
            side_effect=mock_create,
        ),
    ):
        asyncio.run(
            sync_replicas(
                AsyncMock(),
                model,
                AsyncMock(),
                placement_override=placement_override,
            )
        )

    assert len(created_instances) == 1
    assert created_instances[0].placement_override.gpu_selector.gpu_ids == [
        "host4090:cuda:0"
    ]


def test_sync_replicas_deletes_specified_scale_in_instances_as_whole_replicas():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    instances = [
        new_model_instance(1, "test-1", model.id, 1, ModelInstanceStateEnum.RUNNING, [0, 1]),
        new_model_instance(2, "test-2", model.id, 1, ModelInstanceStateEnum.RUNNING, [2, 3]),
        new_model_instance(3, "test-3", model.id, 1, ModelInstanceStateEnum.RUNNING, [4, 5]),
    ]
    deleted_instances = []

    async def mock_delete(instance):
        deleted_instances.append(instance)
        return instance

    with (
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            return_value=instances,
        ),
        patch(
            'gpustack.server.services.ModelInstanceService.delete',
            side_effect=mock_delete,
        ),
    ):
        asyncio.run(
            sync_replicas(
                AsyncMock(),
                model,
                AsyncMock(),
                scale_in_instance_ids=[2, 3],
            )
        )

    assert [instance.id for instance in deleted_instances] == [2, 3]
    assert deleted_instances[0].gpu_indexes == [2, 3]


def test_sync_replicas_rejects_invalid_specified_scale_in_instances():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    instances = [
        new_model_instance(1, "test-1", model.id),
        new_model_instance(2, "test-2", model.id),
        new_model_instance(3, "test-3", model.id),
    ]

    with patch(
        'gpustack.schemas.models.ModelInstance.all_by_field',
        return_value=instances,
    ):
        with pytest.raises(ValueError, match="scale_in_instance_ids"):
            asyncio.run(
                sync_replicas(
                    AsyncMock(),
                    model,
                    AsyncMock(),
                    scale_in_instance_ids=[2],
                )
            )
