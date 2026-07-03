import asyncio
from datetime import datetime
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
from gpustack.server.controllers import find_scale_down_candidates, safe_event_attr, sync_replicas
from gpustack.server.bus import Event, EventType
from tests.fixtures.workers.fixtures import (
    linux_nvidia_19_4090_24gx2,
    linux_nvidia_2_4080_16gx2,
    linux_cpu_1,
)

from unittest.mock import patch, AsyncMock

from tests.utils.model import new_model, new_model_instance


def test_find_scale_down_candidates():
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

        candidates = asyncio.run(find_scale_down_candidates(mis, m))

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


def test_model_event_data_is_safe_to_access_after_publish():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    event = Event(type=EventType.UPDATED, data=model.model_copy(deep=True))

    assert event.data.id == 1
    assert event.data.name == "test"


def test_convert_to_public_class_handles_model_construct_snapshot():
    from gpustack.schemas.models import ModelInstance

    instance = new_model_instance(6, "test-6", 1)
    instance.source = new_model(1, "test", huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct").source
    instance.__dict__["created_at"] = datetime.now()
    instance.__dict__["updated_at"] = datetime.now()
    snapshot = instance.__class__.model_construct(**instance.model_dump())

    public = ModelInstance._convert_to_public_class(snapshot)

    assert not hasattr(snapshot, "_sa_instance_state")
    assert public["id"] == 6
    assert public["name"] == "test-6"


def test_convert_to_public_class_uses_public_schema_for_complete_data():
    from gpustack.schemas.models import ModelInstance

    now = datetime.now()
    instance = new_model_instance(7, "test-7", 1)
    instance.source = new_model(1, "test", huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct").source
    instance.huggingface_repo_id = "Qwen/Qwen2.5-7B-Instruct"
    instance.__dict__["created_at"] = now
    instance.__dict__["updated_at"] = now

    public = ModelInstance._convert_to_public_class(instance)

    assert public.id == 7
    assert public.name == "test-7"


def test_safe_event_attr_handles_detached_orm_attribute_errors():
    class DetachedLike:
        @property
        def name(self):
            raise RuntimeError("detached")

    assert safe_event_attr(DetachedLike(), "name") == "<unknown>"


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


def test_sync_replicas_serializes_scale_down_for_same_model():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    remaining_instances = {
        1: new_model_instance(1, "test-1", model.id, 1, ModelInstanceStateEnum.RUNNING, [0]),
        2: new_model_instance(2, "test-2", model.id, 1, ModelInstanceStateEnum.RUNNING, [1]),
    }
    deleted_instance_ids = []

    async def mock_all_by_field(*args, **kwargs):
        return list(remaining_instances.values())

    async def mock_delete(instance):
        await asyncio.sleep(0.01)
        remaining_instances.pop(instance.id, None)
        deleted_instance_ids.append(instance.id)
        return instance

    async def mock_find_scale_down_candidates(instances, model):
        return [ModelInstanceScore(model_instance=instances[0], score=0)]

    async def run_concurrent_syncs():
        await asyncio.gather(
            sync_replicas(
                AsyncMock(),
                model,
                AsyncMock(),
                scale_in_instance_ids=[2],
            ),
            sync_replicas(AsyncMock(), model, AsyncMock()),
        )

    with (
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            side_effect=mock_all_by_field,
        ),
        patch(
            'gpustack.server.services.ModelInstanceService.delete',
            side_effect=mock_delete,
        ),
        patch(
            'gpustack.server.controllers.find_scale_down_candidates',
            side_effect=mock_find_scale_down_candidates,
        ),
    ):
        asyncio.run(run_concurrent_syncs())

    assert deleted_instance_ids == [2]
    assert list(remaining_instances) == [1]


def test_sync_replicas_rechecks_current_instances_before_delete():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    stale_instances = [
        new_model_instance(1, "test-1", model.id, 1, ModelInstanceStateEnum.RUNNING, [0]),
        new_model_instance(2, "test-2", model.id, 1, ModelInstanceStateEnum.RUNNING, [1]),
    ]
    current_instances = [stale_instances[0]]
    all_by_field_results = [stale_instances, current_instances]
    deleted_instances = []

    async def mock_all_by_field(*args, **kwargs):
        return all_by_field_results.pop(0)

    async def mock_find_scale_down_candidates(instances, model):
        return [ModelInstanceScore(model_instance=instances[0], score=0)]

    async def mock_delete(instance):
        deleted_instances.append(instance)
        return instance

    with (
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            side_effect=mock_all_by_field,
        ),
        patch(
            'gpustack.server.controllers.find_scale_down_candidates',
            side_effect=mock_find_scale_down_candidates,
        ),
        patch(
            'gpustack.server.services.ModelInstanceService.delete',
            side_effect=mock_delete,
        ),
    ):
        asyncio.run(sync_replicas(AsyncMock(), model, AsyncMock()))

    assert deleted_instances == []


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
