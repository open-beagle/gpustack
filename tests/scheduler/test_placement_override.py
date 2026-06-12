import asyncio

from gpustack.policies.worker_filters.gpu_matching_filter import GPUMatchingFilter
from gpustack.scheduler.placement_override import get_model_for_instance_scheduling
from gpustack.schemas.models import (
    BackendEnum,
    GPUSelector,
    ModelInstancePlacementOverride,
    ModelInstanceStateEnum,
)
from tests.fixtures.workers.fixtures import (
    linux_nvidia_1_4090_24gx1,
    linux_nvidia_2_4080_16gx2,
)
from tests.utils.model import new_model, new_model_instance


def test_instance_placement_override_filters_workers_without_mutating_model():
    model = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend=BackendEnum.VLLM,
        backend_parameters=[],
    )
    instance = new_model_instance(1, "test_name-1", model.id)
    instance.placement_override = ModelInstancePlacementOverride(
        gpu_selector=GPUSelector(gpu_ids=["host4080:cuda:1"])
    )

    scheduling_model = get_model_for_instance_scheduling(model, instance)

    workers, _ = asyncio.run(
        GPUMatchingFilter(scheduling_model).filter(
            [linux_nvidia_1_4090_24gx1(), linux_nvidia_2_4080_16gx2()]
        )
    )

    assert [worker.name for worker in workers] == ["host4080"]
    assert scheduling_model.gpu_selector.gpu_ids == ["host4080:cuda:1"]
    assert model.gpu_selector is None


def test_instance_without_placement_override_uses_persisted_model_selector():
    model = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend=BackendEnum.VLLM,
        backend_parameters=[],
        gpu_selector=GPUSelector(gpu_ids=["host4090:cuda:0"]),
    )
    instance = new_model_instance(1, "test_name-1", model.id)

    scheduling_model = get_model_for_instance_scheduling(model, instance)

    assert scheduling_model is model
    assert scheduling_model.gpu_selector.gpu_ids == ["host4090:cuda:0"]


def test_scheduled_retry_keeps_placement_override_for_rescheduling():
    model = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend=BackendEnum.VLLM,
        backend_parameters=[],
    )
    instance = new_model_instance(1, "test_name-1", model.id)
    instance.state = ModelInstanceStateEnum.SCHEDULED
    instance.worker_id = 1
    instance.placement_override = ModelInstancePlacementOverride(
        gpu_selector=GPUSelector(gpu_ids=["host4080:cuda:1"])
    )

    scheduling_model = get_model_for_instance_scheduling(model, instance)

    assert scheduling_model.gpu_selector.gpu_ids == ["host4080:cuda:1"]
