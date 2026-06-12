from unittest.mock import AsyncMock, patch

import asyncio

from gpustack.routes.models import update_model
from gpustack.schemas.models import (
    GPUSelector,
    ModelPlacementOverride,
    ModelUpdate,
    PlacementOverrideReplicaGroup,
    SourceEnum,
)
from tests.utils.model import new_model


def test_update_model_with_placement_override_updates_model_once():
    model = new_model(
        1,
        "test",
        2,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
    )
    model_in = ModelUpdate(
        name="test",
        replicas=2,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        placement_override=ModelPlacementOverride(
            new_replica_groups=[
                PlacementOverrideReplicaGroup(
                    gpu_selector=GPUSelector(gpu_ids=["host4090:cuda:0"])
                )
            ]
        ),
    )

    update_mock = AsyncMock(return_value=model)
    sync_mock = AsyncMock()

    with (
        patch("gpustack.schemas.models.Model.one_by_id", return_value=model),
        patch("gpustack.routes.models.validate_model_in", new=AsyncMock()),
        patch("gpustack.routes.models.ModelService") as service_cls,
        patch("gpustack.routes.models.sync_replicas", new=sync_mock),
    ):
        service_cls.return_value.update = update_mock

        result = asyncio.run(update_model(AsyncMock(), model.id, model_in))

    assert result is model
    assert update_mock.await_count == 1
    sync_mock.assert_awaited_once()
