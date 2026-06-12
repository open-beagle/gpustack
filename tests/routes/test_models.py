from unittest.mock import AsyncMock, patch

import asyncio

from gpustack.routes.models import update_model
from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.models import (
    GPUSelector,
    ModelPlacementOverride,
    ModelUpdate,
    PlacementOverrideReplicaGroup,
    SourceEnum,
)
from tests.utils.model import new_model, new_model_instance


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


def test_update_model_with_scale_in_instance_ids_does_not_persist_request_field():
    model = new_model(
        1,
        "test",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
    )
    model_in = ModelUpdate(
        name="test",
        replicas=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        scale_in_instance_ids=[10, 11],
    )

    save_mock = AsyncMock()
    publish_mock = AsyncMock()
    sync_mock = AsyncMock()

    with (
        patch("gpustack.schemas.models.Model.one_by_id", return_value=model),
        patch("gpustack.routes.models.validate_model_in", new=AsyncMock()),
        patch(
            "gpustack.routes.models.ModelInstance.all_by_field",
            return_value=[
                new_model_instance(10, "test-10", model.id),
                new_model_instance(11, "test-11", model.id),
                new_model_instance(12, "test-12", model.id),
            ],
        ),
        patch("gpustack.routes.models.Model.save", new=save_mock),
        patch("gpustack.routes.models.Model._publish_event", new=publish_mock),
        patch("gpustack.routes.models.delete_cache_by_key", new=AsyncMock()),
        patch("gpustack.routes.models.ModelService") as service_cls,
        patch("gpustack.routes.models.sync_replicas", new=sync_mock),
    ):
        result = asyncio.run(update_model(AsyncMock(), model.id, model_in))

    assert result is model
    service_cls.return_value.update.assert_not_called()
    save_mock.assert_awaited_once()
    assert not hasattr(model, "scale_in_instance_ids")
    sync_mock.assert_awaited_once()
    assert sync_mock.await_args.kwargs["scale_in_instance_ids"] == [10, 11]
    publish_mock.assert_awaited_once()


def test_update_model_validates_scale_in_instance_ids_before_persisting_replicas():
    model = new_model(
        1,
        "test",
        3,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
    )
    model_in = ModelUpdate(
        name="test",
        replicas=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct",
        scale_in_instance_ids=[999],
    )

    save_mock = AsyncMock()

    with (
        patch("gpustack.schemas.models.Model.one_by_id", return_value=model),
        patch("gpustack.routes.models.validate_model_in", new=AsyncMock()),
        patch(
            "gpustack.routes.models.ModelInstance.all_by_field",
            return_value=[],
        ),
        patch("gpustack.routes.models.Model.save", new=save_mock),
        patch("gpustack.routes.models.ModelService") as service_cls,
    ):
        try:
            asyncio.run(update_model(AsyncMock(), model.id, model_in))
        except BadRequestException:
            pass
        else:
            raise AssertionError("指定实例无效时应返回 BadRequest")

    service_cls.return_value.update.assert_not_called()
    save_mock.assert_not_awaited()
