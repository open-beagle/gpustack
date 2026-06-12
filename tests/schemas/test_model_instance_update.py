from gpustack.schemas.models import (
    ModelInstanceCreate,
    ModelInstanceInternalCreate,
    ModelInstanceInternalUpdate,
    ModelInstanceUpdate,
)


def _instance_payload():
    return {
        "name": "instance-a",
        "model_id": 1,
        "model_name": "model-a",
        "source": "local_path",
        "local_path": "/models/a",
    }


def test_model_instance_create_does_not_expose_placement_override():
    assert "placement_override" not in ModelInstanceCreate.model_fields

    create = ModelInstanceCreate(
        **_instance_payload(),
        placement_override={"gpu_selector": {"gpu_ids": ["host-a:cuda:0"]}},
    )

    assert not hasattr(create, "placement_override")


def test_model_instance_internal_create_exposes_placement_override():
    assert "placement_override" in ModelInstanceInternalCreate.model_fields

    create = ModelInstanceInternalCreate(
        **_instance_payload(),
        placement_override={"gpu_selector": {"gpu_ids": ["host-a:cuda:0"]}},
    )

    assert create.placement_override.gpu_selector.gpu_ids == ["host-a:cuda:0"]


def test_model_instance_update_does_not_expose_placement_override():
    assert "placement_override" not in ModelInstanceUpdate.model_fields

    update = ModelInstanceUpdate(
        **_instance_payload(),
        placement_override={"gpu_selector": {"gpu_ids": ["host-a:cuda:0"]}},
    )

    assert not hasattr(update, "placement_override")


def test_model_instance_internal_update_exposes_placement_override():
    assert "placement_override" in ModelInstanceInternalUpdate.model_fields

    update = ModelInstanceInternalUpdate(
        **_instance_payload(),
        placement_override={"gpu_selector": {"gpu_ids": ["host-a:cuda:0"]}},
    )

    assert update.placement_override.gpu_selector.gpu_ids == ["host-a:cuda:0"]
