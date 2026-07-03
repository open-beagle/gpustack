import copy

from gpustack.schemas.models import Model, ModelInstance


def get_model_for_instance_scheduling(
    model: Model, instance: ModelInstance
) -> Model:
    if (
        not instance.placement_override
        or not instance.placement_override.gpu_selector
        or not instance.placement_override.gpu_selector.gpu_ids
    ):
        return model

    scheduling_model = copy.deepcopy(model)
    scheduling_model.gpu_selector = instance.placement_override.gpu_selector
    return scheduling_model
