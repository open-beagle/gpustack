from gpustack.schemas.models import (
    CategoryEnum,
    is_audio_model,
    is_embedding_model,
    is_image_model,
    model_categories,
    is_renaker_model,
)


class LegacyModel:
    backend = None
    categories = None


def test_category_helpers_treat_legacy_null_categories_as_empty():
    model = LegacyModel()

    assert is_audio_model(model) is False
    assert is_image_model(model) is False
    assert is_embedding_model(model) is False
    assert is_renaker_model(model) is False
    assert model_categories(model) == []


class LLMModel:
    backend = None
    categories = [CategoryEnum.LLM]


def test_model_categories_preserves_non_empty_categories():
    assert model_categories(LLMModel()) == [CategoryEnum.LLM]
