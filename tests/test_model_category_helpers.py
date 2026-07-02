from gpustack.schemas.models import (
    BackendEnum,
    CategoryEnum,
    SourceEnum,
    get_backend,
    is_audio_model,
    is_embedding_model,
    is_image_model,
    is_diffusers_model,
    is_vllm_omni_model,
    is_video_model,
    model_categories,
    is_renaker_model,
)


class LegacyModel:
    backend = None
    categories = None
    source = SourceEnum.HUGGING_FACE
    huggingface_filename = None
    huggingface_repo_id = None
    model_scope_file_path = None
    model_scope_model_id = None
    local_path = None


def test_category_helpers_treat_legacy_null_categories_as_empty():
    model = LegacyModel()

    assert is_audio_model(model) is False
    assert is_image_model(model) is False
    assert is_video_model(model) is False
    assert is_embedding_model(model) is False
    assert is_renaker_model(model) is False
    assert model_categories(model) == []


class LLMModel:
    backend = None
    categories = [CategoryEnum.LLM]


def test_model_categories_preserves_non_empty_categories():
    assert model_categories(LLMModel()) == [CategoryEnum.LLM]


class ImageModel:
    backend = None
    categories = [CategoryEnum.IMAGE]
    source = SourceEnum.MODEL_SCOPE
    name = "qwen-image"
    huggingface_filename = None
    huggingface_repo_id = None
    model_scope_file_path = None
    model_scope_model_id = None
    local_path = None


def test_get_backend_defaults_image_model_to_vllm_omni():
    assert get_backend(ImageModel()) == BackendEnum.VLLM_OMNI


def test_get_backend_preserves_explicit_backend_for_image_model():
    model = ImageModel()
    model.backend = BackendEnum.VLLM

    assert get_backend(model) == BackendEnum.VLLM


class LocalModel:
    backend = None
    categories = []
    source = SourceEnum.LOCAL_PATH
    name = "local-model"
    huggingface_filename = None
    huggingface_repo_id = None
    model_scope_file_path = None
    model_scope_model_id = None

    def __init__(self, local_path):
        self.local_path = local_path


def test_get_backend_defaults_diffusers_local_model_to_vllm_omni(tmp_path):
    model_dir = tmp_path / "qwen-image"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    model = LocalModel(str(model_dir))

    assert is_diffusers_model(model)
    assert get_backend(model) == BackendEnum.VLLM_OMNI


class NamedModel:
    backend = None
    categories = [CategoryEnum.LLM]
    source = SourceEnum.HUGGING_FACE
    huggingface_filename = None
    model_scope_file_path = None
    local_path = None

    def __init__(self, name, repo_id=None):
        self.name = name
        self.huggingface_repo_id = repo_id
        self.model_scope_model_id = None


class VideoModel(NamedModel):
    categories = [CategoryEnum.VIDEO]


class AudioModel(NamedModel):
    categories = [CategoryEnum.SPEECH_TO_TEXT]


def test_get_backend_defaults_omni_named_model_to_vllm_omni():
    model = NamedModel("Qwen2.5-Omni-7B")

    assert is_vllm_omni_model(model)
    assert get_backend(model) == BackendEnum.VLLM_OMNI


def test_get_backend_keeps_omni_gguf_model_on_llama_box():
    model = NamedModel("Qwen2.5-Omni-7B-Q4_K_M")
    model.huggingface_filename = "Qwen2.5-Omni-7B-Q4_K_M.gguf"

    assert get_backend(model) == BackendEnum.LLAMA_BOX


def test_get_backend_defaults_video_model_to_vllm_omni():
    model = VideoModel("wan-video")

    assert is_video_model(model)
    assert is_vllm_omni_model(model)
    assert get_backend(model) == BackendEnum.VLLM_OMNI


def test_get_backend_keeps_audio_model_on_vox_box_by_default():
    model = AudioModel("whisper-large")

    assert is_audio_model(model)
    assert is_vllm_omni_model(model) is False
    assert get_backend(model) == BackendEnum.VOX_BOX


def test_explicit_vllm_omni_audio_model_does_not_use_vox_box_helpers():
    model = AudioModel("whisper-large")
    model.backend = BackendEnum.VLLM_OMNI

    assert is_audio_model(model) is False
    assert get_backend(model) == BackendEnum.VLLM_OMNI


def test_get_backend_keeps_regular_llm_on_vllm():
    model = NamedModel("Qwen2.5-VL-7B")

    assert is_vllm_omni_model(model) is False
    assert get_backend(model) == BackendEnum.VLLM
