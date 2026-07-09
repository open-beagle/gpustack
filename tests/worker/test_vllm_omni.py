from gpustack.schemas.models import CategoryEnum
from gpustack.worker.backends.vllm_omni_args import (
    build_vllm_omni_arguments,
    detect_model_type,
)


def _arguments(backend_parameters=None, gpu_count=0):
    return build_vllm_omni_arguments(
        "/models/qwen-image",
        "qwen-image-2512",
        "qwen-image-2512",
        [CategoryEnum.IMAGE],
        backend_parameters,
        40000,
        gpu_count,
    )


def test_vllm_omni_arguments_enable_omni_mode():
    arguments = _arguments()

    assert arguments[:2] == ["serve", "/models/qwen-image"]
    assert "--omni" in arguments
    assert arguments[arguments.index("--served-model-name") + 1] == "qwen-image-2512"


def test_vllm_omni_arguments_do_not_duplicate_user_omni_flag():
    arguments = _arguments(["--omni", "--trust-remote-code"])

    assert arguments.count("--omni") == 1
    assert "--trust-remote-code" in arguments


def test_vllm_omni_diffusion_arguments_add_num_gpus_for_multi_gpu():
    arguments = _arguments(gpu_count=2)

    assert arguments[arguments.index("--num-gpus") + 1] == "2"


def test_vllm_omni_diffusion_arguments_keep_user_num_gpus():
    arguments = _arguments(["--num-gpus=4"], gpu_count=2)

    assert "--num-gpus=4" in arguments
    assert "--num-gpus" not in arguments


def test_vllm_omni_detects_video_model_type():
    assert detect_model_type("Wan2.2-T2V", [CategoryEnum.VIDEO]) == "video"


def test_vllm_omni_detects_omnigen_model_type_without_category():
    assert detect_model_type("OmniGen2", []) == "diffusion"


def test_vllm_omni_keeps_audio_model_type_when_explicitly_selected():
    assert detect_model_type("whisper-large", [CategoryEnum.SPEECH_TO_TEXT]) == "audio"
