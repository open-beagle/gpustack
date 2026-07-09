from gpustack.utils.command import (
    ensure_bool_parameter,
    find_bool_parameter,
    find_parameter,
)


def build_vllm_omni_arguments(
    model_path: str,
    model_name: str,
    served_model_name: str,
    model_categories: list,
    backend_parameters: list,
    port: int,
    gpu_count: int = 0,
) -> list:
    arguments = [
        "serve",
        model_path,
    ]

    model_type = detect_model_type(model_name, model_categories)
    if model_type == "diffusion":
        arguments.extend(get_diffusion_arguments(backend_parameters, gpu_count))
    elif model_type == "audio":
        arguments.extend(get_audio_arguments())

    if backend_parameters:
        arguments.extend(backend_parameters)

    arguments = ensure_bool_parameter(
        arguments,
        "omni",
        existing_parameters=backend_parameters,
    )

    built_in_arguments = [
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
    ]
    arguments.extend(built_in_arguments)

    return arguments


def detect_model_type(model_name: str, categories: list) -> str:
    model_name = model_name.lower()
    categories = categories or []

    diffusion_keywords = [
        "qwen-image",
        "omnigen",
        "flux",
        "z-image",
        "stable-diffusion",
        "sd3",
        "sdxl",
        "dit",
        "lumina",
        "hunyuan-dit",
        "pixart",
    ]
    if "image" in categories or any(kw in model_name for kw in diffusion_keywords):
        return "diffusion"

    video_keywords = ["video", "wan", "cogvideo", "ltx-video", "hunyuan-video"]
    if "video" in categories or any(kw in model_name for kw in video_keywords):
        return "video"

    audio_keywords = ["whisper", "tts", "stt", "audio", "speech"]
    if any(cat in categories for cat in ["speech_to_text", "text_to_speech"]):
        return "audio"
    if any(kw in model_name for kw in audio_keywords):
        return "audio"

    return "llm"


def get_diffusion_arguments(backend_parameters: list, gpu_count: int = 0) -> list:
    if (
        gpu_count <= 1
        or find_parameter(backend_parameters, ["num-gpus"]) is not None
        or find_parameter(backend_parameters, ["tensor-parallel-size", "tp"])
        is not None
        or find_bool_parameter(backend_parameters, ["use-hsdp"])
    ):
        return []

    return ["--num-gpus", str(gpu_count)]


def get_audio_arguments() -> list:
    return []
