from types import SimpleNamespace

import pytest

from gpustack.schemas.models import BackendEnum, Model, SourceEnum, get_gguf_runtime
from gpustack.utils.command import ensure_bool_parameter
from gpustack.worker import serve_manager
from gpustack.worker.backends.llama_box import (
    LlamaBoxServer,
    normalize_llama_cpp_parameters,
)
from gpustack.worker.tools_manager import (
    get_llama_cpp_package_name,
    get_llama_cpp_version_dir_name,
)


def test_ensure_bool_parameter_adds_default_metrics_flag():
    arguments = ["--host", "0.0.0.0"]

    got = ensure_bool_parameter(arguments, "metrics")

    assert got == ["--host", "0.0.0.0", "--metrics"]
    assert arguments == ["--host", "0.0.0.0"]


def test_ensure_bool_parameter_does_not_duplicate_user_metrics_flag():
    arguments = ["--host", "0.0.0.0", "--metrics"]

    got = ensure_bool_parameter(arguments, "metrics")

    assert got is arguments


def test_ensure_bool_parameter_respects_existing_metrics_parameter():
    arguments = ["--host", "0.0.0.0"]

    got = ensure_bool_parameter(
        arguments,
        "metrics",
        existing_parameters=["--metrics"],
    )

    assert got is arguments


def test_ensure_bool_parameter_respects_existing_metrics_equals_parameter():
    arguments = ["--host", "0.0.0.0"]

    got = ensure_bool_parameter(
        arguments,
        "metrics",
        existing_parameters=["--metrics=true"],
    )

    assert got is arguments


def test_get_gguf_runtime_defaults_to_llama_box():
    assert (
        get_gguf_runtime(SimpleNamespace(env=None, backend_parameters=None))
        == "llama-box"
    )


def test_get_gguf_runtime_prefers_env():
    assert (
        get_gguf_runtime(
            SimpleNamespace(
                env={"GPUSTACK_GGUF_RUNTIME": "llama-cpp"},
                backend_parameters=["--gpustack-runtime=llama-box"],
            )
        )
        == "llama-cpp"
    )


def test_get_gguf_runtime_reads_backend_parameter():
    assert (
        get_gguf_runtime(
            SimpleNamespace(env={}, backend_parameters=["--gpustack-runtime=llama-cpp"])
        )
        == "llama-cpp"
    )


def test_llama_cpp_runtime_defaults_distributed_inference_to_false():
    model = Model(
        name="qwen-gguf",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="unsloth/Qwen-GGUF",
        huggingface_filename="qwen.gguf",
        backend=BackendEnum.LLAMA_BOX,
        env={"GPUSTACK_GGUF_RUNTIME": "llama-cpp"},
    )

    assert model.distributed_inference_across_workers is False


def test_llama_cpp_runtime_rejects_distributed_inference():
    with pytest.raises(
        ValueError,
        match=(
            "Distributed inference across workers is not supported "
            "for the llama.cpp runtime"
        ),
    ):
        Model(
            name="qwen-gguf",
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="unsloth/Qwen-GGUF",
            huggingface_filename="qwen.gguf",
            backend=BackendEnum.LLAMA_BOX,
            env={"GPUSTACK_GGUF_RUNTIME": "llama-cpp"},
            distributed_inference_across_workers=True,
        )


def test_normalize_llama_cpp_parameters_removes_internal_and_unsupported_flags():
    got = normalize_llama_cpp_parameters(
        [
            "--gpustack-runtime=llama-cpp",
            "--ctx-size=16384",
            "--rpc",
            "10.0.0.1:40064",
            "--images",
            "--mmproj",
            "mmproj.gguf",
            "--metrics",
        ]
    )

    assert got == ["--ctx-size", "16384", "--mmproj", "mmproj.gguf"]


def test_normalize_llama_cpp_parameters_preserves_regular_runtime_flag():
    got = normalize_llama_cpp_parameters(
        [
            "--runtime",
            "some-llama-cpp-value",
        ]
    )

    assert got == ["--runtime", "some-llama-cpp-value"]


def test_get_llama_cpp_package_name_matches_uploaded_artifact():
    assert (
        get_llama_cpp_package_name("b8322")
        == "llama-cpp-cuda-12.8.1-b8322-linux-x64"
    )


def test_get_llama_cpp_version_dir_name():
    assert (
        get_llama_cpp_version_dir_name("b8322", "linux", "amd64", "cuda")
        == "llama.cpp-b8322-linux-amd64-cuda"
    )


def test_llama_cpp_runtime_rejects_model_level_distributed_inference():
    server = object.__new__(LlamaBoxServer)
    server._model = SimpleNamespace(distributed_inference_across_workers=True)
    server._model_instance = SimpleNamespace(distributed_servers=None)

    with pytest.raises(
        RuntimeError,
        match="does not support distributed inference across workers",
    ):
        server._ensure_llama_cpp_supported()


def test_llama_box_health_check_falls_back_to_v1_models(monkeypatch):
    requested_urls = []

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

    def fake_get(url, timeout):
        requested_urls.append(url)
        if url.endswith("/health"):
            return Response(404)
        return Response(200)

    monkeypatch.setattr(serve_manager.requests, "get", fake_get)

    assert (
        serve_manager.is_ready(
            BackendEnum.LLAMA_BOX,
            SimpleNamespace(port=18080, worker_ip="10.0.0.10"),
        )
        is True
    )
    assert requested_urls == [
        "http://127.0.0.1:18080/health",
        "http://127.0.0.1:18080/v1/models",
    ]


def test_llama_box_health_check_falls_back_after_health_exception(monkeypatch):
    requested_urls = []

    class Response:
        status_code = 200

    def fake_get(url, timeout):
        requested_urls.append(url)
        if url.endswith("/health"):
            raise RuntimeError("health endpoint failed")
        return Response()

    monkeypatch.setattr(serve_manager.requests, "get", fake_get)

    assert (
        serve_manager.is_ready(
            BackendEnum.LLAMA_BOX,
            SimpleNamespace(port=18080, worker_ip="10.0.0.10"),
        )
        is True
    )
    assert requested_urls == [
        "http://127.0.0.1:18080/health",
        "http://127.0.0.1:18080/v1/models",
    ]
