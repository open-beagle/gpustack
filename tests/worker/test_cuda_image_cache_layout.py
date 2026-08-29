import re
from pathlib import Path

from gpustack.worker.tools_manager import (
    BUILTIN_LLAMA_CPP_CUDA_VERSION,
    get_llama_cpp_package_name,
)


CUDA_VERSION = "13.0.3"
GPUSTACK_VERSION = "0.7.17"
UI_VERSION = "0.7.11"
VLLM_VERSION = "0.26.0"


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _toml_version(pyproject: str, dependency: str) -> str:
    match = re.search(
        rf'^{re.escape(dependency)} = \{{version = "([^"]+)"',
        pyproject,
        re.MULTILINE,
    )
    assert match is not None
    return match.group(1)


def test_cuda_release_versions_are_kept_in_sync():
    pyproject = _read("pyproject.toml")
    base_workflow = _read(".github/workflows/release-cuda-base.yml")
    release_workflow = _read(".github/workflows/release-cuda.yml")
    app_dockerfile = _read(".beagle/cuda.dockerfile")

    assert f'version = "{GPUSTACK_VERSION}"' in pyproject
    assert _toml_version(pyproject, "vllm") == VLLM_VERSION
    assert _toml_version(pyproject, "vllm-omni") == VLLM_VERSION
    assert 'transformers = ">=5.6.0"' in pyproject

    runtime_tag = f"cuda{CUDA_VERSION}-vllm{VLLM_VERSION}-omni{VLLM_VERSION}"
    assert f'RUNTIME_TAG: "{runtime_tag}"' in base_workflow
    assert f'RUNTIME_TAG: "{runtime_tag}"' in release_workflow
    assert f"gpustack:{runtime_tag}" in app_dockerfile
    assert f'VERSION: "v{GPUSTACK_VERSION}"' in release_workflow
    assert f'UI_VERSION: "v{UI_VERSION}"' in release_workflow


def test_cuda_app_uses_bundled_cuda_13_llama_box():
    app_dockerfile = _read(".beagle/cuda.dockerfile")

    assert "GPUSTACK_DISABLE_DYNAMIC_LINK_LLAMA_BOX=true" not in app_dockerfile


def test_cuda_base_builds_and_installs_cuda_13_llama_box():
    workflow = _read(".github/workflows/release-cuda-base.yml")
    dockerfile = _read(".beagle/cuda-base.dockerfile")
    build_script = _read(".beagle/build-cuda-llama-box.sh")
    lock = _read(".beagle/llama-box.lock")

    assert "LLAMA_BOX_VERSION=v0.0.171" in lock
    assert "LLAMA_BOX_COMMIT=437e8041d4db2747d016c2d020415695f53a3159" in lock
    assert "https://github.com/gpustack/llama-box.git" in build_script
    assert "--recurse-submodules" in build_script
    assert "-DGGML_CUDA=ON" in build_script
    assert "-DGGML_RPC=ON" in build_script
    assert "-DBUILD_SHARED_LIBS=ON" in build_script
    assert "llama-box unexpectedly links against CUDA 12" in build_script
    assert "lib(cudart|cublas).*\\.so\\.12" in build_script
    assert "dist/llama-box.tar.gz" in workflow
    assert "bash .beagle/build-cuda-llama-box.sh" in workflow
    assert "gpustack/llama-box/${LLAMA_BOX_PACKAGE}" not in workflow
    assert "COPY ./dist/llama-box.tar.gz /tmp/llama-box.tar.gz" in dockerfile
    assert "llama-box-default" in dockerfile
    assert "llama-box-rpc-server" in dockerfile
    assert "llama-box unexpectedly links against CUDA 12" in dockerfile


def test_cuda_base_builds_and_installs_cuda_13_llama_cpp():
    workflow = _read(".github/workflows/release-cuda-base.yml")
    dockerfile = _read(".beagle/cuda-base.dockerfile")
    build_script = _read(".beagle/build-cuda-tools.sh")

    image_registry = "nvidia/cuda"
    assert f'BASE_IMG: {image_registry}:{CUDA_VERSION}-runtime-ubuntu24.04' in workflow
    assert (
        f'LLAMA_BUILD_IMG: {image_registry}:{CUDA_VERSION}-devel-ubuntu24.04'
        in workflow
    )
    assert "bash .beagle/build-cuda-llama-box.sh" in workflow
    assert 'CUDA_ARCHITECTURES: "75;80;86;89;90;100;120"' in workflow
    assert "LLAMA_CPP_VERSION:" not in workflow

    assert f"ARG BASE={image_registry}:{CUDA_VERSION}-runtime-ubuntu24.04" in dockerfile
    assert "python3-venv" in dockerfile
    assert "python3 -m venv /opt/gpustack/venv" in dockerfile
    assert "PATH=/opt/gpustack/venv/bin:${PATH}" in dockerfile
    assert (
        "/opt/gpustack/venv/lib/python3.12/site-packages/nvidia/nccl/lib" in dockerfile
    )
    assert "libgl1-mesa-glx" not in dockerfile
    assert "libgl1" in dockerfile
    assert "cuda-nvcc-13-0" in dockerfile
    assert "cuda-nvrtc-13-0" in dockerfile
    assert "libcublas-dev-13-0" in dockerfile
    assert "ARG LLAMA_CPP_VERSION" not in dockerfile
    assert "COPY ./.beagle/llama.cpp.lock /tmp/llama.cpp.lock" in dockerfile
    assert "COPY ./dist/llama.cpp.tar.gz /tmp/llama.cpp.tar.gz" in dockerfile
    assert ". /tmp/llama.cpp.lock" in dockerfile
    assert "ggml-rpc-server" in dockerfile
    assert "ggml-rpc-server" in build_script
    assert "llama-box-default" in dockerfile
    assert "python3 -m pip check" in dockerfile
    assert "release 13.0" in dockerfile
    assert "12-8" not in dockerfile

    assert "https://github.com/ggml-org/llama.cpp.git" in build_script
    assert "/etc/apt/sources.list.d/ubuntu.sources" in build_script
    assert "https://mirrors.aliyun.com/ubuntu" in build_script
    assert build_script.index("mirrors.aliyun.com") < build_script.index(
        "apt-get update"
    )
    assert "git ls-remote --tags --refs --sort=-version:refname" in build_script
    assert 'LLAMA_CPP_VERSION="${LLAMA_CPP_VERSION:-' not in build_script
    assert 'LOCK_FILE=/workspace/.beagle/llama.cpp.lock' in build_script
    assert 'mv -f "${LOCK_TMP}" "${LOCK_FILE}"' in build_script
    assert build_script.index("test -x") < build_script.index('mv -f "${LOCK_TMP}"')
    assert "lib(cudart|cublas).*\\.so\\.12" in build_script
    assert "60;61;70" not in build_script


def test_llama_cpp_package_targets_cuda_13():
    assert BUILTIN_LLAMA_CPP_CUDA_VERSION == CUDA_VERSION
    assert (
        get_llama_cpp_package_name("b8322") == "llama-cpp-cuda-13.0.3-b8322-linux-x64"
    )


def test_cuda_base_has_no_transformers_monkey_patch():
    prepare_script = _read(".beagle/prepare_cuda_base.py")

    assert '"torch": "2.11.0"' in prepare_script
    assert '"vllm": "0.26.0"' in prepare_script
    assert '"vllm-omni": "0.26.0"' in prepare_script
    assert 'torch.version.cuda != "13.0"' in prepare_script
    assert "patch_transformers" not in prepare_script
    assert "received_keys" not in prepare_script
