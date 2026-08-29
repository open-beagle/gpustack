import io
import re
import subprocess
import tarfile
from pathlib import Path

from gpustack.worker.tools_manager import (
    BUILTIN_LLAMA_CPP_CUDA_VERSION,
    get_llama_cpp_package_name,
)


CUDA_VERSION = "13.0.3"
GPUSTACK_VERSION = "0.7.7"
UI_VERSION = "0.7.7"
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
    assert (
        "LLAMA_BOX_SHA256="
        "81b49650f237649996b204b37f3ce5bc3e92aa4a39d3d1539301440dc3c0ab2d"
    ) in lock
    assert "https://github.com/gpustack/llama-box.git" in build_script
    assert "--recurse-submodules" in build_script
    assert "-DGGML_CUDA=ON" in build_script
    assert "-DGGML_RPC=ON" in build_script
    assert "-DBUILD_SHARED_LIBS=ON" in build_script
    assert "lib(cudart|cublas).*\\.so\\.12" in build_script
    assert "dist/llama-box.tar.gz" in workflow
    assert "bash .beagle/build-cuda-llama-box.sh" in workflow
    assert (
        "https://cache.ali.wodcloud.com/vscode/gpustack/llama-box/"
        "${LLAMA_BOX_PACKAGE}.tar.gz"
    ) in workflow
    assert "actual_sha256" in workflow
    assert "--validate-package dist/llama-box.tar.gz" in workflow
    cache_validation = workflow[
        workflow.index('actual_sha256="$(sha256sum') : workflow.index(
            'if [ "${cache_hit}" != "true" ]'
        )
    ]
    assert "docker run" not in cache_validation
    assert "cache_hit" in workflow
    assert "mc cp" not in workflow
    assert "mc mirror" not in workflow
    assert "unsafe path" in build_script
    assert "unsupported special files" in build_script
    assert "NEEDED.*lib(cudart|cublas)" in build_script
    assert "RPATH/RUNPATH" in build_script
    assert "cuda_stub_dir=/usr/local/cuda/lib64/stubs" in build_script
    assert 'ln -s "${cuda_stub_dir}/libcuda.so"' in build_script
    assert build_script.index('if [ "${1:-}" = "--validate-package" ]') < (
        build_script.index("apt-get update")
    )
    assert 'sha256sum "${DIST_DIR}/llama-box.tar.gz"' in build_script
    assert "COPY ./dist/llama-box.tar.gz /tmp/llama-box.tar.gz" in dockerfile
    assert "llama-box-default" in dockerfile
    assert "llama-box-rpc-server" in dockerfile
    assert "llama-box has invalid CUDA runtime dependencies" in dockerfile
    assert "not found|lib(cudart|cublas).*\\.so\\.12" in dockerfile
    assert "cuda_stub_dir=/usr/local/cuda/lib64/stubs" in dockerfile
    assert 'ln -s "${cuda_stub_dir}/libcuda.so"' in dockerfile
    assert 'LD_LIBRARY_PATH="${llama_box_path}:${cuda_stub_dir}' in dockerfile


def test_cuda_llama_box_package_validator_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.tar.gz"
    payload = b"unsafe"
    with tarfile.open(archive_path, "w:gz") as archive:
        entry = tarfile.TarInfo("../escape")
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))

    result = subprocess.run(
        [
            "bash",
            ".beagle/build-cuda-llama-box.sh",
            "--validate-package",
            str(archive_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unsafe path" in result.stderr


def test_cuda_llama_box_package_validator_accepts_valid_elf(tmp_path):
    source_path = tmp_path / "llama-box.c"
    executable_path = tmp_path / "llama-box"
    archive_path = tmp_path / "valid.tar.gz"
    source_path.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    subprocess.run(
        [
            "gcc",
            str(source_path),
            "-Wl,-rpath,$ORIGIN",
            "-o",
            str(executable_path),
        ],
        check=True,
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(executable_path, arcname="llama-box")

    result = subprocess.run(
        [
            "bash",
            ".beagle/build-cuda-llama-box.sh",
            "--validate-package",
            str(archive_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Validated llama-box package" in result.stdout


def test_cuda_llama_box_package_validator_rejects_cuda_12_needed(tmp_path):
    library_source = tmp_path / "cudart.c"
    main_source = tmp_path / "llama-box.c"
    library_path = tmp_path / "libcudart.so.12"
    executable_path = tmp_path / "llama-box"
    archive_path = tmp_path / "cuda12.tar.gz"
    library_source.write_text("int cuda_test(void) { return 0; }\n", encoding="utf-8")
    main_source.write_text(
        "extern int cuda_test(void); int main(void) { return cuda_test(); }\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-Wl,-soname,libcudart.so.12",
            str(library_source),
            "-o",
            str(library_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "gcc",
            str(main_source),
            "-Wl,-rpath,$ORIGIN",
            f"-L{tmp_path}",
            "-Wl,--no-as-needed",
            "-l:libcudart.so.12",
            "-o",
            str(executable_path),
        ],
        check=True,
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(executable_path, arcname="llama-box")
        archive.add(library_path, arcname="libcudart.so.12")

    result = subprocess.run(
        [
            "bash",
            ".beagle/build-cuda-llama-box.sh",
            "--validate-package",
            str(archive_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid CUDA ABI dependencies" in result.stderr


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
