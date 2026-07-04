from pathlib import Path


def test_cuda_dockerfile_keeps_large_stable_layers_before_wheel():
    dockerfile = Path(".beagle/cuda.dockerfile").read_text()

    deps_layer = dockerfile.index("COPY ./dist/requirements-vllm.txt")
    tools_layer = dockerfile.index("COPY ./dist/gpustack-tools-cuda.tar.gz")
    wheel_layer = dockerfile.index("COPY ./dist/*.whl")
    version_label = dockerfile.index("LABEL version=$VERSION")

    assert deps_layer < tools_layer < wheel_layer < version_label


def test_build_script_creates_cuda_tools_archive_before_docker_build():
    build_script = Path(".beagle/build.sh").read_text()

    assert "gpustack-tools-cuda.tar.gz" in build_script
    assert "fastapi pydantic sqlmodel sqlalchemy" in build_script
    assert "save_archive" in build_script
