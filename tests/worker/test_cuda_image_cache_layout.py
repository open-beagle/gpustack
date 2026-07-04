from pathlib import Path


def test_cuda_dockerfile_keeps_large_stable_layers_before_wheel():
    dockerfile = Path(".beagle/cuda.dockerfile").read_text()

    deps_layer = dockerfile.index("COPY ./dist/requirements-vllm.txt")
    tools_layer = dockerfile.index("COPY ./dist/gpustack-tools-cuda.tar.gz")
    wheel_layer = dockerfile.index("COPY ./dist/*.whl")
    version_label = dockerfile.index("LABEL version=$VERSION")

    assert deps_layer < tools_layer < wheel_layer < version_label


def test_corex_dockerfile_keeps_tools_layer_before_wheel():
    dockerfile = Path(".beagle/corex.dockerfile").read_text()

    deps_layer = dockerfile.index("COPY ./dist/requirements-vllm.txt")
    tools_layer = dockerfile.index("COPY ./dist/gpustack-tools-corex.tar.gz")
    wheel_layer = dockerfile.index("COPY ./dist/*.whl")

    assert deps_layer < tools_layer < wheel_layer


def test_external_tools_layer_sets_runtime_tool_path():
    cuda_dockerfile = Path(".beagle/cuda.dockerfile").read_text()
    corex_dockerfile = Path(".beagle/corex.dockerfile").read_text()
    expected_env = "GPUSTACK_THIRD_PARTY_BIN=/opt/gpustack/third_party/bin"

    assert expected_env in cuda_dockerfile
    assert expected_env in corex_dockerfile


def test_build_script_creates_cuda_tools_archive_before_docker_build():
    build_script = Path(".beagle/build.sh").read_text()

    assert "gpustack-tools-cuda.tar.gz" in build_script
    assert "gpustack-tools-corex.tar.gz" in build_script
    assert "fastapi pydantic sqlmodel sqlalchemy" in build_script
    assert 'local tools_bin="$tools_venv/third_party/bin"' in build_script
    assert 'GPUSTACK_THIRD_PARTY_BIN="$tools_bin"' in build_script
    assert "save_archive" in build_script


def test_build_script_skips_llama_box_for_corex_tools_archive():
    build_script = Path(".beagle/build.sh").read_text()

    assert "generate_tools_archive cuda gpustack-tools-cuda.tar.gz true" in build_script
    assert (
        "generate_tools_archive corex gpustack-tools-corex.tar.gz false"
        in build_script
    )
    assert "if '${include_llama_box}' == 'true':" in build_script
