import os
import subprocess
import textwrap
from pathlib import Path

import pytest


def _load_runtime_env() -> dict[str, str]:
    values = {}
    for line in Path(".beagle/cuda-runtime.env").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        value = value.strip('"')
        values[key] = value
    return values


def test_cuda_app_dockerfile_uses_runtime_base_and_only_installs_wheel():
    runtime_env = _load_runtime_env()
    dockerfile = Path(".beagle/cuda.dockerfile").read_text()

    assert runtime_env["WINDSTACK_CUDA_BASE_IMAGE"] in dockerfile
    assert "COPY ./dist/requirements-vllm.txt" not in dockerfile
    assert "COPY ./dist/gpustack-tools-cuda.tar.gz" not in dockerfile
    assert "PYPI_MIRROR" not in dockerfile
    assert "PYPI_HOST" not in dockerfile
    assert "pip3 install" in dockerfile
    assert "--no-deps --force-reinstall" in dockerfile
    assert "GPUSTACK_THIRD_PARTY_BIN=/opt/gpustack/third_party/bin" in dockerfile
    assert "GPUSTACK_THIRD_PARTY_BIN=/var/lib/gpustack/third_party/bin" not in dockerfile
    assert "download_llama_box()" in dockerfile
    assert "download_gguf_parser()" in dockerfile
    assert "download_fastfetch()" in dockerfile
    assert "install_llama_cpp()" in dockerfile


def test_cuda_base_dockerfile_keeps_only_stable_runtime_layers():
    dockerfile = Path(".beagle/cuda-base.dockerfile").read_text()

    deps_layer = dockerfile.index("COPY ./dist/requirements-vllm.txt")
    env_layer = dockerfile.index("PIPX_HOME=/var/lib/gpustack/pipx")

    assert deps_layer < env_layer
    assert "COPY ./dist/gpustack-tools-cuda.tar.gz" not in dockerfile
    assert "GPUSTACK_THIRD_PARTY_BIN=/opt/gpustack/third_party/bin" not in dockerfile


def test_corex_dockerfile_downloads_tools_without_tools_archive_layer():
    dockerfile = Path(".beagle/corex.dockerfile").read_text()

    deps_layer = dockerfile.index("COPY ./dist/requirements-vllm.txt")
    wheel_layer = dockerfile.index("COPY ./dist/*.whl")

    assert deps_layer < wheel_layer
    assert "COPY ./dist/gpustack-tools-corex.tar.gz" not in dockerfile
    assert "download_gguf_parser()" in dockerfile
    assert "download_fastfetch()" in dockerfile
    assert "GPUSTACK_THIRD_PARTY_BIN=/opt/gpustack/third_party/bin" in dockerfile
    assert "GPUSTACK_THIRD_PARTY_BIN=/var/lib/gpustack/third_party/bin" not in dockerfile


def test_cuda_runtime_base_tag_is_kept_in_sync():
    runtime_env = _load_runtime_env()
    cuda_dockerfile = Path(".beagle/cuda.dockerfile").read_text()
    pipeline = Path(".beagle.yml").read_text()

    assert runtime_env["WINDSTACK_CUDA_BASE_IMAGE"] in cuda_dockerfile
    assert f'version: "{runtime_env["WINDSTACK_CUDA_BASE_TAG"]}"' in pipeline
    assert "repo: wod/windstackbase" in pipeline


def test_cuda_runtime_versions_match_python_dependencies():
    runtime_env = _load_runtime_env()
    pyproject = Path("pyproject.toml").read_text()

    assert f'vllm = {{version = "{runtime_env["VLLM_VERSION"]}"' in pyproject
    assert (
        f'vllm-omni = {{version = "{runtime_env["VLLM_OMNI_VERSION"]}"'
        in pyproject
    )
    assert f'vllm{runtime_env["VLLM_VERSION"]}' in runtime_env[
        "WINDSTACK_CUDA_BASE_TAG"
    ]
    assert f'omni{runtime_env["VLLM_OMNI_VERSION"]}' in runtime_env[
        "WINDSTACK_CUDA_BASE_TAG"
    ]


def test_build_script_exports_runtime_requirements_without_tool_archives():
    build_script = Path(".beagle/build.sh").read_text()

    assert "requirements-vllm.txt" in build_script
    assert "generate_tools_archive" not in build_script
    assert "gpustack-tools-cuda.tar.gz" not in build_script
    assert "gpustack-tools-corex.tar.gz" not in build_script


def test_cuda_pipeline_has_conditional_runtime_base_build():
    pipeline = Path(".beagle.yml").read_text()
    runtime_env = _load_runtime_env()

    assert "BUILD_RUNTIME_ASSETS" not in pipeline
    assert "branch:\n    - release-cuda" in pipeline
    assert "event:\n    - push" in pipeline
    assert "name: docker-cuda-base" in pipeline
    assert "if ! .beagle/should-build-cuda-base.sh; then" in pipeline
    assert "/opt/bin/devops-docker" in pipeline
    assert "target:\n    - cuda-base" not in pipeline
    assert "release-cuda-base" not in pipeline
    assert "dockerfile: .beagle/cuda-base.dockerfile" in pipeline
    assert "repo: wod/windstackbase" in pipeline
    assert f'version: "{runtime_env["WINDSTACK_CUDA_BASE_TAG"]}"' in pipeline


def test_cuda_base_change_detector_tracks_runtime_inputs():
    detector = Path(".beagle/should-build-cuda-base.sh").read_text()

    assert ".beagle/cuda-runtime.env" in detector
    assert ".beagle/cuda-base.dockerfile" in detector
    assert ".beagle/build.sh" in detector
    assert "pyproject.toml" in detector
    assert "poetry.lock" in detector
    assert "gpustack/worker/tools_manager.py" not in detector


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _commit_all(repo: Path, message: str) -> str:
    _run_git(repo, "add", ".")
    _run_git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        message,
    )
    return _run_git(repo, "rev-parse", "HEAD")


def _write(repo: Path, relative_path: str, content: str) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def _make_detector_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    _write(
        repo,
        ".beagle/should-build-cuda-base.sh",
        Path(".beagle/should-build-cuda-base.sh").read_text(),
    )
    (repo / ".beagle/should-build-cuda-base.sh").chmod(0o755)
    _write(
        repo,
        ".beagle/cuda-runtime.env",
        """
        CUDA_BASE_IMAGE=registry-vpc.cn-qingdao.aliyuncs.com/wod/cuda:12.8.1-runtime-ubuntu22.04
        WINDSTACK_CUDA_BASE_REPO=registry-vpc.cn-qingdao.aliyuncs.com/wod/windstackbase
        WINDSTACK_CUDA_BASE_TAG=cuda12.8.1-py3.10-vllm0.22.1-omni0.22.0
        WINDSTACK_CUDA_BASE_IMAGE=registry-vpc.cn-qingdao.aliyuncs.com/wod/windstackbase:cuda12.8.1-py3.10-vllm0.22.1-omni0.22.0
        VLLM_VERSION=0.22.1
        VLLM_OMNI_VERSION=0.22.0
        TRANSFORMERS_SPEC=">=5.6.0,<5.9.0"
        """,
    )
    _write(
        repo,
        ".beagle/cuda-base.dockerfile",
        """
        ARG BASE=registry-vpc.cn-qingdao.aliyuncs.com/wod/cuda:12.8.1-runtime-ubuntu22.04
        FROM $BASE
        RUN apt-get update && apt-get install -y python3 python3-pip cuda-nvcc-12-8
        COPY ./dist/requirements-vllm.txt /tmp/requirements-vllm.txt
        RUN pip3 install -r /tmp/requirements-vllm.txt
        ENV PIPX_HOME=/var/lib/gpustack/pipx
        """,
    )
    _write(
        repo,
        ".beagle/build.sh",
        """
        # 从 wheel 元数据导出运行时依赖清单，供 CUDA 基础镜像和 CoreX 镜像安装稳定依赖。
        python3 - <<'PY'
        requirements_path = "dist/requirements-vllm.txt"
        PY
        """,
    )
    _write(
        repo,
        ".beagle.yml",
        """
        - name: docker-cuda-base
          commands:
            - |
              if ! .beagle/should-build-cuda-base.sh; then
                exit 0
              fi
          settings:
            base: registry-vpc.cn-qingdao.aliyuncs.com/wod/cuda:12.8.1-runtime-ubuntu22.04
            dockerfile: .beagle/cuda-base.dockerfile
            repo: wod/windstackbase
            version: "cuda12.8.1-py3.10-vllm0.22.1-omni0.22.0"
        """,
    )
    _write(
        repo,
        "pyproject.toml",
        """
        [tool.poetry]
        name = "gpustack"
        version = "0.7.5"

        [tool.poetry.dependencies]
        python = ">=3.10,<3.13"
        openai = ">=2.0.0,<3.0.0"
        ray = {version = "2.48.0", extras = ["default"]}
        vllm = {version = "0.22.1", optional = true}
        vllm-omni = {version = "0.22.0", optional = true}
        mistral_common = {version = "^1.4.3", optional = true, extras = ["opencv"]}
        transformers = ">=5.6.0,<5.9.0"
        bitsandbytes = {version = "^0.45.2", optional = true}
        timm = {version = "^1.0.15", optional = true}

        [tool.poetry.extras]
        vllm = ["vllm", "vllm-omni", "mistral_common", "bitsandbytes", "timm"]
        all = ["vllm", "vllm-omni", "mistral_common", "bitsandbytes", "timm"]
        """,
    )
    _write(
        repo,
        "poetry.lock",
        """
        [[package]]
        name = "vllm"
        version = "0.22.1"

        [[package]]
        name = "transformers"
        version = "5.7.0"
        """,
    )
    _write(
        repo,
        "gpustack/worker/tools_manager.py",
        '''
        BUILTIN_LLAMA_BOX_VERSION = "v0.0.171"
        BUILTIN_GGUF_PARSER_VERSION = "v0.22.1"
        BUILTIN_LLAMA_CPP_VERSION = "b8322"
        BUILTIN_LLAMA_CPP_CUDA_VERSION = "12.8.1"

        class ToolsManager:
            def download_llama_box(self):
                version = BUILTIN_LLAMA_BOX_VERSION
                url_path = f"gpustack/llama-box/releases/download/{version}/linux.zip"

            def _download_llama_box(self, version, target_dir, file_name, disabled_dynamic_link=False):
                platform_name = self._get_llama_box_platform_name()
                url_path = f"gpustack/llama-box/releases/download/{version}/{platform_name}.zip"

            def _get_llama_box_platform_name(self):
                return "llama-box-linux-amd64-cuda-12.8"

            def download_gguf_parser(self):
                version = BUILTIN_GGUF_PARSER_VERSION
                url_path = f"gpustack/gguf-parser-go/releases/download/{version}/gguf-parser-linux-amd64"

            def _get_gguf_parser_platform_name(self):
                return "linux-amd64"

            def download_fastfetch(self):
                version = "2.25.0.1"
                url_path = f"gpustack/fastfetch/releases/download/{version}/fastfetch-linux-amd64.zip"

            def _get_fastfetch_platform_name(self):
                return "linux-amd64"

            def save_archive(self, archive):
                pass

            def remove_cached_tools(self):
                pass
        ''',
    )
    return repo


def _run_detector(repo: Path, before: str, after: str) -> int:
    env = os.environ.copy()
    env.pop("BUILD_RUNTIME_ASSETS", None)
    env["DRONE_COMMIT_BEFORE"] = before
    env["DRONE_COMMIT_AFTER"] = after
    result = subprocess.run(
        [".beagle/should-build-cuda-base.sh"],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode


def _run_detector_with_env(repo: Path, env_updates: dict[str, str]) -> int:
    env = os.environ.copy()
    env.pop("BUILD_RUNTIME_ASSETS", None)
    env.update(env_updates)
    result = subprocess.run(
        [".beagle/should-build-cuda-base.sh"],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode


def test_cuda_base_change_detector_fetches_missing_before_commit(tmp_path):
    repo = _make_detector_repo(tmp_path)
    before = _commit_all(repo, "initial")

    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace('version = "0.7.5"', 'version = "0.7.6"'),
        encoding="utf-8",
    )
    after = _commit_all(repo, "application version change")

    origin = tmp_path / "origin.git"
    shallow = tmp_path / "shallow"
    _run_git(tmp_path, "clone", "--bare", str(repo), str(origin))
    _run_git(
        tmp_path,
        "clone",
        "--depth=1",
        f"file://{origin}",
        str(shallow),
    )

    assert _run_detector(shallow, before, after) == 1


def test_cuda_base_change_detector_skips_when_missing_commit_cannot_be_fetched(
    tmp_path,
):
    repo = _make_detector_repo(tmp_path)
    before = _commit_all(repo, "initial")

    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace('version = "0.7.5"', 'version = "0.7.6"'),
        encoding="utf-8",
    )
    after = _commit_all(repo, "application version change")

    origin = tmp_path / "origin.git"
    shallow = tmp_path / "shallow"
    _run_git(tmp_path, "clone", "--bare", str(repo), str(origin))
    _run_git(
        tmp_path,
        "clone",
        "--depth=1",
        f"file://{origin}",
        str(shallow),
    )
    _run_git(shallow, "remote", "remove", "origin")

    assert _run_detector(shallow, before, after) == 1


def test_cuda_base_change_detector_skips_when_commit_range_is_missing(tmp_path):
    repo = _make_detector_repo(tmp_path)
    _commit_all(repo, "initial")

    assert _run_detector_with_env(repo, {}) == 1


def test_cuda_base_change_detector_ignores_non_runtime_project_metadata(tmp_path):
    repo = _make_detector_repo(tmp_path)
    before = _commit_all(repo, "initial")

    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace('version = "0.7.5"', 'version = "0.7.6"'),
        encoding="utf-8",
    )
    after = _commit_all(repo, "application version change")

    assert _run_detector(repo, before, after) == 1


@pytest.mark.parametrize(
    ("relative_path", "old", "new"),
    [
        (
            ".beagle/cuda-runtime.env",
            "WINDSTACK_CUDA_BASE_TAG=cuda12.8.1-py3.10-vllm0.22.1-omni0.22.0",
            "WINDSTACK_CUDA_BASE_TAG=cuda12.8.1-py3.10-vllm0.23.0-omni0.22.0",
        ),
        (
            ".beagle/cuda-base.dockerfile",
            "cuda-nvcc-12-8",
            "cuda-nvcc-12-9",
        ),
        (
            "pyproject.toml",
            'vllm = {version = "0.22.1"',
            'vllm = {version = "0.23.0"',
        ),
        (
            "pyproject.toml",
            'vllm-omni = {version = "0.22.0"',
            'vllm-omni = {version = "0.23.0"',
        ),
        (
            "pyproject.toml",
            'transformers = ">=5.6.0,<5.9.0"',
            'transformers = ">=5.8.0,<5.10.0"',
        ),
        (
            "poetry.lock",
            'version = "5.7.0"',
            'version = "5.8.0"',
        ),
    ],
)
def test_cuda_base_change_detector_detects_runtime_input_changes(
    tmp_path, relative_path, old, new
):
    repo = _make_detector_repo(tmp_path)
    before = _commit_all(repo, "initial")

    changed_file = repo / relative_path
    changed_file.write_text(
        changed_file.read_text().replace(old, new), encoding="utf-8"
    )
    after = _commit_all(repo, "runtime input change")

    assert _run_detector(repo, before, after) == 0


def test_cuda_base_change_detector_ignores_tool_manager_changes(tmp_path):
    repo = _make_detector_repo(tmp_path)
    before = _commit_all(repo, "initial")

    tools_manager = repo / "gpustack/worker/tools_manager.py"
    tools_manager.write_text(
        tools_manager.read_text().replace(
            'BUILTIN_LLAMA_BOX_VERSION = "v0.0.171"',
            'BUILTIN_LLAMA_BOX_VERSION = "v0.0.172"',
        ),
        encoding="utf-8",
    )
    after = _commit_all(repo, "llama box version change")

    assert _run_detector(repo, before, after) == 1
