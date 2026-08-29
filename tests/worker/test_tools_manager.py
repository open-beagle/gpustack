import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from gpustack.schemas.models import BackendEnum
from gpustack.worker.tools_manager import (
    BUILTIN_LLAMA_BOX_VERSION,
    is_disabled_dynamic_link,
    ToolsManager,
)


def test_install_llama_cpp_discovers_built_in_version(tmp_path, monkeypatch):
    third_party_bin = tmp_path / "third-party"
    version = "b8322"
    version_dir = f"llama.cpp-{version}-linux-amd64-cuda"
    command = third_party_bin / "llama.cpp" / version_dir / "llama-server"
    command.parent.mkdir(parents=True)
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    (third_party_bin / "versions.json").write_text(
        json.dumps({version_dir: version}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GPUSTACK_THIRD_PARTY_BIN", str(third_party_bin))

    manager = ToolsManager(system="linux", arch="amd64", device="cuda")

    assert manager.install_llama_cpp() == command


def test_download_llama_box_reuses_bundled_cuda_build(tmp_path, monkeypatch):
    third_party_bin = tmp_path / "third-party"
    version_dir = f"llama-box-{BUILTIN_LLAMA_BOX_VERSION}-linux-amd64-cuda"
    command = third_party_bin / "llama-box" / version_dir / "llama-box"
    command.parent.mkdir(parents=True)
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    command.chmod(0o755)
    (third_party_bin / "versions.json").write_text(
        json.dumps({version_dir: BUILTIN_LLAMA_BOX_VERSION}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GPUSTACK_THIRD_PARTY_BIN", str(third_party_bin))
    monkeypatch.setenv("GPUSTACK_DISABLE_DYNAMIC_LINK_LLAMA_BOX", "true")
    monkeypatch.setattr(
        "gpustack.worker.tools_manager.platform.system", lambda: "linux"
    )
    monkeypatch.setattr("gpustack.worker.tools_manager.platform.arch", lambda: "amd64")
    monkeypatch.setattr("gpustack.worker.tools_manager.platform.device", lambda: "cuda")

    manager = ToolsManager(system="linux", arch="amd64", device="cuda")

    with patch.object(manager, "_download_llama_box") as download:
        manager.download_llama_box()

    download.assert_not_called()
    assert is_disabled_dynamic_link(BUILTIN_LLAMA_BOX_VERSION) is False
    assert (command.parent / "llama-box-rpc-server").resolve() == command
    assert (
        third_party_bin / "llama-box" / "llama-box-default"
    ).resolve() == command.parent


def test_dynamic_link_disable_env_is_preserved_without_bundled_cuda_build(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GPUSTACK_THIRD_PARTY_BIN", str(tmp_path / "third-party"))
    monkeypatch.setenv("GPUSTACK_DISABLE_DYNAMIC_LINK_LLAMA_BOX", "true")
    monkeypatch.setattr(
        "gpustack.worker.tools_manager.platform.system", lambda: "linux"
    )
    monkeypatch.setattr("gpustack.worker.tools_manager.platform.arch", lambda: "amd64")
    monkeypatch.setattr("gpustack.worker.tools_manager.platform.device", lambda: "cuda")

    assert is_disabled_dynamic_link(BUILTIN_LLAMA_BOX_VERSION) is True


def test_bundled_override_is_not_applied_to_other_platforms(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUSTACK_THIRD_PARTY_BIN", str(tmp_path / "third-party"))
    monkeypatch.setenv("GPUSTACK_DISABLE_DYNAMIC_LINK_LLAMA_BOX", "true")
    monkeypatch.setattr(
        "gpustack.worker.tools_manager.platform.system", lambda: "darwin"
    )
    monkeypatch.setattr("gpustack.worker.tools_manager.platform.arch", lambda: "arm64")
    monkeypatch.setattr("gpustack.worker.tools_manager.platform.device", lambda: "mps")

    assert is_disabled_dynamic_link(BUILTIN_LLAMA_BOX_VERSION) is True


def test_prepare_versioned_backend_supports_vllm_omni():
    manager = ToolsManager(data_dir="/tmp/gpustack", bin_dir="/tmp/gpustack/bin")

    with patch.object(manager, "install_versioned_package_by_pipx") as install:
        manager.prepare_versioned_backend(BackendEnum.VLLM_OMNI, "v0.22.0")

    install.assert_called_once_with("vllm-omni", "v0.22.0")


def test_install_versioned_package_adds_sitecustomize_to_pipx_venv(tmp_path):
    bin_dir = tmp_path / "bin"
    venv_bin = tmp_path / "pipx" / "venvs" / "vllm-omni-v0-22-0" / "bin"
    site_packages = tmp_path / "site-packages"
    bin_dir.mkdir()
    venv_bin.mkdir(parents=True)
    command = venv_bin / "vllm-omni"
    python = venv_bin / "python"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    command.chmod(0o755)
    python.symlink_to(sys.executable)
    target = bin_dir / "vllm-omni_v0.22.0"
    target.symlink_to(command)

    manager = ToolsManager(data_dir=str(tmp_path), bin_dir=str(bin_dir))

    with patch("gpustack.worker.tools_manager.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"{site_packages}\n",
            stderr="",
        )
        manager._install_sitecustomize_for_versioned_package(target)

    installed = site_packages / "sitecustomize.py"
    assert installed.exists()
    source = Path(__file__).resolve().parents[2] / "gpustack" / "_sitecustomize.py"
    assert installed.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
