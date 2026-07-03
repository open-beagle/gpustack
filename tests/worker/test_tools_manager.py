import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from gpustack.schemas.models import BackendEnum
from gpustack.worker.tools_manager import ToolsManager


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
