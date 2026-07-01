from unittest.mock import patch

from gpustack.schemas.models import BackendEnum
from gpustack.worker.tools_manager import ToolsManager


def test_prepare_versioned_backend_supports_vllm_omni():
    manager = ToolsManager(data_dir="/tmp/gpustack", bin_dir="/tmp/gpustack/bin")

    with patch.object(manager, "install_versioned_package_by_pipx") as install:
        manager.prepare_versioned_backend(BackendEnum.VLLM_OMNI, "v0.22.0")

    install.assert_called_once_with("vllm-omni", "v0.22.0")
