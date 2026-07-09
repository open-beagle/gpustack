"""
vLLM-Omni backend for GPUStack.

vLLM-Omni extends vLLM to support omni-modality model inference and serving,
including diffusion models like Z-Image, Flux, etc.

Reference: https://github.com/vllm-project/vllm-omni
"""

import logging
import os
import subprocess
import sys

from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.utils.command import (
    get_versioned_command,
    get_command_path,
)
from gpustack.utils.envs import sanitize_env
from gpustack.worker.backends.base import InferenceServer
from gpustack.worker.backends.vllm_omni_args import (
    build_vllm_omni_arguments,
    detect_model_type,
    get_audio_arguments,
    get_diffusion_arguments,
)

logger = logging.getLogger(__name__)


class VLLMOmniServer(InferenceServer):
    """
    vLLM-Omni inference server for omni-modality models.
    
    Supports:
    - Diffusion models (Z-Image, Flux, SD3, etc.)
    - Audio models
    - Video models
    - Multi-modal generation
    """

    def start(self):
        try:
            # Get vllm-omni command path
            command_path = get_command_path("vllm-omni")
            if self._model.backend_version:
                command_path = os.path.join(
                    self._config.bin_dir,
                    get_versioned_command("vllm-omni", self._model.backend_version),
                )

            arguments = self._build_arguments()

            logger.info(f"Starting vLLM-Omni server: {command_path}")
            logger.debug(f"Run vLLM-Omni with arguments: {' '.join(arguments)}")

            env = os.environ.copy()
            env = self.get_inference_running_env(env)

            # Log environment variables for debugging
            if logger.isEnabledFor(logging.DEBUG):
                env_view = sanitize_env(env)
                logger.info(
                    f"With environment variables:{os.linesep}"
                    f"{os.linesep.join(f'{k}={v}' for k, v in sorted(env_view.items()))}"
                )

            result = subprocess.run(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
                cwd=self._model_path,
            )
            self.exit_with_code(result.returncode)

        except Exception as e:
            error_message = f"Failed to run the vLLM-Omni server: {e}"
            logger.error(error_message)
            try:
                patch_dict = {
                    "state_message": error_message,
                    "state": ModelInstanceStateEnum.ERROR,
                }
                self._update_model_instance(self._model_instance.id, **patch_dict)
            except Exception as ue:
                logger.error(f"Failed to update model instance: {ue}")
            sys.exit(1)

    def _build_arguments(self) -> list:
        """Build command line arguments for vLLM-Omni server."""
        return build_vllm_omni_arguments(
            self._model_path,
            self._model.name,
            self._model_instance.model_name,
            self._model.categories,
            self._model.backend_parameters,
            self._model_instance.port,
            len(self._model_instance.gpu_indexes or []),
        )

    def _detect_model_type(self) -> str:
        """
        Detect the model type based on model name or categories.
        
        Returns:
            str: Model type ('diffusion', 'audio', 'llm', etc.)
        """
        return detect_model_type(self._model.name, self._model.categories)

    def _get_diffusion_arguments(self) -> list:
        """Get arguments specific to diffusion models."""
        return get_diffusion_arguments()

    def _get_audio_arguments(self) -> list:
        """Get arguments specific to audio models."""
        return get_audio_arguments()
