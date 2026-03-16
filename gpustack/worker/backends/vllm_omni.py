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
from typing import Optional

from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.utils.command import (
    find_parameter,
    get_versioned_command,
    get_command_path,
)
from gpustack.utils.envs import sanitize_env
from gpustack.worker.backends.base import InferenceServer

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
        arguments = [
            "serve",
            self._model_path,
        ]

        # Determine model type and add appropriate arguments
        model_type = self._detect_model_type()
        if model_type == "diffusion":
            arguments.extend(self._get_diffusion_arguments())
        elif model_type == "audio":
            arguments.extend(self._get_audio_arguments())

        # Add user-defined backend parameters
        if self._model.backend_parameters:
            arguments.extend(self._model.backend_parameters)

        # Built-in arguments (cannot be overridden)
        built_in_arguments = [
            "--host",
            "0.0.0.0",
            "--port",
            str(self._model_instance.port),
            "--served-model-name",
            self._model_instance.model_name,
        ]
        arguments.extend(built_in_arguments)

        return arguments

    def _detect_model_type(self) -> str:
        """
        Detect the model type based on model name or categories.
        
        Returns:
            str: Model type ('diffusion', 'audio', 'llm', etc.)
        """
        model_name = self._model.name.lower()
        categories = self._model.categories or []

        # Check for diffusion/image models
        diffusion_keywords = [
            "flux", "z-image", "stable-diffusion", "sd3", "sdxl",
            "dit", "lumina", "hunyuan-dit", "pixart"
        ]
        if "image" in categories or any(kw in model_name for kw in diffusion_keywords):
            return "diffusion"

        # Check for audio models
        audio_keywords = ["whisper", "tts", "stt", "audio", "speech"]
        if any(cat in categories for cat in ["speech_to_text", "text_to_speech"]):
            return "audio"
        if any(kw in model_name for kw in audio_keywords):
            return "audio"

        return "llm"

    def _get_diffusion_arguments(self) -> list:
        """Get arguments specific to diffusion models."""
        return []

    def _get_audio_arguments(self) -> list:
        """Get arguments specific to audio models."""
        args = []
        # Add audio-specific configurations here
        return args
