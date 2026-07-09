import glob
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple
import psutil

from gpustack.schemas.workers import Worker
from gpustack.utils import platform
from gpustack.schemas.models import (
    BackendEnum,
    ModelInstance,
    ModelInstanceStateEnum,
    get_gguf_runtime,
    is_embedding_model,
    is_image_model,
    is_renaker_model,
)
from gpustack.utils.command import (
    ensure_bool_parameter,
    find_parameter,
    normalize_parameters,
)
from gpustack.utils.third_party import third_party_bin_path
from gpustack.worker.backends.base import InferenceServer
from gpustack.worker.tools_manager import (
    BUILTIN_LLAMA_CPP_VERSION,
    get_llama_box_command,
    get_llama_cpp_command,
    get_llama_cpp_version_dir_name,
    is_disabled_dynamic_link,
    BUILTIN_LLAMA_BOX_VERSION,
    ToolsManager,
)

logger = logging.getLogger(__name__)


class LlamaBoxServer(InferenceServer):
    def start(self):  # noqa: C901
        if get_gguf_runtime(self._model) == BackendEnum.LLAMA_CPP:
            return self._start_llama_cpp()

        return self._start_llama_box()

    def _start_llama_box(self):  # noqa: C901
        # Launch llama-box from <third_party>/bin/llama-box/llama-box-default,
        # if allowing dynamic linking binary and builtin version is used,
        # otherwise use the user-provided binary path in the config,
        # i.e. <bin_dir>/llama-box/llama-box-<version> or <bin_dir>/llama-box/static/llama-box-<version>.
        version = (
            self._model.backend_version
            if self._model.backend_version
            else BUILTIN_LLAMA_BOX_VERSION
        )
        disabled_dynamic_link = (
            is_disabled_dynamic_link(version) and self._config.bin_dir is not None
        )
        if not disabled_dynamic_link and version == BUILTIN_LLAMA_BOX_VERSION:
            base_path = str(third_party_bin_path("llama-box/llama-box-default"))
        else:
            base_path = os.path.join(
                self._config.bin_dir,
                "llama-box",
                f'{"static" if disabled_dynamic_link else ""}',
                f"llama-box-{version}",
            )
        command_path = get_llama_box_command(base_path)

        layers = -1
        claim = self._model_instance.computed_resource_claim
        if claim is not None and claim.offload_layers is not None:
            layers = claim.offload_layers

        workers = self._clientset.workers.list()
        worker_map = {worker.id: worker for worker in workers.items}
        rpc_servers, rpc_server_tensor_split = get_rpc_servers(
            self._model_instance, worker_map
        )

        default_parallel = "4"
        if is_renaker_model(self._model) or is_embedding_model(self._model):
            default_parallel = "1"

        arguments = [
            "--host",
            "0.0.0.0",
            "--embeddings",
            "--gpu-layers",
            str(layers),
            "--parallel",
            default_parallel,
            "--ctx-size",
            "8192",
            "--port",
            str(self._model_instance.port),
            "--model",
            self._model_path,
            "--alias",
            self._model.name,
            "--no-mmap",
            "--no-warmup",
        ]
        arguments = ensure_metrics_enabled(arguments, self._model.backend_parameters)

        if is_renaker_model(self._model):
            arguments.append("--rerank")

        if is_image_model(self._model):
            # TODO support multi-GPU for image models
            arguments.extend(["--images", "--image-vae-tiling", "--tensor-split", "1"])

        mmproj = find_parameter(self._model.backend_parameters, ["mmproj"])
        default_mmproj = get_mmproj_file(self._model_path)
        if mmproj is None and default_mmproj:
            arguments.extend(["--mmproj", default_mmproj])
            # Enable `--max-projected-cache` to optimize chatting experience,
            # cause llama-box will ignore unknown parameters,
            # we can safely add this parameter without breaking previous version.
            arguments.extend(["--max-projected-cache", "10"])

        if rpc_servers:
            rpc_servers_argument = ",".join(rpc_servers)
            arguments.extend(["--rpc", rpc_servers_argument])

        # legacy support for tensor split field is empty
        main_worker_tensor_split = []
        if (
            self._model_instance.gpu_indexes
            and len(self._model_instance.gpu_indexes) > 0
        ):
            vram_claims = claim.vram.values()
            main_worker_tensor_split = vram_claims

        legacy_tensor_split = []
        if rpc_server_tensor_split:
            legacy_tensor_split.extend(rpc_server_tensor_split)
            legacy_tensor_split.extend(main_worker_tensor_split)
        elif len(main_worker_tensor_split) > 1:
            legacy_tensor_split.extend(main_worker_tensor_split)

        tensor_split = legacy_tensor_split
        if self._model_instance.computed_resource_claim.tensor_split:
            tensor_split = self._model_instance.computed_resource_claim.tensor_split

        user_tensor_split = find_parameter(
            self._model.backend_parameters, ["ts", "tensor-split"]
        )
        if user_tensor_split is None and tensor_split:
            tensor_split_argument = ",".join(
                [str(int(tensor / (1024 * 1024))) for tensor in tensor_split]
            )  # convert to MiB to prevent overflow

            arguments.extend(["--tensor-split", tensor_split_argument])

        if self._model.backend_parameters:
            self.normalize_mmproj_path()
            # append user-provided parameters
            for param in self._model.backend_parameters:
                if "=" not in param:
                    arguments.append(param)
                else:
                    key, value = param.split('=', 1)
                    arguments.extend([key, value])

        try:
            logger.info("Starting llama-box server")
            logger.debug(
                f"Run llama-box: {command_path} with arguments: {' '.join(arguments)}"
            )
            if self._model.env:
                logger.debug(
                    f"Model environment variables: {', '.join(f'{key}={value}' for key, value in self._model.env.items())}"
                )

            env = self.get_inference_running_env()
            cwd = str(command_path.parent)
            if platform.system() == "linux":
                ld_library_path = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = (
                    ":".join([cwd, ld_library_path]) if ld_library_path else cwd
                )
            proc = subprocess.Popen(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
                cwd=cwd,
            )

            set_priority(proc.pid)
            exit_code = proc.wait()
            self.exit_with_code(exit_code)

        except Exception as e:
            error_message = f"Failed to run the llama-box server: {e}"
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

    def _start_llama_cpp(self):
        try:
            self._ensure_llama_cpp_supported()
            command_path = self._get_llama_cpp_command_path()
            arguments = self._build_llama_cpp_arguments()

            logger.info("Starting llama.cpp server")
            logger.debug(
                f"Run llama.cpp: {command_path} with arguments: {' '.join(arguments)}"
            )
            if self._model.env:
                logger.debug(
                    f"Model environment variables: {', '.join(f'{key}={value}' for key, value in self._model.env.items())}"
                )

            env = self.get_inference_running_env()
            cwd = str(command_path.parent)
            if platform.system() == "linux":
                ld_library_path = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = (
                    ":".join([cwd, ld_library_path]) if ld_library_path else cwd
                )

            proc = subprocess.Popen(
                [command_path] + arguments,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=env,
                cwd=cwd,
            )

            set_priority(proc.pid)
            exit_code = proc.wait()
            self.exit_with_code(exit_code)

        except Exception as e:
            error_message = f"Failed to run the llama.cpp server: {e}"
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

    def _ensure_llama_cpp_supported(self):
        if self._model.distributed_inference_across_workers:
            raise RuntimeError(
                "llama.cpp runtime does not support distributed inference across "
                "workers. Please disable distributed_inference_across_workers or "
                "use llama-box runtime."
            )

        subordinate_workers = (
            self._model_instance.distributed_servers.subordinate_workers
            if self._model_instance.distributed_servers
            and self._model_instance.distributed_servers.subordinate_workers
            else []
        )
        if subordinate_workers:
            raise RuntimeError(
                "llama.cpp runtime does not support llama-box RPC distributed "
                "serving. Please disable distributed_inference_across_workers or "
                "use llama-box runtime."
            )

    def _get_llama_cpp_command_path(self) -> Path:
        configured_path = (self._model.env or {}).get("GPUSTACK_LLAMA_CPP_SERVER_PATH")
        if configured_path:
            command_path = Path(configured_path)
            if not command_path.is_file():
                raise FileNotFoundError(f"llama-server not found: {command_path}")
            return command_path

        version = (
            self._model.backend_version
            if self._model.backend == BackendEnum.LLAMA_CPP
            and self._model.backend_version
            else (self._model.env or {}).get(
                "GPUSTACK_LLAMA_CPP_VERSION", BUILTIN_LLAMA_CPP_VERSION
            )
        )
        base_path = Path(
            str(
                third_party_bin_path(
                    "llama.cpp",
                    get_llama_cpp_version_dir_name(
                        version,
                        platform.system(),
                        platform.arch(),
                        platform.device(),
                    ),
                )
            )
        )
        command_path = get_llama_cpp_command(base_path)
        if command_path.is_file():
            return command_path

        path_command = shutil.which("llama-server")
        if path_command:
            return Path(path_command)

        tools_manager = ToolsManager(
            tools_download_base_url=self._config.tools_download_base_url,
            data_dir=self._config.data_dir,
            bin_dir=self._config.bin_dir,
            pipx_path=self._config.pipx_path,
            device=platform.device(),
        )
        return tools_manager.install_llama_cpp(version)

    def _build_llama_cpp_arguments(self) -> List[str]:
        layers = -1
        claim = self._model_instance.computed_resource_claim
        if claim is not None and claim.offload_layers is not None:
            layers = claim.offload_layers

        default_parallel = "4"
        if is_renaker_model(self._model) or is_embedding_model(self._model):
            default_parallel = "1"

        arguments = [
            "--host",
            "0.0.0.0",
            "--gpu-layers",
            str(layers),
            "--parallel",
            default_parallel,
            "--ctx-size",
            "8192",
            "--port",
            str(self._model_instance.port),
            "--model",
            self._model_path,
            "--alias",
            self._model.name,
            "--no-mmap",
        ]

        mmproj = find_parameter(self._model.backend_parameters, ["mmproj"])
        default_mmproj = get_mmproj_file(self._model_path)
        if mmproj is None and default_mmproj:
            arguments.extend(["--mmproj", default_mmproj])

        tensor_split = []
        if claim and claim.tensor_split:
            tensor_split = claim.tensor_split
        elif (
            self._model_instance.gpu_indexes
            and len(self._model_instance.gpu_indexes) > 1
            and claim
            and claim.vram
        ):
            tensor_split = list(claim.vram.values())

        user_tensor_split = find_parameter(
            self._model.backend_parameters, ["ts", "tensor-split"]
        )
        if user_tensor_split is None and tensor_split:
            tensor_split_argument = ",".join(
                [str(int(tensor / (1024 * 1024))) for tensor in tensor_split]
            )
            arguments.extend(["--tensor-split", tensor_split_argument])

        self.normalize_mmproj_path()
        user_parameters = normalize_llama_cpp_parameters(
            self._model.backend_parameters or []
        )
        if user_parameters:
            arguments.extend(user_parameters)

        return arguments

    def normalize_mmproj_path(self):
        """
        We provide a syntax sugar for the user to specify the mmproj file relative to the model path.
        So, users can specify --mmproj=mmproj.gguf instead of --mmproj=/path/to/mmproj.gguf.
        This function normalizes the file path to the same directory of the model path and set it back to the backend parameters.
        """

        model_dir = Path(self._model_path).parent
        mmproj_param = "--mmproj"
        for i, param in enumerate(self._model.backend_parameters or []):
            if '=' in param:
                key, value = param.split('=', 1)
                if key == mmproj_param and value and not Path(value).is_absolute():
                    self._model.backend_parameters[i] = (
                        f"{mmproj_param}={model_dir / value}"
                    )
            else:
                if param == mmproj_param and i + 1 < len(
                    self._model.backend_parameters
                ):
                    value = self._model.backend_parameters[i + 1]
                    if value and not Path(value).is_absolute():
                        self._model.backend_parameters[i + 1] = str(model_dir / value)


class LlamaCppServer(LlamaBoxServer):
    def start(self):
        return self._start_llama_cpp()


def get_mmproj_file(model_path: str) -> str:
    directory = os.path.dirname(model_path)
    pattern = os.path.join(directory, '*mmproj*.gguf')
    files = glob.glob(pattern)

    if files:
        return files[0]


def set_priority(pid: int):
    if platform.system() != "windows":
        return

    try:
        priority_class = psutil.ABOVE_NORMAL_PRIORITY_CLASS
        proc = psutil.Process(pid)
        proc.nice(priority_class)
        logger.debug(f"Set process {proc.pid} priority to {priority_class}")
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        logger.error(f"Failed to set priority for process {pid}: {e}")


def ensure_metrics_enabled(arguments: List[str], backend_parameters: List[str]) -> List[str]:
    return ensure_bool_parameter(
        arguments,
        "metrics",
        existing_parameters=backend_parameters,
    )


def normalize_llama_cpp_parameters(parameters: List[str]) -> List[str]:
    removes = [
        "gpustack-runtime",
        "rpc",
        "embeddings",
        "images",
        "image-vae-tiling",
        "max-projected-cache",
        "no-warmup",
        "metrics",
    ]
    return normalize_parameters(parameters, removes=removes)


def get_rpc_servers(
    model_instance: ModelInstance, worker_map: Dict[int, Worker]
) -> Tuple[List[str], List[int]]:
    rpc_servers = []
    rpc_tensor_split = []
    if (
        model_instance.distributed_servers
        and model_instance.distributed_servers.subordinate_workers
    ):
        for rpc_server in model_instance.distributed_servers.subordinate_workers:
            r_worker = worker_map.get(rpc_server.worker_id)
            r_ip = r_worker.ip
            r_port = r_worker.status.rpc_servers.get(rpc_server.gpu_indexes[0]).port
            r_ts = list((rpc_server.computed_resource_claim or {}).vram.values())

            rpc_tensor_split.extend(r_ts)
            rpc_servers.append(f"{r_ip}:{r_port}")
    return rpc_servers, rpc_tensor_split
