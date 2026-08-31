import argparse
import logging
import sys

from gpustack.utils.envs import get_gpustack_env
from gpustack.worker.preheat_credential import (
    WorkerCredentialBootstrapError,
    bootstrap_remote_worker_credential,
)


logger = logging.getLogger(__name__)


def setup_bootstrap_worker_credential_cmd(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        "bootstrap-worker-credential",
        help="为已升级的远程 Worker 写入模型预热专用凭据。",
        description="管理员 API key 仅从标准输入读取，不会写入命令行或输出。",
    )
    parser.add_argument(
        "-s",
        "--server-url",
        required=not bool(get_gpustack_env("SERVER_URL")),
        default=get_gpustack_env("SERVER_URL"),
        help="GPUStack Server 地址。",
    )
    parser.add_argument(
        "--data-dir",
        required=not bool(get_gpustack_env("DATA_DIR")),
        default=get_gpustack_env("DATA_DIR"),
        help="远程 Worker 的数据目录。",
    )
    parser.add_argument(
        "--api-key-stdin",
        action="store_true",
        required=True,
        help="从标准输入读取管理员 API key。",
    )
    parser.set_defaults(func=run)


def run(args):
    try:
        admin_api_key = sys.stdin.read().strip()
        bootstrap_remote_worker_credential(
            args.server_url,
            args.data_dir,
            admin_api_key,
        )
        print("Worker 专用凭据引导完成。")
    except WorkerCredentialBootstrapError as error:
        logger.fatal("Worker 专用凭据引导失败：%s", error)
        raise SystemExit(1)
