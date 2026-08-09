#!/bin/bash

set -euo pipefail

TOOLS_DOWNLOAD_BASE_URL="${TOOLS_DOWNLOAD_BASE_URL:-https://cache.ali.wodcloud.com/vscode}"
TOOLS_SYSTEM="${SYSTEM:-linux}"
TOOLS_ARCH="${ARCH:-amd64}"
TOOLS_DEVICE="${DEVICE:-cuda}"
export TOOLS_DOWNLOAD_BASE_URL TOOLS_SYSTEM TOOLS_ARCH TOOLS_DEVICE

echo "=========================================="
echo "安装 GPUStack Runtime Tools"
echo "下载地址: ${TOOLS_DOWNLOAD_BASE_URL}"
echo "系统: ${TOOLS_SYSTEM}"
echo "架构: ${TOOLS_ARCH}"
echo "设备: ${TOOLS_DEVICE}"
echo "=========================================="

python3 <<'PY'
import os

from gpustack.worker.tools_manager import ToolsManager

tools_manager = ToolsManager(
    tools_download_base_url=os.environ["TOOLS_DOWNLOAD_BASE_URL"],
    system=os.environ["TOOLS_SYSTEM"],
    arch=os.environ["TOOLS_ARCH"],
    device=os.environ["TOOLS_DEVICE"],
)
tools_manager.download_gguf_parser()
tools_manager.download_fastfetch()
PY

echo "=========================================="
echo "Runtime tools 安装完成"
echo "=========================================="
