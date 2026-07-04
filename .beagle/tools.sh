#!/bin/bash

set -e

# 配置
TOOLS_DIR="${TOOLS_DIR:-./.downloads/gpustack}"
S3_BUCKET="aliyun/vscode/gpustack"
ARCH="${ARCH:-amd64}"
DEVICE="${DEVICE:-cuda}"

# 版本定义
LLAMA_BOX_VERSION="v0.0.171"
GGUF_PARSER_GO_VERSION="v0.22.1"
FASTFETCH_VERSION="2.25.0.1"

# 标记是否有下载
NEED_UPLOAD=0

# 清理旧的下载
rm -rf "$TOOLS_DIR"
mkdir -p "$TOOLS_DIR"

echo "=========================================="
echo "下载 GPUStack Tools"
echo "架构: $ARCH"
echo "设备: $DEVICE"
echo "=========================================="

# 下载 llama-box
echo "下载 llama-box $LLAMA_BOX_VERSION..."
mkdir -p "$TOOLS_DIR/llama-box/releases/download/$LLAMA_BOX_VERSION"

case "$ARCH-$DEVICE" in
  amd64-cuda)
    # 动态链接版本
    LLAMA_BOX_FILE="dl-llama-box-linux-amd64-cuda-12.4.zip"
    ;;
  amd64-corex|amd64-cpu)
    # CoreX 目前没有 llama-box 专用构建，使用通用 CPU 动态链接版本
    LLAMA_BOX_FILE="dl-llama-box-linux-amd64-cpu.zip"
    ;;
  arm64-cann)
    # 静态链接版本（CANN 需要静态链接）
    LLAMA_BOX_FILE="llama-box-linux-arm64-cann-8.0.zip"
    ;;
  *)
    echo "不支持的架构-设备组合: $ARCH-$DEVICE"
    exit 1
    ;;
esac

echo "检查 S3 中是否已存在 $LLAMA_BOX_FILE..."
if ! mc stat "$S3_BUCKET/llama-box/releases/download/$LLAMA_BOX_VERSION/$LLAMA_BOX_FILE" > /dev/null 2>&1; then
  echo "文件不存在，开始下载..."
  curl -fL -x $SOCKS5_PROXY_LOCAL -o "$TOOLS_DIR/llama-box/releases/download/$LLAMA_BOX_VERSION/$LLAMA_BOX_FILE" \
    "https://github.com/gpustack/llama-box/releases/download/$LLAMA_BOX_VERSION/$LLAMA_BOX_FILE"
  NEED_UPLOAD=1
else
  echo "文件已存在于 S3，跳过下载"
fi

# 下载 gguf-parser-go
echo "下载 gguf-parser-go $GGUF_PARSER_GO_VERSION..."
mkdir -p "$TOOLS_DIR/gguf-parser-go/releases/download/$GGUF_PARSER_GO_VERSION"

case "$ARCH" in
  amd64)
    GGUF_FILE="gguf-parser-linux-amd64"
    echo "检查 S3 中是否已存在 $GGUF_FILE..."
    if ! mc stat "$S3_BUCKET/gguf-parser-go/releases/download/$GGUF_PARSER_GO_VERSION/$GGUF_FILE" > /dev/null 2>&1; then
      echo "文件不存在，开始下载..."
      curl -fL -x $SOCKS5_PROXY_LOCAL -o "$TOOLS_DIR/gguf-parser-go/releases/download/$GGUF_PARSER_GO_VERSION/$GGUF_FILE" \
        "https://github.com/gpustack/gguf-parser-go/releases/download/$GGUF_PARSER_GO_VERSION/$GGUF_FILE"
      NEED_UPLOAD=1
    else
      echo "文件已存在于 S3，跳过下载"
    fi
    ;;
  arm64)
    GGUF_FILE="gguf-parser-linux-arm64"
    echo "检查 S3 中是否已存在 $GGUF_FILE..."
    if ! mc stat "$S3_BUCKET/gguf-parser-go/releases/download/$GGUF_PARSER_GO_VERSION/$GGUF_FILE" > /dev/null 2>&1; then
      echo "文件不存在，开始下载..."
      curl -fL -x $SOCKS5_PROXY_LOCAL -o "$TOOLS_DIR/gguf-parser-go/releases/download/$GGUF_PARSER_GO_VERSION/$GGUF_FILE" \
        "https://github.com/gpustack/gguf-parser-go/releases/download/$GGUF_PARSER_GO_VERSION/$GGUF_FILE"
      NEED_UPLOAD=1
    else
      echo "文件已存在于 S3，跳过下载"
    fi
    ;;
esac

# 下载 fastfetch
echo "下载 fastfetch $FASTFETCH_VERSION..."
mkdir -p "$TOOLS_DIR/fastfetch/releases/download/$FASTFETCH_VERSION"

case "$ARCH" in
  amd64)
    FASTFETCH_FILE="fastfetch-linux-amd64.zip"
    echo "检查 S3 中是否已存在 $FASTFETCH_FILE..."
    if ! mc stat "$S3_BUCKET/fastfetch/releases/download/$FASTFETCH_VERSION/$FASTFETCH_FILE" > /dev/null 2>&1; then
      echo "文件不存在，开始下载..."
      curl -fL -x $SOCKS5_PROXY_LOCAL -o "$TOOLS_DIR/fastfetch/releases/download/$FASTFETCH_VERSION/$FASTFETCH_FILE" \
        "https://github.com/gpustack/fastfetch/releases/download/$FASTFETCH_VERSION/$FASTFETCH_FILE"
      NEED_UPLOAD=1
    else
      echo "文件已存在于 S3，跳过下载"
    fi
    ;;
  arm64)
    FASTFETCH_FILE="fastfetch-linux-aarch64.zip"
    echo "检查 S3 中是否已存在 $FASTFETCH_FILE..."
    if ! mc stat "$S3_BUCKET/fastfetch/releases/download/$FASTFETCH_VERSION/$FASTFETCH_FILE" > /dev/null 2>&1; then
      echo "文件不存在，开始下载..."
      curl -fL -x $SOCKS5_PROXY_LOCAL -o "$TOOLS_DIR/fastfetch/releases/download/$FASTFETCH_VERSION/$FASTFETCH_FILE" \
        "https://github.com/gpustack/fastfetch/releases/download/$FASTFETCH_VERSION/$FASTFETCH_FILE"
      NEED_UPLOAD=1
    else
      echo "文件已存在于 S3，跳过下载"
    fi
    ;;
esac

echo "=========================================="
echo "上传 Tools 到 S3"
echo "=========================================="

if [ $NEED_UPLOAD -eq 1 ]; then
  echo "有新文件下载，开始上传..."
  mc mirror --overwrite "$TOOLS_DIR/llama-box" "$S3_BUCKET/llama-box"
  mc mirror --overwrite "$TOOLS_DIR/gguf-parser-go" "$S3_BUCKET/gguf-parser-go"
  mc mirror --overwrite "$TOOLS_DIR/fastfetch" "$S3_BUCKET/fastfetch"
  echo "上传完成！"
else
  echo "没有新文件，跳过上传"
fi

echo "=========================================="
echo "完成！"
echo "=========================================="
