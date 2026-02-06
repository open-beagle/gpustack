#!/bin/bash

set -ex

# 配置
TOOLS_DIR="${TOOLS_DIR:-./.downloads/gpustack}"
S3_BUCKET="aliyun/vscode/gpustack"
ARCH="${ARCH:-amd64}"
DEVICE="${DEVICE:-cuda}"

# 版本定义
LLAMA_BOX_VERSION="v0.0.140"
GGUF_PARSER_GO_VERSION="v0.13.8"
FASTFETCH_VERSION="2.25.0.1"

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
    curl -fL -o "$TOOLS_DIR/llama-box/releases/download/$LLAMA_BOX_VERSION/llama-box-linux-amd64-cuda-12.4.zip" \
      "https://github.com/gpustack/llama-box/releases/download/$LLAMA_BOX_VERSION/llama-box-linux-amd64-cuda-12.4.zip"
    ;;
  arm64-cann)
    curl -fL -o "$TOOLS_DIR/llama-box/releases/download/$LLAMA_BOX_VERSION/llama-box-linux-arm64-cann-8.0.zip" \
      "https://github.com/gpustack/llama-box/releases/download/$LLAMA_BOX_VERSION/llama-box-linux-arm64-cann-8.0.zip"
    ;;
  *)
    echo "不支持的架构-设备组合: $ARCH-$DEVICE"
    exit 1
    ;;
esac

# 下载 gguf-parser-go
echo "下载 gguf-parser-go $GGUF_PARSER_GO_VERSION..."
mkdir -p "$TOOLS_DIR/gguf-parser-go/releases/download/$GGUF_PARSER_GO_VERSION"

case "$ARCH" in
  amd64)
    curl -fL -o "$TOOLS_DIR/gguf-parser-go/releases/download/$GGUF_PARSER_GO_VERSION/gguf-parser-linux-amd64" \
      "https://github.com/gpustack/gguf-parser-go/releases/download/$GGUF_PARSER_GO_VERSION/gguf-parser-linux-amd64"
    ;;
  arm64)
    curl -fL -o "$TOOLS_DIR/gguf-parser-go/releases/download/$GGUF_PARSER_GO_VERSION/gguf-parser-linux-arm64" \
      "https://github.com/gpustack/gguf-parser-go/releases/download/$GGUF_PARSER_GO_VERSION/gguf-parser-linux-arm64"
    ;;
esac

# 下载 fastfetch
echo "下载 fastfetch $FASTFETCH_VERSION..."
mkdir -p "$TOOLS_DIR/fastfetch/releases/download/$FASTFETCH_VERSION"

case "$ARCH" in
  amd64)
    curl -fL -o "$TOOLS_DIR/fastfetch/releases/download/$FASTFETCH_VERSION/fastfetch-linux-amd64.zip" \
      "https://github.com/gpustack/fastfetch/releases/download/$FASTFETCH_VERSION/fastfetch-linux-amd64.zip"
    ;;
  arm64)
    curl -fL -o "$TOOLS_DIR/fastfetch/releases/download/$FASTFETCH_VERSION/fastfetch-linux-aarch64.zip" \
      "https://github.com/gpustack/fastfetch/releases/download/$FASTFETCH_VERSION/fastfetch-linux-aarch64.zip"
    ;;
esac

echo "=========================================="
echo "上传 Tools 到 S3"
echo "=========================================="

# 上传到 S3
mc cp -r "$TOOLS_DIR" "$S3_BUCKET/"

echo "=========================================="
echo "完成！"
echo "Tools 已上传到: $S3_BUCKET/gpustack"
echo "=========================================="
