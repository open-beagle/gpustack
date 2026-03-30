ARG BASE=registry.cn-qingdao.aliyuncs.com/wod/cuda:12.8.1-runtime-ubuntu22.04

FROM $BASE

ARG AUTHOR=mengkzhaoyun@gmail.com
ARG VERSION=dev

LABEL maintainer=$AUTHOR version=$VERSION

ENV DEBIAN_FRONTEND=noninteractive

# 配置 Ubuntu 镜像源（使用阿里云镜像）
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list

# 安装系统依赖 + 设置时区
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    wget \
    tzdata \
    python3 \
    python3-pip \
    pipx \
    python3-dev \
    tini \
    gcc \
    g++ \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb1 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    cuda-nvcc-12-8 \
    libcublas-dev-12-8 \
    libcurand-dev-12-8 \
    cuda-nvrtc-dev-12-8 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

ARG PYPI_MIRROR=http://mirrors.cloud.aliyuncs.com/pypi/simple/
ARG PYPI_HOST=mirrors.cloud.aliyuncs.com

# 安装 GPUStack、vLLM 和 vLLM-Omni
COPY ./dist/*.whl /tmp/
RUN WHEEL_PACKAGE_STACK="$(ls /tmp/*.whl)[vllm]" && \
    python3 -m pip install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} --upgrade pip && \
    pip3 install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} --no-cache-dir --default-timeout=12000 $WHEEL_PACKAGE_STACK && \
    # 安装 vLLM-Omni 用于支持 Diffusion 模型（Z-Image、Flux 等）
    pip3 install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple/ --no-cache-dir --default-timeout=12000 vllm-omni==0.18.0 && \
    # 强制升级 transformers 以支持最新模型架构
    pip3 install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} --no-cache-dir "transformers>=5.4.0" && \
    # 修复 transformers RoPE 验证类型错误 (set -= list)
    python3 -c "\
import transformers.modeling_rope_utils as m; \
p = m.__file__; \
c = open(p, 'r').read(); \
c = c.replace('received_keys -= ignore_keys', 'received_keys -= set(ignore_keys)'); \
open(p, 'w').write(c)" && \
    rm -rf /tmp/*.whl

# 下载工具
RUN gpustack download-tools --device cuda --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'

# 设置目录
RUN mkdir -p /var/lib/gpustack && \
    chmod -R 0755 /var/lib/gpustack

# 设置环境变量
ENV PIPX_HOME=/var/lib/gpustack/pipx \
    PIPX_LOCAL_VENVS=/var/lib/gpustack/pipx/venvs \
    PIPX_BIN_DIR=/var/lib/gpustack/bin \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
