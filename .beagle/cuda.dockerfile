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
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

# 安装 GPUStack、vLLM 和 vLLM-Omni
COPY ./dist/*.whl /tmp/
RUN WHEEL_PACKAGE="$(ls /tmp/*.whl)[vllm]" && \
    python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip && \
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip3 install --no-cache-dir --default-timeout=12000 $WHEEL_PACKAGE && \
    # 安装 vLLM-Omni 用于支持 Diffusion 模型（Z-Image、Flux 等）
    pip3 install --no-cache-dir --default-timeout=12000 vllm-omni && \
    # 强制升级 transformers 以支持最新模型架构
    pip3 install --no-cache-dir "transformers>=5.3.0" && \
    rm -rf /tmp/*.whl

# 下载工具
RUN gpustack download-tools --device cuda

# 设置目录
RUN mkdir -p /var/lib/gpustack && \
    chmod -R 0755 /var/lib/gpustack

# 设置环境变量
ENV PIPX_HOME=/var/lib/gpustack/pipx \
    PIPX_LOCAL_VENVS=/var/lib/gpustack/pipx/venvs \
    PIPX_BIN_DIR=/var/lib/gpustack/bin \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
