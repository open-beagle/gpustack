ARG BASE=nvidia/cuda:12.8.2-runtime-ubuntu22.04

FROM $BASE

ARG AUTHOR=mengkzhaoyun@gmail.com

LABEL maintainer=$AUTHOR

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
    libcurand-dev-12-8 \
    cuda-nvrtc-dev-12-8 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

ARG PYPI_MIRROR=https://mirrors.aliyun.com/pypi/simple/
ARG PYPI_HOST=mirrors.aliyun.com
ARG PYPI_EXTRA_INDEX_URL=https://pypi.org/simple/

# 安装稳定运行时依赖层。只有 CUDA/Python/vLLM/vLLM-Omni/transformers 等底层版本变化时才需要重打该基础镜像。
COPY ./dist/requirements-vllm.txt /tmp/requirements-vllm.txt
COPY ./.beagle/prepare_cuda_base.py /tmp/prepare_cuda_base.py
RUN python3 -m pip install -i ${PYPI_MIRROR} --extra-index-url ${PYPI_EXTRA_INDEX_URL} --trusted-host ${PYPI_HOST} --upgrade pip && \
    python3 -m pip install -i ${PYPI_MIRROR} --extra-index-url ${PYPI_EXTRA_INDEX_URL} --trusted-host ${PYPI_HOST} --no-cache-dir --upgrade 'pipx==1.7.1' 'argcomplete>=1.9.4' && \
    python3 -m pip install -i ${PYPI_MIRROR} --extra-index-url ${PYPI_EXTRA_INDEX_URL} --trusted-host ${PYPI_HOST} --no-cache-dir --default-timeout=12000 -r /tmp/requirements-vllm.txt && \
    command -v vllm && \
    command -v vllm-omni && \
    pipx --version && \
    python3 -m pip show vllm vllm-omni && \
    python3 /tmp/prepare_cuda_base.py && \
    rm -f /tmp/requirements-vllm.txt /tmp/prepare_cuda_base.py

# 设置目录
RUN mkdir -p /var/lib/gpustack && \
    chmod -R 0755 /var/lib/gpustack

# 设置环境变量
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/compat \
    PIPX_HOME=/var/lib/gpustack/pipx \
    PIPX_LOCAL_VENVS=/var/lib/gpustack/pipx/venvs \
    PIPX_BIN_DIR=/var/lib/gpustack/bin \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ARG VERSION=dev
LABEL version=$VERSION
