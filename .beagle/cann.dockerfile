ARG BASE=registry.cn-qingdao.aliyuncs.com/wod/cann:8.0.RC3-py310-kernel910b

FROM $BASE

ARG AUTHOR=mengkzhaoyun@gmail.com
ARG VERSION=v0.3.2

LABEL maintainer=$AUTHOR version=$VERSION

COPY ./dist/*.whl /tmp/

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|http://mirrors.cloud.aliyuncs.com/ubuntu-ports/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    wget \
    tzdata \
    gcc \
    g++ \
    tini && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG PYPI_MIRROR=http://mirrors.cloud.aliyuncs.com/pypi/simple/
ARG PYPI_HOST=mirrors.cloud.aliyuncs.com

RUN python3 -m pip install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} pipx

RUN WHEEL_PACKAGE="$(ls /tmp/*-any.whl)" && \
  # 安装 GPUStack、vLLM
  VLLM_TARGET_DEVICE=empty pip3 install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} --use-pep517 "${WHEEL_PACKAGE}[vllm]" &&\
  # 安装 vLLM-Omni 用于支持 Diffusion 模型（Z-Image、Flux 等）
  VLLM_TARGET_DEVICE=empty pip3 install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} vllm-omni==0.18.0 &&\
  # 强制升级 transformers 以支持最新模型架构
  pip3 install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} "transformers>=5.4.0" &&\
  # 继续补装对应的华为 NPU 架构后端实现插件
  pip3 install vllm-ascend --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/simple -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} &&\
  rm /tmp/*.whl && \
  pip3 cache purge

ENV CANN_VERSION=8.2.rc1
RUN gpustack download-tools \
  --arch arm64 --device npu \
  --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'

# 设置目录
RUN mkdir -p /var/lib/gpustack && \
    chmod -R 0755 /var/lib/gpustack

# 设置环境变量
ENV PIPX_HOME=/var/lib/gpustack/pipx \
    PIPX_LOCAL_VENVS=/var/lib/gpustack/pipx/venvs \
    PIPX_BIN_DIR=/var/lib/gpustack/bin \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
