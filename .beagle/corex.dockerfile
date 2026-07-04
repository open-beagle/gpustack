ARG BASE=registry.cn-qingdao.aliyuncs.com/wod/corex:mr-bi150-4.3.0-x86-ubuntu20.04-py3.10-poc-llm-infer-v1.2.3

FROM ${BASE}

ARG PYPI_MIRROR=http://mirrors.cloud.aliyuncs.com/pypi/simple/
ARG PYPI_HOST=mirrors.cloud.aliyuncs.com

ENV DEBIAN_FRONTEND=noninteractive

COPY ./dist/requirements-vllm.txt /tmp/requirements-vllm.txt
COPY ./dist/*.whl /tmp/
COPY .beagle/corex-constraints.txt /tmp/corex-constraints.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    wget \
    tzdata \
    iproute2 \
    tini \
    pipx \
    python3 \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

# CoreX 基础镜像内置厂商适配的 torch/vLLM 栈，不能用 CUDA/上游 PyPI 版本覆盖。
RUN grep -Ev '^(vllm|vllm-omni|torch|torchvision|torchaudio|triton|nvidia-|cuda-|cupy|bitsandbytes|transformers|mistral_common|timm)([<>=!~ ].*|\[.*|$)' \
      /tmp/requirements-vllm.txt > /tmp/requirements-corex.txt && \
    printf '%s\n' 'transformers<5.0.0' 'vllm<0.12' 'torch<2.11' > /tmp/corex-runtime-constraints.txt && \
    python3 -m pip install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} --upgrade pip && \
    pip3 install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} --no-cache-dir --default-timeout=12000 -r /tmp/corex-constraints.txt && \
    pip3 install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} --no-cache-dir --default-timeout=12000 -r /tmp/requirements-corex.txt -c /tmp/corex-constraints.txt -c /tmp/corex-runtime-constraints.txt && \
    WHEEL_PACKAGE="$(ls /tmp/*.whl)" && \
    pip3 install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} --no-cache-dir --no-deps --force-reinstall "${WHEEL_PACKAGE}" && \
    ln -sf /usr/local/corex-4.3.0/lib64/python3/dist-packages/bin/vllm /usr/local/bin/vllm && \
    command -v vllm && \
    python3 -m pip show vllm && \
    pip cache purge && \
    rm -rf /tmp/*.whl /tmp/requirements-vllm.txt /tmp/requirements-corex.txt /tmp/corex-constraints.txt /tmp/corex-runtime-constraints.txt

RUN gpustack download-tools --device corex --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'

# 设置环境变量
ENV PIPX_HOME=/var/lib/gpustack/pipx \
    PIPX_LOCAL_VENVS=/var/lib/gpustack/pipx/venvs \
    PIPX_BIN_DIR=/var/lib/gpustack/bin \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
