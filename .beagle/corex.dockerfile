ARG BASE=registry.cn-qingdao.aliyuncs.com/wod/corex:mr-bi150-4.3.0-x86-ubuntu20.04-py3.10-poc-llm-infer-v1.2.3

FROM ${BASE}

COPY ./dist/*.whl /dist/

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    tzdata \
    iproute2 \
    tini \
    pipx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install /dist/*.whl && \
    # 安装 vLLM-Omni 用于支持 Diffusion 模型（Z-Image、Flux 等）
    pip install vllm-omni && \
    pip cache purge && \
    rm -rf /dist && \
    ln -s /usr/local/corex-4.3.0/lib64/python3/dist-packages/bin/vllm /usr/local/bin/vllm

RUN gpustack download-tools

# 设置环境变量
ENV PIPX_HOME=/var/lib/gpustack/pipx \
    PIPX_LOCAL_VENVS=/var/lib/gpustack/pipx/venvs \
    PIPX_BIN_DIR=/var/lib/gpustack/bin \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
