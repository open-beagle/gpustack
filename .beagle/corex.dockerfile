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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install /dist/*.whl && \
    pip cache purge && \
    rm -rf /dist && \
    ln -s /usr/local/corex-4.3.0/lib64/python3/dist-packages/bin/vllm /usr/local/bin/vllm

RUN gpustack download-tools

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
