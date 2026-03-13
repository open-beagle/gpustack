ARG BASE=registry.cn-qingdao.aliyuncs.com/wod/musa:rc4.2.0-runtime-ubuntu-amd64

FROM $BASE

ARG AUTHOR=mengkzhaoyun@gmail.com
ARG VERSION=v0.3.2

LABEL maintainer=$AUTHOR version=$VERSION

COPY ./dist/*.whl /tmp/

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    tzdata \
    iproute2 \
    tini \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install -i https://mirrors.cloud.aliyuncs.com/pypi/simple/ --timeout=600 --retries=10 /tmp/*.whl && \
    pip cache purge && rm -rf /tmp/*.whl

RUN gpustack download-tools

ENTRYPOINT [ "gpustack", "start" ]
