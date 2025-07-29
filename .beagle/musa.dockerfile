ARG BASE=registry.cn-guangzhou.aliyuncs.com/cloud-ysf/musa:rc4.2.0-runtime-ubuntu-amd64

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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install /tmp/*.whl && \
    pip cache purge && \
    rm -rf /tmp/*.whl

RUN gpustack download-tools \
  --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'

ENTRYPOINT [ "gpustack", "start" ]
