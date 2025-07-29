ARG BASE=registry.cn-qingdao.aliyuncs.com/wod/cann:8.0.RC3-py310-kernel910b

FROM $BASE

ARG AUTHOR=mengkzhaoyun@gmail.com
ARG VERSION=v0.3.2

LABEL maintainer=$AUTHOR version=$VERSION

COPY ./dist/*.whl /tmp/

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's/http:\/\/ports.ubuntu.com\/ubuntu-ports\//https:\/\/mirrors.tuna.tsinghua.edu.cn\/ubuntu-ports\//g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get install -y \
  gcc \
  python3-dev && \
  apt clean && \
  rm -rf /var/lib/apt/lists/*

RUN WHEEL_PACKAGE="$(ls /tmp/*-any.whl)" && \
  pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
  pip3 install --use-pep517 $WHEEL_PACKAGE &&\
  rm /tmp/*.whl && \
  pip3 cache purge

RUN gpustack download-tools \
  --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode' \
  --arch arm64 --device npu

ENTRYPOINT [ "gpustack", "start" ]
