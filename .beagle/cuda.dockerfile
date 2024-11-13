ARG BASE=registry.cn-qingdao.aliyuncs.com/wod/cuda:12.4.1-py310

FROM $BASE

ARG AUTHOR=mengkzhaoyun@gmail.com
ARG VERSION=v0.3.2

LABEL maintainer=$AUTHOR version=$VERSION

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
  wget \
  tzdata \
  && rm -rf /var/lib/apt/lists/*

COPY dist/*.whl /tmp/

RUN WHEEL_PACKAGE="$(ls /tmp/*_x86_64.whl)[vllm]" && \
  pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
  pip3 install $WHEEL_PACKAGE &&\
  rm /tmp/*.whl && \
  pip3 cache purge

ENTRYPOINT [ "gpustack", "start" ]
