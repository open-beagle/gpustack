ARG BASE=registry.cn-qingdao.aliyuncs.com/wod/cann:8.0.RC3-py310-kernel910b

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

RUN WHEEL_PACKAGE="$(ls /tmp/*_aarch64.whl)" && \
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
  pip3 install $WHEEL_PACKAGE &&\
  rm /tmp/*.whl && \
  pip3 cache purge

ENTRYPOINT [ "gpustack", "start" ]
