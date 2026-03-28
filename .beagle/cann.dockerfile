ARG BASE=registry.cn-qingdao.aliyuncs.com/wod/cann:8.0.RC3-py310-kernel910b

FROM $BASE

ARG AUTHOR=mengkzhaoyun@gmail.com
ARG VERSION=v0.3.2

LABEL maintainer=$AUTHOR version=$VERSION

COPY ./dist/*.whl /tmp/

RUN WHEEL_PACKAGE="$(ls /tmp/*-any.whl)" && \
  pip3 config set global.index-url http://mirrors.cloud.aliyuncs.com/pypi/simple/ && \
  pip3 config set global.trusted-host mirrors.cloud.aliyuncs.com && \
  pip3 install --use-pep517 $WHEEL_PACKAGE &&\
  rm /tmp/*.whl && \
  pip3 cache purge

ENV CANN_VERSION=8.2.rc1
RUN gpustack download-tools \
  --arch arm64 --device npu \
  --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'

ENTRYPOINT [ "gpustack", "start" ]
