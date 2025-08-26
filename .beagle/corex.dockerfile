ARG BASE=gpustack/gpustack:v0.7.0-corex

FROM $BASE

ARG AUTHOR=gaoshiyao@gmail.com
ARG VERSION=v0.7.0

LABEL maintainer=$AUTHOR version=$VERSION
COPY ./dist/*.whl /tmp/

RUN WHEEL_PACKAGE="$(ls /tmp/**-any.whl)[vllm]" && \
  pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
  pip3 install $WHEEL_PACKAGE &&\
  rm /tmp/*.whl && \
  pip3 cache purge

RUN gpustack download-tools 
    # \ --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'

ENTRYPOINT [ "gpustack", "start" ]
