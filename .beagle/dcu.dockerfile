ARG BASE=image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-py3.10

FROM $BASE

ARG AUTHOR=gaoshiyao@gmail.com
ARG VERSION=v0.7.0

LABEL maintainer=$AUTHOR version=$VERSION

ENV PATH="/root/.local/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive

# 首先设置PyPI镜像
RUN pip config set global.index-url http://mirrors.cloud.aliyuncs.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.cloud.aliyuncs.com

ENV PIP_INDEX_URL=http://mirrors.cloud.aliyuncs.com/pypi/simple/
ENV POETRY_SOURCE_0_URL=http://mirrors.cloud.aliyuncs.com/pypi/simple/

RUN apt-get update && apt-get install -y \
    python3-venv \
    tzdata \
    iproute2 \
    iputils-ping \
    build-essential \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY ./dist/*.whl /tmp/

# 增加超时重试参数
RUN pip config set global.index-url http://mirrors.cloud.aliyuncs.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.cloud.aliyuncs.com \
    && python3 -m pip install pipx \
    && pipx ensurepath --force \
    && WHEEL_PACKAGE="$(ls /tmp/*-any.whl)" \
    && pipx install $WHEEL_PACKAGE --pip-args="--index-url http://mirrors.cloud.aliyuncs.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com --timeout=600 --retries=5" \
    && pip cache purge

RUN gpustack download-tools --device dcu \
    && ln -s $(which vllm) /root/.local/share/pipx/venvs/gpustack/bin/vllm

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
