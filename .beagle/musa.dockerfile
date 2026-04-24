ARG BASE=registry.cn-qingdao.aliyuncs.com/wod/musa:rc4.2.0-runtime-ubuntu-amd64

FROM $BASE

ARG AUTHOR=mengkzhaoyun@gmail.com
ARG VERSION=v0.3.2

LABEL maintainer=$AUTHOR version=$VERSION

COPY ./dist/*.whl /tmp/

RUN apt-get update && apt-get install -y \
    python3 \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb1 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    python3-pip \
    wget \
    tzdata \
    iproute2 \
    tini \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install -i http://mirrors.cloud.aliyuncs.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com --timeout=600 --retries=10 /tmp/*.whl && \
    pip cache purge && rm -rf /tmp/*.whl

RUN gpustack download-tools

ENTRYPOINT [ "gpustack", "start" ]
