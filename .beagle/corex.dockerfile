ARG BASE=gpustack/gpustack:v0.7.0-corex

FROM $BASE

ARG AUTHOR=gaoshiyao@gmail.com
ARG VERSION=v0.7.0

LABEL maintainer=$AUTHOR version=$VERSION
COPY ./dist/*.whl /tmp/
COPY .beagle/corex-constraints.txt /tmp/corex-constraints.txt

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    tzdata \
    iproute2 \
    tini \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-deps --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout=600 --retries=10 /tmp/*.whl && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout=600 --retries=10 --upgrade --upgrade-strategy eager -r /tmp/corex-constraints.txt && \
    pip cache purge && rm -rf /tmp/*.whl /tmp/corex-constraints.txt

RUN gpustack download-tools

ENTRYPOINT [ "gpustack", "start" ]
