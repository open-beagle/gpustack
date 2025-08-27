ARG BASE=gpustack/gpustack:v0.7.0-corex

FROM $BASE

ARG AUTHOR=gaoshiyao@gmail.com
ARG VERSION=v0.7.0

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

RUN pip install --upgrade --no-cache-dir numpy==2.2.6 -i $POETRY_PYPI_MIRROR_INDEX

RUN pip install /tmp/*.whl --force-reinstall && \
    pip cache purge && \
    rm -rf /tmp/*.whl

RUN gpustack download-tools

ENTRYPOINT [ "gpustack", "start" ]
