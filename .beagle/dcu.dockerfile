ARG BASE=gpustack/gpustack:v0.7.0-dcu

FROM $BASE

ARG AUTHOR=gaoshiyao@gmail.com
ARG VERSION=v0.7.0

LABEL maintainer=$AUTHOR version=$VERSION

COPY ./dist/*.whl /tmp/

RUN python3 -m pip install pipx \
    && pipx ensurepath --force \
    && WHEEL_PACKAGE="$(ls /tmp/*.whl)[audio]" \
    && echo $WHEEL_PACKAGE \
    && pipx install --force-reinstall $WHEEL_PACKAGE \
    && pip cache purge

RUN gpustack download-tools --device dcu \
    && ln -s $(which vllm) /root/.local/share/pipx/venvs/gpustack/bin/vllm
