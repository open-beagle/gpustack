ARG BASE=gpustack/gpustack:v0.7.0-dcu

FROM $BASE

ARG AUTHOR=gaoshiyao@gmail.com
ARG VERSION=v0.7.0

LABEL maintainer=$AUTHOR version=$VERSION
COPY ./dist/*.whl /tmp/

RUN pip install --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout=600 --retries=10 /tmp/*.whl && \
    pip cache purge && rm -rf /tmp/*.whl /tmp/corex-constraints.txt


RUN python3 -m pip install pipx \
    && pipx ensurepath --force \
    && WHEEL_PACKAGE="$(ls /tmp/*.whl)[audio]" \
    && echo $WHEEL_PACKAGE \
    && pipx install --force-reinstall $WHEEL_PACKAGE \
    && pip cache purge

RUN gpustack download-tools --device dcu \
    && ln -s $(which vllm) /root/.local/share/pipx/venvs/gpustack/bin/vllm
