ARG BASE=registry.cn-qingdao.aliyuncs.com/wod/mr-bi150-4.3.0-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3 AS runtime

FROM $BASE

COPY ./dist/. /workspace/gpustack/dist

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    tzdata \
    iproute2 \
    tini \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=build /workspace/gpustack/dist/*.whl /dist/
RUN pip install /dist/*.whl && \
    pip cache purge && \
    rm -rf /dist

RUN gpustack download-tools

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
