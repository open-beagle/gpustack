ARG BASE=registry-vpc.cn-qingdao.aliyuncs.com/wod/windstackbase:cuda12.8.1-py3.10-vllm0.22.1-omni0.22.0

FROM $BASE

# 安装 GPUStack 应用层。底层运行时依赖和工具已内置在 windstackbase 镜像中。
COPY ./dist/*.whl /tmp/
RUN WHEEL_PACKAGE="$(ls /tmp/*.whl)" && \
    pip3 install --no-cache-dir --no-deps --force-reinstall "${WHEEL_PACKAGE}" && \
    rm -rf /tmp/*.whl

ARG VERSION=dev
LABEL version=$VERSION

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
