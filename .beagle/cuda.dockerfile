ARG BASE=ghcr.io/open-beagle/gpustack:cuda12.8.2-vllm0.24.0-omni0.24.0

FROM $BASE

# 安装 GPUStack 应用层。底层运行时依赖已内置在 CUDA runtime 镜像中。
COPY ./dist/*.whl /tmp/
RUN WHEEL_PACKAGE="$(ls /tmp/*.whl)" && \
    pip3 install --no-cache-dir --no-deps --force-reinstall "${WHEEL_PACKAGE}" && \
    rm -rf /tmp/*.whl

ENV GPUSTACK_THIRD_PARTY_BIN=/opt/gpustack/third_party/bin

COPY ./.beagle/tools-downloads.sh /tmp/gpustack-tools-downloads.sh
RUN TOOLS_DOWNLOAD_BASE_URL=https://cache.ali.wodcloud.com/vscode \
    SYSTEM=linux \
    ARCH=amd64 \
    DEVICE=cuda \
    bash /tmp/gpustack-tools-downloads.sh && \
    rm -f /tmp/gpustack-tools-downloads.sh

ARG VERSION=dev
LABEL version=$VERSION

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
