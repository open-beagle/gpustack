ARG BASE=ghcr.io/open-beagle/gpustack:cuda13.0.3-vllm0.26.0-omni0.26.0

FROM $BASE

# 防止应用镜像继续复用包含 compat 驱动库的旧基础镜像。
RUN case ":${LD_LIBRARY_PATH}:" in \
      *:/usr/local/cuda/compat:*) echo "CUDA compat 不得进入运行时 LD_LIBRARY_PATH" >&2; exit 1 ;; \
      *) exit 0 ;; \
    esac

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
