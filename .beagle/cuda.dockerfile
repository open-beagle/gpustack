ARG BASE=ghcr.io/open-beagle/gpustack:cuda13.0.3-vllm0.26.0-omni0.26.0

FROM $BASE

# 防止应用镜像继续复用包含 compat 驱动库的旧基础镜像。
RUN case ":${LD_LIBRARY_PATH}:" in \
      *:/usr/local/cuda/compat:*) echo "CUDA compat 不得进入运行时 LD_LIBRARY_PATH" >&2; exit 1 ;; \
      *) exit 0 ;; \
    esac

# 安装 GPUStack 应用层。底层运行时依赖已内置在 CUDA runtime 镜像中。
COPY ./dist/*.whl /tmp/
COPY ./dist/requirements-vllm.txt /tmp/requirements-vllm.txt
RUN python3 -m pip install --no-cache-dir --default-timeout=12000 \
        -r /tmp/requirements-vllm.txt && \
    WHEEL_PACKAGE="$(ls /tmp/*.whl)" && \
    pip3 install --no-cache-dir --no-deps --force-reinstall "${WHEEL_PACKAGE}" && \
    python3 -m pip check && \
    python3 -c "import modelscope_hub" && \
    python3 -m gpustack.migrations.validate && \
    rm -f /tmp/*.whl /tmp/requirements-vllm.txt

ENV GPUSTACK_THIRD_PARTY_BIN=/opt/gpustack/third_party/bin \
    HF_ENDPOINT=https://hf-mirror.com

COPY ./.beagle/tools-downloads.sh /tmp/gpustack-tools-downloads.sh
RUN TOOLS_DOWNLOAD_BASE_URL=https://cache.ali.wodcloud.com/vscode \
    SYSTEM=linux \
    ARCH=amd64 \
    DEVICE=cuda \
    bash /tmp/gpustack-tools-downloads.sh && \
    rm -f /tmp/gpustack-tools-downloads.sh

# 配置容器运行时镜像源为阿里云（Apt 与 Pip）
RUN if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
            -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
            /etc/apt/sources.list.d/ubuntu.sources; \
    fi && \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
            -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
            -e 's|http://ports.ubuntu.com/ubuntu-ports|https://mirrors.aliyun.com/ubuntu-ports|g' \
            /etc/apt/sources.list; \
    fi

ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ARG VERSION=dev
LABEL version=$VERSION

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
