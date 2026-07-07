ARG BASE=nvidia/cuda:12.8.2-runtime-ubuntu22.04

FROM $BASE

ARG AUTHOR=mengkzhaoyun@gmail.com

LABEL maintainer=$AUTHOR

ENV DEBIAN_FRONTEND=noninteractive

# 配置 Ubuntu 镜像源（使用阿里云镜像）
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list

# 安装系统依赖 + 设置时区
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    wget \
    tzdata \
    python3 \
    python3-pip \
    pipx \
    python3-dev \
    tini \
    gcc \
    g++ \
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
    cuda-nvcc-12-8 \
    libcurand-dev-12-8 \
    cuda-nvrtc-dev-12-8 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

ARG TOOLS_DOWNLOAD_BASE_URL=https://cache.ali.wodcloud.com/vscode
ARG LLAMA_CPP_VERSION=b8322
# Keep this at 12.8.1 until the matching cached llama.cpp package is available.
ARG LLAMA_CPP_CUDA_VERSION=12.8.1

# 预置 CUDA llama.cpp 大二进制包。该层只随 CUDA/llama.cpp 底层版本变化重建。
RUN set -eux; \
    third_party_bin=/opt/gpustack/third_party/bin; \
    package_name="llama-cpp-cuda-${LLAMA_CPP_CUDA_VERSION}-${LLAMA_CPP_VERSION}-linux-x64"; \
    version_dir="llama.cpp-${LLAMA_CPP_VERSION}-linux-amd64-cuda"; \
    target_dir="${third_party_bin}/llama.cpp/${version_dir}"; \
    tmp_dir="$(mktemp -d)"; \
    mkdir -p "${target_dir}"; \
    curl -fL --retry 5 --retry-delay 2 \
      "${TOOLS_DOWNLOAD_BASE_URL}/gpustack/llama.cpp/${package_name}.tar.gz" \
      -o "${tmp_dir}/${package_name}.tar.gz"; \
    tar -xzf "${tmp_dir}/${package_name}.tar.gz" -C "${target_dir}"; \
    rm -rf "${tmp_dir}"; \
    test -x "${target_dir}/llama-server" || chmod +x "${target_dir}/llama-server"; \
    chmod +x "${target_dir}/llama-server" "${target_dir}/llama-cli" "${target_dir}/rpc-server"; \
    printf '{\n    "%s": "%s"\n}\n' "${version_dir}" "${LLAMA_CPP_VERSION}" > "${third_party_bin}/versions.json"

ARG PYPI_MIRROR=https://mirrors.aliyun.com/pypi/simple/
ARG PYPI_HOST=mirrors.aliyun.com
ARG PYPI_EXTRA_INDEX_URL=https://pypi.org/simple/

# 安装稳定运行时依赖层。只有 CUDA/Python/vLLM/vLLM-Omni/transformers 等底层版本变化时才需要重打该基础镜像。
COPY ./dist/requirements-vllm.txt /tmp/requirements-vllm.txt
RUN python3 -m pip install -i ${PYPI_MIRROR} --extra-index-url ${PYPI_EXTRA_INDEX_URL} --trusted-host ${PYPI_HOST} --upgrade pip && \
    python3 -m pip install -i ${PYPI_MIRROR} --extra-index-url ${PYPI_EXTRA_INDEX_URL} --trusted-host ${PYPI_HOST} --no-cache-dir --upgrade 'pipx==1.7.1' 'argcomplete>=1.9.4' && \
    python3 -m pip install -i ${PYPI_MIRROR} --extra-index-url ${PYPI_EXTRA_INDEX_URL} --trusted-host ${PYPI_HOST} --no-cache-dir --default-timeout=12000 -r /tmp/requirements-vllm.txt && \
    # 修复 transformers RoPE 验证类型错误 (set -= list)
    python3 -c "\
import transformers.modeling_rope_utils as m; \
p = m.__file__; \
c = open(p, 'r').read(); \
c = c.replace('received_keys -= ignore_keys', 'received_keys -= set(ignore_keys)'); \
open(p, 'w').write(c)" && \
    command -v vllm && \
    command -v vllm-omni && \
    python3 -c "\
import importlib.metadata as m; \
from packaging.version import Version; \
assert Version(m.version('argcomplete')) >= Version('1.9.4'), m.version('argcomplete'); \
print('argcomplete', m.version('argcomplete'))" && \
    pipx --version && \
    python3 -m pip show vllm vllm-omni && \
    rm -f /tmp/requirements-vllm.txt

# 设置目录
RUN mkdir -p /var/lib/gpustack && \
    chmod -R 0755 /var/lib/gpustack

# 设置环境变量
ENV PIPX_HOME=/var/lib/gpustack/pipx \
    PIPX_LOCAL_VENVS=/var/lib/gpustack/pipx/venvs \
    PIPX_BIN_DIR=/var/lib/gpustack/bin \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ARG VERSION=dev
LABEL version=$VERSION
