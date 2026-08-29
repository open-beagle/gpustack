ARG BASE=nvidia/cuda:13.0.3-runtime-ubuntu24.04

FROM $BASE

ARG AUTHOR=mengkzhaoyun@gmail.com

LABEL maintainer=$AUTHOR

ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖 + 设置时区
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    wget \
    tzdata \
    python3 \
    python3-pip \
    python3-venv \
    pipx \
    python3-dev \
    tini \
    gcc \
    g++ \
    ffmpeg \
    libgl1 \
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
    cuda-nvcc-13-0 \
    cuda-cudart-13-0 \
    cuda-nvrtc-13-0 \
    cuda-cuobjdump-13-0 \
    libcurand-dev-13-0 \
    libcublas-dev-13-0 \
    libnuma-dev \
    numactl \
    && NCCL_VERSION="$(apt-cache madison libnccl-dev | awk -F'|' '/[+]cuda13[.]0/ {gsub(/^ +| +$/, "", $2); print $2; exit}')" \
    && test -n "${NCCL_VERSION}" \
    && apt-get install -y --no-install-recommends --allow-change-held-packages \
      "libnccl-dev=${NCCL_VERSION}" "libnccl2=${NCCL_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

RUN python3 -m venv /opt/gpustack/venv

ENV PATH=/opt/gpustack/venv/bin:${PATH}

ARG CUDA_VERSION=13.0.3

COPY ./.beagle/llama.cpp.lock /tmp/llama.cpp.lock
COPY ./dist/llama.cpp.tar.gz /tmp/llama.cpp.tar.gz
COPY ./.beagle/llama-box.lock /tmp/llama-box.lock
COPY ./dist/llama-box.tar.gz /tmp/llama-box.tar.gz
RUN set -eux; \
    expected_cuda_version="${CUDA_VERSION}"; \
    . /tmp/llama.cpp.lock; \
    . /tmp/llama-box.lock; \
    test "${CUDA_VERSION}" = "${expected_cuda_version}"; \
    test -n "${LLAMA_CPP_VERSION}"; \
    test -n "${LLAMA_CPP_COMMIT}"; \
    test -n "${LLAMA_BOX_VERSION}"; \
    test -n "${LLAMA_BOX_COMMIT}"; \
    test "${LLAMA_CPP_PACKAGE}" = "llama-cpp-cuda-${CUDA_VERSION}-${LLAMA_CPP_VERSION}-linux-x64"; \
    third_party_bin=/opt/gpustack/third_party/bin; \
    llama_cpp_dir="llama.cpp-${LLAMA_CPP_VERSION}-linux-amd64-cuda"; \
    llama_box_dir="llama-box-${LLAMA_BOX_VERSION}-linux-amd64-cuda"; \
    test "${LLAMA_BOX_PACKAGE}" = "llama-box-cuda-${CUDA_VERSION}-${LLAMA_BOX_VERSION}-linux-x64"; \
    mkdir -p "${third_party_bin}/llama.cpp/${llama_cpp_dir}"; \
    mkdir -p "${third_party_bin}/llama-box/${llama_box_dir}"; \
    tar -xzf /tmp/llama.cpp.tar.gz \
      -C "${third_party_bin}/llama.cpp/${llama_cpp_dir}"; \
    tar -xzf /tmp/llama-box.tar.gz \
      -C "${third_party_bin}/llama-box/${llama_box_dir}"; \
    if find "${third_party_bin}/llama-box/${llama_box_dir}" -maxdepth 1 -type f \
      \( -name llama-box -o -name 'lib*.so*' \) -exec ldd {} \; \
      | grep -Eq 'lib(cudart|cublas).*\.so\.12'; then \
      echo "llama-box unexpectedly links against CUDA 12" >&2; \
      exit 1; \
    fi; \
    ln -s llama-box "${third_party_bin}/llama-box/${llama_box_dir}/llama-box-rpc-server"; \
    ln -s "${llama_box_dir}" "${third_party_bin}/llama-box/llama-box-default"; \
    printf '{\n  "%s": "%s",\n  "%s": "%s"\n}\n' \
      "${llama_cpp_dir}" "${LLAMA_CPP_VERSION}" \
      "${llama_box_dir}" "${LLAMA_BOX_VERSION}" \
      > "${third_party_bin}/versions.json"; \
    install -m 0644 /tmp/llama.cpp.lock "${third_party_bin}/llama.cpp.lock"; \
    install -m 0644 /tmp/llama-box.lock "${third_party_bin}/llama-box.lock"; \
    test -x "${third_party_bin}/llama.cpp/${llama_cpp_dir}/llama-server"; \
    test -x "${third_party_bin}/llama.cpp/${llama_cpp_dir}/ggml-rpc-server"; \
    test -x "${third_party_bin}/llama-box/llama-box-default/llama-box"; \
    test -x "${third_party_bin}/llama-box/llama-box-default/llama-box-rpc-server"; \
    rm -f /tmp/llama.cpp.tar.gz /tmp/llama.cpp.lock \
      /tmp/llama-box.tar.gz /tmp/llama-box.lock

ARG PYPI_MIRROR=https://pypi.org/simple/

# 安装稳定运行时依赖层。只有 CUDA/Python/vLLM/vLLM-Omni/transformers 等底层版本变化时才需要重打该基础镜像。
COPY ./dist/requirements-vllm.txt /tmp/requirements-vllm.txt
COPY ./.beagle/prepare_cuda_base.py /tmp/prepare_cuda_base.py
RUN python3 -m pip install -i ${PYPI_MIRROR} --upgrade pip && \
    python3 -m pip install -i ${PYPI_MIRROR} --no-cache-dir --upgrade 'pipx==1.7.1' 'argcomplete>=1.9.4' && \
    python3 -m pip install -i ${PYPI_MIRROR} --no-cache-dir --default-timeout=12000 -r /tmp/requirements-vllm.txt && \
    command -v vllm && \
    command -v vllm-omni && \
    nvcc --version | grep -F 'release 13.0' && \
    pipx --version && \
    python3 -m pip check && \
    python3 /tmp/prepare_cuda_base.py && \
    rm -f /tmp/requirements-vllm.txt /tmp/prepare_cuda_base.py

# 设置目录
RUN mkdir -p /var/lib/gpustack && \
    chmod -R 0755 /var/lib/gpustack

# 禁止将 CUDA compat 放入运行时搜索路径，避免覆盖宿主机注入的 libcuda。
ENV LD_LIBRARY_PATH=/opt/gpustack/venv/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    PIPX_HOME=/var/lib/gpustack/pipx \
    PIPX_LOCAL_VENVS=/var/lib/gpustack/pipx/venvs \
    PIPX_BIN_DIR=/var/lib/gpustack/bin

RUN case ":${LD_LIBRARY_PATH}:" in \
      *:/usr/local/cuda/compat:*) echo "CUDA compat 不得进入运行时 LD_LIBRARY_PATH" >&2; exit 1 ;; \
      *) exit 0 ;; \
    esac

ARG VERSION=dev
LABEL version=$VERSION
