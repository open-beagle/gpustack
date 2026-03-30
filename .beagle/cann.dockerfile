ARG BASE=registry.cn-qingdao.aliyuncs.com/wod/cann:8.0.RC3-py310-kernel910b

FROM $BASE

ARG AUTHOR=mengkzhaoyun@gmail.com
ARG VERSION=v0.3.2

LABEL maintainer=$AUTHOR version=$VERSION

COPY ./dist/*.whl /tmp/

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|http://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    wget \
    tzdata \
    gcc \
    g++ \
    tini \
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
    libxkbcommon-x11-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG PYPI_MIRROR=http://mirrors.aliyun.com/pypi/simple/
ARG PYPI_HOST=mirrors.aliyun.com

RUN python3 -m pip install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} pipx

RUN WHEEL_PACKAGE="$(ls /tmp/*-any.whl)" && \
  # 只安装基础 GPUStack，刻意丢弃 [vllm] extra 以绕开不支持 ARM64 的 bitsandbytes (量化主要交由华为自有框架)
  pip3 install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} --use-pep517 "${WHEEL_PACKAGE}" &&\
  # 手动补齐不含硬件绑定偏见的多模态与大语言模型运行库
  VLLM_TARGET_DEVICE=empty pip3 install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} "vllm==0.17.0" "mistral_common>=1.4.3" "timm>=1.0.15" &&\
  # 继续补装对应的华为 NPU 架构后端实现插件
  pip3 install vllm-ascend==0.17.0rc1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/simple -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} &&\
  # 强制升级 transformers 以支持最新模型架构
  pip3 install -i ${PYPI_MIRROR} --trusted-host ${PYPI_HOST} "transformers>=5.3.0" &&\
  python3 -c "import transformers.modeling_rope_utils as m; p = m.__file__; c = open(p, 'r').read(); c = c.replace('received_keys -= ignore_keys', 'received_keys -= set(ignore_keys)'); open(p, 'w').write(c)" &&\
  python3 -c "import os; p = '/usr/local/python3.11.14/lib/python3.11/site-packages/torchaudio/_extension/utils.py'; c = open(p, 'r').read() if os.path.exists(p) else ''; c = c.replace('torch.ops.load_library(paths[0])\n    return True', 'try:\n        torch.ops.load_library(paths[0])\n        return True\n    except Exception:\n        return False') if c else c; open(p, 'w').write(c) if c else None" &&\
  rm /tmp/*.whl && \
  pip3 cache purge

ENV CANN_VERSION=8.2.rc1
RUN TORCH_DEVICE_BACKEND_AUTOLOAD=0 gpustack download-tools \
  --arch arm64 --device npu \
  --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'

# 设置目录
RUN mkdir -p /var/lib/gpustack && \
    chmod -R 0755 /var/lib/gpustack

# 设置环境变量
ENV PIPX_HOME=/var/lib/gpustack/pipx \
    PIPX_LOCAL_VENVS=/var/lib/gpustack/pipx/venvs \
    PIPX_BIN_DIR=/var/lib/gpustack/bin \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
