ARG BASE=gpustack/gpustack:v0.7.0-dcu

FROM $BASE

ARG AUTHOR=gaoshiyao@gmail.com
ARG VERSION=v0.7.0

LABEL maintainer=$AUTHOR version=$VERSION

# 复制您编译的包文件
COPY ./dist/*.whl /tmp/

# 彻底移除原有的 gpustack 安装
RUN if pipx list | grep -q gpustack; then pipx uninstall gpustack; fi \
    && python3 -m pip uninstall -y gpustack || true \
    && rm -rf /root/.local/share/pipx/venvs/gpustack \
    && rm -rf /root/.local/pipx/venvs/gpustack

# 清理 pip 缓存并重新安装
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install pipx \
    && pipx ensurepath --force

# 安装您编译的 wheel 包（两种方式可选）
# 方式1：使用 pipx 安装（推荐）
RUN pipx install --force "/tmp/gpustack-0.0.0-py3-none-any.whl[audio]"

# 方式2：或者使用 pip 直接安装到系统
# RUN python3 -m pip install "/tmp/gpustack-0.0.0-py3-none-any.whl[audio]"

# 下载工具并创建符号链接
RUN gpustack download-tools --device dcu \
    && ln -sf $(which vllm) /root/.local/share/pipx/venvs/gpustack/bin/vllm

# 验证安装
RUN python3 -c "import gpustack; print(f'Successfully installed gpustack version: {gpustack.__version__}')" \
    && echo "Installation verification:" \
    && gpustack --version