ARG BASE=registry-vpc.cn-qingdao.aliyuncs.com/wod/windstackbase:cuda12.8.1-py3.10-vllm0.22.1-omni0.22.0

FROM $BASE

# 安装 GPUStack 应用层。底层运行时依赖已内置在 windstackbase 镜像中。
COPY ./dist/*.whl /tmp/
RUN WHEEL_PACKAGE="$(ls /tmp/*.whl)" && \
    pip3 install --no-cache-dir --no-deps --force-reinstall "${WHEEL_PACKAGE}" && \
    rm -rf /tmp/*.whl

ENV GPUSTACK_THIRD_PARTY_BIN=/opt/gpustack/third_party/bin

RUN python3 -c "from gpustack.worker.tools_manager import ToolsManager; tools_manager = ToolsManager(tools_download_base_url='https://cache.ali.wodcloud.com/vscode', system='linux', arch='amd64', device='cuda'); tools_manager.remove_cached_tools(); tools_manager.download_llama_box(); tools_manager.download_gguf_parser(); tools_manager.download_fastfetch(); tools_manager.install_llama_cpp()"

ARG VERSION=dev
LABEL version=$VERSION

ENTRYPOINT [ "tini", "--", "gpustack", "start" ]
