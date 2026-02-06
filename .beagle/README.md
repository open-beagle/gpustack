# gpustack

<https://github.com/open-beagle/gpustack>

## git

```bash
git remote add upstream git@github.com:gpustack/gpustack.git

git fetch upstream

git merge v0.7.1
```

## images

<https://hub.docker.com/r/gpustack/gpustack>

```bash
docker pull gpustack/gpustack:0.7.1 && \
docker tag gpustack/gpustack:0.7.1 registry.cn-qingdao.aliyuncs.com/wod/windstack:0.7.1 && \
docker push registry.cn-qingdao.aliyuncs.com/wod/windstack:0.7.1
```

## base images

```bash
# cuda
docker pull nvidia/cuda:12.8.1-runtime-ubuntu22.04 && \
docker tag nvidia/cuda:12.8.1-runtime-ubuntu22.04 registry.cn-qingdao.aliyuncs.com/wod/cuda:12.8.1-runtime-ubuntu22.04 && \
docker push registry.cn-qingdao.aliyuncs.com/wod/cuda:12.8.1-runtime-ubuntu22.04

# cann
docker pull --platform=linux/arm64 ascendai/cann:8.0.rc3.beta1-910b-ubuntu22.04-py3.10 && \
docker tag ascendai/cann:8.0.rc3.beta1-910b-ubuntu22.04-py3.10 registry.cn-qingdao.aliyuncs.com/wod/cann:8.0.rc3.beta1-910b-ubuntu22.04-py3.10 && \
docker push registry.cn-qingdao.aliyuncs.com/wod/cann:8.0.rc3.beta1-910b-ubuntu22.04-py3.10

# corex
docker pull git.modelhub.org.cn:9443/enginex-iluvatar/mr-bi150-4.3.0-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3 && \
docker tag git.modelhub.org.cn:9443/enginex-iluvatar/mr-bi150-4.3.0-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3 registry-vpc.cn-qingdao.aliyuncs.com/wod/corex:mr-bi150-4.3.0-x86-ubuntu20.04-py3.10-poc-llm-infer-v1.2.3 && \
docker push registry-vpc.cn-qingdao.aliyuncs.com/wod/corex:mr-bi150-4.3.0-x86-ubuntu20.04-py3.10-poc-llm-infer-v1.2.3
```

### cuda

```bash
# default user admin
docker run -d --gpus all -p 6080:6080 --ipc=host --shm-size=2g --name gpustack \
  -v /data/gpustack:/var/lib/gpustack \
  registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-cuda \
  --bootstrap-password 'beagle!@#123' --port 6080 \
  --worker-name <host-name> \
  --worker-s3-host=your_s3_host \
  --worker-s3-access-key=your_access_key \
  --worker-s3-secret-key=your_secret_key

docker rm -f windstack&& rm -rf /data/windstack

# start worker node
docker run -d --gpus all --ipc=host --shm-size=2g --name gpustack \
  -p 10150:10150 -p 40000-41024:40000-41024 \
  -v /data/gpustack:/var/lib/gpustack \
  registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-cuda \
  --server-url http://myserver:6080 --token mytoken \
  --worker-ip <host-ip> \
  --worker-name <host-name> \
  --worker-s3-host=your_s3_host \
  --worker-s3-access-key=your_access_key \
  --worker-s3-secret-key=your_secret_key
```

### NPU

```bash
# default user admin
docker run -d -p 6080:6080 --privileged --ipc=host --shm-size=2g --name windstack \
  -v /usr/share/hwdata:/usr/share/hwdata \
  -v /data/windstack/data:/var/lib/gpustack \
  -e ASCEND_VISIBLE_DEVICES=0-7 \
  -e TZ=Asia/Shanghai \
  registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-cann \
  --bootstrap-password 'beagle!@#123' --port 6080 \
  --worker-name <host-name> \
  --worker-s3-host=your_s3_host \
  --worker-s3-access-key=your_access_key \
  --worker-s3-secret-key=your_secret_key

docker rm -f windstack&& rm -rf /data/windstack

# start worker node
docker run -d --ipc=host --shm-size=2g --name windstack \
  -p 10150:10150 -p 40000-41024:40000-41024 \
  -v /usr/share/hwdata:/usr/share/hwdata \
  -v /data/windstack/data:/var/lib/gpustack \
  -e ASCEND_VISIBLE_DEVICES=0-7 \
  registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-cann \
  --server-url http://myserver:6080 --token mytoken \
  --worker-ip <host-ip> \
  --worker-name <host-name> \
  --worker-s3-host=your_s3_host \
  --worker-s3-access-key=your_access_key \
  --worker-s3-secret-key=your_secret_key
```

### Mthreads（摩尔线程）

```bash
# default user admin
docker run -d -p 6080:6080 --privileged --ipc=host --shm-size=2g --name windstack \
  -v /data/windstack/data:/var/lib/gpustack \
  -e MTHREADS_VISIBLE_DEVICES=0-7 \
  -e TZ=Asia/Shanghai \
  registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-musa \
  --bootstrap-password 'beagle!@#123' --port 6080 \
  --worker-name <host-name> \
  --worker-s3-host=your_s3_host \
  --worker-s3-access-key=your_access_key \
  --worker-s3-secret-key=your_secret_key

docker rm -f windstack&& rm -rf /data/windstack

# start worker node
docker run -d --ipc=host --shm-size=2g --name gpustack \
  -p 10150:10150 -p 40000-41024:40000-41024 \
  -v /data/windstack/data:/var/lib/gpustack \
  -e MTHREADS_VISIBLE_DEVICES=0-7 \
  registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-musa \
  --server-url http://myserver:6080 --token mytoken \
  --worker-ip <host-ip> \
  --worker-name <host-name> \
  --worker-s3-host=your_s3_host \
  --worker-s3-access-key=your_access_key \
  --worker-s3-secret-key=your_secret_key
```

### CoreX （天数智芯）

```bash
# default user admin
docker run -d --name windstack \
  -v /lib/modules:/lib/modules \
  -v /dev:/dev \
  --privileged \
  --cap-add=ALL \
  --pid=host \
  --restart=unless-stopped \
  --network=host \
  --ipc=host \
  --shm-size=2g \
  -v /data/windstack/data:/var/lib/gpustack \
  -e TZ=Asia/Shanghai \
  -e VLLM_TARGET_DEVICE=corex \
  registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-corex \
  --bootstrap-password 'beagle!@#123' --port 6080 \
  --worker-name <host-name> \
  --worker-s3-host=your_s3_host \
  --worker-s3-access-key=your_access_key \
  --worker-s3-secret-key=your_secret_key

docker rm -f windstack && rm -rf /data/windstack

# start worker node
docker run -d --name windstack \
  -v /lib/modules:/lib/modules \
  -v /dev:/dev \
  --privileged \
  --cap-add=ALL \
  --pid=host \
  --restart=unless-stopped \
  --network=host \
  --ipc=host \
  --shm-size=2g \
  -v /data/windstack/data:/var/lib/gpustack \
  -e TZ=Asia/Shanghai \
  -e VLLM_TARGET_DEVICE=corex \
  registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-corex \
  --server-url http://myserver:6080 --token mytoken \
  --worker-ip <host-ip> \
  --worker-name <host-name> \
  --worker-s3-host=your_s3_host \
  --worker-s3-access-key=your_access_key \
  --worker-s3-secret-key=your_secret_key
```

### DCU

```bash
# default user admin
docker run -d -p 6080:6080 --privileged --ipc=host --shm-size=2g --name windstack \
  --restart=unless-stopped \
  --device=/dev/kfd \
  --device=/dev/mkfd \
  --device=/dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  --network=host \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /data/windstack/data:/var/lib/gpustack \
  -e TZ=Asia/Shanghai \
  registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-musa \
  --bootstrap-password 'beagle!@#123' --port 6080 \
  --worker-name <host-name> \
  --worker-s3-host=your_s3_host \
  --worker-s3-access-key=your_access_key \
  --worker-s3-secret-key=your_secret_key

docker rm -f windstack&& rm -rf /data/windstack

# start worker node
docker run -d --ipc=host --shm-size=2g --name windstack \
  -p 10150:10150 -p 40000-41024:40000-41024 \
  --restart=unless-stopped \
  --device=/dev/kfd \
  --device=/dev/mkfd \
  --device=/dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /data/windstack/data:/var/lib/gpustack \
  -e TZ=Asia/Shanghai \
  registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-musa \
  --server-url http://myserver:6080 --token mytoken \
  --worker-ip <host-ip> \
  --worker-name <host-name> \
  --worker-s3-host=your_s3_host \
  --worker-s3-access-key=your_access_key \
  --worker-s3-secret-key=your_secret_key
```

## debug

### 本地调试

```bash
# 安装 poetry
curl -sSL https://install.python-poetry.org | python3 -

# 安装 pnpm（Node.js 包管理器）
npm install -g pnpm

# 配置国内镜像源
poetry config repositories.pypi-mirror https://pypi.tuna.tsinghua.edu.cn/simple

# 配置 pip 使用国内镜像（加速依赖安装）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

# 编译前端（可选，如果需要修改 UI）
bash -c "cd gpustack/ui && pnpm install && pnpm build && cp -r dist/* ../"

# 安装依赖（会自动创建虚拟环境 .venv）
# 注意：首次安装可能需要较长时间（30分钟-1小时），因为需要编译 vLLM 等大型依赖
poetry install

# 激活虚拟环境
poetry shell

# 启动服务（无 UI 模式，仅 API）
poetry run python3 \
  gpustack/main.py start \
  --bootstrap-password='password' \
  --port=6080 \
  --worker-ip=127.0.0.1 \
  --worker-name=local-debug \
  --data-dir=${HOME}/gpustack \
  --tools-download-base-url=https://cache.ali.wodcloud.com/vscode \
  --disable-worker
```

**注意事项：**

1. **前端编译**：如果需要修改 UI，在 `gpustack/ui` 目录下运行编译命令，编译后的文件会自动复制到 `gpustack/ui/` 目录
   - `css/`、`js/`、`static/`、`index.html` 等都是编译产物，不应该提交到 git
2. **Worker 模式**：使用 `--disable-worker` 可以只启动 server，不启动 worker（适合纯 API 调试）
3. **数据目录**：首次启动会在 `${HOME}/gpustack` 创建数据库和缓存
4. **默认账号**：用户名 `admin`，密码 `password`

**验证服务启动：**

```bash
# 检查健康状态
curl http://127.0.0.1:6080/healthz

# 访问 Web UI（浏览器打开）
http://127.0.0.1:6080

# 访问 API 文档
http://127.0.0.1:6080/docs
```

**调试技巧：**

- 如果 `poetry install` 网络超时，可以按 `Ctrl+C` 中断后重试
- 可以使用 `poetry install -vvv` 查看详细安装日志
- 如果只需要调试核心功能，可以跳过某些可选依赖的安装
- 访问 API 文档：`http://127.0.0.1:6080/docs`
- 前端修改后需要重新运行 `pnpm build` 并重启服务

## build

### cuda

```bash
sudo rm -rf .venv dist gpustack/ui && \
docker pull registry.cn-qingdao.aliyuncs.com/wod/python:3.10-bookworm && \
docker run -it --rm \
  -v $PWD/:/go/src/github.com/open-beagle/gpustack \
  -w /go/src/github.com/open-beagle/gpustack \
  -e VERSION=v0.7.1 \
  -e POETRY_PYPI_MIRROR_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple/ \
  registry.cn-qingdao.aliyuncs.com/wod/python:3.10-bookworm \
  bash .beagle/build.sh

docker build \
  --build-arg BASE=registry.cn-qingdao.aliyuncs.com/wod/cuda:12.8.1-runtime-ubuntu22.04 \
  -t registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-cuda \
  -f .beagle/cuda.dockerfile \
  . && \
docker push registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-cuda
```

**注意：** 关于 vLLM 版本管理和多版本支持，请参考 [vLLM 版本管理文档](.beagle/vllm.md)

### corex

```bash
sudo rm -rf rm -rf .venv dist gpustack/ui && \
docker pull registry.cn-qingdao.aliyuncs.com/wod/python:3.10-bookworm && \
docker run -it --rm \
  -v $PWD/:/go/src/github.com/open-beagle/gpustack \
  -w /go/src/github.com/open-beagle/gpustack \
  -e VERSION=v0.7.1 \
  -e POETRY_PYPI_MIRROR_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple/ \
  registry.cn-qingdao.aliyuncs.com/wod/python:3.10-bookworm \
  bash .beagle/build.sh

docker build \
  --build-arg BASE=registry.cn-qingdao.aliyuncs.com/wod/corex:mr-bi150-4.3.0-x86-ubuntu20.04-py3.10-poc-llm-infer-v1.2.3 \
  -t registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-corex \
  -f .beagle/corex.dockerfile \
  . && \
docker push registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-corex
```

### cann

```bash
# cann
docker run -it --rm \
  -v $PWD/:/go/src/github.com/open-beagle/gpustack \
  -w /go/src/github.com/open-beagle/gpustack \
  -e VERSION=v0.7.1 \
  -e POETRY_PYPI_MIRROR_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple/ \
  registry.cn-qingdao.aliyuncs.com/wod/python:3.10-bookworm \
  bash .beagle/build.sh

docker run -it --rm \
  -v $PWD/:/go/src/github.com/open-beagle/gpustack \
  -w /go/src/github.com/open-beagle/gpustack \
  -e DEBIAN_FRONTEND=noninteractive \
  registry.cn-qingdao.aliyuncs.com/wod/cuda:12.8.1-runtime-ubuntu22.04 \
  bash

  apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    tzdata \
    python3 \
    python3-pip && \
  WHEEL_PACKAGE="$(ls /go/src/github.com/open-beagle/gpustack/dist/*.whl)[vllm]" && \
  pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
  pip3 install $WHEEL_PACKAGE

  gpustack download-tools --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'

docker pull registry.cn-qingdao.aliyuncs.com/wod/cann:8.0.rc3.beta1-910b-ubuntu22.04-py3.10 && \
docker run -it --rm \
  -v $PWD/:/go/src/github.com/open-beagle/gpustack \
  -w /go/src/github.com/open-beagle/gpustack \
  -e DEBIAN_FRONTEND=noninteractive \
  registry.cn-qingdao.aliyuncs.com/wod/cann:8.0.rc3.beta1-910b-ubuntu22.04-py3.10 \
  bash

  WHEEL_PACKAGE="$(ls /go/src/github.com/open-beagle/gpustack/dist/*.whl)" && \
  pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
  pip3 install --use-pep517 $WHEEL_PACKAGE

  gpustack download-tools \
    --arch arm64 --device npu \
    --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'

curl -sSL https://install.python-poetry.org | python3 -

poetry config repositories.pypi-mirror https://pypi.tuna.tsinghua.edu.cn/simple
poetry install

python3 \
  gpustack/main.py start \
  --bootstrap-password='beagle!@#123' \
  --port=6080 \
  --worker-ip=127.0.0.1 \
  --worker-name=WSL-Debian \
  --data-dir=${HOME}/gpustack \
  --tools-download-base-url=https://cache.ali.wodcloud.com/vscode

git apply .beagle/v0.7.1-logginglocal.patch
```

## tools

### aarch64 cann

```bash
# gpustack/worker/tools_manager.py
rm -rf ./downloads/gpustack/

# llama-box
# https://github.com/gpustack/llama-box/releases
export LLAMA_BOX_VERSION=v0.0.103 && \
mkdir -p ./downloads/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION} && \
curl -x $SOCKS5_PROXY_LOCAL \
  -o ./downloads/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION}/llama-box-linux-arm64-cann-8.0.zip \
  -fL https://github.com/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION}/llama-box-linux-arm64-cann-8.0.zip

# gguf-parser-go
export GGUF_PARSER_GO_VERSION=v0.13.8 && \
mkdir -p ./downloads/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION} && \
curl -x $SOCKS5_PROXY_LOCAL \
  -o ./downloads/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION}/gguf-parser-linux-arm64 \
  -fL https://github.com/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION}/gguf-parser-linux-arm64

# fastfetch
# https://github.com/gpustack/fastfetch/releases
export FASTFETCH_VERSION=2.25.0.1 && \
mkdir -p ./downloads/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION} && \
curl -x $SOCKS5_PROXY_LOCAL \
  -o ./downloads/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION}/fastfetch-linux-aarch64.zip \
  -fL https://github.com/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION}/fastfetch-linux-aarch64.zip

export FASTFETCH_VERSION=2.25.0.1 && \
mkdir -p ./downloads/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION} && \
curl -x $SOCKS5_PROXY_LOCAL \
  -o ./downloads/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION}/fastfetch-linux-aarch64.rpm \
  -fL https://github.com/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION}/fastfetch-linux-aarch64.rpm

mc cp -r ./downloads/gpustack/ aliyun/vscode/gpustack/
```

### amd64 cuda 12.4

```bash
# gpustack/worker/tools_manager.py
rm -rf ./downloads/gpustack/ && mkdir -p ./downloads/gpustack

# llama-box
# https://github.com/gpustack/llama-box/releases
export LLAMA_BOX_VERSION=v0.0.103 && \
mkdir -p ./downloads/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION} && \
curl -x $SOCKS5_PROXY_LOCAL \
  -o ./downloads/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION}/llama-box-linux-amd64-cuda-12.4.zip \
  -fL https://github.com/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION}/llama-box-linux-amd64-cuda-12.4.zip

# gguf-parser-go
# https://github.com/gpustack/gguf-parser-go
export GGUF_PARSER_GO_VERSION=v0.13.8 && \
mkdir -p ./downloads/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION} && \
curl -x $SOCKS5_PROXY_LOCAL \
  -o ./downloads/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION}/gguf-parser-linux-amd64 \
  -fL https://github.com/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION}/gguf-parser-linux-amd64

# fastfetch
# https://github.com/gpustack/fastfetch/releases
export FASTFETCH_VERSION=2.25.0.1 && \
mkdir -p ./downloads/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION} && \
curl -x $SOCKS5_PROXY_LOCAL \
  -o ./downloads/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION}/fastfetch-linux-amd64.zip \
  -fL https://github.com/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION}/fastfetch-linux-amd64.zip

export FASTFETCH_VERSION=2.25.0.1 && \
mkdir -p ./downloads/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION} && \
curl -x $SOCKS5_PROXY_LOCAL \
  -o ./downloads/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION}/fastfetch-linux-amd64.rpm \
  -fL https://github.com/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION}/fastfetch-linux-amd64.rpm

mc cp -r ./downloads/gpustack/ aliyun/vscode/gpustack/
```

## cache

```bash
# 构建缓存-->推送缓存至服务器
docker run --rm \
  -e PLUGIN_REBUILD=true \
  -e PLUGIN_ENDPOINT=${S3_ENDPOINT_ALIYUN} \
  -e PLUGIN_ACCESS_KEY=${S3_ACCESS_KEY_ALIYUN} \
  -e PLUGIN_SECRET_KEY=${S3_SECRET_KEY_ALIYUN} \
  -e DRONE_REPO_OWNER="open-beagle" \
  -e DRONE_REPO_NAME="gpustack" \
  -e PLUGIN_MOUNT="./.git,./.venv" \
  -v $(pwd):$(pwd) \
  -w $(pwd) \
  registry.cn-qingdao.aliyuncs.com/wod/devops-s3-cache:1.0

# 读取缓存-->将缓存从服务器拉取到本地
docker run --rm \
  -e PLUGIN_RESTORE=true \
  -e PLUGIN_ENDPOINT=${S3_ENDPOINT_ALIYUN} \
  -e PLUGIN_ACCESS_KEY=${S3_ACCESS_KEY_ALIYUN} \
  -e PLUGIN_SECRET_KEY=${S3_SECRET_KEY_ALIYUN} \
  -e DRONE_REPO_OWNER="open-beagle" \
  -e DRONE_REPO_NAME="gpustack" \
  -v $(pwd):$(pwd) \
  -w $(pwd) \
  registry.cn-qingdao.aliyuncs.com/wod/devops-s3-cache:1.0
```
