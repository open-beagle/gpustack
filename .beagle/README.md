# gpustack

<https://github.com/open-beagle/gpustack>

## git

```bash
git remote add upstream git@github.com:gpustack/gpustack.git

git fetch upstream

# 0.3.2
git merge upstream/main
```

## images

<https://hub.docker.com/r/gpustack/gpustack>

```bash
docker pull gpustack/gpustack:0.3.2 && \
docker tag gpustack/gpustack:0.3.2 registry.cn-qingdao.aliyuncs.com/wod/gpustack:0.3.2 && \
docker push registry.cn-qingdao.aliyuncs.com/wod/gpustack:0.3.2
```

## base images

```bash
# cuda
docker pull nvidia/cuda:12.5.1-runtime-ubuntu22.04 && \
docker tag nvidia/cuda:12.5.1-runtime-ubuntu22.04 registry.cn-qingdao.aliyuncs.com/wod/cuda:12.5.1-runtime-ubuntu22.04 && \
docker push registry.cn-qingdao.aliyuncs.com/wod/cuda:12.5.1-runtime-ubuntu22.04

# cann
docker pull ascendai/cann:ubuntu-python3.10-cann8.0.rc3.beta1 && \
docker tag ascendai/cann:ubuntu-python3.10-cann8.0.rc3.beta1 registry.cn-qingdao.aliyuncs.com/wod/cann:ubuntu-python3.10-cann8.0.rc3.beta1 && \
docker push registry.cn-qingdao.aliyuncs.com/wod/cann:ubuntu-python3.10-cann8.0.rc3.beta1
```

## deploy

```bash
# default user admin
docker run -d --gpus all -p 6080:80 --ipc=host --shm-size=2g --name gpustack \
  -v /data/gpustack:/var/lib/gpustack \
  registry.cn-qingdao.aliyuncs.com/wod/gpustack:0.3.2 \
  --bootstrap-password 'beagle!@#123' \
  --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'

docker rm -f gpustack && rm -rf /data/gpustack

# start worker node
docker run -d --gpus all --ipc=host --shm-size=2g --name gpustack \
  -p 10150:10150 -p 40000-41024:40000-41024 \
  -v /data/gpustack:/var/lib/gpustack \
  registry.cn-qingdao.aliyuncs.com/wod/gpustack:0.3.2 \
  --server-url http://myserver --token mytoken \
  --worker-ip <host-ip> \
  --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'
```

## deployNPU

```bash
# default user admin
docker run -d -p 6080:80 --privileged --ipc=host --shm-size=2g --name gpustack \
  -v /usr/share/hwdata:/usr/share/hwdata \
  -v /data/gpustack/data:/var/lib/gpustack \
  -e ASCEND_VISIBLE_DEVICES=0-7 \
  -e TZ=Asia/Shanghai \
  registry.cn-qingdao.aliyuncs.com/wod/gpustack:v0.3.3-cann \
  --bootstrap-password 'beagle!@#123' \
  --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'

docker rm -f gpustack && rm -rf /data/gpustack

# start worker node
docker run -d --ipc=host --shm-size=2g --name gpustack \
  -p 10150:10150 -p 40000-41024:40000-41024 \
  -v /usr/share/hwdata:/usr/share/hwdata \
  -v /data/gpustack/data:/var/lib/gpustack \
  -e ASCEND_VISIBLE_DEVICES=0-7 \
  registry.cn-qingdao.aliyuncs.com/wod/gpustack:v0.3.3-cann \
  --server-url http://myserver:6080 --token mytoken \
  --worker-ip <host-ip>  \
  --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'
```

## build

```bash
# cann
docker run -it --rm \
  -v $PWD/:/go/src/github.com/open-beagle/gpustack \
  -w /go/src/github.com/open-beagle/gpustack \
  -e VERSION=v0.3.3 \
  -e POETRY_PYPI_MIRROR_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple/ \
  registry.cn-qingdao.aliyuncs.com/wod/python:3.10-bookworm \
  bash .beagle/build.sh

docker run -it --rm \
  -v $PWD/:/go/src/github.com/open-beagle/gpustack \
  -w /go/src/github.com/open-beagle/gpustack \
  -e DEBIAN_FRONTEND=noninteractive \
  registry.cn-qingdao.aliyuncs.com/wod/cuda:12.5.1-runtime-ubuntu22.04 \
  bash

  apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    tzdata \
    python3 \
    python3-pip && \
  timedatectl set-timezone Asia/Shanghai && \
  WHEEL_PACKAGE="$(ls /go/src/github.com/open-beagle/gpustack/dist/*.whl)[vllm]" && \
  pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
  pip3 install $WHEEL_PACKAGE

  gpustack download-tools --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'
```

## tools

### aarch64 cann

```bash
rm -rf ./downloads/gpustack/

# llama-box
export LLAMA_BOX_VERSION=v0.0.79 && \
mkdir -p ./downloads/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION} && \
curl -x $SOCKS5_PROXY_LOCAL \
  -o ./downloads/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION}/llama-box-linux-arm64-cann-8.0.zip \
  -fL https://github.com/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION}/llama-box-linux-arm64-cann-8.0.zip

# gguf-parser-go
export GGUF_PARSER_GO_VERSION=v0.12.0 && \
mkdir -p ./downloads/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION} && \
curl -x $SOCKS5_PROXY_LOCAL \
  -o ./downloads/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION}/gguf-parser-linux-arm64 \
  -fL https://github.com/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION}/gguf-parser-linux-arm64

# fastfetch
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
rm -rf ./downloads/gpustack/ && mkdir -p ./downloads/gpustack

# llama-box
export LLAMA_BOX_VERSION=v0.0.79 && \
mkdir -p ./downloads/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION} && \
curl -x $SOCKS5_PROXY_LOCAL \
  -o ./downloads/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION}/llama-box-linux-amd64-cuda-12.4.zip \
  -fL https://github.com/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION}/llama-box-linux-amd64-cuda-12.4.zip

# gguf-parser-go
export GGUF_PARSER_GO_VERSION=v0.12.0 && \
mkdir -p ./downloads/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION} && \
curl -x $SOCKS5_PROXY_LOCAL \
  -o ./downloads/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION}/gguf-parser-linux-amd64 \
  -fL https://github.com/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION}/gguf-parser-linux-amd64

# fastfetch
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
