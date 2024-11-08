# gpustack

<https://github.com/open-beagle/gpustack>

## git

```bash
git remote add upstream git@github.com:gpustack/gpustack.git

git fetch upstream

git merge 0.3.2
```

## images

<https://hub.docker.com/r/gpustack/gpustack>

```bash
docker pull gpustack/gpustack:0.3.2 && \
docker tag gpustack/gpustack:0.3.2 registry.cn-qingdao.aliyuncs.com/wod/gpustack:0.3.2 && \
docker push registry.cn-qingdao.aliyuncs.com/wod/gpustack:0.3.2
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
  --worker-ip <host-ip> --server-url http://myserver --token mytoken
```

## deployNPU

```bash
# default user admin
docker run -d -p 6080:80 --privileged --ipc=host --shm-size=2g --name gpustack \
  -v /usr/share/hwdata:/usr/share/hwdata \
  -v /data/gpustack/data:/var/lib/gpustack \
  -e ASCEND_VISIBLE_DEVICES=0-7 \
  -e TZ=Asia/Shanghai \
  registry.cn-qingdao.aliyuncs.com/wod/gpustack:v0.3.2-cann \
  --bootstrap-password 'beagle!@#123' \
  --tools-download-base-url 'https://cache.ali.wodcloud.com/vscode'

docker rm -f gpustack && rm -rf /data/gpustack

# start worker node
docker run -d --gpus all --ipc=host --shm-size=2g --name gpustack \
  -p 10150:10150 -p 40000-41024:40000-41024 \
  -v /usr/share/hwdata:/usr/share/hwdata \
  -v /data/gpustack/data:/var/lib/gpustack \
  registry.cn-qingdao.aliyuncs.com/wod/gpustack:v0.3.2-cann \
  --worker-ip <host-ip> --server-url http://myserver --token mytoken
```

## build

```bash
# cann
docker run -it --rm \
  -v $PWD/:/go/src/github.com/open-beagle/gpustack \
  -w /go/src/github.com/open-beagle/gpustack \
  -e VERSION=v0.3.2 \
  -e POETRY_PYPI_MIRROR_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple/ \
  registry.cn-qingdao.aliyuncs.com/wod/python:3.10-bookworm \
  bash .beagle/build.sh
```

## tools

```bash
# llama-box
export LLAMA_BOX_VERSION=v0.0.73 && \
mkdir -p ./downloads/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION} && \
curl -x socks5://www.ali.wodcloud.com:1283 \
  -o ./downloads/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION}/llama-box-linux-arm64-cann-8.0.zip \
  -fL https://github.com/gpustack/llama-box/releases/download/${LLAMA_BOX_VERSION}/llama-box-linux-arm64-cann-8.0.zip

# gguf-parser-go
export GGUF_PARSER_GO_VERSION=v0.12.0 && \
mkdir -p ./downloads/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION} && \
curl -x socks5://www.ali.wodcloud.com:1283 \
  -o ./downloads/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION}/gguf-parser-linux-arm64 \
  -fL https://github.com/gpustack/gguf-parser-go/releases/download/${GGUF_PARSER_GO_VERSION}/gguf-parser-linux-arm64

# fastfetch
export FASTFETCH_VERSION=2.25.0.1 && \
mkdir -p ./downloads/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION} && \
curl -x socks5://www.ali.wodcloud.com:1283 \
  -o ./downloads/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION}/fastfetch-linux-aarch64.zip \
  -fL https://github.com/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION}/fastfetch-linux-aarch64.zip

export FASTFETCH_VERSION=2.25.0.1 && \
mkdir -p ./downloads/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION} && \
curl -x socks5://www.ali.wodcloud.com:1283 \
  -o ./downloads/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION}/fastfetch-linux-aarch64.rpm \
  -fL https://github.com/gpustack/fastfetch/releases/download/${FASTFETCH_VERSION}/fastfetch-linux-aarch64.rpm

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
