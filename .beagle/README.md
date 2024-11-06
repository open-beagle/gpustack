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
  --bootstrap-password 'beagle!@#123'

docker rm -f gpustack && rm -rf /data/gpustack

# start worker node
docker run -d --gpus all --ipc=host --shm-size=2g --name gpustack \
  -p 10150:10150 -p 40000-41024:40000-41024 \
  -v /data/gpustack:/var/lib/gpustack \
  registry.cn-qingdao.aliyuncs.com/wod/gpustack:0.3.2 \
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
