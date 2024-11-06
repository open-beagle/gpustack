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
  bash

rm -rf $PWD/.venv
python3 -m venv $PWD/.venv
source $PWD/.venv/bin/activate
git config --global --add safe.directory /go/src/github.com/open-beagle/gpustack
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
make build
```
