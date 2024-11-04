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
