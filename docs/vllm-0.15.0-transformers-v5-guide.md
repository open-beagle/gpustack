# vLLM 0.15.0 + Transformers v5 快速指南

本指南介绍如何在 GPUStack 中使用 vLLM 0.15.0 配合 transformers v5 来运行需要最新 transformers 版本的模型（如 GLM-4.7 Flash）。

## 背景

- vLLM 0.15.0 官方默认不依赖 transformers v5
- 某些新模型（如 GLM-4.7 Flash）需要 transformers v5 才能正常运行
- 通过 `pipx` 的 `--pip-args` 参数，可以强制安装 transformers v5

## 快速开始

### 1. 进入容器

```bash
# 查看运行中的容器
docker ps | grep gpustack

# 进入容器（假设容器名为 gpustack）
docker exec -it gpustack bash
```

### 2. 安装 vLLM 0.15.0 + Transformers v5

```bash
# 设置环境变量（通常容器已设置，保险起见再执行一次）
export PIPX_HOME=/var/lib/gpustack/pipx
export PIPX_BIN_DIR=/var/lib/gpustack/bin
mkdir -p $PIPX_HOME $PIPX_BIN_DIR

# 安装 vLLM 0.15.0 + transformers v5
pipx install --force \
  --suffix _v0.15.0 \
  --pip-args='--index-url https://mirrors.aliyun.com/pypi/simple/ transformers>=5.0.0 torch>=2.5.0' \
  vllm==0.15.0
```

### 3. 验证安装

```bash
# 检查可执行文件是否存在
ls -l /var/lib/gpustack/bin/vllm_v0.15.0

# 检查 transformers 版本
/var/lib/gpustack/pipx/venvs/vllm-v0.15.0/bin/python -c "import transformers; print(transformers.__version__)"
```

### 4. 部署模型

#### 通过 UI 部署

1. 在 GPUStack 控制台创建或编辑模型
2. 展开 **"Advanced"** 高级设置
3. 在 **"Backend Version"** 字段填入：`0.15.0`
4. 选择你的模型（如 `THUDM/glm-4-9b-chat`）
5. 点击部署

#### 通过 CLI 部署

```bash
gpustack models create \
  --name glm-4-flash \
  --backend vllm \
  --backend-version 0.15.0 \
  --huggingface-repo-id THUDM/glm-4-9b-chat
```

#### 通过 API 部署

```bash
curl -X POST http://localhost/v1/models \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "name": "glm-4-flash",
    "backend": "vllm",
    "backend_version": "0.15.0",
    "huggingface_repo_id": "THUDM/glm-4-9b-chat"
  }'
```

## 版本对照表

| vLLM 版本 | Transformers 版本 | 适用场景 |
|-----------|------------------|---------|
| 0.15.0 | >= 5.0.0 | GLM-4.7 Flash 等最新模型 |
| 0.14.1 | >= 4.49.0 | 较新的模型，需要 transformers 4.x |
| 0.11.2 | 默认版本 | GPUStack 默认配置 |

## 常见问题

### Q: 为什么不直接在 vLLM 中集成 transformers v5？

vLLM 0.15.0 发布时，transformers v5 刚刚发布，vLLM 官方还在测试兼容性。通过 `--pip-args` 强制安装可以让你提前使用新特性。

### Q: 这样安装会影响其他模型吗？

不会。使用 `pipx` 安装的每个版本都在独立的虚拟环境中，互不影响。你可以同时运行使用不同 vLLM 版本的多个模型。

### Q: 容器重启后需要重新安装吗？

不需要。只要你的 `/var/lib/gpustack` 目录是持久化挂载的（通常是），安装的 vLLM 版本会一直保留。

### Q: 如何卸载不需要的版本？

```bash
# 方法 1：使用 pipx 卸载
docker exec -it <container> pipx uninstall vllm-v0.15.0

# 方法 2：直接删除目录
docker exec -it <container> rm -rf /var/lib/gpustack/pipx/venvs/vllm-v0.15.0
docker exec -it <container> rm -f /var/lib/gpustack/bin/vllm_v0.15.0
```

### Q: 支持离线安装吗？

支持。如果你已经下载了 `.whl` 文件，可以使用：

```bash
pipx install --force \
  --suffix _v0.15.0 \
  --pip-args='--no-index --find-links=/path/to/wheels transformers>=5.0.0 torch>=2.5.0' \
  vllm==0.15.0
```

## 参考链接

- [vLLM 0.15.0 Release Notes](https://github.com/vllm-project/vllm/releases/tag/v0.15.0)
- [Transformers v5.0.0 Release Notes](https://github.com/huggingface/transformers/releases/tag/v5.0.0)
- [GPUStack 手动安装 vLLM 指南](./manual-vllm-installation.md)
- [GPUStack 后端版本管理](./.beagle/vllm.md)

## 故障排除

### 安装失败：找不到 transformers v5

确保使用了正确的镜像源，或者尝试使用官方 PyPI：

```bash
pipx install --force \
  --suffix _v0.15.0 \
  --pip-args='transformers>=5.0.0 torch>=2.5.0' \
  vllm==0.15.0
```

### 模型启动失败：transformers 版本不兼容

检查安装的 transformers 版本：

```bash
/var/lib/gpustack/pipx/venvs/vllm-v0.15.0/bin/python -c "import transformers; print(transformers.__version__)"
```

如果版本不对，重新安装：

```bash
pipx uninstall vllm-v0.15.0
# 然后重新执行安装命令
```

### 模型运行时报错

查看详细日志：

```bash
docker logs -f <container-name>
```

或在 GPUStack UI 中查看模型实例的日志。
