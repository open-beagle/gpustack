# vLLM 版本管理

## 当前版本

GPUStack v0.7.1 默认使用 **vLLM 0.11.2**

## 版本兼容性

vLLM 0.11.2 带来了许多新特性和性能改进，包括更好的模型支持、KV cache offloading、异步调度优化等。

### vLLM 0.15.0 + Transformers v5

vLLM 0.15.0 虽然默认不依赖 transformers v5，但可以通过 `--pip-args` 强制安装 transformers v5 来支持最新模型（如 GLM-4.7 Flash）。这种组合可以让你使用最新的模型架构，同时享受 vLLM 0.15.0 的性能优化。

## 解决方案

### 方案 1：使用 Backend Version 功能（推荐）

GPUStack 支持为每个模型指定独立的后端版本，会自动使用 pipx 安装到隔离环境。

#### 通过 UI 部署

1. 创建模型时，点击 **"Advanced"** 展开高级设置
2. 在 **"Backend Version"** 字段填写版本号（如 `0.11.0`）
3. 点击部署

#### 通过 CLI 部署

```bash
gpustack models create \
  --name qwen-2.5-7b \
  --backend vllm \
  --backend-version 0.12.0 \
  --huggingface-repo-id Qwen/Qwen2.5-7B-Instruct
```

#### 通过 API 部署

```bash
curl -X POST http://localhost/v1/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qwen-2.5-7b",
    "backend": "vllm",
    "backend_version": "0.12.0",
    "huggingface_repo_id": "Qwen/Qwen2.5-7B-Instruct"
  }'
```

#### 优点

- ✅ **灵活**：不同模型可以使用不同的 vLLM 版本
- ✅ **隔离**：使用 pipx 虚拟环境，互不影响
- ✅ **方便**：无需重新构建镜像
- ✅ **持久**：配置保存在数据库中，容器重启后依然有效

#### 工作原理

1. GPUStack 检测到 `backend_version` 参数
2. 使用 `pipx` 在隔离环境中安装指定版本：

   ```bash
   pipx install vllm==0.12.0
   ```

3. 模型实例启动时使用对应版本的 vLLM
4. 不同版本的 vLLM 安装在不同的虚拟环境中，互不干扰

#### 存储位置

- **自定义版本**：`/var/lib/gpustack/pipx/venvs/vllm-<version>/`
- **默认版本**：系统 Python 环境（`pip list | grep vllm`）

#### 示例场景

```bash
# 模型 A 使用 vLLM 0.11.2（默认版本）
gpustack models create \
  --name model-a \
  --backend vllm \
  --huggingface-repo-id meta-llama/Llama-3.1-8B-Instruct

# 模型 B 使用 vLLM 0.15.0（最新版本）
gpustack models create \
  --name model-b \
  --backend vllm \
  --backend-version 0.15.0 \
  --huggingface-repo-id Qwen/Qwen2.5-7B-Instruct

# 模型 C 使用 vLLM 0.12.0（自定义版本）
gpustack models create \
  --name model-c \
  --backend vllm \
  --backend-version 0.12.0 \
  --huggingface-repo-id deepseek-ai/DeepSeek-V3
```

三个模型可以同时运行，使用各自的 vLLM 版本！

#### 使用 vLLM 0.15.0 + Transformers v5（支持 GLM-4.7 Flash 等新模型）

如果你需要运行 GLM-4.7 Flash 等需要 transformers v5 的模型，可以手动安装带有 transformers v5 的 vLLM 0.15.0：

```bash
# 进入容器
docker exec -it <container-name> bash

# 手动安装 vLLM 0.15.0 + transformers v5
pipx install --force \
  --suffix _v0.15.0 \
  --pip-args='--index-url https://mirrors.aliyun.com/pypi/simple/ transformers>=5.0.0 torch>=2.5.0' \
  vllm==0.15.0
```

然后在部署模型时指定 `backend-version` 为 `0.15.0`：

```bash
gpustack models create \
  --name glm-4-flash \
  --backend vllm \
  --backend-version 0.15.0 \
  --huggingface-repo-id THUDM/glm-4-9b-chat
```

详细步骤请参考：[手动安装 vLLM 指南](../docs/manual-vllm-installation.md)

---

### 方案 2：构建特定版本镜像

如果你希望所有模型默认使用特定版本的 vLLM，可以修改 `pyproject.toml` 后重新构建镜像。

#### 步骤

1. 修改 `pyproject.toml`：

   ```toml
   vllm = {version = "0.12.0", optional = true}
   ```

2. 重新构建：

   ```bash
   sudo rm -rf .venv dist gpustack/ui
   docker run -it --rm \
     -v $PWD/:/go/src/github.com/open-beagle/gpustack \
     -w /go/src/github.com/open-beagle/gpustack \
     -e VERSION=v0.7.1 \
     -e POETRY_PYPI_MIRROR_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple/ \
     registry.cn-qingdao.aliyuncs.com/wod/python:3.10-bookworm \
     bash .beagle/build.sh

   docker build \
     --build-arg BASE=registry.cn-qingdao.aliyuncs.com/wod/cuda:12.5.1-runtime-ubuntu22.04 \
     -t registry.cn-qingdao.aliyuncs.com/wod/windstack:v0.7.1-cuda-vllm0.12 \
     -f .beagle/cuda.dockerfile \
     .
   ```

#### 优点

- ✅ 所有模型默认使用指定版本
- ✅ 无需每次部署时指定版本

#### 缺点

- ❌ 需要重新构建镜像
- ❌ 所有模型必须使用同一版本
- ❌ 版本升级需要重新构建

---

### 方案 3：运行时升级（不推荐）

临时升级容器中的 vLLM 版本，**容器重启后失效**。

```bash
docker exec -it <container-name> pip install vllm==0.12.0 --force-reinstall
```

#### 缺点

- ❌ 容器重启后失效
- ❌ 可能破坏依赖关系
- ❌ 不适合生产环境

---

## 常见问题

### Q: 如何查看当前使用的 vLLM 版本？

**默认版本：**

```bash
docker exec -it <container> pip show vllm
```

**自定义版本：**

```bash
docker exec -it <container> ls -la /var/lib/gpustack/pipx/venvs/
```

### Q: Backend Version 支持哪些后端？

- ✅ vLLM
- ✅ llama-box
- ✅ vox-box
- ✅ Ascend MindIE

### Q: 自定义版本会占用多少空间？

每个 vLLM 版本大约占用 **2-3 GB** 空间（包括依赖）。

### Q: 如何删除不用的自定义版本？

```bash
docker exec -it <container> pipx uninstall vllm-0.12.0
# 或手动删除
docker exec -it <container> rm -rf /var/lib/gpustack/pipx/venvs/vllm-0.12.0
```

### Q: 分布式推理支持自定义版本吗？

**不支持**。使用自定义 backend version 的模型不能跨多个 worker 分布式部署。

如果需要分布式推理，请使用默认版本或重新构建所有 worker 的镜像。

---

## 推荐实践

1. **优先使用方案 1（Backend Version）**：灵活且不需要重新构建镜像
2. **生产环境建议方案 2**：如果所有模型都需要同一版本
3. **避免方案 3**：仅用于临时测试

## 参考链接

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM Release Notes](https://github.com/vllm-project/vllm/releases)
- [GPUStack 后端文档](https://docs.gpustack.ai/latest/user-guide/inference-backends/)
