# 手动在容器内安装指定版本 vLLM 指南

如果你希望在 GPUStack 自动调度前手动准备好特定的 vLLM 版本（例如为了预下载依赖，或在离线环境下安装），可以按照以下步骤操作。

## 1. 进入容器

首先，找到运行中的 gpustack 容器 ID 或名称，然后进入容器终端：

```bash
# 获取容器 ID
docker ps | grep gpustack

# 进入容器 (假设容器名为 gpustack)
docker exec -it gpustack bash
```

## 2. 确认环境变量

GPUStack 的容器默认已经设置了 `pipx` 相关的环境变量。为了保险起见，可以在执行前确认或手动导出一下：

```bash
export PIPX_HOME=/var/lib/gpustack/pipx
export PIPX_BIN_DIR=/var/lib/gpustack/bin
# 确保目录存在
mkdir -p $PIPX_HOME $PIPX_BIN_DIR
```

## 3. 执行安装命令

使用 `pipx` 安装指定版本的 vLLM。

**关键参数说明：**

- `--suffix _<版本号>`: **必须**。GPUStack 根据此后缀识别多版本（例如 `vllm_0.14.1`）。
- `--pip-args`: 可选。用于指定该环境中需要覆盖的依赖（如 `transformers`）。

### 示例：安装 vLLM v0.14.1 并搭配最新 Transformers

```bash
# 安装 vLLM 0.14.1，同时强制安装较新的 transformers 以支持 GLM-4.7 等新模型
pipx install --force \
    --suffix _0.14.1 \
    --pip-args='transformers>=4.49.0 torch>=2.5.0' \
    vllm==0.14.1

pipx install --force \
    --suffix _0.14.1 \
    vllm==0.14.1
```

_(注意：请根据 vLLM 官方文档选择兼容的 torch 版本。如果不指定 pip-args，pipx 会安装 vllm 默认依赖)_

### 示例：完全离线安装（如果有 whl 包）

如果你已经把 `.whl` 文件挂载到了容器里（例如 `/tmp/packages`）：

```bash
pipx install --force \
    --suffix _0.14.1 \
    --pip-args='--no-index --find-links=/tmp/packages' \
    vllm==0.14.1
```

## 4. 验证安装

安装完成后，检查 `bin` 目录下是否生成了对应的可执行文件：

```bash
ls -l /var/lib/gpustack/bin/vllm_0.14.1
```

如果文件存在，说明安装成功。

## 5. 在 GPUStack UI 中使用

回到 GPUStack 网页控制台：

1. 编辑或部署你的模型。
2. 在 **"Backend Version / 后端版本"** 输入框中，准确填入你刚才安装的版本号：`0.14.1`。
3. 保存/部署。

GPUStack 会检测到本地 `bin` 目录下已经存在 `vllm_0.14.1`，因此**不会**再次触发下载，而是直接使用你手动准备好的环境运行模型。
