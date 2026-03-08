# vLLM 0.17.0 快速指南

本指南介绍如何在 GPUStack 中使用 vLLM 0.17.0，该版本带来了重大更新和新模型支持。

## 快速摘要

### ✅ 兼容性确认

- **当前镜像**：`windstack:v0.7.1-cuda` 使用 CUDA 12.8.1
- **vLLM 0.17.0 要求**：CUDA 12.1+（基于 PyTorch 2.10）
- **结论**：✅ 完全兼容，无需升级镜像

### 🆕 主要新特性

1. **PyTorch 2.10 升级**：性能和稳定性提升
2. **FlashAttention 4**：下一代注意力优化
3. **Transformers v5 支持**：兼容最新 HuggingFace 模型
4. **性能模式标志**：`--performance-mode {balanced, interactivity, throughput}`

### 📦 新支持的模型

- **Qwen3.5 系列**：完整支持，包括 GDN 架构、FP8 量化
- **语音识别**：FunASR、FireRedASR2、Qwen3-ASR
- **NVIDIA Nemotron**：embed-vl、rerank-vl、colembed
- **多模态**：Ovis 2.6、OpenPangu-VL、MiniCPM-o

## vLLM 0.17.0 新特性

### 重大更新

- **PyTorch 2.10 升级**：升级到 PyTorch 2.10.0，这是一个破坏性变更
- **FlashAttention 4 集成**：支持下一代注意力性能优化
- **Transformers v5 兼容**：完整支持 HuggingFace Transformers v5
- **性能模式标志**：新增 `--performance-mode {balanced, interactivity, throughput}` 简化性能调优

### 新支持的模型架构

#### 文本和多模态模型

- **Qwen3.5 系列**：完整支持 Qwen3.5 模型家族，包括 GDN (Gated Delta Networks)、FP8 量化、MTP 推测解码
- **COLQwen3**：ColBERT 架构的 Qwen3 变体
- **Ring 2.5**：新的模型架构
- **skt/A.X-K1**：韩国 SK Telecom 的模型
- **Ovis 2.6**：多模态视觉语言模型
- **ColModernVBERT**：现代化的 BERT 变体

#### NVIDIA Nemotron 系列

- **nvidia/llama-nemotron-embed-vl-1b-v2**：视觉语言嵌入模型
- **nvidia/llama-nemotron-rerank-vl-1b-v2**：视觉语言重排序模型
- **nvidia/nemotron-colembed**：ColBERT 嵌入模型

#### 语音识别模型 (ASR)

- **FunASR**：阿里达摩院的语音识别模型
- **FireRedASR2**：新一代 ASR 模型
- **Qwen3-ASR**：支持实时流式语音识别

#### 多模态增强

- **OpenPangu-VL**：支持视频输入
- **MiniCPM-o**：支持 flagos 模式
- **Parakeet**：用于 nemotron-nano-vl 的音频编码器

### CUDA 版本要求

#### 最低要求

- **CUDA 12.1+**：PyTorch 2.10 需要 CUDA 12.1 或更高版本
- **推荐版本**：CUDA 12.4 或 12.6

#### CUDA 12.9+ 已知问题

如果使用 CUDA 12.9+，可能遇到 `CUBLAS_STATUS_INVALID_VALUE` 错误，这是由 CUDA 库不匹配导致的。解决方法：

1. 从 `LD_LIBRARY_PATH` 中移除系统 CUDA 共享库路径（如 `/usr/local/cuda`），或直接 `unset LD_LIBRARY_PATH`
2. 使用 `uv pip install vllm --torch-backend=auto` 安装
3. 使用 `pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129` 安装（匹配系统 CUDA 版本）

#### 当前镜像兼容性

✅ **当前 windstack CUDA 镜像使用 CUDA 12.8.1，完全兼容 vLLM 0.17.0**

- 基础镜像：`registry.cn-qingdao.aliyuncs.com/wod/cuda:12.8.1-runtime-ubuntu22.04`
- CUDA 12.8.1 满足 PyTorch 2.10 的最低要求（CUDA 12.1+）
- 不受 CUDA 12.9+ 已知问题影响

## 背景

Qwen3.5 是阿里云推出的新一代多模态视觉语言模型（VLM），采用了新的 `qwen3vl_merger` projector 架构。

### 问题现象

使用 llama-box 后端部署 Qwen3.5 VL 模型时，会遇到以下错误：

```
E clip_init: failed to load model '/var/lib/gpustack/cache/model_scope/unsloth/Qwen3.5-9B-GGUF/mmproj-F32.gguf':
  load_hparams: unknown projector type: qwen3vl_merger
E srv load: failed to load multimodal project model
```

### 原因分析

- llama-box v0.0.171 不支持 Qwen3.5 VL 的 `qwen3vl_merger` projector 类型
- Qwen3.5 VL 使用了新的多模态架构，需要更新的推理引擎支持
- vLLM 对 Qwen3.5 VL 有更好的支持

## 解决方案

### 方案 1：使用 vLLM 后端（推荐）

vLLM 原生支持 Qwen3.5 VL 模型，无需额外配置。

#### 通过 UI 部署

1. 在 GPUStack 控制台创建模型
2. 选择 Qwen3.5 模型（如 `unsloth/Qwen3.5-9B-GGUF`）
3. 在 **"Backend"** 下拉菜单中选择 **"vLLM"**
4. 点击部署

#### 通过 CLI 部署

```bash
gpustack models create \
  --name qwen3.5-9b \
  --backend vllm \
  --huggingface-repo-id unsloth/Qwen3.5-9B-GGUF
```

#### 通过 API 部署

```bash
curl -X POST http://localhost/v1/models \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "name": "qwen3.5-9b",
    "backend": "vllm",
    "huggingface_repo_id": "unsloth/Qwen3.5-9B-GGUF"
  }'
```

### 方案 2：使用更新版本的 vLLM

如果默认的 vLLM 版本不支持，可以安装更新版本的 vLLM。

#### 1. 进入容器

```bash
# 查看运行中的容器
docker ps | grep windstack

# 进入容器
docker exec -it <container-name> bash
```

#### 2. 安装更新版本的 vLLM

```bash
# 设置环境变量（容器通常已设置）
export PIPX_HOME=/var/lib/gpustack/pipx
export PIPX_BIN_DIR=/var/lib/gpustack/bin
mkdir -p $PIPX_HOME $PIPX_BIN_DIR

# 安装 vLLM 0.17.0（最新版本，完整支持 Qwen3.5 VL）
pipx install --force \
  --suffix _v0.17.0 \
  --pip-args='--index-url https://mirrors.aliyun.com/pypi/simple/' \
  vllm==0.17.0
```

#### 3. 验证安装

```bash
# 检查可执行文件
ls -l /var/lib/gpustack/bin/vllm_v0.17.0

# 检查版本
/var/lib/gpustack/pipx/venvs/vllm-v0.17.0/bin/python -c "import vllm; print(vllm.__version__)"
```

#### 4. 部署模型时指定版本

```bash
gpustack models create \
  --name qwen3.5-9b \
  --backend vllm \
  --backend-version 0.17.0 \
  --huggingface-repo-id unsloth/Qwen3.5-9B-GGUF
```

或在 UI 中：

1. 创建模型时展开 **"Advanced"** 高级设置
2. 在 **"Backend Version"** 字段填入：`0.17.0`
3. 点击部署

### 方案 3：等待 llama-box 更新

llama-box 团队可能会在未来版本中添加对 `qwen3vl_merger` 的支持。

关注更新：

- [llama-box Releases](https://github.com/gpustack/llama-box/releases)
- [llama.cpp Qwen3-VL Support Issue](https://github.com/ggml-org/llama.cpp/issues/16207)

一旦 llama-box 支持 Qwen3.5 VL，可以通过 Backend Version 功能使用新版本：

```bash
gpustack models create \
  --name qwen3.5-9b \
  --backend llama-box \
  --backend-version v0.0.180 \  # 假设未来版本支持
  --huggingface-repo-id unsloth/Qwen3.5-9B-GGUF
```

### 方案 4：使用纯文本版本

如果不需要视觉功能，可以使用 Qwen3.5 的纯文本版本（不带 VL）：

```bash
gpustack models create \
  --name qwen3.5-9b-text \
  --backend llama-box \
  --huggingface-repo-id Qwen/Qwen3.5-9B-Instruct
```

## 支持的模型列表

### Qwen 系列

| 模型                | 类型 | 推荐后端 | 说明                              |
| ------------------- | ---- | -------- | --------------------------------- |
| Qwen3.5-9B-GGUF     | VLM  | vLLM     | 多模态视觉语言模型，支持 GDN 架构 |
| Qwen3.5-9B-Instruct | Text | vLLM     | 纯文本模型，支持 FP8 量化         |
| Qwen3.5-35B-A3B     | VLM  | vLLM     | 大规模多模态模型                  |
| Qwen3-VL-30B-A3B    | VLM  | vLLM     | Qwen3 VL 系列                     |
| Qwen3-ASR           | ASR  | vLLM     | 实时流式语音识别                  |
| COLQwen3            | Text | vLLM     | ColBERT 架构的 Qwen3 变体         |

### NVIDIA Nemotron 系列

| 模型                           | 类型   | 说明               |
| ------------------------------ | ------ | ------------------ |
| llama-nemotron-embed-vl-1b-v2  | Embed  | 视觉语言嵌入模型   |
| llama-nemotron-rerank-vl-1b-v2 | Rerank | 视觉语言重排序模型 |
| nemotron-colembed              | Embed  | ColBERT 嵌入模型   |

### 语音识别模型

| 模型        | 类型 | 说明               |
| ----------- | ---- | ------------------ |
| FunASR      | ASR  | 阿里达摩院语音识别 |
| FireRedASR2 | ASR  | 新一代 ASR 模型    |
| Qwen3-ASR   | ASR  | 实时流式语音识别   |

### 其他多模态模型

| 模型           | 类型 | 说明               |
| -------------- | ---- | ------------------ |
| Ovis 2.6       | VLM  | 多模态视觉语言模型 |
| OpenPangu-VL   | VLM  | 支持视频输入       |
| MiniCPM-o      | VLM  | 支持 flagos 模式   |
| Ring 2.5       | Text | 新的模型架构       |
| skt/A.X-K1     | Text | SK Telecom 模型    |
| ColModernVBERT | Text | 现代化的 BERT 变体 |

## 常见问题

### Q: vLLM 0.17.0 有哪些性能提升？

- **FlashAttention 4**：下一代注意力机制，显著提升推理速度
- **Pipeline Parallel 优化**：异步发送/接收，吞吐量提升 2.9%
- **权重卸载 V2**：通过预取隐藏加载延迟
- **性能模式**：使用 `--performance-mode` 快速调优

### Q: 如何使用性能模式？

```bash
# 平衡模式（默认）
gpustack models create --backend vllm --backend-version 0.17.0 \
  --backend-parameters='{"performance_mode": "balanced"}'

# 交互模式（低延迟）
gpustack models create --backend vllm --backend-version 0.17.0 \
  --backend-parameters='{"performance_mode": "interactivity"}'

# 吞吐量模式（高吞吐）
gpustack models create --backend vllm --backend-version 0.17.0 \
  --backend-parameters='{"performance_mode": "throughput"}'
```

### Q: CUDA 版本不兼容怎么办？

如果遇到 CUDA 版本问题：

1. 检查当前 CUDA 版本：`nvcc --version` 或 `nvidia-smi`
2. 确保 CUDA >= 12.1
3. 如果使用 CUDA 12.9+，参考上面的已知问题解决方法

### Q: 为什么 llama-box 不支持 Qwen3.5 VL？

llama-box 基于 llama.cpp，而 llama.cpp 目前还不支持 Qwen3.5 VL 的新架构。这需要底层 CLIP 模型加载器的更新。

### Q: vLLM 和 llama-box 有什么区别？

- **vLLM**：基于 PyTorch，支持更多模型架构，性能优化好，但内存占用较大
- **llama-box**：基于 llama.cpp，支持 GGUF 量化格式，内存占用小，但支持的模型架构较少

### Q: 使用 vLLM 后端会影响性能吗？

不会。vLLM 针对推理性能做了大量优化（PagedAttention、连续批处理等），在多数场景下性能优于 llama-box。

### Q: 容器重启后需要重新安装 vLLM 版本吗？

不需要。只要 `/var/lib/gpustack` 目录是持久化挂载的，安装的 vLLM 版本会一直保留。

### Q: 如何查看当前使用的后端版本？

```bash
# 查看默认 vLLM 版本
docker exec -it <container> pip show vllm

# 查看所有已安装的自定义版本
docker exec -it <container> ls -la /var/lib/gpustack/pipx/venvs/
```

### Q: 分布式推理支持自定义 vLLM 版本吗？

不支持。使用自定义 backend version 的模型不能跨多个 worker 分布式部署。如果需要分布式推理，请使用默认版本。

### Q: 如何卸载不需要的 vLLM 版本？

```bash
# 使用 pipx 卸载
docker exec -it <container> pipx uninstall vllm-v0.17.0

# 或直接删除目录
docker exec -it <container> rm -rf /var/lib/gpustack/pipx/venvs/vllm-v0.17.0
docker exec -it <container> rm -f /var/lib/gpustack/bin/vllm_v0.17.0
```

## 参考链接

- [vLLM 0.17.0 Release Notes](https://github.com/vllm-project/vllm/releases/tag/v0.17.0)
- [Qwen3.5 官方文档](https://github.com/QwenLM/Qwen3)
- [vLLM 支持的模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [GPUStack vLLM 后端文档](https://docs.gpustack.ai/latest/user-guide/inference-backends/)
- [GPUStack 后端版本管理](./.beagle/vllm.md)
- [PyTorch 2.10 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v2.10.0)
- [llama.cpp Qwen3-VL Support Issue](https://github.com/ggml-org/llama.cpp/issues/16207)

## 故障排除

### CUDA 版本相关

#### CUBLAS_STATUS_INVALID_VALUE 错误（CUDA 12.9+）

如果遇到此错误：

```bash
# 方法 1：移除 LD_LIBRARY_PATH
unset LD_LIBRARY_PATH

# 方法 2：重新安装 vLLM
pipx install --force \
  --suffix _v0.17.0 \
  --pip-args='--extra-index-url https://download.pytorch.org/whl/cu129' \
  vllm==0.17.0
```

#### CUDA 版本过低

如果 CUDA < 12.1，需要升级 CUDA 驱动或使用旧版本 vLLM。

### 模型加载失败：unknown projector type

这是 llama-box 不支持 Qwen3.5 VL 的典型错误。解决方法：

1. 切换到 vLLM 后端
2. 或使用纯文本版本的 Qwen3.5

### vLLM 启动失败：CUDA out of memory

Qwen3.5 VL 是多模态模型，内存占用较大。尝试：

1. 使用更小的模型（如 Qwen3.5-9B 而不是 35B）
2. 启用 GPU 内存分片
3. 减少并发请求数

### 模型推理速度慢

检查：

1. GPU 是否被充分利用（`nvidia-smi`）
2. 是否启用了 KV cache
3. 批处理大小是否合适

查看详细日志：

```bash
docker logs -f <container-name>
```

或在 GPUStack UI 中查看模型实例的日志。
