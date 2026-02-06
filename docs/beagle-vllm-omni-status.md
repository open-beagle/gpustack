# vLLM-Omni 后端实现状态

## 实现状态：✅ 已完成

vLLM-Omni 后端已经**完全实现并集成**到 GPUStack 中。

## 已实现的功能

### 1. 核心代码 ✅

| 组件     | 文件路径                                | 状态                   |
| -------- | --------------------------------------- | ---------------------- |
| 后端实现 | `gpustack/worker/backends/vllm_omni.py` | ✅ 已实现              |
| 枚举定义 | `gpustack/schemas/models.py`            | ✅ 已添加 `VLLM_OMNI`  |
| 服务管理 | `gpustack/worker/serve_manager.py`      | ✅ 已集成              |
| 模型目录 | `gpustack/assets/model-catalog.yaml`    | ✅ 已添加 Z-Image 模型 |

### 2. 功能特性 ✅

- ✅ 自动检测模型类型（diffusion/audio/llm）
- ✅ 支持 Diffusion 图像生成模型
- ✅ 默认启用缓存加速（TeaCache）
- ✅ 支持用户自定义参数
- ✅ 健康检查端点 `/health`
- ✅ 完整的错误处理和日志记录

### 3. 支持的模型 ✅

| 模型名称      | 参数量 | 显存需求 | 状态                |
| ------------- | ------ | -------- | ------------------- |
| Z-Image Turbo | 6B     | ~16GB    | ✅ 已添加到 catalog |
| Z-Image       | 6B     | ~16GB    | ✅ 已添加到 catalog |
| Flux.1        | 12B    | ~24GB    | ⏳ 可手动部署       |
| SD3           | 2B     | ~8GB     | ⏳ 可手动部署       |

## 如何使用

### 方式一：通过 Web UI 部署

1. 打开 GPUStack Web UI
2. 进入 "Models" → "Deploy Model"
3. 搜索 "Z-Image"
4. 选择模型，点击部署

### 方式二：通过 API 部署

```bash
curl -X POST http://localhost/api/v1/models \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "z-image-turbo",
    "source": "huggingface",
    "huggingface_repo_id": "Tongyi-MAI/Z-Image-Turbo",
    "backend": "vllm-omni",
    "backend_parameters": [
      "--num-inference-steps", "9",
      "--guidance-scale", "0.0"
    ],
    "categories": ["image"],
    "replicas": 1
  }'
```

### 方式三：使用图像生成 API

```bash
curl -X POST http://localhost/v1/images/generations \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-image-turbo",
    "prompt": "一只可爱的橘猫坐在窗台上，阳光洒在它身上",
    "n": 1,
    "size": "1024x1024"
  }'
```

## 代码位置

### 核心实现

**`gpustack/worker/backends/vllm_omni.py`**

```python
class VLLMOmniServer(InferenceServer):
    """vLLM-Omni inference server for omni-modality models."""

    def start(self):
        """启动 vLLM-Omni 服务"""

    def _detect_model_type(self) -> str:
        """自动检测模型类型"""

    def _get_diffusion_arguments(self) -> list:
        """获取 Diffusion 模型参数"""
```

### 集成点

**`gpustack/worker/serve_manager.py`**

```python
from gpustack.worker.backends.vllm_omni import VLLMOmniServer

# Line 333-335
elif backend == BackendEnum.VLLM_OMNI:
    VLLMOmniServer(clientset, mi, cfg, worker_id).start()
```

### 模型定义

**`gpustack/assets/model-catalog.yaml`**

```yaml
# Line 2683-2726
- name: Z-Image Turbo
  backend: vllm-omni
  backend_parameters:
    - --num-inference-steps
    - "9"
    - --guidance-scale
    - "0.0"
```

## 验证方法

### 1. 检查后端是否可用

```bash
# 查看支持的后端类型
curl http://localhost/api/v1/models | jq '.[] | select(.backend == "vllm-omni")'
```

### 2. 检查模型目录

```bash
# 搜索 Z-Image 模型
curl http://localhost/api/v1/model-catalog | jq '.[] | select(.name | contains("Z-Image"))'
```

### 3. 部署测试

```bash
# 部署 Z-Image Turbo
# 通过 Web UI 或 API 部署后，检查状态
curl http://localhost/api/v1/models/{model_id}
```

## 常见问题

### Q: vllm-omni 命令找不到？

**A**: 需要安装 vllm-omni：

```bash
pip install vllm-omni
```

### Q: 显存不足？

**A**: 调整参数：

```yaml
backend_parameters:
  - --gpu-memory-utilization
  - "0.8"
  - --max-num-seqs
  - "2"
```

### Q: 生成速度慢？

**A**: 启用缓存加速（默认已启用）：

```yaml
backend_parameters:
  - --cache-method
  - "teacache"
```

## 相关文档

- [vLLM-Omni 使用指南](beagle-vllm-omni-integration-guide.md) - 详细使用文档
- [vLLM 版本管理](../.beagle/vllm.md) - vLLM 版本说明
- [Beagle 变更记录](beagle-changes.md) - 所有定制化变更

## 总结

✅ **vLLM-Omni 后端已完全实现并可用**

- 代码已实现并集成
- Model Catalog 已包含 Z-Image 模型
- 支持通过 Web UI 和 API 部署
- 支持 OpenAI 兼容的图像生成 API

**可以直接使用，无需额外开发！**

---

**文档版本**：v1.0  
**最后更新**：2026-02-06  
**状态**：✅ 已完成
