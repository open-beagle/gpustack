# vLLM-Omni 集成指南

## 概述

本文档描述 GPUStack 集成 vLLM-Omni 后端的方案，用于支持 **Omni-Modality 模型**（全模态模型）的推理服务，包括：

- **Diffusion 图像生成模型**：Z-Image、Flux、Stable Diffusion 3 等
- **音频模型**：语音生成、语音识别等
- **视频模型**：视频生成等
- **多模态生成模型**：文本+图像+音频混合输出

**核心价值**：

- ✅ 支持 Z-Image 等 DiT 架构的图像生成模型
- ✅ 统一的 API 接口，兼容 OpenAI 格式
- ✅ 高性能推理，支持缓存加速
- ✅ 与 GPUStack 现有架构无缝集成

---

## 背景

### 为什么需要 vLLM-Omni？

GPUStack 现有后端的局限性：

| 后端      | 支持的模型类型                | 不支持                    |
| --------- | ----------------------------- | ------------------------- |
| vLLM      | LLM、VLM                      | Diffusion 模型            |
| llama-box | GGUF 格式的 LLM、部分图像模型 | 非 GGUF 的 Diffusion 模型 |
| vox-box   | 音频模型                      | 图像/视频生成             |

**vLLM-Omni** 是 vLLM 官方的扩展项目，专门支持：

- Diffusion Transformer (DiT) 架构
- 非自回归生成模型
- 多模态输入输出

### vLLM-Omni 特性

1. **高效推理**：继承 vLLM 的 PagedAttention 和 KV Cache 优化
2. **缓存加速**：支持 TeaCache、Cache-DiT 等加速方法
3. **分布式部署**：支持多 GPU 并行推理
4. **OpenAI 兼容 API**：标准的 `/v1/images/generations` 接口

---

## 架构设计

### 系统架构

```
GPUStack Server
      │
      ├── Model Catalog (model-catalog.yaml)
      │       └── Z-Image, Flux 等模型定义
      │
      ├── Scheduler
      │       └── 根据 backend 类型分配到对应 Worker
      │
      └── Worker
              ├── serve_manager.py
              │       └── 根据 BackendEnum 启动对应服务
              │
              └── backends/
                      ├── vllm.py          # LLM 推理
                      ├── llama_box.py     # GGUF 模型
                      ├── vox_box.py       # 音频模型
                      └── vllm_omni.py     # 🆕 Omni 模态模型
```

### 数据流

```
用户请求 (POST /v1/images/generations)
      │
      ▼
GPUStack API Gateway
      │
      ▼
Load Balancer (选择可用实例)
      │
      ▼
vLLM-Omni Server (Worker 节点)
      │
      ▼
返回生成的图像
```

---

## 代码改动

### 1. 新增 BackendEnum

**文件**：`gpustack/schemas/models.py`

```python
class BackendEnum(str, Enum):
    LLAMA_BOX = "llama-box"
    VLLM = "vllm"
    VOX_BOX = "vox-box"
    ASCEND_MINDIE = "ascend-mindie"
    VLLM_OMNI = "vllm-omni"  # 🆕 新增
```

### 2. 新增 vLLM-Omni 后端实现

**文件**：`gpustack/worker/backends/vllm_omni.py`

```python
class VLLMOmniServer(InferenceServer):
    """
    vLLM-Omni 推理服务器

    支持：
    - Diffusion 模型 (Z-Image, Flux, SD3)
    - 音频模型
    - 视频模型
    """

    def start(self):
        command_path = get_command_path("vllm-omni")
        arguments = self._build_arguments()

        # 启动 vllm-omni serve
        subprocess.run([command_path] + arguments, ...)

    def _build_arguments(self):
        # 根据模型类型构建参数
        model_type = self._detect_model_type()
        if model_type == "diffusion":
            return self._get_diffusion_arguments()
        ...
```

### 3. 更新 ServeManager

**文件**：`gpustack/worker/serve_manager.py`

```python
from gpustack.worker.backends.vllm_omni import VLLMOmniServer

# 在 serve_model_instance 方法中添加
elif backend == BackendEnum.VLLM_OMNI:
    VLLMOmniServer(clientset, mi, cfg, worker_id).start()
```

### 4. 添加模型目录

**文件**：`gpustack/assets/model-catalog.yaml`

```yaml
- name: Z-Image Turbo
  description: 阿里通义 6B 参数图像生成模型，8 步快速生成
  home: https://github.com/Tongyi-MAI/Z-Image
  icon: /static/catalog_icons/alibaba.png
  categories:
    - image
  licenses:
    - apache-2.0
  templates:
    - quantizations: ["BF16"]
      source: huggingface
      huggingface_repo_id: Tongyi-MAI/Z-Image-Turbo
      replicas: 1
      backend: vllm-omni
      backend_parameters:
        - --num-inference-steps
        - "9"
        - --guidance-scale
        - "0.0"
```

---

## 支持的模型

### Diffusion 图像生成模型

| 模型           | 参数量 | 显存需求 | 特点                         |
| -------------- | ------ | -------- | ---------------------------- |
| Z-Image Turbo  | 6B     | ~16GB    | 8 步快速生成，中英文文字渲染 |
| Z-Image        | 6B     | ~16GB    | 高质量生成，支持负向提示     |
| Flux.1 Dev     | 12B    | ~24GB    | 高质量，支持多种风格         |
| Flux.1 Schnell | 12B    | ~24GB    | 快速版本                     |
| SD3 Medium     | 2B     | ~8GB     | 轻量级，适合入门             |

### 推荐配置

**RTX 4090 (24GB)**：

- Z-Image Turbo ✅
- Z-Image ✅
- Flux.1 (需要优化) ⚠️

**A100 (80GB)**：

- 所有模型 ✅
- 支持批量生成 ✅

---

## 部署指南

### 前置条件

1. **安装 vllm-omni**：

   ```bash
   pip install vllm-omni
   ```

2. **验证安装**：
   ```bash
   vllm-omni --version
   ```

### 方式一：通过 Model Catalog 部署

1. 打开 GPUStack Web UI
2. 进入 "Models" → "Deploy Model"
3. 搜索 "Z-Image" 或 "Flux"
4. 选择量化版本，点击部署

### 方式二：手动部署

```bash
# 通过 API 创建模型
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

---

## API 使用

### 图像生成 API

**端点**：`POST /v1/images/generations`

**请求示例**：

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

**响应示例**：

```json
{
  "created": 1706140800,
  "data": [
    {
      "url": "http://localhost/files/images/xxx.png",
      "revised_prompt": "..."
    }
  ]
}
```

### 支持的参数

| 参数            | 类型    | 说明                     |
| --------------- | ------- | ------------------------ |
| model           | string  | 模型名称                 |
| prompt          | string  | 生成提示词               |
| negative_prompt | string  | 负向提示词（可选）       |
| n               | integer | 生成数量，默认 1         |
| size            | string  | 图像尺寸，如 "1024x1024" |
| quality         | string  | 质量等级：standard/hd    |
| style           | string  | 风格：vivid/natural      |

---

## 后端参数配置

### Z-Image Turbo 推荐参数

```yaml
backend_parameters:
  - --num-inference-steps
  - "9" # 8 步生成（实际 8 次前向）
  - --guidance-scale
  - "0.0" # Turbo 版本不需要 CFG
  - --cache-method
  - "teacache" # 启用缓存加速
```

### Z-Image 推荐参数

```yaml
backend_parameters:
  - --num-inference-steps
  - "50" # 标准 50 步
  - --guidance-scale
  - "4.0" # CFG 引导强度
  - --cfg-normalization
  - "false"
```

### 通用优化参数

```yaml
backend_parameters:
  - --gpu-memory-utilization
  - "0.9" # GPU 显存利用率
  - --max-num-seqs
  - "4" # 最大并发请求数
```

---

## 性能优化

### 1. 缓存加速

vLLM-Omni 支持多种缓存方法：

| 方法      | 加速比 | 质量损失 | 适用场景 |
| --------- | ------ | -------- | -------- |
| TeaCache  | ~2x    | 极小     | 通用     |
| Cache-DiT | ~3-4x  | 小       | 批量生成 |
| DBCache   | ~4x    | 中等     | 实时应用 |

**启用方式**：

```yaml
backend_parameters:
  - --cache-method
  - "teacache"
```

### 2. 批量生成

```yaml
backend_parameters:
  - --max-num-seqs
  - "8" # 增加并发数
```

### 3. 显存优化

```yaml
backend_parameters:
  - --gpu-memory-utilization
  - "0.85" # 预留部分显存
  - --enable-chunked-prefill # 分块预填充
```

---

## 故障排查

### 常见问题

#### 1. 模型加载失败

**错误**：`Unknown model architecture: 'xxx'`

**原因**：vLLM-Omni 版本不支持该模型架构

**解决**：

```bash
# 升级 vllm-omni
pip install --upgrade vllm-omni
```

#### 2. 显存不足

**错误**：`CUDA out of memory`

**解决**：

```yaml
backend_parameters:
  - --gpu-memory-utilization
  - "0.8"
  - --max-num-seqs
  - "2"
```

#### 3. 生成速度慢

**解决**：启用缓存加速

```yaml
backend_parameters:
  - --cache-method
  - "teacache"
```

### 日志查看

```bash
# 查看模型实例日志
tail -f /var/lib/gpustack/log/serve/{instance_id}.log
```

---

## 与其他后端对比

| 特性           | vLLM | llama-box | vLLM-Omni |
| -------------- | ---- | --------- | --------- |
| LLM 推理       | ✅   | ✅        | ✅        |
| VLM 推理       | ✅   | ✅        | ✅        |
| Diffusion 模型 | ❌   | 部分 GGUF | ✅        |
| 音频生成       | ❌   | ❌        | ✅        |
| 视频生成       | ❌   | ❌        | ✅        |
| 分布式推理     | ✅   | ✅        | ✅        |
| 缓存加速       | ✅   | ✅        | ✅        |

---

## 未来规划

### 短期（1-2 个月）

- [ ] 完善 vLLM-Omni 后端实现
- [ ] 添加更多 Diffusion 模型支持
- [ ] 优化图像生成 API 路由

### 中期（3-6 个月）

- [ ] 支持音频生成模型
- [ ] 支持视频生成模型
- [ ] 添加 LoRA 微调支持

### 长期

- [ ] 多模态混合生成
- [ ] 实时流式生成
- [ ] 模型编排和工作流

---

## 参考资料

- [vLLM-Omni GitHub](https://github.com/vllm-project/vllm-omni)
- [vLLM-Omni 官方博客](https://blog.vllm.ai/2025/11/30/vllm-omni.html)
- [Z-Image GitHub](https://github.com/Tongyi-MAI/Z-Image)
- [GPUStack 文档](https://docs.gpustack.ai)

---

**文档版本**：v1.0  
**最后更新**：2026-02-03  
**维护者**：GPUStack 开发团队
