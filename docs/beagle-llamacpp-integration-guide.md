# llama.cpp 后端集成指南

## 概述

本文档描述如何在 GPUStack 中集成 **llama.cpp** 后端，以支持 GGUF 格式模型的高效推理，特别是 Unsloth 提供的优化 GGUF 模型。

**核心价值**：

- ✅ 支持 GGUF 格式模型（Unsloth Dynamic 量化）
- ✅ 极低显存占用（2-bit 到 8-bit 量化）
- ✅ CPU + GPU 混合推理
- ✅ 与现有 llama-box 后端互补

**实现状态**：⏳ 待实现

---

## 背景

### 为什么需要 llama.cpp 后端？

GPUStack 现有后端的局限性：

| 后端      | 支持的模型格式  | GGUF 支持 | 量化支持         | 显存占用 |
| --------- | --------------- | --------- | ---------------- | -------- |
| vLLM      | HF Transformers | ❌        | FP8, INT8, GPTQ  | 高       |
| llama-box | GGUF            | ✅        | K-quants         | 低       |
| llama.cpp | GGUF            | ✅        | K-quants, IQ, MX | 极低     |

**llama.cpp vs llama-box**：

- **llama-box**：GPUStack 内置的 llama.cpp 封装，功能有限
- **llama.cpp**：原生 llama.cpp，支持最新特性和模型架构

### llama.cpp 特性

1. **极致量化**：支持 2-bit、3-bit、4-bit 等多种量化方法
2. **混合推理**：CPU + GPU 协同，充分利用系统资源
3. **低显存占用**：Qwen3.5-35B 仅需 ~20GB（4-bit 量化）
4. **最新架构支持**：快速跟进新模型架构（如 Qwen3.5 VL）

---

## 架构设计

### 系统架构（待实现）

```
GPUStack Server
      │
      ├── Model Catalog (model-catalog.yaml)
      │       └── Unsloth GGUF 模型定义 ⏳
      │
      ├── Scheduler
      │       └── 根据 backend 类型分配到对应 Worker ⏳
      │
      └── Worker
              ├── serve_manager.py ⏳
              │       └── 根据 BackendEnum 启动对应服务
              │
              └── backends/
                      ├── vllm.py          # LLM 推理
                      ├── llama_box.py     # 内置 GGUF 支持
                      ├── vox_box.py       # 音频模型
                      ├── vllm_omni.py     # Omni 模态模型
                      └── llama_cpp.py     # ⏳ 原生 llama.cpp（待实现）
```

### 数据流

```
用户请求 (POST /v1/chat/completions)
      │
      ▼
GPUStack API Gateway
      │
      ▼
Load Balancer (选择可用实例)
      │
      ▼
llama.cpp Server (Worker 节点)
      │
      ▼
返回生成的文本
```

---

## 实现步骤

### 第 1 步：添加 BackendEnum

**文件**：`gpustack/schemas/models.py`

```python
class BackendEnum(str, Enum):
    LLAMA_BOX = "llama-box"
    VLLM = "vllm"
    VOX_BOX = "vox-box"
    ASCEND_MINDIE = "ascend-mindie"
    VLLM_OMNI = "vllm-omni"
    LLAMA_CPP = "llama-cpp"  # ⏳ 待添加
```

### 第 2 步：实现 llama.cpp 后端

**文件**：`gpustack/worker/backends/llama_cpp.py`

参考 `vllm_omni.py` 的实现结构：

```python
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

from gpustack.schemas.models import ModelInstance
from gpustack.worker.backends.base import InferenceServer

logger = logging.getLogger(__name__)


class LlamaCppServer(InferenceServer):
    """
    llama.cpp inference server for GGUF models.

    Supports:
    - GGUF format models (Unsloth, TheBloke, etc.)
    - K-quants, IQ, MXFP4 quantization
    - CPU + GPU hybrid inference
    - OpenAI-compatible API via llama-server
    """

    def start(self):
        """启动 llama.cpp 服务"""
        try:
            logger.info(f"Starting llama.cpp server for model: {self._mi.model_name}")

            # 1. 检查 llama.cpp 是否安装
            self._check_llama_cpp_installation()

            # 2. 下载模型文件
            model_path = self._download_model()

            # 3. 构建命令行参数
            cmd = self._build_command(model_path)

            # 4. 启动服务
            logger.info(f"Launching llama.cpp with command: {' '.join(cmd)}")
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # 5. 监控日志
            self._monitor_logs()

        except Exception as e:
            logger.error(f"Failed to start llama.cpp server: {e}")
            raise

    def _check_llama_cpp_installation(self):
        """检查 llama.cpp 是否安装"""
        # 检查 llama-server 命令是否存在
        try:
            result = subprocess.run(
                ["llama-server", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError("llama-server not found")
            logger.info(f"llama.cpp version: {result.stdout.strip()}")
        except Exception as e:
            raise RuntimeError(
                "llama.cpp not installed. Please install: "
                "https://github.com/ggml-org/llama.cpp"
            ) from e

    def _download_model(self) -> Path:
        """下载 GGUF 模型文件"""
        # 使用 HuggingFace Hub 下载
        # 参考 llama_box.py 的实现
        repo_id = self._mi.huggingface_repo_id

        # 检测量化类型（从文件名或用户参数）
        quant_type = self._detect_quantization_type()

        # 下载模型文件
        # 例如：unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf
        model_file = f"*{quant_type}*.gguf"

        # 实际下载逻辑（使用 huggingface_hub）
        from huggingface_hub import hf_hub_download

        cache_dir = Path(self._cfg.data_dir) / "cache" / "huggingface"
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_file,
            cache_dir=cache_dir,
        )

        logger.info(f"Model downloaded to: {model_path}")
        return Path(model_path)

    def _detect_quantization_type(self) -> str:
        """检测量化类型"""
        # 从用户参数或默认值获取
        # 优先级：用户指定 > 模型默认 > Q4_K_M
        backend_params = self._mi.backend_parameters or {}
        return backend_params.get("quantization", "Q4_K_M")

    def _build_command(self, model_path: Path) -> list:
        """构建 llama-server 命令"""
        cmd = [
            "llama-server",
            "--model", str(model_path),
            "--host", "0.0.0.0",
            "--port", str(self._mi.port),
            "--ctx-size", "16384",  # 默认上下文长度
            "--n-gpu-layers", "999",  # 自动检测 GPU 层数
        ]

        # 添加多模态支持（如果有 mmproj 文件）
        mmproj_path = self._find_mmproj(model_path)
        if mmproj_path:
            cmd.extend(["--mmproj", str(mmproj_path)])
            logger.info(f"Found mmproj file: {mmproj_path}")

        # 添加用户自定义参数
        if self._mi.backend_parameters:
            for key, value in self._mi.backend_parameters.items():
                if key != "quantization":  # 跳过内部参数
                    cmd.extend([f"--{key}", str(value)])

        # 性能优化参数
        cmd.extend([
            "--parallel", "4",  # 并发请求数
            "--batch-size", "512",  # 批处理大小
            "--ubatch-size", "512",  # 微批处理大小
        ])

        return cmd

    def _find_mmproj(self, model_path: Path) -> Optional[Path]:
        """查找 mmproj 文件（用于多模态模型）"""
        # 在同一目录下查找 mmproj-*.gguf
        model_dir = model_path.parent
        mmproj_files = list(model_dir.glob("mmproj-*.gguf"))

        if mmproj_files:
            return mmproj_files[0]
        return None

    def _monitor_logs(self):
        """监控服务日志"""
        # 参考 vllm_omni.py 的实现
        for line in iter(self._process.stdout.readline, ""):
            if line:
                logger.info(line.strip())

                # 检测服务启动成功
                if "HTTP server listening" in line:
                    logger.info("llama.cpp server started successfully")
                    break

                # 检测错误
                if "error" in line.lower() or "failed" in line.lower():
                    logger.error(f"llama.cpp error: {line.strip()}")

    def health_check(self) -> bool:
        """健康检查"""
        try:
            import requests
            response = requests.get(
                f"http://localhost:{self._mi.port}/health",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False
```

### 第 3 步：集成到 ServeManager

**文件**：`gpustack/worker/serve_manager.py`

```python
from gpustack.worker.backends.llama_cpp import LlamaCppServer

# 在 serve_model_instance 方法中添加
elif backend == BackendEnum.LLAMA_CPP:
    LlamaCppServer(clientset, mi, cfg, worker_id).start()
```

### 第 4 步：添加到 Model Catalog

**文件**：`gpustack/assets/model-catalog.yaml`

```yaml
# Unsloth Qwen3.5 GGUF 模型
- name: Qwen3.5-9B (GGUF Q4)
  backend: llama-cpp
  source: huggingface
  huggingface_repo_id: unsloth/Qwen3.5-9B-GGUF
  backend_parameters:
    quantization: UD-Q4_K_XL
    ctx-size: "16384"
    n-gpu-layers: "999"
  categories:
    - text-generation
  description: Qwen3.5-9B with Unsloth Dynamic Q4 quantization
  memory_requirements:
    min_ram_gb: 8
    min_vram_gb: 6

- name: Qwen3.5-35B-A3B (GGUF Q4)
  backend: llama-cpp
  source: huggingface
  huggingface_repo_id: unsloth/Qwen3.5-35B-A3B-GGUF
  backend_parameters:
    quantization: UD-Q4_K_XL
    ctx-size: "16384"
    n-gpu-layers: "999"
  categories:
    - text-generation
  description: Qwen3.5-35B-A3B with Unsloth Dynamic Q4 quantization
  memory_requirements:
    min_ram_gb: 24
    min_vram_gb: 20
```

---

## 支持的模型

### Unsloth GGUF 模型

| 模型              | 量化类型 | 显存需求 | 推荐硬件      |
| ----------------- | -------- | -------- | ------------- |
| Qwen3.5-0.8B      | Q4_K_XL  | ~1GB     | 任意 GPU      |
| Qwen3.5-2B        | Q4_K_XL  | ~2GB     | 任意 GPU      |
| Qwen3.5-4B        | Q4_K_XL  | ~4GB     | 任意 GPU      |
| Qwen3.5-9B        | Q4_K_XL  | ~6GB     | RTX 3060 12GB |
| Qwen3.5-27B       | Q4_K_XL  | ~18GB    | RTX 4090 24GB |
| Qwen3.5-35B-A3B   | Q4_K_XL  | ~20GB    | RTX 4090 24GB |
| Qwen3.5-122B-A10B | Q4_K_XL  | ~70GB    | A100 80GB     |
| Qwen3.5-397B-A17B | Q3_K_XL  | ~192GB   | 多卡或 CPU    |

### 量化类型对比

| 量化类型   | 位数 | 质量 | 速度 | 显存占用 |
| ---------- | ---- | ---- | ---- | -------- |
| UD-Q2_K_XL | 2    | 中   | 快   | 极低     |
| UD-Q3_K_XL | 3    | 良   | 快   | 低       |
| UD-Q4_K_XL | 4    | 优   | 中   | 中       |
| UD-Q5_K_XL | 5    | 优+  | 中   | 中高     |
| UD-Q8_0    | 8    | 极优 | 慢   | 高       |

---

## 部署指南

### 前置条件

1. **安装 llama.cpp**：

   ```bash
   # 方法 1：从源码编译（推荐）
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release -j
   sudo cp build/bin/llama-server /usr/local/bin/

   # 方法 2：使用预编译二进制
   # 从 https://github.com/ggml-org/llama.cpp/releases 下载
   ```

2. **验证安装**：
   ```bash
   llama-server --version
   ```

### 方式一：通过 Model Catalog 部署

1. 打开 GPUStack Web UI
2. 进入 "Models" → "Deploy Model"
3. 搜索 "Qwen3.5 GGUF"
4. 选择量化版本，点击部署

### 方式二：手动部署

```bash
# 通过 API 创建模型
curl -X POST http://localhost/api/v1/models \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qwen3.5-9b-gguf",
    "source": "huggingface",
    "huggingface_repo_id": "unsloth/Qwen3.5-9B-GGUF",
    "backend": "llama-cpp",
    "backend_parameters": {
      "quantization": "UD-Q4_K_XL",
      "ctx-size": "16384",
      "n-gpu-layers": "999"
    },
    "categories": ["text-generation"],
    "replicas": 1
  }'
```

---

## API 使用

### Chat Completions API

**端点**：`POST /v1/chat/completions`

**请求示例**：

```bash
curl -X POST http://localhost/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b-gguf",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 2048
  }'
```

---

## 性能优化

### 1. GPU 层数优化

```yaml
backend_parameters:
  n-gpu-layers: "35" # 根据显存调整
```

### 2. 批处理优化

```yaml
backend_parameters:
  parallel: "8" # 并发请求数
  batch-size: "512" # 批处理大小
```

### 3. 上下文长度

```yaml
backend_parameters:
  ctx-size: "32768" # 根据需求调整
```

---

## 与其他后端对比

| 特性         | vLLM | llama-box | llama.cpp |
| ------------ | ---- | --------- | --------- |
| GGUF 支持    | ❌   | ✅        | ✅        |
| 显存占用     | 高   | 低        | 极低      |
| 推理速度     | 快   | 中        | 中        |
| CPU 推理     | ❌   | ✅        | ✅        |
| 最新模型支持 | 快   | 慢        | 快        |
| Unsloth GGUF | ❌   | 部分      | ✅        |
| 多模态支持   | ✅   | 部分      | ✅        |

---

## 参考资料

- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [Unsloth GGUF 文档](https://unsloth.ai/docs/zh/mo-xing/qwen3.5)
- [vLLM-Omni 集成指南](beagle-vllm-omni-integration-guide.md) - 参考实现
- [GPUStack 文档](https://docs.gpustack.ai)

---

**文档版本**：v1.0  
**最后更新**：2026-03-08  
**状态**：⏳ 待实现
