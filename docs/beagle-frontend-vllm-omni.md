# 如何在前端显示 vLLM-Omni 模型

## 问题

vLLM-Omni 后端已经完全实现，但是在 Web UI 的 Model Catalog 中看不到 Z-Image 等模型。

## 原因

GPUStack 有两个 Model Catalog 文件：

1. **`model-catalog.yaml`** - 用于可以访问 HuggingFace 的环境
2. **`model-catalog-modelscope.yaml`** - 用于无法访问 HuggingFace 的环境（使用 ModelScope）

在中国大陆等无法访问 HuggingFace 的环境中，系统会自动使用 ModelScope catalog。

**问题**：Z-Image 模型最初只添加到了 `model-catalog.yaml`，没有添加到 `model-catalog-modelscope.yaml`。

## 解决方案

### 已完成的修复 ✅

已将 Z-Image 模型添加到 `model-catalog-modelscope.yaml` 文件中：

**文件**：`gpustack/assets/model-catalog-modelscope.yaml`

```yaml
# vLLM-Omni Diffusion Models
- name: Z-Image Turbo
  description: Z-Image-Turbo is a 6B parameter text-to-image model from Alibaba Tongyi-MAI. It delivers high-fidelity images in 8 steps with sub-second inference latency, supports bilingual text rendering (English & Chinese), and is optimized for photorealistic image generation.
  home: https://github.com/Tongyi-MAI/Z-Image
  icon: /static/catalog_icons/alibaba.png
  categories:
    - image
  licenses:
    - apache-2.0
  release_date: "2025-05-30"
  templates:
    - quantizations: ["BF16"]
      source: model_scope
      model_scope_model_id: Tongyi-MAI/Z-Image-Turbo
      replicas: 1
      backend: vllm-omni
      backend_parameters:
        - --num-inference-steps
        - "9"
        - --guidance-scale
        - "0.0"
- name: Z-Image
  description: Z-Image is a 6B parameter foundation model for high-quality image generation. It focuses on rich aesthetics, strong diversity, and controllability, well-suited for creative generation and fine-tuning.
  home: https://github.com/Tongyi-MAI/Z-Image
  icon: /static/catalog_icons/alibaba.png
  categories:
    - image
  licenses:
    - apache-2.0
  release_date: "2025-05-30"
  templates:
    - quantizations: ["BF16"]
      source: model_scope
      model_scope_model_id: Tongyi-MAI/Z-Image
      replicas: 1
      backend: vllm-omni
      backend_parameters:
        - --num-inference-steps
        - "50"
        - --guidance-scale
        - "4.0"
```

### 验证修复

1. **检查 API**：

```bash
curl -s 'http://localhost:6080/v1/model-sets?search=Z-Image' | python3 -m json.tool
```

应该返回 2 个模型：Z-Image Turbo 和 Z-Image。

2. **检查 Web UI**：

打开 `http://localhost:6080`，进入 "Models" → "Deploy Model"，搜索 "Z-Image"，应该能看到两个模型。

## 前端代码说明

### 前端是预编译的

GPUStack 的前端 UI 是预编译的静态文件，从腾讯云 COS 下载：

```bash
https://gpustack-ui-1303613262.cos.accelerate.myqcloud.com/releases/v0.7.1.tar.gz
```

**前端不需要修改**，因为：

1. 前端通过 API 动态加载 Model Catalog
2. 前端会自动显示所有 `backend` 类型的模型
3. 只要后端 API 返回模型，前端就会显示

### Model Catalog 加载流程

```
启动服务
    ↓
检查是否能访问 HuggingFace
    ↓
    ├─ 能访问 → 使用 model-catalog.yaml
    └─ 不能访问 → 使用 model-catalog-modelscope.yaml
    ↓
加载 YAML 文件
    ↓
解析为 ModelSet 对象
    ↓
通过 API 提供给前端
    ↓
前端显示在 Model Catalog 中
```

### 相关代码

**后端 Catalog 初始化**：`gpustack/server/catalog.py`

```python
def get_builtin_model_catalog_file() -> str:
    huggingface_url = "https://huggingface.co"
    modelscope_url = "https://modelscope.cn"

    model_catalog_file_name = "model-catalog.yaml"

    if not can_access(huggingface_url) and can_access(modelscope_url):
        model_catalog_file_name = "model-catalog-modelscope.yaml"
        logger.info(f"Cannot access {huggingface_url}, using ModelScope model catalog.")

    return str(pkg_resources.files("gpustack.assets").joinpath(model_catalog_file_name))
```

**API 路由**：`gpustack/routes/model_sets.py`

```python
@router.get("", response_model=PaginatedList[ModelSetPublic])
async def get_model_sets(
    params: ListParamsDep,
    search: str = None,
    categories: Optional[List[str]] = Query(None),
    model_catalog: List[ModelSet] = Depends(get_model_catalog),
):
    # 返回 model catalog 给前端
```

## 添加新模型到 Catalog

### 步骤

1. **确定使用哪个 catalog 文件**：
   - 如果模型在 HuggingFace 上：添加到 `model-catalog.yaml`
   - 如果模型在 ModelScope 上：添加到 `model-catalog-modelscope.yaml`
   - **最好两个都添加**

2. **添加模型定义**：

```yaml
- name: 你的模型名称
  description: 模型描述
  home: https://github.com/xxx
  icon: /static/catalog_icons/xxx.png
  categories:
    - image # 或 text, audio, video 等
  licenses:
    - apache-2.0
  release_date: "2025-01-01"
  templates:
    - quantizations: ["BF16"]
      source: model_scope # 或 huggingface
      model_scope_model_id: xxx/yyy # 或 huggingface_repo_id
      replicas: 1
      backend: vllm-omni # 指定后端
      backend_parameters:
        - --your-param
        - "value"
```

3. **验证 YAML 语法**：

```bash
poetry run python3 -c "import yaml; yaml.safe_load(open('gpustack/assets/model-catalog-modelscope.yaml')); print('Valid')"
```

4. **重启服务**：

```bash
# 停止服务
pkill -f gpustack

# 清除缓存（可选）
rm -rf ${HOME}/gpustack/*.db*

# 启动服务
poetry run python3 gpustack/main.py start ...
```

5. **验证**：

```bash
# 检查 API
curl -s 'http://localhost:6080/v1/model-sets?search=你的模型名' | python3 -m json.tool

# 检查 Web UI
# 打开浏览器访问 http://localhost:6080
```

## 常见问题

### Q: 为什么有两个 catalog 文件？

**A**: 因为 HuggingFace 在某些地区无法访问，所以提供了 ModelScope 作为备选。系统会自动检测并选择可用的源。

### Q: 如何知道系统使用了哪个 catalog？

**A**: 查看启动日志：

```
# 使用 HuggingFace
Loaded 82 model sets from model catalog: .../model-catalog.yaml

# 使用 ModelScope
Cannot access https://huggingface.co, using ModelScope model catalog.
Loaded 84 model sets from model catalog: .../model-catalog-modelscope.yaml
```

### Q: 前端需要修改吗？

**A**: 不需要。前端通过 API 动态加载模型列表，只要后端 API 返回模型，前端就会自动显示。

### Q: 如何添加自定义 icon？

**A**: 将图标文件放到 `static/catalog_icons/` 目录，然后在 catalog 中引用：

```yaml
icon: /static/catalog_icons/your-icon.png
```

### Q: 模型添加后看不到？

**A**: 检查：

1. YAML 语法是否正确
2. 是否重启了服务
3. 是否清除了缓存
4. 查看服务日志是否有错误

## 总结

✅ **vLLM-Omni 模型现在可以在前端显示了**

- 已添加到 ModelScope catalog
- 前端不需要修改
- 通过 API 动态加载
- 支持搜索和过滤

**关键点**：

1. 后端已完全实现 ✅
2. Model Catalog 已更新 ✅
3. 前端自动显示 ✅
4. 无需修改前端代码 ✅

---

**文档版本**：v1.0  
**最后更新**：2026-02-06  
**状态**：✅ 已解决
