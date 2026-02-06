# Beagle 定制变更说明

本文档记录了 Beagle 版本相对于上游 GPUStack v0.7.1 的所有定制化变更。

## 变更概览

| Patch 文件              | 变更类型         | 影响范围      |
| ----------------------- | ---------------- | ------------- |
| v0.7.1-apikey.patch     | API Key 格式优化 | 认证模块      |
| v0.7.1-bugfix.patch     | Bug 修复         | 核心功能      |
| v0.7.1-log-filter.patch | 日志过滤         | 日志系统      |
| v0.7.1-s3-project.patch | S3 存储集成      | 模型下载/存储 |
| v0.7.1-vllm.patch       | vLLM 版本升级    | 推理后端      |

---

## 1. API Key 格式优化 (v0.7.1-apikey.patch)

### 变更说明

简化 API Key 格式，移除 `gpustack_` 前缀，使 API Key 更简洁。

### 影响文件

- `gpustack/api/auth.py`
- `gpustack/routes/api_keys.py`

### 主要变更

**认证逻辑调整：**

```python
# 原格式: gpustack_{access_key}_{secret_key}
# 新格式: {access_key}_{secret_key}
```

- 移除了 `API_KEY_PREFIX` 常量的使用
- 修改 token 解析逻辑，从 3 部分改为 2 部分
- 简化 API Key 生成格式

### 使用影响

用户在使用 API Key 时不再需要 `gpustack_` 前缀，格式更加简洁。

---

## 2. Bug 修复 (v0.7.1-bugfix.patch)

### 变更说明

修复多个已知问题，提升系统稳定性。

### 影响文件

- `gpustack/main.py`
- `gpustack/mixins/active_record.py`
- `gpustack/policies/candidate_selectors/ascend_mindie_resource_fit_selector.py`

### 主要变更

**1. 过滤依赖库的弃用警告**

```python
# 在 main.py 中添加警告过滤
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets.legacy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn.protocols.websockets.websockets_impl")
```

**2. 修复 SQLModel 兼容性问题**

```python
# 使用 model_validate 替代已弃用的 from_orm
obj = cls.model_validate(source, update=update)
```

**3. 修复 Ascend MindIE 资源选择器的空值判断**

```python
# 修复 quantize 参数的空值检查
if (self._model_params.quantize is not None
    and self._model_params.quantize.lower().startswith("w8a8")):
```

---

## 3. 日志过滤功能 (v0.7.1-log-filter.patch)

### 变更说明

添加日志过滤功能，支持将日志中的 `gpustack` 替换为 `stack`，实现品牌定制化。

### 影响文件

- `gpustack/worker/logs.py`

### 主要变更

**新增功能：**

- 添加 `filter_gpustack` 参数（默认为 `True`）
- 实现 `filter_log_line()` 函数，自动替换日志中的模块名
- 支持通过 URL 参数控制是否启用过滤

**使用示例：**

```python
# 日志输出转换
# 原始: gpustack.worker.serve_manager - INFO - Starting service
# 过滤后: stack.worker.serve_manager - INFO - Starting service
```

**API 参数：**

```
GET /logs?tail=100&follow=true&filter_gpustack=true
```

---

## 4. S3 存储集成 (v0.7.1-s3-project.patch)

### 变更说明

集成 S3 对象存储支持，允许从 S3 直接加载模型文件，支持私有化部署场景。

### 影响文件

- `gpustack/logginglocal.py` (新增)
- `gpustack/worker/downloader_s3.py` (新增)
- `gpustack/cmd/start.py`
- `gpustack/config/config.py`
- `gpustack/worker/downloaders.py`
- `gpustack/scheduler/calculator.py`
- `gpustack/scheduler/evaluator.py`
- 多个模块的 logging 导入路径调整
- `pyproject.toml` (添加 minio 依赖)

### 主要变更

**1. 新增 S3 配置参数**

```bash
--worker-s3-host              # S3 服务地址
--worker-s3-access-key        # S3 访问密钥
--worker-s3-secret-key        # S3 密钥
--worker-s3-ssl               # 是否使用 SSL
--worker-s3-use-virtual-hosted-style  # 虚拟主机样式
--worker-s3-region            # S3 区域
```

**2. S3 路径格式**

```
s3://beagle_wind/bd-wind/datamodel/{model_id}/{version}/{file}
```

**3. 本地缓存路径**

```
{data_dir}/cache/beagle/{model_id}/{version}/{file}
```

**4. S3Downloader 类功能**

- 支持断点续传
- 分片下载（10MB per chunk）
- 文件锁机制避免并发下载冲突
- 进度条显示
- 自动创建本地缓存目录

**5. 日志模块重命名**
将 `gpustack/logging.py` 重命名为 `gpustack/logginglocal.py`，避免与标准库冲突。

### 环境变量

```bash
STACK_WORKER_S3_HOST
STACK_WORKER_S3_ACCESS_KEY
STACK_WORKER_S3_SECRET_KEY
STACK_WORKER_S3_SSL
STACK_WORKER_S3_USE_VIRTUAL_HOSTED_STYLE
STACK_WORKER_S3_REGION
```

---

## 5. vLLM 版本升级 (v0.7.1-vllm.patch)

### 变更说明

升级 vLLM 推理引擎版本，获得更好的性能和新特性支持。

### 影响文件

- `pyproject.toml`

### 主要变更

```toml
# 原版本: vllm = "0.10.1.1"
# 新版本: vllm = "0.11.2"
```

### 升级收益

- 性能优化
- 支持更多模型架构
- Bug 修复和稳定性提升

详细的 vLLM 版本管理说明请参考：[vLLM 版本管理文档](.beagle/vllm.md)

---

## 应用所有 Patch

### 手动应用

```bash
git apply .beagle/v0.7.1-apikey.patch
git apply .beagle/v0.7.1-bugfix.patch
git apply .beagle/v0.7.1-log-filter.patch
git apply .beagle/v0.7.1-s3-project.patch
git apply .beagle/v0.7.1-vllm.patch
```

### 一键应用

```bash
for patch in .beagle/v0.7.1-*.patch; do
    echo "Applying $patch..."
    git apply "$patch"
done
```

### 验证应用结果

```bash
# 查看已修改的文件
git status

# 查看具体变更
git diff
```

---

## 兼容性说明

### 上游版本

- 基于 GPUStack v0.7.1

### Python 版本

- Python 3.10+

### 依赖变更

- 新增：`minio ^7.2.0` (S3 客户端)
- 升级：`vllm 0.10.1.1 -> 0.11.2`

---

## 注意事项

1. **API Key 格式变更**：如果从旧版本升级，需要重新生成 API Key
2. **S3 配置**：使用 S3 功能需要正确配置 S3 相关参数
3. **日志过滤**：默认启用日志过滤，可通过参数关闭
4. **vLLM 升级**：新版本可能需要重新编译，首次启动时间较长

---

## 技术支持

如有问题，请参考：

- [Beagle API 使用指南](beagle-api-usage-guide.md)
- [Beagle API 复用方案](beagle-api-reuse-solution.md)
- [vLLM 版本管理](../beagle/vllm.md)
