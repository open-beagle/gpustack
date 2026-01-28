# Beagle API 使用指南

## 概述

本文档描述 Beagle 平台基于 GPUStack 的模型管理 API 设计方案，主要解决以下业务需求：

1. **列表页显示 GPU 分配信息**：显示格式为 `worker-01 NVIDIA A10 [0,1]`（节点名 + GPU 型号 + GPU 索引）
2. **详情页显示完整 GPU 信息**：包括 GPU 型号、显存总量、已分配显存、实际使用显存、GPU 利用率
3. **Token 使用量统计**：显示 24 小时内的 Token 使用量（Prompt Tokens、Completion Tokens、总请求数）

---

## 业务场景

### 场景 1：模型实例列表页

**需求**：运维人员需要快速查看所有模型实例的 GPU 分配情况

**显示内容**：

- 实例名称
- 模型名称
- 运行状态
- GPU 分配：`worker-01 NVIDIA A10 [0,1]`

**设计要点**：

- 简洁明了，一行显示完整信息
- 支持多 GPU、多节点的情况
- 性能优化：默认不加载 GPU 详细信息，按需加载

### 场景 2：模型详情页

**需求**：查看模型的详细资源使用情况和性能指标

**显示内容**：

- 所有实例的 GPU 详细信息表格
- 24 小时 Token 使用量统计
- 显存分配和使用情况
- GPU 利用率

**设计要点**：

- 完整的 GPU 信息展示
- 实时或准实时的使用量数据
- 支持按时间范围查询

### 场景 3：资源统计和监控

**需求**：管理员需要了解整体资源使用情况

**显示内容**：

- 按 GPU 型号汇总的资源使用
- 按时间段的 Token 使用趋势
- 按操作类型的使用分布

---

## API 设计方案

### 方案概述

#### 短期方案（基于现有数据）

利用 GPUStack 现有的数据结构：

- `ModelInstance` 表：包含 `worker_id`、`gpu_indexes`
- `Worker` 表：包含 `status.gpu_devices`（GPU 型号、显存等信息）
- `ModelUsage` 表：包含 Token 使用记录

通过关联查询获取所需信息，无需修改数据库结构。

#### 长期方案（优化性能）

考虑以下优化：

- 添加数据库索引提升查询性能
- 使用缓存减少重复查询
- 考虑物化视图存储汇总数据

---

## API 端点设计

### 1. 获取模型实例列表（扩展现有 API）

**端点**：`GET /api/v1/model-instances`

**功能**：获取模型实例列表，支持显示 GPU 分配信息

#### 请求参数

| 参数             | 类型    | 必填 | 说明                          |
| ---------------- | ------- | ---- | ----------------------------- |
| include_gpu_info | boolean | 否   | 是否包含 GPU 信息，默认 false |
| model_id         | integer | 否   | 按模型 ID 过滤                |
| worker_id        | integer | 否   | 按 Worker ID 过滤             |
| state            | string  | 否   | 按状态过滤                    |
| page             | integer | 否   | 页码，默认 1                  |
| perPage          | integer | 否   | 每页数量，默认 10             |

#### 响应格式

```json
{
  "items": [
    {
      "id": 1,
      "name": "llama-3-8b-instance-1",
      "model_id": 1,
      "model_name": "llama-3-8b",
      "state": "running",
      "worker_id": 1,
      "worker_name": "worker-01",
      "worker_ip": "192.168.1.10",
      "gpu_indexes": [0, 1],
      "gpu_display": ["worker-01 NVIDIA A10 [0,1]"],
      "created_at": "2024-01-20T10:00:00Z",
      "updated_at": "2024-01-20T10:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "perPage": 10,
    "total": 50,
    "totalPage": 5
  }
}
```

#### 字段说明

- `gpu_display`：GPU 分配显示文本数组
  - 格式：`{worker_name} {gpu_model} [{gpu_indexes}]`
  - 示例：`"worker-01 NVIDIA A10 [0,1]"`
  - 如果实例使用多种 GPU 型号，会有多条记录

#### 数据获取逻辑

1. 查询 `ModelInstance` 表获取实例列表
2. 如果 `include_gpu_info=true`：
   - 根据 `worker_id` 查询 `Worker` 表
   - 从 `worker.status.gpu_devices` 获取 GPU 信息
   - 根据 `gpu_indexes` 匹配对应的 GPU 设备
   - 按 GPU 型号分组，生成 `gpu_display` 文本

#### GPU 显示格式规则

**单个 GPU**：

```
worker-01 NVIDIA A10 [0]
```

**多个 GPU（同型号）**：

```
worker-01 NVIDIA A10 [0,1,2]
```

**多个 GPU（不同型号）**：

```json
["worker-01 NVIDIA A10 [0,1]", "worker-01 NVIDIA A100 [2]"]
```

**分布式推理（多节点）**：

```json
["worker-01 NVIDIA A10 [0,1]", "worker-02 NVIDIA A10 [0,1]"]
```

---

### 2. 获取模型详细信息（新增 API）

**端点**：`GET /api/v1/stats/models/{id}/details`

**功能**：获取模型的详细 GPU 信息和 24 小时 Token 使用量

#### 路径参数

| 参数 | 类型    | 必填 | 说明    |
| ---- | ------- | ---- | ------- |
| id   | integer | 是   | 模型 ID |

#### 响应格式

```json
{
  "model_id": 1,
  "model_name": "llama-3-8b",
  "instance_count": 2,
  "total_vram_allocated": 17179869184,
  "gpu_details": [
    {
      "worker_id": 1,
      "worker_name": "worker-01",
      "gpu_index": 0,
      "gpu_name": "NVIDIA A10",
      "gpu_vendor": "NVIDIA",
      "vram_total": 25769803776,
      "vram_allocated": 8589934592,
      "vram_used": 7516192768,
      "utilization_rate": 85.5
    }
  ],
  "token_usage_24h": {
    "prompt_tokens": 125000,
    "completion_tokens": 375000,
    "total_tokens": 500000,
    "total_requests": 1250
  }
}
```

#### 字段说明

**GPU 详细信息**：

- `gpu_name`：GPU 完整名称（如 "NVIDIA A10"）
- `gpu_vendor`：GPU 厂商（如 "NVIDIA"）
- `vram_total`：GPU 总显存（字节）
- `vram_allocated`：模型分配的显存（字节）
- `vram_used`：实际使用的显存（字节）
- `utilization_rate`：GPU 利用率（0-100）

**Token 使用统计**：

- `prompt_tokens`：输入 Token 数量
- `completion_tokens`：输出 Token 数量
- `total_tokens`：总 Token 数量
- `total_requests`：总请求次数

#### 数据获取逻辑

1. 查询 `Model` 表获取模型基本信息
2. 查询所有关联的 `ModelInstance`
3. 对每个实例：
   - 查询 `Worker` 获取 GPU 设备信息
   - 从 `computed_resource_claim.vram` 获取分配的显存
   - 从 `worker.status.gpu_devices` 获取实时使用情况
4. 查询 `ModelUsage` 表：
   - 筛选条件：`model_id = {id}` AND `date >= (today - 1 day)`
   - 聚合计算：SUM(prompt_token_count)、SUM(completion_token_count)、SUM(request_count)

---

### 3. 获取模型统计信息（新增 API）

**端点**：`GET /api/v1/stats/models/{id}/stats`

**功能**：获取模型的资源使用汇总和灵活时间范围的统计

#### 路径参数

| 参数 | 类型    | 必填 | 说明    |
| ---- | ------- | ---- | ------- |
| id   | integer | 是   | 模型 ID |

#### 查询参数

| 参数  | 类型    | 必填 | 说明                          |
| ----- | ------- | ---- | ----------------------------- |
| hours | integer | 否   | 统计时间范围（小时），默认 24 |

#### 响应格式

```json
{
  "model_id": 1,
  "model_name": "llama-3-8b",
  "time_range_hours": 24,
  "total_vram_allocated": 17179869184,
  "gpu_summary": [
    {
      "gpu_model": "NVIDIA A10",
      "gpu_vendor": "NVIDIA",
      "count": 2,
      "total_vram": 25769803776,
      "allocated_vram": 17179869184,
      "instances": [
        {
          "worker": "worker-01",
          "gpu_index": 0,
          "instance_name": "llama-3-8b-instance-1"
        }
      ]
    }
  ],
  "usage": {
    "prompt_tokens": 125000,
    "completion_tokens": 375000,
    "total_tokens": 500000,
    "total_requests": 1250
  },
  "operation_breakdown": [
    {
      "operation": "chat_completion",
      "total_tokens": 450000,
      "requests": 1100
    },
    {
      "operation": "completion",
      "total_tokens": 50000,
      "requests": 150
    }
  ]
}
```

#### 字段说明

**GPU 汇总**：

- 按 GPU 型号分组统计
- `count`：该型号 GPU 的数量
- `total_vram`：单个 GPU 的总显存
- `allocated_vram`：该型号所有 GPU 的总分配显存
- `instances`：使用该 GPU 的实例列表

**操作类型分解**：

- `operation`：操作类型（chat_completion、completion、embedding 等）
- `total_tokens`：该操作类型的总 Token 数
- `requests`：该操作类型的请求次数

#### 数据获取逻辑

1. 查询模型的所有实例
2. 按 GPU 型号分组汇总：
   - 统计每种 GPU 的数量
   - 累计分配的显存
   - 记录使用该 GPU 的实例
3. 查询 `ModelUsage` 表：
   - 筛选条件：`model_id = {id}` AND `date >= (now - hours)`
   - 按 `operation` 分组统计

---

## 数据模型设计

### GPU 分配信息（列表页）

```typescript
interface GPUAllocationInfo {
  worker_name: string; // Worker 节点名称
  gpu_model: string; // GPU 型号（简化后）
  gpu_indexes: number[]; // GPU 索引数组
  display_text: string; // 格式化显示文本
}
```

### GPU 详细信息（详情页）

```typescript
interface GPUDetailInfo {
  worker_id: number; // Worker ID
  worker_name: string; // Worker 名称
  gpu_index: number; // GPU 索引
  gpu_name: string; // GPU 完整名称
  gpu_vendor: string; // GPU 厂商
  vram_total: number; // 总显存（字节）
  vram_allocated: number; // 已分配显存（字节）
  vram_used: number; // 实际使用显存（字节）
  utilization_rate: number; // GPU 利用率（0-100）
}
```

### Token 使用统计

```typescript
interface TokenUsageStats {
  prompt_tokens: number; // Prompt Token 数量
  completion_tokens: number; // Completion Token 数量
  total_tokens: number; // 总 Token 数量
  total_requests: number; // 总请求次数
}
```

### GPU 汇总信息

```typescript
interface GPUSummary {
  gpu_model: string; // GPU 型号
  gpu_vendor: string; // GPU 厂商
  count: number; // 数量
  total_vram: number; // 单个 GPU 总显存
  allocated_vram: number; // 总分配显存
  instances: Array<{
    // 使用该 GPU 的实例
    worker: string;
    gpu_index: number;
    instance_name: string;
  }>;
}
```

---

## 实现要点

### 1. GPU 型号名称提取

**问题**：GPU 设备名称可能很长，如 "NVIDIA GeForce RTX 4090"

**解决方案**：

- 移除不必要的词（如 "GeForce"）
- 保留关键信息：厂商 + 型号
- 示例：
  - 输入：`"NVIDIA GeForce RTX 4090"`
  - 输出：`"NVIDIA RTX 4090"`

### 2. GPU 分组逻辑

**场景**：一个实例使用多个 GPU

**处理逻辑**：

1. 获取实例的所有 `gpu_indexes`
2. 查询对应的 GPU 设备信息
3. 按 GPU 型号分组
4. 为每个型号生成一条 `gpu_display` 记录

**示例**：

```
实例使用 GPU: [0, 1, 2]
- GPU 0: NVIDIA A10
- GPU 1: NVIDIA A10
- GPU 2: NVIDIA A100

结果：
[
  "worker-01 NVIDIA A10 [0,1]",
  "worker-01 NVIDIA A100 [2]"
]
```

### 3. Token 使用量查询优化

**挑战**：频繁查询可能影响性能

**优化方案**：

#### 方案 A：数据库索引

```sql
CREATE INDEX idx_model_usage_model_date
ON model_usages(model_id, date);
```

#### 方案 B：缓存策略

- 使用 Redis 缓存统计结果
- 缓存时间：5-10 分钟
- 缓存键：`model_usage:{model_id}:{hours}`

#### 方案 C：物化视图（PostgreSQL）

```sql
CREATE MATERIALIZED VIEW model_usage_24h AS
SELECT
    model_id,
    SUM(prompt_token_count) as prompt_tokens,
    SUM(completion_token_count) as completion_tokens,
    SUM(request_count) as total_requests
FROM model_usages
WHERE date >= CURRENT_DATE - INTERVAL '1 day'
GROUP BY model_id;
```

### 4. 性能考虑

**列表页优化**：

- 默认 `include_gpu_info=false`，减少不必要的查询
- 使用分页限制数据量
- 考虑批量查询 Worker 信息

**详情页优化**：

- 缓存 GPU 信息（变化不频繁）
- 缓存 Token 统计（可接受短暂延迟）
- 使用异步查询并行获取数据

---

## 前端集成指南

### 列表页集成

#### 显示需求

- 表格列：实例名称、模型名称、状态、GPU 分配、创建时间
- GPU 分配列支持多行显示（一个实例可能有多条 GPU 记录）

#### API 调用

```
GET /api/v1/model-instances?include_gpu_info=true&page=1&perPage=20
```

#### 数据处理

```typescript
// 伪代码
instances.forEach((instance) => {
  // gpu_display 是数组，可能有多条
  instance.gpu_display.forEach((display) => {
    // 显示：worker-01 NVIDIA A10 [0,1]
    renderGPUInfo(display);
  });
});
```

### 详情页集成

#### 显示需求

- GPU 信息表格：节点、GPU 索引、型号、显存、利用率
- Token 使用统计卡片：24 小时数据
- 资源汇总：实例数量、总分配显存

#### API 调用

```
GET /api/v1/stats/models/{id}/details
```

#### 显存格式化

```typescript
// 字节转 GB
function formatBytes(bytes: number): string {
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

// 示例
formatBytes(25769803776); // "24.00 GB"
```

### 统计页集成

#### 显示需求

- GPU 使用汇总（按型号）
- Token 使用趋势图
- 操作类型分布饼图

#### API 调用

```
GET /api/v1/stats/models/{id}/stats?hours=168  // 7 天
```

---

## 错误处理

### 常见错误码

| 状态码 | 说明         | 处理建议               |
| ------ | ------------ | ---------------------- |
| 400    | 请求参数错误 | 检查参数格式和取值范围 |
| 401    | 未授权       | 重新登录获取 Token     |
| 403    | 权限不足     | 需要管理员权限         |
| 404    | 资源不存在   | 检查模型 ID 是否正确   |
| 500    | 服务器错误   | 联系技术支持           |

### 错误响应格式

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model with id 999 not found",
    "details": {}
  }
}
```

---

## 数据单位说明

### 显存单位

所有显存相关字段使用**字节（Bytes）**：

| 字节           | GB    | 示例           |
| -------------- | ----- | -------------- |
| 1,073,741,824  | 1 GB  | 小型 GPU       |
| 8,589,934,592  | 8 GB  | RTX 3070       |
| 25,769,803,776 | 24 GB | RTX 4090 / A10 |
| 85,899,345,920 | 80 GB | A100           |

### 转换公式

```
GB = Bytes / (1024 * 1024 * 1024)
MB = Bytes / (1024 * 1024)
```

---

## 测试场景

### 场景 1：单实例单 GPU

**配置**：

- 1 个实例
- 使用 worker-01 的 GPU 0

**预期结果**：

```json
{
  "gpu_display": ["worker-01 NVIDIA A10 [0]"]
}
```

### 场景 2：单实例多 GPU（同型号）

**配置**：

- 1 个实例
- 使用 worker-01 的 GPU 0, 1, 2

**预期结果**：

```json
{
  "gpu_display": ["worker-01 NVIDIA A10 [0,1,2]"]
}
```

### 场景 3：单实例多 GPU（不同型号）

**配置**：

- 1 个实例
- 使用 worker-01 的 GPU 0, 1 (A10) 和 GPU 2 (A100)

**预期结果**：

```json
{
  "gpu_display": ["worker-01 NVIDIA A10 [0,1]", "worker-01 NVIDIA A100 [2]"]
}
```

### 场景 4：分布式推理（多节点）

**配置**：

- 1 个实例
- 使用 worker-01 的 GPU 0, 1
- 使用 worker-02 的 GPU 0, 1

**预期结果**：

```json
{
  "gpu_display": ["worker-01 NVIDIA A10 [0,1]", "worker-02 NVIDIA A10 [0,1]"]
}
```

---

## 安全考虑

### 权限控制

- 所有统计 API 需要**管理员权限**
- 普通用户只能查看自己创建的模型
- API Key 需要有相应的权限范围

### 数据脱敏

- 不暴露内部 IP 地址（可选）
- 不暴露敏感的系统信息
- Token 使用量可能涉及计费，需要权限控制

---

## 未来扩展

### 1. 实时监控

- WebSocket 推送 GPU 状态变化
- 实时更新 Token 使用量
- 告警通知（GPU 利用率异常、显存不足等）

### 2. 更多统计维度

- 按用户统计 Token 使用量
- 按时间段统计（小时、天、周、月）
- 成本统计（基于 GPU 使用时间和 Token 数量）

### 3. 导出功能

- 导出统计报表（CSV、Excel）
- 生成使用报告（PDF）
- API 调用日志导出

### 4. 预测和建议

- 基于历史数据预测资源需求
- GPU 分配优化建议
- 成本优化建议

---

## 总结

本 API 设计方案完整支持以下业务需求：

✅ **列表页显示**：`worker-01 NVIDIA A10 [0,1]` 格式  
✅ **详情页显示**：完整的 GPU 型号、显存、利用率信息  
✅ **Token 统计**：24 小时内的使用量和请求次数  
✅ **灵活扩展**：支持自定义时间范围和多维度统计

### 核心优势

1. **向后兼容**：扩展现有 API，不影响现有功能
2. **性能优化**：按需加载，支持缓存和索引优化
3. **易于集成**：清晰的数据格式，完整的文档
4. **可扩展性**：预留扩展接口，支持未来需求

### 实施建议

1. **第一阶段**：实现基础 API（列表页和详情页）
2. **第二阶段**：添加统计 API 和性能优化
3. **第三阶段**：实现实时监控和高级功能

---

## 附录

### A. GPUStack 现有数据结构

#### ModelInstance 表

- `id`: 实例 ID
- `model_id`: 模型 ID
- `worker_id`: Worker ID
- `gpu_indexes`: GPU 索引数组
- `computed_resource_claim`: 资源声明（包含 vram 分配）

#### Worker 表

- `id`: Worker ID
- `name`: Worker 名称
- `status.gpu_devices`: GPU 设备数组
  - `index`: GPU 索引
  - `name`: GPU 名称
  - `vendor`: GPU 厂商
  - `memory.total`: 总显存
  - `memory.used`: 已使用显存
  - `core.utilization_rate`: 利用率

#### ModelUsage 表

- `model_id`: 模型 ID
- `date`: 日期
- `prompt_token_count`: Prompt Token 数量
- `completion_token_count`: Completion Token 数量
- `request_count`: 请求次数
- `operation`: 操作类型

### B. 参考资料

- GPUStack 官方文档：https://docs.gpustack.ai
- OpenAI API 规范：https://platform.openai.com/docs/api-reference
- RESTful API 设计最佳实践

---

**文档版本**：v1.0  
**最后更新**：2024-01-26  
**维护者**：Beagle 开发团队
