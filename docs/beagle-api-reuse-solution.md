# Beagle API 复用方案 - 基于 GPUStack 现有 API

## 概述

本文档说明如何**不修改 GPUStack 后端代码**，仅通过复用现有 API 来实现 Beagle 平台的三个核心需求：

1. **列表页显示 GPU 分配信息**：`worker-01 NVIDIA A10 [0,1]`
2. **详情页显示完整 GPU 信息**：GPU 型号、显存、利用率等
3. **Token 使用量统计**：24 小时内的 Token 使用量

**核心优势**： 

- ✅ 零后端代码改动
- ✅ 前端页面完全独立
- ✅ 不影响 GPUStack 现有功能
- ✅ 灵活可控，按需加载数据

---

## 方案架构

### 数据流程

```
前端页面
  ↓
调用 GPUStack 现有 API
  ↓
前端组装和格式化数据
  ↓
展示给用户
```

### API 依赖关系

```
需求 1: GPU 分配信息
  → GET /api/v1/model-instances
  → GET /api/v1/workers/{id}

需求 2: GPU 详细信息
  → GET /api/v1/model-instances?model_id={id}
  → GET /api/v1/workers/{id}

需求 3: Token 使用统计
  → GET /api/v1/dashboard/usage/stats
```

---

## 需求 1：列表页 GPU 分配信息

### 业务需求

在模型实例列表页显示 GPU 分配情况，格式：`worker-01 NVIDIA A10 [0,1]`

### 实现方案

#### 步骤 1：获取模型实例列表

**API**: `GET /api/v1/model-instances`

**请求参数**:

```json
{
  "page": 1,
  "perPage": 20
}
```

**响应示例**:

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
      "gpu_indexes": [0, 1],
      "computed_resource_claim": {
        "vram": { "0": 8589934592, "1": 8589934592 }
      }
    }
  ]
}
```

#### 步骤 2：获取 Worker 信息

**API**: `GET /api/v1/workers/{worker_id}`

**响应示例**:

```json
{
  "id": 1,
  "name": "worker-01",
  "ip": "192.168.1.10",
  "status": {
    "gpu_devices": [
      {
        "index": 0,
        "name": "NVIDIA A10",
        "vendor": "NVIDIA",
        "memory": {
          "total": 25769803776,
          "used": 7516192768
        },
        "core": {
          "utilization_rate": 85.5
        }
      },
      {
        "index": 1,
        "name": "NVIDIA A10",
        "vendor": "NVIDIA",
        "memory": {
          "total": 25769803776,
          "used": 7200000000
        }
      }
    ]
  }
}
```


#### 步骤 3：前端组装显示文本

**伪代码**:

```typescript
async function getGPUDisplayText(instance: ModelInstance): Promise<string[]> {
  // 1. 获取 worker 信息
  const worker = await fetch(`/api/v1/workers/${instance.worker_id}`);
  
  // 2. 按 GPU 索引提取设备信息
  const gpuDevices = instance.gpu_indexes.map(index => 
    worker.status.gpu_devices.find(gpu => gpu.index === index)
  );
  
  // 3. 按 GPU 型号分组
  const groupedByModel = groupBy(gpuDevices, gpu => gpu.name);
  
  // 4. 生成显示文本
  const displayTexts = [];
  for (const [gpuModel, gpus] of Object.entries(groupedByModel)) {
    const indexes = gpus.map(gpu => gpu.index).sort();
    displayTexts.push(`${worker.name} ${gpuModel} [${indexes.join(',')}]`);
  }
  
  return displayTexts;
}

// 示例输出
// ["worker-01 NVIDIA A10 [0,1]"]
```

### 性能优化建议

1. **批量查询 Worker**：收集所有 `worker_id`，一次性查询
2. **缓存 Worker 信息**：Worker 信息变化不频繁，可缓存 5-10 分钟
3. **按需加载**：列表页默认不显示 GPU 信息，点击展开时再加载

---

## 需求 2：详情页 GPU 详细信息

### 业务需求

在模型详情页显示：

- GPU 型号、厂商
- 显存总量、已分配显存、实际使用显存
- GPU 利用率

### 实现方案

#### 步骤 1：获取模型的所有实例

**API**: `GET /api/v1/model-instances?model_id={id}`

**响应示例**:

```json
{
  "items": [
    {
      "id": 1,
      "model_id": 1,
      "worker_id": 1,
      "gpu_indexes": [0, 1],
      "computed_resource_claim": {
        "vram": { "0": 8589934592, "1": 8589934592 }
      }
    }
  ]
}
```


#### 步骤 2：获取每个 Worker 的 GPU 详细信息

**API**: `GET /api/v1/workers/{worker_id}`（同需求 1）

#### 步骤 3：前端组装详细信息

**伪代码**:

```typescript
interface GPUDetail {
  worker_id: number;
  worker_name: string;
  gpu_index: number;
  gpu_name: string;
  gpu_vendor: string;
  vram_total: number;        // 总显存（字节）
  vram_allocated: number;    // 已分配显存（字节）
  vram_used: number;         // 实际使用显存（字节）
  utilization_rate: number;  // GPU 利用率（0-100）
}

async function getModelGPUDetails(modelId: number): Promise<GPUDetail[]> {
  // 1. 获取模型的所有实例
  const instances = await fetch(`/api/v1/model-instances?model_id=${modelId}`);
  
  const gpuDetails: GPUDetail[] = [];
  
  for (const instance of instances.items) {
    // 2. 获取 worker 信息
    const worker = await fetch(`/api/v1/workers/${instance.worker_id}`);
    
    // 3. 遍历实例使用的每个 GPU
    for (const gpuIndex of instance.gpu_indexes) {
      const gpuDevice = worker.status.gpu_devices.find(
        gpu => gpu.index === gpuIndex
      );
      
      // 4. 从 computed_resource_claim 获取分配的显存
      const vramAllocated = instance.computed_resource_claim.vram?.[gpuIndex] || 0;
      
      gpuDetails.push({
        worker_id: worker.id,
        worker_name: worker.name,
        gpu_index: gpuIndex,
        gpu_name: gpuDevice.name,
        gpu_vendor: gpuDevice.vendor,
        vram_total: gpuDevice.memory.total,
        vram_allocated: vramAllocated,
        vram_used: gpuDevice.memory.used,
        utilization_rate: gpuDevice.core.utilization_rate || 0
      });
    }
  }
  
  return gpuDetails;
}
```

### 显示格式化

**显存转换**:

```typescript
function formatBytes(bytes: number): string {
  const gb = bytes / (1024 * 1024 * 1024);
  return `${gb.toFixed(2)} GB`;
}

// 示例
formatBytes(25769803776);  // "24.00 GB"
formatBytes(8589934592);   // "8.00 GB"
```

**表格展示示例**:

```
| Worker    | GPU 索引 | GPU 型号      | 总显存   | 已分配   | 实际使用 | 利用率 |
|-----------|---------|--------------|---------|---------|---------|--------|
| worker-01 | 0       | NVIDIA A10   | 24.00GB | 8.00GB  | 7.00GB  | 85.5%  |
| worker-01 | 1       | NVIDIA A10   | 24.00GB | 8.00GB  | 6.71GB  | 78.2%  |
```

---

## 需求 3：Token 使用量统计

### 业务需求

显示模型在 24 小时内的 Token 使用量：

- Prompt Tokens
- Completion Tokens
- 总请求数

### 实现方案

#### 直接使用现有 API

**API**: `GET /api/v1/dashboard/usage/stats`

**请求参数**:

```json
{
  "start_date": "2024-01-25",  // 昨天
  "end_date": "2024-01-26",    // 今天
  "model_ids": [1]             // 指定模型 ID
}
```

**响应示例**:

```json
{
  "api_request_history": [
    {
      "timestamp": 1706140800,
      "value": 1250
    }
  ],
  "prompt_token_history": [
    {
      "timestamp": 1706140800,
      "value": 125000
    }
  ],
  "completion_token_history": [
    {
      "timestamp": 1706140800,
      "value": 375000
    }
  ]
}
```

#### 前端数据处理

**伪代码**:

```typescript
interface TokenUsage24h {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  total_requests: number;
}

async function getTokenUsage24h(modelId: number): Promise<TokenUsage24h> {
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  
  const params = {
    start_date: yesterday.toISOString().split('T')[0],
    end_date: today.toISOString().split('T')[0],
    model_ids: [modelId]
  };
  
  const stats = await fetch('/api/v1/dashboard/usage/stats', { params });
  
  // 汇总所有时间点的数据
  const promptTokens = stats.prompt_token_history.reduce(
    (sum, item) => sum + item.value, 0
  );
  const completionTokens = stats.completion_token_history.reduce(
    (sum, item) => sum + item.value, 0
  );
  const totalRequests = stats.api_request_history.reduce(
    (sum, item) => sum + item.value, 0
  );
  
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
    total_requests: totalRequests
  };
}
```

### 灵活的时间范围

API 支持自定义时间范围，可以轻松扩展：

```typescript
// 7 天统计
const last7Days = {
  start_date: '2024-01-19',
  end_date: '2024-01-26',
  model_ids: [1]
};

// 30 天统计
const last30Days = {
  start_date: '2023-12-27',
  end_date: '2024-01-26',
  model_ids: [1]
};
```

---

## 完整示例：模型详情页

### 页面需求

展示单个模型的完整信息：

1. 基本信息（名称、状态等）
2. GPU 详细信息表格
3. 24 小时 Token 使用统计

### 实现代码

```typescript
interface ModelDetailPage {
  model: Model;
  gpu_details: GPUDetail[];
  token_usage_24h: TokenUsage24h;
}

async function loadModelDetailPage(modelId: number): Promise<ModelDetailPage> {
  // 并行请求所有数据
  const [model, instances, usageStats] = await Promise.all([
    fetch(`/api/v1/models/${modelId}`),
    fetch(`/api/v1/model-instances?model_id=${modelId}`),
    fetch(`/api/v1/dashboard/usage/stats`, {
      params: {
        start_date: getYesterday(),
        end_date: getToday(),
        model_ids: [modelId]
      }
    })
  ]);
  
  // 获取所有 worker 信息（去重）
  const workerIds = [...new Set(instances.items.map(i => i.worker_id))];
  const workers = await Promise.all(
    workerIds.map(id => fetch(`/api/v1/workers/${id}`))
  );
  const workerMap = new Map(workers.map(w => [w.id, w]));
  
  // 组装 GPU 详细信息
  const gpuDetails: GPUDetail[] = [];
  for (const instance of instances.items) {
    const worker = workerMap.get(instance.worker_id);
    
    for (const gpuIndex of instance.gpu_indexes) {
      const gpuDevice = worker.status.gpu_devices.find(
        gpu => gpu.index === gpuIndex
      );
      
      gpuDetails.push({
        worker_id: worker.id,
        worker_name: worker.name,
        gpu_index: gpuIndex,
        gpu_name: gpuDevice.name,
        gpu_vendor: gpuDevice.vendor,
        vram_total: gpuDevice.memory.total,
        vram_allocated: instance.computed_resource_claim.vram?.[gpuIndex] || 0,
        vram_used: gpuDevice.memory.used,
        utilization_rate: gpuDevice.core.utilization_rate || 0
      });
    }
  }
  
  // 汇总 Token 使用量
  const tokenUsage24h = {
    prompt_tokens: sum(usageStats.prompt_token_history),
    completion_tokens: sum(usageStats.completion_token_history),
    total_tokens: sum(usageStats.prompt_token_history) + 
                  sum(usageStats.completion_token_history),
    total_requests: sum(usageStats.api_request_history)
  };
  
  return {
    model,
    gpu_details: gpuDetails,
    token_usage_24h: tokenUsage24h
  };
}
```


---

## 性能优化策略

### 1. 批量查询优化

**问题**：多个实例可能在同一个 Worker 上，重复查询浪费资源

**解决方案**：

```typescript
// 收集所有唯一的 worker_id
const workerIds = [...new Set(instances.map(i => i.worker_id))];

// 批量并行查询
const workers = await Promise.all(
  workerIds.map(id => fetch(`/api/v1/workers/${id}`))
);

// 建立 Map 快速查找
const workerMap = new Map(workers.map(w => [w.id, w]));
```

### 2. 缓存策略

**Worker 信息缓存**（变化不频繁）:

```typescript
const workerCache = new Map<number, { data: Worker, expiry: number }>();

async function getWorkerWithCache(workerId: number): Promise<Worker> {
  const cached = workerCache.get(workerId);
  const now = Date.now();
  
  if (cached && cached.expiry > now) {
    return cached.data;
  }
  
  const worker = await fetch(`/api/v1/workers/${workerId}`);
  workerCache.set(workerId, {
    data: worker,
    expiry: now + 5 * 60 * 1000  // 缓存 5 分钟
  });
  
  return worker;
}
```

**Token 统计缓存**（可接受短暂延迟）:

```typescript
// 缓存 1 分钟，减少频繁查询
const usageStatsCache = new Map<string, { data: any, expiry: number }>();

async function getUsageStatsWithCache(
  modelId: number, 
  startDate: string, 
  endDate: string
): Promise<any> {
  const cacheKey = `${modelId}-${startDate}-${endDate}`;
  const cached = usageStatsCache.get(cacheKey);
  const now = Date.now();
  
  if (cached && cached.expiry > now) {
    return cached.data;
  }
  
  const stats = await fetch('/api/v1/dashboard/usage/stats', {
    params: { model_ids: [modelId], start_date: startDate, end_date: endDate }
  });
  
  usageStatsCache.set(cacheKey, {
    data: stats,
    expiry: now + 60 * 1000  // 缓存 1 分钟
  });
  
  return stats;
}
```

### 3. 按需加载

**列表页懒加载**:

```typescript
// 默认不加载 GPU 信息
<Table>
  {instances.map(instance => (
    <Row>
      <Cell>{instance.name}</Cell>
      <Cell>
        <Button onClick={() => loadGPUInfo(instance.id)}>
          查看 GPU 分配
        </Button>
      </Cell>
    </Row>
  ))}
</Table>
```

**详情页分步加载**:

```typescript
// 先显示基本信息，再异步加载详细数据
async function loadDetailPage(modelId: number) {
  // 1. 立即显示基本信息
  const model = await fetch(`/api/v1/models/${modelId}`);
  renderBasicInfo(model);
  
  // 2. 异步加载 GPU 详情
  const gpuDetails = await loadGPUDetails(modelId);
  renderGPUTable(gpuDetails);
  
  // 3. 异步加载 Token 统计
  const tokenUsage = await loadTokenUsage(modelId);
  renderTokenStats(tokenUsage);
}
```

---

## API 调用频率建议

| 场景           | API                          | 调用频率       | 缓存时间 |
|----------------|------------------------------|---------------|---------|
| 列表页初始加载  | model-instances              | 按需          | 无      |
| 列表页 GPU 信息 | workers                      | 按需/懒加载    | 5 分钟  |
| 详情页基本信息  | models, model-instances      | 页面加载时     | 无      |
| 详情页 GPU 信息 | workers                      | 页面加载时     | 5 分钟  |
| Token 统计     | dashboard/usage/stats        | 页面加载时     | 1 分钟  |
| 实时监控       | workers                      | 轮询 30-60 秒 | 无      |

---

## 错误处理

### 常见场景

**Worker 不存在或离线**:

```typescript
async function getWorkerSafely(workerId: number): Promise<Worker | null> {
  try {
    return await fetch(`/api/v1/workers/${workerId}`);
  } catch (error) {
    if (error.status === 404) {
      console.warn(`Worker ${workerId} not found`);
      return null;
    }
    throw error;
  }
}

// 使用时处理 null
const worker = await getWorkerSafely(instance.worker_id);
if (!worker) {
  return "Worker 离线";
}
```

**GPU 设备信息缺失**:

```typescript
const gpuDevice = worker.status.gpu_devices?.find(
  gpu => gpu.index === gpuIndex
);

if (!gpuDevice) {
  return {
    gpu_name: "未知",
    vram_total: 0,
    vram_used: 0,
    utilization_rate: 0
  };
}
```

**Token 统计数据为空**:

```typescript
const tokenUsage = {
  prompt_tokens: sum(stats.prompt_token_history) || 0,
  completion_tokens: sum(stats.completion_token_history) || 0,
  total_requests: sum(stats.api_request_history) || 0
};

// 显示友好提示
if (tokenUsage.total_requests === 0) {
  return "暂无使用记录";
}
```

---

## 测试场景

### 场景 1：单实例单 GPU

**数据**:
- 1 个实例，使用 worker-01 的 GPU 0

**预期输出**:
```
GPU 分配: worker-01 NVIDIA A10 [0]
```

### 场景 2：单实例多 GPU（同型号）

**数据**:
- 1 个实例，使用 worker-01 的 GPU 0, 1, 2

**预期输出**:
```
GPU 分配: worker-01 NVIDIA A10 [0,1,2]
```

### 场景 3：单实例多 GPU（不同型号）

**数据**:
- 1 个实例，使用 worker-01 的 GPU 0, 1 (A10) 和 GPU 2 (A100)

**预期输出**:
```
GPU 分配:
  - worker-01 NVIDIA A10 [0,1]
  - worker-01 NVIDIA A100 [2]
```

### 场景 4：分布式推理（多节点）

**数据**:
- 1 个实例，使用 worker-01 的 GPU 0, 1 和 worker-02 的 GPU 0, 1

**预期输出**:
```
GPU 分配:
  - worker-01 NVIDIA A10 [0,1]
  - worker-02 NVIDIA A10 [0,1]
```

### 场景 5：多实例共享 Worker

**数据**:
- 实例 A 使用 worker-01 的 GPU 0
- 实例 B 使用 worker-01 的 GPU 1

**优化**:
- 只查询一次 worker-01 的信息
- 使用缓存避免重复请求

---

## 与原方案对比

| 维度           | 原方案（新增 API）        | 复用方案（本方案）      |
|----------------|--------------------------|------------------------|
| 后端改动       | 需要新增 3 个 API 端点    | 零改动                 |
| 前端复杂度     | 低（后端已组装数据）      | 中（需要前端组装）      |
| 性能           | 优（后端优化查询）        | 良（需要前端优化）      |
| 灵活性         | 低（固定数据格式）        | 高（前端自由控制）      |
| 维护成本       | 高（需要维护新 API）      | 低（依赖现有 API）      |
| 上线时间       | 长（需要开发测试）        | 短（立即可用）          |
| 风险           | 中（可能影响现有系统）    | 低（完全独立）          |

---

## 实施建议

### 第一阶段：快速上线（1-2 天）

1. 实现基础功能，不做优化
2. 列表页显示 GPU 分配信息
3. 详情页显示 GPU 详细信息和 Token 统计

### 第二阶段：性能优化（3-5 天）

1. 添加批量查询优化
2. 实现缓存策略
3. 添加按需加载

### 第三阶段：体验优化（可选）

1. 添加加载状态和骨架屏
2. 实现实时数据刷新
3. 添加错误重试机制

---

## 总结

### 核心优势

✅ **零后端改动**：不需要修改 GPUStack 代码  
✅ **快速上线**：利用现有 API，立即可用  
✅ **完全独立**：前端页面不依赖 GPUStack 前端  
✅ **灵活可控**：可以自定义数据格式和展示方式  
✅ **低风险**：不影响现有系统稳定性

### 适用场景

- ✅ 需要快速上线
- ✅ 不想修改 GPUStack 代码
- ✅ 前端有一定开发能力
- ✅ 可以接受前端组装数据的复杂度

### 不适用场景

- ❌ 对性能要求极高（需要毫秒级响应）
- ❌ 前端开发资源不足
- ❌ 需要复杂的数据聚合和计算

---

## 附录

### A. 辅助函数

```typescript
// 按字段分组
function groupBy<T>(array: T[], keyFn: (item: T) => string): Record<string, T[]> {
  return array.reduce((result, item) => {
    const key = keyFn(item);
    if (!result[key]) result[key] = [];
    result[key].push(item);
    return result;
  }, {} as Record<string, T[]>);
}

// 数组求和
function sum(array: { value: number }[]): number {
  return array.reduce((total, item) => total + item.value, 0);
}

// 获取昨天日期
function getYesterday(): string {
  const date = new Date();
  date.setDate(date.getDate() - 1);
  return date.toISOString().split('T')[0];
}

// 获取今天日期
function getToday(): string {
  return new Date().toISOString().split('T')[0];
}
```

### B. GPUStack 现有 API 参考

| API                              | 说明                  | 权限要求 |
|----------------------------------|-----------------------|---------|
| GET /api/v1/models               | 获取模型列表           | 用户    |
| GET /api/v1/models/{id}          | 获取模型详情           | 用户    |
| GET /api/v1/model-instances      | 获取模型实例列表       | 管理员  |
| GET /api/v1/workers              | 获取 Worker 列表      | 管理员  |
| GET /api/v1/workers/{id}         | 获取 Worker 详情      | 管理员  |
| GET /api/v1/dashboard/usage/stats| 获取使用统计           | 管理员  |

---

**文档版本**：v1.0  
**最后更新**：2024-01-26  
**维护者**：Beagle 开发团队
