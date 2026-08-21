"""任务 2 步骤 3：Worker 注册/心跳载荷必须上报统一模型存储协议版本。

新 Worker 固定上报 ``model_storage_protocol_version=1``；Server 依此判断是否为
该 Worker 创建 ModelFile/同步/预热任务（版本 0 或缺省不支持）。
"""

from gpustack.schemas.workers import (
    MODEL_STORAGE_PROTOCOL_VERSION,
    SystemInfo,
    Worker,
)
from gpustack.worker.collector import WorkerStatusCollector


def test_protocol_version_constant_is_one():
    assert MODEL_STORAGE_PROTOCOL_VERSION == 1


def test_collector_registration_payload_reports_protocol_version():
    collector = WorkerStatusCollector(
        worker_ip="127.0.0.1",
        worker_name="test-worker",
        worker_port=10150,
        system_info=SystemInfo(),
    )
    worker = collector.collect(initial=True)
    assert isinstance(worker, Worker)
    # 注册/心跳载荷固定上报协议版本 1（不依赖 worker_manager）。
    assert worker.model_storage_protocol_version == MODEL_STORAGE_PROTOCOL_VERSION
    assert worker.model_storage_protocol_version == 1
