import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from gpustack.routes import runtime
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    SourceEnum,
)
from gpustack.schemas.workers import Worker


def model(id=1, name="qwen", **kwargs):
    values = {
        "id": id,
        "name": name,
        "source": SourceEnum.HUGGING_FACE,
        "huggingface_repo_id": "Qwen/Qwen2.5-7B",
        "categories": [],
    }
    values.update(kwargs)
    return Model(**values)


def instance(id=101, model_id=1, model_name="qwen", **kwargs):
    values = {
        "id": id,
        "name": f"{model_name}-{id}",
        "model_id": model_id,
        "model_name": model_name,
        "source": SourceEnum.HUGGING_FACE,
        "huggingface_repo_id": "Qwen/Qwen2.5-7B",
        "worker_id": 10,
        "worker_name": "worker-a",
        "worker_ip": "10.0.0.10",
        "port": 18080,
        "ports": [18080],
        "pid": None,
        "gpu_indexes": [0],
        "gpu_addresses": ["cuda:0"],
        "state": ModelInstanceStateEnum.RUNNING,
        "updated_at": datetime(2026, 6, 10, 2, 0, tzinfo=timezone.utc),
    }
    values.update(kwargs)
    return ModelInstance(**values)


def run_get_runtime_model_instances(session=None):
    return asyncio.run(runtime.get_runtime_model_instances(session or SimpleNamespace()))


class QueryResult:
    def __init__(self, items):
        self.items = items

    def all(self):
        return self.items


class RuntimeSession:
    def __init__(self, instances=None, models=None, workers=None):
        self.instances = instances or []
        self.models = models or []
        self.workers = workers or []
        self.statements = []

    async def exec(self, statement):
        self.statements.append(statement)
        entity = statement.column_descriptions[0]["entity"]
        if entity is ModelInstance:
            state_filter = statement.compile().params.get("state_1")
            if state_filter:
                return QueryResult(
                    [
                        instance
                        for instance in self.instances
                        if instance.state in state_filter
                    ]
                )
            return QueryResult(self.instances)
        if entity is Model:
            return QueryResult(self.models)
        if entity is Worker:
            return QueryResult(self.workers)
        return QueryResult([])


def worker(id=10, name="worker-a", ip="10.0.0.20"):
    return Worker(
        id=id,
        name=name,
        hostname=name,
        ip=ip,
        port=10150,
        worker_uuid=f"worker-{id}",
    )


def test_runtime_infers_backend_and_builds_vllm_endpoints():
    session = RuntimeSession(instances=[instance(pid=None)], models=[model(id=1)])

    response = run_get_runtime_model_instances(session)

    item = response.instances[0]
    assert item.model_instance_id == 101
    assert item.model_id == 1
    assert item.model_name == "qwen"
    assert item.backend == BackendEnum.VLLM
    assert item.endpoint == "http://10.0.0.10:18080"
    assert item.health_endpoint == "http://10.0.0.10:18080/v1/models"
    assert item.metrics_endpoint == "http://10.0.0.10:18080/metrics"
    assert item.child_pids == []


def test_runtime_uses_llama_box_health_for_gguf_model():
    session = RuntimeSession(
        instances=[instance(model_name="gguf-model")],
        models=[
            model(
                id=1,
                name="gguf-model",
                huggingface_filename="model.gguf",
            )
        ],
    )

    response = run_get_runtime_model_instances(session)

    item = response.instances[0]
    assert item.backend == BackendEnum.LLAMA_BOX
    assert item.health_endpoint == "http://10.0.0.10:18080/health"


def test_runtime_uses_health_for_vllm_omni_backend():
    session = RuntimeSession(
        instances=[instance(model_name="omni")],
        models=[model(id=1, name="omni", backend=BackendEnum.VLLM_OMNI)],
    )

    response = run_get_runtime_model_instances(session)

    item = response.instances[0]
    assert item.backend == BackendEnum.VLLM_OMNI
    assert item.health_endpoint == "http://10.0.0.10:18080/health"


def test_runtime_returns_empty_endpoints_without_worker_ip_or_port():
    session = RuntimeSession(
        instances=[
            instance(worker_id=None, worker_name=None, worker_ip=None, port=None, ports=[])
        ],
        models=[model(id=1)],
    )

    response = run_get_runtime_model_instances(session)

    item = response.instances[0]
    assert item.worker_ip is None
    assert item.endpoint is None
    assert item.health_endpoint is None
    assert item.metrics_endpoint is None
    assert item.ports == []


def test_runtime_returns_non_running_instance_with_state():
    session = RuntimeSession(
        instances=[
            instance(
                id=102,
                model_name="failed-model",
                state=ModelInstanceStateEnum.ERROR,
            )
        ],
        models=[model(id=1)],
    )

    response = run_get_runtime_model_instances(session)

    assert len(response.instances) == 1
    assert response.instances[0].state == ModelInstanceStateEnum.ERROR


def test_runtime_falls_back_to_worker_ip_and_port_when_ports_empty(monkeypatch):
    async def fail_model_one_by_id(session, id):
        raise AssertionError("Model.one_by_id should not be used")

    async def fail_worker_one_by_id(session, id):
        raise AssertionError("Worker.one_by_id should not be used")

    monkeypatch.setattr(runtime.Model, "one_by_id", fail_model_one_by_id)
    monkeypatch.setattr(runtime.Worker, "one_by_id", fail_worker_one_by_id)
    session = RuntimeSession(
        instances=[
            instance(
                worker_ip=None,
                port=18081,
                ports=[],
            )
        ],
        models=[model(id=1)],
        workers=[worker(id=10, ip="10.0.0.20")],
    )

    response = run_get_runtime_model_instances(session)

    item = response.instances[0]
    assert item.worker_ip == "10.0.0.20"
    assert item.ports == [18081]
    assert item.endpoint == "http://10.0.0.20:18081"


def test_runtime_returns_empty_child_pids_when_psutil_raises(monkeypatch):
    class BrokenProcess:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive):
            raise runtime.psutil.NoSuchProcess(self.pid)

    monkeypatch.setattr(runtime.psutil, "Process", BrokenProcess)
    session = RuntimeSession(
        instances=[instance(pid=1234)],
        models=[model(id=1)],
    )

    response = run_get_runtime_model_instances(session)

    assert response.instances[0].child_pids == []


def test_runtime_filters_pending_by_default():
    session = RuntimeSession(
        instances=[
            instance(id=101, state=ModelInstanceStateEnum.RUNNING),
            instance(id=102, state=ModelInstanceStateEnum.PENDING),
        ],
        models=[model(id=1)],
    )

    response = run_get_runtime_model_instances(session)

    assert [item.model_instance_id for item in response.instances] == [101]
    instance_query = session.statements[0]
    active_states = instance_query.compile().params["state_1"]
    assert ModelInstanceStateEnum.PENDING not in active_states
    assert ModelInstanceStateEnum.RUNNING in active_states
    assert ModelInstanceStateEnum.ERROR in active_states


def test_runtime_batches_models_and_workers(monkeypatch):
    async def fail_model_one_by_id(session, id):
        raise AssertionError("Model.one_by_id should not be used")

    async def fail_worker_one_by_id(session, id):
        raise AssertionError("Worker.one_by_id should not be used")

    monkeypatch.setattr(runtime.Model, "one_by_id", fail_model_one_by_id)
    monkeypatch.setattr(runtime.Worker, "one_by_id", fail_worker_one_by_id)
    session = RuntimeSession(
        instances=[
            instance(id=101, model_id=1, worker_id=10, worker_ip=None),
            instance(id=102, model_id=2, worker_id=11, worker_ip=None),
        ],
        models=[model(id=1), model(id=2, name="qwen-2")],
        workers=[worker(id=10, ip="10.0.0.10"), worker(id=11, ip="10.0.0.11")],
    )

    response = run_get_runtime_model_instances(session)

    assert [item.worker_ip for item in response.instances] == ["10.0.0.10", "10.0.0.11"]
    queried_entities = [
        statement.column_descriptions[0]["entity"] for statement in session.statements
    ]
    assert queried_entities == [ModelInstance, Model, Worker]
