import asyncio
from typing import Dict, List, Optional

import psutil
from fastapi import APIRouter
from sqlmodel import col, select

from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    get_backend,
)
from gpustack.schemas.runtime import (
    RuntimeModelInstance,
    RuntimeModelInstancesResponse,
)
from gpustack.schemas.workers import Worker
from gpustack.server.deps import SessionDep

router = APIRouter()

ACTIVE_MODEL_INSTANCE_STATES = [
    ModelInstanceStateEnum.INITIALIZING,
    ModelInstanceStateEnum.STARTING,
    ModelInstanceStateEnum.RUNNING,
    ModelInstanceStateEnum.SCHEDULED,
    ModelInstanceStateEnum.DOWNLOADING,
    ModelInstanceStateEnum.ANALYZING,
    ModelInstanceStateEnum.UNREACHABLE,
    ModelInstanceStateEnum.ERROR,
]

CHILD_PID_STATES = {
    ModelInstanceStateEnum.INITIALIZING,
    ModelInstanceStateEnum.STARTING,
    ModelInstanceStateEnum.RUNNING,
}


@router.get("/model-instances", response_model=RuntimeModelInstancesResponse)
async def get_runtime_model_instances(session: SessionDep):
    instances = await _list_model_instances(session)
    models_by_id = await _models_by_id(session, instances)
    workers_by_id = await _workers_by_id(session, instances)
    child_pids_by_instance_id = await _child_pids_by_instance_id(instances)
    runtime_instances: List[RuntimeModelInstance] = []

    for instance in instances:
        model = models_by_id.get(instance.model_id)
        backend = get_backend(model) if model else None
        worker_ip = _worker_ip(instance, workers_by_id)
        ports = _ports(instance)
        port = ports[0] if ports else None
        endpoint = _endpoint(worker_ip, port)

        runtime_instances.append(
            RuntimeModelInstance(
                model_id=instance.model_id,
                model_instance_id=instance.id,
                model_name=instance.model_name,
                backend=backend,
                worker_id=instance.worker_id,
                worker_name=instance.worker_name,
                worker_ip=worker_ip,
                endpoint=endpoint,
                health_endpoint=_health_endpoint(endpoint, backend),
                metrics_endpoint=f"{endpoint}/metrics" if endpoint else None,
                pid=instance.pid,
                child_pids=child_pids_by_instance_id.get(instance.id, []),
                ports=ports,
                gpu_indexes=instance.gpu_indexes or [],
                gpu_addresses=instance.gpu_addresses or [],
                state=instance.state,
                updated_at=instance.updated_at,
            )
        )

    return RuntimeModelInstancesResponse(instances=runtime_instances)


async def _list_model_instances(session):
    result = await session.exec(
        select(ModelInstance).where(
            col(ModelInstance.state).in_(ACTIVE_MODEL_INSTANCE_STATES)
        )
    )
    return result.all()


async def _models_by_id(
    session, instances: List[ModelInstance]
) -> Dict[int, Model]:
    model_ids = {instance.model_id for instance in instances if instance.model_id}
    if not model_ids:
        return {}

    result = await session.exec(select(Model).where(col(Model.id).in_(model_ids)))
    return {model.id: model for model in result.all() if model.id is not None}


async def _workers_by_id(
    session, instances: List[ModelInstance]
) -> Dict[int, Worker]:
    worker_ids = {
        instance.worker_id
        for instance in instances
        if not instance.worker_ip and instance.worker_id
    }
    if not worker_ids:
        return {}

    result = await session.exec(select(Worker).where(col(Worker.id).in_(worker_ids)))
    return {worker.id: worker for worker in result.all() if worker.id is not None}


def _worker_ip(
    instance: ModelInstance, workers_by_id: Dict[int, Worker]
) -> Optional[str]:
    if instance.worker_ip:
        return instance.worker_ip

    if not instance.worker_id:
        return None

    worker = workers_by_id.get(instance.worker_id)
    return worker.ip if worker else None


def _ports(instance: ModelInstance) -> List[int]:
    ports = instance.ports or []
    if ports:
        return ports
    if instance.port:
        return [instance.port]
    return []


def _endpoint(worker_ip: Optional[str], port: Optional[int]) -> Optional[str]:
    if not worker_ip or not port:
        return None
    return f"http://{worker_ip}:{port}"


def _health_endpoint(endpoint: Optional[str], backend: Optional[str]) -> Optional[str]:
    if not endpoint:
        return None

    if backend in (BackendEnum.LLAMA_BOX, BackendEnum.VLLM_OMNI):
        return f"{endpoint}/health"

    return f"{endpoint}/v1/models"


async def _child_pids_by_instance_id(
    instances: List[ModelInstance],
) -> Dict[int, List[int]]:
    tasks = [
        asyncio.to_thread(_child_pids, instance.pid)
        if instance.pid and instance.state in CHILD_PID_STATES
        else _empty_child_pids()
        for instance in instances
    ]
    child_pids = await asyncio.gather(*tasks)
    return {
        instance.id: pids
        for instance, pids in zip(instances, child_pids)
        if instance.id is not None
    }


async def _empty_child_pids() -> List[int]:
    return []


def _child_pids(pid: Optional[int]) -> List[int]:
    if not pid:
        return []

    try:
        process = psutil.Process(pid)
        return [child.pid for child in process.children(recursive=True)]
    except psutil.Error:
        return []
