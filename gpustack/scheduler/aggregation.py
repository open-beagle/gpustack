from typing import Dict, Iterable, List, Optional, Tuple

from gpustack.policies.base import ModelInstanceScheduleCandidate
from gpustack.policies.utils import get_worker_allocatable_resource
from gpustack.schemas.models import (
    ComputedResourceClaim,
    ModelInstanceSubordinateWorker,
)
from gpustack.schemas.workers import Worker
from sqlalchemy.ext.asyncio import AsyncEngine


def _claim_vram(claim: ComputedResourceClaim, gpu_index: int) -> int:
    if claim.vram is None:
        return 0
    return int(claim.vram.get(gpu_index, claim.vram.get(str(gpu_index), 0)) or 0)


def _gpu_total(worker: Worker, gpu_index: int) -> int:
    for gpu in worker.status.gpu_devices or []:
        if gpu.index == gpu_index and gpu.memory and gpu.memory.total:
            return int(gpu.memory.total)
    return 0


async def projected_load_percent(
    engine: AsyncEngine,
    worker: Worker,
    gpu_indexes: Optional[List[int]],
    claim: ComputedResourceClaim,
) -> Optional[float]:
    allocatable = await get_worker_allocatable_resource(engine, worker)
    if gpu_indexes:
        projected = []
        for gpu_index in gpu_indexes:
            total = _gpu_total(worker, gpu_index)
            if total <= 0:
                return None
            used = max(total - int(allocatable.vram.get(gpu_index, 0)), 0)
            projected.append((used + _claim_vram(claim, gpu_index)) / total * 100)
        return max(projected) if projected else None

    if not worker.status or not worker.status.memory or not worker.status.memory.total:
        return None
    total = int(worker.status.memory.total)
    used = max(total - int(allocatable.ram), 0)
    return (used + int(claim.ram or 0)) / total * 100


async def filter_by_aggregation_rate(
    engine: AsyncEngine,
    candidates: List[ModelInstanceScheduleCandidate],
    workers: Iterable[Worker],
    aggregation_rate: float,
) -> Tuple[List[ModelInstanceScheduleCandidate], Dict[int, float]]:
    worker_map = {worker.id: worker for worker in workers}
    accepted = []
    projected_loads: Dict[int, float] = {}
    for candidate in candidates:
        loads = []
        main_load = await projected_load_percent(
            engine,
            candidate.worker,
            candidate.gpu_indexes,
            candidate.computed_resource_claim,
        )
        if main_load is None:
            continue
        loads.append(main_load)
        valid = True
        for subordinate in candidate.subordinate_workers or []:
            worker = worker_map.get(subordinate.worker_id)
            if worker is None:
                valid = False
                break
            load = await _subordinate_projected_load(engine, worker, subordinate)
            if load is None:
                valid = False
                break
            loads.append(load)
        projected = max(loads)
        projected_loads[id(candidate)] = round(projected, 2)
        if valid and projected <= aggregation_rate:
            accepted.append(candidate)
    return accepted, projected_loads


async def _subordinate_projected_load(
    engine: AsyncEngine,
    worker: Worker,
    subordinate: ModelInstanceSubordinateWorker,
) -> Optional[float]:
    return await projected_load_percent(
        engine,
        worker,
        subordinate.gpu_indexes,
        subordinate.computed_resource_claim,
    )


def candidate_snapshot(
    candidate: ModelInstanceScheduleCandidate,
    projected_load: Optional[float],
) -> dict:
    return {
        "worker_id": candidate.worker.id,
        "worker_name": candidate.worker.name,
        "worker_ip": candidate.worker.ip,
        "gpu_indexes": candidate.gpu_indexes or [],
        "gpu_addresses": candidate.gpu_addresses or [],
        "requested_resources": candidate.computed_resource_claim.model_dump(
            mode="json"
        ),
        "score": candidate.score,
        "projected_load_percent": projected_load,
        "subordinate_workers": [
            {
                "worker_id": worker.worker_id,
                "worker_name": worker.worker_name,
                "worker_ip": worker.worker_ip,
                "gpu_indexes": worker.gpu_indexes or [],
                "gpu_addresses": worker.gpu_addresses or [],
                "requested_resources": worker.computed_resource_claim.model_dump(
                    mode="json"
                ),
            }
            for worker in candidate.subordinate_workers or []
        ],
    }
