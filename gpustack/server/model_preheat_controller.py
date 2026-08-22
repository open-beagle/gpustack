import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheats import (
    ModelPreheatDesiredStateEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatTask,
    ModelPreheatTaskLock,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
    is_terminal_task,
)
from gpustack.schemas.workers import (
    MODEL_STORAGE_PROTOCOL_VERSION,
    Worker,
    WorkerStateEnum,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalInventoryProbeResult:
    worker_uuid: str
    state: str
    error_code: str | None = None
    source: str | None = None


class LocalInventoryProbe(Protocol):
    async def probe(
        self, task: ModelPreheatTask, worker_uuids: list[str]
    ) -> dict[str, LocalInventoryProbeResult]: ...


class MissingLocalInventoryProbe:
    async def probe(self, task, worker_uuids):
        return {
            worker_uuid: LocalInventoryProbeResult(worker_uuid, "missing")
            for worker_uuid in worker_uuids
        }


class StrictS3ReadyProbe:
    """旧构造参数兼容类。

    统一 Artifact 是否可用由创建时精确库存绑定及 Worker 读取 Manifest 决定，
    Server 不再探测 ready.json。
    """

    def __init__(self, config):
        self.config = config

    async def probe(self, task):
        return None


class ModelPreheatController:
    def __init__(
        self,
        engine,
        ready_probe=None,
        inventory_probe=None,
        interval=15,
        s3_inventory=None,
    ):
        self._engine = engine
        self._inventory_probe = inventory_probe or MissingLocalInventoryProbe()
        self._interval = interval

    async def start(self):
        while True:
            await self.reconcile_all()
            try:
                async for event in ModelPreheatWorkerTask.subscribe(self._engine):
                    if event.data is not None and event.data.task_id is not None:
                        await self.reconcile_task(event.data.task_id)
                    break
            except Exception:
                logger.exception("模型预热事件协调失败")
            await asyncio.sleep(self._interval)

    async def reconcile_all(self):
        async with AsyncSession(self._engine) as session:
            task_ids = (
                await session.exec(
                    select(ModelPreheatTask.id).where(
                        ModelPreheatTask.desired_state
                        == ModelPreheatDesiredStateEnum.RUNNING,
                        ModelPreheatTask.execution_state.not_in(
                            [
                                ModelPreheatExecutionStateEnum.READY,
                                ModelPreheatExecutionStateEnum.PARTIAL,
                                ModelPreheatExecutionStateEnum.ERROR,
                                ModelPreheatExecutionStateEnum.CANCELED,
                            ]
                        ),
                    )
                )
            ).all()
        for task_id in task_ids:
            await self.reconcile_task(task_id)

    async def reconcile_task(self, task_id: int):
        try:
            async with AsyncSession(self._engine) as session:
                await self._reconcile(session, task_id)
                task = await session.get(
                    ModelPreheatTask, task_id, populate_existing=True
                )
                if task is not None and is_terminal_task(task):
                    await session.exec(
                        delete(ModelPreheatTaskLock).where(
                            ModelPreheatTaskLock.task_id == task_id
                        )
                    )
                await session.commit()
        except (IntegrityError, OperationalError):
            return

    async def _reconcile(self, session, task_id):
        task = await session.get(ModelPreheatTask, task_id)
        if (
            task is None
            or task.desired_state != ModelPreheatDesiredStateEnum.RUNNING
            or is_terminal_task(task)
        ):
            return

        all_workers = await _current_workers(session)
        targets = {
            worker_uuid: all_workers[worker_uuid]
            for worker_uuid in task.target_worker_uuids
            if worker_uuid in all_workers
        }
        removed = sorted(set(task.target_worker_uuids) - set(targets))
        task.removed_target_worker_uuids = removed
        await _mark_removed_children(session, task, removed)
        if not targets:
            _finish(task, ModelPreheatExecutionStateEnum.ERROR, "no_available_targets")
            return

        children = (
            await session.exec(
                select(ModelPreheatWorkerTask)
                .where(
                    ModelPreheatWorkerTask.task_id == task.id,
                    ModelPreheatWorkerTask.parent_attempt == task.attempt,
                )
                .order_by(ModelPreheatWorkerTask.id)
            )
        ).all()
        _refresh_child_registrations(task, children, all_workers)
        seed_tasks = [
            child
            for child in children
            if child.role == ModelPreheatWorkerTaskRoleEnum.SEED
        ]
        distribution_tasks = [
            child
            for child in children
            if child.role == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE
        ]

        if not children:
            expected_attempt = task.attempt
            expected_state = task.execution_state
            inventory = await self._inventory_probe.probe(task, sorted(all_workers))
            task = await _reload_running_task(
                session, task.id, expected_attempt, expected_state
            )
            if task is None:
                return
            all_workers = await _current_workers(session)
            targets = {
                worker_uuid: all_workers[worker_uuid]
                for worker_uuid in task.target_worker_uuids
                if worker_uuid in all_workers
            }
            if not targets:
                await _cas_parent_update(
                    session,
                    task,
                    execution_state=ModelPreheatExecutionStateEnum.ERROR,
                    state_message="no_available_targets",
                    progress=100,
                    finished_at=datetime.now(timezone.utc),
                )
                return
            valid_targets = sorted(
                worker_uuid
                for worker_uuid in targets
                if getattr(inventory.get(worker_uuid), "state", None) == "valid"
            )
            if len(valid_targets) == len(targets):
                await _cas_parent_update(
                    session,
                    task,
                    execution_state=ModelPreheatExecutionStateEnum.READY,
                    local_cache_hit_worker_uuids=valid_targets,
                    transfer_source="current_node",
                    state_message=None,
                    progress=100,
                    finished_at=datetime.now(timezone.utc),
                )
                return
            target_candidates = sorted(
                worker_uuid
                for worker_uuid in targets
                if getattr(inventory.get(worker_uuid), "state", None) == "candidate"
            )
            peer_candidates = sorted(
                worker_uuid
                for worker_uuid in set(all_workers) - set(targets)
                if getattr(inventory.get(worker_uuid), "state", None) == "candidate"
            )
            if target_candidates or peer_candidates:
                seed_uuid = (target_candidates or peer_candidates)[0]
                worker = all_workers[seed_uuid]
                if not await _cas_parent_update(
                    session,
                    task,
                    execution_state=ModelPreheatExecutionStateEnum.STAGING,
                    local_cache_hit_worker_uuids=valid_targets,
                    seed_worker_uuid=worker.worker_uuid,
                    seed_worker_id=worker.id,
                    seed_source=getattr(
                        inventory.get(worker.worker_uuid), "source", None
                    ),
                    started_at=task.started_at or datetime.now(timezone.utc),
                ):
                    return
                _add_seed_child(session, task, worker)
                return
            if task.artifact_id and task.s3_manifest_path:
                if not await _cas_parent_update(
                    session,
                    task,
                    execution_state=ModelPreheatExecutionStateEnum.DISTRIBUTING,
                    local_cache_hit_worker_uuids=valid_targets,
                    transfer_source="s3",
                    transfer_profile_id=task.s3_profile_id,
                ):
                    return
                _create_distribution_tasks(session, task, targets, set(valid_targets))
                return
            seed_uuid = (
                task.seed_worker_uuid
                if task.seed_worker_uuid in targets
                else sorted(targets)[0]
            )
            worker = targets[seed_uuid]
            if not await _cas_parent_update(
                session,
                task,
                execution_state=ModelPreheatExecutionStateEnum.STAGING,
                local_cache_hit_worker_uuids=valid_targets,
                seed_worker_uuid=worker.worker_uuid,
                seed_worker_id=worker.id,
                seed_source=getattr(inventory.get(worker.worker_uuid), "source", None),
                started_at=task.started_at or datetime.now(timezone.utc),
            ):
                return
            _add_seed_child(session, task, worker)
            return

        active_seed = next(
            (
                child
                for child in seed_tasks
                if child.state != ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED
            ),
            None,
        )
        if active_seed is not None and not distribution_tasks:
            if active_seed.state == ModelPreheatWorkerTaskStateEnum.ERROR:
                _replace_seed(session, task, seed_tasks, all_workers, active_seed)
                return
            if active_seed.state != ModelPreheatWorkerTaskStateEnum.READY:
                return
            result = active_seed.resumable_cursor or {}
            if (
                not task.artifact_id
                or not task.s3_manifest_path
                or result.get("artifact_id") != task.artifact_id
            ):
                _replace_seed(session, task, seed_tasks, all_workers, active_seed)
                return
            local_hits = set(task.local_cache_hit_worker_uuids) & set(targets)
            if (
                active_seed.worker_uuid in targets
                and result.get("local_cache_state") == "valid"
            ):
                local_hits.add(active_seed.worker_uuid)
            task.local_cache_hit_worker_uuids = sorted(local_hits)
            missing = set(targets) - local_hits
            if not missing:
                _finish(task, ModelPreheatExecutionStateEnum.READY)
                return
            task.execution_state = ModelPreheatExecutionStateEnum.DISTRIBUTING
            _create_distribution_tasks(session, task, targets, local_hits)
            return

        if distribution_tasks:
            state, message, local_hits, finished_at = _distribution_outcome(
                task, distribution_tasks, targets
            )
            task.execution_state = state
            task.state_message = message
            task.local_cache_hit_worker_uuids = local_hits
            task.finished_at = finished_at
            if finished_at is not None:
                task.progress = 100


async def _current_workers(session):
    rows = (await session.exec(select(Worker).order_by(Worker.id.desc()))).all()
    latest = {}
    for worker in rows:
        latest.setdefault(worker.worker_uuid, worker)
    return {
        worker_uuid: worker
        for worker_uuid, worker in latest.items()
        if worker.state == WorkerStateEnum.READY
        and worker.model_storage_protocol_version == MODEL_STORAGE_PROTOCOL_VERSION
    }


async def _reload_running_task(session, task_id, attempt, execution_state):
    return (
        await session.exec(
            select(ModelPreheatTask)
            .where(
                ModelPreheatTask.id == task_id,
                ModelPreheatTask.attempt == attempt,
                ModelPreheatTask.desired_state == ModelPreheatDesiredStateEnum.RUNNING,
                ModelPreheatTask.execution_state == execution_state,
            )
            .execution_options(populate_existing=True)
        )
    ).first()


async def _cas_parent_update(session, task, *, execution_state, **values):
    result = await session.exec(
        update(ModelPreheatTask)
        .where(
            ModelPreheatTask.id == task.id,
            ModelPreheatTask.attempt == task.attempt,
            ModelPreheatTask.desired_state == ModelPreheatDesiredStateEnum.RUNNING,
            ModelPreheatTask.execution_state == task.execution_state,
        )
        .values(execution_state=execution_state, **values)
        .execution_options(synchronize_session=False)
    )
    return result.rowcount == 1


def _add_seed_child(session, task, worker):
    session.add(
        ModelPreheatWorkerTask(
            task_id=task.id,
            parent_attempt=task.attempt,
            worker_uuid=worker.worker_uuid,
            worker_id=worker.id,
            role=ModelPreheatWorkerTaskRoleEnum.SEED,
        )
    )


def _create_seed(session, task, worker, inventory):
    task.execution_state = ModelPreheatExecutionStateEnum.STAGING
    task.seed_worker_uuid = worker.worker_uuid
    task.seed_worker_id = worker.id
    task.seed_source = getattr(inventory.get(worker.worker_uuid), "source", None)
    task.started_at = task.started_at or datetime.now(timezone.utc)
    session.add(
        ModelPreheatWorkerTask(
            task_id=task.id,
            parent_attempt=task.attempt,
            worker_uuid=worker.worker_uuid,
            worker_id=worker.id,
            role=ModelPreheatWorkerTaskRoleEnum.SEED,
        )
    )


def _create_distribution_tasks(session, task, workers, local_hits):
    for worker_uuid, worker in sorted(workers.items()):
        if worker_uuid in local_hits:
            continue
        session.add(
            ModelPreheatWorkerTask(
                task_id=task.id,
                parent_attempt=task.attempt,
                worker_uuid=worker_uuid,
                worker_id=worker.id,
                role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
            )
        )


def _replace_seed(session, task, seed_tasks, workers, invalid_seed):
    attempted = {seed.worker_uuid for seed in seed_tasks}
    candidates = sorted(set(workers) - attempted)
    invalid_seed.state = ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED
    invalid_seed.lease_owner = None
    invalid_seed.lease_token_hash = None
    invalid_seed.lease_expires_at = None
    invalid_seed.finished_at = datetime.now(timezone.utc)
    if not candidates:
        _finish(
            task,
            ModelPreheatExecutionStateEnum.ERROR,
            invalid_seed.error_code or "no_available_seed",
        )
        return
    _create_seed(session, task, workers[candidates[0]], {})


def _refresh_child_registrations(task, children, workers):
    for child in children:
        current = workers.get(child.worker_uuid)
        if current is None:
            continue
        if child.worker_id != current.id and child.state in {
            ModelPreheatWorkerTaskStateEnum.PENDING,
            ModelPreheatWorkerTaskStateEnum.RUNNING,
            ModelPreheatWorkerTaskStateEnum.ERROR,
        }:
            child.worker_id = current.id
            child.state = ModelPreheatWorkerTaskStateEnum.PENDING
            child.lease_owner = None
            child.lease_token_hash = None
            child.lease_expires_at = None
            child.error_code = None
            child.state_message = None
            child.finished_at = None
            if child.role == ModelPreheatWorkerTaskRoleEnum.SEED:
                task.seed_worker_id = current.id


async def _mark_removed_children(session, task, removed):
    if not removed:
        return
    children = (
        await session.exec(
            select(ModelPreheatWorkerTask).where(
                ModelPreheatWorkerTask.task_id == task.id,
                ModelPreheatWorkerTask.parent_attempt == task.attempt,
                ModelPreheatWorkerTask.worker_uuid.in_(removed),
            )
        )
    ).all()
    for child in children:
        if child.state not in {
            ModelPreheatWorkerTaskStateEnum.READY,
            ModelPreheatWorkerTaskStateEnum.ERROR,
        }:
            child.state = ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED
            child.lease_owner = None
            child.lease_token_hash = None
            child.lease_expires_at = None


def _distribution_outcome(task, children, current_workers):
    relevant = [child for child in children if child.worker_uuid in current_workers]
    terminal = {
        ModelPreheatWorkerTaskStateEnum.READY,
        ModelPreheatWorkerTaskStateEnum.ERROR,
        ModelPreheatWorkerTaskStateEnum.CANCELED,
        ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED,
    }
    if not relevant or not all(child.state in terminal for child in relevant):
        return (
            ModelPreheatExecutionStateEnum.DISTRIBUTING,
            None,
            sorted(set(task.local_cache_hit_worker_uuids) & set(current_workers)),
            None,
        )
    local_hits = set(task.local_cache_hit_worker_uuids) & set(current_workers)
    ready = local_hits | {
        child.worker_uuid
        for child in relevant
        if child.state == ModelPreheatWorkerTaskStateEnum.READY
    }
    if len(ready) == len(current_workers):
        state, message = ModelPreheatExecutionStateEnum.READY, None
    elif ready:
        state, message = ModelPreheatExecutionStateEnum.PARTIAL, None
    else:
        state, message = ModelPreheatExecutionStateEnum.ERROR, "distribution_failed"
    return state, message, sorted(local_hits), datetime.now(timezone.utc)


def _finish(task, state, message=None):
    task.execution_state = state
    task.state_message = message
    task.progress = 100
    task.finished_at = datetime.now(timezone.utc)
