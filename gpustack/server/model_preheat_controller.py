import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.model_preheat_credentials import ModelPreheatCredentialCipher
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
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
from gpustack.worker.model_preheat.s3_client import ModelPreheatS3Client


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadyProbeResult:
    manifest_digest: str
    generation_id: str
    ready_path: str
    manifest_path: str
    cache_key: str | None = None
    selection_digest: str | None = None
    profile_config_version: int | None = None
    file_count: int | None = None
    total_size: int | None = None


@dataclass(frozen=True)
class LocalInventoryProbeResult:
    worker_uuid: str
    state: str
    error_code: str | None = None


class ReadyProbe(Protocol):
    async def probe(self, task: ModelPreheatTask) -> ReadyProbeResult | None: ...


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
    def __init__(self, config):
        self._cipher = ModelPreheatCredentialCipher(
            getattr(config, "model_preheat_credential_key", None),
            getattr(config, "model_preheat_credential_key_version", None),
            getattr(config, "model_preheat_credential_old_keys", None),
        )

    async def probe(self, task):
        return await asyncio.to_thread(self._probe_sync, task)

    def _probe_sync(self, task):
        profile = _decrypt_profile_snapshot(self._cipher, task)
        client = ModelPreheatS3Client.from_minio(
            endpoint=profile["endpoint"],
            access_key=self._cipher.decrypt(profile["access_key_encrypted"]),
            secret_key=self._cipher.decrypt(profile["secret_key_encrypted"]),
            secure=profile.get("tls_enabled", True),
            tls_verify=profile.get("tls_verify", True),
            region=profile.get("region") or None,
            use_virtual_hosted_style=profile.get("use_virtual_hosted_style", True),
        )
        identity = _identity(task)
        manifest = client.read_ready_manifest(
            profile["bucket"],
            profile.get("prefix", ""),
            identity,
            cache_key=task.cache_key,
            selection_digest=task.selection_digest,
        )
        if manifest is None:
            return None
        return ReadyProbeResult(
            manifest_digest=manifest.digest,
            generation_id=manifest.generation_id,
            ready_path=client.ready_object(profile.get("prefix", ""), manifest),
            manifest_path=client.manifest_object(profile.get("prefix", ""), manifest),
            cache_key=manifest.cache_key,
            selection_digest=manifest.selection_digest,
            profile_config_version=task.s3_profile_config_version,
            file_count=len(manifest.files),
            total_size=manifest.total_size,
        )


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
        self._ready_probe = ready_probe
        self._inventory_probe = inventory_probe or MissingLocalInventoryProbe()
        self._interval = interval
        self._s3_inventory = s3_inventory

    async def _record_verified_ready(
        self,
        session,
        task,
        observed_ready,
        parent_state,
        parent_values,
    ):
        if self._s3_inventory is None:
            return True
        task_snapshot = task.model_copy(deep=True)
        task_id = task.id
        task_attempt = task.attempt
        task_state = task.execution_state
        profile_id = task.s3_profile_id
        cache_key = task.cache_key
        await session.commit()
        owner_token = uuid.uuid4().hex
        async with self._s3_inventory.selection_guard(
            profile_id,
            cache_key,
            owner_token,
            "publication",
        ) as lock_owner:
            if lock_owner is None:
                return None
            verified = await self._ready_probe.probe(task_snapshot)
            current = await _reload_running_task(
                session, task_id, task_attempt, task_state
            )
            if current is None or verified != observed_ready:
                await session.rollback()
                return None
            if not await _cas_parent_update(
                session,
                current,
                execution_state=parent_state,
                **parent_values,
            ):
                await session.rollback()
                return None
            current.execution_state = parent_state
            from gpustack.server.model_preheat_s3_inventory import (
                upsert_verified_publication,
            )

            recorded = await upsert_verified_publication(
                session,
                current,
                verified,
                expected_attempt=current.attempt,
                expected_profile_version=current.s3_profile_config_version,
                lock_owner=lock_owner,
            )
            if not recorded:
                await session.rollback()
                return None
            await self._s3_inventory.release_verified_publication_marker(
                session, current
            )
            await session.commit()
            return True

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
                        ModelPreheatTask.execution_state.notin_(
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
                    if self._s3_inventory is not None:
                        await self._s3_inventory.terminate_publication_marker(
                            session, task
                        )
                    await session.exec(
                        delete(ModelPreheatTaskLock).where(
                            ModelPreheatTaskLock.task_id == task_id
                        )
                    )
                await session.commit()
        except (IntegrityError, OperationalError):
            # 唯一约束决定多实例竞争的唯一赢家。
            return

    async def _reconcile(self, session, task_id):
        task = await session.get(ModelPreheatTask, task_id)
        if (
            task is None
            or task.desired_state != ModelPreheatDesiredStateEnum.RUNNING
            or is_terminal_task(task)
        ):
            return

        current_workers = await _current_workers(session, task.target_worker_uuids)
        removed = sorted(set(task.target_worker_uuids) - set(current_workers))
        task.removed_target_worker_uuids = removed
        if removed:
            await _mark_removed_children(session, task, removed)
        if not current_workers:
            await _cas_parent_update(
                session,
                task,
                execution_state=ModelPreheatExecutionStateEnum.ERROR,
                state_message="no_available_targets",
                progress=100,
                finished_at=datetime.now(timezone.utc),
            )
            return

        children = (
            await session.exec(
                select(ModelPreheatWorkerTask).where(
                    ModelPreheatWorkerTask.task_id == task.id,
                    ModelPreheatWorkerTask.parent_attempt == task.attempt,
                )
            )
        ).all()
        for child in children:
            current = current_workers.get(child.worker_uuid)
            if (
                current is not None
                and child.worker_id != current.id
                and child.state
                not in {
                    ModelPreheatWorkerTaskStateEnum.READY,
                    ModelPreheatWorkerTaskStateEnum.CANCELED,
                    ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED,
                }
            ):
                child.worker_id = current.id
                child.state = ModelPreheatWorkerTaskStateEnum.PENDING
                child.lease_owner = None
                child.lease_token_hash = None
                child.lease_expires_at = None
                child.error_code = None
                child.state_message = None
                child.finished_at = None
                session.add(child)
                if child.role == ModelPreheatWorkerTaskRoleEnum.SEED:
                    task.seed_worker_id = current.id
                    session.add(task)
        seed_tasks = [
            child
            for child in children
            if child.role == ModelPreheatWorkerTaskRoleEnum.SEED
        ]
        distribute_tasks = [
            child
            for child in children
            if child.role == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE
        ]

        if not children:
            expected_attempt = task.attempt
            expected_state = task.execution_state
            ready = None
            if self._ready_probe is not None:
                ready = await self._ready_probe.probe(task)
                task = await _reload_running_task(
                    session, task.id, expected_attempt, expected_state
                )
                if task is None:
                    return
                current_workers = await _current_workers(
                    session, task.target_worker_uuids
                )
            inventory = await self._inventory_probe.probe(task, sorted(current_workers))
            task = await _reload_running_task(
                session, task.id, expected_attempt, expected_state
            )
            if task is None:
                return
            current_workers = await _current_workers(session, task.target_worker_uuids)
            if not current_workers:
                await _cas_parent_update(
                    session,
                    task,
                    execution_state=ModelPreheatExecutionStateEnum.ERROR,
                    state_message="no_available_targets",
                    progress=100,
                    finished_at=datetime.now(timezone.utc),
                )
                return
            valid = sorted(
                worker_uuid
                for worker_uuid, result in inventory.items()
                if worker_uuid in current_workers and result.state == "valid"
            )
            candidates = sorted(
                worker_uuid
                for worker_uuid, result in inventory.items()
                if worker_uuid in current_workers and result.state == "candidate"
            )
            if ready is not None:
                missing = set(current_workers) - set(valid)
                new_state = (
                    ModelPreheatExecutionStateEnum.DISTRIBUTING
                    if missing
                    else ModelPreheatExecutionStateEnum.READY
                )
                parent_values = {
                    "local_cache_hit_worker_uuids": valid,
                    "manifest_digest": ready.manifest_digest,
                    "generation_id": ready.generation_id,
                    "s3_ready_path": ready.ready_path,
                    "s3_manifest_path": ready.manifest_path,
                    "progress": 100 if not missing else task.progress,
                    "finished_at": (
                        datetime.now(timezone.utc) if not missing else None
                    ),
                }
                if self._s3_inventory is not None:
                    current_task_id = task.id
                    if not await self._record_verified_ready(
                        session, task, ready, new_state, parent_values
                    ):
                        return
                    task = await session.get(
                        ModelPreheatTask, current_task_id, populate_existing=True
                    )
                    current_workers = await _current_workers(
                        session, task.target_worker_uuids
                    )
                elif not await _cas_parent_update(
                    session, task, execution_state=new_state, **parent_values
                ):
                    return
                await _create_distribution_tasks(
                    session, task, current_workers, set(valid)
                )
                return

            if self._s3_inventory is not None:
                marker_task = task.model_copy(deep=True)
                current_task_id = task.id
                current_attempt = task.attempt
                current_state = task.execution_state
                await session.commit()
                if not await self._s3_inventory.ensure_publication_marker(marker_task):
                    return
                task = await _reload_running_task(
                    session,
                    current_task_id,
                    current_attempt,
                    current_state,
                )
                if task is None:
                    return
                current_workers = await _current_workers(
                    session, task.target_worker_uuids
                )
                valid = sorted(set(valid) & set(current_workers))
                candidates = sorted(set(candidates) & set(current_workers))
                if not current_workers:
                    return
            seed_uuid = _select_seed(task, current_workers, valid, candidates)
            worker = current_workers[seed_uuid]
            if not await _cas_parent_update(
                session,
                task,
                execution_state=ModelPreheatExecutionStateEnum.STAGING,
                local_cache_hit_worker_uuids=valid,
                seed_worker_uuid=seed_uuid,
                seed_worker_id=worker.id,
                seed_source=getattr(inventory.get(seed_uuid), "source", None),
                started_at=task.started_at or datetime.now(timezone.utc),
            ):
                return
            session.add(
                ModelPreheatWorkerTask(
                    task_id=task.id,
                    parent_attempt=task.attempt,
                    worker_uuid=seed_uuid,
                    worker_id=worker.id,
                    role=ModelPreheatWorkerTaskRoleEnum.SEED,
                )
            )
            await session.flush()
            return

        active_seed = next(
            (
                child
                for child in seed_tasks
                if child.state != ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED
            ),
            None,
        )
        if (
            self._s3_inventory is not None
            and active_seed is not None
            and not distribute_tasks
            and active_seed.state
            in {
                ModelPreheatWorkerTaskStateEnum.PENDING,
                ModelPreheatWorkerTaskStateEnum.RUNNING,
            }
        ):
            marker_task = task.model_copy(deep=True)
            current_task_id = task.id
            current_attempt = task.attempt
            current_state = task.execution_state
            active_seed_id = active_seed.id
            await session.commit()
            if not await self._s3_inventory.ensure_publication_marker(marker_task):
                return
            task = await _reload_running_task(
                session,
                current_task_id,
                current_attempt,
                current_state,
            )
            active_seed = await session.get(
                ModelPreheatWorkerTask, active_seed_id, populate_existing=True
            )
            if task is None or active_seed is None:
                return
            current_workers = await _current_workers(session, task.target_worker_uuids)
        if active_seed is None and not distribute_tasks:
            await _replace_seed(session, task, seed_tasks, current_workers)
            return

        if active_seed is not None and not distribute_tasks:
            if active_seed.state == ModelPreheatWorkerTaskStateEnum.ERROR:
                await _replace_seed(
                    session,
                    task,
                    seed_tasks,
                    current_workers,
                    invalid_seed=active_seed,
                    error_code=active_seed.error_code,
                )
                return
            if active_seed.state != ModelPreheatWorkerTaskStateEnum.READY:
                return
            result = active_seed.resumable_cursor or {}
            expected_attempt = task.attempt
            expected_state = task.execution_state
            ready = (
                await self._ready_probe.probe(task)
                if self._ready_probe is not None
                else None
            )
            task = await _reload_running_task(
                session, task.id, expected_attempt, expected_state
            )
            if task is None:
                return
            current_workers = await _current_workers(session, task.target_worker_uuids)
            active_seed = await session.get(
                ModelPreheatWorkerTask, active_seed.id, populate_existing=True
            )
            if (
                active_seed is None
                or active_seed.parent_attempt != task.attempt
                or active_seed.state != ModelPreheatWorkerTaskStateEnum.READY
            ):
                return
            if (
                ready is None
                or ready.manifest_digest != result.get("manifest_digest")
                or ready.generation_id != result.get("generation_id")
            ):
                await _replace_seed(
                    session,
                    task,
                    seed_tasks,
                    current_workers,
                    invalid_seed=active_seed,
                    error_code="s3_manifest_invalid",
                )
                return
            local_hits = set(task.local_cache_hit_worker_uuids) & set(current_workers)
            if result.get("local_cache_state") == "valid":
                local_hits.add(active_seed.worker_uuid)
            missing = set(current_workers) - local_hits
            new_state = (
                ModelPreheatExecutionStateEnum.DISTRIBUTING
                if missing
                else ModelPreheatExecutionStateEnum.READY
            )
            parent_values = {
                "local_cache_hit_worker_uuids": sorted(local_hits),
                "manifest_digest": ready.manifest_digest,
                "generation_id": ready.generation_id,
                "s3_ready_path": ready.ready_path,
                "s3_manifest_path": ready.manifest_path,
                "progress": 100 if not missing else task.progress,
                "finished_at": datetime.now(timezone.utc) if not missing else None,
            }
            if self._s3_inventory is not None:
                current_task_id = task.id
                if not await self._record_verified_ready(
                    session, task, ready, new_state, parent_values
                ):
                    return
                task = await session.get(
                    ModelPreheatTask, current_task_id, populate_existing=True
                )
                current_workers = await _current_workers(
                    session, task.target_worker_uuids
                )
            elif not await _cas_parent_update(
                session, task, execution_state=new_state, **parent_values
            ):
                return
            await _create_distribution_tasks(
                session,
                task,
                current_workers,
                local_hits,
            )
            await session.flush()
            return

        state, message, local_hits, finished_at = _distribution_outcome(
            task, distribute_tasks, current_workers
        )
        await _cas_parent_update(
            session,
            task,
            execution_state=state,
            state_message=message,
            local_cache_hit_worker_uuids=local_hits,
            progress=100 if finished_at is not None else task.progress,
            finished_at=finished_at,
        )


async def _current_workers(session, target_uuids):
    rows = (
        await session.exec(
            select(Worker)
            .where(Worker.worker_uuid.in_(target_uuids))
            .order_by(Worker.id.desc())
        )
    ).all()
    result = {}
    for worker in rows:
        result.setdefault(worker.worker_uuid, worker)
    return {
        worker_uuid: worker
        for worker_uuid, worker in result.items()
        if worker.state == WorkerStateEnum.READY
    }


async def _mark_removed_children(session, task, removed):
    children = (
        await session.exec(
            select(ModelPreheatWorkerTask).where(
                ModelPreheatWorkerTask.task_id == task.id,
                ModelPreheatWorkerTask.parent_attempt == task.attempt,
                ModelPreheatWorkerTask.worker_uuid.in_(removed),
            )
        )
    ).all()
    existing_uuids = {child.worker_uuid for child in children}
    for child in children:
        if child.state not in {
            ModelPreheatWorkerTaskStateEnum.READY,
            ModelPreheatWorkerTaskStateEnum.ERROR,
        }:
            child.state = ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED
            child.lease_owner = None
            child.lease_token_hash = None
            child.lease_expires_at = None
            session.add(child)
    snapshot = {item.get("worker_uuid"): item for item in task.target_worker_snapshot}
    for worker_uuid in sorted(set(removed) - existing_uuids):
        item = snapshot.get(worker_uuid, {})
        session.add(
            ModelPreheatWorkerTask(
                task_id=task.id,
                parent_attempt=task.attempt,
                worker_uuid=worker_uuid,
                worker_id=item.get("worker_id"),
                role=(
                    ModelPreheatWorkerTaskRoleEnum.SEED
                    if worker_uuid == task.seed_worker_uuid
                    else ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE
                ),
                state=ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED,
                finished_at=datetime.now(timezone.utc),
            )
        )


async def _create_distribution_tasks(session, task, workers, local_hits):
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


def _select_seed(task, workers, valid, candidates=()):
    if valid:
        return valid[0]
    if candidates:
        return candidates[0]
    if task.seed_worker_uuid in workers:
        return task.seed_worker_uuid
    return sorted(workers)[0]


def _distribution_outcome(task, children, current_workers):
    relevant = [child for child in children if child.worker_uuid in current_workers]
    if not relevant:
        return (
            ModelPreheatExecutionStateEnum.ERROR,
            "no_available_targets",
            [],
            datetime.now(timezone.utc),
        )
    terminal = {
        ModelPreheatWorkerTaskStateEnum.READY,
        ModelPreheatWorkerTaskStateEnum.ERROR,
        ModelPreheatWorkerTaskStateEnum.CANCELED,
        ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED,
    }
    if not all(child.state in terminal for child in relevant):
        return (
            ModelPreheatExecutionStateEnum.DISTRIBUTING,
            None,
            sorted(set(task.local_cache_hit_worker_uuids) & set(current_workers)),
            None,
        )
    current_uuids = set(current_workers)
    local_hits = set(task.local_cache_hit_worker_uuids) & current_uuids
    ready_uuids = local_hits | {
        child.worker_uuid
        for child in relevant
        if child.state == ModelPreheatWorkerTaskStateEnum.READY
    }
    if len(ready_uuids) == len(current_workers):
        state, message = ModelPreheatExecutionStateEnum.READY, None
    elif ready_uuids:
        state, message = ModelPreheatExecutionStateEnum.PARTIAL, None
    else:
        state, message = (
            ModelPreheatExecutionStateEnum.ERROR,
            "distribution_failed",
        )
    return state, message, sorted(local_hits), datetime.now(timezone.utc)


def _finish(task, state, message=None):
    task.execution_state = state
    task.state_message = message
    task.progress = 100
    task.finished_at = datetime.now(timezone.utc)


def _identity(task):
    return ModelPreheatIdentity(
        source=task.source,
        model_id=task.model_id,
        revision=task.resolved_revision,
        file_patterns=task.include_patterns,
    )


def _decrypt_profile_snapshot(cipher, task):
    return json.loads(cipher.decrypt(task.s3_profile_snapshot_encrypted))


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


async def _replace_seed(
    session,
    task,
    seed_tasks,
    current_workers,
    *,
    invalid_seed=None,
    error_code=None,
):
    attempted = {seed.worker_uuid for seed in seed_tasks}
    candidates = sorted(set(current_workers) - attempted)
    if not candidates:
        changed = await _cas_parent_update(
            session,
            task,
            execution_state=ModelPreheatExecutionStateEnum.ERROR,
            state_message=error_code or "no_available_seed",
            progress=100,
            finished_at=datetime.now(timezone.utc),
        )
        if changed and invalid_seed is not None:
            _skip_seed(session, invalid_seed)
        return
    seed_uuid = candidates[0]
    worker = current_workers[seed_uuid]
    if not await _cas_parent_update(
        session,
        task,
        execution_state=task.execution_state,
        seed_worker_uuid=seed_uuid,
        seed_worker_id=worker.id,
    ):
        return
    if invalid_seed is not None:
        _skip_seed(session, invalid_seed)
    session.add(
        ModelPreheatWorkerTask(
            task_id=task.id,
            parent_attempt=task.attempt,
            worker_uuid=seed_uuid,
            worker_id=worker.id,
            role=ModelPreheatWorkerTaskRoleEnum.SEED,
        )
    )


def _skip_seed(session, seed):
    seed.state = ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED
    seed.lease_owner = None
    seed.lease_token_hash = None
    seed.lease_expires_at = None
    seed.finished_at = datetime.now(timezone.utc)
    session.add(seed)
