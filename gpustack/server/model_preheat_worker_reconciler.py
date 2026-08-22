import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
    ModelPreheatWorkerObservation,
    distribution_operation_key,
    distribution_selector_digest,
)
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import (
    ModelPreheatExecutionStateEnum,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.bus import EventType
from gpustack.server.model_preheat_connectivity import (
    create_or_reuse_connectivity_check,
    current_registered_workers,
    latest_connectivity_results_for_workers,
)
from gpustack.utils.gpu import normalize_gpu_names


RETRYABLE_DISTRIBUTION_ERRORS = {
    "network_timeout",
    "s3_throttled",
    "s3_read_failed",
    "s3_ready_not_found",
    "worker_execution_failed",
}
MAX_DISTRIBUTION_ATTEMPTS = 5
MAX_DISTRIBUTION_RETRY_DELAY = 300


logger = logging.getLogger(__name__)


class ModelPreheatWorkerReconciler:
    def __init__(
        self,
        engine,
        *,
        ready_probe=None,
        connectivity_creator=create_or_reuse_connectivity_check,
        interval=15,
    ):
        self._engine = engine
        self._ready_probe = ready_probe
        self._connectivity_creator = connectivity_creator
        self._interval = interval

    async def start(self):
        event_task = asyncio.create_task(self._watch_workers())
        try:
            while True:
                try:
                    await self.reconcile_all()
                except Exception:
                    logger.exception("模型预热 worker 增量协调失败")
                await asyncio.sleep(self._interval)
        finally:
            event_task.cancel()
            await asyncio.gather(event_task, return_exceptions=True)

    async def _watch_workers(self):
        async for event in Worker.subscribe(self._engine):
            try:
                await self.handle_event(event)
            except Exception:
                logger.exception("模型预热 worker 事件处理失败")

    async def handle_event(self, event):
        if event.type == EventType.HEARTBEAT or event.data is None:
            return
        worker_uuid = getattr(event.data, "__dict__", {}).get("worker_uuid")
        worker_id = getattr(event.data, "__dict__", {}).get("id")
        if not worker_uuid:
            return
        if event.type == EventType.DELETED:
            if worker_id is not None:
                await self._reconcile_deleted(worker_uuid, worker_id)
            return
        if event.type in {EventType.CREATED, EventType.UPDATED}:
            await self.reconcile_worker(worker_uuid)

    async def reconcile_all(self):
        await self.reconcile_policies()
        async with AsyncSession(self._engine) as session:
            worker_uuids = [
                worker.worker_uuid
                for worker in await current_registered_workers(session)
            ]
        for worker_uuid in worker_uuids:
            await self.reconcile_worker(worker_uuid)

    async def reconcile_policies(self):
        async with AsyncSession(self._engine) as session:
            try:
                tasks = (
                    await session.exec(
                        select(ModelPreheatTask)
                        .where(
                            ModelPreheatTask.keep_new_workers_in_sync.is_(True),
                            ModelPreheatTask.execution_state
                            == ModelPreheatExecutionStateEnum.READY,
                        )
                        .order_by(ModelPreheatTask.id)
                    )
                ).all()
                for task in tasks:
                    await _ensure_policy(session, task)
                await session.flush()
                policies = (
                    await session.exec(
                        select(ModelPreheatDistributionPolicy).execution_options(
                            populate_existing=True
                        )
                    )
                ).all()
                for policy in policies:
                    profile = await session.get(
                        ModelPreheatS3Profile,
                        policy.profile_id,
                        populate_existing=True,
                    )
                    if (
                        profile is None
                        or policy.profile_config_version != profile.config_version
                    ):
                        await _deactivate_policy(session, policy)
                await session.commit()
            except (IntegrityError, OperationalError):
                await session.rollback()

    async def reconcile_policy(self, policy_id):
        async with AsyncSession(self._engine) as session:
            policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
            if policy is None:
                return
            workers = await current_registered_workers(session)
        for worker in workers:
            if worker.state == WorkerStateEnum.READY:
                await self._evaluate_worker(worker, policy_ids=[policy_id])
        async with AsyncSession(self._engine) as session:
            policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
            if policy is not None:
                policy.last_reconciled_at = datetime.now(timezone.utc)
                session.add(policy)
                await session.commit()

    async def reconcile_worker(self, worker_uuid):
        async with AsyncSession(self._engine) as session:
            worker = await _latest_worker(session, worker_uuid)
            if worker is None:
                return
            if worker.state != WorkerStateEnum.READY:
                await _record_worker_state(session, worker)
                return
        await self._ensure_connectivity_checks(worker)
        async with AsyncSession(self._engine) as session:
            current = await _latest_worker(session, worker_uuid)
            if current is None or current.id != worker.id:
                return
            await _record_worker_state(session, current)
            worker = await _latest_worker(session, worker_uuid)
        await self._evaluate_worker(worker)

    async def _ensure_connectivity_checks(self, worker):
        async with AsyncSession(self._engine) as session:
            profiles = (await session.exec(select(ModelPreheatS3Profile))).all()
            profile_ids = [profile.id for profile in profiles]
        for profile_id in profile_ids:
            async with AsyncSession(self._engine) as session:
                profile = await session.get(ModelPreheatS3Profile, profile_id)
                if profile is None:
                    continue
                scope = (
                    f"worker-lifecycle:{profile.id}:{profile.config_version}:"
                    f"{worker.worker_uuid}:{worker.id}:{_network_fingerprint(worker)}"
                )
                await self._connectivity_creator(
                    session,
                    profile,
                    [worker.worker_uuid],
                    idempotency_scope_key=scope,
                    request_hash=hashlib.sha256(scope.encode("utf-8")).hexdigest(),
                    scope_discriminator=_network_fingerprint(worker),
                )

    async def _evaluate_worker(self, worker, policy_ids=None):
        async with AsyncSession(self._engine) as session:
            statement = select(ModelPreheatDistributionPolicy).where(
                ModelPreheatDistributionPolicy.enabled.is_(True)
            )
            if policy_ids is not None:
                statement = statement.where(
                    ModelPreheatDistributionPolicy.id.in_(policy_ids)
                )
            policies = (await session.exec(statement)).all()
            for policy in policies:
                if not _policy_matches_worker(policy, worker):
                    continue
                source_task = await session.get(
                    ModelPreheatTask, policy.created_by_task_id
                )
                profile = await session.get(ModelPreheatS3Profile, policy.profile_id)
                if (
                    source_task is None
                    or profile is None
                    or not policy.enabled
                    or policy.profile_config_version != profile.config_version
                    or source_task.s3_profile_config_version
                    != policy.profile_config_version
                    or source_task.execution_state
                    != ModelPreheatExecutionStateEnum.READY
                ):
                    continue
                connectivity = await latest_connectivity_results_for_workers(
                    session,
                    policy.profile_id,
                    policy.profile_config_version,
                    [worker],
                )
                result = connectivity.get(worker.worker_uuid)
                if (
                    result is None
                    or result[0].state != ModelPreheatWorkerTaskStateEnum.READY
                ):
                    continue
                await session.refresh(source_task)
                await session.refresh(policy)
                await session.refresh(profile)
                current_worker = await _latest_worker(session, worker.worker_uuid)
                if (
                    not source_task.artifact_id
                    or not source_task.s3_manifest_path
                    or not source_task.manifest_digest
                    or not policy.enabled
                    or policy.profile_config_version != profile.config_version
                    or source_task.s3_profile_config_version
                    != policy.profile_config_version
                    or current_worker is None
                    or current_worker.id != worker.id
                    or source_task.execution_state
                    != ModelPreheatExecutionStateEnum.READY
                ):
                    continue
                await _create_or_rebind_distribution_task(
                    session, policy, source_task, current_worker
                )
                policy.last_reconciled_at = datetime.now(timezone.utc)
                session.add(policy)
            try:
                await session.commit()
            except (IntegrityError, OperationalError):
                await session.rollback()

    async def _reconcile_deleted(self, worker_uuid, worker_id):
        async with AsyncSession(self._engine) as session:
            tasks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.worker_uuid == worker_uuid,
                        ModelPreheatWorkerTask.worker_id == worker_id,
                        ModelPreheatWorkerTask.state.in_(
                            [
                                ModelPreheatWorkerTaskStateEnum.PENDING,
                                ModelPreheatWorkerTaskStateEnum.RUNNING,
                                ModelPreheatWorkerTaskStateEnum.PAUSED,
                            ]
                        ),
                    )
                )
            ).all()
            now = datetime.now(timezone.utc)
            for task in tasks:
                task.state = ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED
                task.lease_owner = None
                task.lease_token_hash = None
                task.lease_expires_at = None
                task.finished_at = now
                session.add(task)
            observation = await session.get(ModelPreheatWorkerObservation, worker_uuid)
            current_worker = await _latest_worker(session, worker_uuid)
            current_worker_id = (
                current_worker.id if current_worker is not None else None
            )
            current_worker_state = (
                current_worker.state if current_worker is not None else None
            )
            if (
                observation is not None
                and observation.worker_id == worker_id
                and (current_worker_id is None or current_worker_id == worker_id)
            ):
                observation.ready = False
                session.add(observation)
            await session.commit()
        if (
            current_worker_id is not None
            and current_worker_id != worker_id
            and current_worker_state == WorkerStateEnum.READY
        ):
            await self.reconcile_worker(worker_uuid)


async def _ensure_policy(session, task):
    profile = await session.get(
        ModelPreheatS3Profile, task.s3_profile_id, populate_existing=True
    )
    if profile is None or task.s3_profile_config_version != profile.config_version:
        return None
    worker_selector, gpu_selector = _selectors_for_task(task)
    selector_digest = distribution_selector_digest(worker_selector, gpu_selector)
    existing = (
        await session.exec(
            select(ModelPreheatDistributionPolicy).where(
                ModelPreheatDistributionPolicy.profile_id == task.s3_profile_id,
                ModelPreheatDistributionPolicy.request_digest == task.request_digest,
                ModelPreheatDistributionPolicy.target_scope == task.target_scope,
                ModelPreheatDistributionPolicy.selector_digest == selector_digest,
            )
        )
    ).first()
    if existing is not None:
        expected_version = existing.profile_config_version
        expected_task_id = existing.created_by_task_id
        expected_enabled = existing.enabled
        expected_profile_version_stale = existing.profile_version_stale
        should_update = (
            expected_version != profile.config_version
            or expected_task_id is None
            or task.id > expected_task_id
            or (existing.profile_version_stale and task.id == expected_task_id)
        )
        if should_update:
            enable_current_version = existing.enabled or existing.profile_version_stale
            result = await session.exec(
                update(ModelPreheatDistributionPolicy)
                .where(
                    ModelPreheatDistributionPolicy.id == existing.id,
                    ModelPreheatDistributionPolicy.profile_config_version
                    == expected_version,
                    ModelPreheatDistributionPolicy.created_by_task_id
                    == expected_task_id,
                    ModelPreheatDistributionPolicy.enabled == expected_enabled,
                    ModelPreheatDistributionPolicy.profile_version_stale
                    == expected_profile_version_stale,
                )
                .values(
                    profile_config_version=profile.config_version,
                    created_by_task_id=task.id,
                    enabled=enable_current_version,
                    profile_version_stale=False,
                )
                .execution_options(synchronize_session=False)
            )
            if result.rowcount == 1:
                await _reset_policy_tasks(session, existing.id, task.attempt)
                session.expire(existing)
        return existing
    policy = ModelPreheatDistributionPolicy(
        name=f"{task.model_id} 自动同步"[:255],
        profile_id=task.s3_profile_id,
        profile_config_version=task.s3_profile_config_version,
        request_identity=task.request_identity,
        request_digest=task.request_digest,
        target_scope=task.target_scope,
        worker_selector=worker_selector,
        gpu_selector=gpu_selector,
        selector_digest=selector_digest,
        created_by_task_id=task.id,
    )
    session.add(policy)
    return policy


def _selectors_for_task(task):
    if task.target_scope == ModelPreheatTargetScopeEnum.SAME_GPU_MODEL:
        return {}, {"gpu_names": sorted(normalize_gpu_names(task.target_gpu_names))}
    worker_uuids = (
        [task.seed_worker_uuid]
        if task.target_scope == ModelPreheatTargetScopeEnum.SEED_WORKER
        else task.target_worker_uuids
    )
    return {"worker_uuids": sorted(uuid for uuid in worker_uuids if uuid)}, {}


def _policy_matches_worker(policy, worker):
    selected_uuids = set(policy.worker_selector.get("worker_uuids", []))
    if selected_uuids and worker.worker_uuid not in selected_uuids:
        return False
    selected_gpu_names = normalize_gpu_names(policy.gpu_selector.get("gpu_names", []))
    if selected_gpu_names and not (selected_gpu_names & _worker_gpu_names(worker)):
        return False
    return bool(selected_uuids or selected_gpu_names)


def _worker_gpu_names(worker):
    raw_names = []
    status = worker.status
    if status is not None:
        raw_names.extend(device.name for device in status.gpu_devices or [])
    label_names = worker.labels.get("gpu_names") if worker.labels else None
    if label_names:
        raw_names.extend(label_names.split(","))
    return normalize_gpu_names(raw_names)


async def _deactivate_policy(session, policy):
    policy.enabled = False
    policy.profile_version_stale = True
    session.add(policy)
    await session.exec(
        update(ModelPreheatWorkerTask)
        .where(
            ModelPreheatWorkerTask.distribution_policy_id == policy.id,
            ModelPreheatWorkerTask.state.in_(
                [
                    ModelPreheatWorkerTaskStateEnum.PENDING,
                    ModelPreheatWorkerTaskStateEnum.RUNNING,
                    ModelPreheatWorkerTaskStateEnum.PAUSED,
                ]
            ),
        )
        .values(
            state=ModelPreheatWorkerTaskStateEnum.CANCELED,
            lease_owner=None,
            lease_token_hash=None,
            lease_expires_at=None,
            finished_at=datetime.now(timezone.utc),
        )
        .execution_options(synchronize_session=False)
    )


async def _reset_policy_tasks(session, policy_id, parent_attempt):
    await session.exec(
        update(ModelPreheatWorkerTask)
        .where(
            ModelPreheatWorkerTask.distribution_policy_id == policy_id,
            ModelPreheatWorkerTask.state != ModelPreheatWorkerTaskStateEnum.READY,
        )
        .values(
            parent_attempt=parent_attempt,
            state=ModelPreheatWorkerTaskStateEnum.PENDING,
            lease_owner=None,
            lease_token_hash=None,
            lease_expires_at=None,
            error_code=None,
            state_message=None,
            finished_at=None,
        )
        .execution_options(synchronize_session=False)
    )


async def _create_or_rebind_distribution_task(session, policy, source_task, worker):
    operation_key = distribution_operation_key(
        policy.id, worker.worker_uuid, policy.request_digest
    )
    task = (
        await session.exec(
            select(ModelPreheatWorkerTask).where(
                ModelPreheatWorkerTask.operation_key == operation_key
            )
        )
    ).first()
    if task is None:
        task = ModelPreheatWorkerTask(
            distribution_policy_id=policy.id,
            operation_key=operation_key,
            parent_attempt=source_task.attempt,
            worker_uuid=worker.worker_uuid,
            worker_id=worker.id,
            role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
        )
        session.add(task)
        return task
    if task.worker_id != worker.id:
        task.worker_id = worker.id
        task.parent_attempt = source_task.attempt
        task.state = ModelPreheatWorkerTaskStateEnum.PENDING
        task.lease_owner = None
        task.lease_token_hash = None
        task.lease_expires_at = None
        task.error_code = None
        task.state_message = None
        task.finished_at = None
        session.add(task)
    elif task.state == ModelPreheatWorkerTaskStateEnum.ERROR:
        await _retry_distribution_error(session, task, source_task, worker)
    return task


async def _retry_distribution_error(session, task, source_task, worker):
    if (
        task.error_code not in RETRYABLE_DISTRIBUTION_ERRORS
        or task.attempt >= MAX_DISTRIBUTION_ATTEMPTS
        or task.finished_at is None
    ):
        return False
    delay_seconds = min(
        MAX_DISTRIBUTION_RETRY_DELAY,
        5 * (2 ** max(task.attempt - 1, 0)),
    )
    if task.finished_at + timedelta(seconds=delay_seconds) > datetime.now(timezone.utc):
        return False
    result = await session.exec(
        update(ModelPreheatWorkerTask)
        .where(
            ModelPreheatWorkerTask.id == task.id,
            ModelPreheatWorkerTask.worker_id == worker.id,
            ModelPreheatWorkerTask.parent_attempt == source_task.attempt,
            ModelPreheatWorkerTask.state == ModelPreheatWorkerTaskStateEnum.ERROR,
            ModelPreheatWorkerTask.error_code == task.error_code,
            ModelPreheatWorkerTask.finished_at == task.finished_at,
        )
        .values(
            state=ModelPreheatWorkerTaskStateEnum.PENDING,
            lease_owner=None,
            lease_token_hash=None,
            lease_expires_at=None,
            error_code=None,
            state_message=None,
            finished_at=None,
        )
        .execution_options(synchronize_session=False)
    )
    return result.rowcount == 1


async def _latest_worker(session, worker_uuid):
    return (
        await session.exec(
            select(Worker)
            .where(Worker.worker_uuid == worker_uuid)
            .order_by(Worker.id.desc())
            .execution_options(populate_existing=True)
        )
    ).first()


async def _record_worker_state(session, worker):
    fingerprint = _network_fingerprint(worker)
    observation = await session.get(ModelPreheatWorkerObservation, worker.worker_uuid)
    is_ready = worker.state == WorkerStateEnum.READY
    if observation is None:
        observation = ModelPreheatWorkerObservation(
            worker_uuid=worker.worker_uuid,
            worker_id=worker.id,
            network_fingerprint=fingerprint,
            ready=is_ready,
        )
    else:
        observation.worker_id = worker.id
        observation.network_fingerprint = fingerprint
        observation.ready = is_ready
    session.add(observation)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()


def _network_fingerprint(worker):
    payload = json.dumps(
        [worker.worker_uuid, worker.hostname, worker.ip, worker.port],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
