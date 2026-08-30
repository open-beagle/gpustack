import asyncio
from contextlib import suppress
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
    ModelPreheatDistributionPolicyRun,
    ModelPreheatDistributionPolicyRunStateEnum,
    ModelPreheatDistributionPolicyRunTriggerEnum,
    ModelPreheatDistributionPolicyRunTask,
    ModelPreheatDistributionPolicyTriggerModeEnum,
    ModelPreheatDistributionWorkerSlot,
    ModelPreheatDistributionSelectionModeEnum,
    ModelPreheatWorkerObservation,
    distribution_operation_key,
    distribution_policy_run_operation_key,
    distribution_selector_digest,
)
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatExecutionStateEnum,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.workers import (
    MODEL_STORAGE_PROTOCOL_VERSION,
    Worker,
    WorkerStateEnum,
)
from gpustack.server.bus import EventType
from gpustack.server.model_preheat_connectivity import (
    create_or_reuse_connectivity_check,
    current_registered_workers,
    latest_connectivity_results_for_workers,
)
from gpustack.server.model_preheat_distribution_source import (
    DistributionSourceUnavailable,
    resolve_distribution_source,
    resolve_distribution_sources,
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
ACTIVE_DISTRIBUTION_TASK_STATES = (
    ModelPreheatWorkerTaskStateEnum.PENDING,
    ModelPreheatWorkerTaskStateEnum.RUNNING,
    ModelPreheatWorkerTaskStateEnum.PAUSED,
)


logger = logging.getLogger(__name__)


def _distribution_outcome_item(status, reason=None, *, worker=None, task_id=None):
    item = {
        "task_id": task_id,
        "worker_id": worker.id if worker is not None else None,
        "worker_uuid": worker.worker_uuid if worker is not None else None,
    }
    if reason is not None:
        item["reason"] = reason
    return status, item


def _distribution_reconcile_result(items):
    outcome = {"created": [], "skipped": [], "failed": []}
    for status, item in items:
        outcome[status].append(item)
    if outcome["created"]:
        error_code = (
            "distribution_partial_outcome"
            if outcome["skipped"] or outcome["failed"]
            else None
        )
    elif outcome["failed"]:
        error_code = outcome["failed"][0].get("reason") or "distribution_run_failed"
    else:
        error_code = "distribution_no_eligible_workers"
    return {"outcome": outcome, "error_code": error_code}


def _distribution_outcome_has_created(outcome):
    return bool(isinstance(outcome, dict) and outcome.get("created"))


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
        self._run_lease_owner = uuid4().hex
        self._run_lease_ttl = timedelta(seconds=60)
        self._continuous_safety_interval = timedelta(minutes=5)
        self._next_continuous_safety_at = None

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
            await self.reconcile_worker(worker_uuid, event_driven=True)

    async def reconcile_all(self):
        await self._settle_policy_runs()
        await self.reconcile_policies()
        async with AsyncSession(self._engine) as session:
            workers = await current_registered_workers(session)
        for worker in workers:
            if (
                worker.state == WorkerStateEnum.READY
                and worker.model_storage_protocol_version
                == MODEL_STORAGE_PROTOCOL_VERSION
            ):
                await self.reconcile_worker(worker.worker_uuid, safety_check=True)
        now = datetime.now(timezone.utc)
        if (
            self._next_continuous_safety_at is not None
            and self._next_continuous_safety_at > now
        ):
            return
        self._next_continuous_safety_at = now + self._continuous_safety_interval
        async with AsyncSession(self._engine) as session:
            policy_ids = (
                await session.exec(
                    select(ModelPreheatDistributionPolicy.id).where(
                        ModelPreheatDistributionPolicy.enabled.is_(True),
                        ModelPreheatDistributionPolicy.trigger_mode
                        == ModelPreheatDistributionPolicyTriggerModeEnum.CONTINUOUS,
                    )
                )
            ).all()
        for policy_id in policy_ids:
            await self.reconcile_continuous_policy(policy_id, now)

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
                        policy.blocked_reason = "distribution_profile_not_active"
                        session.add(policy)
                    elif policy.enabled:
                        try:
                            await resolve_distribution_sources(session, policy)
                        except DistributionSourceUnavailable as exc:
                            policy.blocked_reason = str(exc)
                            session.add(policy)
                        else:
                            policy.blocked_reason = None
                            session.add(policy)
                await session.commit()
            except (IntegrityError, OperationalError):
                await session.rollback()

    async def reconcile_policy(
        self, policy_id, run_key=None, lease_check=None, run_id=None
    ):
        outcomes = []
        async with AsyncSession(self._engine) as session:
            policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
            if policy is None:
                return _distribution_reconcile_result(
                    [
                        _distribution_outcome_item(
                            "failed", "distribution_policy_not_found"
                        )
                    ]
                )
            if run_id is not None:
                try:
                    await resolve_distribution_sources(session, policy)
                except DistributionSourceUnavailable as exc:
                    return _distribution_reconcile_result(
                        [_distribution_outcome_item("failed", str(exc))]
                    )
            workers = await current_registered_workers(session)
        if run_id is not None and not workers:
            outcomes.append(
                _distribution_outcome_item(
                    "failed", "distribution_no_registered_workers"
                )
            )
        for worker in workers:
            if lease_check is not None and not lease_check():
                outcomes.append(
                    _distribution_outcome_item(
                        "failed", "distribution_schedule_lease_lost"
                    )
                )
                break
            if run_id is not None:
                if not _policy_selector_matches_worker(policy, worker):
                    continue
                outcomes.extend(
                    await self._evaluate_worker(
                        worker,
                        policy_ids=[policy_id],
                        run_key=run_key,
                        run_id=run_id,
                    )
                )
            elif (
                worker.state == WorkerStateEnum.READY
                and worker.model_storage_protocol_version
                == MODEL_STORAGE_PROTOCOL_VERSION
            ):
                await self._evaluate_worker(
                    worker, policy_ids=[policy_id], run_key=run_key, run_id=run_id
                )
        async with AsyncSession(self._engine) as session:
            policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
            if policy is not None:
                policy.last_reconciled_at = datetime.now(timezone.utc)
                session.add(policy)
                await session.commit()
        return _distribution_reconcile_result(outcomes)

    async def reconcile_manual_policy(self, policy_id):
        return await self._run_policy(
            policy_id,
            ModelPreheatDistributionPolicyRunTriggerEnum.MANUAL,
            datetime.now(timezone.utc),
            unique_suffix=uuid4().hex,
        )

    async def reconcile_continuous_policy(self, policy_id, now):
        window_seconds = int(self._continuous_safety_interval.total_seconds())
        window = datetime.fromtimestamp(
            int(now.timestamp() // window_seconds) * window_seconds,
            tz=timezone.utc,
        )
        return await self._run_policy(
            policy_id,
            ModelPreheatDistributionPolicyRunTriggerEnum.CONTINUOUS,
            window,
        )

    async def _run_policy(self, policy_id, trigger, window, unique_suffix=None):
        operation_key = distribution_policy_run_operation_key(
            policy_id, window, trigger.value, unique_suffix
        )
        now = datetime.now(timezone.utc)
        lease_token = uuid4().hex
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            run = ModelPreheatDistributionPolicyRun(
                policy_id=policy_id,
                trigger=trigger,
                window_start_utc=window,
                operation_key=operation_key,
                lease_owner=self._run_lease_owner,
                lease_token=lease_token,
                lease_expires_at=now + self._run_lease_ttl,
                started_at=now,
            )
            session.add(run)
            try:
                await session.commit()
                await session.refresh(run)
            except (IntegrityError, OperationalError):
                await session.rollback()
                return None
        error_code = None
        result = None
        lease_lost = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._policy_run_lease_heartbeat(run.id, lease_token, lease_lost)
        )
        try:
            result = await self.reconcile_policy(
                policy_id,
                operation_key,
                lease_check=lambda: not lease_lost.is_set(),
                run_id=run.id,
            )
        except Exception as exc:
            error_code = type(exc).__name__
            raise
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            outcome = result.get("outcome") if isinstance(result, dict) else None
            result_error = (
                result.get("error_code") if isinstance(result, dict) else None
            )
            if lease_lost.is_set():
                error_code = error_code or "distribution_run_lease_lost"
            effective_error = error_code or result_error
            has_created = _distribution_outcome_has_created(outcome)
            values = {
                "error_code": effective_error,
                "outcome": outcome,
            }
            if has_created:
                values.update(
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                    finished_at=None,
                )
            else:
                values.update(
                    state=(
                        ModelPreheatDistributionPolicyRunStateEnum.ERROR
                        if effective_error
                        else ModelPreheatDistributionPolicyRunStateEnum.READY
                    ),
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                    finished_at=datetime.now(timezone.utc),
                )
            async with AsyncSession(self._engine) as session:
                await session.exec(
                    update(ModelPreheatDistributionPolicyRun)
                    .where(
                        ModelPreheatDistributionPolicyRun.id == run.id,
                        ModelPreheatDistributionPolicyRun.state
                        == ModelPreheatDistributionPolicyRunStateEnum.PENDING,
                        ModelPreheatDistributionPolicyRun.lease_owner
                        == self._run_lease_owner,
                        ModelPreheatDistributionPolicyRun.lease_token == lease_token,
                    )
                    .values(**values)
                )
                await session.commit()
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            return await session.get(ModelPreheatDistributionPolicyRun, run.id)

    async def _renew_policy_run_lease(self, run_id, token, now=None):
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        async with AsyncSession(self._engine) as session:
            result = await session.exec(
                update(ModelPreheatDistributionPolicyRun)
                .where(
                    ModelPreheatDistributionPolicyRun.id == run_id,
                    ModelPreheatDistributionPolicyRun.state
                    == ModelPreheatDistributionPolicyRunStateEnum.PENDING,
                    ModelPreheatDistributionPolicyRun.lease_owner
                    == self._run_lease_owner,
                    ModelPreheatDistributionPolicyRun.lease_token == token,
                    ModelPreheatDistributionPolicyRun.lease_expires_at > now,
                )
                .values(lease_expires_at=now + self._run_lease_ttl)
            )
            await session.commit()
        return result.rowcount == 1

    async def _policy_run_lease_heartbeat(self, run_id, token, lease_lost):
        while True:
            await asyncio.sleep(max(0.005, self._run_lease_ttl.total_seconds() / 4))
            try:
                renewed = await self._renew_policy_run_lease(run_id, token)
            except Exception:
                renewed = False
            if not renewed:
                lease_lost.set()
                return

    async def reconcile_worker(
        self, worker_uuid, *, event_driven=False, safety_check=True
    ):
        async with AsyncSession(self._engine) as session:
            worker = await _latest_worker(session, worker_uuid)
            if worker is None:
                return
            if (
                worker.state != WorkerStateEnum.READY
                or worker.model_storage_protocol_version
                != MODEL_STORAGE_PROTOCOL_VERSION
            ):
                await _record_worker_state(
                    session,
                    worker,
                    force_not_ready=(
                        worker.model_storage_protocol_version
                        != MODEL_STORAGE_PROTOCOL_VERSION
                    ),
                )
                return
        await self._ensure_connectivity_checks(worker)
        async with AsyncSession(self._engine) as session:
            current = await _latest_worker(session, worker_uuid)
            if (
                current is None
                or current.id != worker.id
                or current.model_storage_protocol_version
                != MODEL_STORAGE_PROTOCOL_VERSION
            ):
                return
            await _record_worker_state(session, current)
            worker = await _latest_worker(session, worker_uuid)
        if event_driven or safety_check:
            await self._evaluate_worker(
                worker,
                allowed_modes=[
                    ModelPreheatDistributionPolicyTriggerModeEnum.CONTINUOUS
                ],
            )

    async def _ensure_connectivity_checks(self, worker):
        async with AsyncSession(self._engine) as session:
            profiles = (
                await session.exec(
                    select(ModelPreheatS3Profile).where(
                        ModelPreheatS3Profile.lifecycle_state
                        == ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE
                    )
                )
            ).all()
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
                    update_profile_pointer=False,
                )

    async def _evaluate_worker(
        self, worker, policy_ids=None, allowed_modes=None, run_key=None, run_id=None
    ):
        outcomes = []
        if worker.state != WorkerStateEnum.READY:
            if run_id is not None:
                outcomes.append(
                    _distribution_outcome_item(
                        "skipped",
                        "distribution_worker_not_ready",
                        worker=worker,
                    )
                )
            return outcomes
        if worker.model_storage_protocol_version != MODEL_STORAGE_PROTOCOL_VERSION:
            if run_id is not None:
                outcomes.append(
                    _distribution_outcome_item(
                        "skipped",
                        "distribution_worker_protocol_unsupported",
                        worker=worker,
                    )
                )
            return outcomes
        async with AsyncSession(self._engine) as session:
            statement = select(ModelPreheatDistributionPolicy).where(
                ModelPreheatDistributionPolicy.enabled.is_(True)
            )
            if policy_ids is not None:
                statement = statement.where(
                    ModelPreheatDistributionPolicy.id.in_(policy_ids)
                )
            if allowed_modes is not None:
                statement = statement.where(
                    ModelPreheatDistributionPolicy.trigger_mode.in_(allowed_modes)
                )
            policies = (await session.exec(statement)).all()
            for policy in policies:
                if not _policy_matches_worker(policy, worker):
                    continue
                try:
                    sources = await resolve_distribution_sources(session, policy)
                except DistributionSourceUnavailable as exc:
                    policy.blocked_reason = str(exc)
                    session.add(policy)
                    if run_id is not None:
                        outcomes.append(
                            _distribution_outcome_item(
                                "failed", str(exc), worker=worker
                            )
                        )
                    continue
                if not sources:
                    if run_id is not None:
                        outcomes.append(
                            _distribution_outcome_item(
                                "failed",
                                "distribution_source_unavailable",
                                worker=worker,
                            )
                        )
                    continue
                profile = sources[0].profile
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
                    if run_id is not None:
                        outcomes.append(
                            _distribution_outcome_item(
                                "skipped",
                                "distribution_connectivity_not_ready",
                                worker=worker,
                            )
                        )
                    continue
                await session.refresh(policy)
                await session.refresh(profile)
                current_worker = await _latest_worker(session, worker.worker_uuid)
                try:
                    sources = await resolve_distribution_sources(session, policy)
                except DistributionSourceUnavailable as exc:
                    policy.blocked_reason = str(exc)
                    session.add(policy)
                    if run_id is not None:
                        outcomes.append(
                            _distribution_outcome_item(
                                "failed", str(exc), worker=worker
                            )
                        )
                    continue
                if (
                    profile.lifecycle_state
                    != ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE
                    or not policy.enabled
                    or policy.profile_config_version != profile.config_version
                    or current_worker is None
                    or current_worker.id != worker.id
                    or current_worker.model_storage_protocol_version
                    != MODEL_STORAGE_PROTOCOL_VERSION
                ):
                    if run_id is not None:
                        outcomes.append(
                            _distribution_outcome_item(
                                "failed",
                                "distribution_policy_changed_during_run",
                                worker=worker,
                            )
                        )
                    continue
                for source in sources:
                    task = await _create_or_rebind_distribution_task(
                        session, policy, source, current_worker, run_key=run_key
                    )
                    if run_id is not None and task is not None:
                        await session.merge(
                            ModelPreheatDistributionPolicyRunTask(
                                run_id=run_id, task_id=task.id
                            )
                        )
                        outcomes.append(
                            _distribution_outcome_item(
                                "created", worker=worker, task_id=task.id
                            )
                        )
                    elif run_id is not None:
                        outcomes.append(
                            _distribution_outcome_item(
                                "failed",
                                "distribution_task_not_created",
                                worker=worker,
                            )
                        )
                policy.last_reconciled_at = datetime.now(timezone.utc)
                policy.blocked_reason = None
                session.add(policy)
            try:
                await session.commit()
            except (IntegrityError, OperationalError):
                await session.rollback()
                if run_id is not None:
                    return [
                        _distribution_outcome_item(
                            "failed",
                            "distribution_outcome_persist_failed",
                            worker=worker,
                        )
                    ]
        return outcomes

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

    async def _settle_policy_runs(self):
        terminal = {
            ModelPreheatWorkerTaskStateEnum.READY,
            ModelPreheatWorkerTaskStateEnum.ERROR,
            ModelPreheatWorkerTaskStateEnum.CANCELED,
            ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED,
        }
        async with AsyncSession(self._engine) as session:
            now = datetime.now(timezone.utc)
            runs = (
                await session.exec(
                    select(ModelPreheatDistributionPolicyRun).where(
                        ModelPreheatDistributionPolicyRun.state
                        == ModelPreheatDistributionPolicyRunStateEnum.PENDING,
                        (
                            ModelPreheatDistributionPolicyRun.lease_expires_at.is_(None)
                            | (ModelPreheatDistributionPolicyRun.lease_expires_at <= now)
                        ),
                    )
                )
            ).all()
            for run in runs:
                tasks = (
                    await session.exec(
                        select(ModelPreheatWorkerTask)
                        .join(
                            ModelPreheatDistributionPolicyRunTask,
                            ModelPreheatDistributionPolicyRunTask.task_id
                            == ModelPreheatWorkerTask.id,
                        )
                        .where(ModelPreheatDistributionPolicyRunTask.run_id == run.id)
                    )
                ).all()
                if not tasks or any(task.state not in terminal for task in tasks):
                    continue
                ready_count = sum(
                    task.state == ModelPreheatWorkerTaskStateEnum.READY
                    for task in tasks
                )
                run.state = (
                    ModelPreheatDistributionPolicyRunStateEnum.READY
                    if ready_count
                    else ModelPreheatDistributionPolicyRunStateEnum.ERROR
                )
                task_error_code = next(
                    (task.error_code for task in tasks if task.error_code), None
                )
                if task_error_code is not None:
                    run.error_code = task_error_code
                elif run.state == ModelPreheatDistributionPolicyRunStateEnum.ERROR:
                    run.error_code = (
                        run.error_code or "distribution_run_failed"
                    )
                run.finished_at = max(
                    (task.finished_at for task in tasks if task.finished_at is not None),
                    default=now,
                )
                run.lease_owner = None
                run.lease_token = None
                run.lease_expires_at = None
                session.add(run)
            await session.commit()


async def _ensure_policy(session, task):
    profile = await session.get(
        ModelPreheatS3Profile, task.s3_profile_id, populate_existing=True
    )
    if (
        profile is None
        or profile.lifecycle_state != ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE
        or task.s3_profile_config_version != profile.config_version
    ):
        return None
    source_artifact = (
        await session.exec(
            select(ModelPreheatArtifact).where(
                ModelPreheatArtifact.profile_id == task.s3_profile_id,
                ModelPreheatArtifact.profile_config_version
                == task.s3_profile_config_version,
                ModelPreheatArtifact.artifact_id == task.artifact_id,
            )
        )
    ).first()
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
        manually_sourced = existing.source_sync_task_id is not None or (
            existing.source_artifact_id is not None
            and existing.created_by_task_id is None
        )
        if manually_sourced:
            return existing
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
                    source_artifact_id=(
                        source_artifact.id if source_artifact is not None else None
                    ),
                    source_sync_task_id=None,
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
        selection_mode=ModelPreheatDistributionSelectionModeEnum.FIXED,
        profile_id=task.s3_profile_id,
        profile_config_version=task.s3_profile_config_version,
        request_identity=task.request_identity,
        request_digest=task.request_digest,
        target_scope=task.target_scope,
        worker_selector=worker_selector,
        gpu_selector=gpu_selector,
        selector_digest=selector_digest,
        created_by_task_id=task.id,
        source_artifact_id=(
            source_artifact.id if source_artifact is not None else None
        ),
        trigger_mode=ModelPreheatDistributionPolicyTriggerModeEnum.CONTINUOUS,
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
    if worker.model_storage_protocol_version != MODEL_STORAGE_PROTOCOL_VERSION:
        return False
    return _policy_selector_matches_worker(policy, worker)


def _policy_selector_matches_worker(policy, worker):
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


async def _create_or_rebind_distribution_task(
    session, policy, source, worker, *, run_key=None
):
    source_artifact_id = (
        source.artifact.artifact_id
        if hasattr(source, "artifact")
        else source.artifact_id
    )
    source_request_digest = (
        source.payload["request_digest"]
        if hasattr(source, "payload")
        else source.request_digest
    )
    operation_key = distribution_operation_key(
        policy.id,
        worker.worker_uuid,
        source_request_digest,
        run_key,
        artifact_id=source_artifact_id,
    )
    task = (
        await session.exec(
            select(ModelPreheatWorkerTask).where(
                ModelPreheatWorkerTask.operation_key == operation_key
            )
        )
    ).first()
    if (
        task is None
        and policy.selection_mode == ModelPreheatDistributionSelectionModeEnum.FIXED
    ):
        legacy_operation_key = distribution_operation_key(
            policy.id, worker.worker_uuid, source_request_digest, run_key
        )
        task = (
            await session.exec(
                select(ModelPreheatWorkerTask).where(
                    ModelPreheatWorkerTask.operation_key == legacy_operation_key
                )
            )
        ).first()
    if task is None:
        slot_claim = await _claim_distribution_worker_slot(
            session, policy.id, source_artifact_id, worker.worker_uuid, operation_key
        )
        if slot_claim is not True:
            return slot_claim
        task = ModelPreheatWorkerTask(
            distribution_policy_id=policy.id,
            distribution_artifact_id=source_artifact_id,
            distribution_request_digest=source_request_digest,
            operation_key=operation_key,
            parent_attempt=source.attempt,
            worker_uuid=worker.worker_uuid,
            worker_id=worker.id,
            role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
        )
        session.add(task)
        await session.flush()
        bound = await session.exec(
            update(ModelPreheatDistributionWorkerSlot)
            .where(
                ModelPreheatDistributionWorkerSlot.policy_id == policy.id,
                ModelPreheatDistributionWorkerSlot.artifact_id == source_artifact_id,
                ModelPreheatDistributionWorkerSlot.worker_uuid == worker.worker_uuid,
                ModelPreheatDistributionWorkerSlot.active_task_id.is_(None),
                ModelPreheatDistributionWorkerSlot.active_operation_key
                == operation_key,
            )
            .values(active_task_id=task.id)
        )
        if bound.rowcount != 1:
            await session.delete(task)
            await session.flush()
            return await _active_slot_task(
                session, policy.id, source_artifact_id, worker.worker_uuid
            )
        return task
    if task.worker_id != worker.id:
        task.worker_id = worker.id
        task.parent_attempt = source.attempt
        task.state = ModelPreheatWorkerTaskStateEnum.PENDING
        task.lease_owner = None
        task.lease_token_hash = None
        task.lease_expires_at = None
        task.error_code = None
        task.state_message = None
        task.finished_at = None
        session.add(task)
    elif task.state == ModelPreheatWorkerTaskStateEnum.ERROR:
        await _retry_distribution_error(session, task, source, worker)
    if task.distribution_artifact_id is None:
        task.distribution_artifact_id = source_artifact_id
        task.distribution_request_digest = source_request_digest
        session.add(task)
    return task


async def _claim_distribution_worker_slot(
    session, policy_id, artifact_id, worker_uuid, operation_key
):
    slot = await _distribution_worker_slot(session, policy_id, artifact_id, worker_uuid)
    if slot is None:
        try:
            async with session.begin_nested():
                session.add(
                    ModelPreheatDistributionWorkerSlot(
                        policy_id=policy_id,
                        artifact_id=artifact_id,
                        worker_uuid=worker_uuid,
                        active_operation_key=operation_key,
                    )
                )
                await session.flush()
            return True
        except IntegrityError:
            slot = await _distribution_worker_slot(
                session, policy_id, artifact_id, worker_uuid
            )
            if slot is None:
                return None

    if slot.active_task_id is None:
        if slot.active_operation_key == operation_key:
            return True
        if slot.active_operation_key is not None:
            return await _active_slot_task(session, policy_id, artifact_id, worker_uuid)
    else:
        active_task = await session.get(ModelPreheatWorkerTask, slot.active_task_id)
        if (
            active_task is not None
            and active_task.state in ACTIVE_DISTRIBUTION_TASK_STATES
        ):
            return active_task

    old_task_id = slot.active_task_id
    old_operation_key = slot.active_operation_key
    conditions = [ModelPreheatDistributionWorkerSlot.id == slot.id]
    conditions.append(
        ModelPreheatDistributionWorkerSlot.active_task_id.is_(None)
        if old_task_id is None
        else ModelPreheatDistributionWorkerSlot.active_task_id == old_task_id
    )
    conditions.append(
        ModelPreheatDistributionWorkerSlot.active_operation_key.is_(None)
        if old_operation_key is None
        else ModelPreheatDistributionWorkerSlot.active_operation_key
        == old_operation_key
    )
    claimed = await session.exec(
        update(ModelPreheatDistributionWorkerSlot)
        .where(*conditions)
        .values(active_task_id=None, active_operation_key=operation_key)
    )
    if claimed.rowcount == 1:
        return True
    return await _active_slot_task(session, policy_id, artifact_id, worker_uuid)


async def _distribution_worker_slot(session, policy_id, artifact_id, worker_uuid):
    return (
        await session.exec(
            select(ModelPreheatDistributionWorkerSlot).where(
                ModelPreheatDistributionWorkerSlot.policy_id == policy_id,
                ModelPreheatDistributionWorkerSlot.artifact_id == artifact_id,
                ModelPreheatDistributionWorkerSlot.worker_uuid == worker_uuid,
            )
        )
    ).first()


async def _active_slot_task(session, policy_id, artifact_id, worker_uuid):
    slot = await _distribution_worker_slot(session, policy_id, artifact_id, worker_uuid)
    if slot is not None and slot.active_task_id is not None:
        task = await session.get(ModelPreheatWorkerTask, slot.active_task_id)
        if task is not None and task.state in ACTIVE_DISTRIBUTION_TASK_STATES:
            return task
    return None


async def _retry_distribution_error(session, task, source, worker):
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
            ModelPreheatWorkerTask.parent_attempt == source.attempt,
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


async def _record_worker_state(session, worker, *, force_not_ready=False):
    fingerprint = _network_fingerprint(worker)
    observation = await session.get(ModelPreheatWorkerObservation, worker.worker_uuid)
    is_ready = worker.state == WorkerStateEnum.READY and not force_not_ready
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
