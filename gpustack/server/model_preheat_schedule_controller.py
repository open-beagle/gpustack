import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import exists, or_, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.model_preheat_credentials import (
    CredentialEncryptionUnavailable,
    ModelPreheatCredentialCipher,
)
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
)
from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum
from gpustack.schemas.model_preheat_schedules import (
    ModelPreheatSchedule,
    ModelPreheatScheduleRun,
    ModelPreheatScheduleRunStateEnum,
    ModelPreheatScheduleRunTriggerEnum,
    ModelPreheatScheduleTriggerModeEnum,
    next_window_start_utc,
    window_end_utc,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatDesiredStateEnum,
    ModelPreheatCreate,
    ModelPreheatExecutionStateEnum,
    ModelPreheatTask,
    ModelPreheatTaskLock,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskStateEnum,
    is_terminal_task,
    operation_key_for,
    selection_digest,
)
from gpustack.server.model_preheat_connectivity import current_ready_workers
from gpustack.server.model_preheat_revision import resolve_model_preheat_revision
from gpustack.server.model_preheat_s3_profile_lifecycle import (
    ModelPreheatS3ProfileNotActive,
    lock_active_profile_for_new_work,
)
from gpustack.server.bus import EventType
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity


logger = logging.getLogger(__name__)


class ScheduleRunConflict(Exception):
    pass


class ScheduleConcurrencyLimit(Exception):
    pass


class ScheduleDisabled(RuntimeError):
    pass


TASK_CREATION_ERROR_CODES = {
    "credential_encryption_unavailable",
    "model_preheat_s3_profile_in_maintenance",
    "model_preheat_s3_profile_not_found",
    "model_preheat_schedule_config_required",
    "seed_worker_not_idle",
    "seed_worker_not_online",
    "target_workers_not_idle",
    "target_workers_not_online",
}


def manual_run_operation_key(user_id: int, idempotency_key: str | None) -> str:
    if not idempotency_key:
        return uuid4().hex
    payload = json.dumps(
        ["model_preheat_schedule.run_now", user_id, idempotency_key],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scheduled_run_operation_key(schedule_id: int, window_start_utc: datetime) -> str:
    payload = json.dumps(
        [
            "model_preheat_schedule.scheduled",
            schedule_id,
            window_start_utc.astimezone(timezone.utc).isoformat(),
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ModelPreheatScheduleController:
    def __init__(
        self,
        engine,
        task_creator=None,
        config=None,
        interval=15,
        claim_hook=None,
    ):
        self._engine = engine
        self._config = config
        self._task_creator = task_creator or self._create_task
        self._interval = interval
        self._claim_hook = claim_hook

    async def start(self):
        while True:
            await self.tick()
            await asyncio.sleep(self._interval)

    async def tick(self, now: datetime | None = None):
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        await self._close_active_windows(now)
        async with AsyncSession(self._engine) as session:
            schedule_ids = (
                await session.exec(
                    select(ModelPreheatSchedule.id).where(
                        ModelPreheatSchedule.enabled.is_(True),
                        ModelPreheatSchedule.trigger_mode
                        == ModelPreheatScheduleTriggerModeEnum.SCHEDULED,
                    )
                )
            ).all()
        for schedule_id in schedule_ids:
            await self._trigger_due_schedule(schedule_id, now)

    async def _create_task(self, session, schedule, created_by_user_id):
        if self._config is None:
            raise RuntimeError("model_preheat_schedule_config_required")
        return await create_scheduled_model_preheat_task(
            session,
            schedule,
            created_by_user_id,
            self._config,
        )

    async def run_now(
        self,
        session,
        schedule: ModelPreheatSchedule,
        created_by_user_id: int,
        idempotency_key: str | None,
        now: datetime | None = None,
    ) -> ModelPreheatScheduleRun:
        if not schedule.enabled:
            raise ScheduleDisabled("model_preheat_schedule_disabled")
        operation_key = manual_run_operation_key(created_by_user_id, idempotency_key)
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        schedule_id = schedule.id
        last_retry_error = None
        for _ in range(schedule.max_concurrency + 2):
            existing = await self._run_for_operation(session, operation_key)
            if existing is not None:
                if existing.schedule_id != schedule_id:
                    raise ScheduleRunConflict
                if existing.state == ModelPreheatScheduleRunStateEnum.PENDING and (
                    existing.task_id is None
                ):
                    await session.delete(existing)
                    await session.commit()
                    continue
                return existing
            try:
                task = await self._resume_paused_task(session, schedule_id, now)
                pause_ack_pending_id = None
                creator_savepoint = None
                if task is None:
                    pause_ack_pending = await self._pause_ack_pending_task(
                        session, schedule_id
                    )
                    if pause_ack_pending is not None:
                        pause_ack_pending_id = pause_ack_pending.id
                    else:
                        task, pause_ack_pending_id, creator_savepoint = (
                            await self._create_task_behind_pause_gate(
                                session,
                                schedule,
                                created_by_user_id,
                            )
                        )
                schedule = await session.get(
                    ModelPreheatSchedule, schedule_id, populate_existing=True
                )
                if schedule is None:
                    raise ScheduleRunConflict
                if pause_ack_pending_id is not None:
                    run = ModelPreheatScheduleRun(
                        schedule_id=schedule_id,
                        window_start_utc=now,
                        window_end_utc=window_end_utc(
                            now, schedule.window_duration_minutes
                        ),
                        trigger=ModelPreheatScheduleRunTriggerEnum.MANUAL,
                        state=ModelPreheatScheduleRunStateEnum.SKIPPED,
                        operation_key=operation_key,
                        task_id=pause_ack_pending_id,
                        error_code="pause_ack_pending",
                        finished_at=now,
                        created_by_user_id=created_by_user_id,
                    )
                    session.add(run)
                    await session.commit()
                    await session.refresh(run)
                    return run
                slot = await self._available_slot(session, schedule)
                if slot is None:
                    raise ScheduleConcurrencyLimit
                run = ModelPreheatScheduleRun(
                    schedule_id=schedule_id,
                    window_start_utc=now,
                    window_end_utc=window_end_utc(
                        now, schedule.window_duration_minutes
                    ),
                    trigger=ModelPreheatScheduleRunTriggerEnum.MANUAL,
                    state=ModelPreheatScheduleRunStateEnum.PENDING,
                    operation_key=operation_key,
                    slot=slot,
                    created_by_user_id=created_by_user_id,
                )
                session.add(run)
                await session.flush()
                run.task_id = task.id
                if await self._task_is_owned_by_run(session, run, task, schedule_id):
                    run.state = ModelPreheatScheduleRunStateEnum.RUNNING
                    run.started_at = now
                else:
                    run.state = ModelPreheatScheduleRunStateEnum.SKIPPED
                    run.error_code = "active_operation_deduplicated"
                    run.finished_at = now
                    run.slot = None
                session.add(run)
                if creator_savepoint is not None:
                    await creator_savepoint.commit()
                await session.commit()
                await session.refresh(run)
                return run
            except ScheduleConcurrencyLimit:
                await session.rollback()
                raise
            except RuntimeError as exc:
                await session.rollback()
                existing = await self._run_for_operation(session, operation_key)
                if existing is not None:
                    if existing.schedule_id != schedule_id:
                        raise ScheduleRunConflict
                    return existing
                schedule = await session.get(ModelPreheatSchedule, schedule_id)
                if schedule is None:
                    raise ScheduleRunConflict
                return await self._persist_failed_manual_run(
                    session,
                    schedule,
                    operation_key,
                    created_by_user_id,
                    now,
                    _task_creation_error_code(exc),
                )
            except (IntegrityError, OperationalError) as exc:
                last_retry_error = exc
                await session.rollback()
                existing = await self._run_for_operation(session, operation_key)
                if existing is not None:
                    if existing.schedule_id != schedule_id:
                        raise ScheduleRunConflict
                    return existing
                schedule = await session.get(ModelPreheatSchedule, schedule_id)
                if schedule is None:
                    raise ScheduleRunConflict
            except Exception:
                await session.rollback()
                raise
        if isinstance(last_retry_error, OperationalError):
            raise last_retry_error
        raise ScheduleConcurrencyLimit

    async def _persist_failed_manual_run(
        self,
        session,
        schedule,
        operation_key,
        created_by_user_id,
        now,
        error_code,
    ):
        run = ModelPreheatScheduleRun(
            schedule_id=schedule.id,
            window_start_utc=now,
            window_end_utc=window_end_utc(now, schedule.window_duration_minutes),
            trigger=ModelPreheatScheduleRunTriggerEnum.MANUAL,
            state=ModelPreheatScheduleRunStateEnum.ERROR,
            operation_key=operation_key,
            error_code=error_code,
            finished_at=now,
            created_by_user_id=created_by_user_id,
        )
        session.add(run)
        try:
            await session.commit()
            await session.refresh(run)
            return run
        except IntegrityError:
            await session.rollback()
            existing = await self._run_for_operation(session, operation_key)
            if existing is None or existing.schedule_id != schedule.id:
                raise ScheduleRunConflict
            return existing

    async def _create_task_behind_pause_gate(
        self,
        session,
        schedule,
        created_by_user_id,
    ):
        savepoint = await session.begin_nested()
        try:
            task = await self._task_creator(session, schedule, created_by_user_id)
            pause_ack_pending = await self._pause_ack_pending_task(session, schedule.id)
            if pause_ack_pending is not None:
                pause_ack_pending_id = pause_ack_pending.id
                await savepoint.rollback()
                return None, pause_ack_pending_id, None
            return task, None, savepoint
        except BaseException:
            if savepoint.is_active:
                await savepoint.rollback()
            raise

    async def _close_active_windows(self, now):
        async with AsyncSession(self._engine) as session:
            pause_notifications = []
            runs = (
                await session.exec(
                    select(ModelPreheatScheduleRun).where(
                        ModelPreheatScheduleRun.state.in_(
                            [
                                ModelPreheatScheduleRunStateEnum.PENDING,
                                ModelPreheatScheduleRunStateEnum.RUNNING,
                                ModelPreheatScheduleRunStateEnum.PAUSED,
                            ]
                        )
                    )
                )
            ).all()
            for run in runs:
                task = (
                    await session.get(ModelPreheatTask, run.task_id)
                    if run.task_id is not None
                    else None
                )
                if task is not None and is_terminal_task(task):
                    run.state = (
                        ModelPreheatScheduleRunStateEnum.READY
                        if task.execution_state
                        in {
                            ModelPreheatExecutionStateEnum.READY,
                            ModelPreheatExecutionStateEnum.PARTIAL,
                        }
                        else ModelPreheatScheduleRunStateEnum.ERROR
                    )
                    run.error_code = (
                        None
                        if run.state == ModelPreheatScheduleRunStateEnum.READY
                        else task.state_message or "model_preheat_task_failed"
                    )
                    run.finished_at = now
                    run.slot = None
                    session.add(run)
                    continue
                if run.window_end_utc > now:
                    continue
                if task is None:
                    run.state = ModelPreheatScheduleRunStateEnum.ERROR
                    run.error_code = "model_preheat_task_not_found"
                else:
                    paused = await self._pause_task(session, task, run, now)
                    if paused:
                        run.state = ModelPreheatScheduleRunStateEnum.PAUSED
                        pause_notifications.extend(
                            (
                                await session.exec(
                                    select(ModelPreheatWorkerTask.id).where(
                                        ModelPreheatWorkerTask.task_id == task.id,
                                        ModelPreheatWorkerTask.parent_attempt
                                        == task.attempt,
                                        ModelPreheatWorkerTask.state
                                        == ModelPreheatWorkerTaskStateEnum.RUNNING,
                                        ModelPreheatWorkerTask.state_message
                                        == "pause_requested",
                                        ModelPreheatWorkerTask.lease_token_hash.is_not(
                                            None
                                        ),
                                    )
                                )
                            ).all()
                        )
                    else:
                        await session.refresh(task)
                        if is_terminal_task(task):
                            self._finish_run_from_task(run, task, now)
                            session.add(run)
                            continue
                        continue
                run.finished_at = now
                run.slot = None
                session.add(run)
            await session.commit()
            for worker_task_id in pause_notifications:
                worker_task = await session.get(
                    ModelPreheatWorkerTask,
                    worker_task_id,
                    populate_existing=True,
                )
                if worker_task is not None:
                    await ModelPreheatWorkerTask._publish_event(
                        EventType.UPDATED, worker_task
                    )

    async def _trigger_due_schedule(self, schedule_id, now):
        while True:
            async with AsyncSession(self._engine) as session:
                schedule = await session.get(ModelPreheatSchedule, schedule_id)
                if (
                    schedule is None
                    or not schedule.enabled
                    or schedule.trigger_mode
                    != ModelPreheatScheduleTriggerModeEnum.SCHEDULED
                ):
                    return
                window_start = schedule.next_window_start_utc
                if window_start is None:
                    schedule.next_window_start_utc = next_window_start_utc(
                        schedule, now
                    )
                    session.add(schedule)
                    await session.commit()
                    return
                if window_start > now:
                    return
                if self._claim_hook is not None:
                    await self._claim_hook(schedule.id, window_start)
                try:
                    started = await self._start_scheduled_window(
                        session, schedule, window_start, now
                    )
                    if started is None:
                        return
                except (IntegrityError, OperationalError):
                    await session.rollback()
                    return

    async def _start_scheduled_window(self, session, schedule, window_start, now):
        schedule_id = schedule.id
        existing = (
            await session.exec(
                select(ModelPreheatScheduleRun).where(
                    ModelPreheatScheduleRun.schedule_id == schedule.id,
                    ModelPreheatScheduleRun.window_start_utc == window_start,
                )
            )
        ).first()
        if existing is not None:
            claimed = await self._claim_schedule_cursor(
                session,
                schedule,
                window_start,
                next_window_start_utc(schedule, window_start),
            )
            if claimed:
                await session.commit()
            return existing

        next_start = next_window_start_utc(schedule, window_start)
        window_end = window_end_utc(window_start, schedule.window_duration_minutes)
        if window_end <= now:
            if not await self._claim_schedule_cursor(
                session, schedule, window_start, next_start
            ):
                await session.rollback()
                return None
            run = ModelPreheatScheduleRun(
                schedule_id=schedule.id,
                window_start_utc=window_start,
                window_end_utc=window_end,
                trigger=ModelPreheatScheduleRunTriggerEnum.SCHEDULED,
                state=ModelPreheatScheduleRunStateEnum.SKIPPED,
                operation_key=scheduled_run_operation_key(schedule.id, window_start),
                error_code="schedule_window_elapsed",
                finished_at=now,
            )
            session.add(run)
            await session.commit()
            return run
        schedule = await session.get(ModelPreheatSchedule, schedule_id)
        if (
            schedule is None
            or not schedule.enabled
            or schedule.trigger_mode != ModelPreheatScheduleTriggerModeEnum.SCHEDULED
        ):
            await session.rollback()
            return None
        slot = await self._available_slot(session, schedule)
        if not await self._claim_schedule_cursor(
            session, schedule, window_start, next_start
        ):
            await session.rollback()
            return None
        if slot is None:
            run = ModelPreheatScheduleRun(
                schedule_id=schedule_id,
                window_start_utc=window_start,
                window_end_utc=window_end,
                trigger=ModelPreheatScheduleRunTriggerEnum.SCHEDULED,
                state=ModelPreheatScheduleRunStateEnum.SKIPPED,
                operation_key=scheduled_run_operation_key(schedule_id, window_start),
                error_code="schedule_concurrency_limit",
                finished_at=now,
                created_by_user_id=schedule.created_by_user_id,
            )
            session.add(run)
            await session.commit()
            return run
        try:
            task = await self._resume_paused_task(session, schedule.id, now)
            pause_ack_pending_id = None
            creator_savepoint = None
            if task is None:
                pause_ack_pending = await self._pause_ack_pending_task(
                    session, schedule.id
                )
                if pause_ack_pending is not None:
                    pause_ack_pending_id = pause_ack_pending.id
                else:
                    task, pause_ack_pending_id, creator_savepoint = (
                        await self._create_task_behind_pause_gate(
                            session,
                            schedule,
                            schedule.created_by_user_id,
                        )
                    )
        except (IntegrityError, OperationalError):
            raise
        except Exception as exc:
            logger.warning(
                "模型预热调度创建任务失败。schedule_id=%s error_type=%s",
                schedule_id,
                type(exc).__name__,
            )
            await session.rollback()
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            if (
                schedule is None
                or schedule.last_window_start_utc != window_start
                or schedule.next_window_start_utc != next_start
            ):
                await session.rollback()
                return None
            run = ModelPreheatScheduleRun(
                schedule_id=schedule_id,
                window_start_utc=window_start,
                window_end_utc=window_end,
                trigger=ModelPreheatScheduleRunTriggerEnum.SCHEDULED,
                state=ModelPreheatScheduleRunStateEnum.ERROR,
                operation_key=scheduled_run_operation_key(schedule_id, window_start),
                error_code="schedule_task_creation_failed",
                finished_at=now,
                created_by_user_id=schedule.created_by_user_id,
            )
            session.add(run)
            await session.commit()
            return run
        schedule = await session.get(
            ModelPreheatSchedule, schedule_id, populate_existing=True
        )
        if schedule is None:
            await session.rollback()
            return None
        if pause_ack_pending_id is not None:
            run = ModelPreheatScheduleRun(
                schedule_id=schedule.id,
                window_start_utc=window_start,
                window_end_utc=window_end,
                trigger=ModelPreheatScheduleRunTriggerEnum.SCHEDULED,
                state=ModelPreheatScheduleRunStateEnum.SKIPPED,
                operation_key=scheduled_run_operation_key(schedule.id, window_start),
                task_id=pause_ack_pending_id,
                error_code="pause_ack_pending",
                finished_at=now,
                created_by_user_id=schedule.created_by_user_id,
            )
            session.add(run)
            await session.commit()
            return run
        run = ModelPreheatScheduleRun(
            schedule_id=schedule.id,
            window_start_utc=window_start,
            window_end_utc=window_end,
            trigger=ModelPreheatScheduleRunTriggerEnum.SCHEDULED,
            state=ModelPreheatScheduleRunStateEnum.PENDING,
            operation_key=scheduled_run_operation_key(schedule.id, window_start),
            slot=slot,
            created_by_user_id=schedule.created_by_user_id,
        )
        session.add(run)
        await session.flush()
        run.task_id = task.id
        if slot is None:
            run.state = ModelPreheatScheduleRunStateEnum.SKIPPED
            run.error_code = "schedule_concurrency_limit"
            run.finished_at = now
            run.slot = None
        elif await self._task_is_owned_by_run(session, run, task, schedule.id):
            run.state = ModelPreheatScheduleRunStateEnum.RUNNING
            run.started_at = now
        else:
            run.state = ModelPreheatScheduleRunStateEnum.SKIPPED
            run.error_code = "active_operation_deduplicated"
            run.finished_at = now
            run.slot = None
        session.add(run)
        if creator_savepoint is not None:
            await creator_savepoint.commit()
        await session.commit()
        return run

    async def _claim_schedule_cursor(self, session, schedule, window_start, next_start):
        result = await session.exec(
            update(ModelPreheatSchedule)
            .where(
                ModelPreheatSchedule.id == schedule.id,
                ModelPreheatSchedule.enabled.is_(True),
                ModelPreheatSchedule.trigger_mode
                == ModelPreheatScheduleTriggerModeEnum.SCHEDULED,
                ModelPreheatSchedule.next_window_start_utc == window_start,
            )
            .values(
                last_window_start_utc=window_start,
                next_window_start_utc=next_start,
            )
            .execution_options(synchronize_session=False)
        )
        return result.rowcount == 1

    async def _pause_task(self, session, task, run, now=None):
        now = now or datetime.now(timezone.utc)
        if task.desired_state == ModelPreheatDesiredStateEnum.PAUSED:
            owned = await session.exec(
                select(ModelPreheatScheduleRun.id).where(
                    ModelPreheatScheduleRun.id == run.id,
                    ModelPreheatScheduleRun.task_id == task.id,
                    ModelPreheatScheduleRun.state
                    == ModelPreheatScheduleRunStateEnum.PAUSED,
                )
            )
            if owned.first() is None:
                return False
            await self._reap_expired_pause_requests(session, task, now)
            return True
        result = await session.exec(
            update(ModelPreheatTask)
            .where(
                ModelPreheatTask.id == task.id,
                ModelPreheatTask.attempt == task.attempt,
                ModelPreheatTask.desired_state == ModelPreheatDesiredStateEnum.RUNNING,
                ModelPreheatTask.execution_state.not_in(
                    [
                        ModelPreheatExecutionStateEnum.READY,
                        ModelPreheatExecutionStateEnum.PARTIAL,
                        ModelPreheatExecutionStateEnum.ERROR,
                        ModelPreheatExecutionStateEnum.CANCELED,
                    ]
                ),
                exists().where(
                    ModelPreheatScheduleRun.id == run.id,
                    ModelPreheatScheduleRun.task_id == ModelPreheatTask.id,
                    ModelPreheatScheduleRun.state.in_(
                        [
                            ModelPreheatScheduleRunStateEnum.PENDING,
                            ModelPreheatScheduleRunStateEnum.RUNNING,
                        ]
                    ),
                ),
            )
            .values(
                paused_from_state=ModelPreheatTask.execution_state,
                desired_state=ModelPreheatDesiredStateEnum.PAUSED,
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            return False
        await session.exec(
            update(ModelPreheatWorkerTask)
            .where(
                ModelPreheatWorkerTask.task_id == task.id,
                ModelPreheatWorkerTask.parent_attempt == task.attempt,
                ModelPreheatWorkerTask.state == ModelPreheatWorkerTaskStateEnum.RUNNING,
                ModelPreheatWorkerTask.lease_token_hash.is_not(None),
                ModelPreheatWorkerTask.lease_expires_at.is_not(None),
                ModelPreheatWorkerTask.lease_expires_at > now,
            )
            .values(state_message="pause_requested")
            .execution_options(synchronize_session=False)
        )
        await session.exec(
            update(ModelPreheatWorkerTask)
            .where(
                ModelPreheatWorkerTask.task_id == task.id,
                ModelPreheatWorkerTask.parent_attempt == task.attempt,
                or_(
                    ModelPreheatWorkerTask.state
                    == ModelPreheatWorkerTaskStateEnum.PENDING,
                    (
                        ModelPreheatWorkerTask.state
                        == ModelPreheatWorkerTaskStateEnum.RUNNING
                    )
                    & or_(
                        ModelPreheatWorkerTask.lease_token_hash.is_(None),
                        ModelPreheatWorkerTask.lease_expires_at.is_(None),
                        ModelPreheatWorkerTask.lease_expires_at <= now,
                    ),
                ),
            )
            .values(
                state=ModelPreheatWorkerTaskStateEnum.PAUSED,
                state_message="paused",
                lease_owner=None,
                lease_token_hash=None,
                lease_expires_at=None,
            )
            .execution_options(synchronize_session=False)
        )
        await self._finish_parent_pause_if_acknowledged(session, task)
        return True

    async def _reap_expired_pause_requests(self, session, task, now):
        parent_locked = await session.exec(
            update(ModelPreheatTask)
            .where(
                ModelPreheatTask.id == task.id,
                ModelPreheatTask.attempt == task.attempt,
                ModelPreheatTask.desired_state == ModelPreheatDesiredStateEnum.PAUSED,
                ModelPreheatTask.execution_state.not_in(
                    [
                        ModelPreheatExecutionStateEnum.READY,
                        ModelPreheatExecutionStateEnum.PARTIAL,
                        ModelPreheatExecutionStateEnum.ERROR,
                        ModelPreheatExecutionStateEnum.CANCELED,
                    ]
                ),
            )
            .values(desired_state=ModelPreheatDesiredStateEnum.PAUSED)
            .execution_options(synchronize_session=False)
        )
        if parent_locked.rowcount != 1:
            return
        await session.exec(
            update(ModelPreheatWorkerTask)
            .where(
                ModelPreheatWorkerTask.task_id == task.id,
                ModelPreheatWorkerTask.parent_attempt == task.attempt,
                ModelPreheatWorkerTask.state == ModelPreheatWorkerTaskStateEnum.RUNNING,
                exists().where(
                    ModelPreheatTask.id == ModelPreheatWorkerTask.task_id,
                    ModelPreheatTask.attempt == ModelPreheatWorkerTask.parent_attempt,
                    ModelPreheatTask.desired_state
                    == ModelPreheatDesiredStateEnum.PAUSED,
                ),
                or_(
                    ModelPreheatWorkerTask.lease_expires_at.is_(None),
                    ModelPreheatWorkerTask.lease_expires_at <= now,
                ),
            )
            .values(
                state=ModelPreheatWorkerTaskStateEnum.PAUSED,
                state_message="paused",
                lease_owner=None,
                lease_token_hash=None,
                lease_expires_at=None,
            )
            .execution_options(synchronize_session=False)
        )
        await self._finish_parent_pause_if_acknowledged(session, task)

    async def _finish_parent_pause_if_acknowledged(self, session, task):
        await session.exec(
            update(ModelPreheatTask)
            .where(
                ModelPreheatTask.id == task.id,
                ModelPreheatTask.attempt == task.attempt,
                ModelPreheatTask.desired_state == ModelPreheatDesiredStateEnum.PAUSED,
                ~exists().where(
                    ModelPreheatWorkerTask.task_id == ModelPreheatTask.id,
                    ModelPreheatWorkerTask.parent_attempt == ModelPreheatTask.attempt,
                    ModelPreheatWorkerTask.state.in_(
                        [
                            ModelPreheatWorkerTaskStateEnum.PENDING,
                            ModelPreheatWorkerTaskStateEnum.RUNNING,
                        ]
                    ),
                ),
            )
            .values(execution_state=ModelPreheatExecutionStateEnum.PAUSED)
            .execution_options(synchronize_session=False)
        )

    async def _pause_ack_pending_task(self, session, schedule_id):
        return (
            await session.exec(
                select(ModelPreheatTask).where(
                    ModelPreheatTask.schedule_id == schedule_id,
                    ModelPreheatTask.desired_state
                    == ModelPreheatDesiredStateEnum.PAUSED,
                    ModelPreheatTask.execution_state
                    != ModelPreheatExecutionStateEnum.PAUSED,
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
        ).first()

    async def _resume_paused_task(self, session, schedule_id, now=None):
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        paused_runs = (
            await session.exec(
                select(ModelPreheatScheduleRun)
                .where(
                    ModelPreheatScheduleRun.schedule_id == schedule_id,
                    ModelPreheatScheduleRun.state
                    == ModelPreheatScheduleRunStateEnum.PAUSED,
                    ModelPreheatScheduleRun.task_id.is_not(None),
                )
                .order_by(ModelPreheatScheduleRun.window_start_utc.desc())
            )
        ).all()
        for paused_run in paused_runs:
            task = await session.get(ModelPreheatTask, paused_run.task_id)
            if (
                task is None
                or task.desired_state != ModelPreheatDesiredStateEnum.PAUSED
                or is_terminal_task(task)
            ):
                continue
            await self._reap_expired_pause_requests(session, task, now)
            task = await session.get(ModelPreheatTask, task.id, populate_existing=True)
            if task.execution_state != ModelPreheatExecutionStateEnum.PAUSED:
                continue
            resumed_state = (
                task.paused_from_state or ModelPreheatExecutionStateEnum.PENDING
            )
            resumed = await session.exec(
                update(ModelPreheatTask)
                .where(
                    ModelPreheatTask.id == task.id,
                    ModelPreheatTask.attempt == task.attempt,
                    ModelPreheatTask.desired_state
                    == ModelPreheatDesiredStateEnum.PAUSED,
                    ModelPreheatTask.execution_state
                    == ModelPreheatExecutionStateEnum.PAUSED,
                )
                .values(
                    desired_state=ModelPreheatDesiredStateEnum.RUNNING,
                    execution_state=resumed_state,
                    paused_from_state=None,
                )
                .execution_options(synchronize_session=False)
            )
            if resumed.rowcount != 1:
                await session.rollback()
                return None
            await session.exec(
                update(ModelPreheatWorkerTask)
                .where(
                    ModelPreheatWorkerTask.task_id == task.id,
                    ModelPreheatWorkerTask.parent_attempt == task.attempt,
                    ModelPreheatWorkerTask.state
                    == ModelPreheatWorkerTaskStateEnum.PAUSED,
                    ModelPreheatWorkerTask.lease_token_hash.is_(None),
                )
                .values(
                    state=ModelPreheatWorkerTaskStateEnum.PENDING,
                    lease_owner=None,
                    lease_token_hash=None,
                    lease_expires_at=None,
                )
                .execution_options(synchronize_session=False)
            )
            paused_run.state = ModelPreheatScheduleRunStateEnum.SKIPPED
            paused_run.error_code = "resumed_in_later_window"
            session.add(paused_run)
            return await session.get(ModelPreheatTask, task.id, populate_existing=True)
        return None

    async def _run_for_operation(self, session, operation_key):
        return (
            await session.exec(
                select(ModelPreheatScheduleRun).where(
                    ModelPreheatScheduleRun.operation_key == operation_key
                )
            )
        ).first()

    async def _task_is_owned_by_run(self, session, run, task, schedule_id):
        if getattr(task, "schedule_id", schedule_id) != schedule_id:
            return False
        runnable_task = (
            await session.exec(
                select(ModelPreheatTask.id).where(
                    ModelPreheatTask.id == task.id,
                    ModelPreheatTask.schedule_id == schedule_id,
                    ModelPreheatTask.desired_state
                    == ModelPreheatDesiredStateEnum.RUNNING,
                    ModelPreheatTask.execution_state.not_in(
                        [
                            ModelPreheatExecutionStateEnum.PAUSED,
                            ModelPreheatExecutionStateEnum.READY,
                            ModelPreheatExecutionStateEnum.PARTIAL,
                            ModelPreheatExecutionStateEnum.ERROR,
                            ModelPreheatExecutionStateEnum.CANCELED,
                        ]
                    ),
                )
            )
        ).first()
        if runnable_task is None:
            return False
        other_active_run = (
            await session.exec(
                select(ModelPreheatScheduleRun.id).where(
                    ModelPreheatScheduleRun.task_id == task.id,
                    ModelPreheatScheduleRun.id != run.id,
                    ModelPreheatScheduleRun.state.in_(
                        [
                            ModelPreheatScheduleRunStateEnum.PENDING,
                            ModelPreheatScheduleRunStateEnum.RUNNING,
                        ]
                    ),
                )
            )
        ).first()
        return other_active_run is None

    async def _available_slot(self, session, schedule):
        occupied = set(
            (
                await session.exec(
                    select(ModelPreheatScheduleRun.slot).where(
                        ModelPreheatScheduleRun.schedule_id == schedule.id,
                        ModelPreheatScheduleRun.slot.is_not(None),
                    )
                )
            ).all()
        )
        if len(occupied) >= schedule.max_concurrency:
            return None
        return next(
            (slot for slot in range(schedule.max_concurrency) if slot not in occupied),
            None,
        )

    @staticmethod
    def _finish_run_from_task(run, task, now):
        run.state = (
            ModelPreheatScheduleRunStateEnum.READY
            if task.execution_state
            in {
                ModelPreheatExecutionStateEnum.READY,
                ModelPreheatExecutionStateEnum.PARTIAL,
            }
            else ModelPreheatScheduleRunStateEnum.ERROR
        )
        run.error_code = (
            None
            if run.state == ModelPreheatScheduleRunStateEnum.READY
            else task.state_message or "model_preheat_task_failed"
        )
        run.finished_at = now
        run.slot = None


def _task_creation_error_code(exc: RuntimeError) -> str:
    code = str(exc) or ""
    return (
        code
        if code in TASK_CREATION_ERROR_CODES
        else "model_preheat_schedule_run_failed"
    )


async def create_scheduled_model_preheat_task(
    session,
    schedule,
    created_by_user_id,
    config,
):
    from gpustack.routes.model_preheats import (
        _active_task_for_operation,
        _ensure_profile_available_on_workers,
        _exact_artifact_match,
        _profile_snapshot,
        _request_identity,
        _resolve_s3_only_seed_worker,
        _resolve_target_workers,
        _target_snapshot,
    )

    ready_workers = await current_ready_workers(session)
    busy_worker_ids = await _busy_worker_ids(session)
    ready_workers_by_uuid = {worker.worker_uuid: worker for worker in ready_workers}
    ready_workers = [
        worker for worker in ready_workers if worker.id not in busy_worker_ids
    ]
    workers_by_uuid = {worker.worker_uuid: worker for worker in ready_workers}
    if schedule.delivery_mode.value == "s3_only":
        # 精确 Artifact 命中不需要任何 Worker；未命中时随后只解析一个 Seed。
        target_worker_ids, seed_worker_id = [], None
    elif schedule.target_scope.value == "selected_workers":
        workers_by_uuid = ready_workers_by_uuid
        missing = set(schedule.target_worker_uuids) - set(workers_by_uuid)
        if missing:
            raise RuntimeError("target_workers_not_online")
        target_worker_ids = [
            workers_by_uuid[worker_uuid].id
            for worker_uuid in schedule.target_worker_uuids
        ]
        seed_worker_id = (
            workers_by_uuid[schedule.seed_worker_uuid].id
            if schedule.seed_worker_uuid
            else None
        )
    else:
        seed = workers_by_uuid.get(schedule.seed_worker_uuid)
        if seed is None:
            if schedule.seed_worker_uuid in ready_workers_by_uuid:
                raise RuntimeError("seed_worker_not_idle")
            raise RuntimeError("seed_worker_not_online")
        target_worker_ids = []
        seed_worker_id = seed.id
    task_in = ModelPreheatCreate(
        source=schedule.source,
        model_id=schedule.model_id,
        revision=schedule.revision,
        include_patterns=schedule.include_patterns,
        exclude_patterns=schedule.exclude_patterns,
        target_scope=schedule.target_scope,
        target_worker_ids=target_worker_ids,
        seed_worker_id=seed_worker_id,
        s3_profile_id=schedule.s3_profile_id,
        s3_backfill_policy=schedule.s3_backfill_policy,
        delivery_mode=schedule.delivery_mode,
        keep_new_workers_in_sync=schedule.keep_new_workers_in_sync,
        connectivity_failure_override=schedule.connectivity_failure_override,
    )
    profile = await session.get(ModelPreheatS3Profile, schedule.s3_profile_id)
    if profile is None:
        raise RuntimeError("model_preheat_s3_profile_not_found")
    if profile.lifecycle_state != ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE:
        raise RuntimeError("model_preheat_s3_profile_in_maintenance")
    resolved_revision = await asyncio.to_thread(
        resolve_model_preheat_revision,
        task_in.source,
        task_in.model_id,
        task_in.revision,
        include_patterns=task_in.include_patterns,
        exclude_patterns=task_in.exclude_patterns,
        token=getattr(config, "huggingface_token", None),
    )
    resolved_revision = resolved_revision or "ollama-pending"
    identity = ModelPreheatIdentity(
        source=task_in.source,
        model_id=task_in.model_id,
        revision=resolved_revision,
        requested_revision=task_in.revision,
        file_patterns=task_in.include_patterns,
        exclude_patterns=task_in.exclude_patterns,
    )
    matched_artifact = await _exact_artifact_match(session, profile, identity)
    if task_in.delivery_mode.value == "s3_only" and matched_artifact is not None:
        workers, seed_worker, target_gpu_names = [], None, []
    elif task_in.delivery_mode.value == "s3_only":
        seed_worker = await _resolve_s3_only_seed_worker(session, task_in)
        if seed_worker.id in busy_worker_ids:
            raise RuntimeError("seed_worker_not_idle")
        workers, target_gpu_names = [], []
        await _ensure_profile_available_on_workers(
            session,
            profile,
            config,
            [seed_worker],
            allow_failure=task_in.connectivity_failure_override,
        )
    else:
        workers, seed_worker, target_gpu_names = await _resolve_target_workers(
            session, task_in
        )
        if task_in.target_scope.value != "selected_workers":
            workers = [worker for worker in workers if worker.id not in busy_worker_ids]
            if not workers:
                raise RuntimeError("target_workers_not_idle")
        await _ensure_profile_available_on_workers(
            session,
            profile,
            config,
            workers,
            allow_failure=task_in.connectivity_failure_override,
        )
    target_snapshot = _target_snapshot(workers)
    target_worker_uuids = [item["worker_uuid"] for item in target_snapshot]
    pattern_digest = selection_digest(
        task_in.include_patterns, task_in.exclude_patterns
    )
    operation_key = operation_key_for(
        profile.id,
        identity.request_digest,
        target_worker_uuids,
        task_in.s3_backfill_policy,
        task_in.delivery_mode,
    )
    existing = await _active_task_for_operation(session, operation_key)
    if existing is not None:
        return existing
    cipher = ModelPreheatCredentialCipher(
        current_key=getattr(config, "model_preheat_credential_key", None),
        current_key_version=getattr(
            config, "model_preheat_credential_key_version", None
        ),
        old_keys=getattr(config, "model_preheat_credential_old_keys", None),
    )
    try:
        profile_snapshot = _profile_snapshot(cipher, profile)
    except CredentialEncryptionUnavailable as exc:
        raise RuntimeError("credential_encryption_unavailable") from exc
    try:
        await lock_active_profile_for_new_work(
            session, profile.id, profile.config_version
        )
    except ModelPreheatS3ProfileNotActive:
        raise RuntimeError("model_preheat_s3_profile_in_maintenance") from None
    task = ModelPreheatTask(
        source=task_in.source,
        model_id=task_in.model_id,
        requested_revision=task_in.revision,
        resolved_revision=resolved_revision,
        include_patterns=task_in.include_patterns,
        exclude_patterns=task_in.exclude_patterns,
        selection_digest=pattern_digest,
        request_identity=_request_identity(identity),
        request_digest=identity.request_digest,
        artifact_id=(matched_artifact.artifact_id if matched_artifact else None),
        seed_worker_uuid=(seed_worker.worker_uuid if seed_worker else None),
        seed_worker_id=(seed_worker.id if seed_worker else None),
        target_scope=task_in.target_scope,
        target_gpu_names=target_gpu_names,
        target_worker_uuids=target_worker_uuids,
        target_worker_snapshot=target_snapshot,
        s3_profile_id=profile.id,
        s3_profile_config_version=profile.config_version,
        s3_profile_snapshot_encrypted=profile_snapshot,
        encryption_key_version=cipher.current_key_version,
        s3_backfill_policy=task_in.s3_backfill_policy,
        delivery_mode=task_in.delivery_mode,
        s3_manifest_path=(matched_artifact.manifest_path if matched_artifact else None),
        manifest_digest=(
            matched_artifact.manifest_digest if matched_artifact else None
        ),
        keep_new_workers_in_sync=task_in.keep_new_workers_in_sync,
        connectivity_failure_override=task_in.connectivity_failure_override,
        bandwidth_limit_mbps=schedule.bandwidth_limit_mbps,
        schedule_id=schedule.id,
        created_by_user_id=created_by_user_id,
    )
    session.add(task)
    await session.flush()
    session.add(
        ModelPreheatTaskLock(
            operation_key=operation_key,
            task_id=task.id,
            lease_expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )
    )
    await session.flush()
    return task


async def _busy_worker_ids(session):
    instances = (
        await session.exec(
            select(ModelInstance).where(
                ModelInstance.state.in_(
                    [
                        ModelInstanceStateEnum.INITIALIZING,
                        ModelInstanceStateEnum.STARTING,
                        ModelInstanceStateEnum.RUNNING,
                    ]
                )
            )
        )
    ).all()
    busy = {
        instance.worker_id for instance in instances if instance.worker_id is not None
    }
    for instance in instances:
        distributed = instance.distributed_servers
        for worker in (
            distributed.subordinate_workers
            if distributed is not None and distributed.subordinate_workers
            else []
        ):
            if worker.worker_id is not None:
                busy.add(worker.worker_id)
    return busy
