import asyncio
from datetime import datetime, timedelta, timezone
import hashlib

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.routes.model_preheats import (
    cancel_model_preheat,
    pause_model_preheat,
    resume_model_preheat,
    retry_model_preheat,
)
from gpustack.routes.model_preheat_worker_tasks import _validate_active_lease
from gpustack.api.exceptions import HTTPException
from gpustack.schemas.model_preheat_schedules import (
    ModelPreheatSchedule,
    ModelPreheatScheduleRun,
    ModelPreheatScheduleRunStateEnum,
    ModelPreheatScheduleRunTriggerEnum,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatDesiredStateEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
    ModelPreheatTaskLock,
    ModelPreheatPublicationMarker,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskLease,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.server.model_preheat_controller import ModelPreheatController
from gpustack.server.model_preheat_s3_inventory import ModelPreheatS3Inventory
from gpustack.server.model_preheat_worker_identity import (
    ModelPreheatWorkerPrincipal,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum


async def _seed(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'actions.db'}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine) as session:
        session.add(
            Worker(
                id=10,
                name="worker-a",
                hostname="worker-a",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="worker-a",
                state=WorkerStateEnum.READY,
            )
        )
        task = ModelPreheatTask(
            source="huggingface",
            model_id="org/model",
            resolved_revision="a" * 40,
            include_patterns=[],
            exclude_patterns=[],
            selection_digest="b" * 64,
            cache_key="c" * 64,
            generation_id="preheat-00000000-0000-4000-8000-000000000001",
            target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
            target_worker_uuids=["worker-a"],
            target_worker_snapshot=[],
            s3_profile_id=1,
            s3_profile_config_version=1,
            s3_profile_snapshot_encrypted={"ciphertext": "encrypted"},
            encryption_key_version="v1",
            s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
            execution_state=ModelPreheatExecutionStateEnum.PUBLISHING,
        )
        session.add(task)
        await session.flush()
        child = ModelPreheatWorkerTask(
            task_id=task.id,
            parent_attempt=1,
            worker_uuid="worker-a",
            worker_id=10,
            role=ModelPreheatWorkerTaskRoleEnum.SEED,
            state=ModelPreheatWorkerTaskStateEnum.RUNNING,
            attempt=3,
            lease_owner="worker-a",
            lease_token_hash=hashlib.sha256(b"token").hexdigest(),
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
        )
        session.add(child)
        session.add(
            ModelPreheatPublicationMarker(
                profile_id=task.s3_profile_id,
                selection_key=task.cache_key,
                generation_id=task.generation_id,
                task_id=task.id,
                parent_attempt=task.attempt,
                profile_config_version=task.s3_profile_config_version,
            )
        )
        await session.commit()
        await session.refresh(task)
        await session.refresh(child)
        return engine, task.id, child.id


async def _invoke(engine, action, task_id):
    async with AsyncSession(engine) as session:
        return await action(session=session, id=task_id)


async def _stored(engine, task_id):
    async with AsyncSession(engine) as session:
        parent = await session.get(ModelPreheatTask, task_id)
        children = (
            await session.exec(
                select(ModelPreheatWorkerTask).where(
                    ModelPreheatWorkerTask.task_id == task_id
                )
            )
        ).all()
        return parent, children


def test_pause_and_resume_are_idempotent_and_restore_persisted_state(tmp_path):
    async def run():
        engine, task_id, _ = await _seed(tmp_path)
        await _invoke(engine, pause_model_preheat, task_id)
        await _invoke(engine, pause_model_preheat, task_id)
        paused = await _stored(engine, task_id)
        await _invoke(engine, resume_model_preheat, task_id)
        await _invoke(engine, resume_model_preheat, task_id)
        resumed = await _stored(engine, task_id)
        await engine.dispose()
        return paused, resumed

    (paused_parent, paused_children), (resumed_parent, resumed_children) = asyncio.run(
        run()
    )
    assert paused_parent.desired_state == ModelPreheatDesiredStateEnum.PAUSED
    assert paused_parent.execution_state == ModelPreheatExecutionStateEnum.PAUSED
    assert paused_parent.paused_from_state == ModelPreheatExecutionStateEnum.PUBLISHING
    assert paused_children[0].state == ModelPreheatWorkerTaskStateEnum.PAUSED
    assert paused_children[0].lease_owner is None
    assert paused_children[0].lease_token_hash is None
    assert paused_children[0].lease_expires_at is None
    assert resumed_parent.desired_state == ModelPreheatDesiredStateEnum.RUNNING
    assert resumed_parent.execution_state == ModelPreheatExecutionStateEnum.PUBLISHING
    assert resumed_parent.paused_from_state is None
    assert resumed_children[0].state == ModelPreheatWorkerTaskStateEnum.PENDING


def test_cancel_clears_all_active_child_leases_and_is_idempotent(tmp_path):
    async def run():
        engine, task_id, _ = await _seed(tmp_path)
        await _invoke(engine, cancel_model_preheat, task_id)
        await _invoke(engine, cancel_model_preheat, task_id)
        stored = await _stored(engine, task_id)
        await engine.dispose()
        return stored

    parent, children = asyncio.run(run())
    assert parent.execution_state == ModelPreheatExecutionStateEnum.CANCELED
    assert children[0].state == ModelPreheatWorkerTaskStateEnum.CANCELED
    assert children[0].lease_owner is None
    assert children[0].lease_token_hash is None
    assert children[0].lease_expires_at is None


def test_cancel_blocked_seed_marks_marker_terminated_without_deleting_it(tmp_path):
    async def run():
        engine, task_id, _ = await _seed(tmp_path)
        await _invoke(engine, cancel_model_preheat, task_id)
        controller = ModelPreheatController(
            engine, s3_inventory=ModelPreheatS3Inventory(engine)
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            marker = (await session.exec(select(ModelPreheatPublicationMarker))).one()
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            result = marker.terminated_at, child.lease_expires_at
        await engine.dispose()
        return result

    terminated_at, lease_expires_at = asyncio.run(run())
    assert terminated_at is not None
    assert lease_expires_at is None


def test_retry_increments_parent_attempt_keeps_generation_and_history(tmp_path):
    async def run():
        engine, task_id, _ = await _seed(tmp_path)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            parent.execution_state = ModelPreheatExecutionStateEnum.ERROR
            session.add(parent)
            await session.commit()
        first = await _invoke(engine, retry_model_preheat, task_id)
        second = await _invoke(engine, retry_model_preheat, task_id)
        stored = await _stored(engine, task_id)
        async with AsyncSession(engine) as session:
            locks = (
                await session.exec(
                    select(ModelPreheatTaskLock).where(
                        ModelPreheatTaskLock.task_id == task_id
                    )
                )
            ).all()
        await engine.dispose()
        return first, second, stored, locks

    first, second, (parent, children), locks = asyncio.run(run())
    assert first.attempt == 2
    assert second.attempt == 2
    assert parent.generation_id == "preheat-00000000-0000-4000-8000-000000000001"
    assert parent.execution_state == ModelPreheatExecutionStateEnum.PENDING
    assert [child.parent_attempt for child in children] == [1]
    assert len(locks) == 1


@pytest.mark.parametrize(
    ("action", "task_desired", "task_execution", "run_state", "run_slot"),
    [
        (
            pause_model_preheat,
            ModelPreheatDesiredStateEnum.RUNNING,
            ModelPreheatExecutionStateEnum.PUBLISHING,
            ModelPreheatScheduleRunStateEnum.RUNNING,
            0,
        ),
        (
            resume_model_preheat,
            ModelPreheatDesiredStateEnum.PAUSED,
            ModelPreheatExecutionStateEnum.PAUSED,
            ModelPreheatScheduleRunStateEnum.PAUSED,
            None,
        ),
        (
            retry_model_preheat,
            ModelPreheatDesiredStateEnum.RUNNING,
            ModelPreheatExecutionStateEnum.ERROR,
            ModelPreheatScheduleRunStateEnum.ERROR,
            None,
        ),
    ],
)
def test_common_actions_reject_schedule_managed_tasks_without_breaking_run_invariants(
    tmp_path,
    action,
    task_desired,
    task_execution,
    run_state,
    run_slot,
):
    async def run():
        engine, task_id, child_id = await _seed(tmp_path)
        async with AsyncSession(engine) as session:
            schedule = ModelPreheatSchedule(
                name="managed",
                cron_expression="0 * * * *",
                timezone="UTC",
                window_duration_minutes=30,
                source="huggingface",
                model_id="org/model",
                target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                target_worker_uuids=["worker-a"],
                s3_profile_id=1,
            )
            session.add(schedule)
            await session.flush()
            task = await session.get(ModelPreheatTask, task_id)
            child = await session.get(ModelPreheatWorkerTask, child_id)
            task.schedule_id = schedule.id
            task.desired_state = task_desired
            task.execution_state = task_execution
            task.paused_from_state = (
                ModelPreheatExecutionStateEnum.PUBLISHING
                if task_execution == ModelPreheatExecutionStateEnum.PAUSED
                else None
            )
            if task_execution == ModelPreheatExecutionStateEnum.PAUSED:
                child.state = ModelPreheatWorkerTaskStateEnum.PAUSED
                child.lease_owner = None
                child.lease_token_hash = None
                child.lease_expires_at = None
            schedule_run = ModelPreheatScheduleRun(
                schedule_id=schedule.id,
                window_start_utc=datetime(2026, 8, 12, 0, 0, tzinfo=timezone.utc),
                window_end_utc=datetime(2026, 8, 12, 0, 30, tzinfo=timezone.utc),
                trigger=ModelPreheatScheduleRunTriggerEnum.SCHEDULED,
                state=run_state,
                operation_key=f"scheduled-{action.__name__}",
                slot=run_slot,
                task_id=task.id,
                started_at=datetime(2026, 8, 12, 0, 0, tzinfo=timezone.utc),
            )
            session.add_all([task, child, schedule_run])
            await session.commit()

        with pytest.raises(HTTPException) as caught:
            await _invoke(engine, action, task_id)

        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatTask, task_id)
            schedule_run = (await session.exec(select(ModelPreheatScheduleRun))).one()
            stored = (
                task.desired_state,
                task.execution_state,
                schedule_run.state,
                schedule_run.slot,
            )
        await engine.dispose()
        return caught.value, stored

    error, stored = asyncio.run(run())
    assert error.status_code == 409
    assert error.message == "model_preheat_schedule_managed_action"
    assert stored == (task_desired, task_execution, run_state, run_slot)


def test_late_result_is_rejected_when_parent_paused_or_retried(tmp_path):
    async def run():
        engine, task_id, child_id = await _seed(tmp_path)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            child = await session.get(ModelPreheatWorkerTask, child_id)
            child_attempt = child.attempt
            parent.desired_state = ModelPreheatDesiredStateEnum.PAUSED
            parent.execution_state = ModelPreheatExecutionStateEnum.PAUSED
            parent.paused_from_state = ModelPreheatExecutionStateEnum.PUBLISHING
            session.add(parent)
            await session.commit()
            lease = ModelPreheatWorkerTaskLease(
                worker_uuid="worker-a",
                worker_id=10,
                attempt=child_attempt,
                lease_token="token",
            )
            identity = ModelPreheatWorkerPrincipal(
                worker_id=10,
                worker_uuid="worker-a",
                credential_id=1,
                token_version=1,
            )
            try:
                await _validate_active_lease(session, child_id, lease, identity)
            except Exception as exc:
                paused_error = getattr(exc, "reason", None) or getattr(
                    exc, "message", str(exc)
                )
            parent.desired_state = ModelPreheatDesiredStateEnum.RUNNING
            parent.execution_state = ModelPreheatExecutionStateEnum.PENDING
            parent.attempt = 2
            session.add(parent)
            await session.commit()
            try:
                await _validate_active_lease(session, child_id, lease, identity)
            except Exception as exc:
                retried_error = getattr(exc, "reason", None) or getattr(
                    exc, "message", str(exc)
                )
        await engine.dispose()
        return paused_error, retried_error

    paused_error, retried_error = asyncio.run(run())
    assert "parent_not_running" in str(paused_error)
    assert "stale_parent_attempt" in str(retried_error)


def test_two_retry_requests_increment_parent_attempt_once(tmp_path):
    async def run():
        engine, task_id, _ = await _seed(tmp_path)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            parent.execution_state = ModelPreheatExecutionStateEnum.ERROR
            session.add(parent)
            await session.commit()

        async def retry():
            async with AsyncSession(engine) as session:
                return await retry_model_preheat(session=session, id=task_id)

        await asyncio.gather(retry(), retry())
        parent, _ = await _stored(engine, task_id)
        await engine.dispose()
        return parent

    parent = asyncio.run(run())
    assert parent.attempt == 2
