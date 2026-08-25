import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
)  # noqa: F401
from gpustack.schemas.model_preheat_schedules import (
    ModelPreheatSchedule,
    ModelPreheatScheduleRun,
    ModelPreheatScheduleRunStateEnum,
    ModelPreheatScheduleTriggerModeEnum,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatDesiredStateEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.models import ModelInstance, ModelInstanceStateEnum, SourceEnum
from gpustack.schemas.workers import (
    GPUDeviceInfo,
    Worker,
    WorkerStateEnum,
    WorkerStatus,
)
from gpustack.model_preheat_credentials import (
    generate_model_preheat_credential_key,
)
from gpustack.routes import model_preheats
from gpustack.routes.model_preheat_schedules import delete_model_preheat_schedule
from gpustack.server.bus import EventType
from gpustack.server import model_preheat_schedule_controller
from gpustack.server.model_preheat_schedule_controller import (
    ModelPreheatScheduleController,
    ScheduleConcurrencyLimit,
    create_scheduled_model_preheat_task,
)
from gpustack.server.model_preheat_s3_profile_lifecycle import (
    ModelPreheatS3ProfileNotActive,
)


UTC = timezone.utc


async def _database(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'schedule-controller.db'}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return engine


async def _add_worker(session, worker_uuid):
    worker = Worker(
        name=worker_uuid,
        hostname=worker_uuid,
        ip="127.0.0.1",
        port=10150,
        worker_uuid=worker_uuid,
        state=WorkerStateEnum.READY,
        model_storage_protocol_version=1,
    )
    session.add(worker)
    await session.flush()
    return worker


async def _seed_schedule(
    engine,
    *,
    max_concurrency=1,
    trigger_mode=ModelPreheatScheduleTriggerModeEnum.SCHEDULED,
):
    async with AsyncSession(engine) as session:
        await _add_worker(session, "worker-a")
        schedule = ModelPreheatSchedule(
            name="hourly",
            trigger_mode=trigger_mode,
            cron_expression=(
                "0 * * * *"
                if trigger_mode == ModelPreheatScheduleTriggerModeEnum.SCHEDULED
                else None
            ),
            timezone="UTC",
            window_duration_minutes=30,
            max_concurrency=max_concurrency,
            source="huggingface",
            model_id="org/model",
            revision="a" * 40,
            include_patterns=[],
            exclude_patterns=[],
            target_scope=ModelPreheatTargetScopeEnum.SAME_GPU_MODEL,
            target_worker_uuids=[],
            seed_worker_uuid="worker-a",
            s3_profile_id=1,
            s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
            next_window_start_utc=(
                datetime(2026, 8, 12, 0, 0, tzinfo=UTC)
                if trigger_mode == ModelPreheatScheduleTriggerModeEnum.SCHEDULED
                else None
            ),
            created_by_user_id=1,
        )
        session.add(schedule)
        await session.commit()
        await session.refresh(schedule)
        return schedule.id


def test_manual_schedule_is_not_ticked_but_can_run_now(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(
            engine, trigger_mode=ModelPreheatScheduleTriggerModeEnum.MANUAL
        )
        creator = RecordingTaskCreator()
        controller = ModelPreheatScheduleController(engine, task_creator=creator)

        await controller.tick(datetime(2026, 8, 12, 1, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            assert (await session.exec(select(ModelPreheatScheduleRun))).all() == []
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            manual_run = await controller.run_now(
                session,
                schedule,
                created_by_user_id=1,
                idempotency_key="manual-mode-run",
                now=datetime(2026, 8, 12, 1, 0, tzinfo=UTC),
            )
            assert manual_run.state == ModelPreheatScheduleRunStateEnum.RUNNING
        assert len(creator.calls) == 1
        await engine.dispose()

    asyncio.run(run())


class RecordingTaskCreator:
    def __init__(self):
        self.calls = []

    async def __call__(self, session, schedule, created_by_user_id):
        workers = (
            await session.exec(
                select(Worker)
                .where(Worker.state == WorkerStateEnum.READY)
                .order_by(Worker.worker_uuid)
            )
        ).all()
        worker_uuids = [worker.worker_uuid for worker in workers]
        self.calls.append(worker_uuids)
        seed = workers[0]
        task = ModelPreheatTask(
            source=schedule.source,
            model_id=schedule.model_id,
            requested_revision=schedule.revision,
            resolved_revision=schedule.revision,
            include_patterns=schedule.include_patterns,
            exclude_patterns=schedule.exclude_patterns,
            selection_digest="b" * 64,
            request_identity={
                "source": schedule.source,
                "model_id": schedule.model_id,
                "requested_revision": schedule.revision,
                "include_patterns": schedule.include_patterns,
                "exclude_patterns": schedule.exclude_patterns,
            },
            request_digest=f"{len(self.calls):064d}",
            seed_worker_uuid=seed.worker_uuid,
            seed_worker_id=seed.id,
            target_scope=schedule.target_scope,
            target_worker_uuids=worker_uuids,
            target_worker_snapshot=[
                {
                    "worker_uuid": worker.worker_uuid,
                    "worker_id": worker.id,
                    "worker_name": worker.name,
                }
                for worker in workers
            ],
            s3_profile_id=schedule.s3_profile_id,
            s3_profile_config_version=1,
            s3_profile_snapshot_encrypted={"ciphertext": "encrypted"},
            encryption_key_version="v1",
            s3_backfill_policy=schedule.s3_backfill_policy,
            schedule_id=schedule.id,
            created_by_user_id=created_by_user_id,
        )
        session.add(task)
        await session.flush()
        session.add(
            ModelPreheatWorkerTask(
                task_id=task.id,
                parent_attempt=task.attempt,
                worker_uuid=seed.worker_uuid,
                worker_id=seed.id,
                role=ModelPreheatWorkerTaskRoleEnum.SEED,
                state=ModelPreheatWorkerTaskStateEnum.PENDING,
            )
        )
        await session.flush()
        return task


async def _seed_pause_ack_pending_task(engine, schedule_id, creator):
    async with AsyncSession(engine) as session:
        schedule = await session.get(ModelPreheatSchedule, schedule_id)
        task = await creator(session, schedule, schedule.created_by_user_id)
        task.desired_state = ModelPreheatDesiredStateEnum.PAUSED
        task.paused_from_state = task.execution_state
        session.add(task)
        task_id = task.id
        await session.commit()
        return task_id


def _defer_pause_gate_until_after_creator(controller):
    original = controller._pause_ack_pending_task
    calls = 0

    async def gate(session, schedule_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return await original(session, schedule_id)

    controller._pause_ack_pending_task = gate


def _record_model_preheat_update_tables(engine):
    tables = []

    def before_cursor_execute(
        connection, cursor, statement, parameters, context, executemany
    ):
        del connection, cursor, parameters, context, executemany
        normalized = " ".join(statement.lower().split())
        if normalized.startswith("update model_preheat_"):
            tables.append(normalized.split()[1].strip('"`'))

    event.listen(engine.sync_engine, "before_cursor_execute", before_cursor_execute)
    return tables, before_cursor_execute


class TwoPartyBarrier:
    def __init__(self):
        self.arrived = 0
        self.both_arrived = asyncio.Event()

    async def __call__(self, schedule_id, window_start_utc):
        self.arrived += 1
        if self.arrived == 2:
            self.both_arrived.set()
        await asyncio.wait_for(self.both_arrived.wait(), timeout=5)


class BarrierTaskCreator(RecordingTaskCreator):
    def __init__(self):
        super().__init__()
        self.arrived = 0
        self.both_arrived = asyncio.Event()

    async def __call__(self, session, schedule, created_by_user_id):
        self.arrived += 1
        if self.arrived == 2:
            self.both_arrived.set()
        await asyncio.wait_for(self.both_arrived.wait(), timeout=5)
        return await super().__call__(session, schedule, created_by_user_id)


def test_duplicate_tick_and_two_servers_create_one_window_task(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        creator = RecordingTaskCreator()
        barrier = TwoPartyBarrier()
        first = ModelPreheatScheduleController(
            engine, task_creator=creator, claim_hook=barrier
        )
        second = ModelPreheatScheduleController(
            engine, task_creator=creator, claim_hook=barrier
        )
        now = datetime(2026, 8, 12, 0, 0, tzinfo=UTC)
        await asyncio.gather(first.tick(now), second.tick(now))
        await first.tick(now)
        async with AsyncSession(engine) as session:
            runs = (await session.exec(select(ModelPreheatScheduleRun))).all()
            tasks = (await session.exec(select(ModelPreheatTask))).all()
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
        await engine.dispose()
        return barrier.arrived, creator.calls, runs, tasks, schedule

    arrived, calls, runs, tasks, schedule = asyncio.run(run())
    assert arrived == 2
    assert len(calls) == 1
    assert len(runs) == 1
    assert len(tasks) == 1
    assert runs[0].window_start_utc == datetime(2026, 8, 12, 0, 0, tzinfo=UTC)
    assert schedule.next_window_start_utc == datetime(2026, 8, 12, 1, 0, tzinfo=UTC)


def test_window_end_pauses_and_next_window_resumes_same_task(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        creator = RecordingTaskCreator()
        controller = ModelPreheatScheduleController(engine, task_creator=creator)
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        await controller.tick(datetime(2026, 8, 12, 0, 31, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            paused_task = (await session.exec(select(ModelPreheatTask))).one()
            paused_child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            paused_run = (await session.exec(select(ModelPreheatScheduleRun))).one()
            paused = (
                paused_task.desired_state,
                paused_task.execution_state,
                paused_child.state,
                paused_run.state,
                paused_run.slot,
            )
        await controller.tick(datetime(2026, 8, 12, 1, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            runs = (
                await session.exec(
                    select(ModelPreheatScheduleRun).order_by(
                        ModelPreheatScheduleRun.window_start_utc
                    )
                )
            ).all()
            task = (await session.exec(select(ModelPreheatTask))).one()
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
        await engine.dispose()
        return creator.calls, paused, runs, task, child

    calls, paused, runs, task, child = asyncio.run(run())
    assert calls == [["worker-a"]]
    assert paused == (
        ModelPreheatDesiredStateEnum.PAUSED,
        ModelPreheatExecutionStateEnum.PAUSED,
        ModelPreheatWorkerTaskStateEnum.PAUSED,
        ModelPreheatScheduleRunStateEnum.PAUSED,
        None,
    )
    assert len(runs) == 2
    assert runs[0].task_id == runs[1].task_id == task.id
    assert task.desired_state == ModelPreheatDesiredStateEnum.RUNNING
    assert child.state == ModelPreheatWorkerTaskStateEnum.PENDING


def test_terminal_task_releases_slot_and_new_worker_joins_next_fresh_run(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed_schedule(engine)
        creator = RecordingTaskCreator()
        controller = ModelPreheatScheduleController(engine, task_creator=creator)
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            first_task = (await session.exec(select(ModelPreheatTask))).one()
            first_task.execution_state = ModelPreheatExecutionStateEnum.READY
            first_task.desired_state = ModelPreheatDesiredStateEnum.RUNNING
            session.add(first_task)
            await _add_worker(session, "worker-b")
            first_task_id = first_task.id
            await session.commit()
        await controller.tick(datetime(2026, 8, 12, 0, 10, tzinfo=UTC))
        await controller.tick(datetime(2026, 8, 12, 1, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            tasks = (
                await session.exec(
                    select(ModelPreheatTask).order_by(ModelPreheatTask.id)
                )
            ).all()
            runs = (
                await session.exec(
                    select(ModelPreheatScheduleRun).order_by(ModelPreheatScheduleRun.id)
                )
            ).all()
        await engine.dispose()
        return first_task_id, creator.calls, tasks, runs

    first_task_id, calls, tasks, runs = asyncio.run(run())
    assert calls == [["worker-a"], ["worker-a", "worker-b"]]
    assert len(tasks) == 2
    assert runs[0].task_id == first_task_id
    assert runs[0].state == ModelPreheatScheduleRunStateEnum.READY
    assert runs[0].slot is None
    assert runs[1].task_id == tasks[1].id


def test_manual_runs_obey_database_concurrency_slots(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine, max_concurrency=2)
        creator = RecordingTaskCreator()
        controller = ModelPreheatScheduleController(engine, task_creator=creator)
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            first = await controller.run_now(session, schedule, 1, "manual-1")
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            second = await controller.run_now(session, schedule, 1, "manual-2")
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            with pytest.raises(ScheduleConcurrencyLimit):
                await controller.run_now(session, schedule, 1, "manual-3")
        await engine.dispose()
        return first.slot, second.slot

    assert asyncio.run(run()) == (0, 1)


def test_manual_runs_with_different_keys_can_share_database_timestamp(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine, max_concurrency=2)
        creator = RecordingTaskCreator()
        controller = ModelPreheatScheduleController(engine, task_creator=creator)
        database_timestamp = datetime(2026, 8, 12, 0, 0, tzinfo=UTC)
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            first = await controller.run_now(
                session,
                schedule,
                1,
                "same-second-1",
                now=database_timestamp,
            )
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            second = await controller.run_now(
                session,
                schedule,
                1,
                "same-second-2",
                now=database_timestamp,
            )
        async with AsyncSession(engine) as session:
            runs = (
                await session.exec(
                    select(ModelPreheatScheduleRun).order_by(ModelPreheatScheduleRun.id)
                )
            ).all()
        await engine.dispose()
        return first, second, runs

    first, second, runs = asyncio.run(run())
    assert first.id != second.id
    assert first.window_start_utc == second.window_start_utc
    assert [run.slot for run in runs] == [0, 1]


def test_concurrent_manual_same_key_returns_one_run_and_task(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine, max_concurrency=2)
        creator = BarrierTaskCreator()
        first = ModelPreheatScheduleController(engine, task_creator=creator)
        second = ModelPreheatScheduleController(engine, task_creator=creator)

        async def invoke(controller):
            async with AsyncSession(engine) as session:
                schedule = await session.get(ModelPreheatSchedule, schedule_id)
                return await controller.run_now(session, schedule, 1, "same-key")

        returned = await asyncio.gather(invoke(first), invoke(second))
        async with AsyncSession(engine) as session:
            runs = (await session.exec(select(ModelPreheatScheduleRun))).all()
            tasks = (await session.exec(select(ModelPreheatTask))).all()
        await engine.dispose()
        return returned, runs, tasks

    returned, runs, tasks = asyncio.run(run())
    assert returned[0].id == returned[1].id
    assert returned[0].task_id == returned[1].task_id
    assert len(runs) == 1
    assert len(tasks) == 1


def test_deduplicated_manual_task_is_not_paused_by_schedule_window(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed_schedule(engine)
        creator = RecordingTaskCreator()
        async with AsyncSession(engine) as session:
            schedule = (await session.exec(select(ModelPreheatSchedule))).one()
            manual_task = await creator(session, schedule, 1)
            manual_task.schedule_id = None
            session.add(manual_task)
            manual_task_id = manual_task.id
            await session.commit()

        async def reuse_manual_task(session, schedule, created_by_user_id):
            del schedule, created_by_user_id
            return await session.get(ModelPreheatTask, manual_task_id)

        controller = ModelPreheatScheduleController(
            engine, task_creator=reuse_manual_task
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        await controller.tick(datetime(2026, 8, 12, 0, 31, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            run = (await session.exec(select(ModelPreheatScheduleRun))).one()
            task = await session.get(ModelPreheatTask, manual_task_id)
            result = run.state, run.error_code, run.slot, task.desired_state
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (
        ModelPreheatScheduleRunStateEnum.SKIPPED,
        "active_operation_deduplicated",
        None,
        ModelPreheatDesiredStateEnum.RUNNING,
    )


def test_default_task_creator_builds_real_scheduled_task_without_plain_credentials(
    tmp_path, monkeypatch
):
    async def skip_connectivity_check(session, profile, config, workers, **kwargs):
        del session, profile, config, workers, kwargs

    monkeypatch.setattr(
        model_preheats,
        "_ensure_profile_available_on_workers",
        skip_connectivity_check,
    )
    monkeypatch.setattr(
        model_preheat_schedule_controller,
        "resolve_model_preheat_revision",
        lambda source, model_id, revision, token=None: revision,
    )

    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        async with AsyncSession(engine) as session:
            session.add(
                ModelPreheatS3Profile(
                    id=1,
                    name="profile",
                    endpoint="https://s3.example.com",
                    bucket="models",
                    access_key_encrypted={"ciphertext": "access-secret"},
                    secret_key_encrypted={"ciphertext": "secret-secret"},
                    encryption_key_version="v1",
                )
            )
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            schedule.target_scope = ModelPreheatTargetScopeEnum.SELECTED_WORKERS
            schedule.target_worker_uuids = ["worker-a"]
            session.add(schedule)
            await session.commit()
        config = SimpleNamespace(
            model_preheat_credential_key=generate_model_preheat_credential_key(),
            model_preheat_credential_key_version="v1",
            model_preheat_credential_old_keys=None,
            huggingface_token=None,
        )
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            task = await create_scheduled_model_preheat_task(
                session, schedule, 1, config
            )
            await session.commit()
            await session.refresh(task)
            result = (
                task.schedule_id,
                task.target_worker_uuids,
                task.s3_profile_snapshot_encrypted,
            )
        await engine.dispose()
        return result

    schedule_id, worker_uuids, encrypted_snapshot = asyncio.run(run())
    assert schedule_id is not None
    assert worker_uuids == ["worker-a"]
    assert "access-secret" not in str(encrypted_snapshot)
    assert "secret-secret" not in str(encrypted_snapshot)


def test_default_task_creator_rejects_profile_maintained_before_final_lock(
    tmp_path, monkeypatch
):
    async def skip_connectivity_check(session, profile, config, workers, **kwargs):
        del session, profile, config, workers, kwargs

    async def reject_stale_active_profile(*args, **kwargs):
        del args, kwargs
        raise ModelPreheatS3ProfileNotActive

    monkeypatch.setattr(
        model_preheats,
        "_ensure_profile_available_on_workers",
        skip_connectivity_check,
    )
    monkeypatch.setattr(
        model_preheat_schedule_controller,
        "resolve_model_preheat_revision",
        lambda source, model_id, revision, token=None: revision,
    )
    monkeypatch.setattr(
        model_preheat_schedule_controller,
        "lock_active_profile_for_new_work",
        reject_stale_active_profile,
    )

    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        async with AsyncSession(engine) as session:
            session.add(
                ModelPreheatS3Profile(
                    id=1,
                    name="profile",
                    endpoint="https://s3.example.com",
                    bucket="models",
                    access_key_encrypted={"ciphertext": "access-secret"},
                    secret_key_encrypted={"ciphertext": "secret-secret"},
                    encryption_key_version="v1",
                )
            )
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            schedule.target_scope = ModelPreheatTargetScopeEnum.SELECTED_WORKERS
            schedule.target_worker_uuids = ["worker-a"]
            session.add(schedule)
            await session.commit()
        config = SimpleNamespace(
            model_preheat_credential_key=generate_model_preheat_credential_key(),
            model_preheat_credential_key_version="v1",
            model_preheat_credential_old_keys=None,
            huggingface_token=None,
        )
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            with pytest.raises(
                RuntimeError, match="model_preheat_s3_profile_in_maintenance"
            ):
                await create_scheduled_model_preheat_task(session, schedule, 1, config)
            task_count = len((await session.exec(select(ModelPreheatTask))).all())
        await engine.dispose()
        return task_count

    assert asyncio.run(run()) == 0


def test_default_task_creator_rejects_busy_target_worker(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        async with AsyncSession(engine) as session:
            worker = (await session.exec(select(Worker))).one()
            session.add(
                ModelInstance(
                    source=SourceEnum.HUGGING_FACE,
                    name="busy-instance",
                    model_name="busy-model",
                    model_id=1,
                    worker_id=worker.id,
                    state=ModelInstanceStateEnum.RUNNING,
                )
            )
            await session.commit()
        config = SimpleNamespace(model_preheat_enabled=True)
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            with pytest.raises(RuntimeError, match="seed_worker_not_idle"):
                await create_scheduled_model_preheat_task(session, schedule, 1, config)
        await engine.dispose()

    asyncio.run(run())


def test_same_gpu_model_creator_does_not_reinclude_busy_worker(tmp_path, monkeypatch):
    async def skip_connectivity_check(session, profile, config, workers, **kwargs):
        del session, profile, config, workers, kwargs

    monkeypatch.setattr(
        model_preheats,
        "_ensure_profile_available_on_workers",
        skip_connectivity_check,
    )
    monkeypatch.setattr(
        model_preheat_schedule_controller,
        "resolve_model_preheat_revision",
        lambda source, model_id, revision, token=None: revision,
    )

    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        async with AsyncSession(engine) as session:
            seed = (await session.exec(select(Worker))).one()
            seed.status = WorkerStatus(gpu_devices=[GPUDeviceInfo(name="NVIDIA A100")])
            busy = await _add_worker(session, "worker-b")
            busy.status = WorkerStatus(gpu_devices=[GPUDeviceInfo(name="NVIDIA A100")])
            session.add(seed)
            session.add(busy)
            session.add(
                ModelInstance(
                    source=SourceEnum.HUGGING_FACE,
                    name="busy-instance",
                    model_name="busy-model",
                    model_id=1,
                    worker_id=busy.id,
                    state=ModelInstanceStateEnum.RUNNING,
                )
            )
            session.add(
                ModelPreheatS3Profile(
                    id=1,
                    name="profile",
                    endpoint="https://s3.example.com",
                    bucket="models",
                    access_key_encrypted={"ciphertext": "access-secret"},
                    secret_key_encrypted={"ciphertext": "secret-secret"},
                    encryption_key_version="v1",
                )
            )
            await session.commit()
        config = SimpleNamespace(
            model_preheat_credential_key=generate_model_preheat_credential_key(),
            model_preheat_credential_key_version="v1",
            model_preheat_credential_old_keys=None,
            huggingface_token=None,
        )
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            task = await create_scheduled_model_preheat_task(
                session, schedule, 1, config
            )
            result = task.target_worker_uuids
        await engine.dispose()
        return result

    assert asyncio.run(run()) == ["worker-a"]


def test_disabled_feature_flag_does_not_block_schedule_runs(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed_schedule(engine)
        creator = RecordingTaskCreator()
        controller = ModelPreheatScheduleController(
            engine,
            task_creator=creator,
            config=SimpleNamespace(model_preheat_enabled=False),
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            runs = (await session.exec(select(ModelPreheatScheduleRun))).all()
        await engine.dispose()
        return creator.calls, runs

    calls, runs = asyncio.run(run())
    assert len(calls) == 1
    assert len(runs) == 1


def test_disable_committed_during_claim_prevents_window_task(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        creator = RecordingTaskCreator()

        async def disable_after_read(claimed_schedule_id, window_start):
            del window_start
            async with AsyncSession(engine) as other:
                schedule = await other.get(ModelPreheatSchedule, claimed_schedule_id)
                schedule.enabled = False
                schedule.next_window_start_utc = None
                other.add(schedule)
                await other.commit()

        controller = ModelPreheatScheduleController(
            engine,
            task_creator=creator,
            claim_hook=disable_after_read,
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            runs = (await session.exec(select(ModelPreheatScheduleRun))).all()
            tasks = (await session.exec(select(ModelPreheatTask))).all()
        await engine.dispose()
        return schedule, runs, tasks

    schedule, runs, tasks = asyncio.run(run())
    assert schedule.enabled is False
    assert schedule.next_window_start_utc is None
    assert runs == []
    assert tasks == []


def test_delete_committed_during_claim_prevents_orphan_task(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        creator = RecordingTaskCreator()

        async def delete_after_read(claimed_schedule_id, window_start):
            del window_start
            async with AsyncSession(engine) as other:
                await delete_model_preheat_schedule(other, claimed_schedule_id)

        controller = ModelPreheatScheduleController(
            engine,
            task_creator=creator,
            claim_hook=delete_after_read,
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            runs = (await session.exec(select(ModelPreheatScheduleRun))).all()
            tasks = (await session.exec(select(ModelPreheatTask))).all()
        await engine.dispose()
        return schedule, runs, tasks

    schedule, runs, tasks = asyncio.run(run())
    assert schedule is None
    assert runs == []
    assert tasks == []


def test_delete_terminal_paused_history_sets_task_schedule_null(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        controller = ModelPreheatScheduleController(
            engine, task_creator=RecordingTaskCreator()
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        await controller.tick(datetime(2026, 8, 12, 0, 31, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            task = (await session.exec(select(ModelPreheatTask))).one()
            task.execution_state = ModelPreheatExecutionStateEnum.READY
            task.progress = 100
            session.add(task)
            await session.commit()
        async with AsyncSession(engine) as session:
            await session.exec(text("PRAGMA foreign_keys=ON"))
            await delete_model_preheat_schedule(session, schedule_id)
        async with AsyncSession(engine) as session:
            task = (await session.exec(select(ModelPreheatTask))).one()
            runs = (await session.exec(select(ModelPreheatScheduleRun))).all()
            result = task.schedule_id, runs
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (None, [])


def test_pause_task_cas_does_not_overwrite_terminal_parent(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed_schedule(engine)
        creator = RecordingTaskCreator()
        controller = ModelPreheatScheduleController(engine, task_creator=creator)
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            task = (await session.exec(select(ModelPreheatTask))).one()
            run = (await session.exec(select(ModelPreheatScheduleRun))).one()
            task.execution_state = ModelPreheatExecutionStateEnum.READY
            task.progress = 100
            session.add(task)
            await session.commit()
            await session.refresh(task)
            await session.refresh(run)
            paused = await controller._pause_task(
                session,
                task,
                run,
                datetime(2026, 8, 12, 0, 31, tzinfo=UTC),
            )
            await session.commit()
            await session.refresh(task)
            result = paused, task.desired_state, task.execution_state
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (
        False,
        ModelPreheatDesiredStateEnum.RUNNING,
        ModelPreheatExecutionStateEnum.READY,
    )


def test_pause_request_keeps_running_child_lease_until_worker_confirmation(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed_schedule(engine)
        creator = RecordingTaskCreator()
        controller = ModelPreheatScheduleController(engine, task_creator=creator)
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            task = (await session.exec(select(ModelPreheatTask))).one()
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            child.state = ModelPreheatWorkerTaskStateEnum.RUNNING
            child.lease_owner = child.worker_uuid
            child.lease_token_hash = "lease-hash"
            child.lease_expires_at = datetime(2026, 8, 12, 1, 0, tzinfo=UTC)
            session.add(child)
            await session.commit()
            await session.refresh(task)
            run = (await session.exec(select(ModelPreheatScheduleRun))).one()
            paused = await controller._pause_task(
                session,
                task,
                run,
                datetime(2026, 8, 12, 0, 31, tzinfo=UTC),
            )
            await session.commit()
        async with AsyncSession(engine) as session:
            task = (await session.exec(select(ModelPreheatTask))).one()
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            result = (
                paused,
                task.desired_state,
                task.execution_state,
                child.state,
                child.state_message,
                child.lease_owner,
                child.lease_token_hash,
            )
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (
        True,
        ModelPreheatDesiredStateEnum.PAUSED,
        ModelPreheatExecutionStateEnum.PENDING,
        ModelPreheatWorkerTaskStateEnum.RUNNING,
        "pause_requested",
        "worker-a",
        "lease-hash",
    )


def test_stale_window_cannot_pause_task_owned_by_new_run(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed_schedule(engine, max_concurrency=2)
        controller = ModelPreheatScheduleController(
            engine, task_creator=RecordingTaskCreator()
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))

        async with AsyncSession(engine) as stale_session:
            task = (await stale_session.exec(select(ModelPreheatTask))).one()
            stale_run = (
                await stale_session.exec(select(ModelPreheatScheduleRun))
            ).one()

            async with AsyncSession(engine) as current_session:
                old_run = await current_session.get(
                    ModelPreheatScheduleRun, stale_run.id
                )
                old_run.state = ModelPreheatScheduleRunStateEnum.SKIPPED
                old_run.slot = None
                current_session.add(old_run)
                new_run = ModelPreheatScheduleRun(
                    schedule_id=old_run.schedule_id,
                    window_start_utc=datetime(2026, 8, 12, 1, 0, tzinfo=UTC),
                    window_end_utc=datetime(2026, 8, 12, 1, 30, tzinfo=UTC),
                    trigger=old_run.trigger,
                    state=ModelPreheatScheduleRunStateEnum.RUNNING,
                    operation_key="new-owner",
                    slot=1,
                    task_id=task.id,
                    created_by_user_id=1,
                )
                current_session.add(new_run)
                await current_session.commit()

            paused = await controller._pause_task(stale_session, task, stale_run)
            await stale_session.commit()

        async with AsyncSession(engine) as session:
            task = (await session.exec(select(ModelPreheatTask))).one()
            result = paused, task.desired_state, task.execution_state
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (
        False,
        ModelPreheatDesiredStateEnum.RUNNING,
        ModelPreheatExecutionStateEnum.PENDING,
    )


def test_window_pause_publishes_paused_child_event_after_commit(tmp_path, monkeypatch):
    published = []

    async def capture(cls, event_type, child):
        published.append((event_type, child.id, child.state, child.state_message))

    monkeypatch.setattr(ModelPreheatWorkerTask, "_publish_event", classmethod(capture))

    async def run():
        engine = await _database(tmp_path)
        await _seed_schedule(engine)
        controller = ModelPreheatScheduleController(
            engine, task_creator=RecordingTaskCreator()
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            child.state = ModelPreheatWorkerTaskStateEnum.RUNNING
            child.lease_owner = child.worker_uuid
            child.lease_token_hash = "lease-hash"
            child.lease_expires_at = datetime(2026, 8, 12, 1, 0, tzinfo=UTC)
            child_id = child.id
            session.add(child)
            await session.commit()
        await controller.tick(datetime(2026, 8, 12, 0, 31, tzinfo=UTC))
        await engine.dispose()
        return child_id

    child_id = asyncio.run(run())
    assert published == [
        (
            EventType.UPDATED,
            child_id,
            ModelPreheatWorkerTaskStateEnum.RUNNING,
            "pause_requested",
        )
    ]


def test_pause_reaper_updates_parent_before_expired_child(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed_schedule(engine)
        controller = ModelPreheatScheduleController(
            engine, task_creator=RecordingTaskCreator()
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            parent = (await session.exec(select(ModelPreheatTask))).one()
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            parent.desired_state = ModelPreheatDesiredStateEnum.PAUSED
            parent.paused_from_state = parent.execution_state
            child.state = ModelPreheatWorkerTaskStateEnum.RUNNING
            child.state_message = "pause_requested"
            child.lease_owner = child.worker_uuid
            child.lease_token_hash = "expired-lease"
            child.lease_expires_at = datetime(2020, 1, 1, tzinfo=UTC)
            session.add(parent)
            session.add(child)
            await session.commit()

        update_tables, listener = _record_model_preheat_update_tables(engine)
        try:
            async with AsyncSession(engine) as session:
                parent = (await session.exec(select(ModelPreheatTask))).one()
                await controller._reap_expired_pause_requests(
                    session,
                    parent,
                    datetime(2026, 8, 12, 0, 31, tzinfo=UTC),
                )
                await session.commit()
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", listener)

        async with AsyncSession(engine) as session:
            parent = (await session.exec(select(ModelPreheatTask))).one()
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            result = parent.execution_state, child.state, update_tables
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (
        ModelPreheatExecutionStateEnum.PAUSED,
        ModelPreheatWorkerTaskStateEnum.PAUSED,
        [
            "model_preheat_tasks",
            "model_preheat_worker_tasks",
            "model_preheat_tasks",
        ],
    )


def test_expired_pause_request_lease_is_reaped_before_resume(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        controller = ModelPreheatScheduleController(
            engine, task_creator=RecordingTaskCreator()
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            child.state = ModelPreheatWorkerTaskStateEnum.RUNNING
            child.lease_owner = child.worker_uuid
            child.lease_token_hash = "expired-lease"
            child.lease_expires_at = datetime(2020, 1, 1, tzinfo=UTC)
            session.add(child)
            await session.commit()
        await controller.tick(datetime(2026, 8, 12, 0, 31, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            resumed = await controller._resume_paused_task(
                session,
                schedule_id,
                datetime(2026, 8, 12, 0, 31, tzinfo=UTC),
            )
            await session.commit()
        async with AsyncSession(engine) as session:
            parent = (await session.exec(select(ModelPreheatTask))).one()
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            result = (
                resumed is not None,
                parent.desired_state,
                child.state,
                child.lease_token_hash,
            )
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (
        True,
        ModelPreheatDesiredStateEnum.RUNNING,
        ModelPreheatWorkerTaskStateEnum.PENDING,
        None,
    )


def test_resume_paused_task_uses_window_time_for_pause_ack(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        controller = ModelPreheatScheduleController(
            engine, task_creator=RecordingTaskCreator()
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            parent = (await session.exec(select(ModelPreheatTask))).one()
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            parent.desired_state = ModelPreheatDesiredStateEnum.PAUSED
            parent.paused_from_state = parent.execution_state
            child.state = ModelPreheatWorkerTaskStateEnum.RUNNING
            child.state_message = "pause_requested"
            child.lease_owner = child.worker_uuid
            child.lease_token_hash = "still-valid-at-window-time"
            child.lease_expires_at = datetime(2026, 8, 15, tzinfo=UTC)
            session.add(parent)
            session.add(child)
            await session.commit()
        async with AsyncSession(engine) as session:
            resumed = await controller._resume_paused_task(
                session,
                schedule_id,
                datetime(2026, 8, 12, 0, 31, tzinfo=UTC),
            )
        await engine.dispose()
        return resumed

    assert asyncio.run(run()) is None


def test_expired_pause_child_is_reaped_when_legacy_marker_was_lost(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed_schedule(engine)
        controller = ModelPreheatScheduleController(
            engine, task_creator=RecordingTaskCreator()
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            child.state = ModelPreheatWorkerTaskStateEnum.RUNNING
            child.lease_owner = child.worker_uuid
            child.lease_token_hash = "active-lease"
            child.lease_expires_at = datetime(2026, 8, 12, 1, 0, tzinfo=UTC)
            session.add(child)
            await session.commit()
        await controller.tick(datetime(2026, 8, 12, 0, 31, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            child.state_message = "downloading"
            child.lease_expires_at = datetime(2026, 8, 12, 0, 31, tzinfo=UTC)
            session.add(child)
            await session.commit()
        await controller.tick(datetime(2026, 8, 12, 0, 32, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            parent = (await session.exec(select(ModelPreheatTask))).one()
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            result = (
                parent.execution_state,
                child.state,
                child.state_message,
                child.lease_token_hash,
            )
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (
        ModelPreheatExecutionStateEnum.PAUSED,
        ModelPreheatWorkerTaskStateEnum.PAUSED,
        "paused",
        None,
    )


def test_stale_pause_reaper_does_not_overwrite_concurrent_parent_resume(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed_schedule(engine)
        controller = ModelPreheatScheduleController(
            engine, task_creator=RecordingTaskCreator()
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            parent = (await session.exec(select(ModelPreheatTask))).one()
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            parent.desired_state = ModelPreheatDesiredStateEnum.PAUSED
            child.state = ModelPreheatWorkerTaskStateEnum.RUNNING
            child.lease_owner = child.worker_uuid
            child.lease_token_hash = "expired-lease"
            child.lease_expires_at = datetime(2026, 8, 12, 0, 30, tzinfo=UTC)
            child.state_message = "pause_requested"
            session.add(parent)
            session.add(child)
            await session.commit()

        async with AsyncSession(engine) as stale_session:
            stale_parent = (await stale_session.exec(select(ModelPreheatTask))).one()
            async with AsyncSession(engine) as resume_session:
                current_parent = await resume_session.get(
                    ModelPreheatTask, stale_parent.id
                )
                current_parent.desired_state = ModelPreheatDesiredStateEnum.RUNNING
                resume_session.add(current_parent)
                await resume_session.commit()
            update_tables, listener = _record_model_preheat_update_tables(engine)
            try:
                await controller._reap_expired_pause_requests(
                    stale_session,
                    stale_parent,
                    datetime(2026, 8, 12, 0, 31, tzinfo=UTC),
                )
                await stale_session.commit()
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", listener)

        async with AsyncSession(engine) as session:
            parent = (await session.exec(select(ModelPreheatTask))).one()
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            result = (
                parent.desired_state,
                child.state,
                child.state_message,
                update_tables,
            )
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (
        ModelPreheatDesiredStateEnum.RUNNING,
        ModelPreheatWorkerTaskStateEnum.RUNNING,
        "pause_requested",
        ["model_preheat_tasks"],
    )


def test_adjacent_window_does_not_run_before_pause_ack(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed_schedule(engine)
        creator = RecordingTaskCreator()
        controller = ModelPreheatScheduleController(engine, task_creator=creator)
        async with AsyncSession(engine) as session:
            schedule = (await session.exec(select(ModelPreheatSchedule))).one()
            schedule.window_duration_minutes = 60
            session.add(schedule)
            await session.commit()
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            child.state = ModelPreheatWorkerTaskStateEnum.RUNNING
            child.lease_owner = child.worker_uuid
            child.lease_token_hash = "active-lease"
            child.lease_expires_at = datetime(2026, 8, 12, 2, 0, tzinfo=UTC)
            session.add(child)
            await session.commit()
        await controller.tick(datetime(2026, 8, 12, 1, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            runs = (
                await session.exec(
                    select(ModelPreheatScheduleRun).order_by(
                        ModelPreheatScheduleRun.window_start_utc
                    )
                )
            ).all()
            parent = (await session.exec(select(ModelPreheatTask))).one()
            result = (
                [(run.state, run.error_code) for run in runs],
                parent.desired_state,
                parent.execution_state,
                len(creator.calls),
            )
        await engine.dispose()
        return result

    states, desired, execution, creator_calls = asyncio.run(run())
    assert states == [
        (ModelPreheatScheduleRunStateEnum.PAUSED, None),
        (ModelPreheatScheduleRunStateEnum.SKIPPED, "pause_ack_pending"),
    ]
    assert desired == ModelPreheatDesiredStateEnum.PAUSED
    assert execution != ModelPreheatExecutionStateEnum.PAUSED
    assert creator_calls == 1


def test_run_now_is_skipped_while_schedule_pause_ack_is_pending(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        controller = ModelPreheatScheduleController(
            engine, task_creator=RecordingTaskCreator()
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            child.state = ModelPreheatWorkerTaskStateEnum.RUNNING
            child.lease_owner = child.worker_uuid
            child.lease_token_hash = "active-lease"
            child.lease_expires_at = datetime(2026, 8, 12, 2, 0, tzinfo=UTC)
            session.add(child)
            await session.commit()
        await controller.tick(datetime(2026, 8, 12, 0, 31, tzinfo=UTC))

        async def deduplicate_to_pausing_task(session, schedule, created_by_user_id):
            del schedule, created_by_user_id
            return (await session.exec(select(ModelPreheatTask))).one()

        controller._task_creator = deduplicate_to_pausing_task
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            manual = await controller.run_now(
                session,
                schedule,
                created_by_user_id=1,
                idempotency_key="pause-pending",
                now=datetime(2026, 8, 12, 0, 32, tzinfo=UTC),
            )
        async with AsyncSession(engine) as session:
            parent = (await session.exec(select(ModelPreheatTask))).one()
            persisted = await session.get(ModelPreheatScheduleRun, manual.id)
            result = (
                persisted.state,
                persisted.error_code,
                persisted.slot,
                persisted.task_id == parent.id,
                parent.desired_state,
                parent.execution_state,
            )
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (
        ModelPreheatScheduleRunStateEnum.SKIPPED,
        "pause_ack_pending",
        None,
        True,
        ModelPreheatDesiredStateEnum.PAUSED,
        ModelPreheatExecutionStateEnum.PENDING,
    )


def test_run_now_rechecks_pause_ack_after_concurrent_window_close(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        controller = ModelPreheatScheduleController(
            engine, task_creator=RecordingTaskCreator()
        )
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            parent = (await session.exec(select(ModelPreheatTask))).one()
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            run = (await session.exec(select(ModelPreheatScheduleRun))).one()
            child.state = ModelPreheatWorkerTaskStateEnum.RUNNING
            child.lease_owner = child.worker_uuid
            child.lease_token_hash = "active-lease"
            child.lease_expires_at = datetime(2026, 8, 12, 2, 0, tzinfo=UTC)
            parent_id = parent.id
            run_id = run.id
            session.add(child)
            await session.commit()

        creator_entered = asyncio.Event()
        release_creator = asyncio.Event()

        async def delayed_deduplication(session, schedule, created_by_user_id):
            del schedule, created_by_user_id
            creator_entered.set()
            await release_creator.wait()
            return await session.get(
                ModelPreheatTask, parent_id, populate_existing=True
            )

        controller._task_creator = delayed_deduplication

        async def invoke_run_now():
            async with AsyncSession(engine) as session:
                schedule = await session.get(ModelPreheatSchedule, schedule_id)
                return await controller.run_now(
                    session,
                    schedule,
                    created_by_user_id=1,
                    idempotency_key="concurrent-pause",
                    now=datetime(2026, 8, 12, 0, 31, tzinfo=UTC),
                )

        manual_run = asyncio.create_task(invoke_run_now())
        await asyncio.wait_for(creator_entered.wait(), timeout=1)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, parent_id)
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            scheduled_run = await session.get(ModelPreheatScheduleRun, run_id)
            parent.desired_state = ModelPreheatDesiredStateEnum.PAUSED
            parent.paused_from_state = parent.execution_state
            child.state_message = "pause_requested"
            scheduled_run.state = ModelPreheatScheduleRunStateEnum.PAUSED
            scheduled_run.slot = None
            scheduled_run.finished_at = datetime(2026, 8, 12, 0, 31, tzinfo=UTC)
            session.add(parent)
            session.add(child)
            session.add(scheduled_run)
            await session.commit()
        release_creator.set()
        manual = await asyncio.wait_for(manual_run, timeout=1)
        async with AsyncSession(engine) as session:
            persisted = await session.get(ModelPreheatScheduleRun, manual.id)
            result = persisted.state, persisted.error_code, persisted.slot
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (
        ModelPreheatScheduleRunStateEnum.SKIPPED,
        "pause_ack_pending",
        None,
    )


def test_run_now_discards_flushed_task_when_post_create_pause_gate_closes(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        creator = RecordingTaskCreator()
        paused_task_id = await _seed_pause_ack_pending_task(
            engine, schedule_id, creator
        )
        controller = ModelPreheatScheduleController(engine, task_creator=creator)
        _defer_pause_gate_until_after_creator(controller)

        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            manual = await controller.run_now(
                session,
                schedule,
                created_by_user_id=1,
                idempotency_key="post-create-pause",
                now=datetime(2026, 8, 12, 0, 1, tzinfo=UTC),
            )

        async with AsyncSession(engine) as session:
            tasks = (await session.exec(select(ModelPreheatTask))).all()
            children = (await session.exec(select(ModelPreheatWorkerTask))).all()
            persisted_run = await session.get(ModelPreheatScheduleRun, manual.id)
            result = (
                persisted_run.state,
                persisted_run.error_code,
                persisted_run.task_id == paused_task_id,
                [task.id for task in tasks] == [paused_task_id],
                [child.task_id for child in children] == [paused_task_id],
            )
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (
        ModelPreheatScheduleRunStateEnum.SKIPPED,
        "pause_ack_pending",
        True,
        True,
        True,
    )


def test_scheduled_window_discards_flushed_task_when_post_create_pause_gate_closes(
    tmp_path,
):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        creator = RecordingTaskCreator()
        paused_task_id = await _seed_pause_ack_pending_task(
            engine, schedule_id, creator
        )
        controller = ModelPreheatScheduleController(engine, task_creator=creator)
        _defer_pause_gate_until_after_creator(controller)

        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))

        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            tasks = (await session.exec(select(ModelPreheatTask))).all()
            children = (await session.exec(select(ModelPreheatWorkerTask))).all()
            run = (await session.exec(select(ModelPreheatScheduleRun))).one()
            result = (
                run.state,
                run.error_code,
                run.task_id == paused_task_id,
                [task.id for task in tasks] == [paused_task_id],
                [child.task_id for child in children] == [paused_task_id],
                schedule.next_window_start_utc,
            )
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (
        ModelPreheatScheduleRunStateEnum.SKIPPED,
        "pause_ack_pending",
        True,
        True,
        True,
        datetime(2026, 8, 12, 1, 0, tzinfo=UTC),
    )


def test_resume_paused_task_cas_does_not_overwrite_concurrent_cancel(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        creator = RecordingTaskCreator()
        controller = ModelPreheatScheduleController(engine, task_creator=creator)
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        await controller.tick(datetime(2026, 8, 12, 0, 31, tzinfo=UTC))
        async with AsyncSession(engine) as stale:
            cached = (await stale.exec(select(ModelPreheatTask))).one()
            task_id = cached.id
            async with AsyncSession(engine) as current:
                task = await current.get(ModelPreheatTask, cached.id)
                task.desired_state = ModelPreheatDesiredStateEnum.CANCELED
                task.execution_state = ModelPreheatExecutionStateEnum.CANCELED
                current.add(task)
                await current.commit()
            resumed = await controller._resume_paused_task(stale, schedule_id)
            await stale.commit()
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatTask, task_id)
            run = (await session.exec(select(ModelPreheatScheduleRun))).one()
            result = resumed, task.desired_state, task.execution_state, run.state
        await engine.dispose()
        return result

    assert asyncio.run(run()) == (
        None,
        ModelPreheatDesiredStateEnum.CANCELED,
        ModelPreheatExecutionStateEnum.CANCELED,
        ModelPreheatScheduleRunStateEnum.PAUSED,
    )


def test_run_now_rolls_back_when_task_creator_commits_then_fails(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed_schedule(engine)

        async def committing_failure(session, schedule, created_by_user_id):
            del schedule, created_by_user_id
            await session.commit()
            raise RuntimeError("preflight_failed")

        controller = ModelPreheatScheduleController(
            engine, task_creator=committing_failure
        )
        async with AsyncSession(engine) as session:
            schedule = (await session.exec(select(ModelPreheatSchedule))).one()
            with pytest.raises(RuntimeError, match="preflight_failed"):
                await controller.run_now(session, schedule, 1, "failure")
        async with AsyncSession(engine) as session:
            runs = (await session.exec(select(ModelPreheatScheduleRun))).all()
        await engine.dispose()
        return runs

    assert asyncio.run(run()) == []


def test_run_now_does_not_mask_persistent_database_lock_as_concurrency_limit(
    tmp_path,
):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)

        async def locked(session, schedule, created_by_user_id):
            del session, schedule, created_by_user_id
            raise OperationalError("INSERT", {}, RuntimeError("database is locked"))

        controller = ModelPreheatScheduleController(engine, task_creator=locked)
        try:
            async with AsyncSession(engine) as session:
                schedule = await session.get(ModelPreheatSchedule, schedule_id)
                with pytest.raises(OperationalError):
                    await controller.run_now(session, schedule, 1, "locked")
        finally:
            await engine.dispose()

    asyncio.run(run())


def test_lowered_concurrency_uses_active_count_not_slot_range(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine, max_concurrency=3)
        creator = RecordingTaskCreator()
        controller = ModelPreheatScheduleController(engine, task_creator=creator)
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            first = await controller.run_now(session, schedule, 1, "first")
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            second = await controller.run_now(session, schedule, 1, "second")
            second_slot = second.slot
            first = await session.get(ModelPreheatScheduleRun, first.id)
            first.slot = None
            first.state = ModelPreheatScheduleRunStateEnum.READY
            schedule.max_concurrency = 1
            session.add(first)
            session.add(schedule)
            await session.commit()
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            with pytest.raises(ScheduleConcurrencyLimit):
                await controller.run_now(session, schedule, 1, "third")
            active = (
                await session.exec(
                    select(ModelPreheatScheduleRun).where(
                        ModelPreheatScheduleRun.slot.is_not(None)
                    )
                )
            ).all()
        await engine.dispose()
        return second_slot, len(active)

    assert asyncio.run(run()) == (1, 1)


def test_single_tick_fast_forwards_all_elapsed_windows(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        controller = ModelPreheatScheduleController(
            engine, task_creator=RecordingTaskCreator()
        )
        await controller.tick(datetime(2026, 8, 12, 3, 0, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            runs = (
                await session.exec(
                    select(ModelPreheatScheduleRun).order_by(
                        ModelPreheatScheduleRun.window_start_utc
                    )
                )
            ).all()
        await engine.dispose()
        return schedule.next_window_start_utc, [run.state for run in runs]

    next_start, states = asyncio.run(run())
    assert next_start == datetime(2026, 8, 12, 4, 0, tzinfo=UTC)
    assert states == [
        ModelPreheatScheduleRunStateEnum.SKIPPED,
        ModelPreheatScheduleRunStateEnum.SKIPPED,
        ModelPreheatScheduleRunStateEnum.SKIPPED,
        ModelPreheatScheduleRunStateEnum.RUNNING,
    ]


def test_due_task_creator_failure_exits_tick_without_hot_loop(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        calls = 0

        async def unavailable(session, schedule, created_by_user_id):
            nonlocal calls
            del session, schedule, created_by_user_id
            calls += 1
            raise RuntimeError("target_workers_not_idle")

        controller = ModelPreheatScheduleController(engine, task_creator=unavailable)
        await asyncio.wait_for(
            controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC)),
            timeout=0.5,
        )
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            cursor = schedule.next_window_start_utc
        await engine.dispose()
        return calls, cursor

    assert asyncio.run(run()) == (
        1,
        datetime(2026, 8, 12, 0, 0, tzinfo=UTC),
    )


def test_run_now_resumes_paused_task_and_closes_previous_run(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        schedule_id = await _seed_schedule(engine)
        creator = RecordingTaskCreator()
        controller = ModelPreheatScheduleController(engine, task_creator=creator)
        await controller.tick(datetime(2026, 8, 12, 0, 0, tzinfo=UTC))
        await controller.tick(datetime(2026, 8, 12, 0, 31, tzinfo=UTC))
        async with AsyncSession(engine) as session:
            schedule = await session.get(ModelPreheatSchedule, schedule_id)
            manual = await controller.run_now(session, schedule, 1, "resume")
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatTask, manual.task_id)
            child = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.task_id == task.id
                    )
                )
            ).one()
            runs = (
                await session.exec(
                    select(ModelPreheatScheduleRun).order_by(ModelPreheatScheduleRun.id)
                )
            ).all()
        await engine.dispose()
        return manual, task, child, runs

    manual, task, child, runs = asyncio.run(run())
    assert manual.state == ModelPreheatScheduleRunStateEnum.RUNNING
    assert task.desired_state == ModelPreheatDesiredStateEnum.RUNNING
    assert child.state == ModelPreheatWorkerTaskStateEnum.PENDING
    assert runs[0].state == ModelPreheatScheduleRunStateEnum.SKIPPED
    assert runs[0].error_code == "resumed_in_later_window"
