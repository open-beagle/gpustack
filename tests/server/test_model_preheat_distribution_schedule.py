import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
    ModelPreheatDistributionPolicyRun,
    ModelPreheatDistributionPolicyRunStateEnum,
    ModelPreheatDistributionPolicyRunTask,
    ModelPreheatDistributionPolicyRunTriggerEnum,
    ModelPreheatDistributionPolicyTriggerModeEnum,
)
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import (
    ModelPreheatTargetScopeEnum,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.server.policy_run_observability import distribution_run_observations


def test_scheduled_distribution_claims_one_window_across_controllers(tmp_path):
    from gpustack.server.model_preheat_distribution_schedule_controller import (
        ModelPreheatDistributionScheduleController,
    )

    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'distribution-schedule.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        now = datetime.now(timezone.utc)
        async with AsyncSession(engine) as session:
            policy = ModelPreheatDistributionPolicy(
                name="scheduled-artifact",
                profile_id=1,
                profile_config_version=1,
                request_identity={"source": "huggingface", "model_id": "org/model"},
                request_digest="a" * 64,
                target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                worker_selector={"worker_uuids": ["worker-a"]},
                gpu_selector={},
                selector_digest="b" * 64,
                trigger_mode=ModelPreheatDistributionPolicyTriggerModeEnum.SCHEDULED,
                cron_expression="* * * * *",
                timezone="UTC",
                next_run_at=now,
            )
            session.add(policy)
            await session.commit()
        calls = []

        class Reconciler:
            async def reconcile_policy(
                self, policy_id, run_key, lease_check=None, run_id=None
            ):
                calls.append((policy_id, run_key, run_id))
                return {
                    "outcome": {"created": [], "skipped": [], "failed": []},
                    "error_code": "distribution_no_eligible_workers",
                }

        controller_a = ModelPreheatDistributionScheduleController(engine, Reconciler())
        controller_b = ModelPreheatDistributionScheduleController(engine, Reconciler())
        await asyncio.gather(
            controller_a.tick(now),
            controller_b.tick(now),
        )
        async with AsyncSession(engine) as session:
            runs = (await session.exec(select(ModelPreheatDistributionPolicyRun))).all()
        await engine.dispose()
        return runs, calls

    runs, calls = asyncio.run(run())
    assert len(runs) == 1
    assert len(calls) == 1
    assert calls[0][2] == runs[0].id
    assert runs[0].state == ModelPreheatDistributionPolicyRunStateEnum.ERROR
    assert runs[0].error_code == "distribution_no_eligible_workers"
    assert runs[0].outcome == {"created": [], "skipped": [], "failed": []}


def test_scheduled_distribution_renews_short_lease_during_reconcile(tmp_path):
    from gpustack.server.model_preheat_distribution_schedule_controller import (
        ModelPreheatDistributionScheduleController,
    )

    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'short-lease.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        now = datetime.now(timezone.utc)
        async with AsyncSession(engine) as session:
            session.add(
                ModelPreheatDistributionPolicy(
                    name="short-lease",
                    profile_id=1,
                    profile_config_version=1,
                    request_identity={"source": "huggingface", "model_id": "org/model"},
                    request_digest="a" * 64,
                    target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                    worker_selector={"worker_uuids": ["worker-a"]},
                    gpu_selector={},
                    selector_digest="b" * 64,
                    trigger_mode=ModelPreheatDistributionPolicyTriggerModeEnum.SCHEDULED,
                    cron_expression="* * * * *",
                    timezone="UTC",
                    next_run_at=now,
                )
            )
            await session.commit()
        calls = []

        class Reconciler:
            async def reconcile_policy(
                self, policy_id, run_key, lease_check=None, run_id=None
            ):
                calls.append(policy_id)
                for _ in range(6):
                    await asyncio.sleep(0.06)
                    assert lease_check()

        first = ModelPreheatDistributionScheduleController(engine, Reconciler())
        second = ModelPreheatDistributionScheduleController(engine, Reconciler())
        first._lease_ttl = timedelta(seconds=0.2)
        second._lease_ttl = timedelta(seconds=0.2)
        task = asyncio.create_task(first.tick(now))
        await asyncio.sleep(0.25)
        await second.tick(now + timedelta(seconds=0.25))
        await task
        await engine.dispose()
        return calls

    assert asyncio.run(run()) == [1]


def test_scheduled_distribution_persists_partial_outcome_and_real_task_link(tmp_path):
    from gpustack.server.model_preheat_distribution_schedule_controller import (
        ModelPreheatDistributionScheduleController,
    )

    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'partial-outcome.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        now = datetime.now(timezone.utc)
        async with AsyncSession(engine, expire_on_commit=False) as session:
            policy = ModelPreheatDistributionPolicy(
                name="partial-outcome",
                profile_id=1,
                profile_config_version=1,
                request_identity={"source": "huggingface", "model_id": "org/model"},
                request_digest="a" * 64,
                target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                worker_selector={"worker_uuids": ["worker-a", "worker-b"]},
                gpu_selector={},
                selector_digest="b" * 64,
                trigger_mode=ModelPreheatDistributionPolicyTriggerModeEnum.SCHEDULED,
                cron_expression="* * * * *",
                timezone="UTC",
                next_run_at=now,
            )
            session.add(policy)
            await session.commit()

        class Reconciler:
            async def reconcile_policy(
                self, policy_id, run_key, lease_check=None, run_id=None
            ):
                async with AsyncSession(engine, expire_on_commit=False) as session:
                    task = ModelPreheatWorkerTask(
                        distribution_policy_id=policy_id,
                        operation_key=f"{run_key}:worker-a",
                        worker_uuid="worker-a",
                        role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                        state=ModelPreheatWorkerTaskStateEnum.READY,
                        progress=100,
                    )
                    session.add(task)
                    await session.flush()
                    session.add(
                        ModelPreheatDistributionPolicyRunTask(
                            run_id=run_id, task_id=task.id
                        )
                    )
                    await session.commit()
                return {
                    "outcome": {
                        "created": [
                            {
                                "task_id": task.id,
                                "worker_id": None,
                                "worker_uuid": "worker-a",
                            }
                        ],
                        "skipped": [],
                        "failed": [
                            {
                                "task_id": None,
                                "worker_id": None,
                                "worker_uuid": "worker-b",
                                "reason": "distribution_connectivity_not_ready",
                            }
                        ],
                    },
                    "error_code": "distribution_partial_outcome",
                }

        controller = ModelPreheatDistributionScheduleController(engine, Reconciler())
        await controller.tick(now)
        async with AsyncSession(engine) as session:
            stored = (
                await session.exec(select(ModelPreheatDistributionPolicyRun))
            ).one()
            observation = (
                await distribution_run_observations(
                    session, [stored], include_tasks=True
                )
            )[stored.id]
        await engine.dispose()
        return stored, observation

    stored, observation = asyncio.run(run())
    assert stored.state == ModelPreheatDistributionPolicyRunStateEnum.PENDING
    assert stored.finished_at is None
    assert stored.error_code == "distribution_partial_outcome"
    assert observation.execution_state.value == "partial_error"
    assert observation.summary.total == 2
    assert observation.summary.ready == 1
    assert observation.summary.error == 1
    assert observation.tasks[1].worker_uuid == "worker-b"
    assert observation.tasks[1].error_code == "distribution_connectivity_not_ready"


def test_schedule_controller_does_not_claim_manual_or_active_planner_runs(tmp_path):
    from gpustack.server.model_preheat_distribution_schedule_controller import (
        ModelPreheatDistributionScheduleController,
    )

    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'claim-boundary.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        now = datetime.now(timezone.utc)
        async with AsyncSession(engine, expire_on_commit=False) as session:
            policy = ModelPreheatDistributionPolicy(
                name="scheduled-artifact",
                profile_id=1,
                profile_config_version=1,
                request_identity={"source": "huggingface", "model_id": "org/model"},
                request_digest="a" * 64,
                target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                worker_selector={"worker_uuids": ["worker-a"]},
                gpu_selector={},
                selector_digest="b" * 64,
                trigger_mode=ModelPreheatDistributionPolicyTriggerModeEnum.SCHEDULED,
                cron_expression="* * * * *",
                timezone="UTC",
                next_run_at=None,
            )
            session.add(policy)
            await session.flush()
            manual_run = ModelPreheatDistributionPolicyRun(
                policy_id=policy.id,
                trigger=ModelPreheatDistributionPolicyRunTriggerEnum.MANUAL,
                window_start_utc=now,
                operation_key="manual-run",
            )
            planned_run = ModelPreheatDistributionPolicyRun(
                policy_id=policy.id,
                trigger=ModelPreheatDistributionPolicyRunTriggerEnum.SCHEDULED,
                window_start_utc=now + timedelta(minutes=1),
                operation_key="planned-run",
                lease_owner="planner",
                lease_token="token",
                lease_expires_at=now + timedelta(minutes=1),
            )
            session.add_all([manual_run, planned_run])
            await session.flush()
            task = ModelPreheatWorkerTask(
                distribution_policy_id=policy.id,
                operation_key="planned-task",
                worker_uuid="worker-a",
                role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                state=ModelPreheatWorkerTaskStateEnum.PENDING,
            )
            session.add(task)
            await session.flush()
            session.add(
                ModelPreheatDistributionPolicyRunTask(
                    run_id=planned_run.id,
                    task_id=task.id,
                )
            )
            await session.commit()
            manual_run_id = manual_run.id
            planned_run_id = planned_run.id

        calls = []

        class Reconciler:
            async def reconcile_policy(
                self, policy_id, run_key, lease_check=None, run_id=None
            ):
                calls.append((policy_id, run_key, run_id))
                return {
                    "outcome": {"created": [], "skipped": [], "failed": []},
                    "error_code": "should_not_run",
                }

        controller = ModelPreheatDistributionScheduleController(engine, Reconciler())
        await controller.tick(now)
        async with AsyncSession(engine) as session:
            manual = await session.get(ModelPreheatDistributionPolicyRun, manual_run_id)
            planned = await session.get(
                ModelPreheatDistributionPolicyRun, planned_run_id
            )
            values = (
                manual.state,
                manual.error_code,
                planned.state,
                planned.error_code,
            )
        await engine.dispose()
        return values, calls

    values, calls = asyncio.run(run())
    assert calls == []
    assert values == (
        ModelPreheatDistributionPolicyRunStateEnum.PENDING,
        None,
        ModelPreheatDistributionPolicyRunStateEnum.PENDING,
        None,
    )


def test_schedule_controller_resumes_expired_partially_planned_run(tmp_path):
    from gpustack.server.model_preheat_distribution_schedule_controller import (
        ModelPreheatDistributionScheduleController,
    )

    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'resume-partial.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        now = datetime.now(timezone.utc)
        async with AsyncSession(engine, expire_on_commit=False) as session:
            policy = ModelPreheatDistributionPolicy(
                name="resume-partial",
                profile_id=1,
                profile_config_version=1,
                request_identity={"source": "huggingface", "model_id": "org/model"},
                request_digest="a" * 64,
                target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                worker_selector={"worker_uuids": ["worker-a", "worker-b"]},
                gpu_selector={},
                selector_digest="b" * 64,
                trigger_mode=ModelPreheatDistributionPolicyTriggerModeEnum.SCHEDULED,
                cron_expression="* * * * *",
                timezone="UTC",
                next_run_at=None,
            )
            session.add(policy)
            await session.flush()
            run = ModelPreheatDistributionPolicyRun(
                policy_id=policy.id,
                trigger=ModelPreheatDistributionPolicyRunTriggerEnum.SCHEDULED,
                window_start_utc=now,
                operation_key="resume-run",
                lease_owner="crashed",
                lease_token="old-token",
                lease_expires_at=now - timedelta(seconds=1),
            )
            session.add(run)
            await session.flush()
            task = ModelPreheatWorkerTask(
                distribution_policy_id=policy.id,
                operation_key="resume-run:worker-a",
                worker_uuid="worker-a",
                role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                state=ModelPreheatWorkerTaskStateEnum.PENDING,
            )
            session.add(task)
            await session.flush()
            session.add(
                ModelPreheatDistributionPolicyRunTask(run_id=run.id, task_id=task.id)
            )
            await session.commit()
            run_id = run.id

        calls = []

        class Reconciler:
            async def reconcile_policy(
                self, policy_id, run_key, lease_check=None, run_id=None
            ):
                calls.append((policy_id, run_key, run_id, lease_check()))
                return {
                    "outcome": {
                        "created": [
                            {
                                "task_id": 1,
                                "worker_id": None,
                                "worker_uuid": "worker-a",
                            }
                        ],
                        "skipped": [],
                        "failed": [],
                    },
                    "error_code": None,
                }

        controller = ModelPreheatDistributionScheduleController(engine, Reconciler())
        await controller.tick(now)
        async with AsyncSession(engine) as session:
            stored = await session.get(ModelPreheatDistributionPolicyRun, run_id)
            values = (
                stored.state,
                stored.lease_owner,
                stored.lease_token,
                stored.lease_expires_at,
                stored.finished_at,
            )
        await engine.dispose()
        return values, calls, run_id

    values, calls, run_id = asyncio.run(run())
    assert calls == [(1, "resume-run", run_id, True)]
    assert values == (
        ModelPreheatDistributionPolicyRunStateEnum.PENDING,
        None,
        None,
        None,
        None,
    )


def test_schedule_controller_does_not_reclaim_planned_run_waiting_for_tasks(tmp_path):
    from gpustack.server.model_preheat_distribution_schedule_controller import (
        ModelPreheatDistributionScheduleController,
    )

    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'planned-waiting.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        now = datetime.now(timezone.utc)
        async with AsyncSession(engine, expire_on_commit=False) as session:
            policy = ModelPreheatDistributionPolicy(
                name="planned-waiting",
                profile_id=1,
                profile_config_version=1,
                request_identity={"source": "huggingface", "model_id": "org/model"},
                request_digest="a" * 64,
                target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                worker_selector={"worker_uuids": ["worker-a"]},
                gpu_selector={},
                selector_digest="b" * 64,
                trigger_mode=ModelPreheatDistributionPolicyTriggerModeEnum.SCHEDULED,
                cron_expression="* * * * *",
                timezone="UTC",
                next_run_at=None,
            )
            session.add(policy)
            await session.flush()
            run = ModelPreheatDistributionPolicyRun(
                policy_id=policy.id,
                trigger=ModelPreheatDistributionPolicyRunTriggerEnum.SCHEDULED,
                window_start_utc=now,
                operation_key="planned-waiting-run",
            )
            session.add(run)
            await session.flush()
            task = ModelPreheatWorkerTask(
                distribution_policy_id=policy.id,
                operation_key="planned-waiting-task",
                worker_uuid="worker-a",
                role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                state=ModelPreheatWorkerTaskStateEnum.RUNNING,
            )
            session.add(task)
            await session.flush()
            session.add(
                ModelPreheatDistributionPolicyRunTask(run_id=run.id, task_id=task.id)
            )
            await session.commit()

        calls = []

        class Reconciler:
            async def reconcile_policy(
                self, policy_id, run_key, lease_check=None, run_id=None
            ):
                calls.append((policy_id, run_key, run_id))
                return {
                    "outcome": {"created": [], "skipped": [], "failed": []},
                    "error_code": "should_not_run",
                }

        controller = ModelPreheatDistributionScheduleController(engine, Reconciler())
        await controller.tick(now + timedelta(seconds=15))
        await engine.dispose()
        return calls

    assert asyncio.run(run()) == []
