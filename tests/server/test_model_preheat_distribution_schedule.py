import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
    ModelPreheatDistributionPolicyRun,
    ModelPreheatDistributionPolicyTriggerModeEnum,
)
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import ModelPreheatTargetScopeEnum


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
            async def reconcile_policy(self, policy_id, run_key, lease_check=None):
                calls.append((policy_id, run_key))

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
            async def reconcile_policy(self, policy_id, run_key, lease_check=None):
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
