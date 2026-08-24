import asyncio
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import func, or_, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
    ModelPreheatDistributionPolicyRun,
    ModelPreheatDistributionPolicyRunStateEnum,
    ModelPreheatDistributionPolicyRunTriggerEnum,
    ModelPreheatDistributionPolicyTriggerModeEnum,
    distribution_policy_run_operation_key,
)
from gpustack.schemas.model_preheat_schedules import next_window_start_utc


class ModelPreheatDistributionScheduleController:
    """以持久运行记录和租约认领定时 Artifact 分发窗口。"""

    def __init__(self, engine, reconciler=None, interval=15):
        self._engine = engine
        self._reconciler = reconciler
        self._interval = interval
        self._lease_owner = uuid4().hex
        self._lease_ttl = timedelta(seconds=60)

    async def start(self):
        while True:
            await self.tick()
            await asyncio.sleep(self._interval)

    async def tick(self, now=None):
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        async with AsyncSession(self._engine) as session:
            pending_ids = (
                await session.exec(
                    select(ModelPreheatDistributionPolicyRun.id).where(
                        ModelPreheatDistributionPolicyRun.state
                        == ModelPreheatDistributionPolicyRunStateEnum.PENDING
                    )
                )
            ).all()
            policy_ids = (
                await session.exec(
                    select(ModelPreheatDistributionPolicy.id).where(
                        ModelPreheatDistributionPolicy.enabled.is_(True),
                        ModelPreheatDistributionPolicy.trigger_mode
                        == ModelPreheatDistributionPolicyTriggerModeEnum.SCHEDULED,
                        ModelPreheatDistributionPolicy.next_run_at.is_not(None),
                        ModelPreheatDistributionPolicy.next_run_at <= now,
                    )
                )
            ).all()
        for run_id in pending_ids:
            await self._claim_and_execute(run_id)
        for policy_id in policy_ids:
            await self._claim_due_run(policy_id, now)

    async def _claim_due_run(self, policy_id, now):
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
            if (
                policy is None
                or not policy.enabled
                or policy.trigger_mode
                != ModelPreheatDistributionPolicyTriggerModeEnum.SCHEDULED
                or policy.next_run_at is None
                or policy.next_run_at > now
            ):
                return
            window_start = policy.next_run_at
            run = ModelPreheatDistributionPolicyRun(
                policy_id=policy.id,
                trigger=ModelPreheatDistributionPolicyRunTriggerEnum.SCHEDULED,
                window_start_utc=window_start,
                operation_key=distribution_policy_run_operation_key(
                    policy.id, window_start
                ),
            )
            policy.last_run_at = window_start
            policy.next_run_at = next_window_start_utc(policy, window_start)
            session.add(policy)
            session.add(run)
            try:
                await session.commit()
                await session.refresh(run)
            except (IntegrityError, OperationalError):
                await session.rollback()
                return
        await self._claim_and_execute(run.id)

    async def _claim_and_execute(self, run_id):
        token = await self._claim_run(run_id)
        if token is None:
            return
        await self._execute_claimed_run(run_id, token)

    async def _claim_run(self, run_id, now=None):
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        token = uuid4().hex
        async with AsyncSession(self._engine) as session:
            claimed = await session.exec(
                update(ModelPreheatDistributionPolicyRun)
                .where(
                    ModelPreheatDistributionPolicyRun.id == run_id,
                    ModelPreheatDistributionPolicyRun.state
                    == ModelPreheatDistributionPolicyRunStateEnum.PENDING,
                    or_(
                        ModelPreheatDistributionPolicyRun.lease_owner.is_(None),
                        ModelPreheatDistributionPolicyRun.lease_expires_at.is_(None),
                        ModelPreheatDistributionPolicyRun.lease_expires_at <= now,
                    ),
                )
                .values(
                    lease_owner=self._lease_owner,
                    lease_token=token,
                    lease_expires_at=now + self._lease_ttl,
                    started_at=func.coalesce(
                        ModelPreheatDistributionPolicyRun.started_at, now
                    ),
                )
            )
            await session.commit()
        return token if claimed.rowcount == 1 else None

    async def _execute_claimed_run(self, run_id, token):
        error_code = None
        try:
            async with AsyncSession(self._engine) as session:
                run = await session.get(ModelPreheatDistributionPolicyRun, run_id)
                if (
                    run is None
                    or run.state != ModelPreheatDistributionPolicyRunStateEnum.PENDING
                    or run.lease_owner != self._lease_owner
                    or run.lease_token != token
                ):
                    return
                policy = await session.get(
                    ModelPreheatDistributionPolicy, run.policy_id
                )
                if (
                    policy is None
                    or not policy.enabled
                    or policy.trigger_mode
                    != ModelPreheatDistributionPolicyTriggerModeEnum.SCHEDULED
                ):
                    error_code = "distribution_policy_not_active"
                elif self._reconciler is None:
                    error_code = "distribution_reconciler_unavailable"
            if error_code is None:
                if not await self._renew_lease(run_id, token):
                    error_code = "distribution_schedule_lease_lost"
                    await self._finish_run(run_id, token, error_code)
                    return
                lease_lost = asyncio.Event()
                heartbeat = asyncio.create_task(
                    self._lease_heartbeat(run_id, token, lease_lost)
                )
                reconcile_task = asyncio.create_task(
                    self._reconciler.reconcile_policy(
                        run.policy_id,
                        run.operation_key,
                        lease_check=lambda: not lease_lost.is_set(),
                    )
                )
                lost_task = asyncio.create_task(lease_lost.wait())
                try:
                    done, _ = await asyncio.wait(
                        {reconcile_task, lost_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if lease_lost.is_set():
                        reconcile_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await reconcile_task
                        error_code = "distribution_schedule_lease_lost"
                    else:
                        lost_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await lost_task
                        await reconcile_task
                finally:
                    heartbeat.cancel()
                    with suppress(asyncio.CancelledError):
                        await heartbeat
        except Exception as exc:
            error_code = type(exc).__name__
        await self._finish_run(run_id, token, error_code)

    async def _renew_lease(self, run_id, token, now=None):
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        async with AsyncSession(self._engine) as session:
            result = await session.exec(
                update(ModelPreheatDistributionPolicyRun)
                .where(
                    ModelPreheatDistributionPolicyRun.id == run_id,
                    ModelPreheatDistributionPolicyRun.state
                    == ModelPreheatDistributionPolicyRunStateEnum.PENDING,
                    ModelPreheatDistributionPolicyRun.lease_owner == self._lease_owner,
                    ModelPreheatDistributionPolicyRun.lease_token == token,
                    ModelPreheatDistributionPolicyRun.lease_expires_at > now,
                )
                .values(lease_expires_at=now + self._lease_ttl)
            )
            await session.commit()
        return result.rowcount == 1

    async def _lease_heartbeat(self, run_id, token, lease_lost):
        while True:
            await asyncio.sleep(max(0.005, self._lease_ttl.total_seconds() / 4))
            try:
                renewed = await self._renew_lease(run_id, token)
            except Exception:
                renewed = False
            if not renewed:
                lease_lost.set()
                return

    async def _finish_run(self, run_id, token, error_code):
        async with AsyncSession(self._engine) as session:
            await session.exec(
                update(ModelPreheatDistributionPolicyRun)
                .where(
                    ModelPreheatDistributionPolicyRun.id == run_id,
                    ModelPreheatDistributionPolicyRun.state
                    == ModelPreheatDistributionPolicyRunStateEnum.PENDING,
                    ModelPreheatDistributionPolicyRun.lease_owner == self._lease_owner,
                    ModelPreheatDistributionPolicyRun.lease_token == token,
                )
                .values(
                    state=(
                        ModelPreheatDistributionPolicyRunStateEnum.ERROR
                        if error_code
                        else ModelPreheatDistributionPolicyRunStateEnum.READY
                    ),
                    error_code=error_code,
                    finished_at=datetime.now(timezone.utc),
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                )
            )
            await session.commit()
