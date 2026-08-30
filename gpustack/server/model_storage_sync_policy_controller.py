import asyncio
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy import String, case, cast, func, or_, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import HTTPException
from gpustack.schemas.model_storage_sync import (
    ModelStorageSyncBatchCreate,
    ModelStorageSyncBatchItem,
    ModelStorageSyncBatchPublic,
    ModelStorageSyncScopeEnum,
    ModelStorageSyncTask,
    ModelStorageSyncTaskStateEnum,
)
from gpustack.schemas.model_storage_sync_policies import (
    ModelStorageSyncPolicy,
    ModelStorageSyncPolicyRun,
    ModelStorageSyncPolicyRunStateEnum,
    ModelStorageSyncPolicyRunTriggerEnum,
    ModelStorageSyncPolicyTriggerModeEnum,
    sync_policy_operation_key,
)
from gpustack.schemas.workers import (
    MODEL_STORAGE_PROTOCOL_VERSION,
    Worker,
    WorkerStateEnum,
)
from gpustack.schemas.users import User
from gpustack.server.model_preheat_idempotency import canonical_request_hash


class SyncPolicyRunConflict(Exception):
    pass


class SyncPolicyDisabled(Exception):
    pass


class SyncPolicyRunLeaseLost(Exception):
    pass


class ModelStorageSyncPolicyController:
    def __init__(self, engine, config=None, interval=15, batch_executor=None):
        self._engine = engine
        self._config = config
        self._interval = interval
        self._batch_executor = batch_executor or self._execute_existing_batch
        # 每个 Server 实例使用独立 owner，避免过期执行者覆盖重新认领后的终态。
        self._lease_owner = uuid4().hex
        self._lease_ttl = timedelta(seconds=60)
        self._missing_child_grace = timedelta(minutes=5)
        self._ready_repair_cursor = 0

    async def start(self):
        while True:
            await self.tick()
            await asyncio.sleep(self._interval)

    async def tick(self, now=None):
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        await self._repair_inconsistent_ready_runs()
        async with AsyncSession(self._engine) as session:
            pending_ids = (
                await session.exec(
                    select(ModelStorageSyncPolicyRun.id).where(
                        ModelStorageSyncPolicyRun.state
                        == ModelStorageSyncPolicyRunStateEnum.PENDING
                    )
                )
            ).all()
        for run_id in pending_ids:
            await self._claim_and_execute(run_id, self._internal_request())

        async with AsyncSession(self._engine) as session:
            policy_ids = (
                await session.exec(
                    select(ModelStorageSyncPolicy.id).where(
                        ModelStorageSyncPolicy.enabled.is_(True),
                        ModelStorageSyncPolicy.trigger_mode
                        == ModelStorageSyncPolicyTriggerModeEnum.SCHEDULED,
                        ModelStorageSyncPolicy.next_run_at.is_not(None),
                        ModelStorageSyncPolicy.next_run_at <= now,
                    )
                )
            ).all()
        for policy_id in policy_ids:
            await self._claim_due_run(policy_id, now)

    async def run_now(self, session, policy, user_id, idempotency_key, request):
        policy_id = policy.id
        request_hash = canonical_request_hash({"policy_id": policy_id})
        operation_key = sync_policy_operation_key(
            "model_storage_sync_policy.run_now", user_id, idempotency_key
        )
        existing = await self._run_for_operation(session, operation_key)
        if existing is not None:
            if existing.policy_id != policy_id or existing.request_hash != request_hash:
                raise SyncPolicyRunConflict
            return existing
        if not policy.enabled:
            raise SyncPolicyDisabled
        now = datetime.now(timezone.utc)
        run = ModelStorageSyncPolicyRun(
            policy_id=policy_id,
            trigger=ModelStorageSyncPolicyRunTriggerEnum.MANUAL,
            window_start_utc=now,
            operation_key=operation_key,
            request_hash=request_hash,
            created_by_user_id=user_id,
        )
        policy.last_run_at = now
        session.add(policy)
        session.add(run)
        try:
            await session.commit()
            await session.refresh(run)
            policy = await session.get(
                ModelStorageSyncPolicy, policy_id, populate_existing=True
            )
        except IntegrityError:
            await session.rollback()
            existing = await self._run_for_operation(session, operation_key)
            if (
                existing is None
                or existing.policy_id != policy.id
                or existing.request_hash != request_hash
            ):
                raise SyncPolicyRunConflict from None
            return existing
        del request
        return run

    async def _claim_due_run(self, policy_id, now):
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            policy = await session.get(ModelStorageSyncPolicy, policy_id)
            if (
                policy is None
                or not policy.enabled
                or policy.trigger_mode
                != ModelStorageSyncPolicyTriggerModeEnum.SCHEDULED
                or policy.next_run_at is None
                or policy.next_run_at > now
            ):
                return
            window_start = policy.next_run_at
            from gpustack.schemas.model_preheat_schedules import next_window_start_utc

            policy.last_run_at = window_start
            policy.next_run_at = next_window_start_utc(policy, window_start)
            run = ModelStorageSyncPolicyRun(
                policy_id=policy.id,
                trigger=ModelStorageSyncPolicyRunTriggerEnum.SCHEDULED,
                window_start_utc=window_start,
                operation_key=sync_policy_operation_key(
                    "model_storage_sync_policy.scheduled",
                    policy.id,
                    window_start.astimezone(timezone.utc).isoformat(),
                ),
                request_hash=canonical_request_hash({"policy_id": policy.id}),
                created_by_user_id=policy.created_by_user_id,
            )
            session.add(policy)
            session.add(run)
            try:
                await session.commit()
                await session.refresh(run)
            except (IntegrityError, OperationalError):
                await session.rollback()
                return
        await self._claim_and_execute(run.id, self._internal_request())

    async def _claim_and_execute(self, run_id, request):
        lease_token = await self._claim_run(run_id)
        if lease_token is None:
            return await self._get_run(run_id)
        return await self._execute_claimed_run(run_id, request, lease_token)

    async def _claim_run(self, run_id, now=None):
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        lease_token = uuid4().hex
        async with AsyncSession(self._engine) as session:
            claimed = await session.exec(
                update(ModelStorageSyncPolicyRun)
                .where(
                    ModelStorageSyncPolicyRun.id == run_id,
                    ModelStorageSyncPolicyRun.state
                    == ModelStorageSyncPolicyRunStateEnum.PENDING,
                    or_(
                        ModelStorageSyncPolicyRun.lease_owner.is_(None),
                        ModelStorageSyncPolicyRun.lease_expires_at.is_(None),
                        ModelStorageSyncPolicyRun.lease_expires_at <= now,
                    ),
                )
                .values(
                    lease_owner=self._lease_owner,
                    lease_token=lease_token,
                    lease_expires_at=now + self._lease_ttl,
                    attempt=case(
                        (
                            or_(
                                ModelStorageSyncPolicyRun.response_payload.is_(None),
                                cast(
                                    ModelStorageSyncPolicyRun.response_payload,
                                    String,
                                )
                                == "null",
                            ),
                            ModelStorageSyncPolicyRun.attempt + 1,
                        ),
                        else_=ModelStorageSyncPolicyRun.attempt,
                    ),
                    started_at=func.coalesce(ModelStorageSyncPolicyRun.started_at, now),
                )
            )
            await session.commit()
            return lease_token if claimed.rowcount == 1 else None

    async def _renew_lease(self, run_id, lease_token, now=None):
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        async with AsyncSession(self._engine) as session:
            renewed = await session.exec(
                update(ModelStorageSyncPolicyRun)
                .where(
                    ModelStorageSyncPolicyRun.id == run_id,
                    ModelStorageSyncPolicyRun.state
                    == ModelStorageSyncPolicyRunStateEnum.PENDING,
                    ModelStorageSyncPolicyRun.lease_owner == self._lease_owner,
                    ModelStorageSyncPolicyRun.lease_token == lease_token,
                    ModelStorageSyncPolicyRun.lease_expires_at > now,
                )
                .values(lease_expires_at=now + self._lease_ttl)
            )
            await session.commit()
            return renewed.rowcount == 1

    async def _lease_heartbeat(self, run_id, lease_token, lease_lost):
        interval = max(0.05, self._lease_ttl.total_seconds() / 3)
        while True:
            await asyncio.sleep(interval)
            try:
                renewed = await self._renew_lease(run_id, lease_token)
            except Exception:
                renewed = False
            if not renewed:
                lease_lost.set()
                return

    async def _execute_claimed_run(self, run_id, request, lease_token):
        try:
            error_code = None
            async with AsyncSession(self._engine, expire_on_commit=False) as session:
                run = await session.get(ModelStorageSyncPolicyRun, run_id)
                if (
                    run is None
                    or run.state != ModelStorageSyncPolicyRunStateEnum.PENDING
                    or run.lease_owner != self._lease_owner
                    or run.lease_token != lease_token
                ):
                    return await self._get_run(run_id)
                policy = await session.get(ModelStorageSyncPolicy, run.policy_id)
                if policy is None:
                    error_code = "model_storage_sync_policy_not_found"
                elif run.response_payload is not None:
                    result = ModelStorageSyncBatchPublic.model_validate(
                        run.response_payload
                    )
                    response_payload = result.model_dump(mode="json")
                else:
                    run_user_id = await self._run_user_id(session, run, lease_token)
                    if run_user_id is None:
                        error_code = "model_storage_sync_policy_system_user_unavailable"
                    else:
                        batch, unavailable = await self._batch_for_policy(
                            session, policy
                        )
                        if batch is None:
                            result = ModelStorageSyncBatchPublic(
                                scope=policy.scope,
                                planned=0,
                                skipped=unavailable,
                            )
                        else:
                            result = await self._execute_batch_with_lease(
                                run_id,
                                lease_token,
                                request,
                                run_user_id,
                                batch,
                                f"sync-policy:{run.operation_key}",
                            )
                            result.scope = policy.scope
                            result.skipped = unavailable + result.skipped
                        response_payload = result.model_dump(mode="json")
            if error_code is not None:
                return await self._finish_run(
                    run_id,
                    ModelStorageSyncPolicyRunStateEnum.ERROR,
                    lease_token,
                    error_code=error_code,
                )
            started_at = run.started_at
            if started_at is not None and started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            missing_is_expired = bool(
                started_at is not None
                and datetime.now(timezone.utc) - started_at >= self._missing_child_grace
            )
            terminal_state, result_error = await self._result_terminal_state(
                result, missing_is_expired=missing_is_expired
            )
            if terminal_state is None:
                return await self._defer_run(
                    run_id, lease_token, response_payload=response_payload
                )
            return await self._finish_run(
                run_id,
                terminal_state,
                lease_token,
                response_payload=response_payload,
                error_code=result_error,
            )
        except SyncPolicyRunLeaseLost:
            return await self._get_run(run_id)
        except HTTPException as exc:
            if str(exc.message) == "model_storage_sync_batch_in_progress":
                await self._release_run_lease(run_id, lease_token)
                return await self._get_run(run_id)
            return await self._finish_run(
                run_id,
                ModelStorageSyncPolicyRunStateEnum.ERROR,
                lease_token,
                error_code=str(exc.message),
            )
        except Exception as exc:
            return await self._finish_run(
                run_id,
                ModelStorageSyncPolicyRunStateEnum.ERROR,
                lease_token,
                error_code=type(exc).__name__,
            )

    async def _execute_batch_with_lease(
        self, run_id, lease_token, request, user_id, batch, idempotency_key
    ):
        if not await self._renew_lease(run_id, lease_token):
            raise SyncPolicyRunLeaseLost
        lease_lost = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._lease_heartbeat(run_id, lease_token, lease_lost)
        )
        try:
            async with AsyncSession(self._engine, expire_on_commit=False) as session:
                batch_task = asyncio.create_task(
                    self._batch_executor(
                        request, session, user_id, batch, idempotency_key
                    )
                )
                lost_task = asyncio.create_task(lease_lost.wait())
                done, _pending = await asyncio.wait(
                    {batch_task, lost_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if lease_lost.is_set():
                    batch_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await batch_task
                    raise SyncPolicyRunLeaseLost
                lost_task.cancel()
                with suppress(asyncio.CancelledError):
                    await lost_task
                return await batch_task
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

    async def _finish_run(
        self, run_id, state, lease_token, response_payload=None, error_code=None
    ):
        now = datetime.now(timezone.utc)
        async with AsyncSession(self._engine) as session:
            await session.exec(
                update(ModelStorageSyncPolicyRun)
                .where(
                    ModelStorageSyncPolicyRun.id == run_id,
                    ModelStorageSyncPolicyRun.state
                    == ModelStorageSyncPolicyRunStateEnum.PENDING,
                    ModelStorageSyncPolicyRun.lease_owner == self._lease_owner,
                    ModelStorageSyncPolicyRun.lease_token == lease_token,
                )
                .values(
                    state=state,
                    response_payload=response_payload,
                    error_code=error_code,
                    finished_at=now,
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                )
            )
            await session.commit()
        return await self._get_run(run_id)

    async def _defer_run(self, run_id, lease_token, response_payload):
        """保存规划结果并释放 lease，等待已创建子任务进入终态。"""
        async with AsyncSession(self._engine) as session:
            await session.exec(
                update(ModelStorageSyncPolicyRun)
                .where(
                    ModelStorageSyncPolicyRun.id == run_id,
                    ModelStorageSyncPolicyRun.state
                    == ModelStorageSyncPolicyRunStateEnum.PENDING,
                    ModelStorageSyncPolicyRun.lease_owner == self._lease_owner,
                    ModelStorageSyncPolicyRun.lease_token == lease_token,
                )
                .values(
                    response_payload=response_payload,
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                )
            )
            await session.commit()
        return await self._get_run(run_id)

    async def _result_terminal_state(
        self,
        result,
        preserve_ready_on_missing=False,
        missing_is_expired=False,
    ):
        error_codes = {item.reason for item in result.failed if item.reason}
        task_ids = [item.task_id for item in result.created if item.task_id is not None]
        if len(task_ids) != len(result.created):
            error_codes.add("model_storage_sync_task_not_found")
        if not task_ids:
            skipped_codes = {item.reason for item in result.skipped if item.reason}
            if skipped_codes:
                error_codes.update(skipped_codes)
            if not error_codes:
                return ModelStorageSyncPolicyRunStateEnum.READY, None
            error_code = (
                next(iter(error_codes))
                if len(error_codes) == 1
                else "model_storage_sync_policy_run_failed"
            )
            return ModelStorageSyncPolicyRunStateEnum.ERROR, error_code
        async with AsyncSession(self._engine) as session:
            tasks = (
                await session.exec(
                    select(ModelStorageSyncTask).where(
                        ModelStorageSyncTask.id.in_(task_ids)
                    )
                )
            ).all()
        tasks_by_id = {task.id: task for task in tasks}
        has_missing = len(tasks_by_id) != len(set(task_ids))
        active_tasks = [
            tasks_by_id[task_id]
            for task_id in task_ids
            if task_id in tasks_by_id
            and tasks_by_id[task_id].state
            not in {
                ModelStorageSyncTaskStateEnum.READY,
                ModelStorageSyncTaskStateEnum.ERROR,
                ModelStorageSyncTaskStateEnum.CANCELED,
            }
        ]
        if active_tasks:
            return None, None
        failed_tasks = [
            tasks_by_id[task_id]
            for task_id in task_ids
            if task_id in tasks_by_id
            if tasks_by_id[task_id].state
            in {
                ModelStorageSyncTaskStateEnum.ERROR,
                ModelStorageSyncTaskStateEnum.CANCELED,
            }
        ]
        if failed_tasks:
            error_codes.update(
                {
                    task.error_code
                    or (
                        "model_storage_sync_task_canceled"
                        if task.state == ModelStorageSyncTaskStateEnum.CANCELED
                        else "model_storage_sync_task_failed"
                    )
                    for task in failed_tasks
                }
            )
        if has_missing:
            if preserve_ready_on_missing:
                if not error_codes:
                    return ModelStorageSyncPolicyRunStateEnum.READY, None
            elif missing_is_expired:
                error_codes.add("model_storage_sync_task_result_unavailable")
            else:
                return None, None
        if not error_codes:
            return ModelStorageSyncPolicyRunStateEnum.READY, None
        error_code = (
            next(iter(error_codes))
            if len(error_codes) == 1
            else "model_storage_sync_policy_run_failed"
        )
        return ModelStorageSyncPolicyRunStateEnum.ERROR, error_code

    async def _repair_inconsistent_ready_runs(self):
        """纠正旧版本曾过早标记 READY 的运行，不重放批次。"""

        async def load_rows(after_id):
            async with AsyncSession(self._engine) as session:
                return (
                    await session.exec(
                        select(
                            ModelStorageSyncPolicyRun.id,
                            ModelStorageSyncPolicyRun.response_payload,
                        )
                        .where(
                            ModelStorageSyncPolicyRun.id > after_id,
                            ModelStorageSyncPolicyRun.state
                            == ModelStorageSyncPolicyRunStateEnum.READY,
                            ModelStorageSyncPolicyRun.response_payload.is_not(None),
                        )
                        .order_by(ModelStorageSyncPolicyRun.id)
                        .limit(500)
                    )
                ).all()

        rows = await load_rows(self._ready_repair_cursor)
        if not rows and self._ready_repair_cursor:
            self._ready_repair_cursor = 0
            rows = await load_rows(0)
        if rows:
            self._ready_repair_cursor = max(run_id for run_id, _payload in rows)
        for run_id, response_payload in rows:
            try:
                result = ModelStorageSyncBatchPublic.model_validate(response_payload)
            except Exception:
                continue
            terminal_state, error_code = await self._result_terminal_state(
                result, preserve_ready_on_missing=True
            )
            if terminal_state == ModelStorageSyncPolicyRunStateEnum.READY:
                continue
            values = {
                "state": terminal_state or ModelStorageSyncPolicyRunStateEnum.PENDING,
                "error_code": error_code,
            }
            if terminal_state is None:
                values["finished_at"] = None
            else:
                values["finished_at"] = datetime.now(timezone.utc)
            async with AsyncSession(self._engine) as session:
                await session.exec(
                    update(ModelStorageSyncPolicyRun)
                    .where(
                        ModelStorageSyncPolicyRun.id == run_id,
                        ModelStorageSyncPolicyRun.state
                        == ModelStorageSyncPolicyRunStateEnum.READY,
                    )
                    .values(**values)
                )
                await session.commit()

    async def _release_run_lease(self, run_id, lease_token):
        async with AsyncSession(self._engine) as session:
            await session.exec(
                update(ModelStorageSyncPolicyRun)
                .where(
                    ModelStorageSyncPolicyRun.id == run_id,
                    ModelStorageSyncPolicyRun.state
                    == ModelStorageSyncPolicyRunStateEnum.PENDING,
                    ModelStorageSyncPolicyRun.lease_owner == self._lease_owner,
                    ModelStorageSyncPolicyRun.lease_token == lease_token,
                )
                .values(
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                )
            )
            await session.commit()

    async def _get_run(self, run_id):
        async with AsyncSession(self._engine) as session:
            return await session.get(ModelStorageSyncPolicyRun, run_id)

    async def _run_user_id(self, session, run, lease_token):
        if run.execution_user_id is not None:
            return run.execution_user_id
        user_id = None
        if run.created_by_user_id is not None:
            creator = await session.get(User, run.created_by_user_id)
            if creator is not None and creator.deleted_at is None:
                user_id = creator.id
        if user_id is None:
            user_id = (
                await session.exec(
                    select(User.id)
                    .where(User.is_admin.is_(True), User.deleted_at.is_(None))
                    .order_by(User.id)
                    .limit(1)
                )
            ).first()
        if user_id is None:
            return None
        bound = await session.exec(
            update(ModelStorageSyncPolicyRun)
            .where(
                ModelStorageSyncPolicyRun.id == run.id,
                ModelStorageSyncPolicyRun.state
                == ModelStorageSyncPolicyRunStateEnum.PENDING,
                ModelStorageSyncPolicyRun.lease_owner == self._lease_owner,
                ModelStorageSyncPolicyRun.lease_token == lease_token,
                ModelStorageSyncPolicyRun.execution_user_id.is_(None),
            )
            .values(execution_user_id=user_id)
        )
        await session.commit()
        if bound.rowcount == 1:
            return user_id
        current = await session.get(
            ModelStorageSyncPolicyRun, run.id, populate_existing=True
        )
        if current is None or current.lease_token != lease_token:
            raise SyncPolicyRunLeaseLost
        return current.execution_user_id

    async def _batch_for_policy(self, session, policy):
        if policy.scope == ModelStorageSyncScopeEnum.SINGLE_MODEL:
            return (
                ModelStorageSyncBatchCreate(
                    profile_id=policy.profile_id,
                    scope=ModelStorageSyncScopeEnum.SINGLE_MODEL,
                    model_file_id=policy.model_file_id,
                ),
                [],
            )
        if policy.scope == ModelStorageSyncScopeEnum.ALL_READY_WORKERS:
            return (
                ModelStorageSyncBatchCreate(
                    profile_id=policy.profile_id,
                    scope=ModelStorageSyncScopeEnum.ALL_READY_WORKERS,
                ),
                [],
            )

        rows = (
            await session.exec(
                select(Worker)
                .where(Worker.worker_uuid.in_(policy.worker_uuids))
                .order_by(Worker.id.desc())
            )
        ).all()
        current = {}
        for worker in rows:
            current.setdefault(worker.worker_uuid, worker)
        eligible_ids = []
        unavailable = []
        for worker_uuid in policy.worker_uuids:
            worker = current.get(worker_uuid)
            if worker is None:
                reason = "worker_not_registered"
            elif worker.state != WorkerStateEnum.READY:
                reason = "worker_not_ready"
            elif (
                worker.model_storage_protocol_version != MODEL_STORAGE_PROTOCOL_VERSION
            ):
                reason = "worker_protocol_unsupported"
            else:
                eligible_ids.append(worker.id)
                continue
            unavailable.append(
                ModelStorageSyncBatchItem(worker_uuid=worker_uuid, reason=reason)
            )
        if not eligible_ids:
            return None, unavailable
        return (
            ModelStorageSyncBatchCreate(
                profile_id=policy.profile_id,
                scope=ModelStorageSyncScopeEnum.SELECTED_WORKERS,
                worker_ids=eligible_ids,
            ),
            unavailable,
        )

    async def _execute_existing_batch(
        self, request, session, user_id, batch, idempotency_key
    ):
        from gpustack.routes.model_storage import create_model_storage_sync_batch

        return await create_model_storage_sync_batch(
            request,
            session,
            SimpleNamespace(id=user_id),
            batch,
            idempotency_key,
        )

    async def _run_for_operation(self, session, operation_key):
        return (
            await session.exec(
                select(ModelStorageSyncPolicyRun).where(
                    ModelStorageSyncPolicyRun.operation_key == operation_key
                )
            )
        ).first()

    def _internal_request(self):
        return SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(server_config=self._config))
        )
