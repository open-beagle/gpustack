import asyncio
from concurrent.futures import ProcessPoolExecutor
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Protocol

from minio import Minio

from gpustack.api.exceptions import HTTPException
from gpustack.schemas.model_preheats import (
    ModelPreheatWorkerTaskClaim,
    ModelPreheatWorkerTaskComplete,
    ModelPreheatWorkerTaskFail,
    ModelPreheatWorkerTaskLease,
    ModelPreheatWorkerTaskProgress,
    ModelPreheatWorkerTaskPublic,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker.model_preheat.executor import (
    execute_profile_connectivity_check,
)


logger = logging.getLogger(__name__)


class ModelPreheatExecutionHandler(Protocol):
    def __call__(
        self,
        payload,
        context: "ModelPreheatExecutionContext",
    ) -> Awaitable[dict]: ...


@dataclass(frozen=True)
class _LeaseIdentity:
    worker_task_id: int
    worker_uuid: str
    worker_id: int
    attempt: int
    token: str

    def request(self):
        return ModelPreheatWorkerTaskLease(
            worker_uuid=self.worker_uuid,
            worker_id=self.worker_id,
            attempt=self.attempt,
            lease_token=self.token,
        )


class ModelPreheatExecutionContext:
    def __init__(self, client, lease: _LeaseIdentity):
        self._client = client
        self._lease = lease

    async def heartbeat(self):
        return await self._client.aheartbeat(
            id=self._lease.worker_task_id,
            lease=self._lease.request(),
        )

    async def progress(
        self,
        progress: float,
        *,
        downloaded_size: Optional[int] = None,
        total_size: Optional[int] = None,
        resumable_cursor: Optional[dict] = None,
        state_message: Optional[str] = None,
    ):
        request = ModelPreheatWorkerTaskProgress(
            **self._lease.request().model_dump(),
            progress=progress,
            downloaded_size=downloaded_size,
            total_size=total_size,
            resumable_cursor=resumable_cursor,
            state_message=state_message,
        )
        return await self._client.aprogress(
            id=self._lease.worker_task_id,
            progress=request,
        )


class ModelPreheatManager:
    def __init__(
        self,
        worker_id: int,
        worker_uuid: str,
        clientset,
        execution_handler: Optional[ModelPreheatExecutionHandler] = None,
        *,
        reconnect_delay: float = 5,
        heartbeat_interval: float = 20,
        reconcile_interval: float = 15,
        max_concurrent_tasks: int = 1,
    ):
        self._worker_id = worker_id
        self._worker_uuid = worker_uuid
        self._client = clientset.model_preheat_worker_tasks
        self._uses_default_handler = execution_handler is None
        self._execution_handler = execution_handler or self._execute_supported_task
        self._reconnect_delay = reconnect_delay
        self._heartbeat_interval = heartbeat_interval
        self._reconcile_interval = reconcile_interval
        self._max_concurrent_tasks = max_concurrent_tasks
        self._active_tasks: dict[int, asyncio.Task] = {}

    async def watch_model_preheat_tasks(self):
        reconciliation = asyncio.create_task(self._reconcile_loop())
        try:
            while True:
                try:
                    await self._client.awatch(
                        callback=self.handle_event,
                        params={
                            "worker_uuid": self._worker_uuid,
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "模型预热任务 watch 中断，准备重连。error_type=%s",
                        type(exc).__name__,
                    )
                    await asyncio.sleep(self._reconnect_delay)
        finally:
            reconciliation.cancel()
            await asyncio.gather(reconciliation, return_exceptions=True)
            await self.shutdown()

    def handle_event(self, event: Event):
        if event.type not in {EventType.CREATED, EventType.UPDATED}:
            return
        try:
            worker_task = ModelPreheatWorkerTaskPublic.model_validate(event.data)
        except Exception as exc:
            logger.warning(
                "忽略无效的模型预热任务事件。error_type=%s",
                type(exc).__name__,
            )
            return
        if (
            worker_task.worker_uuid != self._worker_uuid
            or worker_task.state
            not in {
                ModelPreheatWorkerTaskStateEnum.PENDING,
                ModelPreheatWorkerTaskStateEnum.RUNNING,
            }
            or not self._supports_role(worker_task.role)
            or worker_task.id in self._active_tasks
            or len(self._active_tasks) >= self._max_concurrent_tasks
        ):
            return
        task = asyncio.create_task(self._claim_and_execute(worker_task.id))
        self._active_tasks[worker_task.id] = task
        task.add_done_callback(
            lambda completed, task_id=worker_task.id: self._remove_active_task(
                task_id, completed
            )
        )

    async def shutdown(self):
        tasks = list(self._active_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_tasks.clear()

    async def _reconcile_loop(self):
        while True:
            try:
                for worker_task in await self._list_worker_tasks():
                    self.handle_event(Event(EventType.UPDATED, worker_task))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "模型预热任务租约核对失败。error_type=%s",
                    type(exc).__name__,
                )
            await asyncio.sleep(self._reconcile_interval)

    async def _list_worker_tasks(self):
        items = []
        page = 1
        while True:
            listed = await asyncio.to_thread(
                self._client.list,
                params={
                    "worker_uuid": self._worker_uuid,
                    "state": [
                        ModelPreheatWorkerTaskStateEnum.PENDING.value,
                        ModelPreheatWorkerTaskStateEnum.RUNNING.value,
                    ],
                    "page": page,
                    "perPage": 100,
                },
            )
            items.extend(listed.items)
            if page >= listed.pagination.totalPage:
                return items
            page += 1

    def _supports_role(self, role):
        if not self._uses_default_handler:
            return True
        return role == ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK

    async def _claim_and_execute(self, worker_task_id: int):
        claim_request = ModelPreheatWorkerTaskClaim(
            worker_uuid=self._worker_uuid,
            worker_id=self._worker_id,
        )
        try:
            claim = await self._client.aclaim(
                id=worker_task_id,
                claim=claim_request,
            )
        except HTTPException as exc:
            if exc.status_code == 409:
                return
            logger.error(
                "模型预热任务领取失败。worker_task_id=%s error_type=%s",
                worker_task_id,
                type(exc).__name__,
            )
            return
        except Exception as exc:
            logger.error(
                "模型预热任务领取失败。worker_task_id=%s error_type=%s",
                worker_task_id,
                type(exc).__name__,
            )
            return

        lease = _LeaseIdentity(
            worker_task_id=worker_task_id,
            worker_uuid=self._worker_uuid,
            worker_id=self._worker_id,
            attempt=claim.attempt,
            token=claim.lease_token,
        )
        try:
            payload = await self._client.aget_execution_payload(
                id=worker_task_id,
                worker_uuid=self._worker_uuid,
                worker_id=self._worker_id,
                attempt=claim.attempt,
                token=claim.lease_token,
            )
        except Exception as exc:
            logger.error(
                "模型预热执行参数获取失败。worker_task_id=%s error_type=%s",
                worker_task_id,
                type(exc).__name__,
            )
            await self._fail(lease, "execution_payload_unavailable")
            return

        context = ModelPreheatExecutionContext(self._client, lease)
        execution = asyncio.create_task(self._execution_handler(payload, context))
        heartbeat = asyncio.create_task(self._heartbeat_loop(context))
        try:
            done, _ = await asyncio.wait(
                {execution, heartbeat}, return_when=asyncio.FIRST_COMPLETED
            )
            if heartbeat in done and not heartbeat.cancelled():
                heartbeat_error = heartbeat.exception()
                if heartbeat_error is not None:
                    execution.cancel()
                    await asyncio.gather(execution, return_exceptions=True)
                    return
            result = await execution
        except asyncio.CancelledError:
            execution.cancel()
            raise
        except Exception as exc:
            logger.error(
                "模型预热任务执行失败。worker_task_id=%s error_type=%s",
                worker_task_id,
                type(exc).__name__,
            )
            await self._fail(lease, "worker_execution_failed")
            return
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

        if result.get("state") == "error":
            await self._fail(
                lease,
                result.get("error_code") or "worker_execution_failed",
                result,
            )
            return
        try:
            await self._client.acomplete(
                id=worker_task_id,
                complete=ModelPreheatWorkerTaskComplete(
                    **lease.request().model_dump(),
                    result=result,
                ),
            )
        except HTTPException as exc:
            if exc.status_code != 409:
                logger.error(
                    "模型预热任务完成回写失败。worker_task_id=%s error_type=%s",
                    worker_task_id,
                    type(exc).__name__,
                )
        except Exception as exc:
            logger.error(
                "模型预热任务完成回写失败。worker_task_id=%s error_type=%s",
                worker_task_id,
                type(exc).__name__,
            )

    async def _heartbeat_loop(self, context: ModelPreheatExecutionContext):
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            await context.heartbeat()

    async def _fail(self, lease, error_code, result=None):
        try:
            await self._client.afail(
                id=lease.worker_task_id,
                failure=ModelPreheatWorkerTaskFail(
                    **lease.request().model_dump(),
                    error_code=error_code,
                    state_message=error_code,
                    result=result or {},
                ),
            )
        except HTTPException as exc:
            if exc.status_code != 409:
                logger.error(
                    "模型预热任务失败回写失败。worker_task_id=%s error_type=%s",
                    lease.worker_task_id,
                    type(exc).__name__,
                )
        except Exception as exc:
            logger.error(
                "模型预热任务失败回写失败。worker_task_id=%s error_type=%s",
                lease.worker_task_id,
                type(exc).__name__,
            )

    async def _execute_supported_task(self, payload, context):
        if payload.role != ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK:
            return {
                "state": "error",
                "error_code": "worker_task_handler_unavailable",
            }
        profile = payload.profile.model_dump()
        check_id = payload.task["connectivity_check_id"]
        execution_pool = ProcessPoolExecutor(max_workers=1)
        future = asyncio.get_running_loop().run_in_executor(
            execution_pool,
            execute_profile_connectivity_check,
            profile,
            check_id,
            self._worker_uuid,
            Minio,
        )
        try:
            result = await future
        except asyncio.CancelledError:
            self._terminate_execution_pool(execution_pool)
            raise
        except Exception:
            execution_pool.shutdown(wait=False, cancel_futures=True)
            raise
        execution_pool.shutdown(wait=False, cancel_futures=True)
        return result

    @staticmethod
    def _terminate_execution_pool(execution_pool):
        processes = getattr(execution_pool, "_processes", {}) or {}
        for process in processes.values():
            if process.is_alive():
                process.terminate()
        execution_pool.shutdown(wait=False, cancel_futures=True)

    def _remove_active_task(self, task_id, completed):
        if self._active_tasks.get(task_id) is completed:
            self._active_tasks.pop(task_id, None)
