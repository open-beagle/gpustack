import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gpustack.api.exceptions import HTTPException
from gpustack.schemas.model_preheats import (
    ModelPreheatWorkerTaskPublic,
    ModelPreheatWorkerTaskRoleEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker.model_preheat.manager import ModelPreheatManager


def _public_task():
    now = datetime.now(timezone.utc)
    return ModelPreheatWorkerTaskPublic(
        id=7,
        task_id=3,
        worker_uuid="worker-uuid",
        worker_id=11,
        role="seed",
        state="pending",
        attempt=0,
        progress=0,
        downloaded_size=0,
        total_size=0,
        created_at=now,
        updated_at=now,
    )


class FakeWorkerTasksClient:
    def __init__(self):
        self.claim_count = 0
        self.complete_count = 0
        self.watch_count = 0
        self.list_params = []

    def list(self, params):
        self.list_params.append(params)
        return SimpleNamespace(
            items=[_public_task()],
            pagination=SimpleNamespace(totalPage=1),
        )

    async def awatch(self, callback, params):
        self.watch_count += 1
        event = Event(EventType.CREATED, _public_task().model_dump(mode="json"))
        callback(event)
        callback(event)
        await asyncio.sleep(0.02)
        raise RuntimeError("sse disconnected")

    async def aclaim(self, id, claim):
        self.claim_count += 1
        if self.claim_count > 1:
            raise HTTPException(409, "Conflict", "task_not_claimable")
        task = _public_task().model_dump()
        task.update(
            attempt=1,
            state="running",
            lease_token="lease-token",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=30),
        )
        return SimpleNamespace(**task)

    async def aget_execution_payload(self, **kwargs):
        return SimpleNamespace(worker_task_id=7, attempt=1)

    async def aheartbeat(self, task_id, lease):
        return None

    async def acomplete(self, id, complete):
        self.complete_count += 1

    async def afail(self, id, failure):
        raise AssertionError("executor should not fail")


def test_sse_reconnect_and_duplicate_events_start_only_one_execution():
    async def run():
        worker_tasks = FakeWorkerTasksClient()
        clientset = SimpleNamespace(model_preheat_worker_tasks=worker_tasks)
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def executor(payload, context):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return {"ok": True}

        manager = ModelPreheatManager(
            worker_id=11,
            worker_uuid="worker-uuid",
            clientset=clientset,
            execution_handler=executor,
            reconnect_delay=0.01,
            heartbeat_interval=60,
            reconcile_interval=0.01,
        )
        watch = asyncio.create_task(manager.watch_model_preheat_tasks())
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.sleep(0.08)
        release.set()
        await asyncio.sleep(0.03)
        watch.cancel()
        try:
            await watch
        except asyncio.CancelledError:
            pass
        await manager.shutdown()
        return worker_tasks, calls

    worker_tasks, calls = asyncio.run(run())
    assert worker_tasks.watch_count >= 2
    assert worker_tasks.claim_count >= 1
    assert worker_tasks.complete_count == 1
    assert calls == 1
    assert worker_tasks.list_params
    assert worker_tasks.list_params[0]["state"] == ["pending", "running"]


def test_periodic_reconciliation_claims_task_after_existing_lease_expires():
    class TakeoverClient(FakeWorkerTasksClient):
        def __init__(self):
            super().__init__()
            self.claimable = False

        async def awatch(self, callback, params):
            await asyncio.Event().wait()

        async def aclaim(self, id, claim):
            self.claim_count += 1
            if not self.claimable:
                raise HTTPException(409, "Conflict", "task_not_claimable")
            task = _public_task().model_dump()
            task.update(
                attempt=2,
                state="running",
                lease_token="new-token",
                lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=30),
            )
            return SimpleNamespace(**task)

    async def run():
        worker_tasks = TakeoverClient()
        executed = asyncio.Event()

        async def executor(payload, context):
            executed.set()
            return {"ok": True}

        manager = ModelPreheatManager(
            worker_id=11,
            worker_uuid="worker-uuid",
            clientset=SimpleNamespace(model_preheat_worker_tasks=worker_tasks),
            execution_handler=executor,
            reconcile_interval=0.01,
        )
        watch = asyncio.create_task(manager.watch_model_preheat_tasks())
        await asyncio.sleep(0.04)
        assert worker_tasks.claim_count >= 1
        worker_tasks.claimable = True
        await asyncio.wait_for(executed.wait(), timeout=1)
        watch.cancel()
        try:
            await watch
        except asyncio.CancelledError:
            pass
        return worker_tasks

    worker_tasks = asyncio.run(run())
    assert worker_tasks.claim_count >= 2
    assert worker_tasks.complete_count == 1


def test_default_manager_does_not_claim_unimplemented_file_roles():
    async def run():
        worker_tasks = FakeWorkerTasksClient()
        task = _public_task().model_copy(
            update={"role": ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE}
        )

        async def awatch(callback, params):
            callback(Event(EventType.CREATED, task.model_dump(mode="json")))
            await asyncio.Event().wait()

        worker_tasks.awatch = awatch
        worker_tasks.list = lambda params: SimpleNamespace(
            items=[task], pagination=SimpleNamespace(totalPage=1)
        )
        manager = ModelPreheatManager(
            worker_id=11,
            worker_uuid="worker-uuid",
            clientset=SimpleNamespace(model_preheat_worker_tasks=worker_tasks),
            reconcile_interval=0.01,
        )
        watch = asyncio.create_task(manager.watch_model_preheat_tasks())
        await asyncio.sleep(0.05)
        watch.cancel()
        try:
            await watch
        except asyncio.CancelledError:
            pass
        return worker_tasks.claim_count

    assert asyncio.run(run()) == 0
