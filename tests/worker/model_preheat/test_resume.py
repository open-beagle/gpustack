import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import HTTPException
from gpustack.routes.model_preheat_worker_tasks import (
    claim_model_preheat_worker_task,
    update_model_preheat_worker_task_progress,
)
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
)  # noqa: F401
from gpustack.schemas.model_preheats import (
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskClaim,
    ModelPreheatWorkerTaskProgress,
    ModelPreheatWorkerTaskPublic,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.workers import (
    MODEL_STORAGE_PROTOCOL_VERSION,
    Worker,
    WorkerStateEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.server.model_preheat_worker_identity import (
    ModelPreheatWorkerPrincipal,
)
from gpustack.worker.model_preheat.manager import ModelPreheatManager


def _public_task(state):
    now = datetime.now(timezone.utc)
    return ModelPreheatWorkerTaskPublic(
        id=7,
        task_id=3,
        worker_uuid="worker-uuid",
        worker_id=11,
        role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
        state=state,
        attempt=1,
        progress=20,
        downloaded_size=10,
        total_size=100,
        created_at=now,
        updated_at=now,
    )


class ResumeClient:
    def __init__(self):
        self.claim_count = 0
        self.complete_count = 0
        self.completed = asyncio.Event()
        self.pause_confirmed = asyncio.Event()
        self.progress_requests = []

    async def aclaim(self, id, claim):
        self.claim_count += 1
        claimed = _public_task(ModelPreheatWorkerTaskStateEnum.RUNNING).model_dump()
        claimed.update(
            lease_token="new-lease",
            lease_expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
            attempt=2,
        )
        return SimpleNamespace(**claimed)

    async def aget_execution_payload(self, **kwargs):
        return SimpleNamespace(worker_task_id=7, attempt=kwargs["attempt"])

    async def aheartbeat(self, id, lease):
        return None

    async def aprogress(self, id, progress):
        self.progress_requests.append(progress)
        if progress.state_message == "paused":
            self.pause_confirmed.set()
        return None

    async def acomplete(self, id, complete):
        self.complete_count += 1
        self.completed.set()

    async def afail(self, id, failure):
        raise AssertionError("恢复执行不应失败")


def test_paused_event_waits_and_duplicate_pending_event_claims_once():
    async def run():
        client = ResumeClient()
        executions = []

        async def execute(payload, context):
            del context
            executions.append(payload.attempt)
            return {"state": "ready"}

        manager = ModelPreheatManager(
            worker_id=11,
            worker_uuid="worker-uuid",
            clientset=SimpleNamespace(model_preheat_worker_tasks=client),
            execution_handler=execute,
            heartbeat_interval=60,
        )
        paused = _public_task(ModelPreheatWorkerTaskStateEnum.PAUSED)
        manager.handle_event(Event(EventType.UPDATED, paused.model_dump(mode="json")))
        await asyncio.sleep(0)
        claims_while_paused = client.claim_count

        pending = paused.model_copy(
            update={"state": ModelPreheatWorkerTaskStateEnum.PENDING}
        )
        event = Event(EventType.UPDATED, pending.model_dump(mode="json"))
        manager.handle_event(event)
        manager.handle_event(event)
        await asyncio.wait_for(client.completed.wait(), timeout=1)
        await manager.shutdown()
        return claims_while_paused, client, executions

    claims_while_paused, client, executions = asyncio.run(run())
    assert claims_while_paused == 0
    assert client.claim_count == 1
    assert client.complete_count == 1
    assert executions == [2]


def test_busy_worker_does_not_defer_preheat_claim():
    async def run():
        client = ResumeClient()

        async def execute(payload, context):
            del payload, context
            return {"state": "ready"}

        manager = ModelPreheatManager(
            worker_id=11,
            worker_uuid="worker-uuid",
            clientset=SimpleNamespace(model_preheat_worker_tasks=client),
            execution_handler=execute,
            idle_check=lambda: False,
            heartbeat_interval=60,
        )
        pending = _public_task(ModelPreheatWorkerTaskStateEnum.PENDING)
        manager.handle_event(Event(EventType.UPDATED, pending.model_dump(mode="json")))
        await asyncio.wait_for(client.completed.wait(), timeout=1)
        await manager.shutdown()
        return client.claim_count

    assert asyncio.run(run()) == 1


def test_running_preheat_continues_when_worker_becomes_busy():
    async def run():
        client = ResumeClient()
        idle = True
        started = asyncio.Event()
        finish = asyncio.Event()

        async def execute(payload, context):
            del payload, context
            started.set()
            await finish.wait()
            return {"state": "ready"}

        manager = ModelPreheatManager(
            worker_id=11,
            worker_uuid="worker-uuid",
            clientset=SimpleNamespace(model_preheat_worker_tasks=client),
            execution_handler=execute,
            idle_check=lambda: idle,
            heartbeat_interval=0.01,
        )
        pending = _public_task(ModelPreheatWorkerTaskStateEnum.PENDING)
        manager.handle_event(Event(EventType.UPDATED, pending.model_dump(mode="json")))
        await asyncio.wait_for(started.wait(), timeout=1)
        idle = False
        await asyncio.sleep(0.03)
        finish.set()
        await asyncio.wait_for(client.completed.wait(), timeout=1)
        await manager.shutdown()
        return client.complete_count

    assert asyncio.run(run()) == 1


def test_paused_event_immediately_stops_active_handler_and_confirms_with_lease():
    async def run():
        client = ResumeClient()
        started = asyncio.Event()
        canceled = asyncio.Event()

        async def execute(payload, context):
            del payload
            await context.progress(
                25,
                downloaded_size=2,
                total_size=8,
                resumable_cursor={
                    "completed_files": ["weights/model%207b.bin"],
                    "staging_exists": True,
                },
                state_message="downloading",
            )
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                canceled.set()

        manager = ModelPreheatManager(
            worker_id=11,
            worker_uuid="worker-uuid",
            clientset=SimpleNamespace(model_preheat_worker_tasks=client),
            execution_handler=execute,
            heartbeat_interval=60,
        )
        running = _public_task(ModelPreheatWorkerTaskStateEnum.RUNNING)
        manager.handle_event(Event(EventType.UPDATED, running.model_dump(mode="json")))
        await asyncio.wait_for(started.wait(), timeout=1)
        paused = running.model_copy(update={"state_message": "pause_requested"})
        manager.handle_event(Event(EventType.UPDATED, paused.model_dump(mode="json")))
        await asyncio.wait_for(client.pause_confirmed.wait(), timeout=1)
        await manager.shutdown()
        return canceled.is_set(), client.progress_requests[-1]

    canceled, confirmation = asyncio.run(run())
    assert canceled is True
    assert confirmation.attempt == 2
    assert confirmation.lease_token == "new-lease"
    assert confirmation.state_message == "paused"
    assert confirmation.resumable_cursor == {
        "completed_files": ["weights/model%207b.bin"],
        "staging_exists": True,
    }


def test_existing_paused_event_immediately_stops_handler_without_schedule_ack():
    async def run():
        client = ResumeClient()
        started = asyncio.Event()
        canceled = asyncio.Event()

        async def execute(payload, context):
            del payload, context
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                canceled.set()

        manager = ModelPreheatManager(
            worker_id=11,
            worker_uuid="worker-uuid",
            clientset=SimpleNamespace(model_preheat_worker_tasks=client),
            execution_handler=execute,
            heartbeat_interval=60,
        )
        running = _public_task(ModelPreheatWorkerTaskStateEnum.RUNNING)
        manager.handle_event(Event(EventType.UPDATED, running.model_dump(mode="json")))
        await asyncio.wait_for(started.wait(), timeout=1)
        paused = running.model_copy(
            update={"state": ModelPreheatWorkerTaskStateEnum.PAUSED}
        )
        manager.handle_event(Event(EventType.UPDATED, paused.model_dump(mode="json")))
        try:
            await asyncio.wait_for(canceled.wait(), timeout=0.2)
            stopped_before_shutdown = True
        except asyncio.TimeoutError:
            stopped_before_shutdown = False
        pause_confirmed_before_shutdown = client.pause_confirmed.is_set()
        progress_before_shutdown = list(client.progress_requests)
        await manager.shutdown()
        return (
            stopped_before_shutdown,
            pause_confirmed_before_shutdown,
            progress_before_shutdown,
        )

    assert asyncio.run(run()) == (True, False, [])


def test_pause_requested_while_loading_payload_confirms_with_claimed_lease():
    async def run():
        client = ResumeClient()
        payload_started = asyncio.Event()
        release_payload = asyncio.Event()

        async def loading_payload(**kwargs):
            payload_started.set()
            await release_payload.wait()
            return SimpleNamespace(worker_task_id=7, attempt=kwargs["attempt"])

        client.aget_execution_payload = loading_payload
        manager = ModelPreheatManager(
            worker_id=11,
            worker_uuid="worker-uuid",
            clientset=SimpleNamespace(model_preheat_worker_tasks=client),
            execution_handler=lambda payload, context: None,
            heartbeat_interval=60,
        )
        running = _public_task(ModelPreheatWorkerTaskStateEnum.RUNNING)
        manager.handle_event(Event(EventType.UPDATED, running.model_dump(mode="json")))
        await asyncio.wait_for(payload_started.wait(), timeout=1)
        pause_requested = running.model_copy(
            update={"state_message": "pause_requested"}
        )
        manager.handle_event(
            Event(EventType.UPDATED, pause_requested.model_dump(mode="json"))
        )
        await asyncio.wait_for(client.pause_confirmed.wait(), timeout=1)
        release_payload.set()
        await manager.shutdown()
        return client.progress_requests[-1]

    confirmation = asyncio.run(run())
    assert confirmation.attempt == 2
    assert confirmation.lease_token == "new-lease"
    assert confirmation.state_message == "paused"


def test_repeated_pause_requested_event_does_not_interrupt_boundary_ack():
    async def run():
        client = ResumeClient()
        started = asyncio.Event()
        boundary_reported = asyncio.Event()
        release_boundary = asyncio.Event()

        async def execute(payload, context):
            del payload
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.shield(
                    context.progress(
                        50,
                        resumable_cursor={
                            "completed_files": ["weights/model%207b.bin"],
                            "staging_exists": True,
                        },
                        state_message="downloading",
                    )
                )
                boundary_reported.set()
                await asyncio.shield(release_boundary.wait())
                raise

        manager = ModelPreheatManager(
            worker_id=11,
            worker_uuid="worker-uuid",
            clientset=SimpleNamespace(model_preheat_worker_tasks=client),
            execution_handler=execute,
            heartbeat_interval=60,
        )
        running = _public_task(ModelPreheatWorkerTaskStateEnum.RUNNING)
        manager.handle_event(Event(EventType.UPDATED, running.model_dump(mode="json")))
        await asyncio.wait_for(started.wait(), timeout=1)
        pause_requested = running.model_copy(
            update={"state_message": "pause_requested"}
        )
        event = Event(EventType.UPDATED, pause_requested.model_dump(mode="json"))
        manager.handle_event(event)
        await asyncio.wait_for(boundary_reported.wait(), timeout=1)
        manager.handle_event(event)
        release_boundary.set()
        await asyncio.wait_for(client.pause_confirmed.wait(), timeout=1)
        await manager.shutdown()
        return client.progress_requests[-1]

    confirmation = asyncio.run(run())
    assert confirmation.state_message == "paused"
    assert confirmation.resumable_cursor == {
        "completed_files": ["weights/model%207b.bin"],
        "staging_exists": True,
    }


async def _database(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'resume.db'}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return engine


async def _seed_resumed_worker_task(session):
    worker = Worker(
        name="worker",
        hostname="worker",
        ip="127.0.0.1",
        port=10150,
        worker_uuid="worker-uuid",
        state=WorkerStateEnum.READY,
        model_storage_protocol_version=MODEL_STORAGE_PROTOCOL_VERSION,
    )
    session.add(worker)
    await session.flush()
    request_identity = {
        "source": "huggingface",
        "model_id": "org/model",
        "requested_revision": None,
        "include_patterns": [],
        "exclude_patterns": [],
    }
    task = ModelPreheatTask(
        source="huggingface",
        model_id="org/model",
        resolved_revision="a" * 40,
        include_patterns=[],
        exclude_patterns=[],
        selection_digest="b" * 64,
        request_identity=request_identity,
        request_digest="c" * 64,
        cache_key="c" * 64,
        generation_id="preheat-resume",
        seed_worker_uuid=worker.worker_uuid,
        seed_worker_id=worker.id,
        target_scope=ModelPreheatTargetScopeEnum.SEED_WORKER,
        target_worker_uuids=[worker.worker_uuid],
        target_worker_snapshot=[],
        s3_profile_id=1,
        s3_profile_config_version=1,
        s3_profile_snapshot_encrypted={"ciphertext": "encrypted"},
        encryption_key_version="v1",
        s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
    )
    session.add(task)
    await session.flush()
    worker_task = ModelPreheatWorkerTask(
        task_id=task.id,
        parent_attempt=task.attempt,
        worker_uuid=worker.worker_uuid,
        worker_id=worker.id,
        role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
        state=ModelPreheatWorkerTaskStateEnum.PENDING,
        attempt=1,
        progress=20,
    )
    session.add(worker_task)
    worker_id = worker.id
    worker_task_id = worker_task.id
    await session.flush()
    worker_task_id = worker_task.id
    await session.commit()
    return worker_id, worker_task_id


def test_resumed_pending_claim_increments_attempt_and_rejects_old_lease(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        async with AsyncSession(engine) as session:
            worker_id, worker_task_id = await _seed_resumed_worker_task(session)
        identity = ModelPreheatWorkerPrincipal(
            worker_id=worker_id,
            worker_uuid="worker-uuid",
            credential_id=1,
            token_version=1,
        )
        async with AsyncSession(engine) as session:
            claimed = await claim_model_preheat_worker_task(
                session,
                worker_task_id,
                ModelPreheatWorkerTaskClaim(
                    worker_uuid="worker-uuid", worker_id=worker_id
                ),
                identity,
            )
        old_progress = ModelPreheatWorkerTaskProgress(
            worker_uuid="worker-uuid",
            worker_id=worker_id,
            attempt=1,
            lease_token="old-lease",
            progress=80,
            state_message="downloading",
        )
        async with AsyncSession(engine) as session:
            with pytest.raises(HTTPException) as stale_error:
                await update_model_preheat_worker_task_progress(
                    session, worker_task_id, old_progress, identity
                )
            persisted = await session.get(ModelPreheatWorkerTask, worker_task_id)
            result = claimed, stale_error.value, persisted.attempt, persisted.progress
        await engine.dispose()
        return result

    claimed, stale_error, attempt, progress = asyncio.run(run())
    assert claimed.attempt == 2
    assert claimed.lease_token != "old-lease"
    assert stale_error.status_code == 409
    assert stale_error.message == "stale_attempt"
    assert (attempt, progress) == (2, 20)
