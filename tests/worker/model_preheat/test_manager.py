import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import FastAPI
import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.exceptions import HTTPException
from gpustack.client.generated_clientset import ClientSet
from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
    generate_model_preheat_credential_key,
)
from gpustack.routes import model_preheat_worker_tasks
from gpustack.schemas.model_preheats import (
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatPublicationMarker,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskComplete,
    ModelPreheatWorkerTaskProgress,
    ModelPreheatWorkerTaskPublic,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.bus import Event, EventType
from gpustack.server.db import get_engine, get_session
from gpustack.server.model_preheat_worker_identity import (
    issue_model_preheat_worker_credential,
)
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


class RestartProtocolClient:
    def __init__(self, generated_client, engine, worker_id):
        self._generated_client = generated_client
        self._engine = engine
        self._worker_id = worker_id
        self.claim_requests = []
        self.claims = []
        self.completed = asyncio.Event()

    def list(self, params):
        return asyncio.run(self._list_from_database(params))

    async def _list_from_database(self, params):
        states = [ModelPreheatWorkerTaskStateEnum(value) for value in params["state"]]
        async with AsyncSession(self._engine) as session:
            rows = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.worker_uuid == params["worker_uuid"],
                        ModelPreheatWorkerTask.worker_id == self._worker_id,
                        ModelPreheatWorkerTask.state.in_(states),
                    )
                )
            ).all()
            items = [ModelPreheatWorkerTaskPublic.model_validate(row) for row in rows]
        return SimpleNamespace(
            items=items,
            pagination=SimpleNamespace(totalPage=1),
        )

    async def awatch(self, callback, params):
        del callback, params
        await asyncio.Event().wait()

    async def aclaim(self, id, claim):
        claimed = await self._generated_client.aclaim(id=id, claim=claim)
        self.claim_requests.append(claim.model_dump(mode="json"))
        self.claims.append(claimed)
        return claimed

    async def aget_execution_payload(self, **kwargs):
        return await self._generated_client.aget_execution_payload(**kwargs)

    async def aheartbeat(self, id, lease):
        return await self._generated_client.aheartbeat(id=id, lease=lease)

    async def aprogress(self, id, progress):
        return await self._generated_client.aprogress(id=id, progress=progress)

    async def acomplete(self, id, complete):
        completed = await self._generated_client.acomplete(id=id, complete=complete)
        self.completed.set()
        return completed

    async def afail(self, id, failure):
        return await self._generated_client.afail(id=id, failure=failure)


async def _restart_protocol_fixture(tmp_path):
    key = generate_model_preheat_credential_key()
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'manager-restart.db'}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    app = FastAPI()
    app.state.server_config = SimpleNamespace(
        model_preheat_credential_key=key,
        model_preheat_credential_key_version="v1",
        model_preheat_credential_old_keys=None,
    )

    async def session_override():
        async with AsyncSession(engine) as session:
            yield session

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_engine] = lambda: engine
    app.include_router(
        model_preheat_worker_tasks.router,
        prefix="/v1/model-preheat-worker-tasks",
    )
    exceptions.register_handlers(app)

    cipher = ModelPreheatCredentialCipher(key, "v1")
    snapshot = cipher.encrypt(
        json.dumps(
            {
                "endpoint": "https://s3.example.com",
                "bucket": "models",
                "prefix": "cache",
                "tls_enabled": True,
                "tls_verify": True,
                "region": "cn-test-1",
                "use_virtual_hosted_style": False,
                "access_key_encrypted": cipher.encrypt("access-plain"),
                "secret_key_encrypted": cipher.encrypt("secret-plain"),
            }
        )
    )
    async with AsyncSession(engine) as session:
        worker = Worker(
            name="worker-a",
            hostname="worker-a",
            ip="127.0.0.1",
            port=10150,
            worker_uuid="worker-uuid",
            state=WorkerStateEnum.READY,
        )
        session.add(worker)
        await session.flush()
        task = ModelPreheatTask(
            source="modelscope",
            model_id="Qwen/Test",
            resolved_revision="commit-1",
            include_patterns=[],
            exclude_patterns=[],
            selection_digest="selection",
            cache_key="cache-key",
            generation_id="preheat-00000000-0000-4000-8000-000000000001",
            seed_worker_uuid=worker.worker_uuid,
            seed_worker_id=worker.id,
            target_scope=ModelPreheatTargetScopeEnum.SEED_WORKER,
            target_worker_uuids=[worker.worker_uuid],
            target_worker_snapshot=[],
            s3_profile_id=1,
            s3_profile_config_version=1,
            s3_profile_snapshot_encrypted=snapshot,
            encryption_key_version="v1",
            s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
        )
        session.add(task)
        await session.flush()
        worker_task = ModelPreheatWorkerTask(
            task_id=task.id,
            worker_uuid=worker.worker_uuid,
            worker_id=worker.id,
            role=ModelPreheatWorkerTaskRoleEnum.SEED,
        )
        session.add(worker_task)
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
        await session.flush()
        worker_id = worker.id
        worker_uuid = worker.worker_uuid
        worker_task_id = worker_task.id
        await session.commit()
        credential = await issue_model_preheat_worker_credential(
            session, worker_id, worker_uuid
        )
        return app, engine, worker_id, worker_task_id, credential


def _restart_ready_result():
    return {
        "state": "ready",
        "manifest_digest": "a" * 64,
        "ready_path": "model-cache/v1/source/model/revision/selection/ready.json",
        "manifest_path": "model-cache/v1/source/model/revision/selection/generations/g/.gpustack-manifest.json",
        "generation_id": "preheat-00000000-0000-4000-8000-000000000001",
        "local_cache_state": "valid",
        "uploaded": 1,
        "skipped": 0,
        "downloaded": 0,
        "total_size": 3,
    }


def test_worker_restart_reclaims_expired_upload_attempt(tmp_path):
    async def run():
        app, engine, worker_id, worker_task_id, credential = (
            await _restart_protocol_fixture(tmp_path)
        )
        transport = httpx.ASGITransport(app=app)
        async_client = httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"X-GPUStack-Worker-Credential": credential},
        )
        first_watch = None
        second_watch = None
        first_started = asyncio.Event()
        try:
            first_clientset = ClientSet("http://testserver")
            first_clientset.set_model_preheat_worker_credential(credential)
            first_clientset.http_client.set_async_httpx_client(async_client)
            first_client = RestartProtocolClient(
                first_clientset.model_preheat_worker_tasks, engine, worker_id
            )

            async def interrupted_upload(payload, context):
                assert payload.worker_task_id == worker_task_id
                assert payload.attempt == 1
                assert payload.profile.access_key == "access-plain"
                assert payload.profile.secret_key == "secret-plain"
                await context.progress(
                    40,
                    resumable_cursor={"staging_exists": True},
                    state_message="uploading",
                )
                first_started.set()
                await asyncio.Event().wait()

            first_manager = ModelPreheatManager(
                worker_id=worker_id,
                worker_uuid="worker-uuid",
                clientset=SimpleNamespace(model_preheat_worker_tasks=first_client),
                execution_handler=interrupted_upload,
                heartbeat_interval=60,
                reconcile_interval=0.01,
            )
            first_watch = asyncio.create_task(first_manager.watch_model_preheat_tasks())
            await asyncio.wait_for(first_started.wait(), timeout=5)
            first_watch.cancel()
            await asyncio.gather(first_watch, return_exceptions=True)
            first_watch = None

            first_claim = first_client.claims[0]
            async with AsyncSession(engine) as session:
                running = await session.get(ModelPreheatWorkerTask, worker_task_id)
                first_persisted = (
                    running.state,
                    running.attempt,
                    running.progress,
                    running.resumable_cursor,
                )
                running.lease_expires_at = datetime.now(timezone.utc) - timedelta(
                    seconds=1
                )
                session.add(running)
                await session.commit()

            second_clientset = ClientSet("http://testserver")
            second_clientset.set_model_preheat_worker_credential(credential)
            second_clientset.http_client.set_async_httpx_client(async_client)
            second_client = RestartProtocolClient(
                second_clientset.model_preheat_worker_tasks, engine, worker_id
            )

            async def resumed_upload(payload, context):
                del context
                assert payload.worker_task_id == worker_task_id
                assert payload.attempt == 2
                return _restart_ready_result()

            second_manager = ModelPreheatManager(
                worker_id=worker_id,
                worker_uuid="worker-uuid",
                clientset=SimpleNamespace(model_preheat_worker_tasks=second_client),
                execution_handler=resumed_upload,
                heartbeat_interval=60,
                reconcile_interval=0.01,
            )
            second_watch = asyncio.create_task(
                second_manager.watch_model_preheat_tasks()
            )
            await asyncio.wait_for(second_client.completed.wait(), timeout=5)
            second_watch.cancel()
            await asyncio.gather(second_watch, return_exceptions=True)
            second_watch = None

            second_claim = second_client.claims[0]
            stale_complete = ModelPreheatWorkerTaskComplete(
                worker_uuid="worker-uuid",
                worker_id=worker_id,
                attempt=first_claim.attempt,
                lease_token=first_claim.lease_token,
                result=_restart_ready_result(),
            )
            with pytest.raises(HTTPException) as stale_complete_error:
                await first_clientset.model_preheat_worker_tasks.acomplete(
                    id=worker_task_id,
                    complete=stale_complete,
                )
            stale_progress = ModelPreheatWorkerTaskProgress(
                worker_uuid="worker-uuid",
                worker_id=worker_id,
                attempt=first_claim.attempt,
                lease_token=first_claim.lease_token,
                progress=80,
                state_message="uploading",
            )
            with pytest.raises(HTTPException) as stale_progress_error:
                await first_clientset.model_preheat_worker_tasks.aprogress(
                    id=worker_task_id,
                    progress=stale_progress,
                )

            async with AsyncSession(engine) as session:
                ready = await session.get(ModelPreheatWorkerTask, worker_task_id)
                final_persisted = ready.state, ready.attempt, ready.progress

            return (
                worker_id,
                first_client.claim_requests,
                second_client.claim_requests,
                first_claim,
                second_claim,
                first_persisted,
                final_persisted,
                stale_complete_error.value,
                stale_progress_error.value,
            )
        finally:
            for watch in (first_watch, second_watch):
                if watch is not None:
                    watch.cancel()
            pending = [watch for watch in (first_watch, second_watch) if watch]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await async_client.aclose()
            await engine.dispose()

    (
        worker_id,
        first_claim_requests,
        second_claim_requests,
        first_claim,
        second_claim,
        first_persisted,
        final_persisted,
        stale_complete_error,
        stale_progress_error,
    ) = asyncio.run(run())
    expected_claim = {"worker_uuid": "worker-uuid", "worker_id": worker_id}
    assert first_claim_requests == [expected_claim]
    assert second_claim_requests == [expected_claim]
    assert first_claim.attempt == 1
    assert second_claim.attempt == 2
    assert second_claim.lease_token != first_claim.lease_token
    assert first_persisted == (
        ModelPreheatWorkerTaskStateEnum.RUNNING,
        1,
        40,
        {"staging_exists": True},
    )
    assert final_persisted == (ModelPreheatWorkerTaskStateEnum.READY, 2, 100)
    assert stale_complete_error.status_code == 409
    assert stale_complete_error.message == "stale_attempt"
    assert stale_progress_error.status_code == 409
    assert stale_progress_error.message == "stale_attempt"


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


def test_registered_seed_handler_is_claimed_and_completed():
    async def run():
        worker_tasks = FakeWorkerTasksClient()
        task = _public_task()

        async def execution_payload(**kwargs):
            return SimpleNamespace(
                worker_task_id=7,
                attempt=1,
                role=ModelPreheatWorkerTaskRoleEnum.SEED,
            )

        worker_tasks.aget_execution_payload = execution_payload

        async def seed_handler(payload, context):
            del payload, context
            return {
                "state": "ready",
                "manifest_digest": "a" * 64,
                "ready_path": "ready.json",
                "manifest_path": "manifest.json",
                "generation_id": "a" * 64,
                "local_cache_state": "valid",
                "uploaded": 0,
                "skipped": 2,
                "downloaded": 0,
                "total_size": 10,
            }

        async def awatch(callback, params):
            callback(Event(EventType.CREATED, task.model_dump(mode="json")))
            await asyncio.Event().wait()

        worker_tasks.awatch = awatch
        manager = ModelPreheatManager(
            worker_id=11,
            worker_uuid="worker-uuid",
            clientset=SimpleNamespace(model_preheat_worker_tasks=worker_tasks),
            role_handlers={ModelPreheatWorkerTaskRoleEnum.SEED: seed_handler},
            reconcile_interval=60,
        )
        watch = asyncio.create_task(manager.watch_model_preheat_tasks())
        await asyncio.sleep(0.05)
        watch.cancel()
        try:
            await watch
        except asyncio.CancelledError:
            pass
        return worker_tasks

    worker_tasks = asyncio.run(run())
    assert worker_tasks.claim_count == 1
    assert worker_tasks.complete_count == 1
