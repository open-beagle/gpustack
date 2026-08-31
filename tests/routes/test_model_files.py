import asyncio
import json
from types import SimpleNamespace

import anyio
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
    generate_model_preheat_credential_key,
)
from gpustack.routes import model_files as model_files_routes
from gpustack.routes import model_storage
from gpustack.routes.model_files import (
    _stream_model_files,
    create_model_file,
    delete_model_file,
    get_model_file,
    get_model_files,
    update_model_file,
)
from gpustack.schemas.common import ListParams
from gpustack.server.bus import Event, EventType, event_bus
from gpustack.schemas.links import ModelInstanceModelFileLink
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_storage_sync import (
    ModelFileTransferSourceEnum,
    ModelStorageSyncBatchCreate,
    ModelStorageSyncScopeEnum,
    ModelStorageSyncTask,
    ModelStorageSyncTaskCreate,
    ModelStorageSyncTaskStateEnum,
)
from gpustack.schemas.model_file_download_executions import (
    ModelFileDownloadExecution,
)
from gpustack.schemas.model_files import (
    ModelFile,
    ModelFileCreate,
    ModelFileStateEnum,
    ModelFileUpdate,
)
from gpustack.schemas.models import Model, ModelInstance, SourceEnum
from gpustack.schemas.users import User
from gpustack.schemas.workers import (
    MODEL_STORAGE_PROTOCOL_VERSION,
    Worker,
    WorkerStateEnum,
)
from gpustack.server.db import get_engine, get_session


def test_model_storage_sync_task_is_owned_by_model_file():
    foreign_key = next(
        foreign_key
        for foreign_key in ModelStorageSyncTask.__table__.foreign_keys
        if foreign_key.parent.name == "model_file_id"
    )

    assert foreign_key.target_fullname == "model_files.id"
    assert foreign_key.ondelete == "CASCADE"


@pytest.mark.parametrize(
    ("streamer", "event_model"),
    [
        (model_files_routes._stream_model_files, ModelFile),
        (model_storage._stream_sync_tasks, ModelStorageSyncTask),
    ],
)
def test_streaming_cancellation_closes_session_outside_cancel_scope(
    monkeypatch, streamer, event_model
):
    session_started = anyio.Event()
    session_closed = anyio.Event()

    class BlockingSession:
        def __init__(self, _engine):
            pass

        async def get(self, *_args):
            session_started.set()
            await anyio.sleep_forever()

        async def close(self):
            await anyio.sleep(0)
            session_closed.set()

    async def subscribe(_cls, _engine):
        yield Event(EventType.UPDATED, SimpleNamespace(id=1))

    async def consume():
        stream = streamer(object())
        try:
            await anext(stream)
        except StopAsyncIteration:
            pass

    monkeypatch.setattr(model_files_routes, "AsyncSession", BlockingSession)
    monkeypatch.setattr(model_storage, "AsyncSession", BlockingSession)
    monkeypatch.setattr(event_model, "subscribe", classmethod(subscribe))

    async def run():
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(consume)
            await session_started.wait()
            task_group.cancel_scope.cancel()
        assert session_closed.is_set()

    anyio.run(run)


def test_model_file_crud_cascades_sync_tasks(tmp_path):
    asyncio.run(_run_model_file_crud(tmp_path))


def test_model_file_list_filters_source_and_state(tmp_path):
    asyncio.run(_run_model_file_list_filters_source_and_state(tmp_path))


async def _run_model_file_list_filters_source_and_state(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'model-file-list.db'}",
        poolclass=NullPool,
    )
    async with engine.begin() as connection:
        await connection.run_sync(
            SQLModel.metadata.create_all,
            tables=[
                Worker.__table__,
                Model.__table__,
                ModelInstance.__table__,
                ModelFile.__table__,
                ModelInstanceModelFileLink.__table__,
                ModelFileDownloadExecution.__table__,
            ],
        )

    async with AsyncSession(engine, expire_on_commit=False) as session:
        worker = await Worker.create(
            session,
            Worker(
                name="worker-a",
                hostname="worker-a",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="worker-a-uuid",
                state=WorkerStateEnum.READY,
                model_storage_protocol_version=1,
            ),
        )
        ready_modelscope = await ModelFile.create(
            session,
            ModelFile(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="org/ready",
                worker_id=worker.id,
                state=ModelFileStateEnum.READY,
                resolved_paths=["/models/ready"],
            ),
        )
        await ModelFile.create(
            session,
            ModelFile(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="org/downloading",
                worker_id=worker.id,
                state=ModelFileStateEnum.DOWNLOADING,
            ),
        )

        page = await get_model_files(
            engine,
            session,
            ListParams(page=1, perPage=100),
            source=SourceEnum.MODEL_SCOPE,
            state=ModelFileStateEnum.READY,
        )

        assert [item.id for item in page.items] == [ready_modelscope.id]
        assert page.pagination.total == 1

    await engine.dispose()


@pytest.mark.parametrize(
    "state",
    [
        ModelStorageSyncTaskStateEnum.PENDING,
        ModelStorageSyncTaskStateEnum.SCANNING,
        ModelStorageSyncTaskStateEnum.PUBLISHING,
    ],
)
def test_delete_model_file_rejects_active_sync_task(tmp_path, state):
    asyncio.run(_run_delete_model_file_rejects_active_sync_task(tmp_path, state))


async def _run_delete_model_file_rejects_active_sync_task(tmp_path, state):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'model-file-active-task.db'}",
        poolclass=NullPool,
    )
    event.listen(
        engine.sync_engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    tables = [
        Worker.__table__,
        User.__table__,
        Model.__table__,
        ModelInstance.__table__,
        ModelFile.__table__,
        ModelInstanceModelFileLink.__table__,
        ModelPreheatS3Profile.__table__,
        ModelStorageSyncTask.__table__,
    ]
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all, tables=tables)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        worker = await Worker.create(
            session,
            Worker(
                name="worker-a",
                hostname="worker-a",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="worker-a-uuid",
                state=WorkerStateEnum.READY,
            ),
        )
        model_file = await ModelFile.create(
            session,
            ModelFile(
                source=SourceEnum.LOCAL_PATH,
                local_path="/models/original.gguf",
                worker_id=worker.id,
                state=ModelFileStateEnum.READY,
                resolved_paths=["/models/original.gguf"],
            ),
        )
        profile = await ModelPreheatS3Profile.create(
            session,
            ModelPreheatS3Profile(
                name="center-cache",
                endpoint="https://s3.example.com",
                bucket="models",
                access_key_encrypted={},
                secret_key_encrypted={},
                encryption_key_version="v1",
            ),
        )
        task = await ModelStorageSyncTask.create(
            session,
            ModelStorageSyncTask(
                model_file_id=model_file.id,
                worker_id=worker.id,
                worker_uuid=worker.worker_uuid,
                profile_id=profile.id,
                profile_config_version=1,
                request_identity={},
                request_digest="d" * 64,
                source="modelscope",
                model_id="example/model",
                resolved_revision="sha",
                credential_snapshot_encrypted={},
                encryption_key_version="v1",
                state=state,
            ),
        )

        try:
            await delete_model_file(session, model_file.id)
            assert False, "活动同步任务必须阻止删除"
        except Exception as exc:
            assert getattr(exc, "status_code", None) == 409
            assert getattr(exc, "message", None) == "model_file_has_active_sync_task"

        assert await ModelFile.one_by_id(session, model_file.id) is not None
        assert await ModelStorageSyncTask.one_by_id(session, task.id) is not None

    await engine.dispose()


def test_model_file_sse_update_enters_and_leaves_combined_filter(tmp_path):
    asyncio.run(_run_model_file_sse_update_enters_and_leaves_combined_filter(tmp_path))


async def _run_model_file_sse_update_enters_and_leaves_combined_filter(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'model-file-sse-filter.db'}",
        poolclass=NullPool,
    )
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        worker = await Worker.create(
            session,
            Worker(
                name="worker-a",
                hostname="worker-a",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="worker-a-uuid",
                state=WorkerStateEnum.READY,
            ),
        )
        await ModelFile.create(
            session,
            ModelFile(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="team/already-ready",
                worker_id=worker.id,
                state=ModelFileStateEnum.READY,
                resolved_paths=["/models/already-ready"],
            ),
        )
        candidate = await ModelFile.create(
            session,
            ModelFile(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="team/needle-model",
                worker_id=worker.id,
                state=ModelFileStateEnum.DOWNLOADING,
            ),
        )

        def search_filter(model_file):
            return "needle" in str(model_file.model_scope_model_id or "")

        stream = _stream_model_files(
            engine,
            fields={
                "worker_id": worker.id,
                "source": SourceEnum.MODEL_SCOPE,
                "state": ModelFileStateEnum.READY,
            },
            filter_func=search_filter,
        )
        next_event = asyncio.create_task(anext(stream))
        while not event_bus.subscribers.get("modelfile"):
            await asyncio.sleep(0)

        candidate.state = ModelFileStateEnum.READY
        candidate.resolved_paths = ["/models/needle-model"]
        await candidate.update(session)
        entered = json.loads(await asyncio.wait_for(next_event, timeout=2))
        assert entered["type"] == EventType.UPDATED.value
        assert entered["data"]["id"] == candidate.id

        next_event = asyncio.create_task(anext(stream))
        candidate.state = ModelFileStateEnum.DOWNLOADING
        await candidate.update(session)
        left = json.loads(await asyncio.wait_for(next_event, timeout=2))
        assert left["type"] == EventType.DELETED.value
        assert left["data"]["id"] == candidate.id
        await stream.aclose()

    await engine.dispose()


@pytest.mark.parametrize(
    "query",
    ["source=not-a-source", "state=not-a-state"],
)
def test_model_file_list_rejects_invalid_filter_enum_over_http(tmp_path, query):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'model-file-http-filter.db'}",
        poolclass=NullPool,
    )
    app = FastAPI()

    async def session_override():
        async with AsyncSession(engine) as session:
            yield session

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_engine] = lambda: engine
    app.include_router(model_files_routes.router, prefix="/model-files")
    with TestClient(app) as client:
        response = client.get(f"/model-files?{query}")
    assert response.status_code == 422
    asyncio.run(engine.dispose())


@pytest.mark.parametrize("create_kind", ["single", "batch"])
def test_sync_creation_commits_before_delete_is_rejected(
    tmp_path, monkeypatch, create_kind
):
    asyncio.run(
        _run_sync_creation_commits_before_delete_is_rejected(
            tmp_path, monkeypatch, create_kind
        )
    )


@pytest.mark.parametrize("create_kind", ["single", "batch"])
def test_delete_holds_lock_before_sync_creation_observes_missing_model_file(
    tmp_path, monkeypatch, create_kind
):
    asyncio.run(
        _run_delete_holds_lock_before_sync_creation_observes_missing_model_file(
            tmp_path, monkeypatch, create_kind
        )
    )


async def _create_sync_work(
    create_kind,
    request,
    session,
    user,
    profile_id,
    model_file_id,
):
    if create_kind == "single":
        return await model_storage.create_model_storage_sync_task(
            request,
            session,
            user,
            ModelStorageSyncTaskCreate(
                model_file_id=model_file_id,
                profile_id=profile_id,
            ),
            None,
        )
    return await model_storage.create_model_storage_sync_batch(
        request,
        session,
        user,
        ModelStorageSyncBatchCreate(
            profile_id=profile_id,
            scope=ModelStorageSyncScopeEnum.SINGLE_MODEL,
            model_file_id=model_file_id,
        ),
        None,
    )


async def _seed_sync_race_database(tmp_path, filename):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / filename}",
        poolclass=NullPool,
        connect_args={"timeout": 10},
    )
    event.listen(
        engine.sync_engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    key = generate_model_preheat_credential_key()
    cipher = ModelPreheatCredentialCipher(
        current_key=key,
        current_key_version="v1",
    )
    async with AsyncSession(engine, expire_on_commit=False) as session:
        user = User(
            username="admin",
            is_admin=True,
            hashed_password="unused",
        )
        session.add(user)
        worker = Worker(
            name="worker-a",
            hostname="worker-a",
            ip="127.0.0.1",
            port=10150,
            worker_uuid="worker-a-uuid",
            state=WorkerStateEnum.READY,
            model_storage_protocol_version=MODEL_STORAGE_PROTOCOL_VERSION,
        )
        session.add(worker)
        await session.flush()
        profile = ModelPreheatS3Profile(
            name="center-cache",
            endpoint="https://s3.example.com",
            bucket="models",
            access_key_encrypted=cipher.encrypt("AK"),
            secret_key_encrypted=cipher.encrypt("SK"),
            encryption_key_version="v1",
        )
        session.add(profile)
        model_file = ModelFile(
            source=SourceEnum.MODEL_SCOPE,
            model_scope_model_id="team/race-model",
            worker_id=worker.id,
            worker_uuid_snapshot=worker.worker_uuid,
            worker_name_snapshot=worker.name,
            state=ModelFileStateEnum.READY,
            resolved_paths=["/models/race-model"],
            requested_revision="main",
            resolved_revision="abcdef123456",
        )
        session.add(model_file)
        await session.commit()
        ids = user.id, profile.id, model_file.id

    app = SimpleNamespace(
        state=SimpleNamespace(
            server_config=SimpleNamespace(
                model_preheat_credential_key=key,
                model_preheat_credential_key_version="v1",
                model_preheat_credential_old_keys=None,
            )
        )
    )
    request = Request({"type": "http", "app": app})
    return engine, request, ids


async def _run_sync_creation_commits_before_delete_is_rejected(
    tmp_path, monkeypatch, create_kind
):
    engine, request, (user_id, profile_id, model_file_id) = (
        await _seed_sync_race_database(tmp_path, f"create-first-{create_kind}.db")
    )
    original_lock = model_files_routes.lock_model_file_for_sync_or_delete
    create_locked = asyncio.Event()
    allow_create = asyncio.Event()
    delete_attempted = asyncio.Event()

    async def create_lock(session, row_id):
        row = await original_lock(session, row_id)
        create_locked.set()
        await allow_create.wait()
        return row

    async def delete_lock(session, row_id):
        delete_attempted.set()
        return await original_lock(session, row_id)

    monkeypatch.setattr(
        model_storage, "lock_model_file_for_sync_or_delete", create_lock
    )
    monkeypatch.setattr(
        model_files_routes, "lock_model_file_for_sync_or_delete", delete_lock
    )

    async with (
        AsyncSession(engine, expire_on_commit=False) as create_session,
        AsyncSession(engine, expire_on_commit=False) as delete_session,
    ):
        user = await create_session.get(User, user_id)
        create_task = asyncio.create_task(
            _create_sync_work(
                create_kind,
                request,
                create_session,
                user,
                profile_id,
                model_file_id,
            )
        )
        await asyncio.wait_for(create_locked.wait(), timeout=2)
        delete_task = asyncio.create_task(
            delete_model_file(delete_session, model_file_id, cleanup=True)
        )
        await asyncio.wait_for(delete_attempted.wait(), timeout=2)
        allow_create.set()
        created = await asyncio.wait_for(create_task, timeout=5)
        delete_result = await asyncio.gather(delete_task, return_exceptions=True)

    if create_kind == "single":
        created_task_id = created.id
    else:
        assert len(created.created) == 1
        created_task_id = created.created[0].task_id
    error = delete_result[0]
    assert getattr(error, "status_code", None) == 409
    assert getattr(error, "message", None) == "model_file_has_active_sync_task"
    async with AsyncSession(engine) as verify_session:
        assert await verify_session.get(ModelFile, model_file_id) is not None
        assert (
            await verify_session.get(ModelStorageSyncTask, created_task_id) is not None
        )
    await engine.dispose()


async def _run_delete_holds_lock_before_sync_creation_observes_missing_model_file(
    tmp_path, monkeypatch, create_kind
):
    engine, request, (user_id, profile_id, model_file_id) = (
        await _seed_sync_race_database(tmp_path, f"delete-first-{create_kind}.db")
    )
    original_lock = model_files_routes.lock_model_file_for_sync_or_delete
    delete_locked = asyncio.Event()
    allow_delete = asyncio.Event()
    create_attempted = asyncio.Event()

    async def delete_lock(session, row_id):
        row = await original_lock(session, row_id)
        delete_locked.set()
        await allow_delete.wait()
        return row

    async def create_lock(session, row_id):
        create_attempted.set()
        return await original_lock(session, row_id)

    monkeypatch.setattr(
        model_files_routes, "lock_model_file_for_sync_or_delete", delete_lock
    )
    monkeypatch.setattr(
        model_storage, "lock_model_file_for_sync_or_delete", create_lock
    )

    async with (
        AsyncSession(engine, expire_on_commit=False) as delete_session,
        AsyncSession(engine, expire_on_commit=False) as create_session,
    ):
        user = await create_session.get(User, user_id)
        delete_task = asyncio.create_task(
            delete_model_file(delete_session, model_file_id, cleanup=True)
        )
        await asyncio.wait_for(delete_locked.wait(), timeout=2)
        create_task = asyncio.create_task(
            _create_sync_work(
                create_kind,
                request,
                create_session,
                user,
                profile_id,
                model_file_id,
            )
        )
        await asyncio.wait_for(create_attempted.wait(), timeout=2)
        allow_delete.set()
        await asyncio.wait_for(delete_task, timeout=5)
        create_result = (
            await asyncio.wait_for(
                asyncio.gather(create_task, return_exceptions=True), timeout=5
            )
        )[0]

    if create_kind == "single":
        assert getattr(create_result, "status_code", None) == 404
        assert getattr(create_result, "message", None) == "model_file_not_found"
    else:
        assert create_result.created == []
        assert len(create_result.failed) == 1
        assert create_result.failed[0].reason == "model_file_not_found"
    async with AsyncSession(engine) as verify_session:
        assert await verify_session.get(ModelFile, model_file_id) is None
        tasks = (
            await verify_session.exec(
                select(ModelStorageSyncTask).where(
                    ModelStorageSyncTask.model_file_id == model_file_id
                )
            )
        ).all()
        assert tasks == []
    await engine.dispose()


async def _run_model_file_crud(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'model-files.db'}",
        poolclass=NullPool,
    )
    event.listen(
        engine.sync_engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )

    tables = [
        Worker.__table__,
        User.__table__,
        Model.__table__,
        ModelInstance.__table__,
        ModelFile.__table__,
        ModelInstanceModelFileLink.__table__,
        ModelPreheatS3Profile.__table__,
        ModelFileDownloadExecution.__table__,
        ModelStorageSyncTask.__table__,
    ]
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all, tables=tables)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        worker = Worker(
            name="worker-a",
            hostname="worker-a",
            ip="127.0.0.1",
            port=10150,
            worker_uuid="worker-a-uuid",
            state=WorkerStateEnum.READY,
            model_storage_protocol_version=1,
        )
        worker = await Worker.create(session, worker)

        created = await create_model_file(
            session,
            ModelFileCreate(
                source=SourceEnum.LOCAL_PATH,
                local_path="/models/original.gguf",
                worker_id=worker.id,
            ),
        )
        assert created.worker_uuid_snapshot == "worker-a-uuid"
        assert created.worker_name_snapshot == "worker-a"
        fetched = await get_model_file(session, created.id)
        assert fetched.worker_name == "worker-a"
        assert fetched.worker_available is True

        worker.state = WorkerStateEnum.NOT_READY
        session.add(worker)
        await session.commit()
        assert (await get_model_file(session, created.id)).worker_available is False
        worker.state = WorkerStateEnum.READY
        worker.model_storage_protocol_version = 0
        session.add(worker)
        await session.commit()
        assert (await get_model_file(session, created.id)).worker_available is False
        worker.model_storage_protocol_version = 1
        session.add(worker)
        await session.commit()
        assert fetched.local_path == "/models/original.gguf"
        assert fetched.state == ModelFileStateEnum.DOWNLOADING

        update = ModelFileUpdate(**fetched.model_dump())
        update.state = ModelFileStateEnum.READY
        update.resolved_paths = ["/models/original.gguf"]
        updated = await update_model_file(session, fetched.id, update)
        assert updated.state == ModelFileStateEnum.READY
        assert updated.resolved_paths == ["/models/original.gguf"]

        profile = ModelPreheatS3Profile(
            name="center-cache",
            endpoint="https://s3.example.com",
            bucket="models",
            access_key_encrypted={},
            secret_key_encrypted={},
            encryption_key_version="v1",
        )
        profile = await ModelPreheatS3Profile.create(session, profile)
        download_execution = await ModelFileDownloadExecution.one_by_field(
            session, "model_file_id", created.id
        )
        download_execution.transfer_source = ModelFileTransferSourceEnum.S3
        download_execution.transfer_profile_id = profile.id
        download_execution.source_worker_id = worker.id
        session.add(download_execution)
        await session.commit()

        transferred = await get_model_file(session, created.id)
        assert transferred.source == SourceEnum.LOCAL_PATH
        assert transferred.transfer_source == ModelFileTransferSourceEnum.S3
        assert transferred.transfer_profile_id == profile.id
        assert transferred.transfer_profile_name == "center-cache"
        assert transferred.source_worker_id == worker.id
        assert transferred.source_worker_name == "worker-a"

        stream = _stream_model_files(engine)
        initial_event = json.loads(await anext(stream))
        assert initial_event["data"]["transfer_profile_name"] == "center-cache"
        assert initial_event["data"]["source_worker_name"] == "worker-a"

        next_event = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.05)
        model_file = await ModelFile.one_by_id(session, created.id)
        model_file.state_message = "ordinary-update"
        await model_file.update(session)
        updated_event = json.loads(await asyncio.wait_for(next_event, timeout=2))
        assert updated_event["data"]["state_message"] == "ordinary-update"
        assert updated_event["data"]["transfer_profile_name"] == "center-cache"
        assert updated_event["data"]["source_worker_name"] == "worker-a"

        next_event = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.05)
        incomplete_event = ModelFile(
            id=created.id,
            source=SourceEnum.LOCAL_PATH,
            local_path="/models/original.gguf",
            worker_id=worker.id,
            state=ModelFileStateEnum.READY,
        )
        await ModelFile._publish_event(EventType.UPDATED, incomplete_event)
        reloaded_event = json.loads(await asyncio.wait_for(next_event, timeout=2))
        assert reloaded_event["data"]["created_at"] is not None
        assert reloaded_event["data"]["updated_at"] is not None

        next_event = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.05)
        incomplete_created_event = ModelFile(
            id=created.id,
            source=SourceEnum.LOCAL_PATH,
            local_path="/models/original.gguf",
            worker_id=worker.id,
            state=ModelFileStateEnum.READY,
        )
        await ModelFile._publish_event(EventType.CREATED, incomplete_created_event)
        reloaded_created_event = json.loads(
            await asyncio.wait_for(next_event, timeout=2)
        )
        assert reloaded_created_event["data"]["created_at"] is not None
        assert reloaded_created_event["data"]["updated_at"] is not None

        for event_type in (EventType.UPDATED, EventType.CREATED):
            next_event = asyncio.create_task(anext(stream))
            await asyncio.sleep(0.05)
            expired_event = await session.get(ModelFile, created.id)
            session.expire(expired_event, ["created_at", "updated_at"])
            assert "created_at" not in expired_event.__dict__
            assert "updated_at" not in expired_event.__dict__
            await ModelFile._publish_event(event_type, expired_event)
            reloaded_expired_event = json.loads(
                await asyncio.wait_for(next_event, timeout=2)
            )
            assert reloaded_expired_event["type"] == event_type.value
            assert reloaded_expired_event["data"]["id"] == created.id
            assert reloaded_expired_event["data"]["created_at"] is not None
            assert reloaded_expired_event["data"]["updated_at"] is not None

        next_event = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.05)
        string_timestamp_event = SimpleNamespace(
            id=created.id,
            source=SourceEnum.LOCAL_PATH,
            local_path="/models/original.gguf",
            worker_id=worker.id,
            state=ModelFileStateEnum.READY,
            created_at="2026-08-31 01:30:00",
            updated_at="2026-08-31 01:30:00",
        )
        await ModelFile._publish_event(EventType.UPDATED, string_timestamp_event)
        reloaded_string_event = json.loads(
            await asyncio.wait_for(next_event, timeout=2)
        )
        assert reloaded_string_event["data"]["id"] == created.id
        assert reloaded_string_event["data"]["created_at"] is not None
        assert reloaded_string_event["data"]["updated_at"] is not None
        await stream.aclose()

        sync_task = ModelStorageSyncTask(
            model_file_id=created.id,
            worker_id=worker.id,
            worker_uuid=worker.worker_uuid,
            profile_id=profile.id,
            profile_config_version=1,
            request_identity={},
            request_digest="d" * 64,
            source="modelscope",
            model_id="example/model",
            resolved_revision="sha",
            credential_snapshot_encrypted={},
            encryption_key_version="v1",
            state=ModelStorageSyncTaskStateEnum.READY,
            transfer_source=ModelFileTransferSourceEnum.S3,
            transfer_profile_id=profile.id,
            source_worker_id=worker.id,
        )
        sync_task = await ModelStorageSyncTask.create(session, sync_task)

        await delete_model_file(session, created.id)

    async with AsyncSession(engine) as session:
        assert await ModelFile.one_by_id(session, created.id) is None
        assert (
            await ModelFileDownloadExecution.one_by_field(
                session, "model_file_id", created.id
            )
            is None
        )
        assert await ModelStorageSyncTask.one_by_id(session, sync_task.id) is None

    await engine.dispose()
