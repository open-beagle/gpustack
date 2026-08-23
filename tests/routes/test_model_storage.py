"""任务 3：模型同步 API 定向测试。

覆盖：请求只接受 model_file_id + profile_id；READY 模型、来源与 Worker 归属
校验、Profile 版本；Idempotency-Key 重放；活动任务去重；客户端不能提交对象
Key；完成后写入统一 Artifact 库存（CAS 绑定）；Public schema 分别返回模型
source、本次 transfer_source、S3 Profile 与来源 Worker，不混用字段；凭据快照
不进入 Public schema / SSE / 日志。
"""

import asyncio
import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.auth import get_admin_user, get_current_user
from gpustack.api.exceptions import HTTPException
from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
    generate_model_preheat_credential_key,
)
from gpustack.client.generated_http_client import HTTPClient
from gpustack.client.generated_model_storage_sync_task_client import (
    ModelStorageSyncTaskClient,
)
from gpustack.routes import model_storage
from gpustack.schemas.model_files import (
    ModelFile,
    ModelFileStateEnum,
)
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatInventoryManifestStateEnum,
    ModelPreheatIdempotencyRecord,
)
from gpustack.schemas.model_storage_sync import (
    ModelFileTransferSourceEnum,
    ModelStorageSyncTask,
    ModelStorageSyncTaskStateEnum,
)
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_engine, get_session
from gpustack.server.model_preheat_worker_identity import (
    ModelPreheatWorkerPrincipal,
    get_model_preheat_worker_identity,
)
from gpustack.server.model_preheat_s3_profile_lifecycle import (
    ModelPreheatS3ProfileNotActive,
)
from gpustack.server.bus import Event, EventType

API = "/v1/model-storage-sync-tasks"
DETAIL = "/v1/model-storage-sync-tasks/{id}"
WORKER_TASKS_ROOT = "/v1/model-storage-worker-tasks"
WORKER_EXEC = "/v1/model-storage-worker-tasks/{id}/execution-payload"
WORKER_COMPLETE = "/v1/model-storage-worker-tasks/{id}/complete"
WORKER_FAIL = "/v1/model-storage-worker-tasks/{id}/fail"


def _fetch_task_lease_token(app, task_id):
    """从数据库解密任务的执行 lease token（测试专用）。"""
    from gpustack.schemas.model_storage_sync import ModelStorageSyncTask as TaskORM

    async def read():
        async with AsyncSession(_engine(app), expire_on_commit=False) as session:
            task = await session.get(TaskORM, task_id)
            cipher = _cipher_from_app(app)
            return cipher.decrypt(task.lease_token_encrypted)

    return _run(app, read())


def _complete_route_with_lease_injection(app):
    """为既有 complete 测试注入 lease token：包装真实 complete 路由。

    历史测试的 complete 请求体没有 lease_token（字段新增前）；这里在路由
    入口按任务解密出真实 lease 并注入到 complete 对象，其余逻辑（契约、
    CAS、库存、终态语义）全部走真实路由。需要验证 lease 语义的新测试直接
    调用真实路由（不带注入）。
    """
    original_complete = model_storage.complete_model_storage_sync_task

    async def wrapper(
        request,
        session,
        task_id,
        complete,
        identity,
    ):
        if not getattr(complete, "lease_token", None):
            task_row = await ModelStorageSyncTask.one_by_id(session, task_id)
            if task_row is not None and task_row.lease_token_encrypted is not None:
                cipher = _cipher_from_app(request.app)
                complete.lease_token = cipher.decrypt(task_row.lease_token_encrypted)
        return await original_complete(request, session, task_id, complete, identity)

    return model_storage.complete_model_storage_sync_task, wrapper


def _patch_complete_with_lease(monkeypatch, app):
    original, wrapper = _complete_route_with_lease_injection(app)
    monkeypatch.setattr(model_storage, "complete_model_storage_sync_task", wrapper)
    return original


@pytest.fixture
def app(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'storage.db'}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )

    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    asyncio.run(create_tables())

    test_app = FastAPI()
    test_app.state.server_config = SimpleNamespace(
        model_preheat_credential_key=generate_model_preheat_credential_key(),
        model_preheat_credential_key_version="v1",
        model_preheat_credential_old_keys=None,
        force_auth_localhost=True,
    )
    test_app.state.test_engine = engine

    async def session_override():
        async with AsyncSession(engine) as session:
            yield session

    async def admin_user_override():
        return User(id=1, username="admin", is_admin=True, hashed_password="")

    test_app.dependency_overrides[get_session] = session_override

    def engine_override():
        return engine

    test_app.dependency_overrides[get_engine] = engine_override
    test_app.dependency_overrides[get_admin_user] = admin_user_override
    test_app.dependency_overrides[get_current_user] = admin_user_override
    admin_router = APIRouter(dependencies=[Depends(get_admin_user)])
    admin_router.include_router(model_storage.router)
    test_app.include_router(admin_router, prefix="/v1")
    # Worker 侧端点（受 Worker 身份约束），挂载在 api 级（非管理员作用域）。
    test_app.include_router(
        model_storage.worker_router,
        prefix="/v1/model-storage-worker-tasks",
    )
    exceptions.register_handlers(test_app)

    yield test_app

    test_app.dependency_overrides.clear()
    asyncio.run(engine.dispose())


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def _engine(app):
    return app.state.test_engine


def _cipher_from_app(app):
    config = app.state.server_config
    return ModelPreheatCredentialCipher(
        current_key=config.model_preheat_credential_key,
        current_key_version=config.model_preheat_credential_key_version,
        old_keys=config.model_preheat_credential_old_keys,
    )


async def _seed_ids(app, **seed_kwargs):
    """创建 worker/profile/model_file 并返回 (profile_id, model_file_id)。

    id 在 commit 前（flush 后）捕获，避免 post-commit 惰性刷新导致
    MissingGreenlet。
    """
    async with AsyncSession(_engine(app), expire_on_commit=False) as session:
        cipher = _cipher_from_app(app)
        _worker, profile, model_file = await _seed(session, cipher, **seed_kwargs)
        profile_id, model_file_id = profile.id, model_file.id
        await session.commit()
    return profile_id, model_file_id


async def _seed(
    session,
    cipher,
    *,
    source=SourceEnum.MODEL_SCOPE,
    model_id="Qwen/Test",
    state=ModelFileStateEnum.READY,
    worker_state=WorkerStateEnum.READY,
    requested_revision="master",
    resolved_revision="8f73c6a91b",
):
    worker = Worker(
        name="worker-a",
        hostname="worker-a",
        ip="127.0.0.1",
        port=10150,
        worker_uuid="worker-a-uuid",
        state=worker_state,
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
        config_version=3,
    )
    session.add(profile)
    await session.flush()
    model_file = ModelFile(
        source=source,
        model_scope_model_id=model_id if source == SourceEnum.MODEL_SCOPE else None,
        huggingface_repo_id=model_id if source == SourceEnum.HUGGING_FACE else None,
        worker_id=worker.id,
        resolved_paths=(
            ["/models/Qwen/Test"] if state == ModelFileStateEnum.READY else []
        ),
        state=state,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
    )
    session.add(model_file)
    await session.flush()
    return worker, profile, model_file


def _run(app, coro):
    return asyncio.run(coro)


def test_capabilities_reports_credential_encryption(app, client):
    response = client.get("/v1/model-storage/capabilities")
    assert response.status_code == 200
    assert response.json() == {"credential_encryption_available": True}
    assert "model_preheat_credential_key" not in response.text


def test_capabilities_false_when_key_unavailable(app, client):
    app.state.server_config.model_preheat_credential_key = None
    response = client.get("/v1/model-storage/capabilities")
    assert response.status_code == 200
    assert response.json() == {"credential_encryption_available": False}


def test_create_sync_task_only_accepts_model_file_and_profile_id(app, client):
    profile_id, model_file_id = _run(app, _seed_ids(app))
    response = client.post(
        API,
        json={"model_file_id": model_file_id, "profile_id": profile_id},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"] == "modelscope"
    assert body["model_id"] == "Qwen/Test"
    assert body["profile_config_version"] == 3
    assert body["state"] == "pending"
    assert body["artifact_id"] is None
    assert "credential_snapshot_encrypted" not in body
    assert "encryption_key_version" not in body
    assert "secret_key" not in body
    assert "access_key" not in body


def test_batch_sync_selected_workers_plans_ready_models_and_replays(app, client):
    profile_id, first_model_file_id = _run(app, _seed_ids(app))

    async def seed_more():
        async with AsyncSession(_engine(app), expire_on_commit=False) as session:
            first_model_file = await session.get(ModelFile, first_model_file_id)
            first_worker_id = first_model_file.worker_id
            ready_worker = Worker(
                name="worker-b",
                hostname="worker-b",
                ip="127.0.0.2",
                port=10150,
                worker_uuid="worker-b-uuid",
                state=WorkerStateEnum.READY,
            )
            not_ready_worker = Worker(
                name="worker-c",
                hostname="worker-c",
                ip="127.0.0.3",
                port=10150,
                worker_uuid="worker-c-uuid",
                state=WorkerStateEnum.NOT_READY,
            )
            session.add_all([ready_worker, not_ready_worker])
            await session.flush()
            second = ModelFile(
                source=SourceEnum.HUGGING_FACE,
                huggingface_repo_id="org/second",
                worker_id=ready_worker.id,
                resolved_paths=["/models/org/second"],
                state=ModelFileStateEnum.READY,
                requested_revision="main",
                resolved_revision="b" * 40,
            )
            invalid = ModelFile(
                source=SourceEnum.LOCAL_PATH,
                local_path="/models/local-only",
                worker_id=ready_worker.id,
                resolved_paths=["/models/local-only"],
                state=ModelFileStateEnum.READY,
            )
            session.add_all([second, invalid])
            await session.commit()
            return first_worker_id, ready_worker.id, not_ready_worker.id, second.id

    (
        first_worker_id,
        ready_worker_id,
        not_ready_worker_id,
        second_model_file_id,
    ) = _run(app, seed_more())
    first = client.post(
        "/v1/model-storage-sync-batches",
        headers={"Idempotency-Key": "batch-selected-1"},
        json={
            "profile_id": profile_id,
            "scope": "selected_workers",
            "worker_ids": [first_worker_id, ready_worker_id, not_ready_worker_id],
        },
    )
    replay = client.post(
        "/v1/model-storage-sync-batches",
        headers={"Idempotency-Key": "batch-selected-1"},
        json={
            "profile_id": profile_id,
            "scope": "selected_workers",
            "worker_ids": [first_worker_id, ready_worker_id, not_ready_worker_id],
        },
    )
    reused_for_different_scope = client.post(
        "/v1/model-storage-sync-batches",
        headers={"Idempotency-Key": "batch-selected-1"},
        json={
            "profile_id": profile_id,
            "scope": "all_ready_workers",
        },
    )
    empty = client.post(
        "/v1/model-storage-sync-batches",
        headers={"Idempotency-Key": "batch-empty-1"},
        json={
            "profile_id": profile_id,
            "scope": "selected_workers",
            "worker_ids": [not_ready_worker_id],
        },
    )

    assert first.status_code == 200, first.text
    assert replay.status_code == 200, replay.text
    assert replay.json() == first.json()
    assert reused_for_different_scope.status_code == 409
    assert reused_for_different_scope.json()["message"] == "idempotency_key_reused"
    assert {item["model_file_id"] for item in first.json()["created"]} == {
        first_model_file_id,
        second_model_file_id,
    }
    assert [item["task_id"] for item in replay.json()["created"]] == [
        item["task_id"] for item in first.json()["created"]
    ]
    assert {item["worker_id"] for item in first.json()["skipped"]} == {
        not_ready_worker_id
    }
    assert [item["reason"] for item in first.json()["failed"]] == [
        "model_identity_invalid"
    ]
    assert empty.status_code == 200, empty.text
    assert empty.json()["planned"] == 0
    assert empty.json()["created"] == []
    assert empty.json()["failed"] == []
    assert empty.json()["skipped"][0]["reason"] == "worker_not_ready"
    empty_replay = client.post(
        "/v1/model-storage-sync-batches",
        headers={"Idempotency-Key": "batch-empty-1"},
        json={
            "profile_id": profile_id,
            "scope": "selected_workers",
            "worker_ids": [not_ready_worker_id],
        },
    )
    assert empty_replay.status_code == 200
    assert empty_replay.json() == empty.json()


def test_batch_sync_single_and_all_ready_scope_contract(app, client):
    profile_id, model_file_id = _run(app, _seed_ids(app))

    missing_single = client.post(
        "/v1/model-storage-sync-batches",
        json={"profile_id": profile_id, "scope": "single_model"},
    )
    missing_selected = client.post(
        "/v1/model-storage-sync-batches",
        json={"profile_id": profile_id, "scope": "selected_workers"},
    )
    single = client.post(
        "/v1/model-storage-sync-batches",
        headers={"Idempotency-Key": "batch-single-1"},
        json={
            "profile_id": profile_id,
            "scope": "single_model",
            "model_file_id": model_file_id,
        },
    )
    all_ready = client.post(
        "/v1/model-storage-sync-batches",
        headers={"Idempotency-Key": "batch-all-1"},
        json={"profile_id": profile_id, "scope": "all_ready_workers"},
    )

    assert missing_single.status_code == 422
    assert missing_selected.status_code == 422
    assert single.status_code == 200, single.text
    assert all_ready.status_code == 200, all_ready.text
    assert single.json()["created"][0]["model_file_id"] == model_file_id
    assert (
        all_ready.json()["created"][0]["task_id"]
        == single.json()["created"][0]["task_id"]
    )


def test_batch_all_identity_failures_replay_original_result_after_model_change(
    app, client
):
    profile_id, model_file_id = _run(app, _seed_ids(app, source=SourceEnum.LOCAL_PATH))
    request_body = {
        "profile_id": profile_id,
        "scope": "single_model",
        "model_file_id": model_file_id,
    }
    headers = {"Idempotency-Key": "batch-all-identity-failed"}

    first = client.post(
        "/v1/model-storage-sync-batches", headers=headers, json=request_body
    )
    assert first.status_code == 200, first.text
    assert first.json()["created"] == []
    assert first.json()["failed"][0]["reason"] == "model_identity_invalid"

    async def make_identity_valid():
        async with AsyncSession(_engine(app), expire_on_commit=False) as session:
            model_file = await session.get(ModelFile, model_file_id)
            model_file.source = SourceEnum.MODEL_SCOPE
            model_file.local_path = None
            model_file.model_scope_model_id = "Qwen/Test"
            model_file.requested_revision = "main"
            model_file.resolved_revision = "a" * 40
            session.add(model_file)
            await session.commit()

    _run(app, make_identity_valid())
    replay = client.post(
        "/v1/model-storage-sync-batches", headers=headers, json=request_body
    )

    assert replay.status_code == 200, replay.text
    assert replay.json() == first.json()


def test_concurrent_batch_replay_waits_for_child_idempotency_record(app, monkeypatch):
    """并发重放不得在批次标记已提交、子记录未写入时返回空结果。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    import gpustack.routes.model_storage as model_storage_route

    original_create = model_storage_route.create_model_storage_sync_task
    owner_reached_child_creation = threading.Barrier(2)
    release_owner = threading.Event()
    first_call_lock = threading.Lock()
    first_call = True

    async def blocked_first_child(*args, **kwargs):
        nonlocal first_call
        with first_call_lock:
            should_block = first_call
            first_call = False
        if should_block:
            owner_reached_child_creation.wait(timeout=5)
            assert release_owner.wait(timeout=5)
        return await original_create(*args, **kwargs)

    monkeypatch.setattr(
        model_storage_route,
        "create_model_storage_sync_task",
        blocked_first_child,
    )
    key = "batch-concurrent-replay"
    responses = []

    def post():
        with TestClient(app) as concurrent_client:
            responses.append(
                concurrent_client.post(
                    "/v1/model-storage-sync-batches",
                    headers={"Idempotency-Key": key},
                    json={
                        "profile_id": profile_id,
                        "scope": "single_model",
                        "model_file_id": model_file_id,
                    },
                )
            )

    owner = threading.Thread(target=post)
    owner.start()
    owner_reached_child_creation.wait(timeout=5)
    replay = threading.Thread(target=post)
    replay.start()
    release_owner.set()
    owner.join(timeout=5)
    replay.join(timeout=5)

    assert not owner.is_alive()
    assert not replay.is_alive()
    assert len(responses) == 2
    assert all(response.status_code == 200 for response in responses)
    task_ids = [response.json()["created"][0]["task_id"] for response in responses]
    assert len(set(task_ids)) == 1
    assert all(response.json()["created"] for response in responses)
    assert responses[0].json() == responses[1].json()


def test_create_rejects_non_ready_model_file(app, client):
    profile_id, model_file_id = _run(
        app, _seed_ids(app, state=ModelFileStateEnum.DOWNLOADING)
    )
    response = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    assert response.status_code == 409


def test_create_rejects_unsupported_source(app, client):
    profile_id, model_file_id = _run(app, _seed_ids(app, source=SourceEnum.LOCAL_PATH))
    response = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    assert response.status_code == 409


def test_create_rejects_client_supplied_object_key(app, client):
    """客户端不能提交任意对象 Key：多余字段被拒绝（422）。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    response = client.post(
        API,
        json={
            "model_file_id": model_file_id,
            "profile_id": profile_id,
            "target_path": "evil/object/key",
            "object_key": "evil/object/key",
        },
    )
    assert response.status_code == 422


def test_create_missing_model_file_is_404(app, client):
    profile_id, _ = _run(app, _seed_ids(app))
    response = client.post(API, json={"model_file_id": 9999, "profile_id": profile_id})
    assert response.status_code == 404


def test_create_missing_profile_is_404(app, client):
    _, model_file_id = _run(app, _seed_ids(app))
    response = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": 9999}
    )
    assert response.status_code == 404


def test_active_task_dedup_returns_existing(app, client):
    profile_id, model_file_id = _run(app, _seed_ids(app))
    first = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    assert first.status_code == 200
    second = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]


def test_idempotency_key_replay_returns_same_task(app, client):
    profile_id, model_file_id = _run(app, _seed_ids(app))
    key = "sync-key-1"
    first = client.post(
        API,
        json={"model_file_id": model_file_id, "profile_id": profile_id},
        headers={"Idempotency-Key": key},
    )
    assert first.status_code == 200
    second = client.post(
        API,
        json={"model_file_id": model_file_id, "profile_id": profile_id},
        headers={"Idempotency-Key": key},
    )
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]


async def _set_task_terminal(app, task_id, **fields):
    async with AsyncSession(_engine(app), expire_on_commit=False) as session:
        task = await session.get(ModelStorageSyncTask, task_id)
        for key, value in fields.items():
            setattr(task, key, value)
        session.add(task)
        await session.commit()


async def _change_profile_target(app, profile_id):
    async with AsyncSession(_engine(app), expire_on_commit=False) as session:
        profile = await session.get(ModelPreheatS3Profile, profile_id)
        profile.endpoint = "https://new-s3.example.com"
        profile.bucket = "new-models"
        profile.prefix = "new-prefix"
        profile.config_version += 1
        session.add(profile)
        await session.commit()


def _request(app):
    return Request({"type": "http", "app": app})


def test_detail_separates_source_transfer_profile_and_worker(app, client):
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    task_id = created.json()["id"]
    # 模拟 Worker 完成后回写 transfer 字段与来源 Worker。
    _run(
        app,
        _set_task_terminal(
            app,
            task_id,
            state=ModelStorageSyncTaskStateEnum.READY,
            transfer_source=ModelFileTransferSourceEnum.S3,
            transfer_profile_id=profile_id,
            artifact_id="a" * 64,
            **{"source_worker_id": created.json()["worker_id"]},
        ),
    )
    # 当前 Profile 后续修改不得改写任务历史目标。
    _run(app, _change_profile_target(app, profile_id))
    response = client.get(DETAIL.format(id=task_id))
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "modelscope"
    assert body["transfer_source"] == "s3"
    assert body["profile"]["id"] == profile_id
    assert body["profile"]["name"] == "center-cache"
    assert body["profile"]["endpoint"] == "https://s3.example.com"
    assert body["profile"]["bucket"] == "models"
    assert body["profile"]["prefix"] == ""
    assert body["profile"]["config_version"] == 3
    assert body["source_worker_id"] == body["worker_id"]
    assert body["source_worker_name"] == "worker-a"
    assert body["artifact_id"] == "a" * 64
    assert "access_key" not in body["profile"]
    assert "secret_key" not in body["profile"]
    assert "credential_snapshot_encrypted" not in body

    listing = client.get(API).json()["items"][0]
    assert listing["source_worker_name"] == "worker-a"
    assert listing["profile_name"] == "center-cache"
    assert listing["profile_endpoint"] == "https://s3.example.com"
    assert listing["profile_bucket"] == "models"
    assert listing["profile_prefix"] == ""
    assert listing["started_at"] is None
    assert listing["finished_at"] is None
    assert "access_key" not in listing
    assert "secret_key" not in listing

    async def initial_stream_event():
        stream = model_storage._stream_sync_tasks(
            _engine(app), cipher=_cipher_from_app(app)
        )
        try:
            return json.loads(await anext(stream))
        finally:
            await stream.aclose()

    stream_data = _run(app, initial_stream_event())["data"]
    assert stream_data["source_worker_name"] == "worker-a"
    assert stream_data["profile_name"] == "center-cache"
    assert stream_data["profile_endpoint"] == "https://s3.example.com"
    assert stream_data["profile_bucket"] == "models"
    assert stream_data["profile_prefix"] == ""
    assert "access_key" not in stream_data
    assert "secret_key" not in stream_data
    assert "credential_snapshot_encrypted" not in stream_data


def test_sync_history_decryption_unavailable_degrades_without_credentials(app, client):
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    task_id = created.json()["id"]

    original_key = app.state.server_config.model_preheat_credential_key
    app.state.server_config.model_preheat_credential_key = None
    try:
        detail_response = client.get(DETAIL.format(id=task_id))
        list_response = client.get(API)
        assert detail_response.status_code == 200
        assert list_response.status_code == 200

        detail = detail_response.json()
        listing = list_response.json()["items"][0]
        assert detail["profile"]["id"] == profile_id
        assert detail["profile"]["name"] == "center-cache"
        assert detail["profile"]["config_version"] == 3
        for field in ("endpoint", "bucket", "prefix"):
            assert detail["profile"][field] is None
            assert listing[f"profile_{field}"] is None

        async def initial_stream_event():
            stream = model_storage._stream_sync_tasks(
                _engine(app), cipher=_cipher_from_app(app)
            )
            try:
                return json.loads(await anext(stream))
            finally:
                await stream.aclose()

        stream_data = _run(app, initial_stream_event())["data"]
        assert stream_data["profile_name"] == "center-cache"
        for field in ("endpoint", "bucket", "prefix"):
            assert stream_data[f"profile_{field}"] is None

        serialized = json.dumps(
            {"detail": detail, "listing": listing, "stream": stream_data}
        )
        for forbidden in (
            "AK",
            "SK",
            "access_key",
            "secret_key",
            "credential_snapshot_encrypted",
            "lease_token_encrypted",
        ):
            assert forbidden not in serialized
    finally:
        app.state.server_config.model_preheat_credential_key = original_key


def test_sync_stream_reloads_event_missing_timestamps_by_id(app, client, monkeypatch):
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    ).json()

    async def subscribe(cls, engine):
        del cls, engine
        yield Event(
            type=EventType.UPDATED,
            data=ModelStorageSyncTask(id=created["id"]),
        )

    monkeypatch.setattr(
        ModelStorageSyncTask,
        "subscribe",
        classmethod(subscribe),
    )

    async def read_event():
        stream = model_storage._stream_sync_tasks(
            _engine(app), cipher=_cipher_from_app(app)
        )
        try:
            return json.loads(await anext(stream))["data"]
        finally:
            await stream.aclose()

    event = _run(app, read_event())
    assert event["id"] == created["id"]
    assert event["created_at"] is not None
    assert event["updated_at"] is not None


def test_cancel_active_task_marks_canceled(app, client):
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    task_id = created.json()["id"]
    response = client.delete(DETAIL.format(id=task_id))
    assert response.status_code == 200
    detail = client.get(DETAIL.format(id=task_id))
    assert detail.json()["state"] == "canceled"


def test_artifacts_list_matches_profile_and_config_version(app, client):
    profile_id, _ = _run(app, _seed_ids(app))

    async def seed_artifacts():
        async with AsyncSession(_engine(app), expire_on_commit=False) as session:
            now = datetime.now(timezone.utc)
            for version, artifact_id in [
                (3, "b" * 64),  # 当前 config version
                (2, "d" * 64),  # 旧版本，不应命中
            ]:
                session.add(
                    ModelPreheatArtifact(
                        profile_id=profile_id,
                        profile_config_version=version,
                        artifact_id=artifact_id,
                        source="modelscope",
                        model_id="Qwen/Test",
                        resolved_revision="8f73c6a91b",
                        include_patterns=[],
                        exclude_patterns=[],
                        manifest_path=f"models/modelscope/Qwen/Test/{artifact_id[:4]}/manifest.json",
                        manifest_digest="c" * 64,
                        file_count=1,
                        total_size=10,
                        manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
                        last_verified_at=now,
                    )
                )
            await session.commit()

    _run(app, seed_artifacts())
    response = client.get(f"/v1/model-storage-profiles/{profile_id}/artifacts")
    assert response.status_code == 200
    items = response.json()
    assert [item["artifact_id"] for item in items] == ["b" * 64]


def test_artifacts_list_404_for_missing_profile(app, client):
    response = client.get("/v1/model-storage-profiles/9999/artifacts")
    assert response.status_code == 404


def test_refresh_artifacts_returns_stable_error_when_service_unavailable(app, client):
    profile_id, _ = _run(app, _seed_ids(app))
    response = client.post(f"/v1/model-storage-profiles/{profile_id}/artifacts/refresh")
    assert response.status_code == 503


def _worker_principal(worker_id, worker_uuid):
    return ModelPreheatWorkerPrincipal(
        worker_id=worker_id,
        worker_uuid=worker_uuid,
        credential_id=1,
        token_version=1,
    )


async def _seed_inventory_artifact(
    app,
    *,
    profile_id,
    profile_config_version=3,
    artifact_id="b" * 64,
    source="modelscope",
    model_id="Qwen/Test",
    resolved_revision="8f73c6a91b",
    include_patterns=None,
    exclude_patterns=None,
    state=ModelPreheatInventoryManifestStateEnum.VALID,
):
    """写入一条统一 Artifact 库存（用于验证预绑定精确匹配）。"""
    async with AsyncSession(_engine(app), expire_on_commit=False) as session:
        session.add(
            ModelPreheatArtifact(
                profile_id=profile_id,
                profile_config_version=profile_config_version,
                artifact_id=artifact_id,
                source=source,
                model_id=model_id,
                resolved_revision=resolved_revision,
                include_patterns=(
                    include_patterns if include_patterns is not None else []
                ),
                exclude_patterns=(
                    exclude_patterns if exclude_patterns is not None else []
                ),
                manifest_path=f"models/{source}/{model_id}/{artifact_id[:4]}/manifest.json",
                manifest_digest="c" * 64,
                file_count=1,
                total_size=10,
                manifest_state=state,
                last_verified_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()


def _create_task(app, profile_id, model_file_id):
    with TestClient(app) as client:
        response = client.post(
            API, json={"model_file_id": model_file_id, "profile_id": profile_id}
        )
    assert response.status_code == 200, response.text
    return response.json()


def test_create_sync_task_rejects_maintenance_profile(app):
    profile_id, model_file_id = _run(app, _seed_ids(app))

    async def maintain():
        async with AsyncSession(app.state.test_engine) as session:
            profile = await session.get(ModelPreheatS3Profile, profile_id)
            profile.lifecycle_state = (
                ModelPreheatS3ProfileLifecycleStateEnum.MAINTENANCE
            )
            session.add(profile)
            await session.commit()

    _run(app, maintain())
    with TestClient(app) as client:
        response = client.post(
            API, json={"model_file_id": model_file_id, "profile_id": profile_id}
        )

    assert response.status_code == 409
    assert response.json()["message"] == "s3_profile_in_maintenance"


def test_create_sync_task_rejects_profile_maintained_before_final_lock(
    app, monkeypatch
):
    profile_id, model_file_id = _run(app, _seed_ids(app))

    async def reject_stale_active_profile(*args, **kwargs):
        del args, kwargs
        raise ModelPreheatS3ProfileNotActive

    monkeypatch.setattr(
        model_storage,
        "lock_active_profile_for_new_work",
        reject_stale_active_profile,
    )
    with TestClient(app) as client:
        response = client.post(
            API, json={"model_file_id": model_file_id, "profile_id": profile_id}
        )

    async def task_count():
        async with AsyncSession(_engine(app)) as session:
            return len((await session.exec(select(ModelStorageSyncTask))).all())

    assert response.status_code == 409
    assert response.json()["message"] == "s3_profile_in_maintenance"
    assert _run(app, task_count()) == 0


def test_worker_complete_cas_binds_artifact_from_null(app):
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            # 完成后 CAS 绑定 artifact_id（仅从 NULL）。
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 200
            detail = client.get(DETAIL.format(id=task_id))
            assert detail.json()["artifact_id"] == "e" * 64
            assert detail.json()["state"] == "ready"
            assert detail.json()["transfer_source"] == "s3"
            # 不同 artifact 的重复完成：稳定冲突（不再一律 200），不覆盖。
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "f" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 3,
                    "total_size": 20,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 409
            detail = client.get(DETAIL.format(id=task_id))
            assert detail.json()["artifact_id"] == "e" * 64
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_worker_complete_rejected_after_cancel(app):
    """任务被取消后，Worker 的 complete 不得把状态改回 ready（稳定 409）。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]
    with TestClient(app) as client:
        cancel = client.delete(DETAIL.format(id=task_id))
        assert cancel.status_code == 200
        assert client.get(DETAIL.format(id=task_id)).json()["state"] == "canceled"

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            # CAS 失败不再一律 200：已取消（终态）稳定冲突。
            assert response.status_code == 409
            detail = client.get(DETAIL.format(id=task_id))
            assert detail.json()["state"] == "canceled"
            assert detail.json()["artifact_id"] is None
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_worker_execution_payload_403_for_other_worker(app):
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    task_id = created["id"]

    async def override():
        # 其他 Worker（worker_uuid 不匹配）不得读取执行 payload。
        return _worker_principal(999, "other-worker-uuid")

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.get(WORKER_EXEC.format(id=task_id))
            assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_worker_execution_payload_returns_credentials_only_for_authorized(app):
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.get(WORKER_EXEC.format(id=task_id))
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["state"] == "publishing"
            # 执行 payload 含明文 S3 凭据与可信本地源路径。
            assert body["profile"]["access_key"] == "AK"
            assert body["source_paths"] == ["/models/Qwen/Test"]
            assert body["request_identity"] == {
                "source": "modelscope",
                "model_id": "Qwen/Test",
                "requested_revision": "master",
                "include_patterns": [],
                "exclude_patterns": [],
            }
            assert "/models/" not in json.dumps(body["request_identity"])
            assert response.headers.get("cache-control") == "no-store"
            detail = client.get(DETAIL.format(id=task_id)).json()
            assert detail["state"] == "publishing"
            assert detail["started_at"] is not None
            first_started_at = detail["started_at"]

            replay = client.get(WORKER_EXEC.format(id=task_id))
            assert replay.status_code == 200
            assert replay.json()["state"] == "publishing"
            assert (
                client.get(DETAIL.format(id=task_id)).json()["started_at"]
                == first_started_at
            )

        async def used_at():
            async with AsyncSession(app.state.test_engine) as session:
                profile = await session.get(ModelPreheatS3Profile, profile_id)
                return profile.ever_used_at

        assert _run(app, used_at()) is not None
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_worker_execution_payload_rejects_terminal_task(app):
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    _run(
        app,
        _set_task_terminal(
            app,
            created["id"],
            state=ModelStorageSyncTaskStateEnum.CANCELED,
            finished_at=datetime.now(timezone.utc),
        ),
    )

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.get(WORKER_EXEC.format(id=created["id"]))
            assert response.status_code == 409
            assert response.json()["message"] == "sync_task_already_terminal"
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


# ---------------------------------------------------------------------------
# Review 子阶段 A：Worker 私有任务根 list/watch 端点 + 创建时 Worker 可用性
# ---------------------------------------------------------------------------


def test_worker_tasks_root_route_registered_for_generated_client():
    """真实 client URL 与真实路由收集必须一一对应：Worker 不再 404 pending。

    生成 client 以 ``GET {base}/v1/model-storage-worker-tasks?watch=true``
    监听任务。这里把真实的 ``model_storage.worker_router`` 按
    ``routes.py`` 使用的同一 prefix 挂到 FastAPI，收集真实挂载后的完整
    路径，并断言 client 的全部端点 URL（含根 list/watch 端点）都能命中。
    """
    collect_app = FastAPI()
    collect_app.include_router(
        model_storage.worker_router,
        prefix="/v1/model-storage-worker-tasks",
    )
    registered = {
        (route.path, method)
        for route in collect_app.routes
        if hasattr(route, "methods")
        for method in route.methods or set()
    }
    # 根 list/watch 端点必须注册（此前缺失导致真实 Worker 404 pending）。
    assert ("/v1/model-storage-worker-tasks", "GET") in registered

    # 生成 client 的全部端点 URL 都必须存在对应路由。
    http_client = object.__new__(HTTPClient)
    http_client._base_url = "http://127.0.0.1:80"
    client = ModelStorageSyncTaskClient(http_client)
    endpoints = [
        (client._url, "GET"),  # 根 list/watch（watch=true 参数在请求上）
        (f"{client._url}/1/execution-payload", "GET"),
        (f"{client._url}/1/complete", "POST"),
        (f"{client._url}/1/fail", "POST"),
    ]
    import re

    def _match(path, method):
        # 路由以模板（如 /{task_id}）注册：按路径段与花括号占位匹配。
        for route_path, route_method in registered:
            if route_method != method:
                continue
            pattern = re.escape(route_path)
            pattern = re.sub(r"\\{[^}]+\\}", "[^/]+", pattern)
            if re.fullmatch(pattern, path):
                return True
        return False

    for url, method in endpoints:
        path = url.removeprefix("http://127.0.0.1:80")
        assert _match(path, method), f"client URL 无对应路由: {url}"


async def _seed_two_workers_tasks(app):
    """两个 Worker（A/B）各持有一个 READY 模型文件并各创建一个同步任务。

    返回 (worker_a, task_a_id, task_b_id, worker_b)。
    """
    async with AsyncSession(_engine(app), expire_on_commit=False) as session:
        cipher = _cipher_from_app(app)
        worker_a = Worker(
            name="worker-a",
            hostname="worker-a",
            ip="127.0.0.1",
            port=10150,
            worker_uuid="worker-a-uuid",
            state=WorkerStateEnum.READY,
        )
        worker_b = Worker(
            name="worker-b",
            hostname="worker-b",
            ip="127.0.0.1",
            port=10151,
            worker_uuid="worker-b-uuid",
            state=WorkerStateEnum.READY,
        )
        session.add_all([worker_a, worker_b])
        await session.flush()
        profile = ModelPreheatS3Profile(
            name="center-cache",
            endpoint="https://s3.example.com",
            bucket="models",
            access_key_encrypted=cipher.encrypt("AK"),
            secret_key_encrypted=cipher.encrypt("SK"),
            encryption_key_version="v1",
            config_version=3,
        )
        session.add(profile)
        model_file_a = ModelFile(
            source=SourceEnum.MODEL_SCOPE,
            model_scope_model_id="Qwen/A",
            worker_id=worker_a.id,
            resolved_paths=["/models/Qwen/A"],
            state=ModelFileStateEnum.READY,
            requested_revision="master",
            resolved_revision="1111",
        )
        model_file_b = ModelFile(
            source=SourceEnum.MODEL_SCOPE,
            model_scope_model_id="Qwen/B",
            worker_id=worker_b.id,
            resolved_paths=["/models/Qwen/B"],
            state=ModelFileStateEnum.READY,
            requested_revision="master",
            resolved_revision="2222",
        )
        session.add_all([model_file_a, model_file_b])
        await session.flush()
        profile_id, worker_a_id, worker_b_id = profile.id, worker_a.id, worker_b.id
        model_file_a_id, model_file_b_id = model_file_a.id, model_file_b.id
        await session.commit()

    with TestClient(app) as client:
        response_a = client.post(
            API,
            json={
                "model_file_id": model_file_a_id,
                "profile_id": profile_id,
            },
        )
        assert response_a.status_code == 200, response_a.text
        response_b = client.post(
            API,
            json={
                "model_file_id": model_file_b_id,
                "profile_id": profile_id,
            },
        )
        assert response_b.status_code == 200, response_b.text
    return (
        SimpleNamespace(id=worker_a_id, worker_uuid="worker-a-uuid"),
        response_a.json()["id"],
        response_b.json()["id"],
        SimpleNamespace(id=worker_b_id, worker_uuid="worker-b-uuid"),
    )


def test_worker_tasks_root_list_scoped_to_authenticated_principal(app):
    """根 list 端点：只返回认证 Worker 自己的任务，client 不能越权。"""
    worker_a, task_a_id, task_b_id, worker_b = _run(app, _seed_two_workers_tasks(app))

    async def override():
        return _worker_principal(worker_a.id, worker_a.worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            # 不带 worker_id：只返回本 Worker 任务。
            response = client.get(WORKER_TASKS_ROOT)
            assert response.status_code == 200, response.text
            ids = [item["id"] for item in response.json()["items"]]
            assert ids == [task_a_id]
            assert task_b_id not in ids
            # worker_id 与 principal 一致：合法，结果相同。
            response = client.get(WORKER_TASKS_ROOT, params={"worker_id": worker_a.id})
            assert response.status_code == 200
            assert [item["id"] for item in response.json()["items"]] == [task_a_id]
            # worker_id 指向其他 Worker：越权，403。
            response = client.get(WORKER_TASKS_ROOT, params={"worker_id": worker_b.id})
            assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_worker_tasks_root_watch_route_uses_authenticated_principal_fields(
    app, monkeypatch
):
    """私有根 watch 端点：watch 分支把认证 principal 的 worker_id/worker_uuid
    作为 ``streaming`` 字段传入，并返回 SSE 响应。

    直接调用路由函数并 mock ``_stream_sync_tasks``（返回有限
    异步生成器），不驱动真实无限 SSE 流，从而：
    - 断言 watch 分支以认证 Worker 的 ``worker_id``/``worker_uuid`` 过滤
      （身份隔离的唯一数据来源）；
    - 断言返回 ``text/event-stream``（复用项目 SSE watch 协议）；
    - 客户端越权 ``worker_id`` 触发 403。

    HTTP 层可达性（不再 404）由
    ``test_worker_tasks_root_route_registered_for_generated_client`` 以真实
    client URL → 真实路由收集证明；principal 到最新 Worker 的认证映射由
    ``get_model_preheat_worker_identity`` 依赖保证。
    """
    from gpustack.schemas.common import ListParams

    worker_a, _task_a_id, _task_b_id, worker_b = _run(app, _seed_two_workers_tasks(app))
    principal = _worker_principal(worker_a.id, worker_a.worker_uuid)

    captured: dict = {}

    async def fake_streaming(engine, fields=None, cipher=None):
        captured["fields"] = fields
        captured["cipher"] = cipher
        yield '{"type":"CREATED","data":{"id":1}}\n\n'
        yield  # 让生成器正常结束，避免无限流

    monkeypatch.setattr(model_storage, "_stream_sync_tasks", fake_streaming)

    # watch 分支：字段必须来自认证 principal。
    params = ListParams(page=1, perPage=100, watch=True)

    async def consume_first(resp):
        async for _line in resp.body_iterator:
            return _line

    def call_and_consume(worker_id):
        response = asyncio.run(
            model_storage.list_or_watch_model_storage_sync_tasks(
                request=_request(app),
                engine=_engine(app),
                session=None,
                params=params,
                identity=principal,
                worker_id=worker_id,
            )
        )
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        asyncio.run(consume_first(response))
        return response

    expected_fields = {
        "worker_id": worker_a.id,
        "worker_uuid": worker_a.worker_uuid,
    }
    call_and_consume(None)
    assert captured["fields"] == expected_fields
    assert isinstance(captured["cipher"], ModelPreheatCredentialCipher)

    # 客户端 worker_id 与 principal 一致：合法，仍按 principal 字段过滤。
    captured.clear()
    call_and_consume(worker_a.id)
    assert captured["fields"] == expected_fields

    # 客户端 worker_id 指向其他 Worker：越权，403。
    captured.clear()
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            model_storage.list_or_watch_model_storage_sync_tasks(
                request=_request(app),
                engine=_engine(app),
                session=None,
                params=params,
                identity=principal,
                worker_id=worker_b.id,
            )
        )
    assert excinfo.value.status_code == 403
    assert captured == {}


def test_create_rejects_offline_worker(app):
    """创建同步任务：ModelFile 的 Worker 非 READY（离线）时必须拒绝。"""
    profile_id, model_file_id = _run(
        app, _seed_ids(app, worker_state=WorkerStateEnum.UNREACHABLE)
    )
    response = _client_post(app, model_file_id, profile_id)
    assert response.status_code == 409


def test_create_rejects_stale_worker_registration(app):
    """创建同步任务：ModelFile 绑定的是同 UUID 的旧注册记录时拒绝。

    同 ``worker_uuid`` 存在更新的注册行时，旧 ``worker_id`` 不再是最新
    注册；执行端会永久 ``worker_not_current``，因此创建端必须稳定拒绝。
    改绑到同 UUID 最新注册（READY）后应可创建。
    """

    async def seed_stale():
        async with AsyncSession(_engine(app), expire_on_commit=False) as session:
            cipher = _cipher_from_app(app)
            worker_old = Worker(
                name="worker-a",
                hostname="worker-a",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="worker-a-uuid",
                state=WorkerStateEnum.READY,
            )
            session.add(worker_old)
            await session.flush()
            worker_new = Worker(
                name="worker-a-re",
                hostname="worker-a-re",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="worker-a-uuid",
                state=WorkerStateEnum.READY,
            )
            session.add(worker_new)
            await session.flush()
            profile = ModelPreheatS3Profile(
                name="center-cache",
                endpoint="https://s3.example.com",
                bucket="models",
                access_key_encrypted=cipher.encrypt("AK"),
                secret_key_encrypted=cipher.encrypt("SK"),
                encryption_key_version="v1",
                config_version=3,
            )
            session.add(profile)
            model_file = ModelFile(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="Qwen/Test",
                worker_id=worker_old.id,  # 旧注册记录
                resolved_paths=["/models/Qwen/Test"],
                state=ModelFileStateEnum.READY,
                requested_revision="master",
                resolved_revision="8f73c6a91b",
            )
            session.add(model_file)
            await session.flush()
            profile_id, model_file_id, latest_worker_id = (
                profile.id,
                model_file.id,
                worker_new.id,
            )
            await session.commit()
        return profile_id, model_file_id, latest_worker_id

    profile_id, model_file_id, latest_worker_id = _run(app, seed_stale())
    response = _client_post(app, model_file_id, profile_id)
    assert response.status_code == 409

    async def rebind_latest():
        async with AsyncSession(_engine(app), expire_on_commit=False) as session:
            model_file = await session.get(ModelFile, model_file_id)
            model_file.worker_id = latest_worker_id
            session.add(model_file)
            await session.commit()

    _run(app, rebind_latest())
    response = _client_post(app, model_file_id, profile_id)
    assert response.status_code == 200, response.text


def _client_post(app, model_file_id, profile_id):
    with TestClient(app) as client:
        return client.post(
            API, json={"model_file_id": model_file_id, "profile_id": profile_id}
        )


# ---------------------------------------------------------------------------
# Review 子阶段 B：幂等 / 并发原子性
# ---------------------------------------------------------------------------


def _task_count(app):
    async def count():
        async with AsyncSession(_engine(app)) as session:
            return len((await session.exec(select(ModelStorageSyncTask))).all())

    return _run(app, count())


def _slot_task_id(app, model_file_id, profile_id):
    async def read_slot():
        from gpustack.routes.model_storage import _dedupe_key
        from gpustack.schemas.model_storage_sync import (
            ModelStorageSyncTaskDedupeSlot,
        )

        async with AsyncSession(_engine(app)) as session:
            slot = (
                await session.exec(
                    select(ModelStorageSyncTaskDedupeSlot).where(
                        ModelStorageSyncTaskDedupeSlot.dedupe_key
                        == _dedupe_key(model_file_id, profile_id)
                    )
                )
            ).first()
            return slot.task_id if slot is not None else None

    return _run(app, read_slot())


def test_idempotency_key_record_uses_sync_task_resource_type(app, client):
    """Idempotency-Key 记录的 resource_type 必须是 model_storage_sync_task。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    response = client.post(
        API,
        json={"model_file_id": model_file_id, "profile_id": profile_id},
        headers={"Idempotency-Key": "b-phase-key-1"},
    )
    assert response.status_code == 200, response.text

    async def inspect_record():
        async with AsyncSession(_engine(app)) as session:
            return (
                await session.exec(
                    select(ModelPreheatIdempotencyRecord).where(
                        ModelPreheatIdempotencyRecord.idempotency_key == "b-phase-key-1"
                    )
                )
            ).one()

    record = _run(app, inspect_record())
    assert record.resource_type == "model_storage_sync_task"
    assert record.resource_id == response.json()["id"]


def test_request_hash_is_stable_across_profile_config_version_change(app, client):
    """request hash 只包含稳定请求语义（model_file_id + profile_id +
    request_digest），不混入 profile_config_version 等可变派生状态：
    Profile 配置版本变化后，同一 Idempotency-Key 仍是等价重放而不是 409。
    """
    profile_id, model_file_id = _run(app, _seed_ids(app))
    key = "b-phase-stable-hash"
    first = client.post(
        API,
        json={"model_file_id": model_file_id, "profile_id": profile_id},
        headers={"Idempotency-Key": key},
    )
    assert first.status_code == 200, first.text

    # 任务进入终态并让 Profile 配置版本变化。
    _run(
        app,
        _set_task_terminal(
            app, first.json()["id"], state=ModelStorageSyncTaskStateEnum.READY
        ),
    )

    async def bump_profile_version():
        async with AsyncSession(_engine(app), expire_on_commit=False) as session:
            profile = await session.get(ModelPreheatS3Profile, profile_id)
            profile.config_version += 1
            session.add(profile)
            await session.commit()

    _run(app, bump_profile_version())

    # 同一语义请求 + 同一 Idempotency-Key：hash 稳定，返回既有任务而非 409。
    replay = client.post(
        API,
        json={"model_file_id": model_file_id, "profile_id": profile_id},
        headers={"Idempotency-Key": key},
    )
    assert replay.status_code == 200, replay.text
    assert replay.json()["id"] == first.json()["id"]
    # 重放返回既有任务（其 profile_config_version 固定为创建时值）。
    assert (
        replay.json()["profile_config_version"]
        == first.json()["profile_config_version"]
    )


def test_concurrent_create_with_same_idempotency_key_single_task(app):
    """并发相同 Idempotency-Key 创建：任务与记录原子提交，不产生重复任务，
    竞争者得到既有等价结果。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    key = "b-phase-concurrent-key"
    results: list = []
    barrier = threading.Barrier(2)

    def post():
        with TestClient(app) as client:
            barrier.wait()
            response = client.post(
                API,
                json={"model_file_id": model_file_id, "profile_id": profile_id},
                headers={"Idempotency-Key": key},
            )
        results.append(response)

    threads = [threading.Thread(target=post) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for response in results:
        assert response.status_code == 200, response.text
    assert len({response.json()["id"] for response in results}) == 1
    assert _task_count(app) == 1


def test_concurrent_different_requests_same_idempotency_key_no_ghost_task(app):
    """不同请求并发复用同一 Key：仅一个请求成功，冲突方不得返回已回滚任务。"""
    profile_id, first_model_file_id = _run(app, _seed_ids(app))

    async def add_second_model_file():
        async with AsyncSession(_engine(app), expire_on_commit=False) as session:
            first = await session.get(ModelFile, first_model_file_id)
            second = ModelFile(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="Qwen/Other",
                worker_id=first.worker_id,
                resolved_paths=["/models/Qwen/Other"],
                state=ModelFileStateEnum.READY,
                requested_revision="release",
                resolved_revision="9f84d7c61a",
            )
            session.add(second)
            await session.flush()
            second_id = second.id
            await session.commit()
            return second_id

    second_model_file_id = _run(app, add_second_model_file())
    key = "different-requests-one-key"
    results = []
    barrier = threading.Barrier(2)

    def post(model_file_id):
        with TestClient(app) as client:
            barrier.wait()
            response = client.post(
                API,
                json={"model_file_id": model_file_id, "profile_id": profile_id},
                headers={"Idempotency-Key": key},
            )
        results.append(response)

    threads = [
        threading.Thread(target=post, args=(first_model_file_id,)),
        threading.Thread(target=post, args=(second_model_file_id,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(response.status_code for response in results) == [200, 409]
    successful_task_id = next(
        response.json()["id"] for response in results if response.status_code == 200
    )

    async def persisted_task_ids():
        async with AsyncSession(_engine(app)) as session:
            return {
                task.id
                for task in (await session.exec(select(ModelStorageSyncTask))).all()
            }

    assert _run(app, persisted_task_ids()) == {successful_task_id}


def test_concurrent_create_without_key_single_active_task(app):
    """并发无 Idempotency-Key 创建同一 (model_file_id, profile_id)：
    数据库级去重保证只有一个活动任务，竞争者返回既有任务。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    results: list = []
    barrier = threading.Barrier(2)

    def post():
        with TestClient(app) as client:
            barrier.wait()
            results.append(
                client.post(
                    API,
                    json={"model_file_id": model_file_id, "profile_id": profile_id},
                )
            )

    threads = [threading.Thread(target=post) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for response in results:
        assert response.status_code == 200, response.text
    assert len({response.json()["id"] for response in results}) == 1
    assert _task_count(app) == 1
    assert _slot_task_id(app, model_file_id, profile_id) == results[0].json()["id"]


def test_worker_complete_releases_slot_allows_new_task(app):
    """Worker complete（ready 终态）在同一事务释放槽位：新任务可立即创建。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]
    assert _slot_task_id(app, model_file_id, profile_id) == task_id

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 200
        assert _slot_task_id(app, model_file_id, profile_id) is None
        # 新任务可以立即创建（同 (model_file_id, profile_id)）。
        again = _create_task(app, profile_id, model_file_id)
        assert again["id"] != task_id
        assert _slot_task_id(app, model_file_id, profile_id) == again["id"]
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_worker_fail_releases_slot_allows_new_task(app):
    """Worker fail（error 终态）在同一事务释放槽位。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]
    assert _slot_task_id(app, model_file_id, profile_id) == task_id

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/v1/model-storage-worker-tasks/{task_id}/fail",
                json={
                    "error_code": "s3_publish_failed",
                    "lease_token": _fetch_task_lease_token(app, task_id),
                },
            )
            assert response.status_code == 200
            assert client.get(DETAIL.format(id=task_id)).json()["state"] == "error"
        assert _slot_task_id(app, model_file_id, profile_id) is None
        again = _create_task(app, profile_id, model_file_id)
        assert again["id"] != task_id
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_cancel_releases_slot_allows_new_task(app, client):
    """取消活动任务：同一事务释放槽位，新任务可立即创建。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    first = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    task_id = first.json()["id"]
    assert _slot_task_id(app, model_file_id, profile_id) == task_id
    cancel = client.delete(DETAIL.format(id=task_id))
    assert cancel.status_code == 200
    assert _slot_task_id(app, model_file_id, profile_id) is None
    again = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    assert again.status_code == 200
    assert again.json()["id"] != task_id


# ---------------------------------------------------------------------------
# 定向复审 8 项 Important：回滚标量 / 幂等重放 / lease / 库存版本 / 冻结文件
# 选择 / moving revision / Worker list 脱敏 / 并发（含线程异常警告）
# ---------------------------------------------------------------------------


async def _set_model_file_resolved_paths(app, model_file_id, resolved_paths):
    from gpustack.schemas.model_files import ModelFile as ModelFileORM

    async with AsyncSession(_engine(app), expire_on_commit=False) as session:
        model_file = await session.get(ModelFileORM, model_file_id)
        model_file.resolved_paths = list(resolved_paths)
        session.add(model_file)
        await session.commit()


def test_execution_payload_uses_frozen_paths_not_current_modelfile(app):
    """执行文件选择在任务创建时冻结：execution payload 不得重读当前 ModelFile。

    创建任务后修改 ModelFile.resolved_paths；Worker 拉取 execution payload 时
    必须仍返回创建时冻结的源路径与扫描规约（root/patterns），而不是当前值。
    """
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    task_id = created["id"]
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]

    # 创建后改变当前 ModelFile.resolved_paths（模拟重新下载/修改）。
    _run(
        app,
        _set_model_file_resolved_paths(app, model_file_id, ["/models/Qwen/CHANGED"]),
    )

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.get(WORKER_EXEC.format(id=task_id))
            assert response.status_code == 200, response.text
            body = response.json()
            # 冻结：仍是创建时源路径（单目录 /models/Qwen/Test），非当前 CHANGED。
            assert body["source_paths"] == ["/models/Qwen/Test"]
            # 冻结扫描规约：root 与 patterns 来自创建时 compute_scan_spec。
            from gpustack.server.model_storage_scan_spec import compute_scan_spec

            frozen_root, frozen_patterns = compute_scan_spec(["/models/Qwen/Test"])
            assert body["scan_spec"]["root"] == frozen_root
            assert body["scan_spec"]["include_patterns"] == list(frozen_patterns)
            # 一次性 lease token 非空（明文只在此受约束端点出现）。
            assert body["lease_token"]
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_idempotency_replay_stable_across_ready_and_profile_change(app, client):
    """Idempotency-Key 重放：任务 READY 且 Profile/ModelFile 变化后仍返回原任务。

    覆盖：创建后任务进入 READY、Profile 配置版本/凭据变化、ModelFile revision
    变化，同一 Key 重放都稳定返回原任务（不 409、不新建）。
    """
    profile_id, model_file_id = _run(app, _seed_ids(app))
    key = "replay-stable-key"
    first = client.post(
        API,
        json={"model_file_id": model_file_id, "profile_id": profile_id},
        headers={"Idempotency-Key": key},
    )
    assert first.status_code == 200, first.text
    task_id = first.json()["id"]

    # 任务进入 READY 终态。
    _run(
        app,
        _set_task_terminal(app, task_id, state=ModelStorageSyncTaskStateEnum.READY),
    )
    # Profile 配置版本变化 + 凭据变化。

    async def bump_profile():
        async with AsyncSession(_engine(app), expire_on_commit=False) as session:
            profile = await session.get(ModelPreheatS3Profile, profile_id)
            profile.config_version += 1
            profile.endpoint = "https://s3.example.com/v2"
            session.add(profile)
            await session.commit()

    _run(app, bump_profile())
    # ModelFile resolved revision 变化。
    _run(
        app,
        _set_model_file_resolved_paths(app, model_file_id, ["/models/Qwen/Test"]),
    )

    replay = client.post(
        API,
        json={"model_file_id": model_file_id, "profile_id": profile_id},
        headers={"Idempotency-Key": key},
    )
    assert replay.status_code == 200, replay.text
    assert replay.json()["id"] == task_id
    # 重放返回原任务（profile_config_version 固定为创建时值）。
    assert (
        replay.json()["profile_config_version"]
        == first.json()["profile_config_version"]
    )
    # 不产生新任务。
    assert _task_count(app) == 1


def test_idempotency_key_concurrent_hit_binds_existing_task(app):
    """并发活动任务命中新 Idempotency-Key：同一事务/后续把 Key 绑定既有任务，
    冲突语义稳定（Key 记录持久化到既有任务，后续重放返回同一任务而非 409）。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    key = "concurrent-hit-key"
    # 先创建一个活动任务（无 Key）。
    existing = _create_task(app, profile_id, model_file_id)
    # 再带 Key 创建同键：活动任务去重命中既有任务，并把 Key 绑定到既有任务。
    second = _client_post_with_key(app, model_file_id, profile_id, key)
    assert second.status_code == 200, second.text
    assert second.json()["id"] == existing["id"]

    # Key 必须被持久化绑定到既有活动任务（resource_id == 既有任务 id）。
    async def read_record():
        async with AsyncSession(_engine(app)) as session:
            return (
                await session.exec(
                    select(ModelPreheatIdempotencyRecord).where(
                        ModelPreheatIdempotencyRecord.idempotency_key == key
                    )
                )
            ).first()

    record = _run(app, read_record())
    assert record is not None, "活动任务命中新 Key 必须持久化该 Key 绑定"
    assert record.resource_id == existing["id"]
    assert record.resource_type == "model_storage_sync_task"
    # 后续同一 Key 重放：稳定返回既有任务（而不是 409 或新建）。
    replay = _client_post_with_key(app, model_file_id, profile_id, key)
    assert replay.status_code == 200, replay.text
    assert replay.json()["id"] == existing["id"]
    assert _task_count(app) == 1


def _client_post_with_key(app, model_file_id, profile_id, key):
    with TestClient(app) as client:
        return client.post(
            API,
            json={"model_file_id": model_file_id, "profile_id": profile_id},
            headers={"Idempotency-Key": key},
        )


def test_complete_requires_lease_token(app):
    """complete 必须携带有效 lease token：无 lease/错 lease 稳定 409。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            # 无 lease_token（契约字段必选）：422。
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": created["request_digest"],
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 422
            # 错 lease_token：稳定 409，不推进终态。
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": "wrong-lease-token",
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 409
            detail = client.get(DETAIL.format(id=task_id))
            assert detail.json()["state"] != "ready"
            assert detail.json()["artifact_id"] is None
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_complete_requires_manifest_path(app):
    """Worker 不能省略 manifest_path，Server 不得用当前 Profile prefix 补写。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    task_id = created["id"]

    async def override():
        return _worker_principal(created["worker_id"], created["worker_uuid"])

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                },
            )
            assert response.status_code == 422
            assert client.get(DETAIL.format(id=task_id)).json()["state"] != "ready"
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_complete_equivalent_replay_idempotent_app(app):
    """同一已完成等价重放（同一 lease + 同一 artifact + 同一 manifest）幂等成功。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]
    lease = _fetch_task_lease_token(app, task_id)

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    body = {
        "artifact_id": "e" * 64,
        "request_digest": created["request_digest"],
        "lease_token": lease,
        "file_count": 2,
        "total_size": 10,
        "manifest_digest": "c" * 64,
        "manifest_path": "models/test/manifest.json",
    }
    try:
        with TestClient(app) as client:
            first = client.post(WORKER_COMPLETE.format(id=task_id), json=body)
            assert first.status_code == 200
            assert client.get(DETAIL.format(id=task_id)).json()["state"] == "ready"
            # 等价重放：完全相同的 body → 幂等成功（200），artifact 不变。
            replay = client.post(WORKER_COMPLETE.format(id=task_id), json=body)
            assert replay.status_code == 200, replay.text
            detail = client.get(DETAIL.format(id=task_id))
            assert detail.json()["state"] == "ready"
            assert detail.json()["artifact_id"] == "e" * 64
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_complete_conflicting_replay_stable_conflict(app):
    """已完成任务上携带不同 artifact / 不同 manifest 的重放：稳定 409，不覆盖。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]
    lease = _fetch_task_lease_token(app, task_id)

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": lease,
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            # 不同 artifact：稳定 409，不覆盖原绑定。
            conflict = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "f" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": lease,
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert conflict.status_code == 409
            # 不同 manifest_digest：稳定 409。
            conflict2 = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": lease,
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "d" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert conflict2.status_code == 409
            # 原绑定未被覆盖。
            assert (
                client.get(DETAIL.format(id=task_id)).json()["artifact_id"] == "e" * 64
            )
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_fail_requires_lease_token_and_terminal_conflict(app):
    """fail 必须携带有效 lease：错 lease 409；任务已终态时 fail 稳定 409（不折叠 200）。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]
    lease = _fetch_task_lease_token(app, task_id)

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            # 错 lease：409，任务仍活动。
            bad = client.post(
                WORKER_FAIL.format(id=task_id),
                json={"error_code": "boom", "lease_token": "wrong"},
            )
            assert bad.status_code == 409
            assert client.get(DETAIL.format(id=task_id)).json()["state"] != "error"
            # 正常 fail：200 → error 终态。
            ok = client.post(
                WORKER_FAIL.format(id=task_id),
                json={"error_code": "s3_publish_failed", "lease_token": lease},
            )
            assert ok.status_code == 200
            assert client.get(DETAIL.format(id=task_id)).json()["state"] == "error"
            # 已终态再 fail：稳定 409（不折叠为 200）。
            again = client.post(
                WORKER_FAIL.format(id=task_id),
                json={"error_code": "s3_publish_failed", "lease_token": lease},
            )
            assert again.status_code == 409
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_complete_uses_task_profile_config_version_not_current(app):
    """complete 使用任务冻结版本和 Worker 回传路径，不读取当前 prefix。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))

    async def set_frozen_prefix():
        async with AsyncSession(_engine(app), expire_on_commit=False) as session:
            profile = await session.get(ModelPreheatS3Profile, profile_id)
            profile.prefix = "frozen-prefix"
            session.add(profile)
            await session.commit()

    _run(app, set_frozen_prefix())
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]
    lease = _fetch_task_lease_token(app, task_id)
    created_version = created["profile_config_version"]

    # 任务创建后把当前 Profile 配置版本提升（模拟 Profile 变化）。
    async def bump():
        async with AsyncSession(_engine(app), expire_on_commit=False) as session:
            profile = await session.get(ModelPreheatS3Profile, profile_id)
            profile.config_version += 10
            profile.prefix = "changed-after-task"
            session.add(profile)
            await session.commit()

    _run(app, bump())

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "7" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": lease,
                    "file_count": 1,
                    "total_size": 5,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "frozen-prefix/modelscope/Qwen/Test/manifest.json",
                },
            )
            assert response.status_code == 200, response.text
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)

    artifacts = _run(app, _inventory_artifact(app, "7" * 64))
    assert len(artifacts) == 1
    # 库存归属版本必须是任务创建时固定的版本，而非变化后的当前版本。
    assert artifacts[0].profile_config_version == created_version
    assert (
        artifacts[0].manifest_path == "frozen-prefix/modelscope/Qwen/Test/manifest.json"
    )


def test_create_rejects_missing_resolved_revision_for_dev_branch(app):
    """requested_revision=dev 不能在 resolved_revision 缺失时冒充解析结果。"""
    profile_id, model_file_id = _run(
        app, _seed_ids(app, resolved_revision=None, requested_revision="dev")
    )
    response = client_post(app, model_file_id, profile_id)
    assert response.status_code == 409


def test_create_rejects_missing_resolved_revision_for_release_branch(app):
    """requested_revision=release 同样不能通过有限别名表逃逸。"""
    profile_id, model_file_id = _run(
        app, _seed_ids(app, resolved_revision=None, requested_revision="release")
    )
    response = client_post(app, model_file_id, profile_id)
    assert response.status_code == 409


def test_create_accepts_concrete_resolved_revision(app):
    """resolved_revision 为具体 commit（requested 可为 moving alias）时允许创建。"""
    profile_id, model_file_id = _run(
        app, _seed_ids(app, resolved_revision="8f73c6a91b", requested_revision="master")
    )
    response = client_post(app, model_file_id, profile_id)
    assert response.status_code == 200, response.text
    assert response.json()["resolved_revision"] == "8f73c6a91b"


def client_post(app, model_file_id, profile_id):
    with TestClient(app) as client:
        return client.post(
            API, json={"model_file_id": model_file_id, "profile_id": profile_id}
        )


def test_worker_list_response_model_excludes_sensitive_fields(app):
    """Worker list 端点显式 Public response_model：credential_snapshot_encrypted /
    encryption_key_version / lease_token_encrypted 不泄露。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.get(WORKER_TASKS_ROOT)
            assert response.status_code == 200, response.text
            body = response.text
            items = response.json()["items"]
            assert len(items) == 1
            # 敏感字段不进入任何 item 或响应文本。
            for field in (
                "credential_snapshot_encrypted",
                "encryption_key_version",
                "lease_token_encrypted",
            ):
                assert field not in items[0]
                assert field not in body
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_concurrent_create_no_unhandled_thread_exception(app):
    """并发创建（同 Key 与无 Key）：所有线程均返回、无未处理线程异常。

    通过把 ``PytestUnhandledThreadExceptionWarning`` 提升为 error（在
    conftest 或 -W 标志下），确保 IntegrityError 回滚路径不会在后台线程
    抛未捕获异常（MissingGreenlet 等）。此用例用 barrier 对齐 4 线程，
    全部成功返回且只产生 1 个活动任务。
    """
    profile_id, model_file_id = _run(app, _seed_ids(app))
    key = "thread-safe-key"
    results: list = []
    errors: list = []
    barrier = threading.Barrier(4)

    def post():
        try:
            with TestClient(app) as client:
                barrier.wait()
                response = client.post(
                    API,
                    json={"model_file_id": model_file_id, "profile_id": profile_id},
                    headers={"Idempotency-Key": key},
                )
            results.append(response)
        except Exception as exc:  # noqa: BLE001 - 捕获线程内未处理异常
            errors.append(exc)

    threads = [threading.Thread(target=post) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # 所有线程均返回且无未处理异常。
    assert not errors, f"线程内未处理异常: {errors}"
    assert len(results) == 4
    for response in results:
        assert response.status_code == 200, response.text
    # 只产生一个任务。
    assert len({response.json()["id"] for response in results}) == 1
    assert _task_count(app) == 1


def test_no_orphan_task_on_active_slot_conflict(app):
    """槽位被既有活动任务持有时：新创建不得产生重复任务或遗留任务。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    existing = _create_task(app, profile_id, model_file_id)
    assert _task_count(app) == 1
    # 同键再次创建：走槽位兜底查询返回既有任务，无重复。
    response = _client_post(app, model_file_id, profile_id)
    assert response.status_code == 200
    assert response.json()["id"] == existing["id"]
    assert _task_count(app) == 1


# ---------------------------------------------------------------------------
# Review 子阶段 C：Artifact 精确匹配、发布语义与 complete 契约
# ---------------------------------------------------------------------------


async def _inventory_artifact(app, artifact_id):
    async with AsyncSession(_engine(app)) as session:
        rows = (
            await session.exec(
                select(ModelPreheatArtifact).where(
                    ModelPreheatArtifact.artifact_id == artifact_id
                )
            )
        ).all()
        return rows


def _seed_artifact_for_task(app, profile_id, **overrides):
    """按任务实际 request identity 精确构造库存（默认匹配 seed 的任务）。

    seed 的 resolved_paths=["/models/Qwen/Test"]（单目录）→ 实际文件选择
    patterns 由 compute_scan_spec 推导（["**"]，整目录全量）。库存必须与之
    一致才能预绑定。
    """
    from gpustack.server.model_storage_scan_spec import compute_scan_spec

    _root, seed_patterns = compute_scan_spec(["/models/Qwen/Test"])
    defaults = dict(
        profile_id=profile_id,
        artifact_id="b" * 64,
        source="modelscope",
        model_id="Qwen/Test",
        resolved_revision="8f73c6a91b",
        include_patterns=list(seed_patterns),
        exclude_patterns=[],
    )
    defaults.update(overrides)
    _run(app, _seed_inventory_artifact(app, **defaults))


def test_create_prebinds_artifact_on_exact_match(app, client):
    """预绑定必须精确匹配 resolved revision + 实际文件选择。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    _seed_artifact_for_task(app, profile_id)
    response = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    assert response.status_code == 200, response.text
    assert response.json()["artifact_id"] == "b" * 64
    # request_identity 的 include_patterns 反映实际文件选择（非粗粒度空集）。

    async def read_task():
        async with AsyncSession(_engine(app), expire_on_commit=False) as session:
            task = await session.get(ModelStorageSyncTask, response.json()["id"])
            return task.request_identity

    identity = _run(app, read_task())
    from gpustack.server.model_storage_scan_spec import compute_scan_spec

    _root, expected = compute_scan_spec(["/models/Qwen/Test"])
    assert identity["include_patterns"] == list(expected)
    assert set(identity) == {
        "source",
        "model_id",
        "requested_revision",
        "include_patterns",
        "exclude_patterns",
    }
    assert "/models/" not in json.dumps(identity)


def test_create_rejects_prebind_on_revision_mismatch(app, client):
    """同一模型不同 resolved revision 的旧库存不得预绑定。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    _seed_artifact_for_task(app, profile_id, resolved_revision="0000000deadbeef")
    response = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    assert response.status_code == 200
    assert response.json()["artifact_id"] is None


def test_create_rejects_prebind_on_pattern_mismatch(app, client):
    """同一模型不同文件选择（patterns）的旧库存不得预绑定。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    # patterns 与任务实际文件选择（["Test","Test/**"]）不同。
    _seed_artifact_for_task(app, profile_id, include_patterns=["other/**"])
    response = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    assert response.status_code == 200
    assert response.json()["artifact_id"] is None


def test_create_no_prebind_when_multiple_valid_artifacts(app, client):
    """多个合法库存（无法唯一确定）时不预绑定，保持 NULL 供 Worker 绑定。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    _seed_artifact_for_task(app, profile_id, artifact_id="a" * 64)
    _seed_artifact_for_task(app, profile_id, artifact_id="b" * 64)
    response = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    assert response.status_code == 200
    assert response.json()["artifact_id"] is None


def test_prebound_complete_advances_ready_and_writes_inventory(app):
    """预绑定 artifact 后 complete 仍必须推进终态并写库存（不破坏 CAS）。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    _seed_artifact_for_task(app, profile_id)
    created = _create_task(app, profile_id, model_file_id)
    assert created["artifact_id"] == "b" * 64
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "b" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 200, response.text
            detail = client.get(DETAIL.format(id=task_id))
            assert detail.json()["state"] == "ready"
            assert detail.json()["artifact_id"] == "b" * 64
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)

    artifacts = _run(app, _inventory_artifact(app, "b" * 64))
    assert len(artifacts) == 1
    assert artifacts[0].manifest_state == ModelPreheatInventoryManifestStateEnum.VALID
    assert artifacts[0].file_count == 2


def test_complete_rejects_forged_artifact_id(app):
    """伪造 artifact_id（与预绑定不一致）必须拒绝（稳定冲突），不覆盖。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    _seed_artifact_for_task(app, profile_id)  # 预绑定 "b"*64
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "f" * 64,  # 与预绑定 "b"*64 不同
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 409
            assert client.get(DETAIL.format(id=task_id)).json()["state"] != "ready"
            # 任务保持活动态、artifact_id 未被伪造值覆盖。
            assert (
                client.get(DETAIL.format(id=task_id)).json()["artifact_id"] == "b" * 64
            )
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_complete_rejects_wrong_request_digest(app):
    """request_digest 与任务不一致（过期/重放/串任务）必须拒绝。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": "f" * 64,  # 与任务 request_digest 不同
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 409
            assert client.get(DETAIL.format(id=task_id)).json()["state"] != "ready"
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_complete_rejects_negative_stats(app):
    """负 file_count/total_size 必须被契约拒绝（422）。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": -1,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 422
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 2,
                    "total_size": -5,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 422
            # 契约校验在写库前失败，任务仍是活动态。
            assert client.get(DETAIL.format(id=task_id)).json()["state"] != "ready"
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_complete_requires_valid_manifest_digest(app):
    """manifest_digest 非 64 位小写十六进制必须被契约拒绝（422）。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "not-a-sha",
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_complete_by_stale_worker_rejected(app):
    """旧注册 Worker（非当前注册）complete 必须被 lease/身份隔离拒绝（403）。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_uuid = created["worker_uuid"]
    task_id = created["id"]

    async def seed_new_registration():
        async with AsyncSession(_engine(app), expire_on_commit=False) as session:
            # 同一 worker_uuid 重新注册（新 id 成为最新），任务仍指向旧 id。
            session.add(
                Worker(
                    name="worker-a-new",
                    hostname="worker-a",
                    ip="127.0.0.1",
                    port=10152,
                    worker_uuid=worker_uuid,
                    state=WorkerStateEnum.READY,
                )
            )
            await session.commit()

    _run(app, seed_new_registration())

    async def override():
        # 用任务的（旧）worker_id 认证，但该注册已不是最新 → worker_not_current。
        return _worker_principal(created["worker_id"], worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 403
            assert client.get(DETAIL.format(id=task_id)).json()["state"] != "ready"
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_complete_by_other_worker_rejected(app):
    """其他 Worker（worker_uuid 不匹配）complete 必须被拒绝（403）。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    task_id = created["id"]

    async def override():
        return _worker_principal(999, "other-worker-uuid")

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "e" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_complete_writes_and_updates_inventory(app):
    """complete 成功后写入/更新统一 Artifact 库存（profile + 当前 config version）。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]
    artifact_id = "9" * 64

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": artifact_id,
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 3,
                    "total_size": 33,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 200
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)

    artifacts = _run(app, _inventory_artifact(app, artifact_id))
    assert len(artifacts) == 1
    art = artifacts[0]
    assert art.profile_id == profile_id
    assert art.profile_config_version == 3
    assert art.source == "modelscope"
    assert art.model_id == "Qwen/Test"
    assert art.resolved_revision == "8f73c6a91b"
    # include_patterns 与任务 request_identity 一致（seed 单目录 → ["**"]）。
    from gpustack.server.model_storage_scan_spec import compute_scan_spec

    _root, expected_patterns = compute_scan_spec(["/models/Qwen/Test"])
    assert art.include_patterns == list(expected_patterns)
    assert art.manifest_state == ModelPreheatInventoryManifestStateEnum.VALID
    assert art.file_count == 3
    assert art.total_size == 33
    assert art.manifest_digest == "c" * 64
    assert art.manifest_path  # Server 按 Profile prefix + 身份推导


def test_complete_publishes_updated_event(app, monkeypatch):
    """complete 成功必须广播 UPDATED 事件（终态已写入，Public schema 不含凭据）。

    事件总线跨事件循环投递在 TestClient portal 下不稳定，这里直接捕获路由对
    ``ModelStorageSyncTask._publish_event`` 的调用（类属性补丁对 TestClient
    的独立线程同样生效，patch 在请求前设置、请求后恢复）。
    """
    from gpustack.schemas.model_storage_sync import (
        ModelStorageSyncTask,
        ModelStorageSyncTaskPublic,
    )
    from gpustack.server.bus import EventType

    published: list = []

    async def recorder(cls, event_type, data):
        published.append((event_type, data))

    monkeypatch.setattr(ModelStorageSyncTask, "_publish_event", classmethod(recorder))

    profile_id, model_file_id = _run(app, _seed_ids(app))
    created = _create_task(app, profile_id, model_file_id)
    worker_id, worker_uuid = created["worker_id"], created["worker_uuid"]
    task_id = created["id"]

    async def override():
        return _worker_principal(worker_id, worker_uuid)

    app.dependency_overrides[get_model_preheat_worker_identity] = override
    try:
        with TestClient(app) as client:
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={
                    "artifact_id": "8" * 64,
                    "request_digest": created["request_digest"],
                    "lease_token": _fetch_task_lease_token(app, task_id),
                    "file_count": 2,
                    "total_size": 10,
                    "manifest_digest": "c" * 64,
                    "manifest_path": "models/test/manifest.json",
                },
            )
            assert response.status_code == 200
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)

    updated = [d for et, d in published if et == EventType.UPDATED and d.id == task_id]
    assert updated, "complete 未广播 UPDATED 事件"
    state = updated[0].state
    state = state.value if hasattr(state, "value") else state
    assert state == "ready"
    # Public schema 不携带凭据字段（Worker/SSE 消费路径不含敏感字段）。
    assert (
        "credential_snapshot_encrypted" not in ModelStorageSyncTaskPublic.model_fields
    )


def test_cancel_preserves_slot_transaction_semantics(app, client):
    """取消活动任务：终态 + 槽位释放原子（B 语义在 C 下仍成立）。"""
    profile_id, model_file_id = _run(app, _seed_ids(app))
    _seed_artifact_for_task(app, profile_id)  # 预绑定
    created = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    task_id = created.json()["id"]
    assert _slot_task_id(app, model_file_id, profile_id) == task_id
    assert client.delete(DETAIL.format(id=task_id)).status_code == 200
    assert _slot_task_id(app, model_file_id, profile_id) is None
    # 取消后同键可再创建（槽位已释放）。
    again = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    assert again.status_code == 200
    assert again.json()["id"] != task_id
