"""任务 3：模型同步 API 定向测试。

覆盖：请求只接受 model_file_id + profile_id；READY 模型、来源与 Worker 归属
校验、Profile 版本；Idempotency-Key 重放；活动任务去重；客户端不能提交对象
Key；完成后写入统一 Artifact 库存（CAS 绑定）；Public schema 分别返回模型
source、本次 transfer_source、S3 Profile 与来源 Worker，不混用字段；凭据快照
不进入 Public schema / SSE / 日志。
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.auth import get_admin_user, get_current_user
from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
    generate_model_preheat_credential_key,
)
from gpustack.routes import model_storage
from gpustack.schemas.model_files import (
    ModelFile,
    ModelFileStateEnum,
)
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatInventoryManifestStateEnum,
)
from gpustack.schemas.model_storage_sync import (
    ModelFileTransferSourceEnum,
    ModelStorageSyncTask,
    ModelStorageSyncTaskStateEnum,
)
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_session
from gpustack.server.model_preheat_worker_identity import (
    ModelPreheatWorkerPrincipal,
    get_model_preheat_worker_identity,
)

API = "/v1/model-storage-sync-tasks"
DETAIL = "/v1/model-storage-sync-tasks/{id}"
WORKER_EXEC = "/v1/model-storage-worker-tasks/{id}/execution-payload"
WORKER_COMPLETE = "/v1/model-storage-worker-tasks/{id}/complete"


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
        resolved_paths=["/models/Qwen/Test"] if state == ModelFileStateEnum.READY else [],
        state=state,
        requested_revision="master",
        resolved_revision="8f73c6a91b",
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
    profile_id, model_file_id = _run(
        app, _seed_ids(app)
    )
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


def test_create_rejects_non_ready_model_file(app, client):
    profile_id, model_file_id = _run(
        app, _seed_ids(app, state=ModelFileStateEnum.DOWNLOADING)
    )
    response = client.post(
        API, json={"model_file_id": model_file_id, "profile_id": profile_id}
    )
    assert response.status_code == 409


def test_create_rejects_unsupported_source(app, client):
    profile_id, model_file_id = _run(
        app, _seed_ids(app, source=SourceEnum.LOCAL_PATH)
    )
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
    response = client.post(API, json={"model_file_id": model_file_id, "profile_id": 9999})
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
    response = client.get(DETAIL.format(id=task_id))
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "modelscope"
    assert body["transfer_source"] == "s3"
    assert body["profile"]["id"] == profile_id
    assert body["profile"]["config_version"] == 3
    assert body["source_worker_id"] == body["worker_id"]
    assert body["source_worker_name"] == "worker-a"
    assert body["artifact_id"] == "a" * 64
    assert "credential_snapshot_encrypted" not in body


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


def _create_task(app, profile_id, model_file_id):
    with TestClient(app) as client:
        response = client.post(
            API, json={"model_file_id": model_file_id, "profile_id": profile_id}
        )
    assert response.status_code == 200, response.text
    return response.json()


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
                json={"artifact_id": "e" * 64, "file_count": 2, "total_size": 10},
            )
            assert response.status_code == 200
            detail = client.get(DETAIL.format(id=task_id))
            assert detail.json()["artifact_id"] == "e" * 64
            assert detail.json()["state"] == "ready"
            assert detail.json()["transfer_source"] == "s3"
            # 重复完成不覆盖已绑定的 artifact_id。
            response = client.post(
                WORKER_COMPLETE.format(id=task_id),
                json={"artifact_id": "f" * 64, "file_count": 3, "total_size": 20},
            )
            assert response.status_code == 200
            detail = client.get(DETAIL.format(id=task_id))
            assert detail.json()["artifact_id"] == "e" * 64
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)


def test_worker_complete_rejected_after_cancel(app):
    """任务被取消后，Worker 的 complete 不得把状态改回 ready。"""
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
                json={"artifact_id": "e" * 64, "file_count": 2, "total_size": 10},
            )
            assert response.status_code == 200
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
            # 执行 payload 含明文 S3 凭据与可信本地源路径。
            assert body["profile"]["access_key"] == "AK"
            assert body["source_paths"] == ["/models/Qwen/Test"]
            assert body["request_identity"]["model_id"] == "Qwen/Test"
            assert response.headers.get("cache-control") == "no-store"
    finally:
        app.dependency_overrides.pop(get_model_preheat_worker_identity, None)
