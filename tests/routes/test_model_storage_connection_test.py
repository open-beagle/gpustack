"""任务 3：保存前连接测试（capabilities + connection-tests）定向测试。

覆盖：仅管理员可调用；使用未保存表单分阶段报告连接/Bucket/写/读/删除；
受控解析器拒绝 link-local/云元数据地址、允许 loopback 与 RFC1918；异常路径
也清理临时对象（注入 fake client 验证 remove_object 被调用）；Endpoint 校验
与 Profile CRUD 共用；凭据不进入响应体。

说明：真实 TCP/TLS 与 S3 依赖外部环境（不可用），本测试注入 fake minio
client 并把受控 DNS 解析/TCP 探测短路为成功，从而只验证**真实的分阶段编排、
临时对象清理与受控地址校验**逻辑，不伪造外部可达性。
"""

import asyncio
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
from gpustack.model_preheat_credentials import generate_model_preheat_credential_key
from gpustack.routes import model_storage
from gpustack.server import model_storage_connection_test as ctest
from gpustack.schemas.users import User
from gpustack.server.db import get_session

API = "/v1/model-storage/connection-tests"


class _FakeClient:
    """可注入的 fake minio client，按阶段返回可控结果并记录对象操作。"""

    def __init__(self, *, list_ok=True, put_ok=True, get_ok=True, delete_ok=True):
        self.list_ok = list_ok
        self.put_ok = put_ok
        self.get_ok = get_ok
        self.delete_ok = delete_ok
        self.put_object_names = []
        self.deleted = []
        self.remove_attempts = 0

    def list_objects(self, bucket, prefix=None, **kwargs):
        if not self.list_ok:
            raise Exception("list failed")
        return iter([])

    def put_object(self, bucket, name, data, length, **kwargs):
        if not self.put_ok:
            raise Exception("write failed")
        # 记录写入的 payload 字节，供 get_object 原样返回以通过摘要校验。
        self._last_payload = data.read()
        self.put_object_names.append(name)

    def get_object(self, bucket, name):
        if not self.get_ok:
            raise Exception("read failed")
        import io

        return io.BytesIO(getattr(self, "_last_payload", b""))

    def remove_object(self, bucket, name):
        self.remove_attempts += 1
        if not self.delete_ok:
            raise Exception("delete failed")
        self.deleted.append(name)


def _short_circuit_network(monkeypatch):
    # 短路受控 DNS 解析与 TCP/TLS 探测为成功（外部网络不可用），
    # 仅验证分阶段编排与对象清理逻辑。
    monkeypatch.setattr(ctest, "resolve_verified_address", lambda endpoint: "127.0.0.1")
    monkeypatch.setattr(ctest, "_probe_tcp", lambda *a, **k: None)


@pytest.fixture
def app(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'ct.db'}",
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
    exceptions.register_handlers(test_app)

    yield test_app

    test_app.dependency_overrides.clear()
    asyncio.run(engine.dispose())


def _body(**overrides):
    payload = {
        "endpoint": "https://127.0.0.1:9000",
        "bucket": "models",
        "prefix": "datamodel",
        "access_key": "AK",
        "secret_key": "SK",
        "tls_enabled": True,
        "tls_verify": True,
        "use_virtual_hosted_style": True,
    }
    payload.update(overrides)
    return payload


def test_connection_test_reports_scope_server_and_stages(app, monkeypatch):
    _short_circuit_network(monkeypatch)
    fake = _FakeClient()
    monkeypatch.setattr(
        model_storage, "_minio_client_factory", lambda **f: fake
    )
    client = TestClient(app)
    response = client.post(API, json=_body())
    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "server"
    for stage in ("connection", "bucket", "write", "read", "delete"):
        assert stage in body
        assert body[stage]["ok"] is True
    assert body["ok"] is True
    # 临时对象在当前 Prefix 下生成并在 finally 中清理。
    assert len(fake.put_object_names) == 1
    assert fake.put_object_names[0].startswith("datamodel/_connection-tests/")
    assert fake.deleted == fake.put_object_names
    # 凭据不进入响应体。
    assert "AK" not in response.text
    assert "SK" not in response.text


def test_connection_test_write_failure_is_stage_specific(app, monkeypatch):
    _short_circuit_network(monkeypatch)
    fake = _FakeClient(put_ok=False)
    monkeypatch.setattr(model_storage, "_minio_client_factory", lambda **f: fake)
    client = TestClient(app)
    response = client.post(API, json=_body())
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["bucket"]["ok"] is True
    assert body["write"]["ok"] is False
    assert body["write"]["error_code"] == "s3_write_failed"
    assert body["read"]["ok"] is False
    assert body["delete"]["ok"] is False


def test_connection_test_cleans_up_on_read_failure(app, monkeypatch):
    _short_circuit_network(monkeypatch)
    fake = _FakeClient(get_ok=False)
    monkeypatch.setattr(model_storage, "_minio_client_factory", lambda **f: fake)
    client = TestClient(app)
    response = client.post(API, json=_body())
    body = response.json()
    assert body["read"]["ok"] is False
    assert body["read"]["error_code"] == "s3_read_failed"
    # 即使读取失败，已写入的临时对象仍在 finally 中清理。
    assert fake.deleted == fake.put_object_names
    assert len(fake.deleted) == 1


def test_connection_test_delete_failure_reported_but_object_attempted(app, monkeypatch):
    _short_circuit_network(monkeypatch)
    fake = _FakeClient(delete_ok=False)
    monkeypatch.setattr(model_storage, "_minio_client_factory", lambda **f: fake)
    client = TestClient(app)
    response = client.post(API, json=_body())
    body = response.json()
    # 写成功、读成功，但删除失败：remove_object 仍被尝试（清理尽力而为），
    # 且未计入成功删除。
    assert fake.remove_attempts == 1
    assert fake.deleted == []
    assert len(fake.put_object_names) == 1


def test_connection_test_requires_admin(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'nadm.db'}",
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
    )

    async def session_override():
        async with AsyncSession(engine) as session:
            yield session

    async def current_user_override():
        return User(id=2, username="viewer", is_admin=False, hashed_password="")

    test_app.dependency_overrides[get_session] = session_override
    test_app.dependency_overrides[get_current_user] = current_user_override
    admin_router = APIRouter(dependencies=[Depends(get_admin_user)])
    admin_router.include_router(model_storage.router)
    test_app.include_router(admin_router, prefix="/v1")
    exceptions.register_handlers(test_app)

    with TestClient(test_app) as test_client:
        response = test_client.post(API, json=_body())

    asyncio.run(engine.dispose())
    assert response.status_code == 403


def test_connection_test_invalid_endpoint_is_422(app, monkeypatch):
    def _boom(**kwargs):
        pytest.fail("run should not be reached for invalid endpoint")

    monkeypatch.setattr(model_storage, "run_model_storage_connection_test", _boom)
    client = TestClient(app)
    response = client.post(API, json=_body(endpoint="ftp://s3.example.com"))
    assert response.status_code == 422


def test_resolve_rejects_link_local_and_metadata():
    # 云元数据地址（link-local）应被拒绝。
    with pytest.raises(ctest.ModelStorageConnectionError) as exc_info:
        ctest.resolve_verified_address("http://169.254.169.254:80")
    assert exc_info.value.code == "s3_forbidden_address"


def test_resolve_allows_rfc1918_and_loopback():
    # RFC1918 内网地址：允许。
    assert ctest.resolve_verified_address("http://10.0.0.5:9000") == "10.0.0.5"
    # loopback：允许（企业内网 MinIO / 本地测试）。
    assert ctest.resolve_verified_address("http://127.0.0.1:9000") == "127.0.0.1"
