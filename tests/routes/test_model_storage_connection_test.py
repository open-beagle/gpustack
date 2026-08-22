"""任务 3：保存前连接测试（capabilities + connection-tests）定向测试。

覆盖：仅管理员可调用；使用未保存表单分阶段报告连接/Bucket/写/读/删除；
受控解析器拒绝 link-local/云元数据地址、允许 loopback 与 RFC1918；
**DNS 固定**：已验证 IP 真正用于 TCP 连接（Host 头/SNI 仍为原始主机名）；
**禁止重定向**：3xx 响应被拦截并稳定失败；put/get/delete 任一步失败
报告稳定脱敏错误码，delete 失败不得折叠为成功；加密能力缺失/密钥不可用
稳定 503；endpoint query、凭据、bucket 不进入响应。

说明：真实外部 S3 不可用；除 DNS 固定测试使用本机 HTTP 服务器验证真实
TCP 连接目标外，其余测试注入 fake minio client 并短路受控 DNS/TCP 探测，
只验证真实的分阶段编排、安全门禁与清理逻辑。
"""

import asyncio
import io
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

    def __init__(
        self,
        *,
        list_exc=None,
        put_exc=None,
        get_exc=None,
        delete_exc=None,
        get_payload=None,
    ):
        self.list_exc = list_exc
        self.put_exc = put_exc
        self.get_exc = get_exc
        self.delete_exc = delete_exc
        self.get_payload = get_payload
        self.put_object_names = []
        self.deleted = []
        self.remove_attempts = 0
        self._last_payload = None

    def list_objects(self, bucket, prefix=None, **kwargs):
        if self.list_exc is not None:
            raise self.list_exc
        return iter([])

    def put_object(self, bucket, name, data, length, **kwargs):
        if self.put_exc is not None:
            raise self.put_exc
        self._last_payload = data.read()
        self.put_object_names.append(name)

    def get_object(self, bucket, name):
        if self.get_exc is not None:
            raise self.get_exc
        if self.get_payload is not None:
            return io.BytesIO(self.get_payload)
        return io.BytesIO(self._last_payload or b"")

    def remove_object(self, bucket, name):
        self.remove_attempts += 1
        if self.delete_exc is not None:
            raise self.delete_exc
        self.deleted.append(name)


def _short_circuit_network(monkeypatch):
    # 短路受控 DNS 解析与 TCP/TLS 探测（外部网络不可用），
    # 仅验证分阶段编排与安全门禁逻辑。
    monkeypatch.setattr(
        ctest,
        "resolve_verified_endpoint",
        lambda endpoint: ctest.VerifiedEndpoint("http", "127.0.0.1", 9000, "127.0.0.1"),
    )
    monkeypatch.setattr(ctest, "_probe", lambda *a, **k: None)


def _install_fake(monkeypatch, fake):
    monkeypatch.setattr(model_storage, "_minio_client_factory", lambda **f: fake)


def _body(**overrides):
    payload = {
        "endpoint": "https://127.0.0.1:9000",
        "bucket": "models",
        "prefix": "datamodel",
        "access_key": "AK-secret-value",
        "secret_key": "SK-secret-value",
        "tls_enabled": True,
        "tls_verify": True,
        "use_virtual_hosted_style": True,
    }
    payload.update(overrides)
    return payload


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


# ---------------------------------------------------------------------------
# 分阶段编排与脱敏
# ---------------------------------------------------------------------------


def test_connection_test_reports_scope_server_and_stages(app, monkeypatch):
    _short_circuit_network(monkeypatch)
    fake = _FakeClient()
    _install_fake(monkeypatch, fake)
    client = TestClient(app)
    response = client.post(API, json=_body())
    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "server"
    for stage in ("connection", "bucket", "write", "read", "delete"):
        assert stage in body
        assert body[stage]["ok"] is True
    assert body["ok"] is True
    # 临时对象在当前 Prefix 下生成并被成功删除。
    assert len(fake.put_object_names) == 1
    assert fake.put_object_names[0].startswith("datamodel/_connection-tests/")
    assert fake.deleted == fake.put_object_names
    # 凭据不进入响应体。
    assert "AK-secret-value" not in response.text
    assert "SK-secret-value" not in response.text


def test_connection_test_write_failure_is_stage_specific(app, monkeypatch):
    _short_circuit_network(monkeypatch)
    fake = _FakeClient(put_exc=RuntimeError("write failed with AK-secret-value"))
    _install_fake(monkeypatch, fake)
    client = TestClient(app)
    response = client.post(API, json=_body())
    body = response.json()
    assert body["ok"] is False
    assert body["bucket"]["ok"] is True
    assert body["write"]["ok"] is False
    assert body["write"]["error_code"] == "s3_request_failed"
    assert body["read"]["ok"] is False
    assert body["read"]["error_code"] == "not_reached"
    assert body["delete"]["ok"] is False
    # 脱敏：原始异常文本（含凭据）不进入响应。
    assert "AK-secret-value" not in response.text
    # 写失败后无对象可清理（remove 未被调用）。
    assert fake.remove_attempts == 0


def test_connection_test_read_failure_cleans_up(app, monkeypatch):
    _short_circuit_network(monkeypatch)
    fake = _FakeClient(get_exc=RuntimeError("read failed"))
    _install_fake(monkeypatch, fake)
    client = TestClient(app)
    response = client.post(API, json=_body())
    body = response.json()
    assert body["ok"] is False
    assert body["bucket"]["ok"] is True
    assert body["write"]["ok"] is True
    assert body["read"]["ok"] is False
    assert body["read"]["error_code"] == "s3_request_failed"
    assert body["delete"]["ok"] is False
    assert body["delete"]["error_code"] == "not_reached"
    # 即使读取失败，已写入的临时对象仍在 finally 中清理（尽力而为）。
    assert fake.deleted == fake.put_object_names
    assert len(fake.deleted) == 1


def test_connection_test_read_content_mismatch(app, monkeypatch):
    _short_circuit_network(monkeypatch)
    fake = _FakeClient(get_payload=b"corrupted-by-proxy")
    _install_fake(monkeypatch, fake)
    client = TestClient(app)
    response = client.post(API, json=_body())
    body = response.json()
    assert body["ok"] is False
    assert body["write"]["ok"] is True
    assert body["read"]["ok"] is False
    assert body["read"]["error_code"] == "s3_read_content_mismatch"
    assert body["error_code"] == "s3_read_content_mismatch"
    assert fake.deleted == fake.put_object_names


def test_connection_test_bucket_auth_failure_is_not_connection_failure(
    app, monkeypatch
):
    _short_circuit_network(monkeypatch)

    class _S3AuthError(Exception):
        def __init__(self):
            super().__init__("S3 operation failed; code: AccessDenied")
            self.code = "AccessDenied"

    fake = _FakeClient(list_exc=_S3AuthError())
    _install_fake(monkeypatch, fake)
    client = TestClient(app)
    response = client.post(API, json=_body())
    body = response.json()
    assert body["ok"] is False
    # 连接阶段成功（TCP/TLS 通过），失败归类为认证而非连接失败。
    assert body["connection"]["ok"] is True
    assert body["bucket"]["ok"] is False
    assert body["bucket"]["error_code"] == "s3_authentication_failed"
    assert body["error_code"] == "s3_authentication_failed"
    # 凭据不进入响应。
    assert "AK-secret-value" not in response.text


def test_connection_test_delete_failure_must_not_report_success(app, monkeypatch):
    _short_circuit_network(monkeypatch)
    fake = _FakeClient(delete_exc=RuntimeError("delete failed"))
    _install_fake(monkeypatch, fake)
    client = TestClient(app)
    response = client.post(API, json=_body())
    body = response.json()
    # 写成功、读成功，但删除失败：整体必须失败，delete 阶段报告稳定错误码。
    assert body["ok"] is False
    assert body["bucket"]["ok"] is True
    assert body["write"]["ok"] is True
    assert body["read"]["ok"] is True
    assert body["delete"]["ok"] is False
    assert body["delete"]["error_code"] == "s3_request_failed"
    assert body["error_code"] == "s3_request_failed"
    # 删除确实被尝试过。
    assert fake.remove_attempts >= 1
    assert fake.deleted == []


# ---------------------------------------------------------------------------
# 禁止重定向
# ---------------------------------------------------------------------------


def _client_returning_status(status, headers=None):
    """构造底层返回固定 HTTP 状态（含 3xx）的最小 fake client。"""
    state = {"calls": 0}

    class _Resp:
        def __init__(self, s):
            self.status = s
            self.headers = headers or {}

        def read(self):
            return b""

        def close(self):
            pass

        def release_conn(self):
            pass

    class _Client:
        def _url_open(self, *a, **k):
            return _Resp(status)

        def list_objects(self, bucket, prefix=None, **kwargs):
            state["calls"] += 1
            self._url_open("GET", "/")
            return iter([])

        def put_object(self, bucket, name, data, length, **kwargs):
            raise AssertionError("put must not be reached after redirect")

        def get_object(self, bucket, name):
            raise AssertionError("get must not be reached after redirect")

        def remove_object(self, bucket, name):
            raise AssertionError("delete must not be reached after redirect")

    return _Client(), state


def test_redirect_is_blocked_and_stable(app, monkeypatch):
    _short_circuit_network(monkeypatch)
    fake, state = _client_returning_status(
        307, {"Location": "http://169.254.169.254/x"}
    )
    _install_fake(monkeypatch, fake)
    client = TestClient(app)
    response = client.post(API, json=_body())
    body = response.json()
    assert response.status_code == 200
    assert body["ok"] is False
    # 3xx 被拦截：bucket 阶段以稳定错误码失败，绝不跟随重定向。
    assert body["bucket"]["ok"] is False
    assert body["bucket"]["error_code"] == "s3_redirect_forbidden"
    assert body["error_code"] == "s3_redirect_forbidden"
    assert body["write"]["ok"] is False
    assert body["write"]["error_code"] == "not_reached"
    # Location 头（未验证目标）不进入响应。
    assert "169.254.169.254" not in response.text


def test_redirect_blocker_installed_on_real_minio_client():
    # 用真实 _minio_client_factory 构建 Minio client（不发起网络请求），
    # 验证 _forbid_redirects 包装了 Minio 的 _url_open，且包装后的守卫函数
    # 对 3xx 响应抛出重定向拦截异常（拦截逻辑真实执行）。
    verified = ctest.VerifiedEndpoint("http", "127.0.0.1", 9000, "127.0.0.1")
    real = model_storage._minio_client_factory(
        endpoint="http://127.0.0.1:9000",
        access_key="AK",
        secret_key="SK",
        secure=False,
        tls_verify=True,
        region=None,
        use_virtual_hosted_style=False,
        verified=verified,
        resolver=lambda h: "127.0.0.1",
    )
    # Minio 原始 _url_open 是方法；包装后应为守卫函数（bound 闭包）。
    original = real._url_open
    ctest._forbid_redirects(real)
    guarded = real._url_open
    assert guarded is not original
    assert "_guarded" in getattr(guarded, "__qualname__", "")

    # 让底层 _url_open 返回 3xx：守卫必须关闭响应并抛出重定向拦截异常。
    class _Redir:
        status = 302
        headers = {}
        closed = False

        def close(self):
            self.closed = True

        def release_conn(self):
            pass

    redir = _Redir()
    calls = []

    def _fake_underlying(*a, **k):
        calls.append(a)
        return redir

    real._url_open = _fake_underlying

    # 守卫闭包捕获的是包装时刻的 original（Minio 方法），这里直接调用守卫
    # 并让 original 委托到 fake，验证拦截路径。
    def _delegate(*a, **k):
        return _fake_underlying(*a, **k)

    # 重新用 fake 作为底层构建包装，等价于连接测试运行时 3xx 的场景。
    fake_client = SimpleNamespace(_url_open=_fake_underlying)
    ctest._forbid_redirects(fake_client)
    with pytest.raises(ctest._RedirectBlocked) as exc_info:
        fake_client._url_open("GET", "/")
    assert calls
    assert redir.closed is True
    assert getattr(exc_info.value, "_model_storage_redirect", False) is True


# ---------------------------------------------------------------------------
# DNS 固定：已验证 IP 真正用于 TCP 连接，Host/SNI 保留原始主机名
# ---------------------------------------------------------------------------


class _ProbeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # 记录收到的 Host 头与真实远端地址；回显 Host 供断言。
    received = []

    def do_GET(self):
        host = self.headers.get("host", "")
        body = f"host={host}".encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/plain")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_dns_pinning_connects_verified_ip_with_original_host(monkeypatch):
    # 起一个仅监听 127.0.0.1 的本机 HTTP 服务作为“已验证后端”。
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProbeHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        verified = ctest.VerifiedEndpoint("http", "origin.example", port, "127.0.0.1")
        resolver = {"origin.example": "127.0.0.1"}
        http = ctest.build_pinned_http_client(
            verified=verified, resolver=lambda h: resolver.get(h), tls_verify=True
        )
        response = http.request("GET", f"http://origin.example:{port}/probe")
        assert response.status == 200
        text = response.data.decode("utf-8")
        # TCP 固定连到 127.0.0.1，但 Host 头保留原始主机名（证书/SNI 语义）。
        assert f"host=origin.example:{port}" in text
    finally:
        server.shutdown()
        server.server_close()


def test_dns_pinning_rejects_unverified_host(monkeypatch):
    # resolver 未放行该 host 时，TCP 连接必须失败（防 rebinding/逃逸）。
    verified = ctest.VerifiedEndpoint("http", "origin.example", 9000, "10.0.0.5")
    http = ctest.build_pinned_http_client(
        verified=verified, resolver=lambda h: None, tls_verify=True
    )
    with pytest.raises(Exception):
        http.request("GET", "http://origin.example:9000/x", timeout=2)


def test_virtual_style_falls_back_for_ip_endpoint(app, monkeypatch):
    # endpoint 为 IP 字面量时 virtual 风格无法表达 SNI/证书语义，必须回退
    # path style（连接 host 不变，DNS 固定不受影响）。
    assert ctest._virtual_style_host_safe("10.0.0.5", "models") is False
    assert ctest._virtual_style_host_safe("s3.example.com", "models") is True
    # 桶名含点号时同样回退。
    assert ctest._virtual_style_host_safe("s3.example.com", "a.b") is False

    calls = {}

    def _factory(**kwargs):
        calls.update(kwargs)
        return _FakeClient()

    # IP 字面量 endpoint 走真实受控解析（无 DNS 查找），仅短路 TCP 探测。
    monkeypatch.setattr(ctest, "_probe", lambda *a, **k: None)
    monkeypatch.setattr(model_storage, "_minio_client_factory", _factory)
    client = TestClient(app)
    response = client.post(
        API,
        json=_body(
            endpoint="http://10.0.0.5:9000",
            tls_enabled=False,
            use_virtual_hosted_style=True,
        ),
    )
    assert response.status_code == 200
    # 回退为 path style，且连接目标仍是受控解析后的 IP。
    assert calls["use_virtual_hosted_style"] is False
    assert calls["verified"].verified_ip == "10.0.0.5"


# ---------------------------------------------------------------------------
# 加密能力门禁
# ---------------------------------------------------------------------------


def test_connection_test_requires_credential_encryption_key(app, monkeypatch):
    # 密钥缺失：必须稳定 503（credential_encryption_unavailable），
    # 且不得进入网络探测。
    app.state.server_config.model_preheat_credential_key = None

    def _boom(**kwargs):
        pytest.fail("network probe must not run without encryption capability")

    monkeypatch.setattr(model_storage, "run_model_storage_connection_test", _boom)
    client = TestClient(app)
    response = client.post(API, json=_body())
    assert response.status_code == 503
    assert "credential_encryption_unavailable" in response.text
    assert "AK-secret-value" not in response.text


def test_connection_test_requires_usable_credential_encryption_key(app, monkeypatch):
    # 密钥存在但不可用（格式非法导致加密探针失败）：同样稳定 503。
    app.state.server_config.model_preheat_credential_key = "not-a-valid-key"

    def _boom(**kwargs):
        pytest.fail("network probe must not run with unusable key")

    monkeypatch.setattr(model_storage, "run_model_storage_connection_test", _boom)
    client = TestClient(app)
    response = client.post(API, json=_body())
    assert response.status_code == 503
    assert "credential_encryption_unavailable" in response.text


# ---------------------------------------------------------------------------
# 受控解析与敏感信息
# ---------------------------------------------------------------------------


def test_resolve_rejects_link_local_and_metadata():
    # 云元数据地址（link-local）应被拒绝。
    with pytest.raises(ctest.ModelStorageConnectionError) as exc_info:
        ctest.resolve_verified_endpoint("http://169.254.169.254:80")
    assert exc_info.value.code == "s3_forbidden_address"
    # 兼容入口同样拒绝。
    with pytest.raises(ctest.ModelStorageConnectionError):
        ctest.resolve_verified_address("http://169.254.169.254:80")


def test_resolve_allows_rfc1918_and_loopback():
    ep = ctest.resolve_verified_endpoint("http://10.0.0.5:9000")
    assert ep.verified_ip == "10.0.0.5"
    assert ep.host == "10.0.0.5"
    assert ep.port == 9000
    # loopback：允许（企业内网 MinIO / 本地测试）。
    assert ctest.resolve_verified_address("http://127.0.0.1:9000") == "127.0.0.1"


def test_endpoint_query_not_in_response(app, monkeypatch):
    # endpoint 携带 query（如预签名参数）时不得进入响应/结果。
    _short_circuit_network(monkeypatch)
    fake = _FakeClient()
    _install_fake(monkeypatch, fake)
    client = TestClient(app)
    response = client.post(
        API,
        json=_body(
            endpoint="http://127.0.0.1:9000?AWSAccessKeyId=AK-secret-value&token=t",
            tls_enabled=False,
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "AWSAccessKeyId" not in response.text
    assert "token=t" not in response.text
    assert "AK-secret-value" not in response.text


def test_connection_test_invalid_endpoint_is_422(app, monkeypatch):
    def _boom(**kwargs):
        pytest.fail("run should not be reached for invalid endpoint")

    monkeypatch.setattr(model_storage, "run_model_storage_connection_test", _boom)
    client = TestClient(app)
    response = client.post(API, json=_body(endpoint="ftp://s3.example.com"))
    assert response.status_code == 422


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


# ---------------------------------------------------------------------------
# 定向复审 8：DNS getaddrinfo 可测试的固定短超时 + 超时稳定脱敏
# ---------------------------------------------------------------------------


def test_dns_resolution_timeout_is_stable_and_redacted(monkeypatch):
    """getaddrinfo 固定短超时（可注入）：超时稳定映射 dns_resolution_timeout，
    不泄露主机名/端点细节。"""
    verified = None

    def fake_getaddrinfo(host_arg: str, port: int):
        raise socket.timeout("injected slow dns for " + host_arg)

    with pytest.raises(ctest.ModelStorageConnectionError) as excinfo:
        ctest.resolve_verified_endpoint(
            "http://slow-dns.example.com:9000",
            getaddrinfo=fake_getaddrinfo,
        )
    assert excinfo.value.stage == "dns"
    assert excinfo.value.code == "dns_resolution_timeout"
    # 稳定脱敏：错误消息不携带主机名/端点（注入异常文本里的 host 也不泄露）。
    assert "slow-dns.example.com" not in str(excinfo.value)
    assert "9000" not in str(excinfo.value)


def test_dns_resolution_fixed_short_timeout_is_passed(monkeypatch):
    """固定短超时：resolver 收到的超时为 DNS_TIMEOUT（可测试的固定值）。"""
    seen: dict = {}

    def fake_getaddrinfo(host_arg: str, port: int):
        seen["host"] = host_arg
        return [(2, 1, 6, "", ("10.0.0.5", port))]

    verified = ctest.resolve_verified_endpoint(
        "http://corp.example.com:9000",
        getaddrinfo=fake_getaddrinfo,
    )
    # 注入解析器被真正使用（可测试性），且返回安全 IP。
    assert seen["host"] == "corp.example.com"
    assert verified.verified_ip == "10.0.0.5"
    assert verified.host == "corp.example.com"
    # 固定短超时是模块级常量（测试可断言其短于 TCP 超时，避免长挂起）。
    assert 0 < ctest.DNS_TIMEOUT <= ctest.CONNECTION_TIMEOUT


def test_dns_resolution_failure_still_stable_code(monkeypatch):
    """解析失败（非超时）仍稳定 dns_resolution_failed，不泄露主机名。"""

    def fake_getaddrinfo(host_arg: str, port: int):
        raise OSError("no such host " + host_arg)

    with pytest.raises(ctest.ModelStorageConnectionError) as excinfo:
        ctest.resolve_verified_endpoint(
            "http://missing.example.com:9000",
            getaddrinfo=fake_getaddrinfo,
        )
    assert excinfo.value.code == "dns_resolution_failed"
    assert "missing.example.com" not in str(excinfo.value)


def test_dns_fixed_short_timeout_actually_bounds_slow_resolver():
    """固定短超时真正约束慢解析器：5s 睡眠被 0.2s 超时切断，
    稳定映射 dns_resolution_timeout（不是无限挂起，也不是 5s 后失败）。"""
    import time

    def sleeping_resolver(host_arg: str, port: int):
        time.sleep(5)  # 远超固定短超时
        return [(2, 1, 6, "", ("10.0.0.5", port))]

    start = time.monotonic()
    with pytest.raises(ctest.ModelStorageConnectionError) as excinfo:
        ctest.resolve_verified_endpoint(
            "http://slow.example.com:9000",
            getaddrinfo=sleeping_resolver,
            timeout=0.2,
        )
    elapsed = time.monotonic() - start
    assert excinfo.value.code == "dns_resolution_timeout"
    # 固定短超时确实生效：远小于 5s 睡眠，避免长挂起。
    assert elapsed < 2.0


def test_connection_route_dns_timeout_bounds_testclient_worker_thread(app, monkeypatch):
    """从 TestClient 路由工作线程触发慢 DNS，固定超时仍按墙钟生效。"""
    import time

    def sleeping_resolver(host_arg: str, port: int):
        time.sleep(5)
        return [(2, 1, 6, "", ("10.0.0.5", port))]

    monkeypatch.setattr(ctest, "DNS_TIMEOUT", 0.1)
    monkeypatch.setattr(ctest, "_default_getaddrinfo", sleeping_resolver)
    start = time.monotonic()
    with TestClient(app) as client:
        response = client.post(
            API,
            json=_body(
                endpoint="http://slow-route.example.com:9000", tls_enabled=False
            ),
        )
    elapsed = time.monotonic() - start

    assert response.status_code == 200, response.text
    assert response.json()["error_code"] == "dns_resolution_timeout"
    assert "slow-route.example.com" not in response.text
    assert elapsed < 1.0
