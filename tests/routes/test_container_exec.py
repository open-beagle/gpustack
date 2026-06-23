import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from gpustack.api import exceptions
from gpustack.routes.admin import container_exec as admin_container_exec
from gpustack.routes.worker import container_exec
from gpustack.schemas.container_exec import ContainerExecCloseReason
from gpustack.schemas.workers import WorkerStateEnum
from gpustack.server.db import get_session


WORKER_TOKEN = "worker-internal-token"


@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.state.config = SimpleNamespace(token=WORKER_TOKEN)
    test_app.include_router(container_exec.router)
    exceptions.register_handlers(test_app)
    yield test_app
    for session in list(container_exec.manager.sessions.values()):
        container_exec.manager.close_session(
            session.session_id, ContainerExecCloseReason.INTERNAL_ERROR
        )


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def admin_app():
    test_app = FastAPI()
    test_app.state.config = SimpleNamespace(token=WORKER_TOKEN)
    test_app.state.server_config = SimpleNamespace(token=WORKER_TOKEN)
    test_app.state.container_exec_workers = [
        SimpleNamespace(
            id=1,
            name="gpu-node-01",
            hostname="gpu-node-01",
            ip="10.0.0.11",
            port=10150,
            state=WorkerStateEnum.READY,
            worker_uuid="ready-worker-uuid",
        ),
        SimpleNamespace(
            id=2,
            name="gpu-node-02",
            hostname="gpu-node-02",
            ip="10.0.0.12",
            port=10150,
            state=WorkerStateEnum.NOT_READY,
            worker_uuid="offline-worker-uuid",
        ),
    ]
    test_app.state.container_exec_forwarder = FakeWorkerForwarder()

    async def session_override():
        yield None

    test_app.dependency_overrides[get_session] = session_override
    test_app.include_router(
        admin_container_exec.router, prefix="/v1/admin/container-exec"
    )
    exceptions.register_handlers(test_app)
    yield test_app
    admin_container_exec.session_store.clear()


@pytest.fixture
def admin_client(admin_app):
    with TestClient(admin_app) as test_client:
        yield test_client


def auth_headers(token=WORKER_TOKEN):
    return {"Authorization": f"Bearer {token}"}


def create_session(client, session_id="cterm_route"):
    response = client.post(
        "/admin/container-exec/sessions",
        headers=auth_headers(),
        json={
            "session_id": session_id,
            "worker_id": 1,
            "worker_uuid": "worker-uuid",
            "cols": 80,
            "rows": 24,
        },
    )
    assert response.status_code == 200
    return response.json()


def bind_message(session_id, ticket):
    return {"op": "bind", "session_id": session_id, "ticket": ticket}


def admin_bind_message(session_id, bind_token):
    return {"op": "bind", "session_id": session_id, "bind_token": bind_token}


class FakeWorkerForwarder:
    def __init__(self):
        self.created_sessions = []
        self.closed_sessions = []
        self.fail_code = None

    async def create_session(self, worker, request):
        if self.fail_code:
            raise admin_container_exec.WorkerExecForwardError(
                self.fail_code, "worker is unreachable"
            )
        self.created_sessions.append((worker, request))
        return {
            "session_id": request["session_id"],
            "ticket": f"worker-ticket-{request['session_id']}",
            "state": "pending",
            "expires_at": request.get("expires_at"),
        }

    async def close_session(self, session):
        self.closed_sessions.append(session.session_id)

    async def pump_websocket(self, websocket, session):
        await websocket.send_text(json.dumps({"op": "bind", "state": "connected"}))
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                await websocket.send_bytes(b"remote:" + message["bytes"])
                continue
            if message.get("text") is None:
                continue
            control = json.loads(message["text"])
            if control.get("op") == "close":
                await websocket.send_text(
                    json.dumps(
                        {
                            "op": "close",
                            "reason": control.get("reason", "user_close"),
                        }
                    )
                )
                return
            if control.get("op") == "ping":
                await websocket.send_text(
                    json.dumps({"op": "pong", "ts": control.get("ts")})
                )


def test_create_session_requires_worker_internal_credential(client):
    payload = {"session_id": "cterm_auth", "cols": 80, "rows": 24}

    assert client.post("/admin/container-exec/sessions", json=payload).status_code == 401
    assert (
        client.post(
            "/admin/container-exec/sessions",
            headers=auth_headers("wrong-token"),
            json=payload,
        ).status_code
        == 401
    )


def test_create_session_registers_pending_without_pty(client):
    response = create_session(client)

    assert response["session_id"] == "cterm_route"
    assert response["ticket"]
    assert response["state"] == "pending"

    session = container_exec.manager.get_session("cterm_route")
    assert session is not None
    assert session.state == "pending"
    assert session.master_fd is None
    assert session.pid is None


def test_delete_session_is_idempotent_for_pending_session(client):
    create_session(client, "cterm_delete")

    first = client.delete(
        "/admin/container-exec/sessions/cterm_delete", headers=auth_headers()
    )
    second = client.delete(
        "/admin/container-exec/sessions/cterm_delete", headers=auth_headers()
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert container_exec.manager.get_session("cterm_delete") is None


def test_delete_session_closes_connected_shell(client):
    response = create_session(client, "cterm_delete_connected")
    ticket = response["ticket"]

    with client.websocket_connect(
        "/admin/container-exec/sessions/cterm_delete_connected/ws",
        headers=auth_headers(),
    ) as websocket:
        websocket.send_text(json.dumps(bind_message("cterm_delete_connected", ticket)))
        assert receive_control_until(websocket, "bind", timeout=3)["state"] == "connected"

        session = container_exec.manager.get_session("cterm_delete_connected")
        pid = session.pid
        delete_response = client.delete(
            "/admin/container-exec/sessions/cterm_delete_connected",
            headers=auth_headers(),
        )

        assert delete_response.status_code == 200
        wait_for_session_removed("cterm_delete_connected", timeout=3)

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_websocket_bind_rejects_expired_ticket(client):
    response = client.post(
        "/admin/container-exec/sessions",
        headers=auth_headers(),
        json={
            "session_id": "cterm_expired",
            "cols": 80,
            "rows": 24,
            "expires_at": "2000-01-01T00:00:00+00:00",
        },
    )
    assert response.status_code == 200
    ticket = response.json()["ticket"]

    with client.websocket_connect(
        "/admin/container-exec/sessions/cterm_expired/ws",
        headers=auth_headers(),
    ) as websocket:
        websocket.send_text(json.dumps(bind_message("cterm_expired", ticket)))
        message = json.loads(websocket.receive_text())

    assert message["op"] == "error"
    assert message["code"] == "bind_token_expired"
    assert container_exec.manager.get_session("cterm_expired") is None


def test_websocket_requires_worker_internal_credential(client):
    create_session(client, "cterm_ws_auth")

    with pytest.raises(Exception):
        with client.websocket_connect("/admin/container-exec/sessions/cterm_ws_auth/ws"):
            pass


def test_websocket_rejects_repeated_bind(client):
    response = create_session(client, "cterm_rebind")
    ticket = response["ticket"]

    with client.websocket_connect(
        "/admin/container-exec/sessions/cterm_rebind/ws",
        headers=auth_headers(),
    ) as websocket:
        websocket.send_text(json.dumps(bind_message("cterm_rebind", ticket)))
        assert json.loads(websocket.receive_text())["op"] == "bind"

        websocket.send_text(json.dumps(bind_message("cterm_rebind", ticket)))
        message = receive_control_until(websocket, "error", timeout=3)

    assert message["op"] == "error"
    assert message["code"] == "bind_token_reused"


def test_websocket_echo_returns_binary_output_and_disconnect_releases_shell(client):
    response = create_session(client, "cterm_echo")
    ticket = response["ticket"]

    with client.websocket_connect(
        "/admin/container-exec/sessions/cterm_echo/ws",
        headers=auth_headers(),
    ) as websocket:
        websocket.send_text(json.dumps(bind_message("cterm_echo", ticket)))
        assert json.loads(websocket.receive_text()) == {"op": "bind", "state": "connected"}

        session = container_exec.manager.get_session("cterm_echo")
        assert session is not None
        assert session.pid is not None
        pid = session.pid

        websocket.send_bytes(b"echo ok\n")
        output = receive_until(websocket, b"ok", timeout=3)
        assert b"ok" in output

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if container_exec.manager.get_session("cterm_echo") is None:
            break
        time.sleep(0.05)

    assert container_exec.manager.get_session("cterm_echo") is None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_websocket_control_messages_support_ping_resize_and_close(client):
    response = create_session(client, "cterm_control")
    ticket = response["ticket"]

    with client.websocket_connect(
        "/admin/container-exec/sessions/cterm_control/ws",
        headers=auth_headers(),
    ) as websocket:
        websocket.send_text(json.dumps(bind_message("cterm_control", ticket)))
        assert receive_control_until(websocket, "bind", timeout=3)["state"] == "connected"

        websocket.send_text(json.dumps({"op": "ping", "ts": 123}))
        assert receive_control_until(websocket, "pong", timeout=3)["ts"] == 123

        websocket.send_text(json.dumps({"op": "resize", "cols": 100, "rows": 40}))
        session = wait_for_session_size("cterm_control", cols=100, rows=40, timeout=3)
        assert session.cols == 100
        assert session.rows == 40

        websocket.send_text(json.dumps({"op": "close", "reason": "user_close"}))
        assert receive_control_until(websocket, "close", timeout=3)["reason"] == "user_close"


def test_admin_targets_returns_server_and_online_workers(admin_client):
    response = admin_client.get("/v1/admin/container-exec/targets")

    assert response.status_code == 200
    targets = response.json()["items"]

    server_target = next(target for target in targets if target["container_role"] == "server")
    assert server_target["status"] == "online"
    assert server_target["worker_id"] is None
    assert server_target["container_name"] == "gpustack"

    worker_targets = [
        target for target in targets if target["container_role"] == "worker"
    ]
    assert [target["worker_id"] for target in worker_targets] == [1]
    assert worker_targets[0]["worker_uuid"] == "ready-worker-uuid"
    assert worker_targets[0]["node_ip"] == "10.0.0.11"


def test_admin_container_exec_router_is_registered_in_v1_admin_router():
    routes_source = (
        Path(__file__).parents[2] / "gpustack" / "routes" / "routes.py"
    ).read_text()

    assert "gpustack.routes.admin" in routes_source
    assert "admin_container_exec.router" in routes_source
    assert 'prefix="/admin/container-exec"' in routes_source


def test_admin_create_session_validates_target_and_creates_remote_pending_session(
    admin_app, admin_client
):
    response = admin_client.post(
        "/v1/admin/container-exec/sessions",
        json={
            "session_id": "cterm_admin_create",
            "target_type": "worker",
            "worker_id": 1,
            "worker_uuid": "ready-worker-uuid",
            "cols": 100,
            "rows": 30,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "cterm_admin_create"
    assert body["bind_token"]
    assert body["state"] == "pending"

    forwarder = admin_app.state.container_exec_forwarder
    assert len(forwarder.created_sessions) == 1
    worker, request = forwarder.created_sessions[0]
    assert worker.id == 1
    assert request["session_id"] == "cterm_admin_create"
    assert request["worker_id"] == 1
    assert request["worker_uuid"] == "ready-worker-uuid"
    assert request["cols"] == 100
    assert request["rows"] == 30


def test_admin_create_session_rejects_offline_target(admin_client):
    response = admin_client.post(
        "/v1/admin/container-exec/sessions",
        json={
            "session_id": "cterm_admin_offline",
            "target_type": "worker",
            "worker_id": 2,
            "worker_uuid": "offline-worker-uuid",
            "cols": 80,
            "rows": 24,
        },
    )

    assert response.status_code == 400
    assert response.json()["message"] == "target_offline"


def test_admin_create_session_validates_worker_uuid(admin_client):
    response = admin_client.post(
        "/v1/admin/container-exec/sessions",
        json={
            "session_id": "cterm_admin_uuid",
            "target_type": "worker",
            "worker_id": 1,
            "worker_uuid": "stale-worker-uuid",
            "cols": 80,
            "rows": 24,
        },
    )

    assert response.status_code == 400
    assert response.json()["message"] == "target_offline"


def test_admin_create_session_maps_worker_unreachable_error(admin_app, admin_client):
    admin_app.state.container_exec_forwarder.fail_code = "worker_unreachable"

    response = admin_client.post(
        "/v1/admin/container-exec/sessions",
        json={
            "session_id": "cterm_admin_unreachable",
            "target_type": "worker",
            "worker_id": 1,
            "worker_uuid": "ready-worker-uuid",
            "cols": 80,
            "rows": 24,
        },
    )

    assert response.status_code == 503
    assert response.json()["message"] == "worker_unreachable"


def test_admin_delete_session_forwards_close_and_removes_session(admin_app, admin_client):
    response = admin_client.post(
        "/v1/admin/container-exec/sessions",
        json={
            "session_id": "cterm_admin_delete",
            "target_type": "worker",
            "worker_id": 1,
            "worker_uuid": "ready-worker-uuid",
            "cols": 80,
            "rows": 24,
        },
    )
    assert response.status_code == 200

    delete_response = admin_client.delete(
        "/v1/admin/container-exec/sessions/cterm_admin_delete"
    )
    second_delete_response = admin_client.delete(
        "/v1/admin/container-exec/sessions/cterm_admin_delete"
    )

    assert delete_response.status_code == 200
    assert second_delete_response.status_code == 200
    assert admin_app.state.container_exec_forwarder.closed_sessions == [
        "cterm_admin_delete"
    ]
    assert admin_container_exec.session_store.get("cterm_admin_delete") is None


def test_admin_websocket_binds_and_pumps_to_worker(admin_client):
    response = admin_client.post(
        "/v1/admin/container-exec/sessions",
        json={
            "session_id": "cterm_admin_ws",
            "target_type": "worker",
            "worker_id": 1,
            "worker_uuid": "ready-worker-uuid",
            "cols": 80,
            "rows": 24,
        },
    )
    bind_token = response.json()["bind_token"]

    with admin_client.websocket_connect(
        "/v1/admin/container-exec/sessions/cterm_admin_ws/ws"
    ) as websocket:
        websocket.send_text(json.dumps(admin_bind_message("cterm_admin_ws", bind_token)))
        assert receive_control_until(websocket, "bind", timeout=3)["state"] == "connected"

        websocket.send_bytes(b"echo ok\n")
        output = receive_until(websocket, b"remote:echo ok", timeout=3)
        assert b"remote:echo ok" in output

        websocket.send_text(json.dumps({"op": "close", "reason": "user_close"}))
        assert receive_control_until(websocket, "close", timeout=3)["reason"] == "user_close"


def test_admin_websocket_rejects_bad_bind_token(admin_client):
    response = admin_client.post(
        "/v1/admin/container-exec/sessions",
        json={
            "session_id": "cterm_admin_bad_bind",
            "target_type": "worker",
            "worker_id": 1,
            "worker_uuid": "ready-worker-uuid",
            "cols": 80,
            "rows": 24,
        },
    )
    assert response.status_code == 200

    with admin_client.websocket_connect(
        "/v1/admin/container-exec/sessions/cterm_admin_bad_bind/ws"
    ) as websocket:
        websocket.send_text(
            json.dumps(admin_bind_message("cterm_admin_bad_bind", "wrong-token"))
        )
        message = receive_control_until(websocket, "error", timeout=3)

    assert message["code"] == "auth_failed"


def receive_until(websocket, marker: bytes, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    chunks = []
    while time.monotonic() < deadline:
        message = websocket.receive()
        if message.get("bytes"):
            chunks.append(message["bytes"])
            data = b"".join(chunks)
            if marker in data:
                return data
        elif message.get("text"):
            control = json.loads(message["text"])
            if control.get("op") == "error":
                raise AssertionError(control)
    raise AssertionError(f"未在 WebSocket 输出中收到标记: {marker!r}")


def receive_control_until(websocket, op: str, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = websocket.receive()
        if message.get("text"):
            control = json.loads(message["text"])
            if control.get("op") == op:
                return control
    raise AssertionError(f"未在 WebSocket 控制消息中收到 op={op!r}")


def wait_for_session_size(session_id: str, cols: int, rows: int, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        session = container_exec.manager.get_session(session_id)
        if session and session.cols == cols and session.rows == rows:
            return session
        time.sleep(0.02)
    raise AssertionError("等待会话 resize 状态更新超时")


def wait_for_session_removed(session_id: str, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if container_exec.manager.get_session(session_id) is None:
            return
        time.sleep(0.02)
    raise AssertionError("等待会话删除超时")
