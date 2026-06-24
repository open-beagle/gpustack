import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import secrets
from typing import Any, Optional

import aiohttp
from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field, field_validator
from sqlmodel import select

from gpustack.api.exceptions import (
    BadRequestException,
    ServiceUnavailableException,
)
from gpustack.routes.worker import container_exec as local_container_exec
from gpustack.schemas.container_exec import ContainerExecCloseReason
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import get_session


DEFAULT_BIND_TIMEOUT_SECONDS = 60
DEFAULT_CONTAINER_NAME = "gpustack"

router = APIRouter()


class AdminContainerExecCreateRequest(BaseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    target_type: str = Field(default="worker")
    worker_id: Optional[int] = None
    worker_uuid: Optional[str] = None
    cols: int = Field(default=80, ge=1, le=1000)
    rows: int = Field(default=24, ge=1, le=1000)
    expires_at: Optional[datetime] = None

    @field_validator("expires_at")
    @classmethod
    def normalize_expires_at(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expires_at must include timezone")
        return value.astimezone(timezone.utc)


class AdminContainerExecSession(BaseModel):
    session_id: str
    target_type: str
    worker_id: Optional[int] = None
    worker_uuid: Optional[str] = None
    bind_token_hash: Optional[str] = None
    worker_ticket: Optional[str] = None
    state: str = "pending"
    expires_at: datetime
    target: dict[str, Any] = Field(default_factory=dict)


class AdminContainerExecSessionStore:
    def __init__(self):
        self._sessions: dict[str, AdminContainerExecSession] = {}

    def create(
        self,
        request: AdminContainerExecCreateRequest,
        target: dict[str, Any],
        worker_ticket: str,
    ) -> tuple[AdminContainerExecSession, str]:
        if request.session_id in self._sessions:
            raise BadRequestException(message="session_already_exists")

        bind_token = secrets.token_urlsafe(32)
        expires_at = request.expires_at or datetime.now(timezone.utc) + timedelta(
            seconds=DEFAULT_BIND_TIMEOUT_SECONDS
        )
        session = AdminContainerExecSession(
            session_id=request.session_id,
            target_type=request.target_type,
            worker_id=request.worker_id,
            worker_uuid=request.worker_uuid,
            bind_token_hash=self._hash_token(bind_token),
            worker_ticket=worker_ticket,
            expires_at=expires_at,
            target=target,
        )
        self._sessions[session.session_id] = session
        return session, bind_token

    def get(self, session_id: str) -> Optional[AdminContainerExecSession]:
        return self._sessions.get(session_id)

    def bind(self, session_id: str, bind_token: str) -> AdminContainerExecSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise BadRequestException(message="session_not_found")
        if session.state != "pending" or not session.bind_token_hash:
            raise BadRequestException(message="auth_failed")
        if datetime.now(timezone.utc) >= session.expires_at:
            self._sessions.pop(session_id, None)
            raise BadRequestException(message="bind_token_expired")
        if not secrets.compare_digest(
            session.bind_token_hash, self._hash_token(bind_token)
        ):
            raise BadRequestException(message="auth_failed")

        session.bind_token_hash = None
        session.state = "connected"
        return session

    def remove(self, session_id: str) -> Optional[AdminContainerExecSession]:
        return self._sessions.pop(session_id, None)

    def clear(self):
        self._sessions.clear()

    def _hash_token(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()


class WorkerExecForwardError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class WorkerExecForwarder:
    async def create_session(self, worker: Any, request: dict[str, Any]) -> dict:
        url = self._worker_http_url(worker, "/admin/container-exec/sessions")
        timeout = aiohttp.ClientTimeout(total=10, sock_connect=5)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as client:
            try:
                async with client.post(
                    url,
                    json=request,
                    headers=self._auth_headers(worker),
                ) as response:
                    if response.status == status.HTTP_400_BAD_REQUEST:
                        error = await self._read_error(response)
                        raise WorkerExecForwardError(error, error)
                    if response.status == status.HTTP_401_UNAUTHORIZED:
                        raise WorkerExecForwardError("auth_failed", "worker auth failed")
                    if response.status >= status.HTTP_400_BAD_REQUEST:
                        raise WorkerExecForwardError(
                            "worker_unreachable", "worker returned error"
                        )
                    return await response.json()
            except aiohttp.ClientError as exc:
                raise WorkerExecForwardError(
                    "worker_unreachable", "failed to reach worker"
                ) from exc

    async def close_session(self, session: AdminContainerExecSession) -> None:
        worker = _target_worker_namespace(session.target)
        url = self._worker_http_url(
            worker, f"/admin/container-exec/sessions/{session.session_id}"
        )
        timeout = aiohttp.ClientTimeout(total=10, sock_connect=5)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as client:
            try:
                async with client.delete(url, headers=self._auth_headers(worker)) as response:
                    if response.status in (
                        status.HTTP_401_UNAUTHORIZED,
                        status.HTTP_403_FORBIDDEN,
                    ):
                        raise WorkerExecForwardError("auth_failed", "worker auth failed")
                    if response.status >= status.HTTP_400_BAD_REQUEST:
                        raise WorkerExecForwardError(
                            "worker_unreachable", "worker returned close error"
                        )
                    return
            except aiohttp.ClientError as exc:
                raise WorkerExecForwardError(
                    "worker_unreachable", "failed to close worker session"
                ) from exc

    async def pump_websocket(
        self, websocket: WebSocket, session: AdminContainerExecSession
    ) -> None:
        worker = _target_worker_namespace(session.target)
        url = self._worker_ws_url(
            worker, f"/admin/container-exec/sessions/{session.session_id}/ws"
        )
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=5)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as client:
            try:
                async with client.ws_connect(
                    url,
                    headers=self._auth_headers(worker),
                ) as worker_ws:
                    await worker_ws.send_str(
                        json.dumps(
                            {
                                "op": "bind",
                                "session_id": session.session_id,
                                "ticket": session.worker_ticket,
                            },
                            separators=(",", ":"),
                        )
                    )
                    await _pump_websockets(websocket, worker_ws)
            except aiohttp.ClientError as exc:
                raise WorkerExecForwardError(
                    "worker_unreachable", "failed to connect worker websocket"
                ) from exc

    def _auth_headers(self, worker: Any) -> dict[str, str]:
        token = worker.token
        return {"Authorization": f"Bearer {token}"}

    def _worker_http_url(self, worker: Any, path: str) -> str:
        return f"http://{worker.ip}:{worker.port}{path}"

    def _worker_ws_url(self, worker: Any, path: str) -> str:
        return f"ws://{worker.ip}:{worker.port}{path}"

    async def _read_error(self, response: aiohttp.ClientResponse) -> str:
        try:
            body = await response.json()
            return body.get("message") or body.get("reason") or "worker_unreachable"
        except Exception:
            return "worker_unreachable"


session_store = AdminContainerExecSessionStore()
default_forwarder = WorkerExecForwarder()


@router.get("/targets")
async def list_targets(request: Request, session: Any = Depends(get_session)):
    workers = await _list_workers(request, session)
    return {"items": [_server_target()] + [_worker_target(w) for w in workers]}


@router.post("/sessions")
async def create_session(
    request: Request,
    create_request: AdminContainerExecCreateRequest,
    session: Any = Depends(get_session),
):
    try:
        target, worker = await _resolve_target(request, session, create_request)
        if create_request.target_type == "server":
            remote_response = await _create_local_server_session(create_request)
        else:
            remote_response = await _forwarder(request).create_session(
                worker, _worker_create_payload(create_request)
            )
    except WorkerExecForwardError as exc:
        _raise_forward_error(exc.code)

    remote_ticket = remote_response.get("ticket")
    if not remote_ticket:
        raise ServiceUnavailableException(message="worker_unreachable")

    try:
        admin_session, bind_token = session_store.create(
            create_request, target, remote_ticket
        )
    except Exception:
        if create_request.target_type == "server":
            local_container_exec.manager.close_session(
                create_request.session_id, ContainerExecCloseReason.INTERNAL_ERROR
            )
        else:
            await _forwarder(request).close_session(
                AdminContainerExecSession(
                    session_id=create_request.session_id,
                    target_type=create_request.target_type,
                    worker_id=create_request.worker_id,
                    worker_uuid=create_request.worker_uuid,
                    expires_at=create_request.expires_at or datetime.now(timezone.utc),
                    target=target,
                )
            )
        raise
    return {
        "session_id": admin_session.session_id,
        "bind_token": bind_token,
        "state": admin_session.state,
        "expires_at": admin_session.expires_at,
        "ws_url": f"/v1/admin/container-exec/sessions/{admin_session.session_id}/ws",
    }


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str):
    session = session_store.remove(session_id)
    if session is None:
        return {"session_id": session_id, "state": "closed"}

    try:
        if session.target_type == "server":
            local_container_exec.manager.close_session(
                session_id, ContainerExecCloseReason.USER_CLOSE
            )
        else:
            await _forwarder(request).close_session(session)
    except WorkerExecForwardError:
        pass

    return {"session_id": session_id, "state": "closed"}


@router.websocket("/sessions/{session_id}/ws")
async def websocket_session(websocket: WebSocket, session_id: str):
    await websocket.accept()
    bound = False
    try:
        bind_message = await websocket.receive_text()
        payload = json.loads(bind_message)
        if payload.get("op") != "bind" or payload.get("session_id") != session_id:
            await _send_ws_error(websocket, "auth_failed", "invalid bind request")
            return
        try:
            session = session_store.bind(session_id, payload.get("bind_token") or "")
        except BadRequestException as exc:
            await _send_ws_error(websocket, exc.message, exc.message)
            return

        try:
            bound = True
            if session.target_type == "server":
                await _pump_local_server(websocket, session)
            else:
                await _forwarder(websocket).pump_websocket(websocket, session)
        except WorkerExecForwardError as exc:
            await _send_ws_error(websocket, exc.code, exc.message)
    except (WebSocketDisconnect, json.JSONDecodeError):
        return
    finally:
        if bound:
            session_store.remove(session_id)


async def _list_workers(request: Request, session: Any) -> list[Any]:
    injected_workers = getattr(request.app.state, "container_exec_workers", None)
    if injected_workers is not None:
        return list(injected_workers)
    if session is None:
        return []
    result = await session.exec(select(Worker))
    return list(result.all())


async def _resolve_target(
    request: Request, session: Any, create_request: AdminContainerExecCreateRequest
) -> tuple[dict[str, Any], Any]:
    if create_request.target_type == "server":
        return _server_target(), None
    if create_request.target_type != "worker":
        raise BadRequestException(message="target_offline")
    if create_request.worker_id is None or not create_request.worker_uuid:
        raise BadRequestException(message="target_offline")

    workers = await _list_workers(request, session)
    for worker in workers:
        if (
            worker.id == create_request.worker_id
            and worker.worker_uuid == create_request.worker_uuid
            and _is_online(worker)
        ):
            target = _worker_target(worker)
            target["token"] = _worker_token(request)
            return target, _worker_namespace(worker, _worker_token(request))
    raise BadRequestException(message="target_offline")


async def _create_local_server_session(
    create_request: AdminContainerExecCreateRequest,
) -> dict[str, Any]:
    from gpustack.schemas.container_exec import ContainerExecCreateRequest

    session, ticket = local_container_exec.manager.create_session(
        ContainerExecCreateRequest(
            session_id=create_request.session_id,
            worker_id=None,
            worker_uuid=None,
            cols=create_request.cols,
            rows=create_request.rows,
            expires_at=create_request.expires_at,
        )
    )
    return {
        "session_id": session.session_id,
        "ticket": ticket,
        "state": session.state,
        "expires_at": session.expires_at,
    }


async def _pump_local_server(
    websocket: WebSocket, session: AdminContainerExecSession
) -> None:
    send_lock = asyncio.Lock()
    await local_container_exec._bind_session(
        websocket,
        session.session_id,
        {
            "session_id": session.session_id,
            "ticket": session.worker_ticket,
        },
        send_lock,
    )
    local_session = local_container_exec.manager.get_session(session.session_id)
    if not local_session or local_session.state != "connected":
        return

    output_task = asyncio.create_task(
        local_container_exec._pump_output(websocket, session.session_id, send_lock)
    )
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                local_container_exec.manager.write_stdin(
                    session.session_id, message["bytes"]
                )
                continue
            if message.get("text") is not None:
                should_close = await local_container_exec._handle_control_message(
                    websocket, session.session_id, message["text"], send_lock
                )
                if should_close:
                    break
    finally:
        output_task.cancel()
        await asyncio.gather(output_task, return_exceptions=True)
        local_container_exec.manager.close_session(
            session.session_id, ContainerExecCloseReason.TARGET_LOST
        )


async def _pump_websockets(
    websocket: WebSocket, worker_ws: aiohttp.ClientWebSocketResponse
):
    async def client_to_worker():
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                await worker_ws.close()
                return
            if message.get("bytes") is not None:
                await worker_ws.send_bytes(message["bytes"])
            elif message.get("text") is not None:
                await worker_ws.send_str(message["text"])

    async def worker_to_client():
        async for message in worker_ws:
            if message.type == aiohttp.WSMsgType.BINARY:
                await websocket.send_bytes(message.data)
            elif message.type == aiohttp.WSMsgType.TEXT:
                await websocket.send_text(message.data)
            elif message.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            ):
                return

    tasks = [
        asyncio.create_task(client_to_worker()),
        asyncio.create_task(worker_to_client()),
    ]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*done, *pending, return_exceptions=True)


def _forwarder(request_or_websocket: Any):
    app = request_or_websocket.app
    return getattr(app.state, "container_exec_forwarder", default_forwarder)


def _worker_create_payload(
    create_request: AdminContainerExecCreateRequest,
) -> dict[str, Any]:
    payload = create_request.model_dump(mode="json", exclude_none=True)
    payload.pop("target_type", None)
    return payload


def _server_target() -> dict[str, Any]:
    return {
        "target_type": "server",
        "node_id": "server",
        "worker_id": None,
        "worker_uuid": None,
        "worker_name": "server",
        "node_ip": "127.0.0.1",
        "container_role": "server",
        "container_name": DEFAULT_CONTAINER_NAME,
        "risk_level": "normal",
        "risk_flags": ["unknown"],
        "status": "online",
    }


def _worker_target(worker: Any) -> dict[str, Any]:
    return {
        "target_type": "worker",
        "node_id": str(worker.id),
        "worker_id": worker.id,
        "worker_uuid": worker.worker_uuid,
        "worker_name": worker.name,
        "node_ip": worker.ip,
        "container_role": "worker",
        "container_name": DEFAULT_CONTAINER_NAME,
        "risk_level": "normal",
        "risk_flags": ["unknown"],
        "status": _worker_status(worker),
        "port": worker.port,
    }


def _is_online(worker: Any) -> bool:
    return worker.state == WorkerStateEnum.READY


def _worker_status(worker: Any) -> str:
    if worker.state == WorkerStateEnum.READY:
        return "online"
    if worker.state == WorkerStateEnum.UNREACHABLE:
        return "unreachable"
    return "offline"


def _worker_namespace(worker: Any, token: str) -> Any:
    return _target_worker_namespace(
        {
            "worker_id": worker.id,
            "worker_uuid": worker.worker_uuid,
            "node_ip": worker.ip,
            "port": worker.port,
            "token": token,
        }
    )


def _target_worker_namespace(target: dict[str, Any]) -> Any:
    return type(
        "ContainerExecWorkerTarget",
        (),
        {
            "id": target.get("worker_id"),
            "worker_uuid": target.get("worker_uuid"),
            "ip": target.get("node_ip"),
            "port": target.get("port"),
            "token": target.get("token"),
        },
    )()


def _worker_token(request: Request) -> str:
    config = getattr(request.app.state, "server_config", None) or getattr(
        request.app.state, "config", None
    )
    return getattr(config, "token", None)


def _raise_forward_error(code: str):
    if code == "shell_not_found":
        raise BadRequestException(message="shell_not_found")
    if code == "auth_failed":
        raise BadRequestException(message="auth_failed")
    raise ServiceUnavailableException(message=code or "worker_unreachable")


async def _send_ws_error(websocket: WebSocket, code: str, message: str):
    await websocket.send_text(
        json.dumps(
            {"op": "error", "code": code, "message": message},
            separators=(",", ":"),
        )
    )
