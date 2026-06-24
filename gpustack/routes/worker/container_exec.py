import asyncio
import json
import secrets

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status

from gpustack.api.auth import worker_auth
from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.container_exec import (
    ContainerExecCloseReason,
    ContainerExecCreateRequest,
    ContainerExecResizeMessage,
)
from gpustack.worker.container_exec_manager import (
    ContainerExecManager,
    ContainerExecSessionError,
)


router = APIRouter()
manager = ContainerExecManager()
_cleanup_task: asyncio.Task | None = None
_CLEANUP_INTERVAL_SECONDS = 30


@router.get("/admin/container-exec/capabilities", dependencies=[Depends(worker_auth)])
async def get_capabilities():
    return {"container_exec": True}


async def start_cleanup_task() -> None:
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_loop())


async def stop_cleanup_task() -> None:
    global _cleanup_task
    if _cleanup_task is None:
        return
    _cleanup_task.cancel()
    await asyncio.gather(_cleanup_task, return_exceptions=True)
    _cleanup_task = None


async def _cleanup_loop() -> None:
    while True:
        manager.cleanup_expired_sessions()
        manager.check_session_limits()
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)


@router.post("/admin/container-exec/sessions", dependencies=[Depends(worker_auth)])
async def create_session(request: ContainerExecCreateRequest):
    try:
        session, ticket = manager.create_session(request)
    except ContainerExecSessionError as exc:
        raise BadRequestException(message=str(exc))

    return {
        "session_id": session.session_id,
        "ticket": ticket,
        "state": session.state,
        "expires_at": session.expires_at,
    }


@router.delete(
    "/admin/container-exec/sessions/{session_id}", dependencies=[Depends(worker_auth)]
)
async def delete_session(session_id: str):
    manager.close_session(session_id, ContainerExecCloseReason.USER_CLOSE)
    return {"session_id": session_id, "state": "closed"}


@router.websocket("/admin/container-exec/sessions/{session_id}/ws")
async def websocket_session(websocket: WebSocket, session_id: str):
    if not _validate_websocket_credential(websocket):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    output_task: asyncio.Task | None = None
    send_lock = asyncio.Lock()
    bound = False

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("text") is not None:
                should_close = await _handle_control_message(
                    websocket, session_id, message["text"], send_lock
                )
                if should_close:
                    break
                if not bound and manager.get_session(session_id):
                    session = manager.get_session(session_id)
                    bound = bool(session and session.state == "connected")
                    if bound:
                        output_task = asyncio.create_task(
                            _pump_output(websocket, session_id, send_lock)
                        )
                continue

            data = message.get("bytes")
            if data is None:
                continue
            if not bound:
                await _send_error(websocket, "not_bound", "session is not bound", send_lock)
                continue
            try:
                manager.write_stdin(session_id, data)
            except ContainerExecSessionError as exc:
                await _send_error(websocket, "target_lost", str(exc), send_lock)
                break
    except WebSocketDisconnect:
        pass
    finally:
        if output_task:
            output_task.cancel()
            await asyncio.gather(output_task, return_exceptions=True)
        manager.close_session(session_id, ContainerExecCloseReason.TARGET_LOST)


async def _handle_control_message(
    websocket: WebSocket, path_session_id: str, text: str, send_lock: asyncio.Lock
) -> bool:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        await _send_error(
            websocket, "invalid_control_message", "invalid JSON control", send_lock
        )
        return False

    op = payload.get("op")
    if op == "bind":
        await _bind_session(websocket, path_session_id, payload, send_lock)
        return False
    if op == "resize":
        await _resize_session(websocket, path_session_id, payload, send_lock)
        return False
    if op == "ping":
        await _send_control(websocket, {"op": "pong", "ts": payload.get("ts")}, send_lock)
        return False
    if op == "pong":
        return False
    if op == "close":
        manager.close_session(path_session_id, ContainerExecCloseReason.USER_CLOSE)
        await _send_control(
            websocket,
            {"op": "close", "reason": payload.get("reason", "user_close")},
            send_lock,
        )
        return True
    if op == "error":
        manager.close_session(path_session_id, ContainerExecCloseReason.TARGET_LOST)
        return True

    await _send_error(
        websocket, "unsupported_control_op", f"unsupported op: {op}", send_lock
    )
    return False


async def _bind_session(
    websocket: WebSocket,
    path_session_id: str,
    payload: dict,
    send_lock: asyncio.Lock,
):
    requested_session_id = payload.get("session_id")
    ticket = payload.get("ticket") or payload.get("bind_token")
    if requested_session_id != path_session_id or not ticket:
        await _send_error(websocket, "auth_failed", "invalid bind request", send_lock)
        return

    try:
        manager.bind_session(path_session_id, ticket)
    except ContainerExecSessionError as exc:
        await _send_error(
            websocket, _error_code_from_manager_error(exc), str(exc), send_lock
        )
        return

    await _send_control(websocket, {"op": "bind", "state": "connected"}, send_lock)


async def _resize_session(
    websocket: WebSocket, session_id: str, payload: dict, send_lock: asyncio.Lock
):
    try:
        message = ContainerExecResizeMessage(cols=payload["cols"], rows=payload["rows"])
        manager.resize_session(session_id, message)
    except (KeyError, ContainerExecSessionError, ValueError) as exc:
        await _send_error(websocket, "invalid_resize", str(exc), send_lock)


async def _pump_output(
    websocket: WebSocket, session_id: str, send_lock: asyncio.Lock
):
    while True:
        data = manager.read_stdout(session_id)
        if data:
            async with send_lock:
                await websocket.send_bytes(data)
        await asyncio.sleep(0.02)


def _validate_websocket_credential(websocket: WebSocket) -> bool:
    authorization = websocket.headers.get("authorization")
    if not authorization:
        return False
    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credential:
        return False

    expected = getattr(websocket.app.state.config, "token", None)
    return bool(expected and secrets.compare_digest(credential, expected))


def _error_code_from_manager_error(exc: ContainerExecSessionError) -> str:
    message = str(exc)
    if "expired" in message:
        return "bind_token_expired"
    if "already bound" in message:
        return "bind_token_reused"
    if "mismatch" in message:
        return "auth_failed"
    if "not found" in message:
        return "session_not_found"
    if "shell not found" in message:
        return "shell_not_found"
    return "internal_error"


async def _send_control(
    websocket: WebSocket, payload: dict, send_lock: asyncio.Lock
):
    async with send_lock:
        await websocket.send_text(json.dumps(payload, separators=(",", ":")))


async def _send_error(
    websocket: WebSocket, code: str, message: str, send_lock: asyncio.Lock
):
    await _send_control(
        websocket,
        {"op": "error", "code": code, "message": message},
        send_lock,
    )
