import asyncio
from datetime import timedelta
import importlib.util
import os
from pathlib import Path
import time

import pytest

from gpustack.schemas.container_exec import (
    ContainerExecCloseReason,
    ContainerExecCreateRequest,
    ContainerExecResizeMessage,
)


MANAGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "gpustack"
    / "worker"
    / "container_exec_manager.py"
)
SPEC = importlib.util.spec_from_file_location("container_exec_manager", MANAGER_PATH)
container_exec_manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(container_exec_manager)

ContainerExecManager = container_exec_manager.ContainerExecManager
ContainerExecSessionError = container_exec_manager.ContainerExecSessionError


def create_request(**kwargs):
    data = {
        "session_id": "cterm_test",
        "worker_id": 1,
        "worker_uuid": "worker-uuid",
        "cols": 80,
        "rows": 24,
    }
    data.update(kwargs)
    return ContainerExecCreateRequest(**data)


@pytest.fixture
def manager():
    mgr = ContainerExecManager(
        bind_timeout_seconds=0.5,
        session_ttl_seconds=5,
        idle_timeout_seconds=5,
        max_input_frame_size=1024,
        max_output_buffer_size=4096,
    )
    yield mgr
    for session in list(mgr.sessions.values()):
        mgr.close_session(session.session_id, ContainerExecCloseReason.INTERNAL_ERROR)


def test_create_session_only_registers_pending_session(manager):
    session, ticket = manager.create_session(create_request())

    assert ticket
    assert session.session_id == "cterm_test"
    assert session.state == "pending"
    assert session.ticket_hash != ticket
    assert session.master_fd is None
    assert session.pid is None
    assert manager.get_session("cterm_test") is session


def test_bind_creates_pty_and_shell(manager):
    session, ticket = manager.create_session(create_request())

    bound = manager.bind_session(session.session_id, ticket)

    try:
        assert bound.state == "connected"
        assert bound.master_fd is not None
        assert bound.pid is not None
        assert bound.pgid == os.getpgid(bound.pid)
        assert os.environ.get("CPS_CLUSTER_TERMINAL_SESSION_ID") != bound.session_id
    finally:
        manager.close_session(bound.session_id, ContainerExecCloseReason.USER_CLOSE)


def test_repeated_bind_fails(manager):
    session, ticket = manager.create_session(create_request())
    manager.bind_session(session.session_id, ticket)

    with pytest.raises(ContainerExecSessionError, match="already bound"):
        manager.bind_session(session.session_id, ticket)


def test_expired_ticket_cannot_bind():
    mgr = ContainerExecManager(bind_timeout_seconds=0.01)
    session, ticket = mgr.create_session(create_request())
    time.sleep(0.03)

    with pytest.raises(ContainerExecSessionError, match="expired"):
        mgr.bind_session(session.session_id, ticket)

    assert mgr.get_session(session.session_id) is None


def test_close_session_exits_shell_process(manager):
    session, ticket = manager.create_session(create_request())
    bound = manager.bind_session(session.session_id, ticket)
    pid = bound.pid

    manager.close_session(bound.session_id, ContainerExecCloseReason.USER_CLOSE)

    assert manager.get_session(bound.session_id) is None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_non_utf8_output_can_be_read(manager):
    session, ticket = manager.create_session(create_request())
    bound = manager.bind_session(session.session_id, ticket)

    try:
        manager.write_stdin(bound.session_id, b"printf '\\377\\376ok\\n'\n")
        output = await read_until(bound, b"\xff\xfeok", timeout=2)

        assert b"\xff\xfeok" in output
    finally:
        manager.close_session(bound.session_id, ContainerExecCloseReason.USER_CLOSE)


def test_resize_updates_pty_size(manager):
    session, ticket = manager.create_session(create_request())
    bound = manager.bind_session(session.session_id, ticket)

    try:
        manager.resize_session(
            bound.session_id, ContainerExecResizeMessage(cols=100, rows=40)
        )

        assert bound.cols == 100
        assert bound.rows == 40
    finally:
        manager.close_session(bound.session_id, ContainerExecCloseReason.USER_CLOSE)


def test_stdin_rejects_oversized_frame(manager):
    session, ticket = manager.create_session(create_request())
    bound = manager.bind_session(session.session_id, ticket)

    try:
        with pytest.raises(ContainerExecSessionError, match="input frame"):
            manager.write_stdin(bound.session_id, b"x" * 2048)
    finally:
        manager.close_session(bound.session_id, ContainerExecCloseReason.USER_CLOSE)


def test_read_stdout_caps_each_output_chunk():
    mgr = ContainerExecManager(max_output_buffer_size=8)
    session, ticket = mgr.create_session(create_request())
    bound = mgr.bind_session(session.session_id, ticket)

    try:
        mgr.write_stdin(bound.session_id, b"printf '0123456789abcdef\\n'\n")

        output = mgr.read_stdout(bound.session_id, max_bytes=1024)

        assert len(output) <= 8
        assert mgr.get_session(bound.session_id) is bound
    finally:
        mgr.close_session(bound.session_id, ContainerExecCloseReason.USER_CLOSE)


@pytest.mark.asyncio
async def test_output_buffer_limit_is_not_total_bytes_limit():
    mgr = ContainerExecManager(max_output_buffer_size=128)
    session, ticket = mgr.create_session(create_request())
    bound = mgr.bind_session(session.session_id, ticket)

    try:
        for index in range(4):
            marker = f"chunk-{index}".encode()
            mgr.write_stdin(bound.session_id, b"printf '" + marker + b"\\n'\n")
            output = await read_until(bound, marker, timeout=2)
            assert marker in output

        assert bound.bytes_out > 128
        assert mgr.get_session(bound.session_id) is bound
    finally:
        mgr.close_session(bound.session_id, ContainerExecCloseReason.USER_CLOSE)


def test_cleanup_expired_pending_session(manager):
    session, _ = manager.create_session(
        create_request(expires_at=manager.now() - timedelta(seconds=1))
    )

    removed = manager.cleanup_expired_sessions()

    assert removed == [session.session_id]
    assert manager.get_session(session.session_id) is None


def test_check_session_limits_closes_idle_session(manager):
    session, ticket = manager.create_session(create_request())
    bound = manager.bind_session(session.session_id, ticket)
    bound.last_activity_at = manager.now() - timedelta(seconds=10)

    closed = manager.check_session_limits()

    assert bound.session_id in closed
    assert manager.get_session(bound.session_id) is None


async def read_until(session, marker: bytes, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    chunks = []
    while time.monotonic() < deadline:
        data = session.manager.read_stdout(session.session_id, max_bytes=4096)
        if data:
            chunks.append(data)
            joined = b"".join(chunks)
            if marker in joined:
                return joined
        await asyncio.sleep(0.02)
    pytest.fail(f"Timed out waiting for {marker!r}")
