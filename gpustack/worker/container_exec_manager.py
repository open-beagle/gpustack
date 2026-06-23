from __future__ import annotations

from datetime import datetime, timedelta, timezone
import errno
import fcntl
import hashlib
import os
import pty
import secrets
import select
import signal
import struct
import termios
import threading
import time
from typing import Dict, Optional

from gpustack.schemas.container_exec import (
    ContainerExecCloseReason,
    ContainerExecCreateRequest,
    ContainerExecResizeMessage,
    ContainerExecSession,
)


DEFAULT_BIND_TIMEOUT_SECONDS = 60
DEFAULT_SESSION_TTL_SECONDS = 30 * 60
DEFAULT_IDLE_TIMEOUT_SECONDS = 10 * 60
DEFAULT_MAX_INPUT_FRAME_SIZE = 64 * 1024
DEFAULT_MAX_OUTPUT_BUFFER_SIZE = 1024 * 1024
DEFAULT_TERMINATE_GRACE_SECONDS = 3
DEFAULT_WORKDIR = "/var/lib/gpustack"


class ContainerExecSessionError(RuntimeError):
    pass


class ContainerExecManager:
    def __init__(
        self,
        *,
        bind_timeout_seconds: float = DEFAULT_BIND_TIMEOUT_SECONDS,
        session_ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
        max_input_frame_size: int = DEFAULT_MAX_INPUT_FRAME_SIZE,
        max_output_buffer_size: int = DEFAULT_MAX_OUTPUT_BUFFER_SIZE,
        terminate_grace_seconds: float = DEFAULT_TERMINATE_GRACE_SECONDS,
    ):
        self.bind_timeout_seconds = bind_timeout_seconds
        self.session_ttl_seconds = session_ttl_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        self.max_input_frame_size = max_input_frame_size
        self.max_output_buffer_size = max_output_buffer_size
        self.terminate_grace_seconds = terminate_grace_seconds
        self.sessions: Dict[str, ContainerExecSession] = {}
        self._lock = threading.RLock()

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def create_session(
        self, request: ContainerExecCreateRequest
    ) -> tuple[ContainerExecSession, str]:
        with self._lock:
            if request.session_id in self.sessions:
                raise ContainerExecSessionError("session already exists")

            created_at = self.now()
            expires_at = request.expires_at or created_at + timedelta(
                seconds=self.bind_timeout_seconds
            )
            ticket = secrets.token_urlsafe(32)
            session = ContainerExecSession(
                session_id=request.session_id,
                worker_id=request.worker_id,
                worker_uuid=request.worker_uuid,
                cols=request.cols,
                rows=request.rows,
                ticket_hash=self._hash_ticket(ticket),
                created_at=created_at,
                expires_at=expires_at,
                last_activity_at=created_at,
                manager=self,
            )
            self.sessions[session.session_id] = session
            return session, ticket

    def get_session(self, session_id: str) -> Optional[ContainerExecSession]:
        with self._lock:
            return self.sessions.get(session_id)

    def bind_session(self, session_id: str, ticket: str) -> ContainerExecSession:
        with self._lock:
            session = self._get_existing_session(session_id)
            if session.state != "pending":
                raise ContainerExecSessionError("session already bound")
            if self._is_expired(session):
                self._remove_pending_session(session)
                raise ContainerExecSessionError("session ticket expired")
            if not session.ticket_hash or not secrets.compare_digest(
                session.ticket_hash, self._hash_ticket(ticket)
            ):
                raise ContainerExecSessionError("session ticket mismatch")

            self._start_pty(session)
            now = self.now()
            session.state = "connected"
            session.ticket_hash = None
            session.connected_at = now
            session.last_activity_at = now
            return session

    def resize_session(
        self, session_id: str, message: ContainerExecResizeMessage
    ) -> None:
        session = self._get_connected_session(session_id)
        self._set_pty_size(session.master_fd, message.rows, message.cols)
        session.rows = message.rows
        session.cols = message.cols
        session.last_activity_at = self.now()

    def write_stdin(self, session_id: str, data: bytes) -> None:
        session = self._get_connected_session(session_id)
        if len(data) > self.max_input_frame_size:
            raise ContainerExecSessionError("input frame exceeds limit")
        try:
            os.write(session.master_fd, data)
        except OSError as exc:
            self.close_session(session_id, ContainerExecCloseReason.TARGET_LOST)
            raise ContainerExecSessionError("failed to write stdin") from exc
        session.bytes_in += len(data)
        session.last_activity_at = self.now()

    def read_stdout(self, session_id: str, *, max_bytes: int = 65536) -> bytes:
        session = self._get_connected_session(session_id)
        chunks = []
        total = 0

        while True:
            try:
                readable, _, _ = select.select([session.master_fd], [], [], 0)
            except (OSError, ValueError):
                self.close_session(session_id, ContainerExecCloseReason.TARGET_LOST)
                return b""

            if not readable:
                break

            remaining = min(max_bytes, self.max_output_buffer_size) - total
            if remaining <= 0:
                break

            try:
                data = os.read(session.master_fd, remaining)
            except OSError as exc:
                if exc.errno in (errno.EIO, errno.EBADF):
                    self.close_session(session_id, ContainerExecCloseReason.TARGET_LOST)
                    break
                raise

            if not data:
                self.close_session(session_id, ContainerExecCloseReason.TARGET_LOST)
                break

            chunks.append(data)
            total += len(data)
            session.bytes_out += len(data)
            session.last_activity_at = self.now()

            if total >= max_bytes:
                break

        return b"".join(chunks)

    def close_session(
        self,
        session_id: str,
        reason: ContainerExecCloseReason = ContainerExecCloseReason.USER_CLOSE,
    ) -> None:
        with self._lock:
            session = self.sessions.pop(session_id, None)
            if session is not None:
                session.state = "closed"
                session.closed_at = self.now()
                session.close_reason = reason
                master_fd = session.master_fd
                pgid = session.pgid
                pid = session.pid
                session.master_fd = None

        if session is None:
            return

        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass

        if pid is not None:
            self._terminate_process_group(pgid, pid)

    def cleanup_expired_sessions(self) -> list[str]:
        removed = []
        with self._lock:
            sessions = list(self.sessions.values())
        for session in sessions:
            if session.state == "pending" and self._is_expired(session):
                self._remove_pending_session(session)
                removed.append(session.session_id)
        return removed

    def check_session_limits(self) -> list[str]:
        closed = []
        now = self.now()
        with self._lock:
            sessions = list(self.sessions.values())
        for session in sessions:
            if session.state == "pending" and self._is_expired(session, now):
                self._remove_pending_session(session)
                closed.append(session.session_id)
                continue
            if session.state != "connected":
                continue

            ttl_expired = session.connected_at and now >= session.connected_at + timedelta(
                seconds=self.session_ttl_seconds
            )
            idle_expired = now >= session.last_activity_at + timedelta(
                seconds=self.idle_timeout_seconds
            )
            if ttl_expired or idle_expired:
                self.close_session(session.session_id, ContainerExecCloseReason.TIMEOUT)
                closed.append(session.session_id)
        return closed

    def _get_existing_session(self, session_id: str) -> ContainerExecSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise ContainerExecSessionError("session not found")
        return session

    def _get_connected_session(self, session_id: str) -> ContainerExecSession:
        session = self._get_existing_session(session_id)
        if session.state != "connected" or session.master_fd is None:
            raise ContainerExecSessionError("session is not connected")
        return session

    def _remove_pending_session(self, session: ContainerExecSession) -> None:
        with self._lock:
            self.sessions.pop(session.session_id, None)
            session.state = "expired"
            session.ticket_hash = None
            session.closed_at = self.now()
            session.close_reason = ContainerExecCloseReason.EXPIRED

    def _is_expired(
        self, session: ContainerExecSession, now: Optional[datetime] = None
    ) -> bool:
        return (now or self.now()) >= session.expires_at

    def _start_pty(self, session: ContainerExecSession) -> None:
        shell_path = self._select_shell()
        cwd = DEFAULT_WORKDIR if os.path.isdir(DEFAULT_WORKDIR) else "/"
        master_fd, slave_fd = pty.openpty()

        try:
            self._set_pty_size(master_fd, session.rows, session.cols)
            pid = os.fork()
        except Exception:
            os.close(master_fd)
            os.close(slave_fd)
            raise

        if pid == 0:
            self._exec_shell_child(slave_fd, shell_path, cwd, session.session_id)

        os.close(slave_fd)
        self._set_nonblocking(master_fd)
        session.master_fd = master_fd
        session.pid = pid
        session.pgid = self._wait_for_child_pgid(pid)
        session.shell_path = shell_path
        session.cwd = cwd

    def _exec_shell_child(
        self, slave_fd: int, shell_path: str, cwd: str, session_id: str
    ) -> None:
        try:
            os.setsid()
            if hasattr(termios, "TIOCSCTTY"):
                try:
                    fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
                except OSError:
                    pass
            os.dup2(slave_fd, 0)
            os.dup2(slave_fd, 1)
            os.dup2(slave_fd, 2)
            if slave_fd > 2:
                os.close(slave_fd)
            os.chdir(cwd)
            env = os.environ.copy()
            env["CPS_CLUSTER_TERMINAL_SESSION_ID"] = session_id
            env.setdefault("TERM", "xterm-256color")
            os.execve(shell_path, [shell_path, "-i"], env)
        except BaseException:
            os._exit(127)

    def _select_shell(self) -> str:
        for shell_path in ("/bin/bash", "/bin/sh"):
            if os.path.exists(shell_path) and os.access(shell_path, os.X_OK):
                return shell_path
        raise ContainerExecSessionError("shell not found")

    def _wait_for_child_pgid(self, pid: int) -> int:
        deadline = time.monotonic() + 1
        while True:
            try:
                pgid = os.getpgid(pid)
                if pgid == pid:
                    return pgid
            except ProcessLookupError:
                raise ContainerExecSessionError("shell process exited before bind")
            except PermissionError:
                pass
            if time.monotonic() >= deadline:
                raise ContainerExecSessionError("shell process group was not isolated")
            time.sleep(0.01)

    def _terminate_process_group(self, pgid: Optional[int], pid: int) -> None:
        if pgid is None:
            self._wait_child(pid, block=True)
            return

        for sig in (signal.SIGHUP, signal.SIGTERM):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                self._wait_child(pid)
                return

        deadline = time.monotonic() + self.terminate_grace_seconds
        while time.monotonic() < deadline:
            if self._wait_child(pid):
                return
            if not self._process_group_exists(pgid):
                return
            time.sleep(0.02)

        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self._wait_child(pid, block=True)

    def _wait_child(self, pid: int, *, block: bool = False) -> bool:
        deadline = time.monotonic() + 1
        while True:
            try:
                waited_pid, _ = os.waitpid(pid, 0 if block else os.WNOHANG)
            except ChildProcessError:
                return True
            if waited_pid == pid:
                return True
            if not block and waited_pid == 0:
                return False
            if time.monotonic() >= deadline:
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    return True
                return True
            time.sleep(0.02)

    def _process_group_exists(self, pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False

    def _set_nonblocking(self, fd: int) -> None:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def _set_pty_size(self, fd: int, rows: int, cols: int) -> None:
        winsize = struct.pack("HHHH", rows, cols, 0, 0)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

    def _hash_ticket(self, ticket: str) -> str:
        return hashlib.sha256(ticket.encode("utf-8")).hexdigest()
