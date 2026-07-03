from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContainerExecCloseReason(str, Enum):
    USER_CLOSE = "user_close"
    TARGET_LOST = "target_lost"
    TIMEOUT = "timeout"
    EXPIRED = "expired"
    OUTPUT_OVERFLOW = "output_overflow"
    INTERNAL_ERROR = "internal_error"


class ContainerExecCreateRequest(BaseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
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


class ContainerExecResizeMessage(BaseModel):
    cols: int = Field(ge=1, le=1000)
    rows: int = Field(ge=1, le=1000)


class ContainerExecSession(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    worker_id: Optional[int] = None
    worker_uuid: Optional[str] = None
    state: str = "pending"
    cols: int = 80
    rows: int = 24
    ticket_hash: Optional[str] = None
    created_at: datetime
    expires_at: datetime
    connected_at: Optional[datetime] = None
    last_activity_at: datetime
    closed_at: Optional[datetime] = None
    close_reason: Optional[ContainerExecCloseReason] = None
    shell_path: Optional[str] = None
    cwd: Optional[str] = None
    master_fd: Optional[int] = None
    pid: Optional[int] = None
    pgid: Optional[int] = None
    bytes_in: int = 0
    bytes_out: int = 0
    manager: Any = Field(default=None, exclude=True)
