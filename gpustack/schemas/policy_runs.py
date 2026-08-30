from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class PolicyRunExecutionStateEnum(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PAUSED = "paused"
    READY = "ready"
    PARTIAL_ERROR = "partial_error"
    ERROR = "error"
    SKIPPED = "skipped"


class PolicyRunTaskPublic(SQLModel):
    id: Optional[int] = None
    model_file_id: Optional[int] = None
    model_id: Optional[str] = None
    worker_id: Optional[int] = None
    worker_uuid: Optional[str] = None
    worker_name: Optional[str] = None
    worker_ip: Optional[str] = None
    artifact_id: Optional[str] = None
    state: str
    progress: float = 0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    error_code: Optional[str] = None
    state_message: Optional[str] = None


class PolicyRunSummary(SQLModel):
    total: int = 0
    pending: int = 0
    running: int = 0
    paused: int = 0
    ready: int = 0
    error: int = 0
    failed: int = 0
    skipped: int = 0
    progress: float = 0
    downloaded_bytes: int = 0
    total_bytes: int = 0


class PolicyRunObservation(SQLModel):
    execution_state: PolicyRunExecutionStateEnum
    summary: PolicyRunSummary = Field(default_factory=PolicyRunSummary)
    tasks: list[PolicyRunTaskPublic] = Field(default_factory=list)
