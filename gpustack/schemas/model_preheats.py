import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import field_validator, model_validator
from sqlalchemy import Column, ForeignKey, Index, UniqueConstraint
from sqlmodel import Field, SQLModel, Text

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import JSON, PaginatedList, UTCDateTime
from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentityError,
    encode_path,
    normalize_source,
)


class ModelPreheatDesiredStateEnum(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    CANCELED = "canceled"


class ModelPreheatExecutionStateEnum(str, Enum):
    PENDING = "pending"
    RESOLVING = "resolving"
    SCANNING = "scanning"
    STAGING = "staging"
    PUBLISHING = "publishing"
    DISTRIBUTING = "distributing"
    PAUSED = "paused"
    READY = "ready"
    PARTIAL = "partial"
    ERROR = "error"
    CANCELED = "canceled"


class ModelPreheatWorkerTaskRoleEnum(str, Enum):
    SEED = "seed"
    DISTRIBUTE = "distribute"
    CONNECTIVITY_CHECK = "connectivity_check"
    INVENTORY = "inventory"


class ModelPreheatWorkerTaskStateEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    READY = "ready"
    ERROR = "error"
    CANCELED = "canceled"
    SKIPPED_WORKER_REMOVED = "skipped_worker_removed"


class ModelPreheatConnectivityCheckStateEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ModelPreheatBackfillPolicyEnum(str, Enum):
    ALWAYS = "always"
    WHEN_MISSING = "when_missing"
    NEVER = "never"


class ModelPreheatTargetScopeEnum(str, Enum):
    SEED_WORKER = "seed_worker"
    SAME_GPU_MODEL = "same_gpu_model"
    SELECTED_WORKERS = "selected_workers"


class ModelPreheatTask(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_tasks"

    id: Optional[int] = Field(default=None, primary_key=True)
    attempt: int = 1
    source: str
    model_id: str
    requested_revision: Optional[str] = None
    resolved_revision: str
    include_patterns: list[str] = Field(sa_column=Column(JSON, nullable=False))
    exclude_patterns: list[str] = Field(sa_column=Column(JSON, nullable=False))
    selection_digest: str
    cache_key: str
    generation_id: str
    desired_state: ModelPreheatDesiredStateEnum = ModelPreheatDesiredStateEnum.RUNNING
    execution_state: ModelPreheatExecutionStateEnum = (
        ModelPreheatExecutionStateEnum.PENDING
    )
    paused_from_state: Optional[ModelPreheatExecutionStateEnum] = None
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    progress: float = 0
    seed_worker_uuid: Optional[str] = None
    seed_worker_id: Optional[int] = None
    seed_source: Optional[str] = None
    target_scope: ModelPreheatTargetScopeEnum
    target_gpu_names: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    target_worker_uuids: list[str] = Field(sa_column=Column(JSON, nullable=False))
    target_worker_snapshot: list[dict] = Field(sa_column=Column(JSON, nullable=False))
    local_cache_hit_worker_uuids: list[str] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    removed_target_worker_uuids: list[str] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    s3_profile_id: int = Field(
        sa_column=Column(
            ForeignKey("model_preheat_s3_profiles.id", ondelete="RESTRICT"),
            nullable=False,
        )
    )
    s3_profile_config_version: int
    s3_profile_snapshot_encrypted: dict = Field(sa_column=Column(JSON, nullable=False))
    encryption_key_version: str
    s3_backfill_policy: ModelPreheatBackfillPolicyEnum
    s3_ready_path: Optional[str] = None
    s3_manifest_path: Optional[str] = None
    manifest_digest: Optional[str] = None
    keep_new_workers_in_sync: bool = False
    schedule_id: Optional[int] = None
    created_by_user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
    )
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    finished_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )


class ModelPreheatWorkerTask(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_worker_tasks"
    __table_args__ = (
        UniqueConstraint(
            "task_id",
            "parent_attempt",
            "worker_uuid",
            "role",
            name="uix_preheat_task_attempt_worker_role",
        ),
        UniqueConstraint(
            "connectivity_check_id",
            "worker_uuid",
            "role",
            name="uix_preheat_check_worker_role",
        ),
        Index("ix_preheat_worker_uuid_state", "worker_uuid", "state"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("model_preheat_tasks.id", ondelete="CASCADE"), nullable=True
        ),
    )
    connectivity_check_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("model_preheat_s3_connectivity_checks.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )
    parent_attempt: int = 1
    worker_uuid: str
    worker_id: Optional[int] = None
    role: ModelPreheatWorkerTaskRoleEnum
    state: ModelPreheatWorkerTaskStateEnum = ModelPreheatWorkerTaskStateEnum.PENDING
    attempt: int = 0
    lease_owner: Optional[str] = None
    lease_token_hash: Optional[str] = None
    lease_expires_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    last_heartbeat_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    error_code: Optional[str] = None
    progress: float = 0
    local_staging_dir: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    downloaded_size: int = 0
    total_size: int = 0
    resumable_cursor: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    finished_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )


class ModelPreheatWorkerTaskPublic(SQLModel):
    id: int
    task_id: Optional[int] = None
    connectivity_check_id: Optional[int] = None
    parent_attempt: int = 1
    worker_uuid: str
    worker_id: Optional[int] = None
    role: ModelPreheatWorkerTaskRoleEnum
    state: ModelPreheatWorkerTaskStateEnum
    attempt: int
    last_heartbeat_at: Optional[datetime] = None
    state_message: Optional[str] = None
    error_code: Optional[str] = None
    progress: float
    downloaded_size: int
    total_size: int
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


ModelPreheatWorkerTasksPublic = PaginatedList[ModelPreheatWorkerTaskPublic]


class ModelPreheatWorkerTaskClaim(SQLModel):
    worker_uuid: str
    worker_id: int


class ModelPreheatWorkerTaskClaimed(ModelPreheatWorkerTaskPublic):
    lease_token: str = Field(repr=False)
    lease_expires_at: datetime


class ModelPreheatWorkerTaskLease(SQLModel):
    worker_uuid: str
    worker_id: int
    attempt: int
    lease_token: str = Field(repr=False)


class ModelPreheatWorkerTaskProgress(ModelPreheatWorkerTaskLease):
    progress: float = Field(ge=0, le=100)
    downloaded_size: Optional[int] = Field(default=None, ge=0)
    total_size: Optional[int] = Field(default=None, ge=0)
    resumable_cursor: Optional[dict] = None
    state_message: Optional[str] = None


class ModelPreheatWorkerTaskComplete(ModelPreheatWorkerTaskLease):
    result: dict = Field(default_factory=dict)


class ModelPreheatWorkerTaskFail(ModelPreheatWorkerTaskLease):
    error_code: str
    state_message: Optional[str] = None
    result: dict = Field(default_factory=dict)


class ModelPreheatExecutionProfile(SQLModel):
    endpoint: str
    bucket: str
    prefix: str = ""
    tls_enabled: bool = True
    tls_verify: bool = True
    region: str = ""
    use_virtual_hosted_style: bool = True
    access_key: str = Field(repr=False)
    secret_key: str = Field(repr=False)


class ModelPreheatWorkerTaskExecutionPayload(SQLModel):
    worker_task_id: int
    attempt: int
    role: ModelPreheatWorkerTaskRoleEnum
    task: dict
    profile: ModelPreheatExecutionProfile = Field(repr=False)


class ModelPreheatS3ConnectivityCheck(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_s3_connectivity_checks"
    __table_args__ = (
        UniqueConstraint("active_key", name="uix_preheat_connectivity_active"),
        UniqueConstraint(
            "idempotency_key", name="uix_preheat_connectivity_idempotency"
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(
        sa_column=Column(
            ForeignKey("model_preheat_s3_profiles.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    profile_config_version: int
    idempotency_key: Optional[str] = None
    request_hash: Optional[str] = None
    scope_key: Optional[str] = None
    active_key: Optional[str] = None
    state: ModelPreheatConnectivityCheckStateEnum = (
        ModelPreheatConnectivityCheckStateEnum.PENDING
    )
    target_worker_uuids: list[str] = Field(sa_column=Column(JSON, nullable=False))
    success_count: int = 0
    failed_count: int = 0
    not_checked_count: int = 0
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    finished_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )


class ModelPreheatConnectivityWorkerPublic(SQLModel):
    worker_uuid: str
    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    state: ModelPreheatWorkerTaskStateEnum
    readable: bool = False
    writable: bool = False
    deletable: bool = False
    cleanup_failed: bool = False
    latency_ms: Optional[int] = None
    error_code: Optional[str] = None
    failed_stage: Optional[str] = None


class ModelPreheatConnectivityCheckPublic(SQLModel):
    id: int
    profile_id: int
    profile_config_version: int
    state: ModelPreheatConnectivityCheckStateEnum
    summary: dict[str, int]
    workers: list[ModelPreheatConnectivityWorkerPublic]
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class ModelPreheatIdempotencyRecord(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_idempotency_records"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "operation", "idempotency_key", name="uix_preheat_idempotency"
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(sa_column=Column(ForeignKey("users.id"), nullable=False))
    operation: str
    idempotency_key: str
    request_hash: str
    resource_type: str
    resource_id: int
    response_status: int = 200
    expires_at: datetime = Field(sa_column=Column(UTCDateTime, nullable=False))


class ModelPreheatTaskLock(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_task_locks"
    __table_args__ = (UniqueConstraint("operation_key", name="uix_preheat_operation"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    operation_key: str
    task_id: int = Field(
        sa_column=Column(
            ForeignKey("model_preheat_tasks.id", ondelete="CASCADE"), nullable=False
        )
    )
    lease_expires_at: datetime = Field(sa_column=Column(UTCDateTime, nullable=False))


class ModelPreheatPublishLock(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_publish_locks"
    __table_args__ = (
        UniqueConstraint("s3_profile_id", "cache_key", name="uix_preheat_publish"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    s3_profile_id: int = Field(
        sa_column=Column(
            ForeignKey("model_preheat_s3_profiles.id", ondelete="RESTRICT"),
            nullable=False,
        )
    )
    cache_key: str
    task_id: int = Field(
        sa_column=Column(
            ForeignKey("model_preheat_tasks.id", ondelete="CASCADE"), nullable=False
        )
    )
    lease_expires_at: datetime = Field(sa_column=Column(UTCDateTime, nullable=False))


class ModelPreheatCreate(SQLModel):
    source: str
    model_id: str
    revision: Optional[str] = None
    include_patterns: list[str] = []
    exclude_patterns: list[str] = []
    target_scope: ModelPreheatTargetScopeEnum = (
        ModelPreheatTargetScopeEnum.SELECTED_WORKERS
    )
    target_worker_ids: list[int] = []
    seed_worker_id: Optional[int] = None
    s3_profile_id: int
    s3_backfill_policy: ModelPreheatBackfillPolicyEnum = (
        ModelPreheatBackfillPolicyEnum.WHEN_MISSING
    )
    keep_new_workers_in_sync: bool = False

    @field_validator("target_worker_ids")
    @classmethod
    def normalize_target_worker_ids(cls, values: list[int]):
        normalized = sorted(values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("duplicate_target_worker_id")
        return normalized

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str):
        try:
            return normalize_source(value)
        except ModelPreheatIdentityError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str):
        try:
            encode_path(value)
        except ModelPreheatIdentityError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: Optional[str]):
        if value is None:
            return value
        try:
            encode_path(value)
        except ModelPreheatIdentityError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @field_validator("include_patterns", "exclude_patterns")
    @classmethod
    def normalize_patterns(cls, values: list[str]):
        try:
            normalized = sorted(encode_path(value) for value in values)
        except ModelPreheatIdentityError as exc:
            raise ValueError(str(exc)) from exc
        if len(normalized) != len(set(normalized)):
            raise ValueError("duplicate_pattern")
        return normalized

    @model_validator(mode="after")
    def validate_target_scope(self):
        if (
            self.target_scope == ModelPreheatTargetScopeEnum.SELECTED_WORKERS
            and not self.target_worker_ids
        ):
            raise ValueError("target_worker_ids_required")
        if (
            self.target_scope
            in {
                ModelPreheatTargetScopeEnum.SEED_WORKER,
                ModelPreheatTargetScopeEnum.SAME_GPU_MODEL,
            }
            and self.seed_worker_id is None
        ):
            raise ValueError("seed_worker_id_required")
        return self


class ModelPreheatTaskPublic(SQLModel):
    id: int
    attempt: int
    source: str
    model_id: str
    requested_revision: Optional[str] = None
    resolved_revision: str
    include_patterns: list[str]
    exclude_patterns: list[str]
    selection_digest: str
    cache_key: str
    generation_id: str
    desired_state: ModelPreheatDesiredStateEnum
    execution_state: ModelPreheatExecutionStateEnum
    paused_from_state: Optional[ModelPreheatExecutionStateEnum] = None
    target_scope: ModelPreheatTargetScopeEnum
    target_worker_uuids: list[str]
    target_worker_snapshot: list[dict]
    s3_profile_id: int
    s3_profile_config_version: int
    s3_backfill_policy: ModelPreheatBackfillPolicyEnum
    keep_new_workers_in_sync: bool
    created_at: datetime
    updated_at: datetime
    deduplicated: bool = False


ModelPreheatTasksPublic = PaginatedList[ModelPreheatTaskPublic]


def selection_digest(include_patterns: list[str], exclude_patterns: list[str]) -> str:
    payload = json.dumps(
        {"exclude_patterns": exclude_patterns, "include_patterns": include_patterns},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_key_for(
    source: str,
    model_id: str,
    resolved_revision: str,
    pattern_digest: str,
) -> str:
    raw = "\0".join(("1", source, model_id, resolved_revision, pattern_digest))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def operation_key_for(
    profile_id: int,
    cache_key: str,
    target_worker_uuids: list[str],
    backfill_policy: ModelPreheatBackfillPolicyEnum,
) -> str:
    payload = json.dumps(
        {
            "backfill_policy": backfill_policy.value,
            "cache_key": cache_key,
            "profile_id": profile_id,
            "target_worker_uuids": sorted(target_worker_uuids),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_terminal_task(task: ModelPreheatTask) -> bool:
    return task.execution_state in {
        ModelPreheatExecutionStateEnum.READY,
        ModelPreheatExecutionStateEnum.PARTIAL,
        ModelPreheatExecutionStateEnum.ERROR,
        ModelPreheatExecutionStateEnum.CANCELED,
    }
