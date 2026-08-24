import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import field_validator, model_validator
from sqlalchemy import Column, ForeignKey, Index, String, UniqueConstraint
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


class ModelPreheatInventoryManifestStateEnum(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"
    STALE = "stale"


class ModelPreheatInventoryJobStateEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    READY = "ready"
    ERROR = "error"


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
    # 请求身份（规范化的 source/model_id/requested_revision/Patterns）与其摘要，
    # 取代旧的 cache_key。Schedule 与分发策略同样保存请求身份，
    # 每次实例化任务时再解析不可变 revision 并绑定 Artifact。
    request_identity: dict = Field(sa_column=Column(JSON, nullable=False))
    request_digest: str
    desired_state: ModelPreheatDesiredStateEnum = ModelPreheatDesiredStateEnum.RUNNING
    execution_state: ModelPreheatExecutionStateEnum = (
        ModelPreheatExecutionStateEnum.PENDING
    )
    paused_from_state: Optional[ModelPreheatExecutionStateEnum] = None
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    progress: float = 0
    # Artifact 两阶段绑定：创建任务时库存精确命中则直接绑定，否则保持 NULL，
    # 由 Worker 解析并扫描后用任务 ID + request digest + Worker UUID + lease token
    # 通过 CAS（WHERE artifact_id IS NULL）原子绑定，不能覆盖已有值。
    artifact_id: Optional[str] = None
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
    s3_manifest_path: Optional[str] = None
    manifest_digest: Optional[str] = None
    keep_new_workers_in_sync: bool = False
    # 任务执行结果来源字段：模型身份（source/model_id/revision）保持不变，
    # 这些字段仅记录本次传输路径，不参与 Artifact ID 计算。
    transfer_source: Optional[str] = None
    transfer_profile_id: Optional[int] = None
    source_worker_id: Optional[int] = None
    schedule_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("model_preheat_schedules.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    bandwidth_limit_mbps: Optional[int] = None
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
        UniqueConstraint(
            "operation_key", name="uix_preheat_worker_distribution_operation"
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
    distribution_policy_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("model_preheat_distribution_policies.id", ondelete="RESTRICT"),
            nullable=True,
        ),
    )
    operation_key: Optional[str] = None
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


class ModelPreheatWorkerIdentity(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_worker_identities"
    __table_args__ = (
        UniqueConstraint("worker_id", name="uix_preheat_worker_identity_worker"),
        Index("ix_preheat_worker_identity_uuid", "worker_uuid"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    worker_id: int = Field(
        sa_column=Column(ForeignKey("workers.id", ondelete="CASCADE"), nullable=False)
    )
    worker_uuid: str
    token_hash: Optional[str] = None
    token_version: int = 0
    bootstrap_required: bool = True
    expires_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    revoked_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )


class ModelPreheatWorkerTaskPublic(SQLModel):
    id: int
    task_id: Optional[int] = None
    connectivity_check_id: Optional[int] = None
    distribution_policy_id: Optional[int] = None
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
    source_fallback_enabled: bool = True
    access_key: str = Field(repr=False)
    secret_key: str = Field(repr=False)


class ModelPreheatTrustedLocalCandidate(SQLModel):
    source: str
    root: str
    paths: list[str]
    repository_complete: bool


class ModelPreheatWorkerTaskExecutionPayload(SQLModel):
    worker_task_id: int
    attempt: int
    role: ModelPreheatWorkerTaskRoleEnum
    resumable_cursor: Optional[dict] = None
    task: dict
    profile: ModelPreheatExecutionProfile = Field(repr=False)
    trusted_local_candidate: Optional[ModelPreheatTrustedLocalCandidate] = None


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
    batch_lease_token: Optional[str] = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    batch_lease_expires_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
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


class ModelPreheatArtifact(SQLModel, BaseModelMixin, table=True):
    """统一 Artifact 库存。

    库存不是第二份事实来源：只认合法 ``<artifact_id>/manifest.json``，
    手工刷新按 Profile 扫描合法 Manifest 重建本表，数据库记录丢失时可恢复。
    查询必须同时精确匹配 ``profile_id + profile_config_version``；
    Profile 配置变化递增 config_version 并把旧版本库存标记 stale。
    现有库存数据直接清空，不做字段转换。
    """

    __tablename__ = "model_preheat_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "profile_config_version",
            "artifact_id",
            name="uix_preheat_artifact_profile_version_artifact",
        ),
        Index(
            "ix_preheat_artifact_profile_state_version",
            "profile_id",
            "profile_config_version",
            "manifest_state",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    profile_id: int = Field(
        sa_column=Column(
            ForeignKey("model_preheat_s3_profiles.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    # 新任务查询必须精确匹配配置版本；旧版本记录仅供已创建任务读取，
    # 待无活动任务引用后删除数据库缓存，不删除旧 S3 对象。
    profile_config_version: int
    artifact_id: str
    source: str
    model_id: str
    resolved_revision: str
    include_patterns: list[str] = Field(sa_column=Column(JSON, nullable=False))
    exclude_patterns: list[str] = Field(sa_column=Column(JSON, nullable=False))
    manifest_path: str = Field(sa_column=Column(Text, nullable=False))
    manifest_digest: str
    file_count: int
    total_size: int
    manifest_state: ModelPreheatInventoryManifestStateEnum
    last_verified_at: datetime = Field(sa_column=Column(UTCDateTime, nullable=False))
    created_by_task_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("model_preheat_tasks.id", ondelete="SET NULL"), nullable=True
        ),
    )


class ModelPreheatArtifactPublic(SQLModel):
    artifact_id: str
    source: str
    model_id: str
    resolved_revision: str
    include_patterns: list[str]
    exclude_patterns: list[str]
    manifest_path: str
    manifest_digest: str
    file_count: int
    total_size: int
    manifest_state: ModelPreheatInventoryManifestStateEnum
    last_verified_at: datetime
    created_by_task_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime


ModelPreheatArtifactsPage = PaginatedList[ModelPreheatArtifactPublic]


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
            normalized = sorted((encode_path(value), value) for value in values)
        except ModelPreheatIdentityError as exc:
            raise ValueError(str(exc)) from exc
        encoded = [item[0] for item in normalized]
        if len(encoded) != len(set(encoded)):
            raise ValueError("duplicate_pattern")
        return [item[1] for item in normalized]

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
    request_identity: dict
    request_digest: str
    artifact_id: Optional[str] = None
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
    transfer_source: Optional[str] = None
    transfer_profile_id: Optional[int] = None
    source_worker_id: Optional[int] = None
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
    request_digest: str,
    target_worker_uuids: list[str],
    backfill_policy: ModelPreheatBackfillPolicyEnum,
) -> str:
    payload = json.dumps(
        {
            "backfill_policy": backfill_policy.value,
            "request_digest": request_digest,
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


from gpustack.schemas import (
    model_preheat_schedules as _model_preheat_schedules,
)  # noqa: E402,F401


# 注册 worker task 外键引用的 Task 9 表，兼容仅导入本模块后 create_all。
from gpustack.schemas import (
    model_preheat_distribution_policies as _distribution_policies,
)  # noqa: E402,F401
