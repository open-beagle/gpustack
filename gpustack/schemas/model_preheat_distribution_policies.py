import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.triggers.cron import CronTrigger
from pydantic import ConfigDict, field_validator, model_validator
from sqlalchemy import Column, ForeignKey, String, UniqueConstraint
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import JSON, PaginatedList, UTCDateTime
from gpustack.schemas.model_preheats import ModelPreheatTargetScopeEnum


class ModelPreheatDistributionPolicyTriggerModeEnum(str, Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    CONTINUOUS = "continuous"


class ModelPreheatDistributionPolicyRunTriggerEnum(str, Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    CONTINUOUS = "continuous"


class ModelPreheatDistributionPolicyRunStateEnum(str, Enum):
    PENDING = "pending"
    READY = "ready"
    ERROR = "error"


class ModelPreheatDistributionPolicy(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_distribution_policies"
    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "request_digest",
            "target_scope",
            "selector_digest",
            name="uix_preheat_distribution_policy_selector",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    enabled: bool = True
    profile_version_stale: bool = False
    trigger_mode: ModelPreheatDistributionPolicyTriggerModeEnum = Field(
        default=ModelPreheatDistributionPolicyTriggerModeEnum.CONTINUOUS,
        sa_column=Column(String(32), nullable=False),
    )
    cron_expression: Optional[str] = Field(
        default=None, sa_column=Column(String(255), nullable=True)
    )
    timezone: str = Field(default="UTC", sa_column=Column(String(64), nullable=False))
    profile_id: int = Field(
        sa_column=Column(
            ForeignKey("model_preheat_s3_profiles.id", ondelete="RESTRICT"),
            nullable=False,
        )
    )
    profile_config_version: int
    # 用请求身份与其摘要替代旧 cache_key；每次实例化任务时再解析不可变 revision
    # 选择或生成 Artifact，不再依赖旧 cache_key。
    request_identity: dict = Field(sa_column=Column(JSON, nullable=False))
    request_digest: str
    target_scope: ModelPreheatTargetScopeEnum
    worker_selector: dict = Field(sa_column=Column(JSON, nullable=False))
    gpu_selector: dict = Field(sa_column=Column(JSON, nullable=False))
    selector_digest: str
    created_by_task_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("model_preheat_tasks.id", ondelete="RESTRICT"), nullable=True
        ),
    )
    source_artifact_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("model_preheat_artifacts.id", ondelete="RESTRICT"),
            nullable=True,
        ),
    )
    source_sync_task_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("model_storage_sync_tasks.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    last_reconciled_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    next_run_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    last_run_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    blocked_reason: Optional[str] = None


class ModelPreheatDistributionPolicyRun(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_distribution_policy_runs"
    __table_args__ = (
        UniqueConstraint("operation_key", name="uix_distribution_policy_operation"),
        UniqueConstraint(
            "policy_id",
            "window_start_utc",
            name="uix_distribution_policy_window",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    policy_id: int = Field(
        sa_column=Column(
            ForeignKey("model_preheat_distribution_policies.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    trigger: ModelPreheatDistributionPolicyRunTriggerEnum = Field(
        sa_column=Column(String(32), nullable=False)
    )
    state: ModelPreheatDistributionPolicyRunStateEnum = Field(
        default=ModelPreheatDistributionPolicyRunStateEnum.PENDING,
        sa_column=Column(String(32), nullable=False),
    )
    window_start_utc: datetime = Field(sa_column=Column(UTCDateTime, nullable=False))
    operation_key: str
    lease_owner: Optional[str] = None
    lease_token: Optional[str] = None
    lease_expires_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    error_code: Optional[str] = None
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    finished_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )


class ModelPreheatDistributionWorkerSlot(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_distribution_worker_slots"
    __table_args__ = (
        UniqueConstraint(
            "policy_id", "worker_uuid", name="uix_distribution_policy_worker_slot"
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    policy_id: int = Field(
        sa_column=Column(
            ForeignKey("model_preheat_distribution_policies.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    worker_uuid: str = Field(sa_column=Column(String(255), nullable=False))
    active_task_id: Optional[int] = None
    active_operation_key: Optional[str] = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )


class ModelPreheatWorkerObservation(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_worker_observations"

    worker_uuid: str = Field(primary_key=True)
    worker_id: int
    network_fingerprint: str
    ready: bool = False


class ModelPreheatDistributionPolicyPublic(SQLModel):
    id: int
    name: str
    enabled: bool
    trigger_mode: ModelPreheatDistributionPolicyTriggerModeEnum
    cron_expression: Optional[str] = None
    timezone: str
    profile_id: int
    profile_config_version: int
    request_identity: dict
    request_digest: str
    target_scope: ModelPreheatTargetScopeEnum
    worker_selector: dict
    gpu_selector: dict
    created_by_task_id: Optional[int] = None
    source_artifact_id: Optional[int] = None
    source_artifact: Optional[str] = None
    source_sync_task_id: Optional[int] = None
    last_reconciled_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None
    blocked_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ModelPreheatDistributionPolicyUpdate(SQLModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    enabled: Optional[bool] = None
    trigger_mode: Optional[ModelPreheatDistributionPolicyTriggerModeEnum] = None
    cron_expression: Optional[str] = None
    timezone: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        if value is not None and (not value.strip() or len(value) > 255):
            raise ValueError("invalid_policy_name")
        return value.strip() if value is not None else value


ModelPreheatDistributionPoliciesPublic = PaginatedList[
    ModelPreheatDistributionPolicyPublic
]


class ModelPreheatDistributionPolicyCreate(SQLModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    trigger_mode: ModelPreheatDistributionPolicyTriggerModeEnum = (
        ModelPreheatDistributionPolicyTriggerModeEnum.MANUAL
    )
    cron_expression: Optional[str] = None
    timezone: str = "UTC"
    profile_id: Optional[int] = None
    artifact_id: Optional[str] = None
    sync_task_id: Optional[int] = None
    target_scope: ModelPreheatTargetScopeEnum
    worker_selector: dict = Field(default_factory=dict)
    gpu_selector: dict = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_create_name(cls, value):
        value = value.strip()
        if not value or len(value) > 255:
            raise ValueError("invalid_policy_name")
        return value

    @model_validator(mode="after")
    def validate_source_and_selector(self):
        if (self.artifact_id is None) == (self.sync_task_id is None):
            raise ValueError("distribution_source_required")
        if self.artifact_id is not None and self.profile_id is None:
            raise ValueError("profile_id_required")
        worker_uuids = self.worker_selector.get("worker_uuids", [])
        gpu_names = self.gpu_selector.get("gpu_names", [])
        if self.target_scope == ModelPreheatTargetScopeEnum.SAME_GPU_MODEL:
            if not gpu_names or worker_uuids:
                raise ValueError("invalid_gpu_selector")
        elif not worker_uuids or gpu_names:
            raise ValueError("invalid_worker_selector")
        if self.trigger_mode == ModelPreheatDistributionPolicyTriggerModeEnum.SCHEDULED:
            if self.cron_expression is None:
                raise ValueError("cron_expression_required")
            try:
                CronTrigger.from_crontab(
                    self.cron_expression, timezone=ZoneInfo(self.timezone)
                )
            except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
                raise ValueError("invalid_distribution_schedule") from exc
        elif self.cron_expression is not None:
            raise ValueError("cron_expression_not_allowed")
        else:
            try:
                ZoneInfo(self.timezone)
            except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
                raise ValueError("invalid_timezone") from exc
        return self


def distribution_selector_digest(worker_selector: dict, gpu_selector: dict) -> str:
    payload = json.dumps(
        {"gpu_selector": gpu_selector, "worker_selector": worker_selector},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def distribution_operation_key(
    policy_id: int, worker_uuid: str, request_digest: str, run_key: str | None = None
) -> str:
    payload = json.dumps(
        (
            [policy_id, worker_uuid, request_digest]
            if run_key is None
            else [policy_id, worker_uuid, request_digest, run_key]
        ),
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def distribution_policy_run_operation_key(
    policy_id: int,
    window_start: datetime,
    trigger: str = "scheduled",
    unique_suffix: str | None = None,
) -> str:
    payload = json.dumps(
        [
            "distribution_policy",
            trigger,
            policy_id,
            window_start.isoformat(),
            unique_suffix,
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
