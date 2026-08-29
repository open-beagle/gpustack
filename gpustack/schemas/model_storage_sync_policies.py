import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.triggers.cron import CronTrigger
from pydantic import ConfigDict, field_validator, model_validator
from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import JSON, PaginatedList, UTCDateTime
from gpustack.schemas.model_storage_sync import ModelStorageSyncScopeEnum
from gpustack.schemas.policy_runs import (
    PolicyRunExecutionStateEnum,
    PolicyRunSummary,
    PolicyRunTaskPublic,
)


class ModelStorageSyncPolicyTriggerModeEnum(str, Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"


class ModelStorageSyncPolicyRunTriggerEnum(str, Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"


class ModelStorageSyncPolicyRunStateEnum(str, Enum):
    PENDING = "pending"
    READY = "ready"
    ERROR = "error"


class ModelStorageSyncPolicy(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_storage_sync_policies"
    __table_args__ = (UniqueConstraint("name", name="uix_storage_sync_policy_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(sa_column=Column(String(255), nullable=False))
    enabled: bool = True
    trigger_mode: ModelStorageSyncPolicyTriggerModeEnum
    cron_expression: Optional[str] = Field(
        default=None, sa_column=Column(String(255), nullable=True)
    )
    timezone: str = Field(sa_column=Column(String(64), nullable=False))
    profile_id: int = Field(
        sa_column=Column(
            ForeignKey("model_preheat_s3_profiles.id", ondelete="RESTRICT"),
            nullable=False,
        )
    )
    scope: ModelStorageSyncScopeEnum
    model_file_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("model_files.id", ondelete="RESTRICT"), nullable=True
        ),
    )
    worker_uuids: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    created_by_user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
    )
    next_run_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    last_run_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )


class ModelStorageSyncPolicyRun(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_storage_sync_policy_runs"
    __table_args__ = (
        UniqueConstraint("operation_key", name="uix_storage_sync_policy_operation"),
        UniqueConstraint(
            "policy_id",
            "window_start_utc",
            name="uix_storage_sync_policy_window",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    policy_id: int = Field(
        sa_column=Column(
            ForeignKey("model_storage_sync_policies.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    trigger: ModelStorageSyncPolicyRunTriggerEnum
    state: ModelStorageSyncPolicyRunStateEnum = (
        ModelStorageSyncPolicyRunStateEnum.PENDING
    )
    window_start_utc: datetime = Field(sa_column=Column(UTCDateTime, nullable=False))
    operation_key: str = Field(sa_column=Column(String(64), nullable=False))
    request_hash: str = Field(sa_column=Column(String(64), nullable=False))
    attempt: int = 0
    lease_owner: Optional[str] = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    lease_token: Optional[str] = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    lease_expires_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    response_payload: Optional[dict] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    error_code: Optional[str] = Field(
        default=None, sa_column=Column(String(255), nullable=True)
    )
    created_by_user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
    )
    execution_user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(ForeignKey("users.id", ondelete="RESTRICT"), nullable=True),
    )
    finished_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )


class ModelStorageSyncPolicyBase(SQLModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    trigger_mode: ModelStorageSyncPolicyTriggerModeEnum = (
        ModelStorageSyncPolicyTriggerModeEnum.MANUAL
    )
    cron_expression: Optional[str] = None
    timezone: str = "UTC"
    profile_id: int
    scope: ModelStorageSyncScopeEnum = ModelStorageSyncScopeEnum.SINGLE_MODEL
    model_file_id: Optional[int] = None
    worker_uuids: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        value = value.strip()
        if not value or len(value) > 255:
            raise ValueError("invalid_sync_policy_name")
        return value

    @field_validator("cron_expression")
    @classmethod
    def validate_cron(cls, value):
        if value is None:
            return None
        try:
            CronTrigger.from_crontab(value, timezone=timezone.utc)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid_cron_expression") from exc
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("invalid_timezone") from exc
        return value

    @field_validator("worker_uuids")
    @classmethod
    def normalize_worker_uuids(cls, values):
        normalized = sorted(value.strip() for value in values)
        if any(not value for value in normalized) or len(normalized) != len(
            set(normalized)
        ):
            raise ValueError("invalid_worker_uuid")
        return normalized

    @model_validator(mode="after")
    def validate_selector(self):
        if (
            self.trigger_mode == ModelStorageSyncPolicyTriggerModeEnum.SCHEDULED
            and self.cron_expression is None
        ):
            raise ValueError("cron_expression_required")
        if self.scope == ModelStorageSyncScopeEnum.SINGLE_MODEL:
            if self.model_file_id is None or self.worker_uuids:
                raise ValueError("invalid_single_model_selector")
        elif self.scope == ModelStorageSyncScopeEnum.SELECTED_WORKERS:
            if not self.worker_uuids or self.model_file_id is not None:
                raise ValueError("invalid_worker_selector")
        elif self.model_file_id is not None or self.worker_uuids:
            raise ValueError("invalid_all_workers_selector")
        return self


class ModelStorageSyncPolicyCreate(ModelStorageSyncPolicyBase):
    enabled: bool = True


class ModelStorageSyncPolicyUpdate(SQLModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    enabled: Optional[bool] = None
    trigger_mode: Optional[ModelStorageSyncPolicyTriggerModeEnum] = None
    cron_expression: Optional[str] = None
    timezone: Optional[str] = None
    profile_id: Optional[int] = None
    scope: Optional[ModelStorageSyncScopeEnum] = None
    model_file_id: Optional[int] = None
    worker_uuids: Optional[list[str]] = None


class ModelStorageSyncPolicyPublic(ModelStorageSyncPolicyBase):
    id: int
    enabled: bool
    created_by_user_id: Optional[int] = None
    next_run_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None
    latest_run: Optional["ModelStorageSyncPolicyRunPublic"] = None
    created_at: datetime
    updated_at: datetime


class ModelStorageSyncPolicyRunPublic(SQLModel):
    id: int
    policy_id: int
    trigger: ModelStorageSyncPolicyRunTriggerEnum
    state: ModelStorageSyncPolicyRunStateEnum
    execution_state: PolicyRunExecutionStateEnum
    summary: PolicyRunSummary = Field(default_factory=PolicyRunSummary)
    tasks: list[PolicyRunTaskPublic] = Field(default_factory=list)
    window_start_utc: datetime
    attempt: int
    response_payload: Optional[dict] = None
    error_code: Optional[str] = None
    created_by_user_id: Optional[int] = None
    finished_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


ModelStorageSyncPoliciesPublic = PaginatedList[ModelStorageSyncPolicyPublic]
ModelStorageSyncPolicyRunsPublic = PaginatedList[ModelStorageSyncPolicyRunPublic]


def sync_policy_operation_key(kind: str, *values) -> str:
    payload = json.dumps([kind, *values], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
