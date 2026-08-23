from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.triggers.cron import CronTrigger
from pydantic import ConfigDict, field_validator, model_validator
from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import JSON, PaginatedList, UTCDateTime
from gpustack.schemas.model_preheats import (
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatTargetScopeEnum,
)
from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentityError,
    encode_path,
    normalize_source,
)


class ModelPreheatScheduleRunTriggerEnum(str, Enum):
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class ModelPreheatScheduleTriggerModeEnum(str, Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"


class ModelPreheatScheduleRunStateEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    READY = "ready"
    SKIPPED = "skipped"
    ERROR = "error"


class ModelPreheatSchedule(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_schedules"
    __table_args__ = (UniqueConstraint("name", name="uix_preheat_schedule_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(sa_column=Column(String(255), nullable=False))
    enabled: bool = True
    trigger_mode: ModelPreheatScheduleTriggerModeEnum = Field(
        default=ModelPreheatScheduleTriggerModeEnum.SCHEDULED,
        sa_column=Column(String(32), nullable=False),
    )
    cron_expression: Optional[str] = Field(
        default=None, sa_column=Column(String(255), nullable=True)
    )
    timezone: str = Field(sa_column=Column(String(64), nullable=False))
    window_duration_minutes: int
    max_concurrency: int = 1
    bandwidth_limit_mbps: Optional[int] = None
    source: str = Field(sa_column=Column(String(32), nullable=False))
    model_id: str = Field(sa_column=Column(String(512), nullable=False))
    revision: Optional[str] = Field(
        default=None, sa_column=Column(String(512), nullable=True)
    )
    include_patterns: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    exclude_patterns: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    target_scope: ModelPreheatTargetScopeEnum
    target_worker_uuids: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    seed_worker_uuid: Optional[str] = Field(
        default=None, sa_column=Column(String(255), nullable=True)
    )
    s3_profile_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("model_preheat_s3_profiles.id", ondelete="RESTRICT"),
            nullable=False,
        )
    )
    s3_backfill_policy: ModelPreheatBackfillPolicyEnum = (
        ModelPreheatBackfillPolicyEnum.WHEN_MISSING
    )
    keep_new_workers_in_sync: bool = False
    created_by_user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    next_window_start_utc: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    last_window_start_utc: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )


class ModelPreheatScheduleRun(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_schedule_runs"
    __table_args__ = (
        UniqueConstraint(
            "schedule_id",
            "window_start_utc",
            "operation_key",
            name="uix_preheat_schedule_window",
        ),
        UniqueConstraint("operation_key", name="uix_preheat_schedule_run_operation"),
        UniqueConstraint("schedule_id", "slot", name="uix_preheat_schedule_slot"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    schedule_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("model_preheat_schedules.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    window_start_utc: datetime = Field(sa_column=Column(UTCDateTime, nullable=False))
    window_end_utc: datetime = Field(sa_column=Column(UTCDateTime, nullable=False))
    trigger: ModelPreheatScheduleRunTriggerEnum
    state: ModelPreheatScheduleRunStateEnum = ModelPreheatScheduleRunStateEnum.PENDING
    operation_key: str = Field(sa_column=Column(String(64), nullable=False))
    slot: Optional[int] = None
    task_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_preheat_tasks.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    created_by_user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    error_code: Optional[str] = Field(
        default=None, sa_column=Column(String(255), nullable=True)
    )
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    finished_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )


class ModelPreheatScheduleBase(SQLModel):
    name: str
    trigger_mode: ModelPreheatScheduleTriggerModeEnum = (
        ModelPreheatScheduleTriggerModeEnum.SCHEDULED
    )
    cron_expression: Optional[str] = None
    timezone: str = "UTC"
    window_duration_minutes: int = Field(ge=1, le=10080)
    max_concurrency: int = Field(default=1, ge=1, le=32)
    bandwidth_limit_mbps: Optional[int] = Field(default=None, ge=1, le=100000)
    source: str
    model_id: str
    revision: Optional[str] = None
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    target_scope: ModelPreheatTargetScopeEnum = (
        ModelPreheatTargetScopeEnum.SELECTED_WORKERS
    )
    target_worker_uuids: list[str] = Field(default_factory=list)
    seed_worker_uuid: Optional[str] = None
    s3_profile_id: int
    s3_backfill_policy: ModelPreheatBackfillPolicyEnum = (
        ModelPreheatBackfillPolicyEnum.WHEN_MISSING
    )
    keep_new_workers_in_sync: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        value = value.strip()
        if not value or len(value) > 255:
            raise ValueError("invalid_schedule_name")
        return value

    @field_validator("cron_expression")
    @classmethod
    def validate_cron_expression(cls, value):
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

    @field_validator("source")
    @classmethod
    def validate_source(cls, value):
        try:
            return normalize_source(value)
        except ModelPreheatIdentityError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("model_id", "revision")
    @classmethod
    def validate_path_value(cls, value):
        if value is None:
            return value
        try:
            encode_path(value)
        except ModelPreheatIdentityError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @field_validator("include_patterns", "exclude_patterns")
    @classmethod
    def normalize_patterns(cls, values):
        try:
            normalized = sorted((encode_path(value), value) for value in values)
        except ModelPreheatIdentityError as exc:
            raise ValueError(str(exc)) from exc
        encoded = [item[0] for item in normalized]
        if len(encoded) != len(set(encoded)):
            raise ValueError("duplicate_pattern")
        return [item[1] for item in normalized]

    @field_validator("target_worker_uuids")
    @classmethod
    def normalize_worker_uuids(cls, values):
        normalized = sorted(value.strip() for value in values)
        if any(not value for value in normalized) or len(normalized) != len(
            set(normalized)
        ):
            raise ValueError("invalid_target_worker_uuid")
        return normalized

    @model_validator(mode="after")
    def validate_target(self):
        if self.trigger_mode == ModelPreheatScheduleTriggerModeEnum.SCHEDULED:
            if self.cron_expression is None:
                raise ValueError("cron_expression_required")
        else:
            self.cron_expression = None
        if (
            self.target_scope == ModelPreheatTargetScopeEnum.SELECTED_WORKERS
            and not self.target_worker_uuids
        ):
            raise ValueError("target_worker_uuids_required")
        if (
            self.target_scope
            in {
                ModelPreheatTargetScopeEnum.SEED_WORKER,
                ModelPreheatTargetScopeEnum.SAME_GPU_MODEL,
            }
            and not self.seed_worker_uuid
        ):
            raise ValueError("seed_worker_uuid_required")
        if (
            self.target_scope == ModelPreheatTargetScopeEnum.SELECTED_WORKERS
            and self.seed_worker_uuid
            and self.target_worker_uuids
            and self.seed_worker_uuid not in self.target_worker_uuids
        ):
            raise ValueError("seed_worker_not_in_target_scope")
        return self


class ModelPreheatScheduleCreate(ModelPreheatScheduleBase):
    pass


class ModelPreheatScheduleUpdate(SQLModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    enabled: Optional[bool] = None
    trigger_mode: Optional[ModelPreheatScheduleTriggerModeEnum] = None
    cron_expression: Optional[str] = None
    timezone: Optional[str] = None
    window_duration_minutes: Optional[int] = Field(default=None, ge=1, le=10080)
    max_concurrency: Optional[int] = Field(default=None, ge=1, le=32)
    bandwidth_limit_mbps: Optional[int] = Field(default=None, ge=1, le=100000)
    source: Optional[str] = None
    model_id: Optional[str] = None
    revision: Optional[str] = None
    include_patterns: Optional[list[str]] = None
    exclude_patterns: Optional[list[str]] = None
    target_scope: Optional[ModelPreheatTargetScopeEnum] = None
    target_worker_uuids: Optional[list[str]] = None
    seed_worker_uuid: Optional[str] = None
    s3_profile_id: Optional[int] = None
    s3_backfill_policy: Optional[ModelPreheatBackfillPolicyEnum] = None
    keep_new_workers_in_sync: Optional[bool] = None

    @field_validator("enabled")
    @classmethod
    def validate_enabled(cls, value):
        if value is None:
            raise ValueError("enabled_required")
        return value


class ModelPreheatSchedulePublic(ModelPreheatScheduleBase):
    id: int
    enabled: bool
    created_by_user_id: Optional[int] = None
    next_window_start_utc: Optional[datetime] = None
    last_window_start_utc: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class ModelPreheatScheduleRunPublic(SQLModel):
    id: int
    schedule_id: int
    window_start_utc: datetime
    window_end_utc: datetime
    trigger: ModelPreheatScheduleRunTriggerEnum
    state: ModelPreheatScheduleRunStateEnum
    task_id: Optional[int] = None
    error_code: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


ModelPreheatSchedulesPublic = PaginatedList[ModelPreheatSchedulePublic]
ModelPreheatScheduleRunsPublic = PaginatedList[ModelPreheatScheduleRunPublic]


def next_window_start_utc(schedule, after: datetime) -> datetime:
    if after.tzinfo is None:
        raise ValueError("timezone_aware_datetime_required")
    if (
        schedule.trigger_mode != ModelPreheatScheduleTriggerModeEnum.SCHEDULED
        or schedule.cron_expression is None
    ):
        raise ValueError("scheduled_trigger_required")
    zone = ZoneInfo(schedule.timezone)
    trigger = CronTrigger.from_crontab(schedule.cron_expression, timezone=zone)
    cursor = (after.astimezone(timezone.utc) + timedelta(microseconds=1)).astimezone(
        zone
    )
    candidate = trigger.get_next_fire_time(None, cursor)
    while candidate is not None:
        candidate_utc = candidate.astimezone(timezone.utc)
        round_trip = candidate_utc.astimezone(zone)
        if (
            candidate.fold == 0
            and round_trip.replace(tzinfo=None) == candidate.replace(tzinfo=None)
            and round_trip.fold == candidate.fold
        ):
            return candidate_utc
        candidate = trigger.get_next_fire_time(candidate, candidate)
    raise ValueError("schedule_has_no_next_window")


def window_end_utc(window_start: datetime, duration_minutes: int) -> datetime:
    if window_start.tzinfo is None:
        raise ValueError("timezone_aware_datetime_required")
    return window_start.astimezone(timezone.utc) + timedelta(minutes=duration_minutes)
