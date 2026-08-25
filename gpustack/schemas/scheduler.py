from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import ConfigDict, field_validator
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    Enum as SAEnum,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import JSON, PaginatedList, UTCDateTime


class SchedulingOutcome(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class PlacementEvaluationReplicaGroup(SQLModel):
    gpu_ids: list[str]


class PlacementEvaluationRequest(SQLModel):
    model_id: int
    replica_groups: list[PlacementEvaluationReplicaGroup]
    independent: bool = False
    discover: bool = False


class PlacementEvaluationReplicaResult(SQLModel):
    group_index: int
    fit: bool
    reason_code: str
    reason: str
    candidate_targets: list[dict]
    selected_targets: list[dict]


class PlacementEvaluationResponse(SQLModel):
    fit: bool
    results: list[PlacementEvaluationReplicaResult]


class SchedulerPolicy(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "scheduler_policies"
    __table_args__ = (
        CheckConstraint(
            "aggregation_rate > 0 AND aggregation_rate <= 100",
            name="ck_scheduler_policy_rate",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(sa_column=Column(String(64), nullable=False, unique=True))
    name: str = Field(sa_column=Column(String(255), nullable=False))
    algorithm: str = Field(sa_column=Column(String(64), nullable=False))
    aggregation_rate: float
    enabled: bool = True
    runtime_revision: int = Field(sa_column=Column(BigInteger, nullable=False))
    updated_by: str = Field(sa_column=Column(String(255), nullable=False))


class SchedulerPolicyPublic(SQLModel):
    code: str
    name: str
    algorithm: str
    aggregation_rate: float
    enabled: bool
    runtime_revision: int
    updated_by: str
    updated_at: datetime


class SchedulerPolicyUpdate(SQLModel):
    model_config = ConfigDict(extra="forbid")

    aggregation_rate: float
    enabled: bool
    expected_revision: int
    target_revision: int

    @field_validator("aggregation_rate")
    @classmethod
    def validate_rate(cls, value: float) -> float:
        if value <= 0 or value > 100:
            raise ValueError("aggregation_rate_must_be_in_range_0_100")
        return round(value, 2)

    @field_validator("expected_revision", "target_revision")
    @classmethod
    def validate_revision(cls, value: int) -> int:
        if value < 1:
            raise ValueError("scheduler_policy_revision_must_be_positive")
        return value


class SchedulingAttemptEvent(SQLModel, table=True):
    __tablename__ = "scheduling_attempt_events"
    __table_args__ = (
        UniqueConstraint(
            "workload_id", "attempt_no", name="uix_scheduling_attempt_workload_attempt"
        ),
        CheckConstraint(
            "outcome IN ('success', 'failed')", name="ck_scheduling_attempt_outcome"
        ),
        CheckConstraint("latency_ms >= 0", name="ck_scheduling_attempt_latency"),
        Index("ix_scheduling_attempt_policy_occurred", "policy_code", "occurred_at"),
        Index("ix_scheduling_attempt_workload", "workload_id"),
    )

    event_id: str = Field(sa_column=Column(String(36), primary_key=True))
    workload_id: str = Field(sa_column=Column(String(255), nullable=False))
    attempt_no: int = Field(sa_column=Column(Integer, nullable=False))
    policy_code: str = Field(sa_column=Column(String(64), nullable=False))
    policy_revision: int = Field(sa_column=Column(BigInteger, nullable=False))
    requested_replicas: int = Field(sa_column=Column(Integer, nullable=False))
    requested_resources: dict = Field(sa_column=Column(JSON, nullable=False))
    candidate_targets: list[dict] = Field(sa_column=Column(JSON, nullable=False))
    selected_targets: list[dict] = Field(sa_column=Column(JSON, nullable=False))
    outcome: SchedulingOutcome = Field(
        sa_column=Column(
            SAEnum(
                SchedulingOutcome,
                values_callable=lambda outcomes: [
                    outcome.value for outcome in outcomes
                ],
                native_enum=False,
                length=16,
            ),
            nullable=False,
        )
    )
    reason_code: str = Field(sa_column=Column(String(128), nullable=False))
    reason: str = Field(sa_column=Column(Text, nullable=False))
    latency_ms: int = Field(sa_column=Column(BigInteger, nullable=False))
    trace_id: Optional[str] = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    occurred_at: datetime = Field(sa_column=Column(UTCDateTime, nullable=False))


class SchedulingAttemptEventPublic(SQLModel):
    event_id: str
    workload_id: str
    attempt_no: int
    policy_code: str
    policy_revision: int
    requested_replicas: int
    requested_resources: dict
    candidate_targets: list[dict]
    selected_targets: list[dict]
    outcome: SchedulingOutcome
    reason_code: str
    reason: str
    latency_ms: int
    trace_id: Optional[str]
    occurred_at: datetime


SchedulingAttemptEventsPublic = PaginatedList[SchedulingAttemptEventPublic]
