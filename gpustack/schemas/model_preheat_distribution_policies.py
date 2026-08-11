import hashlib
import json
from datetime import datetime
from typing import Optional

from pydantic import ConfigDict, field_validator
from sqlalchemy import Column, ForeignKey, UniqueConstraint
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import JSON, PaginatedList, UTCDateTime
from gpustack.schemas.model_preheats import ModelPreheatTargetScopeEnum


class ModelPreheatDistributionPolicy(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_distribution_policies"
    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "cache_key",
            "target_scope",
            "selector_digest",
            name="uix_preheat_distribution_policy_selector",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    enabled: bool = True
    profile_version_stale: bool = False
    profile_id: int = Field(
        sa_column=Column(
            ForeignKey("model_preheat_s3_profiles.id", ondelete="RESTRICT"),
            nullable=False,
        )
    )
    profile_config_version: int
    cache_key: str
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
    last_reconciled_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
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
    profile_id: int
    profile_config_version: int
    cache_key: str
    target_scope: ModelPreheatTargetScopeEnum
    worker_selector: dict
    gpu_selector: dict
    created_by_task_id: Optional[int] = None
    last_reconciled_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class ModelPreheatDistributionPolicyUpdate(SQLModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    enabled: Optional[bool] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        if value is not None and (not value.strip() or len(value) > 255):
            raise ValueError("invalid_policy_name")
        return value.strip() if value is not None else value


ModelPreheatDistributionPoliciesPublic = PaginatedList[
    ModelPreheatDistributionPolicyPublic
]


def distribution_selector_digest(worker_selector: dict, gpu_selector: dict) -> str:
    payload = json.dumps(
        {"gpu_selector": gpu_selector, "worker_selector": worker_selector},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def distribution_operation_key(policy_id: int, worker_uuid: str, cache_key: str) -> str:
    payload = json.dumps(
        [policy_id, worker_uuid, cache_key], separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
