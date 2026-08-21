from datetime import datetime
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

from pydantic import field_validator
from pydantic import computed_field
from sqlalchemy import Column, UniqueConstraint
from sqlmodel import Field, SQLModel, Text

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import JSON, PaginatedList, UTCDateTime


class ModelPreheatS3ConnectivityStateEnum(str, Enum):
    NO_WORKERS = "no_workers"
    PENDING = "pending"
    CHECKING = "checking"
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    STALE = "stale"


class ModelPreheatS3ProvisioningSourceEnum(str, Enum):
    MANUAL = "manual"
    WORKER_LOCAL_S3 = "worker_local_s3"


PROVISIONING_KEY_WORKER_LOCAL_S3 = "worker_local_s3"
DEFAULT_SLOT_GLOBAL = "global"


def normalize_model_preheat_s3_prefix(prefix: Optional[str]) -> str:
    if prefix is None:
        return ""
    if any(ord(char) < 32 or ord(char) == 127 for char in prefix):
        raise ValueError("invalid_prefix_control_character")
    if "\\" in prefix:
        raise ValueError("invalid_prefix_backslash")
    if prefix.startswith("/"):
        raise ValueError("invalid_prefix_absolute_path")

    parts = []
    for part in prefix.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            raise ValueError("invalid_prefix_parent_reference")
        parts.append(part)
    return "/".join(parts)


class ModelPreheatS3ProfileBase(SQLModel):
    name: str
    description: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    endpoint: str
    bucket: str
    prefix: str = ""
    tls_enabled: bool = True
    tls_verify: bool = True
    region: Optional[str] = ""
    use_virtual_hosted_style: bool = True

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str):
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("invalid_endpoint_scheme")
        return value

    @field_validator("prefix")
    @classmethod
    def validate_prefix(cls, value: Optional[str]):
        return normalize_model_preheat_s3_prefix(value)


class ModelPreheatS3Profile(ModelPreheatS3ProfileBase, BaseModelMixin, table=True):
    __tablename__ = "model_preheat_s3_profiles"
    __table_args__ = (
        UniqueConstraint("name", name="uix_model_preheat_s3_profiles_name"),
        # 手工 Profile 的 provisioning_key 为 NULL；系统 Profile 固定为 worker_local_s3。
        # 普通唯一约束保证跨数据库最多存在一个同来源的系统 Profile。
        UniqueConstraint(
            "provisioning_key",
            name="uix_model_preheat_s3_profiles_provisioning_key",
        ),
        # 默认 Profile 的 default_slot 固定为 global，其他为 NULL。
        # 依靠普通唯一约束保证 SQLite/PostgreSQL/MySQL 最多存在一个默认 Profile。
        UniqueConstraint(
            "default_slot", name="uix_model_preheat_s3_profiles_default_slot"
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    access_key_encrypted: dict = Field(sa_column=Column(JSON, nullable=False))
    secret_key_encrypted: dict = Field(sa_column=Column(JSON, nullable=False))
    encryption_key_version: str
    config_version: int = 1
    provisioning_source: ModelPreheatS3ProvisioningSourceEnum = Field(
        default=ModelPreheatS3ProvisioningSourceEnum.MANUAL
    )
    provisioning_key: Optional[str] = None
    system_managed: bool = False
    default_slot: Optional[str] = None
    source_fallback_enabled: bool = True
    connectivity_state: ModelPreheatS3ConnectivityStateEnum = Field(
        default=ModelPreheatS3ConnectivityStateEnum.PENDING
    )
    last_connectivity_check_id: Optional[int] = None
    last_connectivity_checked_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )


class ModelPreheatS3ProfileCreate(ModelPreheatS3ProfileBase):
    access_key: str
    secret_key: str
    # 手工 Profile：provisioning_source 固定 manual、provisioning_key 为 NULL、
    # 默认 source_fallback_enabled 开启；默认槽位由 UI 单独设置（default_slot）。
    source_fallback_enabled: bool = True
    default_slot: Optional[str] = None


class ModelPreheatS3ProfileUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    endpoint: Optional[str] = None
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    tls_enabled: Optional[bool] = None
    tls_verify: Optional[bool] = None
    region: Optional[str] = None
    use_virtual_hosted_style: Optional[bool] = None
    source_fallback_enabled: Optional[bool] = None
    default_slot: Optional[str] = None

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: Optional[str]):
        if value is None:
            return value
        return ModelPreheatS3ProfileBase.validate_endpoint(value)

    @field_validator("prefix")
    @classmethod
    def validate_prefix(cls, value: Optional[str]):
        if value is None:
            return value
        return normalize_model_preheat_s3_prefix(value)


class ModelPreheatS3ProfilePublic(ModelPreheatS3ProfileBase):
    id: int
    credential_configured: bool
    provisioning_source: ModelPreheatS3ProvisioningSourceEnum
    provisioning_key: Optional[str] = None
    system_managed: bool
    default_slot: Optional[str] = None
    source_fallback_enabled: bool
    config_version: int
    connectivity_state: ModelPreheatS3ConnectivityStateEnum
    last_connectivity_check_id: Optional[int] = None
    last_connectivity_checked_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_default(self) -> bool:
        """Public API 的 is_default 由 default_slot 派生，避免两个字段漂移。"""
        return self.default_slot == DEFAULT_SLOT_GLOBAL


ModelPreheatS3ProfilesPublic = PaginatedList[ModelPreheatS3ProfilePublic]
