from datetime import datetime
from enum import Enum
import hashlib
from typing import Optional
from urllib.parse import urlparse

from pydantic import field_validator
from pydantic import computed_field
from sqlalchemy import Column, Enum as SQLEnum, String, UniqueConstraint
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


class ModelPreheatS3ProfileLifecycleStateEnum(str, Enum):
    ACTIVE = "active"
    MAINTENANCE = "maintenance"


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


def model_preheat_s3_storage_key(endpoint: str, bucket: str) -> str:
    """生成跨数据库一致的 S3 位置标识，不包含受系统控制的 prefix。"""
    parsed = urlparse(endpoint)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid_endpoint_port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("invalid_endpoint_port")
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if port in {80, 443} and (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        port = None
    normalized_endpoint = f"{scheme}://{host}"
    if port is not None:
        normalized_endpoint = f"{normalized_endpoint}:{port}"
    location = f"{normalized_endpoint}|{bucket.strip().lower()}"
    return hashlib.sha256(location.encode("utf-8")).hexdigest()


def validate_model_preheat_s3_endpoint(value: str) -> str:
    """校验新写入的 Endpoint；Public/ORM 必须能承载待修复的历史值。"""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("invalid_endpoint_scheme")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid_endpoint_port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("invalid_endpoint_port")
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid_endpoint_format")
    return value


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
    inventory_refresh_interval_seconds: Optional[int] = Field(default=None, ge=60)

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
        # active Profile 的 Endpoint + Bucket 必须唯一；维护中的历史/重复配置保留。
        UniqueConstraint(
            "active_storage_key",
            name="uix_model_preheat_s3_profiles_active_storage_key",
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
    lifecycle_state: ModelPreheatS3ProfileLifecycleStateEnum = Field(
        default=ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE,
        sa_column=Column(
            SQLEnum(
                ModelPreheatS3ProfileLifecycleStateEnum,
                values_callable=lambda enum_class: [item.value for item in enum_class],
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
    )
    active_storage_key: Optional[str] = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    ever_used_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    default_slot: Optional[str] = None
    source_fallback_enabled: bool = True
    connectivity_state: ModelPreheatS3ConnectivityStateEnum = Field(
        default=ModelPreheatS3ConnectivityStateEnum.PENDING
    )
    last_connectivity_check_id: Optional[int] = None
    last_connectivity_checked_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    inventory_last_attempt_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    inventory_last_success_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    inventory_last_scan_count: int = 0
    inventory_last_error_code: Optional[str] = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    # 刷新租约只用于跨 Server 的库存互斥，不进入 Public API。
    inventory_refresh_owner: Optional[str] = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    inventory_refresh_config_version: Optional[int] = None
    inventory_refresh_lease_expires_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )


class ModelPreheatS3ProfileCreate(ModelPreheatS3ProfileBase):
    access_key: str
    secret_key: str
    # 手工 Profile：provisioning_source 固定 manual、provisioning_key 为 NULL、
    # 默认 source_fallback_enabled 开启；默认槽位由 UI 单独设置（default_slot）。
    source_fallback_enabled: bool = True
    default_slot: Optional[str] = None

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str):
        return validate_model_preheat_s3_endpoint(value)


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
    lifecycle_state: Optional[ModelPreheatS3ProfileLifecycleStateEnum] = None
    inventory_refresh_interval_seconds: Optional[int] = Field(default=None, ge=60)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: Optional[str]):
        if value is None:
            return value
        return validate_model_preheat_s3_endpoint(value)

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
    lifecycle_state: ModelPreheatS3ProfileLifecycleStateEnum
    ever_used_at: Optional[datetime] = None
    default_slot: Optional[str] = None
    source_fallback_enabled: bool
    config_version: int
    connectivity_state: ModelPreheatS3ConnectivityStateEnum
    last_connectivity_check_id: Optional[int] = None
    last_connectivity_checked_at: Optional[datetime] = None
    inventory_refresh_interval_seconds: Optional[int] = None
    inventory_last_attempt_at: Optional[datetime] = None
    inventory_last_success_at: Optional[datetime] = None
    inventory_last_scan_count: int = 0
    inventory_last_error_code: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_default(self) -> bool:
        """Public API 的 is_default 由 default_slot 派生，避免两个字段漂移。"""
        return self.default_slot == DEFAULT_SLOT_GLOBAL


ModelPreheatS3ProfilesPublic = PaginatedList[ModelPreheatS3ProfilePublic]
