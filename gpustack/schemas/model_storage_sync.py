from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, ForeignKey, Index, JSON, Text
from sqlmodel import Field, SQLModel
from pydantic import ConfigDict

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList, UTCDateTime
from gpustack.schemas.model_file_download_executions import (
    ModelFileTransferSourceEnum,
)


class ModelStorageSyncTaskStateEnum(str, Enum):
    PENDING = "pending"
    SCANNING = "scanning"
    PUBLISHING = "publishing"
    READY = "ready"
    ERROR = "error"
    CANCELED = "canceled"


class ModelStorageSyncTask(SQLModel, BaseModelMixin, table=True):
    """模型同步任务（原“模型归档任务”）。

    输入只有 ``model_file_id + profile_id``：Server 从 ModelFile 推导请求身份、
    本地路径和目标对象 Key，浏览器不得提交任意目标路径。

    任务创建时固定 request identity、Profile ID、配置版本和加密凭据快照，
    ``artifact_id`` 可为空；库存精确命中时直接固定 Artifact，否则 Worker 扫描后
    使用任务、request digest、Worker UUID 和 lease token 将 ``artifact_id``
    从 NULL CAS 绑定。后续修改默认 Profile 或凭据不改变已创建任务。
    """

    __tablename__ = "model_storage_sync_tasks"
    __table_args__ = (
        Index("ix_model_storage_sync_model_file_profile", "model_file_id", "profile_id"),
        Index("ix_model_storage_sync_state", "state"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    model_file_id: int = Field(
        sa_column=Column(
            ForeignKey("model_files.id", ondelete="CASCADE"), nullable=False
        ),
    )
    worker_id: int = Field(
        sa_column=Column(
            ForeignKey("workers.id", ondelete="RESTRICT"), nullable=False
        ),
    )
    worker_uuid: str
    profile_id: int = Field(
        sa_column=Column(
            ForeignKey("model_preheat_s3_profiles.id", ondelete="RESTRICT"),
            nullable=False,
        ),
    )
    profile_config_version: int
    # 固定请求身份与其摘要
    request_identity: dict = Field(sa_column=Column(JSON, nullable=False))
    request_digest: str
    source: str
    model_id: str
    resolved_revision: str
    # 任务私有加密凭据快照（AES-GCM），只进入受 Worker 身份约束的执行 payload，
    # 不进入 Public schema、SSE 或日志。
    credential_snapshot_encrypted: dict = Field(
        sa_column=Column(JSON, nullable=False)
    )
    encryption_key_version: str
    # 库存精确命中时直接固定，否则保持 NULL，由 Worker CAS 绑定。
    artifact_id: Optional[str] = None
    state: ModelStorageSyncTaskStateEnum = ModelStorageSyncTaskStateEnum.PENDING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    error_code: Optional[str] = None
    file_count: int = 0
    total_size: int = 0
    # 任务执行结果来源字段
    transfer_source: Optional[ModelFileTransferSourceEnum] = None
    transfer_profile_id: Optional[int] = None
    source_worker_id: Optional[int] = None
    created_by_user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("users.id", ondelete="SET NULL"), nullable=True
        ),
    )
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    finished_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )


class ModelStorageSyncTaskPublic(SQLModel):
    """Public schema 不返回凭据快照或加密字段。"""

    id: int
    model_file_id: int
    worker_id: int
    worker_uuid: str
    profile_id: int
    profile_config_version: int
    request_digest: str
    source: str
    model_id: str
    resolved_revision: str
    artifact_id: Optional[str] = None
    state: ModelStorageSyncTaskStateEnum
    state_message: Optional[str] = None
    error_code: Optional[str] = None
    file_count: int
    total_size: int
    transfer_source: Optional[ModelFileTransferSourceEnum] = None
    transfer_profile_id: Optional[int] = None
    source_worker_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


ModelStorageSyncTasksPublic = PaginatedList[ModelStorageSyncTaskPublic]


# ---------------------------------------------------------------------------
# 任务 3：模型同步 API 的 Public / Create 契约
# ---------------------------------------------------------------------------


class ModelStorageSyncTaskCreate(SQLModel):
    """同步任务创建请求。

    只接受 ``model_file_id`` 与 ``profile_id``：Server 从 ModelFile 推导请求
    身份、本地路径与目标对象 Key，浏览器不得提交任意目标路径或对象 Key。
    """

    model_config = ConfigDict(extra="forbid")

    model_file_id: int
    profile_id: int


class ModelStorageSyncTaskProfilePublic(SQLModel):
    """同步任务详情中内嵌的 S3 Profile 摘要（组装时读取，不复制凭据）。"""

    id: int
    name: str
    config_version: int
    system_managed: bool


class ModelStorageSyncTaskDetail(SQLModel):
    """同步任务详情：分别返回模型 source、本次 transfer_source、S3 Profile 与来源
    Worker，字段不混用；不含凭据快照或加密字段。"""

    id: int
    model_file_id: int
    worker_id: int
    worker_uuid: str
    # 模型身份来源（ModelScope / Hugging Face）。
    source: str
    model_id: str
    resolved_revision: str
    request_digest: str
    profile_config_version: int
    profile: Optional[ModelStorageSyncTaskProfilePublic] = None
    # 本次传输路径与来源 Worker（与模型身份分离记录）。
    transfer_source: Optional[ModelFileTransferSourceEnum] = None
    transfer_profile_id: Optional[int] = None
    source_worker_id: Optional[int] = None
    source_worker_name: Optional[str] = None
    artifact_id: Optional[str] = None
    state: ModelStorageSyncTaskStateEnum
    state_message: Optional[str] = None
    error_code: Optional[str] = None
    file_count: int
    total_size: int
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class ModelStorageSyncCapabilitiesPublic(SQLModel):
    """``GET /model-storage/capabilities``：只返回布尔能力，不返回密钥或敏感配置。"""

    credential_encryption_available: bool


class ModelStorageConnectionTestRequest(SQLModel):
    """``POST /model-storage/connection-tests``：接受尚未保存的 Profile 表单。

    凭据只用于本次 Server 侧短生命周期检查，不入库、不写日志、不进入 SSE。
    """

    endpoint: str
    bucket: str
    prefix: str = ""
    access_key: str = Field(repr=False)
    secret_key: str = Field(repr=False)
    tls_enabled: bool = True
    tls_verify: bool = True
    region: Optional[str] = None
    use_virtual_hosted_style: bool = True


class ModelStorageConnectionStagePublic(SQLModel):
    """单个阶段（连接/Bucket/写/读/删除）的结果；权限不足不被折叠为笼统连接失败。"""

    ok: bool
    error_code: Optional[str] = None


class ModelStorageConnectionTestPublic(SQLModel):
    """连接测试结果：``scope=server``，分阶段报告连接、Bucket、写、读、删除。"""

    scope: str = "server"
    ok: bool
    connection: ModelStorageConnectionStagePublic
    bucket: ModelStorageConnectionStagePublic
    write: ModelStorageConnectionStagePublic
    read: ModelStorageConnectionStagePublic
    delete: ModelStorageConnectionStagePublic
    error_code: Optional[str] = None


# ---------------------------------------------------------------------------
# 任务 3：Worker 侧同步执行（受 Worker 身份约束）的契约
# ---------------------------------------------------------------------------


class ModelStorageSyncExecutionProfile(SQLModel):
    """仅供当前 Worker 使用的明文 S3 连接配置（受 Worker 身份约束）。

    不进 Public schema、SSE 或日志；凭据只进入该执行 payload。
    """

    endpoint: str
    bucket: str
    prefix: str = ""
    tls_enabled: bool = True
    tls_verify: bool = True
    region: str = ""
    use_virtual_hosted_style: bool = True
    access_key: str = Field(repr=False)
    secret_key: str = Field(repr=False)


class ModelStorageSyncExecutionPayload(SQLModel):
    """Worker 领取后拉取一次性的执行配置（含解密后的 Profile 凭据）。"""

    task_id: int
    state: ModelStorageSyncTaskStateEnum
    source: str
    model_id: str
    resolved_revision: str
    request_identity: dict
    request_digest: str
    # 可信本地源路径（ModelFile.resolved_paths），供 Worker 扫描本地模型。
    source_paths: list[str]
    profile: ModelStorageSyncExecutionProfile


class ModelStorageSyncTaskComplete(SQLModel):
    """Worker 完成：CAS 绑定 ``artifact_id``（仅从 NULL），并回写文件数/容量。"""

    artifact_id: str
    file_count: int
    total_size: int


class ModelStorageSyncTaskFail(SQLModel):
    """Worker 失败：回写稳定错误码。"""

    error_code: str
# 避免未使用告警，同时保证外键引用的表在仅导入本模块时也可 create_all。
__all__ = [
    "ModelStorageSyncTask",
    "ModelStorageSyncTaskPublic",
    "ModelStorageSyncTasksPublic",
    "ModelStorageSyncTaskStateEnum",
    "ModelStorageSyncTaskCreate",
    "ModelStorageSyncTaskProfilePublic",
    "ModelStorageSyncTaskDetail",
    "ModelStorageSyncCapabilitiesPublic",
    "ModelStorageConnectionTestRequest",
    "ModelStorageConnectionStagePublic",
    "ModelStorageConnectionTestPublic",
]
