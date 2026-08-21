from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, ForeignKey, JSON, Text, UniqueConstraint
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList, UTCDateTime


class ModelFileDownloadExecutionStateEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    READY = "ready"
    ERROR = "error"
    CANCELED = "canceled"


# 传输来源，与模型身份（source/model_id/revision）分离记录。
# 命中 S3 时 source 仍是 ModelScope/Hugging Face，transfer_source=s3 仅表示本次从 S3 获取。
class ModelFileTransferSourceEnum(str, Enum):
    CURRENT_NODE = "current_node"
    PEER_VIA_S3 = "peer_via_s3"
    S3 = "s3"
    MODELSCOPE = "modelscope"
    HUGGINGFACE = "huggingface"


class ModelFileDownloadExecution(SQLModel, BaseModelMixin, table=True):
    """普通下载私有执行配置。

    由路由创建与实例控制器创建两条路径在同一事务中为 ModelFile 创建唯一执行记录，
    固定 request identity、目标 Worker、默认 Profile ID/config version 和 AES-GCM
    凭据快照。没有默认 Profile 时仍创建明确的无 S3 执行记录（profile 字段为 NULL）。
    凭据快照只进入受 Worker 身份约束的领取 payload，不进入 Public schema、SSE 或日志。
    """

    __tablename__ = "model_file_download_executions"

    __table_args__ = (
        UniqueConstraint(
            "model_file_id", name="uix_model_file_download_execution_model_file"
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    # 每个 ModelFile 唯一执行记录，创建后固定不变。
    model_file_id: int = Field(
        sa_column=Column(
            ForeignKey("model_files.id", ondelete="CASCADE"), nullable=False
        ),
    )
    # 固定请求身份及其摘要；移动 revision 领取时才解析 resolved revision。
    request_identity: dict = Field(sa_column=Column(JSON, nullable=False))
    request_digest: str
    # 目标 Worker 身份；只允许版本匹配且属于该 ModelFile 的 Worker 领取。
    target_worker_id: int = Field(
        sa_column=Column(
            ForeignKey("workers.id", ondelete="CASCADE"), nullable=False
        )
    )
    target_worker_uuid: str
    # 默认 Profile 与其配置版本；无默认 Profile 时为 NULL（明确的无 S3 执行）。
    default_profile_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("model_preheat_s3_profiles.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    default_profile_config_version: Optional[int] = None
    # 任务私有加密凭据快照（AES-GCM），仅进入受 Worker 身份约束的领取 payload。
    credential_snapshot_encrypted: Optional[dict] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    encryption_key_version: Optional[str] = None
    state: ModelFileDownloadExecutionStateEnum = (
        ModelFileDownloadExecutionStateEnum.PENDING
    )
    # 领取与执行固定配置
    claimed_by_worker_uuid: Optional[str] = None
    claimed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )
    # 任务执行结果来源字段
    transfer_source: Optional[ModelFileTransferSourceEnum] = None
    transfer_profile_id: Optional[int] = None
    source_worker_id: Optional[int] = None
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    error_code: Optional[str] = None
    finished_at: Optional[datetime] = Field(
        default=None, sa_column=Column(UTCDateTime, nullable=True)
    )


class ModelFileDownloadExecutionPublic(SQLModel):
    """Public schema 不返回凭据快照或加密字段。"""

    model_file_id: int
    request_digest: str
    target_worker_id: int
    default_profile_id: Optional[int] = None
    default_profile_config_version: Optional[int] = None
    state: ModelFileDownloadExecutionStateEnum
    transfer_source: Optional[ModelFileTransferSourceEnum] = None
    transfer_profile_id: Optional[int] = None
    source_worker_id: Optional[int] = None
    state_message: Optional[str] = None
    error_code: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    finished_at: Optional[datetime] = None


ModelFileDownloadExecutionsPublic = PaginatedList[ModelFileDownloadExecutionPublic]
