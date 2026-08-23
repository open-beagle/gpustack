from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, ForeignKey, Index, JSON, Text, UniqueConstraint
from sqlmodel import Field, SQLModel
from pydantic import ConfigDict
from pydantic import field_validator

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

    任务创建时固定 request identity（含冻结的执行文件选择扫描规约）、
    Profile ID、配置版本和加密凭据快照，``artifact_id`` 可为空；库存精确
    命中时直接固定 Artifact，否则 Worker 扫描后使用任务、request digest、
    Worker UUID 和 lease token 将 ``artifact_id`` 从 NULL CAS 绑定。后续
    修改默认 Profile 或凭据不改变已创建任务。
    """

    __tablename__ = "model_storage_sync_tasks"
    __table_args__ = (
        Index(
            "ix_model_storage_sync_model_file_profile", "model_file_id", "profile_id"
        ),
        Index("ix_model_storage_sync_state", "state"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    model_file_id: int = Field(
        sa_column=Column(
            ForeignKey("model_files.id", ondelete="CASCADE"), nullable=False
        ),
    )
    worker_id: int = Field(
        sa_column=Column(ForeignKey("workers.id", ondelete="RESTRICT"), nullable=False),
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
    # 任务私有加密执行快照（Profile 凭据 + 物理 source_paths/scan_spec），只进入
    # 受 Worker 身份约束的执行 payload，不进入 canonical request identity、
    # Public schema、SSE 或日志。
    credential_snapshot_encrypted: dict = Field(sa_column=Column(JSON, nullable=False))
    encryption_key_version: str
    # 库存精确命中时直接固定，否则保持 NULL，由 Worker CAS 绑定。
    artifact_id: Optional[str] = None
    # 执行 lease token 的 AES-GCM 加密快照（任务私有，明文只进入受 Worker
    # 身份约束的执行 payload；complete/fail 用其验证执行归属，防串任务与
    # 过期重放）。
    lease_token_encrypted: Optional[dict] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    state: ModelStorageSyncTaskStateEnum = ModelStorageSyncTaskStateEnum.PENDING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    error_code: Optional[str] = None
    file_count: int = 0
    total_size: int = 0
    # 完成时固定的发布结果（等价重放幂等判定使用：同一已完成任务的同一
    # lease 重放必须与这些固定值一致）。
    manifest_digest: Optional[str] = None
    manifest_path: Optional[str] = None
    # 任务执行结果来源字段
    transfer_source: Optional[ModelFileTransferSourceEnum] = None
    transfer_profile_id: Optional[int] = None
    source_worker_id: Optional[int] = None
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


class ModelStorageSyncTaskDedupeSlot(SQLModel, BaseModelMixin, table=True):
    """活动同步任务的数据库级去重槽。

    同一 ``(model_file_id, profile_id)`` 同时只允许一个活动任务
    （pending/scanning/publishing）：活动任务持有 ``dedupe_key`` 唯一槽位，
    进入终态（ready/error/canceled）时在同一事务中删除槽位行。``dedupe_key``
    唯一约束是三库（SQLite/PostgreSQL/MySQL）通用的数据库级并发保证：竞争
    创建时后到者得到唯一冲突并整体回滚，而不是重复任务。
    """

    __tablename__ = "model_storage_sync_task_dedupe_slots"
    __table_args__ = (
        UniqueConstraint("dedupe_key", name="uix_model_storage_sync_dedupe_key"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    dedupe_key: str
    task_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("model_storage_sync_tasks.id", ondelete="CASCADE"),
            nullable=True,
        ),
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
    source_worker_name: Optional[str] = None
    profile_name: Optional[str] = None
    profile_endpoint: Optional[str] = None
    profile_bucket: Optional[str] = None
    profile_prefix: Optional[str] = None
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


class ModelStorageSyncScopeEnum(str, Enum):
    SINGLE_MODEL = "single_model"
    SELECTED_WORKERS = "selected_workers"
    ALL_READY_WORKERS = "all_ready_workers"


class ModelStorageSyncBatchCreate(SQLModel):
    profile_id: int
    scope: ModelStorageSyncScopeEnum = ModelStorageSyncScopeEnum.SINGLE_MODEL
    model_file_id: Optional[int] = None
    worker_ids: list[int] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ModelStorageSyncBatchItem(SQLModel):
    model_file_id: Optional[int] = None
    worker_id: Optional[int] = None
    task_id: Optional[int] = None
    reason: Optional[str] = None


class ModelStorageSyncBatchPublic(SQLModel):
    scope: ModelStorageSyncScopeEnum
    planned: int
    created: list[ModelStorageSyncBatchItem] = Field(default_factory=list)
    skipped: list[ModelStorageSyncBatchItem] = Field(default_factory=list)
    failed: list[ModelStorageSyncBatchItem] = Field(default_factory=list)


class ModelStorageSyncBatchResult(SQLModel, table=True):
    """顶层 Idempotency-Key 对应的完整批量同步响应快照。"""

    __tablename__ = "model_storage_sync_batch_results"
    __table_args__ = (
        UniqueConstraint(
            "idempotency_record_id",
            name="uix_model_storage_sync_batch_result_idempotency",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    idempotency_record_id: int = Field(
        sa_column=Column(
            ForeignKey("model_preheat_idempotency_records.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    response_payload: dict = Field(sa_column=Column(JSON, nullable=False))


class ModelStorageSyncTaskProfilePublic(SQLModel):
    """同步任务详情中内嵌的 S3 Profile 摘要（组装时读取，不复制凭据）。"""

    id: int
    name: str
    endpoint: Optional[str] = None
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    tls_enabled: Optional[bool] = None
    tls_verify: Optional[bool] = None
    region: Optional[str] = None
    use_virtual_hosted_style: Optional[bool] = None
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
    """Worker 领取后拉取一次性的执行配置（含解密后的 Profile 凭据）。

    所有执行语义都在任务创建时冻结：``scan_spec`` 是冻结的文件选择扫描
    规约（root + include/exclude patterns），``source_paths`` 是创建时冻结
    的源路径；Worker 不得重读当前 ModelFile 或按当前状态重算规约。
    ``lease_token`` 为本次执行的一次性 lease：complete/fail 必须携带，
    Server 校验通过后任务绑定才生效。
    """

    task_id: int
    state: ModelStorageSyncTaskStateEnum
    source: str
    model_id: str
    resolved_revision: str
    request_identity: dict
    request_digest: str
    # 冻结的可信本地源路径（创建时固定的 ModelFile.resolved_paths）。
    source_paths: list[str]
    # 冻结的扫描规约：root 与 include/exclude patterns（创建时由
    # compute_scan_spec 计算，与 request_identity.include_patterns 一致）。
    scan_spec: dict
    # 一次性执行 lease token（complete/fail 必须回传）。
    lease_token: str
    profile: ModelStorageSyncExecutionProfile


class ModelStorageSyncTaskComplete(SQLModel):
    """Worker 完成：CAS 绑定/确认 ``artifact_id`` 并推进终态。

    契约约束（任务 3 子阶段 C）：

    - ``lease_token`` 必须等于任务执行 payload 签发的一次性 lease：
      Server 据此拒绝无 lease、错 lease（串任务）的完成请求。
    - ``request_digest`` 必须等于任务创建时固定的 ``request_digest``。
    - ``artifact_id`` 必须与任务当前绑定一致：未绑定时由本请求绑定，已预绑定
      时只允许确认同一 artifact（CAS 保证，不覆盖为其他值）。
    - ``file_count`` / ``total_size`` 必须为非负整数。
    - ``manifest_digest`` 为发布 Manifest 的 SHA-256（64 位小写十六进制），
      必选：用于写入统一 Artifact 库存（库存要求 digest 非空）。
    - ``manifest_path`` 为 Manifest 对象 Key，必选；必须由 Worker 使用任务
      创建时冻结的 Profile 快照生成，Server 不读取当前 Profile prefix 推导。

    等价重放幂等：任务已处于 ready 终态且 ``request_digest``/``lease_token``/
    ``artifact_id``/``manifest_digest`` 与完成时固定值全部一致时，重复
    complete 幂等成功（200，不重复写库存）；任何一项不一致（不同 artifact
    或过期执行）稳定冲突（409）。
    """

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    request_digest: str
    lease_token: str
    file_count: int = Field(ge=0)
    total_size: int = Field(ge=0)
    manifest_digest: str
    manifest_path: str = Field(min_length=1)

    @field_validator("manifest_digest")
    @classmethod
    def _validate_manifest_digest(cls, value: str) -> str:
        """manifest_digest 必须是 64 位小写十六进制 SHA-256。"""
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("manifest_digest must be sha256 hex")
        return value


class ModelStorageSyncTaskFail(SQLModel):
    """Worker 失败：回写稳定错误码（需携带执行 lease token）。"""

    lease_token: str
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
