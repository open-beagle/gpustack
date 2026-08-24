from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import model_validator
from sqlmodel import JSON, BigInteger, Column, Field, Relationship, SQLModel, Text

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList
from gpustack.schemas.links import ModelInstanceModelFileLink
from gpustack.schemas.model_file_download_executions import (
    ModelFileTransferSourceEnum,
)
from gpustack.schemas.models import ModelSource, ModelInstance


class ModelFileStateEnum(str, Enum):
    ERROR = "error"
    DOWNLOADING = "downloading"
    READY = "ready"


class ModelFileBase(SQLModel, ModelSource):
    local_dir: Optional[str] = None
    worker_id: Optional[int] = None
    cleanup_on_delete: Optional[bool] = None

    size: Optional[int] = Field(sa_column=Column(BigInteger), default=None)
    download_progress: Optional[float] = None
    resolved_paths: List[str] = Field(sa_column=Column(JSON), default=[])
    state: ModelFileStateEnum = ModelFileStateEnum.DOWNLOADING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # 节点模型下载成功时保存 requested/resolved revision；Hugging Face、ModelScope
    # 新下载必填，其他来源可为空。resolved revision 与 Artifact/Manifest 保持一致。
    requested_revision: Optional[str] = None
    resolved_revision: Optional[str] = None


class ModelFile(ModelFileBase, BaseModelMixin, table=True):
    __tablename__ = 'model_files'
    id: Optional[int] = Field(default=None, primary_key=True)

    # Unique index of the model source
    source_index: Optional[str] = Field(index=True, unique=True, default=None)
    worker_uuid_snapshot: Optional[str] = None
    worker_name_snapshot: Optional[str] = None

    instances: list[ModelInstance] = Relationship(
        sa_relationship_kwargs={"lazy": "selectin"},
        back_populates="model_files",
        link_model=ModelInstanceModelFileLink,
    )


class ModelFileCreate(ModelFileBase):
    pass


class ModelFileUpdate(ModelFileBase):
    pass


class ModelFilePublic(
    ModelFileBase,
):
    id: int
    transfer_source: Optional[ModelFileTransferSourceEnum] = None
    transfer_profile_id: Optional[int] = None
    transfer_profile_name: Optional[str] = None
    source_worker_id: Optional[int] = None
    source_worker_name: Optional[str] = None
    worker_uuid_snapshot: Optional[str] = None
    worker_name_snapshot: Optional[str] = None
    worker_name: Optional[str] = None
    worker_available: bool = False
    revision_kind: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def infer_revision_kind(self):
        if self.revision_kind is None:
            if self.resolved_revision:
                self.revision_kind = (
                    "local_snapshot"
                    if self.resolved_revision.startswith("local-snapshot-")
                    else "upstream"
                )
            elif self.state == ModelFileStateEnum.READY and self.resolved_paths:
                self.revision_kind = "local_snapshot"
        return self


ModelFilesPublic = PaginatedList[ModelFilePublic]
