from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import JSON, BigInteger, Column, ForeignKey, Text
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import PaginatedList


class ModelCacheTaskStateEnum(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    READY = "ready"
    ERROR = "error"


class ModelCacheTask(SQLModel, BaseModelMixin, table=True):
    __tablename__ = "model_cache_tasks"

    id: Optional[int] = Field(default=None, primary_key=True)
    model_file_id: int = Field(
        sa_column=Column(ForeignKey("model_files.id", ondelete="RESTRICT"), nullable=False)
    )
    worker_id: int = Field(
        sa_column=Column(ForeignKey("workers.id", ondelete="RESTRICT"), nullable=False)
    )
    model_id: str = Field(index=True)
    target_path: str = Field(sa_column=Column(Text, nullable=False))
    source_paths: list[str] = Field(sa_column=Column(JSON, nullable=False))
    state: ModelCacheTaskStateEnum = ModelCacheTaskStateEnum.PENDING
    progress: float = 0
    uploaded_size: int = Field(default=0, sa_column=Column(BigInteger, nullable=False))
    total_size: int = Field(default=0, sa_column=Column(BigInteger, nullable=False))
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_by_user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
    )
    finished_at: Optional[datetime] = None


class ModelCacheTaskCreate(SQLModel):
    model_id: str


class ModelCacheTaskUpdate(SQLModel):
    state: ModelCacheTaskStateEnum
    progress: float = 0
    uploaded_size: int = 0
    total_size: int = 0
    error_message: Optional[str] = None


class ModelCacheTaskPublic(SQLModel):
    id: int
    model_file_id: int
    worker_id: int
    model_id: str
    target_path: str
    source_paths: list[str]
    state: ModelCacheTaskStateEnum
    progress: float
    uploaded_size: int
    total_size: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    finished_at: Optional[datetime]


ModelCacheTasksPublic = PaginatedList[ModelCacheTaskPublic]


class ModelCacheModelPublic(SQLModel):
    model_id: str
    s3_path: str
    file_count: int
    total_size: int
    updated_at: datetime


class ModelCacheModelsPublic(SQLModel):
    items: list[ModelCacheModelPublic]


class ModelCacheFilePublic(SQLModel):
    path: str
    size: int
    updated_at: datetime


class ModelCacheFilesPublic(SQLModel):
    items: list[ModelCacheFilePublic]


class ModelCacheDeleteResult(SQLModel):
    model_id: str
    deleted_file_count: int
    deleted_size: int
