from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import ConfigDict
from sqlalchemy import Column
from sqlmodel import Field, SQLModel, Text
from gpustack.mixins.active_record import ActiveRecordMixin
from gpustack.schemas.common import UTCDateTime


class OperationEnum(str, Enum):
    COMPLETION = "completion"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    IMAGE_GENERATION = "image_generation"
    AUDIO_SPEECH = "audio_speech"
    AUDIO_TRANSCRIPTION = "audit_transcription"


class ModelUsage(SQLModel, ActiveRecordMixin, table=True):
    __tablename__ = 'model_usages'
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(default=None, foreign_key="users.id")
    model_id: int = Field(default=None, foreign_key="models.id")
    date: date
    prompt_token_count: int
    completion_token_count: int
    request_count: int
    operation: OperationEnum

    model_config = ConfigDict(protected_namespaces=())


class ModelUsageLog(SQLModel, ActiveRecordMixin, table=True):
    __tablename__ = 'model_usage_logs'
    id: Optional[int] = Field(default=None, primary_key=True)
    request_id: Optional[str] = Field(default=None, index=True)
    call_time: datetime = Field(sa_column=Column(UTCDateTime(), index=True))
    date: date = Field(index=True)
    hour: int = Field(index=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    api_key_id: Optional[int] = Field(
        default=None, foreign_key="api_keys.id", index=True
    )
    api_key_access_key: Optional[str] = None
    model_id: Optional[int] = Field(default=None, foreign_key="models.id", index=True)
    model_name: Optional[str] = Field(default=None, index=True)
    operation: OperationEnum = Field(index=True)
    source_ip: Optional[str] = Field(default=None, index=True)
    raw_forwarded_for: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    prompt_token_count: int = 0
    completion_token_count: int = 0
    total_token_count: int = 0
    usage_available: bool = False
    status_code: Optional[int] = Field(default=None, index=True)
    success: bool = Field(default=False, index=True)
    duration_ms: Optional[int] = None
    ttft_ms: Optional[int] = None
    tokens_per_second: Optional[float] = None
    error_code: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    worker_id: Optional[int] = Field(default=None, index=True)
    worker_name: Optional[str] = None
    worker_ip: Optional[str] = None
    model_instance_id: Optional[int] = Field(default=None, index=True)

    model_config = ConfigDict(protected_namespaces=())
