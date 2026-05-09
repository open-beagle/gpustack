from datetime import date as Date, datetime as DateTime
from enum import Enum
from typing import Optional

from pydantic import ConfigDict
from sqlalchemy import Column, UniqueConstraint
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
    date: Date
    prompt_token_count: int
    completion_token_count: int
    request_count: int
    operation: OperationEnum

    model_config = ConfigDict(protected_namespaces=())


class ModelUsageLog(SQLModel, ActiveRecordMixin, table=True):
    __tablename__ = 'model_usage_logs'
    id: Optional[int] = Field(default=None, primary_key=True)
    request_id: Optional[str] = Field(default=None, index=True)
    call_time: DateTime = Field(sa_column=Column(UTCDateTime(), index=True))
    date: Date = Field(index=True)
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


class ModelUsageHourlyStat(SQLModel, ActiveRecordMixin, table=True):
    __tablename__ = 'model_usage_hourly_stats'
    __table_args__ = (
        UniqueConstraint(
            'date',
            'hour',
            'api_key_id',
            'model_id',
            'source_ip',
            'operation',
            'worker_id',
            name='uix_model_usage_hourly_stats_dimensions',
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    date: Date = Field(index=True)
    hour: int = Field(index=True)
    api_key_id: int = Field(default=0, index=True)
    api_key_access_key: Optional[str] = None
    model_id: int = Field(default=0, index=True)
    model_name: Optional[str] = Field(default=None, index=True)
    source_ip: str = Field(default="", index=True)
    operation: OperationEnum = Field(index=True)
    worker_id: int = Field(default=0, index=True)
    worker_name: Optional[str] = None
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    prompt_token_count: int = 0
    completion_token_count: int = 0
    total_token_count: int = 0
    duration_ms_sum: int = 0
    last_call_time: DateTime = Field(sa_column=Column(UTCDateTime(), index=True))

    model_config = ConfigDict(protected_namespaces=())


class ModelUsageDailyStat(SQLModel, ActiveRecordMixin, table=True):
    __tablename__ = 'model_usage_daily_stats'
    __table_args__ = (
        UniqueConstraint(
            'date',
            'api_key_id',
            'model_id',
            'source_ip',
            'operation',
            'worker_id',
            name='uix_model_usage_daily_stats_dimensions',
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)
    date: Date = Field(index=True)
    api_key_id: int = Field(default=0, index=True)
    api_key_access_key: Optional[str] = None
    model_id: int = Field(default=0, index=True)
    model_name: Optional[str] = Field(default=None, index=True)
    source_ip: str = Field(default="", index=True)
    operation: OperationEnum = Field(index=True)
    worker_id: int = Field(default=0, index=True)
    worker_name: Optional[str] = None
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    prompt_token_count: int = 0
    completion_token_count: int = 0
    total_token_count: int = 0
    duration_ms_sum: int = 0
    last_call_time: DateTime = Field(sa_column=Column(UTCDateTime(), index=True))

    model_config = ConfigDict(protected_namespaces=())
