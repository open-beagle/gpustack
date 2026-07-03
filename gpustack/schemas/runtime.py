from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class RuntimeModelInstance(BaseModel):
    model_id: Optional[int] = None
    model_instance_id: int
    model_name: Optional[str] = None
    backend: Optional[str] = None
    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_ip: Optional[str] = None
    endpoint: Optional[str] = None
    health_endpoint: Optional[str] = None
    metrics_endpoint: Optional[str] = None
    pid: Optional[int] = None
    child_pids: List[int] = Field(default_factory=list)
    ports: List[int] = Field(default_factory=list)
    gpu_indexes: List[int] = Field(default_factory=list)
    gpu_addresses: List[str] = Field(default_factory=list)
    state: Optional[str] = None
    updated_at: Optional[datetime] = None


class RuntimeModelInstancesResponse(BaseModel):
    instances: List[RuntimeModelInstance]
