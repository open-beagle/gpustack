import csv
import io
from datetime import date, datetime, time, timezone
from typing import Iterable, Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import Select, and_, case, or_
from sqlmodel import func, select

from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.model_usage import (
    ModelUsageDailyStat,
    ModelUsageHourlyStat,
    ModelUsageLog,
    OperationEnum,
)
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker
from gpustack.server.deps import SessionDep
from gpustack.server.services import ModelUsageStatService


router = APIRouter()


def stat_aggregate_columns(source):
    return [
        func.sum(source.c.request_count).label("request_count"),
        func.sum(source.c.success_count).label("success_count"),
        func.sum(source.c.failure_count).label("failure_count"),
        func.sum(source.c.prompt_tokens).label("prompt_tokens"),
        func.sum(source.c.completion_tokens).label("completion_tokens"),
        func.sum(source.c.total_tokens).label("total_tokens"),
        distinct_non_zero(source.c.api_key_id).label("api_key_count"),
        distinct_non_zero(source.c.model_id).label("model_count"),
        distinct_non_empty(source.c.source_ip).label("source_ip_count"),
        func.max(source.c.last_call_time).label("last_call_time"),
        func.sum(source.c.duration_ms_sum).label("duration_ms_sum"),
    ]


def log_aggregate_columns(source):
    return [
        func.count(source.c.id).label("request_count"),
        func.sum(case((source.c.success == True, 1), else_=0)).label("success_count"),
        func.sum(case((source.c.success == False, 1), else_=0)).label("failure_count"),
        func.sum(source.c.prompt_token_count).label("prompt_tokens"),
        func.sum(source.c.completion_token_count).label("completion_tokens"),
        func.sum(source.c.total_token_count).label("total_tokens"),
        func.count(func.distinct(source.c.api_key_id)).label("api_key_count"),
        func.count(func.distinct(source.c.model_id)).label("model_count"),
        func.count(func.distinct(source.c.source_ip)).label("source_ip_count"),
        func.max(source.c.call_time).label("last_call_time"),
        func.sum(source.c.duration_ms).label("duration_ms_sum"),
    ]


def rolled_up_aggregate_columns(source):
    return [
        func.sum(source.c.request_count).label("request_count"),
        func.sum(source.c.success_count).label("success_count"),
        func.sum(source.c.failure_count).label("failure_count"),
        func.sum(source.c.prompt_tokens).label("prompt_tokens"),
        func.sum(source.c.completion_tokens).label("completion_tokens"),
        func.sum(source.c.total_tokens).label("total_tokens"),
        func.sum(source.c.api_key_count).label("api_key_count"),
        func.sum(source.c.model_count).label("model_count"),
        func.sum(source.c.source_ip_count).label("source_ip_count"),
        func.max(source.c.last_call_time).label("last_call_time"),
        func.sum(source.c.duration_ms_sum).label("duration_ms_sum"),
    ]


def distinct_non_zero(column):
    return func.count(func.distinct(case((column != 0, column), else_=None)))


def distinct_non_empty(column):
    return func.count(func.distinct(case((column != "", column), else_=None)))


def should_use_log_aggregation(
    start_at: Optional[datetime], end_at: Optional[datetime], status: Optional[str]
):
    if status:
        return True
    return not is_day_boundary_start(start_at) or not is_day_boundary_end(end_at)


def is_day_boundary_start(value: Optional[datetime]):
    return value is None or value.timetz().replace(tzinfo=None) == time.min


def is_day_boundary_end(value: Optional[datetime]):
    if value is None:
        return True
    value_time = value.timetz().replace(tzinfo=None)
    return value_time == time.max


def base_statement(
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_ip: Optional[str] = None,
    operation: Optional[OperationEnum] = None,
    status: Optional[str] = None,
    worker_id: Optional[int] = None,
) -> Select:
    statement = select(ModelUsageLog).outerjoin(
        ApiKey, ModelUsageLog.api_key_id == ApiKey.id
    )
    if start_at is not None:
        statement = statement.where(ModelUsageLog.call_time >= start_at)
    if end_at is not None:
        statement = statement.where(ModelUsageLog.call_time <= end_at)
    if api_key:
        api_key_filters = [
            ApiKey.name.contains(api_key),
            ModelUsageLog.api_key_access_key.contains(api_key),
        ]
        if api_key.isdigit():
            api_key_filters.append(ModelUsageLog.api_key_id == int(api_key))
        condition = or_(api_key_filters[0], api_key_filters[1])
        if len(api_key_filters) > 2:
            condition = or_(condition, api_key_filters[2])
        statement = statement.where(condition)
    if model:
        statement = statement.where(ModelUsageLog.model_name.contains(model))
    if source_ip:
        statement = statement.where(ModelUsageLog.source_ip == source_ip)
    if operation:
        statement = statement.where(ModelUsageLog.operation == operation)
    if status == "success":
        statement = statement.where(ModelUsageLog.success == True)
    elif status == "failure":
        statement = statement.where(ModelUsageLog.success == False)
    if worker_id is not None:
        statement = statement.where(ModelUsageLog.worker_id == worker_id)
    return statement


def stat_statement(
    stat_cls,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_ip: Optional[str] = None,
    operation: Optional[OperationEnum] = None,
    status: Optional[str] = None,
    worker_id: Optional[int] = None,
) -> Select:
    selected_columns = [
        stat_cls.id.label("id"),
        stat_cls.date.label("date"),
        stat_cls.api_key_id.label("api_key_id"),
        stat_cls.api_key_access_key.label("api_key_access_key"),
        stat_cls.model_id.label("model_id"),
        stat_cls.model_name.label("model_name"),
        stat_cls.source_ip.label("source_ip"),
        stat_cls.operation.label("operation"),
        stat_cls.worker_id.label("worker_id"),
        stat_cls.worker_name.label("worker_name"),
        stat_cls.request_count.label("request_count"),
        stat_cls.success_count.label("success_count"),
        stat_cls.failure_count.label("failure_count"),
        stat_cls.prompt_token_count.label("prompt_tokens"),
        stat_cls.completion_token_count.label("completion_tokens"),
        stat_cls.total_token_count.label("total_tokens"),
        stat_cls.duration_ms_sum.label("duration_ms_sum"),
        stat_cls.last_call_time.label("last_call_time"),
    ]
    if hasattr(stat_cls, "hour"):
        selected_columns.append(stat_cls.hour.label("hour"))

    statement = select(*selected_columns).outerjoin(ApiKey, stat_cls.api_key_id == ApiKey.id)
    if start_at is not None:
        statement = statement.where(stat_cls.date >= start_at.date())
    if end_at is not None:
        statement = statement.where(stat_cls.date <= end_at.date())
    if api_key:
        condition = or_(
            ApiKey.name.contains(api_key),
            stat_cls.api_key_access_key.contains(api_key),
        )
        if api_key.isdigit():
            condition = or_(condition, stat_cls.api_key_id == int(api_key))
        statement = statement.where(condition)
    if model:
        statement = statement.where(stat_cls.model_name.contains(model))
    if source_ip:
        statement = statement.where(stat_cls.source_ip == source_ip)
    if operation:
        statement = statement.where(stat_cls.operation == operation)
    if status == "success":
        statement = statement.where(stat_cls.success_count > 0)
    elif status == "failure":
        statement = statement.where(stat_cls.failure_count > 0)
    if worker_id is not None:
        statement = statement.where(stat_cls.worker_id == (worker_id or 0))
    return statement


def aggregate_source(
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_ip: Optional[str] = None,
    operation: Optional[OperationEnum] = None,
    status: Optional[str] = None,
    worker_id: Optional[int] = None,
):
    if should_use_log_aggregation(start_at, end_at, status):
        logs = base_statement(
            start_at, end_at, api_key, model, source_ip, operation, status, worker_id
        ).subquery()
        source = aggregate_statement(
            logs,
            [
                logs.c.date,
                logs.c.api_key_id,
                logs.c.api_key_access_key,
                logs.c.model_id,
                logs.c.model_name,
                logs.c.source_ip,
                logs.c.operation,
                logs.c.worker_id,
                logs.c.worker_name,
            ],
            log_aggregate_columns(logs),
        ).subquery()
        return source, stat_aggregate_columns(source), source.c.total_tokens
    source = stat_statement(
        ModelUsageDailyStat,
        start_at, end_at, api_key, model, source_ip, operation, status, worker_id
    ).subquery()
    return source, stat_aggregate_columns(source), source.c.total_tokens


def hourly_source(
    start_at: datetime,
    end_at: datetime,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_ip: Optional[str] = None,
    operation: Optional[OperationEnum] = None,
    status: Optional[str] = None,
    worker_id: Optional[int] = None,
):
    if status:
        logs = base_statement(
            start_at, end_at, api_key, model, source_ip, operation, status, worker_id
        ).subquery()
        source = aggregate_statement(
            logs,
            [logs.c.hour],
            log_aggregate_columns(logs),
        ).subquery()
        return source, rolled_up_aggregate_columns(source)
    source = stat_statement(
        ModelUsageHourlyStat,
        start_at, end_at, api_key, model, source_ip, operation, status, worker_id
    )
    if start_at is not None:
        source = source.where(ModelUsageHourlyStat.hour >= start_at.hour)
    if end_at is not None and end_at.time() != time.max:
        source = source.where(ModelUsageHourlyStat.hour <= end_at.hour)
    source = source.subquery()
    return source, stat_aggregate_columns(source)


def aggregate_statement(source, group_columns, aggregate_columns, order_column=None):
    statement = select(*group_columns, *aggregate_columns).select_from(source)
    if group_columns:
        statement = statement.group_by(*group_columns)
    if order_column is not None:
        statement = statement.order_by(func.sum(order_column).desc())
    return statement


async def paginate(session: SessionDep, statement: Select, page: int, page_size: int):
    total_statement = select(func.count()).select_from(statement.subquery())
    total = (await session.exec(total_statement)).one()
    rows = (await session.exec(statement.offset((page - 1) * page_size).limit(page_size))).all()
    return rows, total


def to_iso(value: Optional[datetime]):
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def to_int(value):
    return int(value or 0)


def to_float(value):
    return float(value or 0)


@router.get("/overview")
async def overview(
    session: SessionDep,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_ip: Optional[str] = None,
    operation: Optional[OperationEnum] = None,
    status: Optional[str] = Query(None, pattern="^(success|failure)$"),
    worker_id: Optional[int] = None,
):
    filtered, _, _ = aggregate_source(
        start_at, end_at, api_key, model, source_ip, operation, status, worker_id
    )
    statement = select(
        func.sum(filtered.c.request_count).label("total_requests"),
        func.sum(filtered.c.success_count).label("success_requests"),
        func.sum(filtered.c.failure_count).label("failure_requests"),
        func.sum(filtered.c.prompt_tokens).label("prompt_tokens"),
        func.sum(filtered.c.completion_tokens).label("completion_tokens"),
        func.sum(filtered.c.total_tokens).label("total_tokens"),
        distinct_non_zero(filtered.c.api_key_id).label("api_key_count"),
        distinct_non_zero(filtered.c.model_id).label("model_count"),
        distinct_non_empty(filtered.c.source_ip).label("source_ip_count"),
        func.sum(filtered.c.duration_ms_sum).label("duration_ms_sum"),
    )
    result = (await session.exec(statement)).one()
    total_requests = to_int(result.total_requests)
    success_requests = to_int(result.success_requests)
    return {
        "total_requests": total_requests,
        "success_requests": success_requests,
        "failure_requests": to_int(result.failure_requests),
        "success_rate": success_requests / total_requests if total_requests else 0,
        "prompt_tokens": to_int(result.prompt_tokens),
        "completion_tokens": to_int(result.completion_tokens),
        "total_tokens": to_int(result.total_tokens),
        "api_key_count": to_int(result.api_key_count),
        "model_count": to_int(result.model_count),
        "source_ip_count": to_int(result.source_ip_count),
        "avg_duration_ms": to_int(result.duration_ms_sum) // total_requests
        if total_requests
        else 0,
        "avg_vram_usage_rate": None,
    }


@router.post("/rebuild-stats")
async def rebuild_stats(
    session: SessionDep,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    if start_date and end_date and start_date > end_date:
        raise BadRequestException("start_date must be less than or equal to end_date")
    rebuilt_log_count = await ModelUsageStatService(session).rebuild(start_date, end_date)
    return {
        "rebuilt_log_count": rebuilt_log_count,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
    }


@router.get("/server-summary")
async def server_summary(session: SessionDep):
    workers = (await session.exec(select(Worker))).all()
    model_instances = (await session.exec(select(ModelInstance))).all()
    total_tokens_by_worker = await get_total_tokens_by_worker(session)
    allocated_vram_by_worker = allocated_vram_by_worker_id(model_instances)
    items = []
    for worker in workers:
        gpus = worker.status.gpu_devices if worker.status else []
        gpu_count = len(gpus or [])
        gpu_memory_bytes = sum_gpu_memory(gpus, "total")
        allocated_vram_bytes = allocated_vram_by_worker.get(worker.id, 0)
        avg_gpu_usage_rate = average_gpu_rate(gpus, "core")
        avg_vram_usage_rate = average_gpu_rate(gpus, "memory")
        gpu_model = first_gpu_model(gpus)
        items.append(
            {
                "worker_id": worker.id,
                "worker_name": worker.name,
                "worker_ip": worker.ip,
                "gpu_model": gpu_model,
                "gpu_count": gpu_count,
                "gpu_memory_bytes": gpu_memory_bytes,
                "avg_gpu_usage_rate": avg_gpu_usage_rate,
                "avg_vram_usage_rate": avg_vram_usage_rate,
                "allocated_vram_bytes": allocated_vram_bytes,
                "total_tokens": total_tokens_by_worker.get(worker.id, 0),
                "remain_resource": remain_resource_text(
                    gpu_memory_bytes, allocated_vram_bytes, gpu_count
                ),
            }
        )
    return {"items": items}


async def get_total_tokens_by_worker(session: SessionDep):
    statement = (
        select(
            ModelUsageLog.worker_id,
            func.sum(ModelUsageLog.total_token_count).label("total_tokens"),
        )
        .where(ModelUsageLog.worker_id.is_not(None))
        .group_by(ModelUsageLog.worker_id)
    )
    rows = (await session.exec(statement)).all()
    return {row.worker_id: to_int(row.total_tokens) for row in rows}


def allocated_vram_by_worker_id(model_instances):
    allocated = {}
    for instance in model_instances:
        allocated[instance.worker_id] = allocated.get(instance.worker_id, 0) + vram_claim(
            instance.computed_resource_claim
        )
        distributed_servers = instance.distributed_servers
        if distributed_servers and distributed_servers.subordinate_workers:
            for subworker in distributed_servers.subordinate_workers:
                allocated[subworker.worker_id] = allocated.get(subworker.worker_id, 0) + vram_claim(
                    subworker.computed_resource_claim
                )
    return allocated


def vram_claim(resource_claim) -> int:
    if resource_claim is None or not resource_claim.vram:
        return 0
    return sum(resource_claim.vram.values())


def sum_gpu_memory(gpus, field: str) -> int:
    total = 0
    for gpu in gpus or []:
        memory = getattr(gpu, "memory", None)
        total += getattr(memory, field, 0) or 0
    return total


def average_gpu_rate(gpus, field: str):
    rates = []
    for gpu in gpus or []:
        usage = getattr(gpu, field, None)
        rate = getattr(usage, "utilization_rate", None) if usage else None
        if rate is not None:
            rates.append(rate)
    if not rates:
        return None
    return sum(rates) / len(rates)


def first_gpu_model(gpus):
    for gpu in gpus or []:
        if gpu.name:
            return gpu.name
    return None


def remain_resource_text(gpu_memory_bytes: int, allocated_vram_bytes: int, gpu_count: int):
    if not gpu_memory_bytes or not gpu_count:
        return None
    memory_per_gpu = gpu_memory_bytes / gpu_count
    remaining_gpus = max(gpu_memory_bytes - allocated_vram_bytes, 0) / memory_per_gpu
    return f"{remaining_gpus:.1f} GPU equivalent"


@router.get("/api-keys")
async def api_keys(
    session: SessionDep,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_ip: Optional[str] = None,
    operation: Optional[OperationEnum] = None,
    status: Optional[str] = Query(None, pattern="^(success|failure)$"),
    worker_id: Optional[int] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    filtered, aggregate_columns, order_column = aggregate_source(
        start_at, end_at, api_key, model, source_ip, operation, status, worker_id
    )
    statement = (
        select(
            filtered.c.api_key_id,
            ApiKey.name.label("api_key_name"),
            filtered.c.api_key_access_key,
            *aggregate_columns,
        )
        .select_from(filtered)
        .outerjoin(ApiKey, filtered.c.api_key_id == ApiKey.id)
        .group_by(filtered.c.api_key_id, ApiKey.name, filtered.c.api_key_access_key)
        .order_by(func.sum(order_column).desc())
    )
    rows, total = await paginate(session, statement, page, page_size)
    return {
        "items": [aggregate_row(row) for row in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/models")
async def models(
    session: SessionDep,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_ip: Optional[str] = None,
    operation: Optional[OperationEnum] = None,
    status: Optional[str] = Query(None, pattern="^(success|failure)$"),
    worker_id: Optional[int] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    filtered, aggregate_columns, order_column = aggregate_source(
        start_at, end_at, api_key, model, source_ip, operation, status, worker_id
    )
    statement = (
        select(
            filtered.c.model_id,
            filtered.c.model_name,
            filtered.c.worker_id,
            filtered.c.worker_name,
            func.count(func.distinct(ModelInstance.id)).label("replicas"),
            *aggregate_columns,
        )
        .select_from(filtered)
        .outerjoin(Model, filtered.c.model_id == Model.id)
        .outerjoin(ModelInstance, filtered.c.model_id == ModelInstance.model_id)
        .group_by(
            filtered.c.model_id,
            filtered.c.model_name,
            filtered.c.worker_id,
            filtered.c.worker_name,
        )
        .order_by(func.sum(order_column).desc())
    )
    rows, total = await paginate(session, statement, page, page_size)
    return {
        "items": [aggregate_row(row) for row in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/source-ips")
async def source_ips(
    session: SessionDep,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_ip: Optional[str] = None,
    operation: Optional[OperationEnum] = None,
    status: Optional[str] = Query(None, pattern="^(success|failure)$"),
    worker_id: Optional[int] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    filtered, aggregate_columns, order_column = aggregate_source(
        start_at, end_at, api_key, model, source_ip, operation, status, worker_id
    )
    statement = (
        select(filtered.c.source_ip, *aggregate_columns)
        .select_from(filtered)
        .group_by(filtered.c.source_ip)
        .order_by(func.sum(order_column).desc())
    )
    rows, total = await paginate(session, statement, page, page_size)
    return {
        "items": [aggregate_row(row) for row in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/daily-logs")
async def daily_logs(
    session: SessionDep,
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_ip: Optional[str] = None,
    operation: Optional[OperationEnum] = None,
    status: Optional[str] = Query(None, pattern="^(success|failure)$"),
    worker_id: Optional[int] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    filtered, aggregate_columns, _ = aggregate_source(
        start_at, end_at, api_key, model, source_ip, operation, status, worker_id
    )
    statement = (
        select(filtered.c.date, *aggregate_columns)
        .select_from(filtered)
        .group_by(filtered.c.date)
        .order_by(filtered.c.date.desc())
    )
    rows, total = await paginate(session, statement, page, page_size)
    return {
        "items": [aggregate_row(row) for row in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/daily-logs/{log_date}/hourly")
async def hourly_logs(
    session: SessionDep,
    log_date: date,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_ip: Optional[str] = None,
    operation: Optional[OperationEnum] = None,
    status: Optional[str] = Query(None, pattern="^(success|failure)$"),
    worker_id: Optional[int] = None,
):
    start_at = datetime.combine(log_date, time.min, tzinfo=timezone.utc)
    end_at = datetime.combine(log_date, time.max, tzinfo=timezone.utc)
    filtered, aggregate_columns = hourly_source(
        start_at, end_at, api_key, model, source_ip, operation, status, worker_id
    )
    statement = (
        select(filtered.c.hour, *aggregate_columns)
        .select_from(filtered)
        .group_by(filtered.c.hour)
        .order_by(filtered.c.hour)
    )
    rows = (await session.exec(statement)).all()
    by_hour = {row.hour: aggregate_row(row) for row in rows if row.hour is not None}
    items = []
    for hour in range(24):
        items.append(
            by_hour.get(
                hour,
                {
                    "hour": hour,
                    "request_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "api_key_count": 0,
                    "model_count": 0,
                    "source_ip_count": 0,
                },
            )
        )
    return {"items": items}


@router.get("/daily-logs/{log_date}/calls")
async def calls(
    session: SessionDep,
    log_date: date,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_ip: Optional[str] = None,
    operation: Optional[OperationEnum] = None,
    status: Optional[str] = Query(None, pattern="^(success|failure)$"),
    worker_id: Optional[int] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    start_at = datetime.combine(log_date, time.min, tzinfo=timezone.utc)
    end_at = datetime.combine(log_date, time.max, tzinfo=timezone.utc)
    statement = (
        base_statement(
            start_at, end_at, api_key, model, source_ip, operation, status, worker_id
        )
        .add_columns(ApiKey.name.label("api_key_name"), User.username)
        .outerjoin(User, ModelUsageLog.user_id == User.id)
        .order_by(ModelUsageLog.call_time.desc(), ModelUsageLog.id.desc())
    )
    rows, total = await paginate(session, statement, page, page_size)
    return {
        "items": [call_row(row) for row in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/daily-logs/{log_date}/calls/export")
async def export_calls(
    session: SessionDep,
    log_date: date,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_ip: Optional[str] = None,
    operation: Optional[OperationEnum] = None,
    status: Optional[str] = Query(None, pattern="^(success|failure)$"),
    worker_id: Optional[int] = None,
):
    start_at = datetime.combine(log_date, time.min, tzinfo=timezone.utc)
    end_at = datetime.combine(log_date, time.max, tzinfo=timezone.utc)
    statement = (
        base_statement(
            start_at, end_at, api_key, model, source_ip, operation, status, worker_id
        )
        .add_columns(ApiKey.name.label("api_key_name"), User.username)
        .outerjoin(User, ModelUsageLog.user_id == User.id)
        .order_by(ModelUsageLog.call_time.desc(), ModelUsageLog.id.desc())
    )
    filename = f"model_usage_calls_{log_date.isoformat()}.csv"
    return StreamingResponse(
        stream_calls_csv(session, statement),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def stream_calls_csv(session: SessionDep, statement, batch_size: int = 500):
    yield calls_csv_header()
    last_call_time = None
    last_id = None
    while True:
        batch_statement = statement
        if last_call_time is not None and last_id is not None:
            batch_statement = batch_statement.where(
                or_(
                    ModelUsageLog.call_time < last_call_time,
                    and_(
                        ModelUsageLog.call_time == last_call_time,
                        ModelUsageLog.id < last_id,
                    ),
                )
            )
        rows = (await session.exec(batch_statement.limit(batch_size))).all()
        if not rows:
            break
        yield calls_csv_rows(rows)
        last_log = rows[-1][0]
        last_call_time = last_log.call_time
        last_id = last_log.id


def calls_csv_header() -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=call_csv_fieldnames())
    writer.writeheader()
    return output.getvalue()


def calls_csv_rows(rows: Iterable) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=call_csv_fieldnames())
    for row in rows:
        writer.writerow(call_row(row))
    return output.getvalue()


def call_csv_fieldnames():
    return [
        "request_id",
        "call_time",
        "api_key_id",
        "api_key_name",
        "api_key_access_key",
        "user_id",
        "username",
        "model_id",
        "model_name",
        "operation",
        "source_ip",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "usage_available",
        "status_code",
        "success",
        "duration_ms",
        "ttft_ms",
        "tokens_per_second",
        "worker_id",
        "worker_name",
        "worker_ip",
        "model_instance_id",
        "error_code",
        "error_type",
        "error_message",
    ]


def aggregate_row(row):
    data = dict(row._mapping)
    for key in (
        "request_count",
        "success_count",
        "failure_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "api_key_count",
        "model_count",
        "source_ip_count",
        "replicas",
        "duration_ms_sum",
    ):
        if key in data:
            data[key] = to_int(data[key])
    if "last_call_time" in data:
        data["last_call_time"] = to_iso(data["last_call_time"])
    if "date" in data and data["date"] is not None:
        data["date"] = data["date"].isoformat()
    for key in ("api_key_id", "model_id", "worker_id"):
        if key in data:
            data[key] = public_dimension_id(data[key])
    if "source_ip" in data:
        data["source_ip"] = public_dimension_text(data["source_ip"])
    return data


def call_row(row):
    log = row[0]
    return {
        "request_id": log.request_id,
        "call_time": to_iso(log.call_time),
        "api_key_id": log.api_key_id,
        "api_key_name": row.api_key_name,
        "api_key_access_key": log.api_key_access_key,
        "user_id": log.user_id,
        "username": row.username,
        "model_id": log.model_id,
        "model_name": log.model_name,
        "operation": log.operation.value if log.operation else None,
        "source_ip": log.source_ip,
        "prompt_tokens": log.prompt_token_count,
        "completion_tokens": log.completion_token_count,
        "total_tokens": log.total_token_count,
        "usage_available": log.usage_available,
        "status_code": log.status_code,
        "success": log.success,
        "duration_ms": log.duration_ms,
        "ttft_ms": log.ttft_ms,
        "tokens_per_second": to_float(log.tokens_per_second) if log.tokens_per_second is not None else None,
        "worker_id": log.worker_id,
        "worker_name": log.worker_name,
        "worker_ip": log.worker_ip,
        "model_instance_id": log.model_instance_id,
        "error_code": log.error_code,
        "error_type": log.error_type,
        "error_message": log.error_message,
    }


def public_dimension_id(value: Optional[int]):
    return value or None


def public_dimension_text(value: Optional[str]):
    return value or None
