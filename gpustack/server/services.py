import asyncio
import json
import logging
import functools
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union
from aiocache import Cache, BaseCache
from sqlmodel import SQLModel, bindparam, cast, col, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.dialects.postgresql import JSONB

from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.model_files import ModelFile
from gpustack.schemas.model_usage import (
    ModelUsage,
    ModelUsageDailyStat,
    ModelUsageHourlyStat,
    ModelUsageLog,
)
from gpustack.schemas.models import Model, ModelInstance, ModelInstanceStateEnum
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker
from gpustack.server.usage_buffer import usage_flush_buffer

logger = logging.getLogger(__name__)
cache = Cache(Cache.MEMORY)


def build_cache_key(func: Callable, *args, **kwargs):
    if kwargs is None:
        kwargs = {}
    ordered_kwargs = sorted(kwargs.items())
    return func.__qualname__ + str(args) + str(ordered_kwargs)


async def delete_cache_by_key(func, *args, **kwargs):
    key = build_cache_key(func, *args, **kwargs)
    logger.trace(f"Deleting cache for key: {key}")
    await cache.delete(key)


async def set_cache_by_key(key: str, value: Any):
    logger.trace(f"Set cache for key: {key}")
    await cache.set(key, value)


_cache_locks: Dict[str, asyncio.Lock] = {}


class locked_cached:
    def __init__(self, ttl: int = 30, cache: BaseCache = cache):
        self.cache = cache
        self.ttl = ttl

    def __call__(self, f):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            return await self.decorator(f, *args, **kwargs)

        wrapper.cache = self.cache
        return wrapper

    async def get_from_cache(self, key: str):
        return await self.cache.get(key)

    async def set_in_cache(self, key: str, value: Any):
        await self.cache.set(key, value, ttl=self.ttl)

    async def decorator(self, f, *args, **kwargs):
        # no self arg
        key = build_cache_key(f, *args[1:], **kwargs)
        value = await self.get_from_cache(key)
        if value is not None:
            return value

        lock = _cache_locks.setdefault(key, asyncio.Lock())

        async with lock:
            value = await self.get_from_cache(key)
            if value is not None:
                return value

            logger.trace(f"cache miss for key: {key}")
            result = await f(*args, **kwargs)

            await self.set_in_cache(key, result)

        return result


class UserService:

    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached(ttl=60)
    async def get_by_id(self, user_id: int) -> Optional[User]:
        result = await User.one_by_id(self.session, user_id)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached(ttl=60)
    async def get_by_username(self, username: str) -> Optional[User]:
        result = await User.one_by_field(self.session, "username", username)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def create(self, user: User):
        return await User.create(self.session, user)

    async def update(self, user: User, source: Union[dict, SQLModel, None] = None):
        result = await user.update(self.session, source)
        await delete_cache_by_key(self.get_by_id, user.id)
        await delete_cache_by_key(self.get_by_username, user.username)
        return result

    async def delete(self, user: User):
        apikeys = await APIKeyService(self.session).get_by_user_id(user.id)
        result = await user.delete(self.session)
        await delete_cache_by_key(self.get_by_id, user.id)
        await delete_cache_by_key(self.get_by_username, user.username)
        for apikey in apikeys:
            await delete_cache_by_key(
                APIKeyService.get_by_access_key, apikey.access_key
            )
        return result


class APIKeyService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached(ttl=60)
    async def get_by_access_key(self, access_key: str) -> Optional[ApiKey]:
        result = await ApiKey.one_by_field(self.session, "access_key", access_key)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def get_by_user_id(self, user_id: int) -> List[ApiKey]:
        results = await ApiKey.all_by_field(self.session, "user_id", user_id)
        if results is None:
            return []
        for result in results:
            self.session.expunge(result)
        return results

    async def delete(self, api_key: ApiKey):
        result = await api_key.delete(self.session)
        await delete_cache_by_key(self.get_by_access_key, api_key.access_key)
        return result


class WorkerService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached(ttl=60)
    async def get_by_id(self, worker_id: int) -> Optional[Worker]:
        result = await Worker.one_by_id(self.session, worker_id)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached(ttl=60)
    async def get_by_name(self, name: str) -> Optional[Worker]:
        result = await Worker.one_by_field(self.session, "name", name)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def update(self, worker: Worker, source: Union[dict, SQLModel, None] = None):
        result = await worker.update(self.session, source)
        await delete_cache_by_key(self.get_by_id, worker.id)
        await delete_cache_by_key(self.get_by_name, worker.name)
        return result

    async def delete(self, worker: Worker):
        result = await worker.delete(self.session)
        await delete_cache_by_key(self.get_by_id, worker.id)
        await delete_cache_by_key(self.get_by_name, worker.name)
        return result


class ModelService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached(ttl=60)
    async def get_by_id(self, model_id: int) -> Optional[Model]:
        result = await Model.one_by_id(self.session, model_id)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached(ttl=60)
    async def get_by_name(self, name: str) -> Optional[Model]:
        result = await Model.one_by_field(self.session, "name", name)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def update(self, model: Model, source: Union[dict, SQLModel, None] = None):
        result = await model.update(self.session, source)
        await delete_cache_by_key(self.get_by_id, model.id)
        await delete_cache_by_key(self.get_by_name, model.name)
        return result

    async def delete(self, model: Model):
        result = await model.delete(self.session)
        await delete_cache_by_key(self.get_by_id, model.id)
        await delete_cache_by_key(self.get_by_name, model.name)
        return result


class ModelInstanceService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached(ttl=60)
    async def get_running_instances(self, model_id: int) -> List[ModelInstance]:
        results = await ModelInstance.all_by_fields(
            self.session,
            fields={"model_id": model_id, "state": ModelInstanceStateEnum.RUNNING},
        )
        if results is None:
            return []

        for result in results:
            self.session.expunge(result)
        return results

    async def create(self, model_instance):
        result = await ModelInstance.create(self.session, model_instance)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        return result

    async def update(
        self, model_instance: ModelInstance, source: Union[dict, SQLModel, None] = None
    ):
        result = await model_instance.update(self.session, source)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        return result

    async def delete(self, model_instance: ModelInstance):
        result = await model_instance.delete(self.session)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        return result


class ModelUsageService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached(ttl=60)
    async def get_by_fields(self, fields: dict) -> ModelUsage:
        result = await ModelUsage.one_by_fields(
            self.session,
            fields=fields,
        )
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def create(self, model_usage: ModelUsage):
        return await ModelUsage.create(self.session, model_usage)

    async def update(
        self,
        model_usage: ModelUsage,
        completion_token_count: int,
        prompt_token_count: int,
    ):
        model_usage.completion_token_count += completion_token_count
        model_usage.prompt_token_count += prompt_token_count
        model_usage.request_count += 1

        key = build_cache_key(
            self.get_by_fields,
            model_usage.user_id,
            model_usage.model_id,
            model_usage.operation,
            model_usage.date,
        )
        await set_cache_by_key(key, model_usage)
        usage_flush_buffer[key] = model_usage
        return model_usage


class ModelUsageLogService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, model_usage_log: ModelUsageLog):
        return await ModelUsageLog.create(self.session, model_usage_log)

    async def add(self, model_usage_log: ModelUsageLog):
        self.session.add(model_usage_log)
        await self.session.flush()
        await self.session.refresh(model_usage_log)
        return model_usage_log


class ModelUsageStatService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def record(self, model_usage_log: ModelUsageLog):
        await self._record_stat(ModelUsageHourlyStat, model_usage_log, include_hour=True)
        await self._record_stat(ModelUsageDailyStat, model_usage_log, include_hour=False)

    async def rebuild(self, start_date=None, end_date=None, batch_size: int = 500):
        await self._delete_existing_stats(start_date, end_date)
        statement = select(ModelUsageLog).order_by(ModelUsageLog.call_time)
        if start_date is not None:
            statement = statement.where(ModelUsageLog.date >= start_date)
        if end_date is not None:
            statement = statement.where(ModelUsageLog.date <= end_date)
        rebuilt_count = 0
        offset = 0
        while True:
            logs = (await self.session.exec(statement.offset(offset).limit(batch_size))).all()
            if not logs:
                break
            for model_usage_log in logs:
                await self.record(model_usage_log)
            await self.session.commit()
            rebuilt_count += len(logs)
            offset += batch_size
        return rebuilt_count

    async def cleanup_before(self, cutoff_date: date, batch_size: int = 500):
        cutoff_time = datetime.combine(cutoff_date, datetime.min.time(), tzinfo=timezone.utc)
        deleted = {}
        for stat_cls in (ModelUsageHourlyStat, ModelUsageDailyStat):
            deleted[stat_cls.__tablename__] = await self._delete_in_batches(
                select(stat_cls).where(stat_cls.date < cutoff_date), batch_size
            )

        deleted[ModelUsageLog.__tablename__] = await self._delete_in_batches(
            select(ModelUsageLog).where(ModelUsageLog.call_time < cutoff_time), batch_size
        )
        return deleted

    async def _delete_in_batches(self, statement, batch_size: int):
        deleted_count = 0
        while True:
            rows = (await self.session.exec(statement.limit(batch_size))).all()
            if not rows:
                break
            for row in rows:
                await self.session.delete(row)
            await self.session.commit()
            deleted_count += len(rows)
        return deleted_count

    async def _delete_existing_stats(self, start_date=None, end_date=None):
        for stat_cls in (ModelUsageHourlyStat, ModelUsageDailyStat):
            statement = select(stat_cls)
            if start_date is not None:
                statement = statement.where(stat_cls.date >= start_date)
            if end_date is not None:
                statement = statement.where(stat_cls.date <= end_date)
            stats = (await self.session.exec(statement)).all()
            for stat in stats:
                await self.session.delete(stat)
        await self.session.commit()

    async def _record_stat(self, stat_cls, model_usage_log: ModelUsageLog, include_hour: bool):
        fields = {
            "date": model_usage_log.date,
            "api_key_id": model_usage_log.api_key_id or 0,
            "model_id": model_usage_log.model_id or 0,
            "source_ip": model_usage_log.source_ip or "",
            "operation": model_usage_log.operation,
            "worker_id": model_usage_log.worker_id or 0,
        }
        if include_hour:
            fields["hour"] = model_usage_log.hour

        return await self._record_stat_once(stat_cls, model_usage_log, fields)

    async def _record_stat_once(self, stat_cls, model_usage_log: ModelUsageLog, fields: dict):
        current_stat = await stat_cls.one_by_fields(self.session, fields)
        if current_stat:
            current_stat.request_count += 1
            if model_usage_log.success:
                current_stat.success_count += 1
            else:
                current_stat.failure_count += 1
            current_stat.prompt_token_count += model_usage_log.prompt_token_count
            current_stat.completion_token_count += model_usage_log.completion_token_count
            current_stat.total_token_count += model_usage_log.total_token_count
            current_stat.duration_ms_sum += model_usage_log.duration_ms or 0
            current_stat.last_call_time = max(
                current_stat.last_call_time, model_usage_log.call_time
            )
            self.session.add(current_stat)
            await self.session.flush()
            return current_stat

        stat = stat_cls(
            **fields,
            api_key_access_key=model_usage_log.api_key_access_key,
            model_name=model_usage_log.model_name,
            worker_name=model_usage_log.worker_name,
            request_count=1,
            success_count=1 if model_usage_log.success else 0,
            failure_count=0 if model_usage_log.success else 1,
            prompt_token_count=model_usage_log.prompt_token_count,
            completion_token_count=model_usage_log.completion_token_count,
            total_token_count=model_usage_log.total_token_count,
            duration_ms_sum=model_usage_log.duration_ms or 0,
            last_call_time=model_usage_log.call_time,
        )
        self.session.add(stat)
        await self.session.flush()
        return stat


class ModelFileService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_resolved_path(self, path: str) -> List[ModelFile]:
        # sqlite
        condition = col(ModelFile.resolved_paths).contains(json.dumps(path))
        if self.session.bind.dialect.name == "postgresql":
            condition = cast(ModelFile.resolved_paths, JSONB).op('?')(
                bindparam("resolved_path", path)
            )

        results = await ModelFile.all_by_fields(
            self.session,
            extra_conditions=[condition],
        )
        if results is None:
            return None

        for result in results:
            self.session.expunge(result)
        return results

    async def get_by_source_index(self, source_index: str) -> List[ModelFile]:
        results = await ModelFile.all_by_field(
            self.session, "source_index", source_index
        )
        if results is None:
            return None

        for result in results:
            self.session.expunge(result)
        return results

    async def create(self, model_file: ModelFile):
        return await ModelFile.create(self.session, model_file)
