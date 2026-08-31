import asyncio
import logging
from datetime import datetime
from typing import Optional

import anyio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import inspect
from sqlalchemy.exc import NoInspectionAvailable
from sqlmodel import String, cast, func, or_, select, update
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    InternalServerErrorException,
    HTTPException,
    NotFoundException,
)
from gpustack.server.deps import ListParamsDep, SessionDep, EngineDep
from gpustack.schemas.model_files import (
    ModelFile,
    ModelFileCreate,
    ModelFilePublic,
    ModelFileStateEnum,
    ModelFileUpdate,
    ModelFilesPublic,
)
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.model_file_download_executions import (
    ModelFileDownloadExecution,
    ModelFileDownloadExecutionStateEnum,
)
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_storage_sync import (
    ModelStorageSyncTask,
    ModelStorageSyncTaskStateEnum,
)
from gpustack.schemas.workers import (
    MODEL_STORAGE_PROTOCOL_VERSION,
    Worker,
    WorkerStateEnum,
)
from gpustack.server.model_file_download_execution_service import (
    create_model_file_with_download_execution,
)
from gpustack.server.bus import EventType

router = APIRouter()
logger = logging.getLogger(__name__)


async def lock_model_file_for_sync_or_delete(session, model_file_id: int):
    """串行化 ModelFile 删除与同步任务创建，避免 CASCADE 吞掉活动任务。"""
    dialect = session.bind.dialect.name
    if dialect in {"postgresql", "mysql"}:
        return (
            await session.exec(
                select(ModelFile).where(ModelFile.id == model_file_id).with_for_update()
            )
        ).first()

    # SQLite 不支持行级 FOR UPDATE；无变化 UPDATE 在当前事务持有写锁。
    result = await session.exec(
        update(ModelFile).where(ModelFile.id == model_file_id).values(id=ModelFile.id)
    )
    if result.rowcount != 1:
        return None
    return await session.get(ModelFile, model_file_id)


@router.get("", response_model=ModelFilesPublic)
async def get_model_files(
    engine: EngineDep,
    session: SessionDep,
    params: ListParamsDep,
    search: str = None,
    worker_id: int = None,
    source: Optional[SourceEnum] = None,
    state: Optional[ModelFileStateEnum] = None,
):
    fields = {}

    if worker_id is not None:
        fields["worker_id"] = worker_id
    if source is not None:
        fields["source"] = source
    if state is not None:
        fields["state"] = state

    def get_filter_func(search):
        if search:
            return lambda data: search_model_file_filter(data, search)
        return None

    if params.watch:
        return StreamingResponse(
            _stream_model_files(
                engine,
                fields=fields,
                filter_func=get_filter_func(search),
            ),
            media_type="text/event-stream",
        )

    extra_conditions = []
    if search:
        lower_search = search.lower()
        extra_conditions.append(
            or_(
                *[
                    func.lower(cast(ModelFile.resolved_paths, String)).like(
                        f"%{lower_search}%"
                    ),
                    func.lower(ModelFile.huggingface_repo_id).like(f"%{lower_search}%"),
                    func.lower(ModelFile.huggingface_filename).like(
                        f"%{lower_search}%"
                    ),
                    func.lower(ModelFile.ollama_library_model_name).like(
                        f"%{lower_search}%"
                    ),
                    func.lower(ModelFile.model_scope_model_id).like(
                        f"%{lower_search}%"
                    ),
                    func.lower(ModelFile.model_scope_file_path).like(
                        f"%{lower_search}%"
                    ),
                    func.lower(ModelFile.local_path).like(f"%{lower_search}%"),
                ]
            )
        )

    result = await ModelFile.paginated_by_query(
        session=session,
        fields=fields,
        extra_conditions=extra_conditions,
        page=params.page,
        per_page=params.perPage,
    )
    return ModelFilesPublic(
        items=await _model_files_public(session, result.items),
        pagination=result.pagination,
    )


async def _stream_model_files(engine, fields=None, filter_func=None):
    """输出带传输来源名称的 ModelFile SSE，初始与增量事件使用同一序列化。"""
    try:
        async for event in ModelFile.subscribe(engine):
            if event.type == EventType.HEARTBEAT:
                yield "\n\n"
                continue
            session = AsyncSession(engine)
            try:
                event.data = await _reload_event_model_file_if_needed(session, event)
                matches = ModelFile._match_fields(event, fields) and (
                    not filter_func or ModelFile._safe_filter(filter_func, event.data)
                )
                if not matches:
                    # UPDATE 离开当前筛选集合时通知客户端移除旧行；对从未进入
                    # 集合的记录发送 DELETE 是幂等的。CREATED 不匹配仍忽略。
                    if event.type == EventType.UPDATED:
                        event.type = EventType.DELETED
                        yield ModelFile._format_event(event)
                    continue
                public_items = await _model_files_public(
                    session, [event.data], skip_incomplete=True
                )
                if not public_items:
                    logger.warning(
                        "Skipping incomplete model file event %s for id %s",
                        event.type,
                        _event_model_id(event.data),
                    )
                    continue
                event.data = public_items[0]
            finally:
                with anyio.CancelScope(shield=True):
                    await session.close()
            yield ModelFile._format_event(event)
    except asyncio.CancelledError:
        return


async def _reload_event_model_file_if_needed(session, event):
    if event.type not in (EventType.CREATED, EventType.UPDATED):
        return event.data
    model_file_id = _event_model_id(event.data)
    if model_file_id is None:
        return event.data
    persisted = await _reload_model_file_by_id(session, model_file_id)
    return persisted if persisted is not None else event.data


def _event_model_id(data):
    values = getattr(data, "__dict__", {})
    if isinstance(values, dict) and values.get("id") is not None:
        return values["id"]
    try:
        identity = inspect(data).identity
    except NoInspectionAvailable:
        return None
    return identity[0] if identity else None


def _event_timestamps_missing(data):
    values = getattr(data, "__dict__", {})
    if not isinstance(values, dict):
        return True
    return any(
        not isinstance(values.get(field), datetime)
        for field in ("created_at", "updated_at")
    )


def search_model_file_filter(data: ModelFile, search: str) -> bool:
    if (
        (
            data.huggingface_repo_id
            and search.lower() in data.huggingface_repo_id.lower()
        )
        or (
            data.huggingface_filename
            and search.lower() in data.huggingface_filename.lower()
        )
        or (
            data.ollama_library_model_name
            and search.lower() in data.ollama_library_model_name.lower()
        )
        or (
            data.model_scope_model_id
            and search.lower() in data.model_scope_model_id.lower()
        )
        or (
            data.model_scope_file_path
            and search.lower() in data.model_scope_file_path.lower()
        )
        or (data.local_path and search.lower() in data.local_path.lower())
        or (data.resolved_paths and search.lower() in data.resolved_paths[0].lower())
    ):
        return True

    return False


@router.get("/{id}", response_model=ModelFilePublic)
async def get_model_file(session: SessionDep, id: int):
    model_file = await ModelFile.one_by_id(session, id)
    if not model_file:
        raise NotFoundException(message=f"Model file {id} not found")
    return (await _model_files_public(session, [model_file]))[0]


async def _model_files_public(session, model_files, skip_incomplete=False):
    model_files = await _reload_model_files_missing_required_fields(session, model_files)
    model_file_ids = [item.id for item in model_files if item.id is not None]
    executions = {}
    if model_file_ids:
        rows = (
            await session.exec(
                select(ModelFileDownloadExecution).where(
                    ModelFileDownloadExecution.model_file_id.in_(model_file_ids)
                )
            )
        ).all()
        executions = {row.model_file_id: row for row in rows}

    profile_ids = {
        row.transfer_profile_id
        for row in executions.values()
        if row.transfer_profile_id is not None
    }
    source_worker_ids = {
        row.source_worker_id
        for row in executions.values()
        if row.source_worker_id is not None
    }
    profiles = {}
    if profile_ids:
        profile_rows = (
            await session.exec(
                select(ModelPreheatS3Profile).where(
                    ModelPreheatS3Profile.id.in_(profile_ids)
                )
            )
        ).all()
        profiles = {row.id: row.name for row in profile_rows}
    source_workers = {}
    if source_worker_ids:
        worker_rows = (
            await session.exec(select(Worker).where(Worker.id.in_(source_worker_ids)))
        ).all()
        source_workers = {row.id: row.name for row in worker_rows}

    owner_worker_ids = {
        row.worker_id for row in model_files if row.worker_id is not None
    }
    owner_workers = {}
    latest_worker_ids_by_uuid = {}
    if owner_worker_ids:
        owner_rows = (
            await session.exec(select(Worker).where(Worker.id.in_(owner_worker_ids)))
        ).all()
        owner_workers = {row.id: row for row in owner_rows}
        owner_uuids = {row.worker_uuid for row in owner_rows}
        current_rows = (
            await session.exec(
                select(Worker)
                .where(Worker.worker_uuid.in_(owner_uuids))
                .order_by(Worker.id.desc())
            )
        ).all()
        for row in current_rows:
            latest_worker_ids_by_uuid.setdefault(row.worker_uuid, row.id)

    result = []
    for model_file in model_files:
        if skip_incomplete and _event_timestamps_missing(model_file):
            continue
        execution = executions.get(model_file.id)
        owner_worker = owner_workers.get(model_file.worker_id)
        worker_available = bool(
            owner_worker is not None
            and latest_worker_ids_by_uuid.get(owner_worker.worker_uuid)
            == owner_worker.id
            and owner_worker.state == WorkerStateEnum.READY
            and owner_worker.model_storage_protocol_version
            == MODEL_STORAGE_PROTOCOL_VERSION
        )
        result.append(
            ModelFilePublic.model_validate(
                _model_file_public_source(model_file),
                update={
                    "transfer_source": execution.transfer_source if execution else None,
                    "transfer_profile_id": (
                        execution.transfer_profile_id if execution else None
                    ),
                    "transfer_profile_name": (
                        profiles.get(execution.transfer_profile_id)
                        if execution
                        else None
                    ),
                    "source_worker_id": (
                        execution.source_worker_id if execution else None
                    ),
                    "source_worker_name": (
                        source_workers.get(execution.source_worker_id)
                        if execution
                        else None
                    ),
                    "worker_name": (
                        owner_worker.name
                        if owner_worker is not None
                        else model_file.worker_name_snapshot
                    ),
                    "worker_available": worker_available,
                },
            )
        )
    return result


def _model_file_public_source(model_file):
    if hasattr(model_file, "model_dump"):
        source = model_file.model_dump()
    else:
        source = {
            key: value
            for key, value in getattr(model_file, "__dict__", {}).items()
            if not key.startswith("_")
        }

    mapper = getattr(model_file, "__mapper__", None)
    if mapper is not None:
        for column in mapper.column_attrs:
            try:
                source[column.key] = getattr(model_file, column.key)
            except Exception:
                continue
    else:
        source.update(
            {
                key: value
                for key, value in getattr(model_file, "__dict__", {}).items()
                if not key.startswith("_")
            }
        )
    return source


async def _reload_model_files_missing_required_fields(session, model_files):
    reloaded = []
    for model_file in model_files:
        model_file_id = getattr(model_file, "id", None)
        if model_file_id is None or not _event_timestamps_missing(model_file):
            reloaded.append(model_file)
            continue
        persisted = await _reload_model_file_by_id(session, model_file_id)
        reloaded.append(persisted if persisted is not None else model_file)
    return reloaded


async def _reload_model_file_by_id(session, model_file_id):
    if not hasattr(session, "exec"):
        return await session.get(ModelFile, model_file_id)
    return (
        await session.exec(
            select(ModelFile)
            .where(ModelFile.id == model_file_id)
            .execution_options(populate_existing=True)
        )
    ).first()


@router.post("", response_model=ModelFilePublic)
async def create_model_file(
    session: SessionDep,
    model_file_in: ModelFileCreate,
    request: Request = None,
):
    fields = {
        "worker_id": model_file_in.worker_id,
        "source_index": model_file_in.model_source_index,
        "local_dir": model_file_in.local_dir,
    }
    existing = await ModelFile.one_by_fields(session, fields)
    if existing:
        raise AlreadyExistsException(
            message="Model file with the same model source already exists on the worker."
        )

    try:
        model_file = ModelFile(
            **model_file_in.model_dump(), source_index=model_file_in.model_source_index
        )
        model_file = await create_model_file_with_download_execution(
            session,
            model_file,
            request.app.state.server_config if request is not None else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(message="Failed to create model file") from e

    return model_file


@router.put("/{id}", response_model=ModelFilePublic)
async def update_model_file(
    session: SessionDep, id: int, model_file_in: ModelFileUpdate
):
    model_file = await ModelFile.one_by_id(session, id)
    if not model_file:
        raise NotFoundException(message=f"Model file {id} not found")

    try:
        await model_file.update(session, model_file_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update model file: {e}")

    return model_file


@router.delete("/{id}")
async def delete_model_file(
    session: SessionDep, id: int, cleanup: Optional[bool] = None
):
    model_file = await lock_model_file_for_sync_or_delete(session, id)
    if not model_file:
        raise NotFoundException(message=f"Model file {id} not found")

    if model_file.instances:
        model_instance_names = ", ".join(
            [model_instance.name for model_instance in model_file.instances]
        )
        raise ConflictException(
            message=f"Cannot delete the model file. It's being used by model instances: {model_instance_names}.",
        )

    active_sync_task = (
        await session.exec(
            select(ModelStorageSyncTask.id).where(
                ModelStorageSyncTask.model_file_id == model_file.id,
                ModelStorageSyncTask.state.in_(
                    (
                        ModelStorageSyncTaskStateEnum.PENDING,
                        ModelStorageSyncTaskStateEnum.SCANNING,
                        ModelStorageSyncTaskStateEnum.PUBLISHING,
                    )
                ),
            )
        )
    ).first()
    if active_sync_task is not None:
        raise ConflictException(message="model_file_has_active_sync_task")

    # cleanup、活动任务检查与 DELETE 必须留在同一个事务中。ActiveRecord 的
    # update/delete helper 会分别提交，导致父记录锁在真正删除前被释放。
    await session.refresh(
        model_file,
        attribute_names=[column.key for column in model_file.__mapper__.column_attrs],
    )
    event_snapshot_values = _model_file_public_source(model_file)
    event_snapshot = ModelFile.model_validate(event_snapshot_values)
    event_snapshot.__dict__.update(event_snapshot_values)
    try:
        if cleanup is not None and model_file.cleanup_on_delete != cleanup:
            model_file.cleanup_on_delete = cleanup
        await session.delete(model_file)
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise InternalServerErrorException(message=f"Failed to delete model file: {e}")
    await ModelFile._publish_event(EventType.DELETED, event_snapshot)


@router.post("/{id}/reset", response_model=ModelFilePublic)
async def reset_model_file(session: SessionDep, id: int):
    model_file = await ModelFile.one_by_id(session, id)
    if not model_file:
        raise NotFoundException(message=f"Model file {id} not found")

    try:
        execution = (
            await session.exec(
                select(ModelFileDownloadExecution).where(
                    ModelFileDownloadExecution.model_file_id == id
                )
            )
        ).first()
        if execution is not None:
            execution.state = ModelFileDownloadExecutionStateEnum.PENDING
            execution.claimed_by_worker_uuid = None
            execution.claimed_at = None
            execution.transfer_source = None
            execution.transfer_profile_id = None
            execution.source_worker_id = None
            execution.state_message = None
            execution.error_code = None
            execution.finished_at = None
            session.add(execution)
        model_file.state = ModelFileStateEnum.DOWNLOADING
        model_file.download_progress = 0
        model_file.state_message = ""

        await model_file.update(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update model file: {e}")

    return model_file
