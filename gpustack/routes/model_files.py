from typing import Optional
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from sqlmodel import String, cast, func, or_, select
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
from gpustack.schemas.model_file_download_executions import (
    ModelFileDownloadExecution,
    ModelFileDownloadExecutionStateEnum,
)
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.workers import Worker
from gpustack.server.model_file_download_execution_service import (
    create_model_file_with_download_execution,
)
from gpustack.server.bus import EventType

router = APIRouter()


@router.get("", response_model=ModelFilesPublic)
async def get_model_files(
    engine: EngineDep,
    session: SessionDep,
    params: ListParamsDep,
    search: str = None,
    worker_id: int = None,
):
    fields = {}

    if worker_id:
        fields["worker_id"] = worker_id

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
    async for event in ModelFile.subscribe(engine):
        if event.type == EventType.HEARTBEAT:
            yield "\n\n"
            continue
        if not ModelFile._match_fields(event, fields):
            continue
        if filter_func and not ModelFile._safe_filter(filter_func, event.data):
            continue
        async with AsyncSession(engine) as session:
            event.data = (await _model_files_public(session, [event.data]))[0]
        yield ModelFile._format_event(event)


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


async def _model_files_public(session, model_files):
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
    worker_ids = {
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
    workers = {}
    if worker_ids:
        worker_rows = (
            await session.exec(select(Worker).where(Worker.id.in_(worker_ids)))
        ).all()
        workers = {row.id: row.name for row in worker_rows}

    result = []
    for model_file in model_files:
        execution = executions.get(model_file.id)
        result.append(
            ModelFilePublic.model_validate(
                model_file,
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
                        workers.get(execution.source_worker_id) if execution else None
                    ),
                },
            )
        )
    return result


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
    model_file = await ModelFile.one_by_id(session, id)
    if not model_file:
        raise NotFoundException(message=f"Model file {id} not found")

    if model_file.instances:
        model_instance_names = ", ".join(
            [model_instance.name for model_instance in model_file.instances]
        )
        raise ConflictException(
            message=f"Cannot delete the model file. It's being used by model instances: {model_instance_names}.",
        )

    try:
        if cleanup is not None and model_file.cleanup_on_delete != cleanup:
            model_file.cleanup_on_delete = cleanup
            await model_file.update(session)

        await model_file.delete(session)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete model file: {e}")


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
