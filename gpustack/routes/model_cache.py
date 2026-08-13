from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import and_
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.schemas.model_cache import (
    ModelCacheDeleteResult,
    ModelCacheFilesPublic,
    ModelCacheModelsPublic,
    ModelCacheTask,
    ModelCachePreview,
    ModelCacheTaskPublic,
    ModelCacheTaskStateEnum,
    ModelCacheTasksPublic,
    ModelCacheTaskUpdate,
)
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.deps import (
    CurrentAdminUserDep,
    EngineDep,
    ListParamsDep,
    SessionDep,
)
from gpustack.server.model_cache_service import (
    ModelCacheConfigurationError,
    ModelCacheService,
)
from gpustack.utils.model_cache import model_object_prefix, validate_model_id


router = APIRouter()
task_router = APIRouter()


def _service(request: Request):
    try:
        return ModelCacheService(request.app.state.server_config)
    except ModelCacheConfigurationError as exc:
        raise ServiceUnavailableException(message=str(exc)) from exc


@router.get("", response_model=ModelCacheModelsPublic)
async def get_model_cache(
    request: Request, search: str = None, organization: str = None
):
    try:
        return _service(request).list_models(search=search, organization=organization)
    except Exception as exc:
        if isinstance(exc, ServiceUnavailableException):
            raise
        raise ServiceUnavailableException(message="local_s3_unavailable") from exc


@router.get("/{organization}/{model_name}/files", response_model=ModelCacheFilesPublic)
async def get_model_cache_files(request: Request, organization: str, model_name: str):
    try:
        return _service(request).list_files(f"{organization}/{model_name}")
    except ValueError as exc:
        raise NotFoundException(message="model_cache_not_found") from exc
    except Exception as exc:
        if isinstance(exc, ServiceUnavailableException):
            raise
        raise ServiceUnavailableException(message="local_s3_unavailable") from exc


@router.delete("/{organization}/{model_name}", response_model=ModelCacheDeleteResult)
async def delete_model_cache(request: Request, organization: str, model_name: str):
    try:
        return _service(request).delete_model(f"{organization}/{model_name}")
    except ValueError as exc:
        raise NotFoundException(message="model_cache_not_found") from exc
    except Exception as exc:
        if isinstance(exc, ServiceUnavailableException):
            raise
        raise ServiceUnavailableException(message="local_s3_unavailable") from exc


@task_router.get("", response_model=ModelCacheTasksPublic)
async def get_model_cache_tasks(
    engine: EngineDep,
    session: SessionDep,
    params: ListParamsDep,
    worker_id: int = None,
    state: ModelCacheTaskStateEnum = None,
):
    fields = {}
    if worker_id is not None:
        fields["worker_id"] = worker_id
    if state is not None:
        fields["state"] = state
    if params.watch:
        return StreamingResponse(
            ModelCacheTask.streaming(engine, fields=fields),
            media_type="text/event-stream",
        )
    return await ModelCacheTask.paginated_by_query(
        session=session,
        fields=fields,
        page=params.page,
        per_page=params.perPage,
    )


@task_router.get("/{id}", response_model=ModelCacheTaskPublic)
async def get_model_cache_task(session: SessionDep, id: int):
    task = await ModelCacheTask.one_by_id(session, id)
    if task is None:
        raise NotFoundException(message="model_cache_task_not_found")
    return task


@task_router.put("/{id}", response_model=ModelCacheTaskPublic)
async def update_model_cache_task(
    session: SessionDep, id: int, update: ModelCacheTaskUpdate
):
    task = await ModelCacheTask.one_by_id(session, id)
    if task is None:
        raise NotFoundException(message="model_cache_task_not_found")
    task.state = update.state
    task.progress = min(max(update.progress, 0), 100)
    task.uploaded_size = max(update.uploaded_size, 0)
    task.total_size = max(update.total_size, 0)
    task.error_message = update.error_message
    if update.state in {ModelCacheTaskStateEnum.READY, ModelCacheTaskStateEnum.ERROR}:
        task.finished_at = datetime.now(timezone.utc)
    await task.update(session)
    return task


@task_router.delete("/{id}", status_code=204)
async def delete_model_cache_task(session: SessionDep, id: int):
    task = await ModelCacheTask.one_by_id(session, id)
    if task is None:
        raise NotFoundException(message="model_cache_task_not_found")
    await task.delete(session)
    return Response(status_code=204)


async def create_model_cache_task(
    request: Request,
    session: SessionDep,
    current_user: CurrentAdminUserDep,
    model_file_id: int,
):
    model_file = await ModelFile.one_by_id(session, model_file_id)
    if model_file is None:
        raise NotFoundException(message="model_file_not_found")
    if (
        model_file.source != SourceEnum.MODEL_SCOPE
        or not model_file.model_scope_model_id
    ):
        raise ConflictException(message="model_cache_requires_modelscope_source")
    model_id = model_file.model_scope_model_id
    try:
        validate_model_id(model_id)
    except ValueError as exc:
        raise ConflictException(message="invalid_model_id") from exc
    if (
        model_file.state != ModelFileStateEnum.READY
        or model_file.worker_id is None
        or not model_file.resolved_paths
    ):
        raise ConflictException(message="model_file_not_ready")
    worker = await Worker.one_by_id(session, model_file.worker_id)
    if worker is None or worker.state != WorkerStateEnum.READY:
        raise ConflictException(message="model_file_worker_not_ready")

    service = _service(request)
    if service.exists(model_id):
        raise AlreadyExistsException(message="model_cache_already_exists")
    active = (
        await session.exec(
            select(ModelCacheTask).where(
                and_(
                    ModelCacheTask.model_id == model_id,
                    ModelCacheTask.state.in_(
                        [
                            ModelCacheTaskStateEnum.PENDING,
                            ModelCacheTaskStateEnum.UPLOADING,
                        ]
                    ),
                )
            )
        )
    ).first()
    if active is not None:
        raise AlreadyExistsException(message="model_cache_task_already_exists")

    config = request.app.state.server_config
    target_path = model_object_prefix(
        urlparse(config.worker_local_s3_modelscope_prefix).path.strip("/"),
        model_id,
    )
    task = ModelCacheTask(
        model_file_id=model_file.id,
        worker_id=model_file.worker_id,
        model_id=model_id,
        target_path=target_path,
        source_paths=list(model_file.resolved_paths),
        total_size=model_file.size or 0,
        created_by_user_id=current_user.id,
    )
    return await ModelCacheTask.create(session, task)


async def get_model_cache_preview(
    request: Request,
    session: SessionDep,
    model_file_id: int,
):
    model_file = await ModelFile.one_by_id(session, model_file_id)
    if model_file is None:
        raise NotFoundException(message="model_file_not_found")
    if (
        model_file.source != SourceEnum.MODEL_SCOPE
        or not model_file.model_scope_model_id
    ):
        raise ConflictException(message="model_cache_requires_modelscope_source")
    try:
        validate_model_id(model_file.model_scope_model_id)
    except ValueError as exc:
        raise ConflictException(message="invalid_model_id") from exc
    service = _service(request)
    return ModelCachePreview(
        model_id=model_file.model_scope_model_id,
        s3_path=service.s3_path(model_file.model_scope_model_id),
        file_count=len(model_file.resolved_paths),
        total_size=model_file.size or 0,
    )
