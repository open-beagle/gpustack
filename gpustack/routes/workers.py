from typing import Annotated, Optional

from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ForbiddenException,
    InternalServerErrorException,
    NotFoundException,
    UnauthorizedException,
)
from gpustack.server.deps import (
    CurrentAdminUserDep,
    EngineDep,
    ListParamsDep,
    SessionDep,
)
from gpustack.schemas.workers import (
    ModelPreheatWorkerCredentialBootstrap,
    WorkerCreate,
    WorkerPublic,
    WorkerUpdate,
    WorkersPublic,
    Worker,
)
from gpustack.server.services import WorkerService
from gpustack.api.auth import SYSTEM_WORKER_USER_PREFIX
from gpustack.server.model_preheat_worker_identity import (
    WORKER_CREDENTIAL_HEADER,
    issue_model_preheat_worker_credential,
    validate_model_preheat_worker_credential,
    validate_model_preheat_worker_registration_credential,
    worker_uuid_has_credential,
)

router = APIRouter()


def normalize_worker_status_for_response(worker: Worker):
    worker.normalize_status_for_state()
    return worker


@router.get("", response_model=WorkersPublic)
async def get_workers(
    engine: EngineDep,
    session: SessionDep,
    params: ListParamsDep,
    name: str = None,
    search: str = None,
    uuid: str = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    fields = {}
    if name:
        fields = {"name": name}
    if uuid:
        fields["worker_uuid"] = uuid

    if params.watch:
        return StreamingResponse(
            Worker.streaming(engine, fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    result = await Worker.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
    )
    result.items = [
        normalize_worker_status_for_response(worker) for worker in result.items
    ]
    return result


@router.get("/{id}", response_model=WorkerPublic)
async def get_worker(session: SessionDep, id: int):
    worker = await Worker.one_by_id(session, id)
    if not worker:
        raise NotFoundException(message="worker not found")

    return normalize_worker_status_for_response(worker)


@router.post(
    "/{id}/model-preheat-credential",
    response_model=ModelPreheatWorkerCredentialBootstrap,
)
async def bootstrap_model_preheat_worker_credential(
    response: Response,
    session: SessionDep,
    current_user: CurrentAdminUserDep,
    id: int,
):
    if current_user.username.startswith(SYSTEM_WORKER_USER_PREFIX):
        raise ForbiddenException(message="Administrator authentication required")
    worker = await Worker.one_by_id(session, id)
    if worker is None:
        raise NotFoundException(message="worker not found")
    worker_id = worker.id
    worker_uuid = worker.worker_uuid
    credential = await issue_model_preheat_worker_credential(
        session, worker_id, worker_uuid, reset_pending=True
    )
    response.headers["Cache-Control"] = "no-store"
    return ModelPreheatWorkerCredentialBootstrap(
        worker_id=worker_id,
        worker_uuid=worker_uuid,
        credential=credential,
    )


@router.post("", response_model=WorkerPublic)
async def create_worker(
    request: Request,
    response: Response,
    session: SessionDep,
    worker_in: WorkerCreate,
    worker_credential: Annotated[
        Optional[str], Header(alias=WORKER_CREDENTIAL_HEADER)
    ] = None,
):
    existing = await Worker.one_by_field(session, "name", worker_in.name)
    if existing:
        raise AlreadyExistsException(message=f"worker f{worker_in.name} already exists")
    await _authorize_worker_registration(
        request, session, worker_in.worker_uuid, worker_credential
    )

    try:
        worker_in.compute_state()
        worker = await Worker.create(session, worker_in)
        await _issue_preheat_credential(request, response, session, worker, True)
        await session.refresh(worker)
        return worker
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create worker: {e}")


@router.put("/{id}", response_model=WorkerPublic)
async def update_worker(
    request: Request,
    response: Response,
    session: SessionDep,
    id: int,
    worker_in: WorkerUpdate,
    rotate_preheat_credential: bool = Header(
        default=False, alias="X-GPUStack-Worker-Registration"
    ),
    worker_credential: Annotated[
        Optional[str], Header(alias=WORKER_CREDENTIAL_HEADER)
    ] = None,
):
    worker = await Worker.one_by_id(session, id)
    if not worker:
        raise NotFoundException(message="worker not found")
    if rotate_preheat_credential:
        await _authorize_worker_registration(
            request, session, worker.worker_uuid, worker_credential
        )

    try:
        worker_in.compute_state()
        await WorkerService(session).update(worker, worker_in)
        await _issue_preheat_credential(
            request,
            response,
            session,
            worker,
            rotate_preheat_credential,
        )
        await session.refresh(worker)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update worker: {e}")

    return worker


async def _issue_preheat_credential(
    request, response, session, worker, registration_request
):
    user = getattr(request.state, "user", None)
    if (
        not registration_request
        or user is None
        or not user.username.startswith(SYSTEM_WORKER_USER_PREFIX)
    ):
        return
    token = await issue_model_preheat_worker_credential(
        session, worker.id, worker.worker_uuid
    )
    response.headers[WORKER_CREDENTIAL_HEADER] = token
    response.headers["Cache-Control"] = "no-store"


async def _authorize_worker_registration(
    request, session, worker_uuid, worker_credential
):
    user = getattr(request.state, "user", None)
    if user is None or not user.username.startswith(SYSTEM_WORKER_USER_PREFIX):
        return
    if not await worker_uuid_has_credential(session, worker_uuid):
        return
    identity = await validate_model_preheat_worker_registration_credential(
        session, worker_credential, worker_uuid
    )
    if identity is None:
        raise UnauthorizedException(message="Invalid worker registration credentials")


@router.delete("/{id}")
async def delete_worker(session: SessionDep, id: int):
    worker = await Worker.one_by_id(session, id)
    if not worker:
        raise NotFoundException(message="worker not found")

    try:
        await WorkerService(session).delete(worker)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete worker: {e}")
