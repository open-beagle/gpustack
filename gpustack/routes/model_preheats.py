import asyncio
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import APIRouter, Header, Request
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlmodel import delete, select

from gpustack.api.exceptions import (
    HTTPException,
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.model_preheat_credentials import (
    CredentialEncryptionUnavailable,
    ModelPreheatCredentialCipher,
)
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3ConnectivityStateEnum,
    ModelPreheatS3Profile,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatCreate,
    ModelPreheatDesiredStateEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatTask,
    ModelPreheatTaskLock,
    ModelPreheatTaskPublic,
    ModelPreheatTasksPublic,
    ModelPreheatTargetScopeEnum,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskStateEnum,
    cache_key_for,
    is_terminal_task,
    operation_key_for,
    selection_digest,
)
from gpustack.schemas.workers import Worker
from gpustack.server.deps import CurrentAdminUserDep, ListParamsDep, SessionDep
from gpustack.server.model_preheat_idempotency import (
    canonical_request_hash,
    get_idempotency_record,
    new_idempotency_record,
)
from gpustack.server.model_preheat_revision import resolve_model_preheat_revision
from gpustack.server.model_preheat_connectivity import (
    connectivity_ttl_from_config,
    create_or_reuse_connectivity_check,
    current_ready_workers,
    latest_connectivity_results_for_workers,
    mark_profile_stale_if_expired,
)


router = APIRouter()
CREATE_OPERATION = "model_preheats.create"
LOCK_TTL = timedelta(hours=24)


@router.get("", response_model=ModelPreheatTasksPublic)
async def get_model_preheats(session: SessionDep, params: ListParamsDep):
    return await ModelPreheatTask.paginated_by_query(
        session=session, page=params.page, per_page=params.perPage
    )


@router.get("/{id}", response_model=ModelPreheatTaskPublic)
async def get_model_preheat(session: SessionDep, id: int):
    task = await ModelPreheatTask.one_by_id(session, id)
    if task is None:
        raise NotFoundException(message="model_preheat_task_not_found")
    return _to_public(task)


@router.post("/{id}/cancel", response_model=ModelPreheatTaskPublic)
async def cancel_model_preheat(session: SessionDep, id: int):
    task = await ModelPreheatTask.one_by_id(session, id)
    if task is None:
        raise NotFoundException(message="model_preheat_task_not_found")
    if not is_terminal_task(task):
        changed = await _transition_parent_and_children(
            session,
            task,
            desired_state=ModelPreheatDesiredStateEnum.CANCELED,
            execution_state=ModelPreheatExecutionStateEnum.CANCELED,
            child_state=ModelPreheatWorkerTaskStateEnum.CANCELED,
        )
        if changed:
            await session.exec(
                update(ModelPreheatTask)
                .where(ModelPreheatTask.id == task.id)
                .values(finished_at=datetime.now(timezone.utc))
            )
            await session.exec(
                delete(ModelPreheatTaskLock).where(
                    ModelPreheatTaskLock.task_id == task.id
                )
            )
    else:
        await release_task_lock_if_terminal(session, task)
    await session.commit()
    await session.refresh(task)
    return _to_public(task)


@router.post("/{id}/pause", response_model=ModelPreheatTaskPublic)
async def pause_model_preheat(session: SessionDep, id: int):
    task = await _task_or_404(session, id)
    if (
        is_terminal_task(task)
        or task.desired_state == ModelPreheatDesiredStateEnum.PAUSED
    ):
        return _to_public(task)
    task.paused_from_state = task.execution_state
    await _transition_parent_and_children(
        session,
        task,
        desired_state=ModelPreheatDesiredStateEnum.PAUSED,
        execution_state=ModelPreheatExecutionStateEnum.PAUSED,
        child_state=ModelPreheatWorkerTaskStateEnum.PAUSED,
    )
    await session.commit()
    await session.refresh(task)
    return _to_public(task)


@router.post("/{id}/resume", response_model=ModelPreheatTaskPublic)
async def resume_model_preheat(session: SessionDep, id: int):
    task = await _task_or_404(session, id)
    if task.desired_state != ModelPreheatDesiredStateEnum.PAUSED:
        return _to_public(task)
    restored_state = task.paused_from_state or ModelPreheatExecutionStateEnum.PENDING
    await _transition_parent_and_children(
        session,
        task,
        desired_state=ModelPreheatDesiredStateEnum.RUNNING,
        execution_state=restored_state,
        child_state=ModelPreheatWorkerTaskStateEnum.PENDING,
        from_child_states={ModelPreheatWorkerTaskStateEnum.PAUSED},
    )
    await session.commit()
    await session.refresh(task)
    return _to_public(task)


@router.post("/{id}/retry", response_model=ModelPreheatTaskPublic)
async def retry_model_preheat(session: SessionDep, id: int):
    task = await _task_or_404(session, id)
    if task.execution_state != ModelPreheatExecutionStateEnum.ERROR:
        return _to_public(task)
    expected_attempt = task.attempt
    operation_key = operation_key_for(
        task.s3_profile_id,
        task.cache_key,
        task.target_worker_uuids,
        task.s3_backfill_policy,
    )
    result = await session.exec(
        update(ModelPreheatTask)
        .where(
            ModelPreheatTask.id == id,
            ModelPreheatTask.attempt == expected_attempt,
            ModelPreheatTask.execution_state == ModelPreheatExecutionStateEnum.ERROR,
        )
        .values(
            attempt=expected_attempt + 1,
            desired_state=ModelPreheatDesiredStateEnum.RUNNING,
            execution_state=ModelPreheatExecutionStateEnum.PENDING,
            paused_from_state=None,
            state_message=None,
            progress=0,
            finished_at=None,
        )
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        await session.rollback()
        task = await _task_or_404(session, id)
        return _to_public(task)
    await session.exec(
        delete(ModelPreheatTaskLock).where(ModelPreheatTaskLock.task_id == id)
    )
    session.add(
        ModelPreheatTaskLock(
            operation_key=operation_key,
            task_id=id,
            lease_expires_at=datetime.now(timezone.utc) + LOCK_TTL,
        )
    )
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "Conflict", "model_preheat_operation_conflict")
    task = await _task_or_404(session, id)
    return _to_public(task)


@router.post("", response_model=ModelPreheatTaskPublic)
async def create_model_preheat(
    request: Request,
    session: SessionDep,
    current_user: CurrentAdminUserDep,
    task_in: ModelPreheatCreate,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    request_hash = canonical_request_hash(
        task_in.model_dump(mode="json", exclude_none=True)
    )
    record = await get_idempotency_record(
        session, current_user.id, CREATE_OPERATION, idempotency_key
    )
    if record is not None:
        if record.request_hash != request_hash:
            raise HTTPException(409, "idempotency_key_reused", "idempotency_key_reused")
        task = await ModelPreheatTask.one_by_id(session, record.resource_id)
        if task is None:
            raise InternalServerErrorException(message="idempotency_resource_not_found")
        return _to_public(task)

    profile = await ModelPreheatS3Profile.one_by_id(session, task_in.s3_profile_id)
    if profile is None:
        raise NotFoundException(message="model_preheat_s3_profile_not_found")
    workers, seed_worker, target_gpu_names = await _resolve_target_workers(
        session, task_in
    )
    if not workers:
        raise InvalidException(message="no_online_workers")
    await _ensure_profile_available_on_workers(
        session, profile, request.app.state.server_config, workers
    )
    target_snapshot = _target_snapshot(workers)
    target_worker_uuids = [item["worker_uuid"] for item in target_snapshot]
    pattern_digest = selection_digest(
        task_in.include_patterns, task_in.exclude_patterns
    )
    resolver = getattr(
        request.app.state,
        "model_preheat_revision_resolver",
        resolve_model_preheat_revision,
    )
    try:
        resolved_revision = await asyncio.to_thread(
            resolver,
            task_in.source,
            task_in.model_id,
            task_in.revision,
            token=getattr(request.app.state.server_config, "huggingface_token", None),
        )
    except Exception:
        raise InvalidException(message="remote_revision_resolution_failed") from None
    cache_key = cache_key_for(
        task_in.source, task_in.model_id, resolved_revision, pattern_digest
    )
    operation_key = operation_key_for(
        profile.id, cache_key, target_worker_uuids, task_in.s3_backfill_policy
    )

    existing = await _active_task_for_operation(session, operation_key)
    if existing is not None:
        resource_id = await _save_idempotency_record(
            session,
            current_user.id,
            idempotency_key,
            request_hash,
            existing.id,
        )
        replay = await ModelPreheatTask.one_by_id(session, resource_id)
        if replay is None:
            raise InternalServerErrorException(message="idempotency_resource_not_found")
        return _to_public(replay, deduplicated=True)

    cipher = _cipher_from_request(request)
    try:
        profile_snapshot = _profile_snapshot(cipher, profile)
    except CredentialEncryptionUnavailable as exc:
        raise ServiceUnavailableException(
            message=f"credential_encryption_unavailable: {exc}"
        )

    task = ModelPreheatTask(
        source=task_in.source,
        model_id=task_in.model_id,
        requested_revision=task_in.revision,
        resolved_revision=resolved_revision,
        include_patterns=task_in.include_patterns,
        exclude_patterns=task_in.exclude_patterns,
        selection_digest=pattern_digest,
        cache_key=cache_key,
        generation_id=f"preheat-{uuid4()}",
        seed_worker_uuid=seed_worker.worker_uuid,
        seed_worker_id=seed_worker.id,
        target_scope=task_in.target_scope,
        target_gpu_names=target_gpu_names,
        target_worker_uuids=target_worker_uuids,
        target_worker_snapshot=target_snapshot,
        s3_profile_id=profile.id,
        s3_profile_config_version=profile.config_version,
        s3_profile_snapshot_encrypted=profile_snapshot,
        encryption_key_version=cipher.current_key_version,
        s3_backfill_policy=task_in.s3_backfill_policy,
        keep_new_workers_in_sync=task_in.keep_new_workers_in_sync,
        created_by_user_id=current_user.id,
    )
    try:
        session.add(task)
        await session.flush()
        session.add(
            ModelPreheatTaskLock(
                operation_key=operation_key,
                task_id=task.id,
                lease_expires_at=datetime.now(timezone.utc) + LOCK_TTL,
            )
        )
        record = new_idempotency_record(
            current_user.id,
            CREATE_OPERATION,
            idempotency_key,
            request_hash,
            task.id,
        )
        if record is not None:
            session.add(record)
        await session.commit()
        await session.refresh(task)
    except IntegrityError:
        await session.rollback()
        record = await get_idempotency_record(
            session, current_user.id, CREATE_OPERATION, idempotency_key
        )
        if record is not None:
            if record.request_hash != request_hash:
                raise HTTPException(
                    409, "idempotency_key_reused", "idempotency_key_reused"
                )
            replay = await ModelPreheatTask.one_by_id(session, record.resource_id)
            if replay is not None:
                return _to_public(replay)
        existing = await _active_task_for_operation(session, operation_key)
        if existing is not None:
            resource_id = await _save_idempotency_record(
                session,
                current_user.id,
                idempotency_key,
                request_hash,
                existing.id,
            )
            replay = await ModelPreheatTask.one_by_id(session, resource_id)
            if replay is None:
                raise InternalServerErrorException(
                    message="idempotency_resource_not_found"
                )
            return _to_public(replay, deduplicated=True)
        raise InternalServerErrorException(message="failed_to_create_model_preheat")
    except Exception as exc:
        await session.rollback()
        raise InternalServerErrorException(
            message=f"failed_to_create_model_preheat: {type(exc).__name__}"
        )
    return _to_public(task)


async def release_task_lock_if_terminal(session, task: ModelPreheatTask) -> bool:
    if not is_terminal_task(task):
        return False
    await session.exec(
        delete(ModelPreheatTaskLock).where(ModelPreheatTaskLock.task_id == task.id)
    )
    await session.flush()
    return True


async def set_task_execution_state(
    session, task: ModelPreheatTask, execution_state: ModelPreheatExecutionStateEnum
) -> bool:
    changed = task.execution_state != execution_state
    if changed:
        task.execution_state = execution_state
        session.add(task)
    if is_terminal_task(task):
        await release_task_lock_if_terminal(session, task)
    return changed


async def _active_task_for_operation(session, operation_key: str):
    lock = (
        await session.exec(
            select(ModelPreheatTaskLock).where(
                ModelPreheatTaskLock.operation_key == operation_key
            )
        )
    ).first()
    if lock is None:
        return None
    task = await ModelPreheatTask.one_by_id(session, lock.task_id)
    if task is None:
        return None
    if is_terminal_task(task):
        await session.delete(lock)
        await session.flush()
        return None
    return task


async def _resolve_target_workers(session, task_in: ModelPreheatCreate):
    ready_workers = await current_ready_workers(session)
    if not ready_workers:
        raise InvalidException(message="no_online_workers")
    workers_by_id = {worker.id: worker for worker in ready_workers}

    if task_in.target_scope == ModelPreheatTargetScopeEnum.SELECTED_WORKERS:
        workers = [
            workers_by_id.get(worker_id) for worker_id in task_in.target_worker_ids
        ]
        if any(worker is None for worker in workers):
            raise InvalidException(message="target_workers_not_online")
        if task_in.seed_worker_id is None:
            seed_worker = min(workers, key=lambda worker: worker.worker_uuid)
        else:
            seed_worker = workers_by_id.get(task_in.seed_worker_id)
            if seed_worker is None:
                raise InvalidException(message="seed_worker_not_online")
            if seed_worker not in workers:
                raise InvalidException(message="seed_worker_not_in_target_scope")
        return workers, seed_worker, []

    seed_worker = workers_by_id.get(task_in.seed_worker_id)
    if seed_worker is None:
        raise InvalidException(message="seed_worker_not_online")
    if task_in.target_scope == ModelPreheatTargetScopeEnum.SEED_WORKER:
        return [seed_worker], seed_worker, []

    target_gpu_names = _normalized_gpu_names(seed_worker)
    if not target_gpu_names:
        raise InvalidException(message="seed_worker_gpu_required")
    workers = [
        worker
        for worker in ready_workers
        if target_gpu_names.intersection(_normalized_gpu_names(worker))
    ]
    return workers, seed_worker, sorted(target_gpu_names)


def _normalized_gpu_names(worker: Worker) -> set[str]:
    gpu_devices = getattr(worker.status, "gpu_devices", None) or []
    return {
        " ".join(gpu.name.split()).casefold()
        for gpu in gpu_devices
        if gpu.name and gpu.name.strip()
    }


async def _ensure_profile_available_on_workers(
    session,
    profile: ModelPreheatS3Profile,
    server_config,
    target_workers: list[Worker],
):
    target_worker_uuids = [worker.worker_uuid for worker in target_workers]
    ttl = connectivity_ttl_from_config(server_config)
    became_stale = await mark_profile_stale_if_expired(session, profile, ttl)
    if became_stale:
        await session.commit()
        await session.refresh(profile)
    current_workers = await current_ready_workers(session)
    current_results = await latest_connectivity_results_for_workers(
        session,
        profile.id,
        profile.config_version,
        current_workers,
    )
    target_results = {
        worker.worker_uuid: current_results.get(worker.worker_uuid)
        for worker in target_workers
    }
    now = datetime.now(timezone.utc)
    target_results_are_fresh = all(
        result is not None
        and result[0].state == ModelPreheatWorkerTaskStateEnum.READY
        and result[1].finished_at is not None
        and result[1].finished_at + ttl >= now
        for result in target_results.values()
    )
    current_worker_results_complete = set(current_results) == {
        worker.worker_uuid for worker in current_workers
    }
    if (
        target_results_are_fresh
        and current_worker_results_complete
        and profile.last_connectivity_check_id is not None
    ):
        return

    if not target_results_are_fresh:
        quick_check = None
        if became_stale or profile.connectivity_state in {
            ModelPreheatS3ConnectivityStateEnum.STALE,
            ModelPreheatS3ConnectivityStateEnum.CHECKING,
        }:
            quick_check = await create_or_reuse_connectivity_check(
                session,
                profile,
                target_worker_uuids=target_worker_uuids,
            )
        if quick_check is not None:
            raise InvalidException(
                message=(
                    "s3_unavailable_on_workers: "
                    f"connectivity_check_id={quick_check.id}"
                )
            )
        raise InvalidException(message="s3_unavailable_on_workers")
    raise InvalidException(message="s3_unavailable_on_workers")


def _target_snapshot(workers: list[Worker]) -> list[dict]:
    return sorted(
        [
            {
                "worker_uuid": worker.worker_uuid,
                "worker_id": worker.id,
                "worker_name": worker.name,
            }
            for worker in workers
        ],
        key=lambda item: item["worker_uuid"],
    )


async def _save_idempotency_record(
    session, user_id, idempotency_key, request_hash, task_id
) -> int:
    record = new_idempotency_record(
        user_id, CREATE_OPERATION, idempotency_key, request_hash, task_id
    )
    if record is None:
        return task_id
    session.add(record)
    try:
        await session.commit()
        return task_id
    except IntegrityError:
        await session.rollback()
        existing = await get_idempotency_record(
            session, user_id, CREATE_OPERATION, idempotency_key
        )
        if existing is None:
            raise InternalServerErrorException(
                message="failed_to_save_idempotency_record"
            )
        if existing.request_hash != request_hash:
            raise HTTPException(409, "idempotency_key_reused", "idempotency_key_reused")
        return existing.resource_id


def _cipher_from_request(request: Request) -> ModelPreheatCredentialCipher:
    config = request.app.state.server_config
    return ModelPreheatCredentialCipher(
        current_key=getattr(config, "model_preheat_credential_key", None),
        current_key_version=getattr(
            config, "model_preheat_credential_key_version", None
        ),
        old_keys=getattr(config, "model_preheat_credential_old_keys", None),
    )


def _profile_snapshot(
    cipher: ModelPreheatCredentialCipher, profile: ModelPreheatS3Profile
):
    snapshot = {
        "endpoint": profile.endpoint,
        "bucket": profile.bucket,
        "prefix": profile.prefix,
        "tls_enabled": profile.tls_enabled,
        "tls_verify": profile.tls_verify,
        "region": profile.region,
        "use_virtual_hosted_style": profile.use_virtual_hosted_style,
        "access_key_encrypted": profile.access_key_encrypted,
        "secret_key_encrypted": profile.secret_key_encrypted,
    }
    return cipher.encrypt(json.dumps(snapshot, separators=(",", ":"), sort_keys=True))


def _to_public(task: ModelPreheatTask, deduplicated: bool = False):
    return ModelPreheatTaskPublic(
        **task.model_dump(
            exclude={"s3_profile_snapshot_encrypted", "encryption_key_version"}
        ),
        created_at=task.created_at,
        updated_at=task.updated_at,
        deduplicated=deduplicated,
    )


async def _task_or_404(session, task_id):
    task = await ModelPreheatTask.one_by_id(session, task_id)
    if task is None:
        raise NotFoundException(message="model_preheat_task_not_found")
    return task


async def _transition_parent_and_children(
    session,
    task,
    *,
    desired_state,
    execution_state,
    child_state,
    from_child_states=None,
):
    expected_desired_state = task.desired_state
    expected_execution_state = task.execution_state
    parent_values = {
        "desired_state": desired_state,
        "execution_state": execution_state,
    }
    if execution_state == ModelPreheatExecutionStateEnum.PAUSED:
        parent_values["paused_from_state"] = expected_execution_state
    elif desired_state == ModelPreheatDesiredStateEnum.RUNNING:
        parent_values["paused_from_state"] = None
    result = await session.exec(
        update(ModelPreheatTask)
        .where(
            ModelPreheatTask.id == task.id,
            ModelPreheatTask.attempt == task.attempt,
            ModelPreheatTask.desired_state == expected_desired_state,
            ModelPreheatTask.execution_state == expected_execution_state,
        )
        .values(**parent_values)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        await session.rollback()
        await session.refresh(task)
        return False
    active_states = from_child_states or {
        ModelPreheatWorkerTaskStateEnum.PENDING,
        ModelPreheatWorkerTaskStateEnum.RUNNING,
        ModelPreheatWorkerTaskStateEnum.PAUSED,
    }
    await session.exec(
        update(ModelPreheatWorkerTask)
        .where(
            ModelPreheatWorkerTask.task_id == task.id,
            ModelPreheatWorkerTask.parent_attempt == task.attempt,
            ModelPreheatWorkerTask.state.in_(active_states),
        )
        .values(
            state=child_state,
            lease_owner=None,
            lease_token_hash=None,
            lease_expires_at=None,
        )
        .execution_options(synchronize_session=False)
    )
    return True
