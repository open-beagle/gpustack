import asyncio
import json
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Header, Request
from sqlalchemy import and_, or_, update
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
    ModelPreheatS3ProfileLifecycleStateEnum,
)
from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
)
from gpustack.schemas.model_preheat_schedules import (
    ModelPreheatScheduleRun,
    ModelPreheatScheduleRunStateEnum,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatCreate,
    ModelPreheatDeliveryModeEnum,
    ModelPreheatDesiredStateEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatArtifact,
    ModelPreheatInventoryManifestStateEnum,
    ModelPreheatTask,
    ModelPreheatIdempotencyRecord,
    ModelPreheatTaskLock,
    ModelPreheatTaskPublic,
    ModelPreheatTasksPublic,
    ModelPreheatTargetScopeEnum,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskPublic,
    ModelPreheatWorkerTaskStateEnum,
    is_terminal_task,
    operation_key_for,
    selection_digest,
)
from gpustack.schemas.workers import MODEL_STORAGE_PROTOCOL_VERSION, Worker
from gpustack.server.deps import CurrentAdminUserDep, ListParamsDep, SessionDep
from gpustack.server.model_preheat_idempotency import (
    canonical_request_hash,
    get_idempotency_record,
    new_idempotency_record,
)
from gpustack.server.model_preheat_s3_profile_lifecycle import (
    ModelPreheatS3ProfileNotActive,
    lock_active_profile_for_new_work,
)
from gpustack.server.model_preheat_revision import resolve_model_preheat_revision
from gpustack.server.model_preheat_connectivity import (
    connectivity_ttl_from_config,
    create_or_reuse_connectivity_check,
    current_ready_workers,
    latest_connectivity_results_for_workers,
    mark_profile_stale_if_expired,
)
from gpustack.utils.gpu import normalize_gpu_names
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity


router = APIRouter()
CREATE_OPERATION = "model_preheats.create"
LOCK_TTL = timedelta(hours=24)


@router.get("", response_model=ModelPreheatTasksPublic)
async def get_model_preheats(session: SessionDep, params: ListParamsDep):
    page = await ModelPreheatTask.paginated_by_query(
        session=session, page=params.page, per_page=params.perPage
    )
    return ModelPreheatTasksPublic(
        items=[_to_public(task) for task in page.items], pagination=page.pagination
    )


@router.get("/{id}", response_model=ModelPreheatTaskPublic)
async def get_model_preheat(session: SessionDep, id: int):
    task = await ModelPreheatTask.one_by_id(session, id)
    if task is None:
        raise NotFoundException(message="model_preheat_task_not_found")
    return _to_public(
        task,
        worker_tasks=await _worker_tasks_public(session, task.id, task.attempt),
    )


@router.delete("/{id}")
async def delete_model_preheat(session: SessionDep, id: int):
    task = await ModelPreheatTask.one_by_id(session, id)
    if task is None:
        raise NotFoundException(message="model_preheat_task_not_found")
    if not is_terminal_task(task):
        raise HTTPException(409, "Conflict", "model_preheat_task_in_use")
    await release_task_lock_if_terminal(session, task)
    await finish_schedule_runs_for_terminal_task(session, task)
    await session.exec(
        update(ModelPreheatScheduleRun)
        .where(ModelPreheatScheduleRun.task_id == task.id)
        .values(task_id=None)
    )
    await session.exec(
        update(ModelPreheatArtifact)
        .where(ModelPreheatArtifact.created_by_task_id == task.id)
        .values(created_by_task_id=None)
    )
    await session.exec(
        update(ModelPreheatDistributionPolicy)
        .where(ModelPreheatDistributionPolicy.created_by_task_id == task.id)
        .values(created_by_task_id=None)
    )
    await session.exec(
        delete(ModelPreheatIdempotencyRecord).where(
            ModelPreheatIdempotencyRecord.resource_type == "model_preheat_task",
            ModelPreheatIdempotencyRecord.resource_id == task.id,
        )
    )
    await session.exec(
        delete(ModelPreheatWorkerTask).where(ModelPreheatWorkerTask.task_id == task.id)
    )
    await session.delete(task)
    await session.commit()
    return {"ok": True}


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
    _reject_schedule_managed_action(task)
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
    _reject_schedule_managed_action(task)
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
        allow_pause_ack_pending=True,
        include_pause_requested_running=True,
    )
    await session.commit()
    await session.refresh(task)
    return _to_public(task)


@router.post("/{id}/retry", response_model=ModelPreheatTaskPublic)
async def retry_model_preheat(session: SessionDep, id: int):
    task = await _task_or_404(session, id)
    _reject_schedule_managed_action(task)
    if task.execution_state != ModelPreheatExecutionStateEnum.ERROR:
        return _to_public(task)
    expected_attempt = task.attempt
    operation_key = operation_key_for(
        task.s3_profile_id,
        task.request_digest,
        task.target_worker_uuids,
        task.s3_backfill_policy,
        task.delivery_mode,
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
    if profile.lifecycle_state != ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE:
        raise InvalidException(message="model_preheat_s3_profile_in_maintenance")
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
            include_patterns=task_in.include_patterns,
            exclude_patterns=task_in.exclude_patterns,
            token=getattr(request.app.state.server_config, "huggingface_token", None),
        )
    except Exception:
        raise InvalidException(message="remote_revision_resolution_failed") from None
    # Registry 无法证明 Ollama tag 不可变时，Seed 会在下载真实单文件后生成快照。
    resolved_revision = resolved_revision or "ollama-pending"
    identity = ModelPreheatIdentity(
        source=task_in.source,
        model_id=task_in.model_id,
        revision=resolved_revision,
        requested_revision=task_in.revision,
        file_patterns=task_in.include_patterns,
        exclude_patterns=task_in.exclude_patterns,
    )
    matched_artifact = await _exact_artifact_match(session, profile, identity)
    if (
        task_in.delivery_mode == ModelPreheatDeliveryModeEnum.S3_ONLY
        and matched_artifact is not None
    ):
        workers, seed_worker, target_gpu_names = [], None, []
    elif task_in.delivery_mode == ModelPreheatDeliveryModeEnum.S3_ONLY:
        seed_worker = await _resolve_s3_only_seed_worker(session, task_in)
        workers, target_gpu_names = [], []
        await _ensure_profile_available_on_workers(
            session,
            profile,
            request.app.state.server_config,
            [seed_worker],
            allow_failure=task_in.connectivity_failure_override,
        )
    else:
        workers, seed_worker, target_gpu_names = await _resolve_target_workers(
            session, task_in
        )
        if not workers:
            raise InvalidException(message="no_online_workers")
        await _ensure_profile_available_on_workers(
            session,
            profile,
            request.app.state.server_config,
            workers,
            allow_failure=task_in.connectivity_failure_override,
        )
    target_snapshot = _target_snapshot(workers)
    target_worker_uuids = [item["worker_uuid"] for item in target_snapshot]
    operation_key = operation_key_for(
        profile.id,
        identity.request_digest,
        target_worker_uuids,
        task_in.s3_backfill_policy,
        task_in.delivery_mode,
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

    try:
        await lock_active_profile_for_new_work(
            session, profile.id, profile.config_version
        )
    except ModelPreheatS3ProfileNotActive:
        raise InvalidException(
            message="model_preheat_s3_profile_in_maintenance"
        ) from None
    task = ModelPreheatTask(
        source=task_in.source,
        model_id=task_in.model_id,
        requested_revision=task_in.revision,
        resolved_revision=resolved_revision,
        include_patterns=task_in.include_patterns,
        exclude_patterns=task_in.exclude_patterns,
        selection_digest=pattern_digest,
        request_identity=_request_identity(identity),
        request_digest=identity.request_digest,
        artifact_id=(matched_artifact.artifact_id if matched_artifact else None),
        seed_worker_uuid=(seed_worker.worker_uuid if seed_worker else None),
        seed_worker_id=(seed_worker.id if seed_worker else None),
        target_scope=task_in.target_scope,
        target_gpu_names=target_gpu_names,
        target_worker_uuids=target_worker_uuids,
        target_worker_snapshot=target_snapshot,
        s3_profile_id=profile.id,
        s3_profile_config_version=profile.config_version,
        s3_profile_snapshot_encrypted=profile_snapshot,
        encryption_key_version=cipher.current_key_version,
        s3_backfill_policy=task_in.s3_backfill_policy,
        delivery_mode=task_in.delivery_mode,
        s3_manifest_path=(matched_artifact.manifest_path if matched_artifact else None),
        manifest_digest=(
            matched_artifact.manifest_digest if matched_artifact else None
        ),
        keep_new_workers_in_sync=task_in.keep_new_workers_in_sync,
        connectivity_failure_override=task_in.connectivity_failure_override,
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


async def finish_schedule_runs_for_terminal_task(session, task: ModelPreheatTask):
    if not is_terminal_task(task):
        return
    state = (
        ModelPreheatScheduleRunStateEnum.READY
        if task.execution_state
        in {
            ModelPreheatExecutionStateEnum.READY,
            ModelPreheatExecutionStateEnum.PARTIAL,
        }
        else ModelPreheatScheduleRunStateEnum.ERROR
    )
    await session.exec(
        update(ModelPreheatScheduleRun)
        .where(
            ModelPreheatScheduleRun.task_id == task.id,
            ModelPreheatScheduleRun.state.in_(
                [
                    ModelPreheatScheduleRunStateEnum.PENDING,
                    ModelPreheatScheduleRunStateEnum.RUNNING,
                    ModelPreheatScheduleRunStateEnum.PAUSED,
                ]
            ),
        )
        .values(
            state=state,
            error_code=(
                None
                if state == ModelPreheatScheduleRunStateEnum.READY
                else task.state_message or "model_preheat_task_failed"
            ),
            finished_at=datetime.now(timezone.utc),
            slot=None,
        )
    )


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
    ready_workers = [
        worker
        for worker in await current_ready_workers(session)
        if worker.model_storage_protocol_version == MODEL_STORAGE_PROTOCOL_VERSION
    ]
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


async def _resolve_s3_only_seed_worker(session, task_in: ModelPreheatCreate):
    ready_workers = [
        worker
        for worker in await current_ready_workers(session)
        if worker.model_storage_protocol_version == MODEL_STORAGE_PROTOCOL_VERSION
    ]
    workers_by_id = {worker.id: worker for worker in ready_workers}
    if task_in.seed_worker_id is not None:
        worker = workers_by_id.get(task_in.seed_worker_id)
        if worker is None:
            raise InvalidException(message="seed_worker_not_online")
        return worker
    selected = [workers_by_id.get(worker_id) for worker_id in task_in.target_worker_ids]
    selected = [worker for worker in selected if worker is not None]
    if selected:
        return min(selected, key=lambda worker: worker.worker_uuid)
    if not ready_workers:
        raise InvalidException(message="no_online_workers")
    return min(ready_workers, key=lambda worker: worker.worker_uuid)


def _normalized_gpu_names(worker: Worker) -> set[str]:
    gpu_devices = getattr(worker.status, "gpu_devices", None) or []
    return normalize_gpu_names(gpu.name for gpu in gpu_devices)


async def _ensure_profile_available_on_workers(
    session,
    profile: ModelPreheatS3Profile,
    server_config,
    target_workers: list[Worker],
    *,
    allow_failure: bool = False,
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
    failed = [
        worker_uuid
        for worker_uuid, result in target_results.items()
        if result is not None
        and result[0].state == ModelPreheatWorkerTaskStateEnum.ERROR
        and result[1].finished_at is not None
        and result[1].finished_at + ttl > datetime.now(timezone.utc)
    ]
    if failed and not allow_failure:
        raise InvalidException(
            message="s3_unavailable_on_workers_confirmation_required"
        )
    # 未检测、检查中、过期及历史结果均为诊断信息。真实 Worker 执行仍会校验 S3。
    return


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
        "source_fallback_enabled": profile.source_fallback_enabled,
        "access_key_encrypted": profile.access_key_encrypted,
        "secret_key_encrypted": profile.secret_key_encrypted,
    }
    return cipher.encrypt(json.dumps(snapshot, separators=(",", ":"), sort_keys=True))


def _request_identity(identity: ModelPreheatIdentity) -> dict:
    return {
        "source": identity.source,
        "model_id": identity.model_path,
        "requested_revision": identity.requested_revision_path,
        "include_patterns": list(identity.file_patterns),
        "exclude_patterns": list(identity.exclude_patterns),
    }


async def _exact_artifact_match(session, profile, identity):
    rows = (
        await session.exec(
            select(ModelPreheatArtifact).where(
                ModelPreheatArtifact.profile_id == profile.id,
                ModelPreheatArtifact.profile_config_version == profile.config_version,
                ModelPreheatArtifact.source == identity.source,
                ModelPreheatArtifact.model_id == identity.model_path,
                ModelPreheatArtifact.resolved_revision == identity.revision,
                ModelPreheatArtifact.manifest_state
                == ModelPreheatInventoryManifestStateEnum.VALID,
            )
        )
    ).all()
    matches = [
        row
        for row in rows
        if tuple(sorted(row.include_patterns or ())) == tuple(identity.file_patterns)
        and tuple(sorted(row.exclude_patterns or ()))
        == tuple(identity.exclude_patterns)
    ]
    return matches[0] if len(matches) == 1 else None


def _to_public(
    task: ModelPreheatTask,
    deduplicated: bool = False,
    worker_tasks: list[ModelPreheatWorkerTaskPublic] | None = None,
):
    values = task.model_dump(
        exclude={"s3_profile_snapshot_encrypted", "encryption_key_version"}
    )
    if values.get("resolved_revision") == "ollama-pending":
        values["resolved_revision"] = None
    return ModelPreheatTaskPublic(
        **values,
        created_at=task.created_at,
        updated_at=task.updated_at,
        deduplicated=deduplicated,
        worker_tasks=worker_tasks or [],
    )


async def _worker_tasks_public(session, task_id, attempt):
    tasks = (
        await session.exec(
            select(ModelPreheatWorkerTask)
            .where(
                ModelPreheatWorkerTask.task_id == task_id,
                ModelPreheatWorkerTask.parent_attempt == attempt,
            )
            .order_by(ModelPreheatWorkerTask.id)
        )
    ).all()
    if not tasks:
        return []
    worker_ids = {task.worker_id for task in tasks if task.worker_id is not None}
    worker_uuids = {task.worker_uuid for task in tasks if task.worker_uuid}
    conditions = []
    if worker_ids:
        conditions.append(Worker.id.in_(worker_ids))
    if worker_uuids:
        conditions.append(Worker.worker_uuid.in_(worker_uuids))
    workers = {}
    if conditions:
        rows = (await session.exec(select(Worker).where(or_(*conditions)))).all()
        for worker in rows:
            workers[("id", worker.id)] = worker
            workers[("uuid", worker.worker_uuid)] = worker
    result = []
    for task in tasks:
        worker = workers.get(("id", task.worker_id)) or workers.get(
            ("uuid", task.worker_uuid)
        )
        result.append(
            ModelPreheatWorkerTaskPublic.model_validate(
                task,
                update={
                    "worker_name": worker.name if worker is not None else None,
                    "worker_ip": worker.ip if worker is not None else None,
                },
            )
        )
    return result


async def _task_or_404(session, task_id):
    task = await ModelPreheatTask.one_by_id(session, task_id)
    if task is None:
        raise NotFoundException(message="model_preheat_task_not_found")
    return task


def _reject_schedule_managed_action(task):
    if task.schedule_id is not None:
        raise HTTPException(
            409,
            "Conflict",
            "model_preheat_schedule_managed_action",
        )


async def _transition_parent_and_children(
    session,
    task,
    *,
    desired_state,
    execution_state,
    child_state,
    from_child_states=None,
    allow_pause_ack_pending=False,
    include_pause_requested_running=False,
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
    parent_update = update(ModelPreheatTask).where(
        ModelPreheatTask.id == task.id,
        ModelPreheatTask.attempt == task.attempt,
        ModelPreheatTask.desired_state == expected_desired_state,
    )
    if allow_pause_ack_pending:
        parent_update = parent_update.where(
            ModelPreheatTask.execution_state.not_in(
                [
                    ModelPreheatExecutionStateEnum.READY,
                    ModelPreheatExecutionStateEnum.PARTIAL,
                    ModelPreheatExecutionStateEnum.ERROR,
                    ModelPreheatExecutionStateEnum.CANCELED,
                ]
            )
        )
    else:
        parent_update = parent_update.where(
            ModelPreheatTask.execution_state == expected_execution_state
        )
    result = await session.exec(
        parent_update.values(**parent_values).execution_options(
            synchronize_session=False
        )
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
    child_condition = ModelPreheatWorkerTask.state.in_(active_states)
    if include_pause_requested_running:
        child_condition = or_(
            child_condition,
            and_(
                ModelPreheatWorkerTask.state == ModelPreheatWorkerTaskStateEnum.RUNNING,
                ModelPreheatWorkerTask.state_message == "pause_requested",
            ),
        )
    child_values = {
        "state": child_state,
        "lease_owner": None,
        "lease_token_hash": None,
        "lease_expires_at": None,
    }
    if desired_state == ModelPreheatDesiredStateEnum.RUNNING:
        child_values["state_message"] = None
    await session.exec(
        update(ModelPreheatWorkerTask)
        .where(
            ModelPreheatWorkerTask.task_id == task.id,
            ModelPreheatWorkerTask.parent_attempt == task.attempt,
            child_condition,
        )
        .values(**child_values)
        .execution_options(synchronize_session=False)
    )
    return True
