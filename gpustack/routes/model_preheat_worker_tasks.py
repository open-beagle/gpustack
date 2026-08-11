import hashlib
import hmac
import json
import secrets
from uuid import UUID
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Header, Query, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, func, or_, update
from sqlmodel import select

from gpustack.api.exceptions import (
    HTTPException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
    ModelPreheatCredentialError,
)
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatDesiredStateEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatExecutionProfile,
    ModelPreheatS3ConnectivityCheck,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskClaim,
    ModelPreheatWorkerTaskClaimed,
    ModelPreheatWorkerTaskComplete,
    ModelPreheatWorkerTaskExecutionPayload,
    ModelPreheatWorkerTaskFail,
    ModelPreheatWorkerTaskLease,
    ModelPreheatWorkerTaskProgress,
    ModelPreheatWorkerTaskPublic,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTasksPublic,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.workers import Worker
from gpustack.server.bus import EventType
from gpustack.server.deps import EngineDep, ListParamsDep, SessionDep
from gpustack.server.model_preheat_connectivity import aggregate_connectivity_check
from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentityError,
    decode_path,
    encode_path,
)


router = APIRouter()
LEASE_TTL = timedelta(seconds=60)
CONNECTIVITY_RESULT_FIELDS = {
    "state",
    "readable",
    "writable",
    "deletable",
    "cleanup_failed",
    "latency_ms",
    "error_code",
    "failed_stage",
}
CONNECTIVITY_STATES = {"ready", "error"}
CONNECTIVITY_STAGES = {
    "client",
    "dns",
    "tcp",
    "tls",
    "auth",
    "list",
    "write",
    "read",
    "delete",
}
SAFE_ERROR_CODES = {
    "s3_client_initialization_failed",
    "dns_resolution_failed",
    "tcp_connection_failed",
    "tls_certificate_verify_failed",
    "tls_handshake_failed",
    "s3_authentication_failed",
    "s3_list_failed",
    "s3_write_failed",
    "s3_read_failed",
    "s3_read_content_mismatch",
    "s3_delete_failed",
    "execution_payload_unavailable",
    "worker_execution_failed",
    "lease_lost",
    "validation_error",
    "credential_error",
    "remote_model_not_found",
    "local_cache_conflict",
    "local_cache_staging_cross_device",
    "local_manifest_conflict",
    "local_manifest_invalid",
    "disk_space_insufficient",
    "network_timeout",
    "s3_throttled",
    "checksum_mismatch",
    "s3_ready_not_found",
    "s3_manifest_invalid",
    "s3_object_conflict",
    "ready_generation_conflict",
    "canceled",
    "local_cache_invalid_cache_key",
    "local_cache_invalid_staging_component",
    "local_cache_lock_unavailable",
    "local_cache_path_escape",
    "local_cache_publish_failed",
    "local_cache_scan_failed",
    "local_cache_staging_cleanup_failed",
    "local_cache_staging_conflict",
    "local_cache_staging_create_failed",
    "local_cache_staging_invalid",
    "local_cache_staging_missing",
    "local_manifest_lock_unavailable",
    "local_manifest_write_failed",
}
PREHEAT_RESULT_FIELDS = {
    "state",
    "error_code",
    "manifest_digest",
    "ready_path",
    "manifest_path",
    "generation_id",
    "local_cache_state",
    "uploaded",
    "skipped",
    "downloaded",
    "total_size",
    "cursor",
}
PREHEAT_RESULT_STATES = {"ready", "error"}
LOCAL_CACHE_STATES = {"valid", "candidate", "missing", "conflict", "error"}
PREHEAT_CURSOR_FIELDS = {"completed_files", "staging_exists"}
MAX_PREHEAT_OBJECT_PATH_LENGTH = 2048
MAX_PREHEAT_CURSOR_FILES = 1024
MAX_PREHEAT_CURSOR_PATH_LENGTH = 1024
MAX_STATE_MESSAGE_LENGTH = 256
STATE_MESSAGE_ALLOWLIST = {
    "distributing",
    "downloading",
    "publishing",
    "uploading",
    "verifying",
}


@router.get("", response_model=ModelPreheatWorkerTasksPublic)
async def get_model_preheat_worker_tasks(
    engine: EngineDep,
    session: SessionDep,
    params: ListParamsDep,
    worker_uuid: Optional[str] = None,
    worker_id: Optional[int] = None,
    state: list[ModelPreheatWorkerTaskStateEnum] = Query(default=[]),
):
    fields = {}
    if worker_uuid is not None:
        fields["worker_uuid"] = worker_uuid
    if worker_id is not None:
        fields["worker_id"] = worker_id
    if params.watch:
        return StreamingResponse(
            ModelPreheatWorkerTask.streaming(
                engine,
                fields=fields,
                filter_func=(lambda task: task.state in state) if state else None,
            ),
            media_type="text/event-stream",
        )
    extra_conditions = []
    if state:
        extra_conditions.append(ModelPreheatWorkerTask.state.in_(state))
    return await ModelPreheatWorkerTask.paginated_by_query(
        session=session,
        fields=fields,
        extra_conditions=extra_conditions,
        page=params.page,
        per_page=params.perPage,
    )


@router.get("/{worker_task_id}", response_model=ModelPreheatWorkerTaskPublic)
async def get_model_preheat_worker_task(session: SessionDep, worker_task_id: int):
    return await _task_or_404(session, worker_task_id)


@router.post("/{worker_task_id}/claim", response_model=ModelPreheatWorkerTaskClaimed)
async def claim_model_preheat_worker_task(
    session: SessionDep,
    worker_task_id: int,
    claim: ModelPreheatWorkerTaskClaim,
):
    await _validate_current_registration(session, claim.worker_uuid, claim.worker_id)
    task = await _task_or_404(session, worker_task_id)
    if task.worker_uuid != claim.worker_uuid:
        _conflict("worker_mismatch")
    if task.distribution_policy_id is not None:
        await _active_distribution_source(session, task.distribution_policy_id)

    now = _utcnow()
    lease_token = secrets.token_urlsafe(32)
    lease_token_hash = _hash_token(lease_token)
    claimable = or_(
        ModelPreheatWorkerTask.state == ModelPreheatWorkerTaskStateEnum.PENDING,
        and_(
            ModelPreheatWorkerTask.state == ModelPreheatWorkerTaskStateEnum.RUNNING,
            ModelPreheatWorkerTask.lease_expires_at.is_not(None),
            ModelPreheatWorkerTask.lease_expires_at <= now,
        ),
    )
    result = await session.exec(
        update(ModelPreheatWorkerTask)
        .where(
            ModelPreheatWorkerTask.id == worker_task_id,
            ModelPreheatWorkerTask.worker_uuid == claim.worker_uuid,
            _is_current_registration(claim.worker_uuid, claim.worker_id),
            claimable,
        )
        .values(
            worker_id=claim.worker_id,
            state=ModelPreheatWorkerTaskStateEnum.RUNNING,
            attempt=ModelPreheatWorkerTask.attempt + 1,
            lease_owner=claim.worker_uuid,
            lease_token_hash=lease_token_hash,
            lease_expires_at=now + LEASE_TTL,
            last_heartbeat_at=now,
            started_at=func.coalesce(ModelPreheatWorkerTask.started_at, now),
            finished_at=None,
        )
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        await session.rollback()
        await _task_or_404(session, worker_task_id)
        _conflict("task_not_claimable")
    await session.commit()
    task = await _refresh_task(session, worker_task_id)
    await _publish(task)
    return ModelPreheatWorkerTaskClaimed(
        **_public(task).model_dump(),
        lease_token=lease_token,
        lease_expires_at=task.lease_expires_at,
    )


@router.post(
    "/{worker_task_id}/heartbeat", response_model=ModelPreheatWorkerTaskClaimed
)
async def heartbeat_model_preheat_worker_task(
    session: SessionDep,
    worker_task_id: int,
    lease: ModelPreheatWorkerTaskLease,
):
    task = await _validate_active_lease(session, worker_task_id, lease)
    now = _utcnow()
    expiry = now + LEASE_TTL
    result = await session.exec(
        _active_lease_update(worker_task_id, lease, now).values(
            lease_expires_at=expiry,
            last_heartbeat_at=now,
        )
    )
    await _commit_validated_update(session, task, result.rowcount)
    task = await _refresh_task(session, worker_task_id)
    await _publish(task)
    return ModelPreheatWorkerTaskClaimed(
        **_public(task).model_dump(),
        lease_token=lease.lease_token,
        lease_expires_at=task.lease_expires_at,
    )


@router.patch("/{worker_task_id}/progress", response_model=ModelPreheatWorkerTaskPublic)
async def update_model_preheat_worker_task_progress(
    session: SessionDep,
    worker_task_id: int,
    progress: ModelPreheatWorkerTaskProgress,
):
    task = await _validate_active_lease(session, worker_task_id, progress)
    cursor = _validated_cursor(task.role, progress.resumable_cursor)
    state_message = _validated_state_message(progress.state_message)
    values = {
        "progress": progress.progress,
        "last_heartbeat_at": _utcnow(),
        "resumable_cursor": cursor,
        "state_message": state_message,
    }
    for field in (
        "downloaded_size",
        "total_size",
    ):
        value = getattr(progress, field)
        if value is not None:
            values[field] = value
    result = await session.exec(
        _active_lease_update(worker_task_id, progress, _utcnow()).values(**values)
    )
    await _commit_validated_update(session, task, result.rowcount)
    task = await _refresh_task(session, worker_task_id)
    await _publish(task)
    return task


@router.post("/{worker_task_id}/complete", response_model=ModelPreheatWorkerTaskPublic)
async def complete_model_preheat_worker_task(
    session: SessionDep,
    worker_task_id: int,
    complete: ModelPreheatWorkerTaskComplete,
):
    task = await _validate_active_lease(
        session,
        worker_task_id,
        complete,
        idempotent_state=ModelPreheatWorkerTaskStateEnum.READY,
    )
    if task.state == ModelPreheatWorkerTaskStateEnum.READY:
        await _aggregate_connectivity(session, task)
        return task
    result_payload = _validated_result(task.role, complete.result)
    now = _utcnow()
    result = await session.exec(
        _active_lease_update(worker_task_id, complete, now).values(
            state=ModelPreheatWorkerTaskStateEnum.READY,
            progress=100,
            resumable_cursor=result_payload,
            last_heartbeat_at=now,
            lease_expires_at=None,
            finished_at=now,
        )
    )
    if result.rowcount != 1:
        await session.rollback()
        task = await _validate_active_lease(
            session,
            worker_task_id,
            complete,
            idempotent_state=ModelPreheatWorkerTaskStateEnum.READY,
        )
        await _aggregate_connectivity(session, task)
        return task
    await session.commit()
    task = await _refresh_task(session, worker_task_id)
    await _aggregate_connectivity(session, task)
    await _publish(task)
    return task


@router.post("/{worker_task_id}/fail", response_model=ModelPreheatWorkerTaskPublic)
async def fail_model_preheat_worker_task(
    session: SessionDep,
    worker_task_id: int,
    failure: ModelPreheatWorkerTaskFail,
):
    task = await _validate_active_lease(
        session,
        worker_task_id,
        failure,
        idempotent_state=ModelPreheatWorkerTaskStateEnum.ERROR,
    )
    if failure.error_code not in SAFE_ERROR_CODES:
        raise HTTPException(422, "Invalid", "invalid_error_code")
    result_payload = _validated_result(task.role, failure.result)
    if task.state == ModelPreheatWorkerTaskStateEnum.ERROR:
        await _aggregate_connectivity(session, task)
        return task
    now = _utcnow()
    result = await session.exec(
        _active_lease_update(worker_task_id, failure, now).values(
            state=ModelPreheatWorkerTaskStateEnum.ERROR,
            error_code=failure.error_code,
            state_message=failure.error_code,
            resumable_cursor=result_payload,
            last_heartbeat_at=now,
            lease_expires_at=None,
            finished_at=now,
        )
    )
    if result.rowcount != 1:
        await session.rollback()
        task = await _validate_active_lease(
            session,
            worker_task_id,
            failure,
            idempotent_state=ModelPreheatWorkerTaskStateEnum.ERROR,
        )
        await _aggregate_connectivity(session, task)
        return task
    await session.commit()
    task = await _refresh_task(session, worker_task_id)
    await _aggregate_connectivity(session, task)
    await _publish(task)
    return task


@router.get(
    "/{worker_task_id}/execution-payload",
    response_model=ModelPreheatWorkerTaskExecutionPayload,
)
async def get_model_preheat_worker_task_execution_payload(
    request: Request,
    response: Response,
    session: SessionDep,
    worker_task_id: int,
    worker_uuid: Annotated[str, Header(alias="X-Worker-UUID")],
    worker_id: Annotated[int, Header(alias="X-Worker-ID")],
    attempt: Annotated[int, Header(alias="X-Task-Attempt")],
    lease_token: Annotated[str, Header(alias="X-Lease-Token")],
):
    lease = ModelPreheatWorkerTaskLease(
        worker_uuid=worker_uuid,
        worker_id=worker_id,
        attempt=attempt,
        lease_token=lease_token,
    )
    worker_task = await _validate_active_lease(session, worker_task_id, lease)
    cipher = _cipher_from_request(request)
    try:
        task_payload, encrypted_profile = await _execution_source(session, worker_task)
        profile = _decrypt_profile(cipher, encrypted_profile)
    except (ModelPreheatCredentialError, KeyError, TypeError, ValueError):
        raise ServiceUnavailableException(message="execution_credentials_unavailable")
    response.headers["Cache-Control"] = "no-store"
    return ModelPreheatWorkerTaskExecutionPayload(
        worker_task_id=worker_task.id,
        attempt=worker_task.attempt,
        role=worker_task.role,
        task=task_payload,
        profile=profile,
    )


async def _execution_source(session, worker_task: ModelPreheatWorkerTask):
    if worker_task.task_id is not None:
        task = await session.get(ModelPreheatTask, worker_task.task_id)
        if task is None:
            raise NotFoundException(message="model_preheat_task_not_found")
        payload = task.model_dump(
            exclude={"s3_profile_snapshot_encrypted", "encryption_key_version"}
        )
        return payload, task.s3_profile_snapshot_encrypted

    if worker_task.distribution_policy_id is not None:
        _, task = await _active_distribution_source(
            session, worker_task.distribution_policy_id
        )
        payload = task.model_dump(
            exclude={"s3_profile_snapshot_encrypted", "encryption_key_version"}
        )
        return payload, task.s3_profile_snapshot_encrypted

    check = await session.get(
        ModelPreheatS3ConnectivityCheck, worker_task.connectivity_check_id
    )
    if check is None:
        raise NotFoundException(message="connectivity_check_not_found")
    profile = await session.get(ModelPreheatS3Profile, check.profile_id)
    if profile is None or profile.config_version != check.profile_config_version:
        _conflict("stale_profile_config")
    encrypted_profile = {
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
    return {"connectivity_check_id": check.id}, encrypted_profile


def _decrypt_profile(cipher, encrypted_profile):
    if isinstance(encrypted_profile, (dict, str)) and (
        not isinstance(encrypted_profile, dict)
        or encrypted_profile.get("algorithm") == "AESGCM"
    ):
        encrypted_profile = json.loads(cipher.decrypt(encrypted_profile))
    return ModelPreheatExecutionProfile(
        endpoint=encrypted_profile["endpoint"],
        bucket=encrypted_profile["bucket"],
        prefix=encrypted_profile.get("prefix", ""),
        tls_enabled=encrypted_profile.get("tls_enabled", True),
        tls_verify=encrypted_profile.get("tls_verify", True),
        region=encrypted_profile.get("region", ""),
        use_virtual_hosted_style=encrypted_profile.get(
            "use_virtual_hosted_style", True
        ),
        access_key=cipher.decrypt(encrypted_profile["access_key_encrypted"]),
        secret_key=cipher.decrypt(encrypted_profile["secret_key_encrypted"]),
    )


async def _validate_active_lease(
    session,
    worker_task_id: int,
    lease: ModelPreheatWorkerTaskLease,
    idempotent_state: Optional[ModelPreheatWorkerTaskStateEnum] = None,
):
    task = await _task_or_404(session, worker_task_id)
    if task.distribution_policy_id is not None:
        await _active_distribution_source(session, task.distribution_policy_id)
    elif task.task_id is not None:
        parent = await session.get(ModelPreheatTask, task.task_id)
        if parent is None:
            _conflict("parent_not_running")
        if task.parent_attempt != parent.attempt:
            _conflict("stale_parent_attempt")
        if (
            parent.desired_state != ModelPreheatDesiredStateEnum.RUNNING
            or parent.execution_state
            in {
                ModelPreheatExecutionStateEnum.PAUSED,
                ModelPreheatExecutionStateEnum.CANCELED,
                ModelPreheatExecutionStateEnum.READY,
                ModelPreheatExecutionStateEnum.PARTIAL,
                ModelPreheatExecutionStateEnum.ERROR,
            }
        ):
            _conflict("parent_not_running")
    await _validate_current_registration(session, lease.worker_uuid, lease.worker_id)
    if task.worker_uuid != lease.worker_uuid:
        _conflict("worker_mismatch")
    if task.worker_id != lease.worker_id:
        _conflict("stale_worker_registration")
    if task.attempt != lease.attempt:
        _conflict("stale_attempt")
    if not task.lease_token_hash or not hmac.compare_digest(
        task.lease_token_hash, _hash_token(lease.lease_token)
    ):
        _conflict("invalid_lease_token")
    if idempotent_state is not None and task.state == idempotent_state:
        return task
    if task.state != ModelPreheatWorkerTaskStateEnum.RUNNING:
        _conflict("task_not_running")
    if task.lease_owner != lease.worker_uuid:
        _conflict("lease_not_owned")
    if task.lease_expires_at is None or task.lease_expires_at <= _utcnow():
        _conflict("lease_expired")
    return task


async def _active_distribution_source(session, policy_id):
    policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
    if policy is None or policy.created_by_task_id is None or not policy.enabled:
        _conflict("distribution_policy_not_active")
    source_task = await session.get(ModelPreheatTask, policy.created_by_task_id)
    profile = await session.get(ModelPreheatS3Profile, policy.profile_id)
    if (
        source_task is None
        or profile is None
        or profile.config_version != policy.profile_config_version
        or source_task.s3_profile_config_version != policy.profile_config_version
        or source_task.execution_state != ModelPreheatExecutionStateEnum.READY
        or source_task.cache_key != policy.cache_key
    ):
        _conflict("distribution_source_not_ready")
    return policy, source_task


async def _validate_current_registration(session, worker_uuid: str, worker_id: int):
    current = (
        await session.exec(
            select(Worker)
            .where(Worker.worker_uuid == worker_uuid)
            .order_by(Worker.id.desc())
        )
    ).first()
    if current is None or current.id != worker_id:
        _conflict("stale_worker_registration")
    return current


def _active_lease_update(worker_task_id, lease, now):
    return (
        update(ModelPreheatWorkerTask)
        .where(
            ModelPreheatWorkerTask.id == worker_task_id,
            ModelPreheatWorkerTask.worker_uuid == lease.worker_uuid,
            ModelPreheatWorkerTask.worker_id == lease.worker_id,
            _is_current_registration(lease.worker_uuid, lease.worker_id),
            ModelPreheatWorkerTask.attempt == lease.attempt,
            ModelPreheatWorkerTask.lease_owner == lease.worker_uuid,
            ModelPreheatWorkerTask.lease_token_hash == _hash_token(lease.lease_token),
            ModelPreheatWorkerTask.lease_expires_at > now,
            ModelPreheatWorkerTask.state == ModelPreheatWorkerTaskStateEnum.RUNNING,
        )
        .execution_options(synchronize_session=False)
    )


async def _commit_validated_update(session, task, rowcount):
    if rowcount != 1:
        await session.rollback()
        _conflict("lease_lost")
    await session.commit()


async def _aggregate_connectivity(session, task):
    if task.connectivity_check_id is not None:
        await aggregate_connectivity_check(session, task.connectivity_check_id)


async def _task_or_404(session, worker_task_id):
    task = await session.get(ModelPreheatWorkerTask, worker_task_id)
    if task is None:
        raise NotFoundException(message="model_preheat_worker_task_not_found")
    return task


async def _refresh_task(session, worker_task_id):
    task = await session.get(
        ModelPreheatWorkerTask,
        worker_task_id,
        populate_existing=True,
    )
    if task is None:
        raise NotFoundException(message="model_preheat_worker_task_not_found")
    return task


async def _publish(task):
    await ModelPreheatWorkerTask._publish_event(EventType.UPDATED, task)


def _public(task):
    return ModelPreheatWorkerTaskPublic.model_validate(task)


def _cipher_from_request(request):
    config = request.app.state.server_config
    return ModelPreheatCredentialCipher(
        current_key=getattr(config, "model_preheat_credential_key", None),
        current_key_version=getattr(
            config, "model_preheat_credential_key_version", None
        ),
        old_keys=getattr(config, "model_preheat_credential_old_keys", None),
    )


def _hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _is_current_registration(worker_uuid, worker_id):
    latest_id = (
        select(func.max(Worker.id))
        .where(Worker.worker_uuid == worker_uuid)
        .scalar_subquery()
    )
    return latest_id == worker_id


def _validated_result(role, value):
    if role in {
        ModelPreheatWorkerTaskRoleEnum.SEED,
        ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
    }:
        return _validated_preheat_result(value)
    if role != ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK:
        if value:
            raise HTTPException(422, "Invalid", "worker_result_not_supported")
        return {}
    if not isinstance(value, dict) or set(value) - CONNECTIVITY_RESULT_FIELDS:
        raise HTTPException(422, "Invalid", "invalid_connectivity_result")
    if value.get("state") not in CONNECTIVITY_STATES:
        raise HTTPException(422, "Invalid", "invalid_connectivity_result")
    if value.get("error_code") not in SAFE_ERROR_CODES | {None}:
        raise HTTPException(422, "Invalid", "invalid_connectivity_result")
    if value.get("failed_stage") not in CONNECTIVITY_STAGES | {None}:
        raise HTTPException(422, "Invalid", "invalid_connectivity_result")
    for field in ("readable", "writable", "deletable", "cleanup_failed"):
        if field in value and not isinstance(value[field], bool):
            raise HTTPException(422, "Invalid", "invalid_connectivity_result")
    latency = value.get("latency_ms")
    if latency is not None and (not isinstance(latency, int) or latency < 0):
        raise HTTPException(422, "Invalid", "invalid_connectivity_result")
    return value


def _validated_preheat_result(value):
    if not isinstance(value, dict) or set(value) - PREHEAT_RESULT_FIELDS:
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    if value.get("state") not in PREHEAT_RESULT_STATES:
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    state = value.get("state")
    if value.get("error_code") not in SAFE_ERROR_CODES | {None}:
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    if value.get("local_cache_state") not in LOCAL_CACHE_STATES | {None}:
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    if state == "ready":
        required = {
            "manifest_digest",
            "ready_path",
            "manifest_path",
            "generation_id",
            "local_cache_state",
            "uploaded",
            "skipped",
            "downloaded",
            "total_size",
        }
        if not required <= set(value) or value.get("error_code") is not None:
            raise HTTPException(422, "Invalid", "invalid_preheat_result")
    else:
        ready_only = {
            "manifest_digest",
            "ready_path",
            "manifest_path",
            "generation_id",
            "uploaded",
            "skipped",
            "downloaded",
            "total_size",
        }
        if value.get("error_code") is None or ready_only & set(value):
            raise HTTPException(422, "Invalid", "invalid_preheat_result")
    if "manifest_digest" in value and not _is_sha256(value["manifest_digest"]):
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    for field in ("ready_path", "manifest_path"):
        if field in value and not _is_canonical_object_path(value[field]):
            raise HTTPException(422, "Invalid", "invalid_preheat_result")
    if "generation_id" in value and not _is_safe_generation_id(value["generation_id"]):
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    for field in ("uploaded", "skipped", "downloaded", "total_size"):
        if field in value and (not isinstance(value[field], int) or value[field] < 0):
            raise HTTPException(422, "Invalid", "invalid_preheat_result")
    cursor = _validated_cursor_value(value.get("cursor"))
    if state == "ready":
        return {
            field: value[field]
            for field in (
                "state",
                "manifest_digest",
                "generation_id",
                "local_cache_state",
                "uploaded",
                "skipped",
                "downloaded",
                "total_size",
            )
        }
    sanitized = {
        "state": value["state"],
        "error_code": value["error_code"],
        "local_cache_state": value.get("local_cache_state", "error"),
    }
    if cursor:
        sanitized["cursor"] = cursor
    return sanitized


def _validated_cursor(role, value):
    if value in (None, {}):
        return None
    if role not in {
        ModelPreheatWorkerTaskRoleEnum.SEED,
        ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
    }:
        raise HTTPException(422, "Invalid", "resumable_cursor_not_supported")
    return _validated_cursor_value(value)


def _validated_cursor_value(value):
    if value in (None, {}):
        return value
    if not isinstance(value, dict) or set(value) - PREHEAT_CURSOR_FIELDS:
        raise HTTPException(422, "Invalid", "invalid_preheat_cursor")
    files = value.get("completed_files", [])
    if (
        not isinstance(files, list)
        or len(files) > MAX_PREHEAT_CURSOR_FILES
        or not all(_is_canonical_cursor_path(path) for path in files)
    ):
        raise HTTPException(422, "Invalid", "invalid_preheat_cursor")
    if "staging_exists" in value and not isinstance(value["staging_exists"], bool):
        raise HTTPException(422, "Invalid", "invalid_preheat_cursor")
    if "staging_exists" in value:
        return {"staging_exists": value["staging_exists"]}
    return {}


def _validated_state_message(value):
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) > MAX_STATE_MESSAGE_LENGTH
        or value not in STATE_MESSAGE_ALLOWLIST
    ):
        raise HTTPException(422, "Invalid", "invalid_state_message")
    return value


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_safe_generation_id(value) -> bool:
    if not isinstance(value, str) or not value.startswith("preheat-"):
        return False
    raw_uuid = value.removeprefix("preheat-")
    try:
        return str(UUID(raw_uuid)) == raw_uuid
    except ValueError:
        return False


def _is_canonical_object_path(value) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_PREHEAT_OBJECT_PATH_LENGTH
    ):
        return False
    try:
        return encode_path(decode_path(value)) == value
    except ModelPreheatIdentityError:
        return False


def _is_canonical_cursor_path(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= MAX_PREHEAT_CURSOR_PATH_LENGTH
        and _is_canonical_object_path(value)
    )


def _utcnow():
    return datetime.now(timezone.utc)


def _conflict(reason):
    raise HTTPException(409, reason, reason)
