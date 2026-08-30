import hashlib
import hmac
import json
import ntpath
import posixpath
import re
import secrets
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, case, exists, func, or_, update
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
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.model_preheats import (
    ModelPreheatDesiredStateEnum,
    ModelPreheatDeliveryModeEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatExecutionProfile,
    ModelPreheatArtifact,
    ModelPreheatInventoryManifestStateEnum,
    ModelPreheatS3ConnectivityCheck,
    ModelPreheatTask,
    ModelPreheatTrustedLocalCandidate,
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
    is_terminal_task,
)
from gpustack.schemas.workers import MODEL_STORAGE_PROTOCOL_VERSION, Worker
from gpustack.schemas.models import SourceEnum
from gpustack.server.bus import EventType
from gpustack.server.deps import EngineDep, ListParamsDep, SessionDep
from gpustack.server.model_preheat_connectivity import aggregate_connectivity_check
from gpustack.server.model_preheat_distribution_source import (
    DistributionSourceUnavailable,
    resolve_distribution_source,
)
from gpustack.server.model_preheat_worker_identity import (
    ModelPreheatWorkerPrincipal,
    get_model_preheat_worker_identity,
)
from gpustack.server.model_preheat_trusted_local import (
    trusted_local_candidate_for_worker,
)
from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentityError,
    decode_path,
    encode_path,
)


router = APIRouter()
WorkerIdentityDep = Annotated[
    ModelPreheatWorkerPrincipal, Depends(get_model_preheat_worker_identity)
]
LEASE_TTL = timedelta(seconds=60)
_LOCAL_SNAPSHOT_REVISION = re.compile(r"^local-snapshot-[0-9a-f]{64}$")
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
    "model_artifact_not_found",
    "request_digest_mismatch",
    "artifact_manifest_conflict",
    "object_content_conflict",
    "local_file_content_mismatch",
    "ollama_artifact_invalid",
}
PREHEAT_RESULT_FIELDS = {
    "state",
    "error_code",
    "request_digest",
    "artifact_id",
    "manifest_digest",
    "manifest_path",
    "file_count",
    "local_cache_state",
    "transfer_source",
    "uploaded",
    "skipped",
    "downloaded",
    "total_size",
    "resolved_revision",
    "local_dir",
    "resolved_paths",
    "error_type",
    "error_message",
    "cursor",
}
PREHEAT_RESULT_STATES = {"ready", "error"}
LOCAL_CACHE_STATES = {"valid", "candidate", "missing", "conflict", "error"}
PREHEAT_CURSOR_FIELDS = {"completed_files", "staging_exists"}
MAX_PREHEAT_OBJECT_PATH_LENGTH = 2048
MAX_PREHEAT_CURSOR_FILES = 1024
MAX_PREHEAT_CURSOR_PATH_LENGTH = 1024
MAX_PREHEAT_RESOLVED_PATHS = 1024
MAX_PREHEAT_RESOLVED_PATH_LENGTH = 4096
MAX_STATE_MESSAGE_LENGTH = 256
MAX_PREHEAT_ERROR_TYPE_LENGTH = 80
STATE_MESSAGE_ALLOWLIST = {
    "distributing",
    "downloading",
    "publishing",
    "uploading",
    "verifying",
    "paused",
}


@router.get("", response_model=ModelPreheatWorkerTasksPublic)
async def get_model_preheat_worker_tasks(
    engine: EngineDep,
    session: SessionDep,
    params: ListParamsDep,
    identity: WorkerIdentityDep,
    worker_uuid: Optional[str] = None,
    worker_id: Optional[int] = None,
    state: list[ModelPreheatWorkerTaskStateEnum] = Query(default=[]),
):
    fields = {
        "worker_uuid": identity.worker_uuid,
        "worker_id": identity.worker_id,
    }
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
async def get_model_preheat_worker_task(
    session: SessionDep, worker_task_id: int, identity: WorkerIdentityDep
):
    task = await _task_or_404(session, worker_task_id)
    _validate_task_identity(task, identity)
    return task


@router.post("/{worker_task_id}/claim", response_model=ModelPreheatWorkerTaskClaimed)
async def claim_model_preheat_worker_task(
    session: SessionDep,
    worker_task_id: int,
    claim: ModelPreheatWorkerTaskClaim,
    identity: WorkerIdentityDep,
):
    _validate_client_identity(claim, identity)
    await _validate_current_registration(
        session, identity.worker_uuid, identity.worker_id
    )
    task = await _task_or_404(session, worker_task_id)
    _validate_task_identity(task, identity)
    if task.distribution_policy_id is not None:
        await _active_distribution_source(
            session,
            task.distribution_policy_id,
            task.distribution_artifact_id,
            task.distribution_request_digest,
        )
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
    claim_conditions = [
        claimable,
        *_execution_allowed_conditions(task.task_id is not None),
    ]
    result = await session.exec(
        update(ModelPreheatWorkerTask)
        .where(
            ModelPreheatWorkerTask.id == worker_task_id,
            ModelPreheatWorkerTask.worker_uuid == identity.worker_uuid,
            _is_current_registration(identity.worker_uuid, identity.worker_id),
            *claim_conditions,
        )
        .values(
            worker_id=identity.worker_id,
            state=ModelPreheatWorkerTaskStateEnum.RUNNING,
            attempt=ModelPreheatWorkerTask.attempt + 1,
            lease_owner=identity.worker_uuid,
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
    profile_id = None
    if task.task_id is not None:
        parent = await session.get(ModelPreheatTask, task.task_id)
        profile_id = parent.s3_profile_id if parent is not None else None
    elif task.distribution_policy_id is not None:
        policy = await session.get(
            ModelPreheatDistributionPolicy, task.distribution_policy_id
        )
        profile_id = policy.profile_id if policy is not None else None
    if profile_id is not None:
        await session.exec(
            update(ModelPreheatS3Profile)
            .where(
                ModelPreheatS3Profile.id == profile_id,
                ModelPreheatS3Profile.ever_used_at.is_(None),
            )
            .values(ever_used_at=now)
        )
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
    identity: WorkerIdentityDep,
):
    task = await _validate_active_lease(
        session,
        worker_task_id,
        lease,
        identity,
    )
    now = _utcnow()
    expiry = now + LEASE_TTL
    result = await session.exec(
        _active_lease_update(
            worker_task_id,
            lease,
            now,
            require_execution_allowed=True,
            has_parent=task.task_id is not None,
        ).values(
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
    identity: WorkerIdentityDep,
):
    pause_confirmation = progress.state_message == "paused"
    task = await _validate_active_lease(
        session,
        worker_task_id,
        progress,
        identity,
        allow_pause_requested=True,
    )
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
    if pause_confirmation:
        values.update(
            state=ModelPreheatWorkerTaskStateEnum.PAUSED,
            lease_owner=None,
            lease_token_hash=None,
            lease_expires_at=None,
        )
        if task.task_id is not None:
            parent_guard = await session.exec(
                update(ModelPreheatTask)
                .where(
                    ModelPreheatTask.id == task.task_id,
                    ModelPreheatTask.attempt == task.parent_attempt,
                    ModelPreheatTask.desired_state
                    == ModelPreheatDesiredStateEnum.PAUSED,
                    ModelPreheatTask.execution_state.not_in(
                        [
                            ModelPreheatExecutionStateEnum.READY,
                            ModelPreheatExecutionStateEnum.PARTIAL,
                            ModelPreheatExecutionStateEnum.ERROR,
                            ModelPreheatExecutionStateEnum.CANCELED,
                        ]
                    ),
                )
                .values(desired_state=ModelPreheatDesiredStateEnum.PAUSED)
                .execution_options(synchronize_session=False)
            )
            if parent_guard.rowcount != 1:
                await session.rollback()
                _conflict("parent_not_running")
    else:
        values["state_message"] = case(
            (
                ModelPreheatWorkerTask.state_message == "pause_requested",
                "pause_requested",
            ),
            else_=state_message,
        )
    result = await session.exec(
        _active_lease_update(
            worker_task_id,
            progress,
            _utcnow(),
            require_pause_requested=pause_confirmation,
        ).values(**values)
    )
    if result.rowcount != 1:
        await session.rollback()
        _conflict("lease_lost")
    if pause_confirmation and task.task_id is not None:
        await session.exec(
            update(ModelPreheatTask)
            .where(
                ModelPreheatTask.id == task.task_id,
                ModelPreheatTask.attempt == task.parent_attempt,
                ModelPreheatTask.desired_state == ModelPreheatDesiredStateEnum.PAUSED,
                ~exists().where(
                    ModelPreheatWorkerTask.task_id == ModelPreheatTask.id,
                    ModelPreheatWorkerTask.parent_attempt == ModelPreheatTask.attempt,
                    ModelPreheatWorkerTask.state.in_(
                        [
                            ModelPreheatWorkerTaskStateEnum.PENDING,
                            ModelPreheatWorkerTaskStateEnum.RUNNING,
                        ]
                    ),
                ),
            )
            .values(execution_state=ModelPreheatExecutionStateEnum.PAUSED)
            .execution_options(synchronize_session=False)
        )
    await session.commit()
    task = await _refresh_task(session, worker_task_id)
    await _publish(task)
    return task


@router.post("/{worker_task_id}/complete", response_model=ModelPreheatWorkerTaskPublic)
async def complete_model_preheat_worker_task(
    session: SessionDep,
    worker_task_id: int,
    complete: ModelPreheatWorkerTaskComplete,
    identity: WorkerIdentityDep,
):
    task = await _validate_active_lease(
        session,
        worker_task_id,
        complete,
        identity,
        idempotent_state=ModelPreheatWorkerTaskStateEnum.READY,
    )
    if task.state == ModelPreheatWorkerTaskStateEnum.READY:
        await _aggregate_connectivity(session, task)
        return await _refresh_task(session, worker_task_id)
    result_payload = _validated_result(task.role, complete.result)
    model_file_event = None
    now = _utcnow()
    if task.task_id is not None and result_payload.get("state") == "ready":
        model_file_event = await _bind_preheat_artifact(
            session, task, result_payload, now
        )
    elif (
        task.distribution_policy_id is not None
        and result_payload.get("state") == "ready"
    ):
        model_file_event = await _bind_distribution_artifact(
            session, task, result_payload, now
        )
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
            identity,
            idempotent_state=ModelPreheatWorkerTaskStateEnum.READY,
        )
        await _aggregate_connectivity(session, task)
        return await _refresh_task(session, worker_task_id)
    await session.commit()
    task = await _refresh_task(session, worker_task_id)
    if model_file_event is not None:
        event_type, model_file = model_file_event
        await session.refresh(model_file)
        await ModelFile._publish_event(event_type, model_file)
    await _aggregate_connectivity(session, task)
    task = await _refresh_task(session, worker_task_id)
    await _publish(task)
    return task


@router.post("/{worker_task_id}/fail", response_model=ModelPreheatWorkerTaskPublic)
async def fail_model_preheat_worker_task(
    session: SessionDep,
    worker_task_id: int,
    failure: ModelPreheatWorkerTaskFail,
    identity: WorkerIdentityDep,
):
    task = await _validate_active_lease(
        session,
        worker_task_id,
        failure,
        identity,
        idempotent_state=ModelPreheatWorkerTaskStateEnum.ERROR,
    )
    if failure.error_code not in SAFE_ERROR_CODES:
        raise HTTPException(422, "Invalid", "invalid_error_code")
    result_payload = _validated_result(task.role, failure.result)
    state_message = _failure_state_message(failure, result_payload)
    if task.state == ModelPreheatWorkerTaskStateEnum.ERROR:
        await _aggregate_connectivity(session, task)
        return await _refresh_task(session, worker_task_id)
    now = _utcnow()
    result = await session.exec(
        _active_lease_update(worker_task_id, failure, now).values(
            state=ModelPreheatWorkerTaskStateEnum.ERROR,
            error_code=failure.error_code,
            state_message=state_message,
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
            identity,
            idempotent_state=ModelPreheatWorkerTaskStateEnum.ERROR,
        )
        await _aggregate_connectivity(session, task)
        return await _refresh_task(session, worker_task_id)
    await session.commit()
    task = await _refresh_task(session, worker_task_id)
    await _aggregate_connectivity(session, task)
    task = await _refresh_task(session, worker_task_id)
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
    identity: WorkerIdentityDep,
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
    worker_task = await _validate_active_lease(
        session,
        worker_task_id,
        lease,
        identity,
    )
    cipher = _cipher_from_request(request)
    try:
        task_payload, encrypted_profile, source_task = await _execution_source(
            session, worker_task
        )
        profile = _decrypt_profile(cipher, encrypted_profile)
    except (ModelPreheatCredentialError, KeyError, TypeError, ValueError):
        raise ServiceUnavailableException(message="execution_credentials_unavailable")
    response.headers["Cache-Control"] = "no-store"
    trusted_local_candidate = None
    if source_task is not None and worker_task.worker_id is not None:
        candidate = await trusted_local_candidate_for_worker(
            session,
            source_task,
            worker_task.worker_uuid,
            worker_task.worker_id,
        )
        if candidate is not None:
            trusted_local_candidate = ModelPreheatTrustedLocalCandidate(
                source=candidate.source,
                root=candidate.root,
                paths=list(candidate.paths),
                repository_complete=candidate.repository_complete,
            )
    return ModelPreheatWorkerTaskExecutionPayload(
        worker_task_id=worker_task.id,
        attempt=worker_task.attempt,
        role=worker_task.role,
        resumable_cursor=worker_task.resumable_cursor,
        task=task_payload,
        profile=profile,
        trusted_local_candidate=trusted_local_candidate,
    )


async def _execution_source(session, worker_task: ModelPreheatWorkerTask):
    if worker_task.task_id is not None:
        task = await session.get(ModelPreheatTask, worker_task.task_id)
        if task is None:
            raise NotFoundException(message="model_preheat_task_not_found")
        payload = task.model_dump(
            exclude={"s3_profile_snapshot_encrypted", "encryption_key_version"}
        )
        return payload, task.s3_profile_snapshot_encrypted, task

    if worker_task.distribution_policy_id is not None:
        _, source = await _active_distribution_source(
            session,
            worker_task.distribution_policy_id,
            worker_task.distribution_artifact_id,
            worker_task.distribution_request_digest,
        )
        return source.payload, source.encrypted_profile, source.preheat_task

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
    return {"connectivity_check_id": check.id}, encrypted_profile, None


def _decrypt_profile(cipher, encrypted_profile):
    if isinstance(encrypted_profile, (dict, str)) and (
        not isinstance(encrypted_profile, dict)
        or encrypted_profile.get("algorithm") == "AESGCM"
    ):
        encrypted_profile = json.loads(cipher.decrypt(encrypted_profile))
    return ModelPreheatExecutionProfile(
        endpoint=encrypted_profile["endpoint"],
        bucket=encrypted_profile["bucket"],
        prefix=encrypted_profile.get("prefix") or "",
        tls_enabled=encrypted_profile.get("tls_enabled", True),
        tls_verify=encrypted_profile.get("tls_verify", True),
        region=encrypted_profile.get("region") or "",
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
    identity: ModelPreheatWorkerPrincipal,
    idempotent_state: Optional[ModelPreheatWorkerTaskStateEnum] = None,
    allow_pause_requested: bool = False,
):
    task = await _task_or_404(session, worker_task_id)
    _validate_client_identity(lease, identity)
    await _validate_current_registration(
        session, identity.worker_uuid, identity.worker_id
    )
    _validate_task_identity(task, identity)
    if task.worker_id != identity.worker_id:
        _conflict("stale_worker_registration")
    if task.attempt != lease.attempt:
        _conflict("stale_attempt")
    if not task.lease_token_hash or not hmac.compare_digest(
        task.lease_token_hash, _hash_token(lease.lease_token)
    ):
        _conflict("invalid_lease_token")
    if idempotent_state is not None and task.state == idempotent_state:
        return task
    if task.state_message == "pause_requested" and not allow_pause_requested:
        _conflict("parent_not_running")
    if task.distribution_policy_id is not None:
        await _active_distribution_source(
            session,
            task.distribution_policy_id,
            task.distribution_artifact_id,
            task.distribution_request_digest,
        )
    elif task.task_id is not None:
        parent = await session.get(ModelPreheatTask, task.task_id)
        if parent is None:
            _conflict("parent_not_running")
        if task.parent_attempt != parent.attempt:
            _conflict("stale_parent_attempt")
        pause_requested = (
            allow_pause_requested
            and task.state == ModelPreheatWorkerTaskStateEnum.RUNNING
            and task.state_message == "pause_requested"
            and parent.desired_state == ModelPreheatDesiredStateEnum.PAUSED
            and not is_terminal_task(parent)
        )
        if not pause_requested and (
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
    if task.state != ModelPreheatWorkerTaskStateEnum.RUNNING:
        _conflict("task_not_running")
    if task.lease_owner != identity.worker_uuid:
        _conflict("lease_not_owned")
    if task.lease_expires_at is None or task.lease_expires_at <= _utcnow():
        _conflict("lease_expired")
    return task


def _validate_client_identity(value, identity):
    if (
        value.worker_uuid != identity.worker_uuid
        or value.worker_id != identity.worker_id
    ):
        _conflict("worker_mismatch")


def _validate_task_identity(task, identity):
    if task.worker_uuid != identity.worker_uuid:
        _conflict("worker_mismatch")
    if task.worker_id is not None and task.worker_id != identity.worker_id:
        _conflict("stale_worker_registration")


async def _active_distribution_source(
    session, policy_id, artifact_id=None, request_digest=None
):
    policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
    if policy is None or not policy.enabled:
        _conflict("distribution_policy_not_active")
    try:
        source = await resolve_distribution_source(
            session, policy, artifact_id, request_digest
        )
    except DistributionSourceUnavailable:
        _conflict("distribution_source_not_ready")
    return policy, source


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
    if current.model_storage_protocol_version != MODEL_STORAGE_PROTOCOL_VERSION:
        _conflict("model_storage_protocol_mismatch")
    return current


def _active_lease_update(
    worker_task_id,
    lease,
    now,
    *,
    require_pause_requested=False,
    require_execution_allowed=False,
    has_parent=False,
):
    statement = (
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
    if require_pause_requested:
        statement = statement.where(
            ModelPreheatWorkerTask.state_message == "pause_requested"
        )
    if require_execution_allowed:
        statement = statement.where(*_execution_allowed_conditions(has_parent))
    return statement


def _execution_allowed_conditions(has_parent):
    conditions = [
        or_(
            ModelPreheatWorkerTask.state_message.is_(None),
            ModelPreheatWorkerTask.state_message != "pause_requested",
        )
    ]
    if has_parent:
        conditions.append(
            exists(
                select(ModelPreheatTask.id).where(
                    ModelPreheatTask.id == ModelPreheatWorkerTask.task_id,
                    ModelPreheatTask.attempt == ModelPreheatWorkerTask.parent_attempt,
                    ModelPreheatTask.desired_state
                    == ModelPreheatDesiredStateEnum.RUNNING,
                    ModelPreheatTask.execution_state.not_in(
                        [
                            ModelPreheatExecutionStateEnum.PAUSED,
                            ModelPreheatExecutionStateEnum.READY,
                            ModelPreheatExecutionStateEnum.PARTIAL,
                            ModelPreheatExecutionStateEnum.ERROR,
                            ModelPreheatExecutionStateEnum.CANCELED,
                        ]
                    ),
                )
            )
        )
    return conditions


async def _commit_validated_update(session, task, rowcount):
    if rowcount != 1:
        await session.rollback()
        _conflict("lease_lost")
    await session.commit()


async def _aggregate_connectivity(session, task):
    if task.connectivity_check_id is not None:
        await aggregate_connectivity_check(session, task.connectivity_check_id)


async def _bind_preheat_artifact(session, worker_task, result, now):
    parent = await session.get(ModelPreheatTask, worker_task.task_id)
    if parent is None or parent.attempt != worker_task.parent_attempt:
        _conflict("stale_parent_attempt")
    if result["request_digest"] != parent.request_digest:
        _conflict("request_digest_mismatch")
    artifact_id = result["artifact_id"]
    if not result["manifest_path"].endswith(f"/{artifact_id}/manifest.json"):
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    if parent.artifact_id is not None and parent.artifact_id != artifact_id:
        _conflict("artifact_binding_conflict")
    if (
        worker_task.role == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE
        and result["resolved_revision"] != parent.resolved_revision
    ):
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    if worker_task.role == ModelPreheatWorkerTaskRoleEnum.SEED:
        values = {
            "artifact_id": artifact_id,
            "s3_manifest_path": result["manifest_path"],
            "manifest_digest": result["manifest_digest"],
        }
        resolved_revision = result["resolved_revision"]
        if parent.resolved_revision == "ollama-pending":
            if (
                not isinstance(resolved_revision, str)
                or _LOCAL_SNAPSHOT_REVISION.fullmatch(resolved_revision) is None
            ):
                raise HTTPException(422, "Invalid", "invalid_preheat_result")
        elif (
            not isinstance(resolved_revision, str)
            or resolved_revision != parent.resolved_revision
        ):
            raise HTTPException(422, "Invalid", "invalid_preheat_result")
        values["resolved_revision"] = resolved_revision
        transfer_source = result["transfer_source"]
        if transfer_source == "current_node":
            transfer_source = (
                "current_node"
                if worker_task.worker_uuid in parent.target_worker_uuids
                else "peer_via_s3"
            )
        values.update(
            transfer_source=transfer_source,
            transfer_profile_id=(
                parent.s3_profile_id
                if transfer_source in {"peer_via_s3", "s3"}
                else None
            ),
            source_worker_id=(
                worker_task.worker_id
                if transfer_source in {"current_node", "peer_via_s3"}
                else None
            ),
        )
        bound = await session.exec(
            update(ModelPreheatTask)
            .where(
                ModelPreheatTask.id == parent.id,
                ModelPreheatTask.attempt == parent.attempt,
                ModelPreheatTask.request_digest == result["request_digest"],
                or_(
                    ModelPreheatTask.artifact_id.is_(None),
                    ModelPreheatTask.artifact_id == artifact_id,
                ),
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        if bound.rowcount != 1:
            _conflict("artifact_binding_conflict")
        inventory = (
            await session.exec(
                select(ModelPreheatArtifact).where(
                    ModelPreheatArtifact.profile_id == parent.s3_profile_id,
                    ModelPreheatArtifact.profile_config_version
                    == parent.s3_profile_config_version,
                    ModelPreheatArtifact.artifact_id == artifact_id,
                )
            )
        ).first()
        if inventory is None:
            inventory = ModelPreheatArtifact(
                profile_id=parent.s3_profile_id,
                profile_config_version=parent.s3_profile_config_version,
                artifact_id=artifact_id,
                source=parent.source,
                model_id=parent.request_identity["model_id"],
                resolved_revision=resolved_revision or parent.resolved_revision,
                include_patterns=parent.request_identity.get("include_patterns", []),
                exclude_patterns=parent.request_identity.get("exclude_patterns", []),
                manifest_path=result["manifest_path"],
                manifest_digest=result["manifest_digest"],
                file_count=result["file_count"],
                total_size=result["total_size"],
                manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
                last_verified_at=now,
                created_by_task_id=parent.id,
            )
            session.add(inventory)
        else:
            inventory.manifest_path = result["manifest_path"]
            inventory.manifest_digest = result["manifest_digest"]
            inventory.file_count = result["file_count"]
            inventory.total_size = result["total_size"]
            inventory.manifest_state = ModelPreheatInventoryManifestStateEnum.VALID
            inventory.last_verified_at = now
            session.add(inventory)
        await session.exec(
            update(ModelPreheatS3Profile)
            .where(
                ModelPreheatS3Profile.id == parent.s3_profile_id,
                ModelPreheatS3Profile.ever_used_at.is_(None),
            )
            .values(ever_used_at=now)
        )
    elif parent.artifact_id != artifact_id:
        _conflict("artifact_binding_conflict")
    if worker_task.role == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE:
        _validate_distributed_model_paths(result)
        model_file_event = await _upsert_preheat_model_file(
            session, parent, worker_task, result, now
        )
    elif _seed_result_has_local_artifact(parent, result):
        _validate_distributed_model_paths(result)
        model_file_event = await _upsert_preheat_model_file(
            session, parent, worker_task, result, now
        )
    else:
        model_file_event = None
    await session.flush()
    return model_file_event


async def _bind_distribution_artifact(session, worker_task, result, now):
    _validate_distributed_model_paths(result)
    policy, source = await _active_distribution_source(
        session,
        worker_task.distribution_policy_id,
        worker_task.distribution_artifact_id,
    )
    artifact = source.artifact
    if (
        worker_task.distribution_request_digest
        and result["request_digest"] != worker_task.distribution_request_digest
    ):
        _conflict("request_digest_mismatch")
    if result["artifact_id"] != artifact.artifact_id:
        _conflict("artifact_binding_conflict")
    if result["resolved_revision"] != artifact.resolved_revision:
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    if not result["manifest_path"].endswith(f"/{artifact.artifact_id}/manifest.json"):
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    parent = SimpleNamespace(
        source=artifact.source,
        model_id=artifact.model_id,
        include_patterns=list(artifact.include_patterns),
        requested_revision=(policy.request_identity or {}).get("requested_revision"),
        resolved_revision=artifact.resolved_revision,
        s3_profile_id=policy.profile_id,
    )
    model_file_event = await _upsert_preheat_model_file(
        session, parent, worker_task, result, now
    )
    await session.flush()
    return model_file_event


def _seed_result_has_local_artifact(parent, result):
    return (
        parent.delivery_mode != ModelPreheatDeliveryModeEnum.S3_ONLY
        and result.get("local_dir") is not None
        and result.get("resolved_paths") is not None
    )


async def _upsert_preheat_model_file(session, parent, worker_task, result, now):
    source = _model_file_source(parent.source)
    model_source = _model_file_source_fields(
        source,
        parent.model_id,
        parent.include_patterns,
        result.get("resolved_paths") or [],
    )
    source_index = _distributed_model_source_index(
        worker_task.worker_id,
        parent.s3_profile_id,
        result["artifact_id"],
    )
    existing = await _existing_preheat_model_file(
        session, source_index, worker_task.worker_id, source, model_source
    )
    worker = (
        await session.get(Worker, worker_task.worker_id)
        if worker_task.worker_id is not None
        else None
    )
    values = {
        "source": source,
        **model_source,
        "local_dir": result["local_dir"],
        "worker_id": worker_task.worker_id,
        "worker_uuid_snapshot": worker_task.worker_uuid,
        "worker_name_snapshot": worker.name if worker is not None else None,
        "cleanup_on_delete": False,
        "size": result["total_size"],
        "download_progress": 100,
        "resolved_paths": result.get("resolved_paths") or [],
        "state": ModelFileStateEnum.READY,
        "state_message": None,
        "requested_revision": parent.requested_revision,
        "resolved_revision": result["resolved_revision"],
    }
    if existing is None:
        model_file = ModelFile(source_index=source_index, **values)
        session.add(model_file)
        return EventType.CREATED, model_file
    for field, value in values.items():
        setattr(existing, field, value)
    existing.source_index = source_index
    session.add(existing)
    return EventType.UPDATED, existing


async def _existing_preheat_model_file(
    session, source_index, worker_id, source, model_source
):
    existing = (
        await session.exec(
            select(ModelFile).where(ModelFile.source_index == source_index)
        )
    ).first()
    if existing is not None or worker_id is None:
        return existing
    conditions = [
        ModelFile.worker_id == worker_id,
        ModelFile.source == source,
        ModelFile.state == ModelFileStateEnum.ERROR,
        ModelFile.local_dir.is_(None),
        ModelFile.resolved_revision.is_(None),
    ]
    for field, value in model_source.items():
        conditions.append(getattr(ModelFile, field) == value)
    candidates = (
        await session.exec(
            select(ModelFile).where(*conditions).order_by(ModelFile.id.desc())
        )
    ).all()
    return candidates[0] if len(candidates) == 1 else None


def _model_file_source(source):
    if source == "modelscope":
        return SourceEnum.MODEL_SCOPE
    if source == "huggingface":
        return SourceEnum.HUGGING_FACE
    if source == "ollama_library":
        return SourceEnum.OLLAMA_LIBRARY
    raise HTTPException(422, "Invalid", "invalid_preheat_result")


def _model_file_source_fields(source, model_id, include_patterns, resolved_paths):
    first_pattern = include_patterns[0] if include_patterns else None
    if source == SourceEnum.MODEL_SCOPE:
        return {
            "model_scope_model_id": model_id,
            "model_scope_file_path": first_pattern if first_pattern else None,
        }
    if source == SourceEnum.HUGGING_FACE:
        return {
            "huggingface_repo_id": model_id,
            "huggingface_filename": first_pattern if first_pattern else None,
        }
    if source == SourceEnum.OLLAMA_LIBRARY:
        return {"ollama_library_model_name": model_id}
    raise HTTPException(422, "Invalid", "invalid_preheat_result")


def _validate_distributed_model_paths(result):
    local_dir = result.get("local_dir")
    resolved_paths = result.get("resolved_paths")
    if not _is_valid_local_dir(local_dir) or not resolved_paths:
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    path_module = _path_module_for_paths([local_dir, *resolved_paths])
    normalized_local_dir = path_module.normpath(local_dir)
    for path in resolved_paths:
        normalized_path = path_module.normpath(path)
        try:
            if (
                path_module.commonpath([normalized_local_dir, normalized_path])
                != normalized_local_dir
            ):
                raise HTTPException(422, "Invalid", "invalid_preheat_result")
        except ValueError:
            raise HTTPException(422, "Invalid", "invalid_preheat_result") from None


def _distributed_model_local_dir(resolved_paths):
    if not resolved_paths:
        return None
    if len(resolved_paths) == 1:
        return resolved_paths[0]
    path_module = _path_module_for_paths(resolved_paths)
    try:
        common = path_module.commonpath(
            [path_module.normpath(path) for path in resolved_paths]
        )
    except ValueError:
        return path_module.dirname(resolved_paths[0])
    if common in resolved_paths:
        return common
    return common if common else path_module.dirname(resolved_paths[0])


def _path_module_for_paths(paths):
    return (
        ntpath
        if any("\\" in path or _has_windows_drive(path) for path in paths)
        else posixpath
    )


def _has_windows_drive(path):
    return len(path) >= 2 and path[1] == ":"


def _distributed_model_source_index(worker_id, profile_id, artifact_id):
    seed = json.dumps(
        {
            "worker_id": worker_id,
            "profile_id": profile_id,
            "artifact_id": artifact_id,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(f"distributed-artifact:{seed}".encode("utf-8")).hexdigest()


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
    return and_(
        latest_id == worker_id,
        exists(
            select(Worker.id).where(
                Worker.id == worker_id,
                Worker.worker_uuid == worker_uuid,
                Worker.model_storage_protocol_version == MODEL_STORAGE_PROTOCOL_VERSION,
            )
        ),
    )


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
            "request_digest",
            "artifact_id",
            "manifest_digest",
            "manifest_path",
            "file_count",
            "local_cache_state",
            "transfer_source",
            "uploaded",
            "skipped",
            "downloaded",
            "total_size",
            "resolved_revision",
        }
        if not required <= set(value) or value.get("error_code") is not None:
            raise HTTPException(422, "Invalid", "invalid_preheat_result")
    else:
        ready_only = {
            "request_digest",
            "artifact_id",
            "manifest_digest",
            "manifest_path",
            "file_count",
            "transfer_source",
            "uploaded",
            "skipped",
            "downloaded",
            "total_size",
            "resolved_revision",
        }
        if value.get("error_code") is None or ready_only & set(value):
            raise HTTPException(422, "Invalid", "invalid_preheat_result")
    for field in ("request_digest", "artifact_id", "manifest_digest"):
        if field in value and not _is_sha256(value[field]):
            raise HTTPException(422, "Invalid", "invalid_preheat_result")
    for field in ("manifest_path",):
        if field in value and not _is_canonical_object_path(value[field]):
            raise HTTPException(422, "Invalid", "invalid_preheat_result")
    if "local_dir" in value and not _is_valid_local_dir(value["local_dir"]):
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    if value.get("transfer_source") not in {
        "current_node",
        "s3",
        "modelscope",
        "huggingface",
        "ollama_library",
        None,
    }:
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    for field in ("uploaded", "skipped", "downloaded", "file_count", "total_size"):
        if field in value and (not isinstance(value[field], int) or value[field] < 0):
            raise HTTPException(422, "Invalid", "invalid_preheat_result")
    resolved_paths = None
    if "resolved_paths" in value:
        resolved_paths = _validated_resolved_paths(value["resolved_paths"])
    cursor = _validated_cursor_value(value.get("cursor"))
    if state == "ready":
        sanitized = {
            field: value[field]
            for field in (
                "state",
                "request_digest",
                "artifact_id",
                "manifest_digest",
                "manifest_path",
                "file_count",
                "local_cache_state",
                "transfer_source",
                "uploaded",
                "skipped",
                "downloaded",
                "total_size",
                "resolved_revision",
                "local_dir",
            )
            if field in value
        }
        if resolved_paths is not None:
            sanitized["resolved_paths"] = resolved_paths
        return sanitized
    sanitized = {
        "state": value["state"],
        "error_code": value["error_code"],
        "local_cache_state": value.get("local_cache_state", "error"),
    }
    error_type = value.get("error_type")
    if error_type is not None:
        if (
            not isinstance(error_type, str)
            or not error_type
            or len(error_type) > MAX_PREHEAT_ERROR_TYPE_LENGTH
            or not error_type.replace("_", "").replace(".", "").isalnum()
        ):
            raise HTTPException(422, "Invalid", "invalid_preheat_result")
        sanitized["error_type"] = error_type
    error_message = _validated_failure_message(value.get("error_message"))
    if error_message is not None:
        sanitized["error_message"] = error_message
    if cursor:
        sanitized["cursor"] = cursor
    return sanitized


def _failure_state_message(failure, result_payload):
    if isinstance(result_payload, dict):
        result_message = _validated_failure_message(result_payload.get("error_message"))
        if result_message is not None:
            return result_message
    return _validated_failure_message(failure.state_message) or failure.error_code


def _validated_failure_message(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    value = "".join(
        " " if ord(char) < 32 or ord(char) == 127 else char for char in value
    )
    value = value.strip()
    if not value:
        return None
    if len(value) > MAX_STATE_MESSAGE_LENGTH:
        value = value[:MAX_STATE_MESSAGE_LENGTH]
    lowered = value.lower()
    sensitive_markers = (
        "access_key",
        "secret_key",
        "token=",
        "password=",
        "authorization",
    )
    if any(marker in lowered for marker in sensitive_markers):
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    return value


def _validated_cursor(role, value):
    if value in (None, {}):
        return None
    if role not in {
        ModelPreheatWorkerTaskRoleEnum.SEED,
        ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
    }:
        raise HTTPException(422, "Invalid", "resumable_cursor_not_supported")
    return _validated_cursor_value(value)


def _validated_resolved_paths(value):
    if (
        not isinstance(value, list)
        or not value
        or len(value) > MAX_PREHEAT_RESOLVED_PATHS
        or not all(
            isinstance(path, str)
            and path
            and _is_abs_path(path)
            and len(path) <= MAX_PREHEAT_RESOLVED_PATH_LENGTH
            and not any(ord(char) < 32 or ord(char) == 127 for char in path)
            for path in value
        )
    ):
        raise HTTPException(422, "Invalid", "invalid_preheat_result")
    return value


def _is_valid_local_dir(value):
    return (
        isinstance(value, str)
        and value
        and _is_abs_path(value)
        and len(value) <= MAX_PREHEAT_RESOLVED_PATH_LENGTH
        and not any(ord(char) < 32 or ord(char) == 127 for char in value)
    )


def _is_abs_path(path):
    return _path_module_for_paths([path]).isabs(path)


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
    validated = {}
    if files:
        validated["completed_files"] = files
    if "staging_exists" in value:
        validated["staging_exists"] = value["staging_exists"]
    return validated


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
