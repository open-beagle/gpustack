from fastapi import APIRouter, Header, Request
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlmodel import delete, select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    HTTPException,
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.model_preheat_credentials import (
    CredentialEncryptionUnavailable,
    ModelPreheatCredentialCipher,
    ModelPreheatCredentialError,
)
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.model_preheat_s3_profiles import (
    DEFAULT_SLOT_GLOBAL,
    ModelPreheatS3ConnectivityStateEnum,
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileCreate,
    ModelPreheatS3ProfileLifecycleStateEnum,
    ModelPreheatS3ProfilePublic,
    ModelPreheatS3ProfilesPublic,
    ModelPreheatS3ProfileUpdate,
    model_preheat_s3_storage_key,
)
from gpustack.schemas.model_file_download_executions import (
    ModelFileDownloadExecution,
    ModelFileDownloadExecutionProfilePin,
    ModelFileDownloadExecutionStateEnum,
)
from gpustack.schemas.model_preheat_schedules import ModelPreheatSchedule
from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatConnectivityCheckPublic,
    ModelPreheatConnectivityWorkerPublic,
    ModelPreheatArtifact,
    ModelPreheatS3ConnectivityCheck,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
)
from gpustack.schemas.model_storage_sync import ModelStorageSyncTask
from gpustack.schemas.workers import Worker
from gpustack.server.deps import CurrentAdminUserDep, ListParamsDep, SessionDep
from gpustack.server.model_preheat_connectivity import (
    ConnectivityCheckIdempotencyConflict,
    aggregate_connectivity_check,
    connectivity_ttl_from_config,
    create_or_reuse_connectivity_check,
    mark_profile_stale_if_expired,
)
from gpustack.server.model_preheat_idempotency import (
    canonical_request_hash,
    get_idempotency_record,
    new_idempotency_record,
)
from gpustack.server.model_storage_bootstrap import parse_local_s3_target

router = APIRouter()
CONNECTIVITY_CHECK_OPERATION = "model_preheat_s3_profile.connectivity_check"
CONNECTION_CONFIG_FIELDS = {
    "endpoint",
    "bucket",
    "prefix",
    "access_key",
    "secret_key",
    "tls_enabled",
    "tls_verify",
    "region",
    "use_virtual_hosted_style",
}
SYSTEM_MANAGED_EDITABLE_FIELDS = {
    "default_slot",
    "tls_enabled",
    "tls_verify",
    "use_virtual_hosted_style",
    "source_fallback_enabled",
    "lifecycle_state",
}


class ProfileConfigConflict(Exception):
    pass


@router.get("", response_model=ModelPreheatS3ProfilesPublic)
async def get_profiles(request: Request, session: SessionDep, params: ListParamsDep):
    statement = (
        select(ModelPreheatS3Profile)
        .order_by(ModelPreheatS3Profile.created_at.desc())
        .offset((params.page - 1) * params.perPage)
        .limit(params.perPage)
    )
    profiles = (await session.exec(statement)).all()
    total = len(await ModelPreheatS3Profile.all(session))
    total_page = (total + params.perPage - 1) // params.perPage
    await _persist_expired_profiles_as_stale(
        session, profiles, request.app.state.server_config
    )
    return PaginatedList[ModelPreheatS3ProfilePublic](
        items=[_to_public(profile) for profile in profiles],
        pagination=Pagination(
            page=params.page,
            perPage=params.perPage,
            total=total,
            totalPage=total_page,
        ),
    )


@router.get("/{id}", response_model=ModelPreheatS3ProfilePublic)
async def get_profile(request: Request, session: SessionDep, id: int):
    profile = await _get_profile(session, id)
    await _persist_expired_profiles_as_stale(
        session, [profile], request.app.state.server_config
    )
    return _to_public(profile)


@router.post("", response_model=ModelPreheatS3ProfilePublic)
async def create_profile(
    request: Request,
    session: SessionDep,
    profile_in: ModelPreheatS3ProfileCreate,
):
    if profile_in.prefix:
        raise HTTPException(422, "Invalid", "manual_profile_prefix_forbidden")
    existing = await ModelPreheatS3Profile.one_by_field(
        session, "name", profile_in.name
    )
    if existing:
        raise AlreadyExistsException(
            message=f"Model preheat S3 profile {profile_in.name} already exists"
        )
    same_storage = (
        await session.exec(
            select(ModelPreheatS3Profile.id).where(
                ModelPreheatS3Profile.active_storage_key
                == model_preheat_s3_storage_key(profile_in.endpoint, profile_in.bucket)
            )
        )
    ).first()
    if same_storage is not None:
        raise HTTPException(409, "Conflict", "profile_storage_conflict")

    cipher = _cipher_from_request(request)
    try:
        profile = ModelPreheatS3Profile(
            **profile_in.model_dump(exclude={"access_key", "secret_key"}),
            access_key_encrypted=cipher.encrypt(profile_in.access_key),
            secret_key_encrypted=cipher.encrypt(profile_in.secret_key),
            encryption_key_version=cipher.current_key_version,
            active_storage_key=model_preheat_s3_storage_key(
                profile_in.endpoint, profile_in.bucket
            ),
        )
        # 显式使用 default_slot 占用默认槽位；Public API 的 is_default 由槽位派生。
        if profile.default_slot == DEFAULT_SLOT_GLOBAL:
            await _unset_other_defaults(session)
        profile = await ModelPreheatS3Profile.create(session, profile)
    except CredentialEncryptionUnavailable as exc:
        raise ServiceUnavailableException(
            message=f"credential_encryption_unavailable: {exc}"
        )
    except HTTPException:
        raise
    except IntegrityError:
        raise HTTPException(409, "Conflict", "profile_storage_conflict")
    except Exception as exc:
        raise InternalServerErrorException(
            message=f"Failed to create model preheat S3 profile: {type(exc).__name__}"
        )

    await create_or_reuse_connectivity_check(session, profile)
    return _to_public(profile)


@router.patch("/{id}", response_model=ModelPreheatS3ProfilePublic)
async def update_profile(
    request: Request,
    session: SessionDep,
    id: int,
    profile_in: ModelPreheatS3ProfileUpdate,
):
    profile = await _get_profile(session, id)
    # 系统引导 Profile（worker-local-s3）的定位与凭据由 Server 管理；UI 只可
    # 调整运行策略和默认选择。TLS/寻址改变后同样必须重新校验连通性。
    cipher = _cipher_from_request(request)
    update_data = profile_in.model_dump(exclude_unset=True)
    if profile.system_managed and not set(update_data).issubset(
        SYSTEM_MANAGED_EDITABLE_FIELDS
    ):
        raise HTTPException(
            403,
            "system_profile_read_only",
            "system_profile_read_only",
        )

    if not profile.system_managed and "prefix" in update_data and update_data["prefix"]:
        raise HTTPException(422, "Invalid", "manual_profile_prefix_forbidden")
    lifecycle_state = update_data.get("lifecycle_state", profile.lifecycle_state)
    if (
        lifecycle_state == ModelPreheatS3ProfileLifecycleStateEnum.MAINTENANCE
        and update_data.get("default_slot") == DEFAULT_SLOT_GLOBAL
    ):
        raise HTTPException(
            409,
            "maintenance_profile_not_defaultable",
            "maintenance_profile_not_defaultable",
        )

    if "name" in update_data:
        existing = await ModelPreheatS3Profile.one_by_field(
            session, "name", update_data["name"]
        )
        if existing and existing.id != id:
            raise AlreadyExistsException(
                message=(
                    f"Model preheat S3 profile {update_data['name']} already exists"
                )
            )

    access_key = update_data.pop("access_key", None)
    secret_key = update_data.pop("secret_key", None)
    credential_changed = access_key is not None or secret_key is not None
    connection_config_changed = credential_changed or any(
        field in update_data and getattr(profile, field) != update_data[field]
        for field in CONNECTION_CONFIG_FIELDS
        if field not in {"access_key", "secret_key"}
    )

    endpoint = update_data.get("endpoint", profile.endpoint)
    bucket = update_data.get("bucket", profile.bucket)
    if lifecycle_state == ModelPreheatS3ProfileLifecycleStateEnum.MAINTENANCE:
        update_data["default_slot"] = None
        update_data["active_storage_key"] = None
    elif lifecycle_state == ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE:
        update_data["active_storage_key"] = model_preheat_s3_storage_key(
            endpoint, bucket
        )

    try:
        _ensure_current_key_configured(cipher)
        if access_key is not None:
            update_data["access_key_encrypted"] = cipher.encrypt(access_key)
        else:
            rotated = _rotate_if_needed(cipher, profile.access_key_encrypted)
            if rotated is not None:
                update_data["access_key_encrypted"] = rotated
        if secret_key is not None:
            update_data["secret_key_encrypted"] = cipher.encrypt(secret_key)
        else:
            rotated = _rotate_if_needed(cipher, profile.secret_key_encrypted)
            if rotated is not None:
                update_data["secret_key_encrypted"] = rotated

        if (
            credential_changed
            or "access_key_encrypted" in update_data
            or "secret_key_encrypted" in update_data
        ):
            update_data["encryption_key_version"] = cipher.current_key_version
        if connection_config_changed:
            update_data["connectivity_state"] = (
                ModelPreheatS3ConnectivityStateEnum.PENDING
            )
            update_data["last_connectivity_check_id"] = None
            update_data["last_connectivity_checked_at"] = None
        # 显式使用 default_slot：请求 default_slot="global" 时在同一事务转移槽位，
        # 先清除其他默认 Profile 的槽位并 flush（否则 SQLite 唯一约束检查时
        # 仍读到未落库的 'global' 旧值而误报冲突），再为本 Profile 占位；
        # 仍依赖唯一约束保证跨数据库最多一个默认，且整段处于同一事务，
        # 后续 CAS 失败可整体回滚（包括已清除的其他默认槽位）。
        if update_data.get("default_slot") == DEFAULT_SLOT_GLOBAL:
            await _unset_other_defaults(session, profile.id)
            await session.flush()
        await _update_profile_with_cas(
            session,
            profile,
            update_data,
            increment_config_version=connection_config_changed,
        )
    except CredentialEncryptionUnavailable as exc:
        raise ServiceUnavailableException(
            message=f"credential_encryption_unavailable: {exc}"
        )
    except ModelPreheatCredentialError as exc:
        raise ServiceUnavailableException(
            message=f"credential_encryption_unavailable: {exc}"
        )
    except IntegrityError:
        # 唯一约束冲突：可能是重名，也可能是并发默认槽位抢占（default_slot 唯一）。
        # 后者返回稳定错误允许用户重试。
        await session.rollback()
        raise HTTPException(
            409,
            "profile_storage_or_default_conflict",
            "profile_storage_or_default_conflict",
        )
    except ProfileConfigConflict:
        raise HTTPException(409, "profile_config_conflict", "profile_config_conflict")
    except Exception as exc:
        raise InternalServerErrorException(
            message=f"Failed to update model preheat S3 profile: {type(exc).__name__}"
        )

    if connection_config_changed:
        await create_or_reuse_connectivity_check(session, profile)
    return _to_public(profile)


@router.post(
    "/{id}/connectivity-checks", response_model=ModelPreheatConnectivityCheckPublic
)
async def create_connectivity_check(
    request: Request,
    session: SessionDep,
    current_user: CurrentAdminUserDep,
    id: int,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    profile = await _get_profile(session, id)
    request_hash = canonical_request_hash(
        {"profile_id": profile.id, "profile_config_version": profile.config_version}
    )
    operation = CONNECTIVITY_CHECK_OPERATION
    check_idempotency_scope = (
        canonical_request_hash(
            {
                "user_id": current_user.id,
                "operation": operation,
                "idempotency_key": idempotency_key,
            }
        )
        if idempotency_key
        else None
    )
    record = await get_idempotency_record(
        session, current_user.id, operation, idempotency_key
    )
    if record is not None:
        if record.request_hash != request_hash:
            raise HTTPException(409, "idempotency_key_reused", "idempotency_key_reused")
        check = await ModelPreheatS3ConnectivityCheck.one_by_id(
            session, record.resource_id
        )
        if check is None:
            raise InternalServerErrorException(message="idempotency_resource_not_found")
        return await _connectivity_check_public(
            session, check, request.app.state.server_config
        )

    try:
        check = await create_or_reuse_connectivity_check(
            session,
            profile,
            idempotency_scope_key=check_idempotency_scope,
            request_hash=request_hash,
        )
    except ConnectivityCheckIdempotencyConflict:
        raise HTTPException(409, "idempotency_key_reused", "idempotency_key_reused")
    if check is None:
        raise InvalidException(message="no_online_workers")
    record = new_idempotency_record(
        current_user.id,
        operation,
        idempotency_key,
        request_hash,
        check.id,
    )
    check_id = check.id
    if record is not None:
        session.add(record)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            existing = await get_idempotency_record(
                session, current_user.id, operation, idempotency_key
            )
            if existing is None:
                raise
            if existing.request_hash != request_hash:
                raise HTTPException(
                    409, "idempotency_key_reused", "idempotency_key_reused"
                )
            check_id = existing.resource_id
    check = await ModelPreheatS3ConnectivityCheck.one_by_id(session, check_id)
    return await _connectivity_check_public(
        session, check, request.app.state.server_config
    )


@router.get(
    "/{id}/connectivity-checks/{check_id}",
    response_model=ModelPreheatConnectivityCheckPublic,
)
async def get_connectivity_check(
    request: Request, session: SessionDep, id: int, check_id: int
):
    await _get_profile(session, id)
    check = await ModelPreheatS3ConnectivityCheck.one_by_id(session, check_id)
    if check is None or check.profile_id != id:
        raise NotFoundException(message="model_preheat_connectivity_check_not_found")
    return await _connectivity_check_public(
        session, check, request.app.state.server_config
    )


@router.delete("/{id}")
async def delete_profile(request: Request, session: SessionDep, id: int):
    cipher = _cipher_from_request(request)
    try:
        _ensure_current_key_configured(cipher)
    except CredentialEncryptionUnavailable as exc:
        raise ServiceUnavailableException(
            message=f"credential_encryption_unavailable: {exc}"
        )
    profile = await _get_profile(session, id)
    if profile.system_managed and parse_local_s3_target(
        request.app.state.server_config
    ):
        raise HTTPException(
            409, "Conflict", "system_profile_declared_by_startup_config"
        )
    if profile.ever_used_at is not None:
        raise HTTPException(409, "Conflict", "profile_has_been_used")
    artifact = (
        await session.exec(
            select(ModelPreheatArtifact.id).where(
                ModelPreheatArtifact.profile_id == profile.id
            )
        )
    ).first()
    if artifact is not None:
        raise HTTPException(409, "Conflict", "model_preheat_artifact_uses_profile")
    policy = (
        await session.exec(
            select(ModelPreheatDistributionPolicy).where(
                ModelPreheatDistributionPolicy.profile_id == profile.id
            )
        )
    ).first()
    if policy is not None:
        raise HTTPException(409, "Conflict", "distribution_policy_uses_profile")
    download_executions = (
        await session.exec(
            select(ModelFileDownloadExecution)
            .join(
                ModelFileDownloadExecutionProfilePin,
                ModelFileDownloadExecutionProfilePin.execution_id
                == ModelFileDownloadExecution.id,
            )
            .where(ModelFileDownloadExecutionProfilePin.profile_id == profile.id)
        )
    ).all()
    removable_execution_ids = []
    for execution in download_executions:
        if (
            execution.state == ModelFileDownloadExecutionStateEnum.PENDING
            and execution.claimed_by_worker_uuid is None
        ):
            removable_execution_ids.append(execution.id)
            continue
        raise HTTPException(
            409,
            "model_file_download_execution_uses_profile",
            "model_file_download_execution_uses_profile",
        )
    schedule = (
        await session.exec(
            select(ModelPreheatSchedule.id).where(
                ModelPreheatSchedule.s3_profile_id == profile.id
            )
        )
    ).first()
    if schedule is not None:
        raise HTTPException(
            409,
            "model_preheat_schedule_uses_profile",
            "model_preheat_schedule_uses_profile",
        )
    task = (
        await session.exec(
            select(ModelPreheatTask.id).where(
                ModelPreheatTask.s3_profile_id == profile.id
            )
        )
    ).first()
    if task is not None:
        raise HTTPException(
            409,
            "model_preheat_task_uses_profile",
            "model_preheat_task_uses_profile",
        )
    sync_task = (
        await session.exec(
            select(ModelStorageSyncTask.id).where(
                ModelStorageSyncTask.profile_id == profile.id
            )
        )
    ).first()
    if sync_task is not None:
        raise HTTPException(
            409,
            "model_storage_sync_task_uses_profile",
            "model_storage_sync_task_uses_profile",
        )
    try:
        if removable_execution_ids:
            detached = await _detach_unclaimed_download_executions(
                session, profile.id, removable_execution_ids
            )
            if not detached:
                raise HTTPException(
                    409,
                    "model_file_download_execution_uses_profile",
                    "model_file_download_execution_uses_profile",
                )
            await session.exec(
                delete(ModelFileDownloadExecutionProfilePin).where(
                    ModelFileDownloadExecutionProfilePin.execution_id.in_(
                        removable_execution_ids
                    )
                )
            )
        await profile.delete(session)
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "profile_is_in_use", "profile_is_in_use")
    except HTTPException:
        raise
    except Exception as exc:
        raise InternalServerErrorException(
            message=f"Failed to delete model preheat S3 profile: {type(exc).__name__}"
        )


async def _detach_unclaimed_download_executions(
    session, profile_id: int, execution_ids: list[int]
) -> bool:
    result = await session.exec(
        update(ModelFileDownloadExecution)
        .where(
            ModelFileDownloadExecution.id.in_(execution_ids),
            ModelFileDownloadExecution.state
            == ModelFileDownloadExecutionStateEnum.PENDING,
            ModelFileDownloadExecution.claimed_by_worker_uuid.is_(None),
            ModelFileDownloadExecution.claimed_at.is_(None),
            ModelFileDownloadExecution.default_profile_id == profile_id,
        )
        .values(
            default_profile_id=None,
            default_profile_config_version=None,
            credential_snapshot_encrypted=None,
            encryption_key_version=None,
        )
        .execution_options(synchronize_session=False)
    )
    if result.rowcount == len(execution_ids):
        return True
    await session.rollback()
    return False


async def _get_profile(session, id: int) -> ModelPreheatS3Profile:
    profile = await ModelPreheatS3Profile.one_by_id(session, id)
    if not profile:
        raise NotFoundException(message="Model preheat S3 profile not found")
    return profile


async def _unset_other_defaults(session, profile_id: int | None = None):
    statement = select(ModelPreheatS3Profile).where(
        ModelPreheatS3Profile.default_slot == DEFAULT_SLOT_GLOBAL
    )
    if profile_id is not None:
        statement = statement.where(ModelPreheatS3Profile.id != profile_id)
    profiles = (await session.exec(statement)).all()
    for profile in profiles:
        profile.default_slot = None
        session.add(profile)


async def _update_profile_with_cas(
    session, profile, update_data, *, increment_config_version: bool
):
    expected_config_version = profile.config_version
    expected_lifecycle_state = profile.lifecycle_state
    expected_default_slot = profile.default_slot
    expected_active_storage_key = profile.active_storage_key
    values = dict(update_data)
    if increment_config_version:
        values["config_version"] = ModelPreheatS3Profile.config_version + 1
    result = await session.exec(
        update(ModelPreheatS3Profile)
        .where(
            ModelPreheatS3Profile.id == profile.id,
            ModelPreheatS3Profile.config_version == expected_config_version,
            ModelPreheatS3Profile.lifecycle_state == expected_lifecycle_state,
            ModelPreheatS3Profile.default_slot == expected_default_slot,
            ModelPreheatS3Profile.active_storage_key == expected_active_storage_key,
        )
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount == 0:
        await session.rollback()
        raise ProfileConfigConflict
    if result.rowcount != 1:
        await session.rollback()
        raise RuntimeError("unexpected_profile_config_update_count")
    await session.commit()
    await session.refresh(profile)


async def _persist_expired_profiles_as_stale(session, profiles, config):
    ttl = connectivity_ttl_from_config(config)
    changed = False
    for profile in profiles:
        changed = await mark_profile_stale_if_expired(session, profile, ttl) or changed
    if not changed:
        return
    await session.commit()
    for profile in profiles:
        await session.refresh(profile)


def _cipher_from_request(request: Request) -> ModelPreheatCredentialCipher:
    config = request.app.state.server_config
    return ModelPreheatCredentialCipher(
        current_key=getattr(config, "model_preheat_credential_key", None),
        current_key_version=getattr(
            config, "model_preheat_credential_key_version", None
        ),
        old_keys=getattr(config, "model_preheat_credential_old_keys", None),
    )


def _ensure_current_key_configured(cipher: ModelPreheatCredentialCipher):
    if not cipher.current_key:
        raise CredentialEncryptionUnavailable("credential_encryption_unavailable")


def _rotate_if_needed(cipher: ModelPreheatCredentialCipher, encrypted):
    _, rotated = cipher.decrypt_and_rotate(encrypted)
    return rotated


def _to_public(profile: ModelPreheatS3Profile) -> ModelPreheatS3ProfilePublic:
    return ModelPreheatS3ProfilePublic(
        id=profile.id,
        name=profile.name,
        description=profile.description,
        endpoint=profile.endpoint,
        bucket=profile.bucket,
        prefix=profile.prefix,
        credential_configured=bool(
            profile.access_key_encrypted and profile.secret_key_encrypted
        ),
        tls_enabled=profile.tls_enabled,
        tls_verify=profile.tls_verify,
        region=profile.region,
        use_virtual_hosted_style=profile.use_virtual_hosted_style,
        default_slot=profile.default_slot,
        provisioning_source=profile.provisioning_source,
        provisioning_key=profile.provisioning_key,
        system_managed=profile.system_managed,
        lifecycle_state=profile.lifecycle_state,
        ever_used_at=profile.ever_used_at,
        source_fallback_enabled=profile.source_fallback_enabled,
        config_version=profile.config_version,
        connectivity_state=profile.connectivity_state,
        last_connectivity_check_id=profile.last_connectivity_check_id,
        last_connectivity_checked_at=profile.last_connectivity_checked_at,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


async def _connectivity_check_public(session, check, config):
    check = await aggregate_connectivity_check(session, check.id)
    profile = await ModelPreheatS3Profile.one_by_id(session, check.profile_id)
    if profile is not None and await mark_profile_stale_if_expired(
        session, profile, connectivity_ttl_from_config(config)
    ):
        await session.commit()
        await session.refresh(check)
    tasks = (
        await session.exec(
            select(ModelPreheatWorkerTask).where(
                ModelPreheatWorkerTask.connectivity_check_id == check.id,
                ModelPreheatWorkerTask.role
                == ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
            )
        )
    ).all()
    worker_ids = [task.worker_id for task in tasks if task.worker_id is not None]
    workers = {}
    if worker_ids:
        workers = {
            worker.id: worker
            for worker in (
                await session.exec(select(Worker).where(Worker.id.in_(worker_ids)))
            ).all()
        }
    result_workers = []
    for task in sorted(tasks, key=lambda item: item.worker_uuid):
        result = task.resumable_cursor or {}
        worker = workers.get(task.worker_id)
        result_workers.append(
            ModelPreheatConnectivityWorkerPublic(
                worker_uuid=task.worker_uuid,
                worker_id=task.worker_id,
                worker_name=worker.name if worker else None,
                state=task.state,
                readable=bool(result.get("readable", False)),
                writable=bool(result.get("writable", False)),
                deletable=bool(result.get("deletable", False)),
                cleanup_failed=bool(result.get("cleanup_failed", False)),
                latency_ms=result.get("latency_ms"),
                error_code=result.get("error_code") or task.error_code,
                failed_stage=result.get("failed_stage"),
            )
        )
    return ModelPreheatConnectivityCheckPublic(
        id=check.id,
        profile_id=check.profile_id,
        profile_config_version=check.profile_config_version,
        state=check.state,
        summary={
            "success": check.success_count,
            "failed": check.failed_count,
            "not_checked": check.not_checked_count,
        },
        workers=result_workers,
        created_at=check.created_at,
        updated_at=check.updated_at,
        started_at=check.started_at,
        finished_at=check.finished_at,
    )
