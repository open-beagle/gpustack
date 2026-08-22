from fastapi import APIRouter, Header, Request
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

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
    ModelPreheatS3ProfilePublic,
    ModelPreheatS3ProfilesPublic,
    ModelPreheatS3ProfileUpdate,
)
from gpustack.schemas.model_file_download_executions import (
    ModelFileDownloadExecutionProfilePin,
)
from gpustack.schemas.model_preheat_schedules import ModelPreheatSchedule
from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatConnectivityCheckPublic,
    ModelPreheatConnectivityWorkerPublic,
    ModelPreheatS3ConnectivityCheck,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
)
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
    existing = await ModelPreheatS3Profile.one_by_field(
        session, "name", profile_in.name
    )
    if existing:
        raise AlreadyExistsException(
            message=f"Model preheat S3 profile {profile_in.name} already exists"
        )

    cipher = _cipher_from_request(request)
    try:
        profile = ModelPreheatS3Profile(
            **profile_in.model_dump(exclude={"access_key", "secret_key"}),
            access_key_encrypted=cipher.encrypt(profile_in.access_key),
            secret_key_encrypted=cipher.encrypt(profile_in.secret_key),
            encryption_key_version=cipher.current_key_version,
        )
        # 显式使用 default_slot 占用默认槽位；Public API 的 is_default 由槽位派生。
        if profile.default_slot == DEFAULT_SLOT_GLOBAL:
            await _unset_other_defaults(session)
        profile = await ModelPreheatS3Profile.create(session, profile)
    except CredentialEncryptionUnavailable as exc:
        raise ServiceUnavailableException(
            message=f"credential_encryption_unavailable: {exc}"
        )
    except IntegrityError:
        raise AlreadyExistsException(
            message=f"Model preheat S3 profile {profile_in.name} already exists"
        )
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
    # 系统引导 Profile（worker-local-s3）由 Server 管理，UI 可查看/检测/选择，
    # 仅允许重新选择默认槽位（default_slot），连接/凭据及其他字段一律禁止
    # 直接编辑：连接/凭据变化只能通过启动参数在重启时更新。
    cipher = _cipher_from_request(request)
    update_data = profile_in.model_dump(exclude_unset=True)
    if profile.system_managed and set(update_data) != {"default_slot"}:
        raise HTTPException(
            403,
            "system_profile_read_only",
            "system_profile_read_only",
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
        if connection_config_changed:
            await _update_connection_config_with_cas(session, profile, update_data)
        else:
            await profile.update(session, update_data)
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
        raise HTTPException(
            409,
            "default_slot_conflict",
            "default_slot_conflict",
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
    policy = (
        await session.exec(
            select(ModelPreheatDistributionPolicy).where(
                ModelPreheatDistributionPolicy.profile_id == profile.id
            )
        )
    ).first()
    if policy is not None:
        raise HTTPException(409, "Conflict", "distribution_policy_uses_profile")
    # 系统引导 Profile 不允许删除（避免破坏启动引导与任务/策略外键）。
    if profile.system_managed:
        raise HTTPException(
            409,
            "system_profile_not_deletable",
            "system_profile_not_deletable",
        )
    # 当前默认 Profile 不允许直接删除：必须先把另一条 Profile 设为默认，
    # 避免删除瞬间改变普通下载行为。
    if profile.default_slot == DEFAULT_SLOT_GLOBAL:
        raise HTTPException(
            409,
            "default_profile_not_deletable",
            "default_profile_not_deletable",
        )
    download_execution = (
        await session.exec(
            select(ModelFileDownloadExecutionProfilePin.execution_id).where(
                ModelFileDownloadExecutionProfilePin.profile_id == profile.id
            )
        )
    ).first()
    if download_execution is not None:
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
    try:
        await profile.delete(session)
    except Exception as exc:
        raise InternalServerErrorException(
            message=f"Failed to delete model preheat S3 profile: {type(exc).__name__}"
        )


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


async def _update_connection_config_with_cas(session, profile, update_data):
    expected_config_version = profile.config_version
    values = dict(update_data)
    values["config_version"] = ModelPreheatS3Profile.config_version + 1
    result = await session.exec(
        update(ModelPreheatS3Profile)
        .where(
            ModelPreheatS3Profile.id == profile.id,
            ModelPreheatS3Profile.config_version == expected_config_version,
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
