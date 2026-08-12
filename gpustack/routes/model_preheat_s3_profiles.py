import base64
import hashlib
import hmac
import json
from typing import Literal

from fastapi import APIRouter, Header, Query, Request
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
    ModelPreheatS3ConnectivityStateEnum,
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileCreate,
    ModelPreheatS3ProfilePublic,
    ModelPreheatS3ProfilesPublic,
    ModelPreheatS3ProfileUpdate,
)
from gpustack.schemas.model_preheat_schedules import ModelPreheatSchedule
from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatCachedModel,
    ModelPreheatCachedModelPublic,
    ModelPreheatCachedModelsPage,
    ModelPreheatConnectivityCheckPublic,
    ModelPreheatConnectivityWorkerPublic,
    ModelPreheatInventoryJob,
    ModelPreheatInventoryJobPublic,
    ModelPreheatInventoryManifestStateEnum,
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
from gpustack.server.model_preheat_s3_inventory import ModelPreheatS3Inventory

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
MAX_INVENTORY_CURSOR_BYTES = 2048


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


@router.get("/{id}/cached-models", response_model=ModelPreheatCachedModelsPage)
async def get_cached_models(
    request: Request,
    session: SessionDep,
    id: int,
    limit: int = Query(default=50, ge=1, le=100),
    manifest_state: ModelPreheatInventoryManifestStateEnum | None = None,
    source: str | None = Query(default=None, max_length=32),
    cursor: str | None = Query(default=None, max_length=MAX_INVENTORY_CURSOR_BYTES),
):
    await _get_profile(session, id)
    filters = {
        "manifest_state": manifest_state.value if manifest_state else None,
        "source": source,
    }
    last_cache_key = None
    if cursor is not None:
        payload = _decode_inventory_cursor(request, cursor)
        if (
            payload.get("v") != 1
            or payload.get("profile_id") != id
            or payload.get("limit") != limit
            or payload.get("filters") != filters
            or not isinstance(payload.get("last_cache_key"), str)
            or len(payload["last_cache_key"]) > 256
        ):
            raise HTTPException(422, "Invalid", "invalid_inventory_cursor")
        last_cache_key = payload["last_cache_key"]

    statement = select(ModelPreheatCachedModel).where(
        ModelPreheatCachedModel.profile_id == id
    )
    if manifest_state is not None:
        statement = statement.where(
            ModelPreheatCachedModel.manifest_state == manifest_state
        )
    if source is not None:
        statement = statement.where(ModelPreheatCachedModel.source == source)
    if last_cache_key is not None:
        statement = statement.where(ModelPreheatCachedModel.cache_key > last_cache_key)
    rows = (
        await session.exec(
            statement.order_by(ModelPreheatCachedModel.cache_key.asc()).limit(limit + 1)
        )
    ).all()
    has_more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = None
    if has_more and rows:
        next_cursor = _encode_inventory_cursor(
            request,
            {
                "v": 1,
                "profile_id": id,
                "limit": limit,
                "filters": filters,
                "last_cache_key": rows[-1].cache_key,
            },
        )
    return ModelPreheatCachedModelsPage(
        items=[ModelPreheatCachedModelPublic.model_validate(row) for row in rows],
        next_cursor=next_cursor,
    )


@router.post(
    "/{id}/inventory-jobs",
    response_model=ModelPreheatInventoryJobPublic,
    status_code=202,
)
async def create_inventory_job(
    request: Request,
    session: SessionDep,
    id: int,
    kind: Literal["refresh", "gc"] = "refresh",
):
    profile = await _get_profile(session, id)
    service = getattr(request.app.state, "model_preheat_s3_inventory", None)
    if service is None:
        service = ModelPreheatS3Inventory(
            session.bind, config=request.app.state.server_config
        )
    if kind == "gc":
        job = await service.create_gc_job(session, profile.id, profile.config_version)
    else:
        job = await service.create_refresh_job(
            session, profile.id, profile.config_version
        )
    return ModelPreheatInventoryJobPublic.model_validate(job)


@router.get(
    "/{id}/inventory-jobs/{job_id}", response_model=ModelPreheatInventoryJobPublic
)
async def get_inventory_job(session: SessionDep, id: int, job_id: int):
    await _get_profile(session, id)
    job = await session.get(ModelPreheatInventoryJob, job_id)
    if job is None or job.profile_id != id:
        raise NotFoundException(message="model_preheat_inventory_job_not_found")
    return ModelPreheatInventoryJobPublic.model_validate(job)


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
        if profile.is_default:
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
    cipher = _cipher_from_request(request)
    update_data = profile_in.model_dump(exclude_unset=True)

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
        if update_data.get("is_default") is True:
            await _unset_other_defaults(session, profile.id)
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
        raise AlreadyExistsException(message="Model preheat S3 profile already exists")
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
        ModelPreheatS3Profile.is_default == True  # noqa: E712
    )
    if profile_id is not None:
        statement = statement.where(ModelPreheatS3Profile.id != profile_id)
    profiles = (await session.exec(statement)).all()
    for profile in profiles:
        profile.is_default = False
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
        is_default=profile.is_default,
        config_version=profile.config_version,
        connectivity_state=profile.connectivity_state,
        last_connectivity_check_id=profile.last_connectivity_check_id,
        last_connectivity_checked_at=profile.last_connectivity_checked_at,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _inventory_cursor_key(request: Request) -> bytes:
    config = request.app.state.server_config
    key = getattr(config, "model_preheat_inventory_cursor_key", None) or getattr(
        config, "jwt_secret_key", None
    )
    if not isinstance(key, str) or len(key) < 16:
        raise ServiceUnavailableException(message="inventory_cursor_key_unavailable")
    return key.encode("utf-8")


def _encode_inventory_cursor(request: Request, payload: dict) -> str:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = hmac.new(_inventory_cursor_key(request), body, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(body + signature).rstrip(b"=").decode("ascii")


def _decode_inventory_cursor(request: Request, cursor: str) -> dict:
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(cursor + padding)
        if base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != cursor:
            raise ValueError
        if len(raw) <= 32 or len(raw) > MAX_INVENTORY_CURSOR_BYTES:
            raise ValueError
        body, signature = raw[:-32], raw[-32:]
        expected = hmac.new(
            _inventory_cursor_key(request), body, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError
        return payload
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(422, "Invalid", "invalid_inventory_cursor") from None


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
    ModelPreheatInventoryJob,
    ModelPreheatInventoryJobPublic,
    ModelPreheatInventoryManifestStateEnum,
