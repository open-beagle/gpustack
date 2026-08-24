from datetime import datetime, timezone

from fastapi import APIRouter, Header, Request
from pydantic import ValidationError
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from gpustack.api.exceptions import HTTPException, InvalidException, NotFoundException
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
)
from gpustack.schemas.model_preheat_schedules import next_window_start_utc
from gpustack.schemas.model_storage_sync_policies import (
    ModelStorageSyncPoliciesPublic,
    ModelStorageSyncPolicy,
    ModelStorageSyncPolicyCreate,
    ModelStorageSyncPolicyPublic,
    ModelStorageSyncPolicyRun,
    ModelStorageSyncPolicyRunPublic,
    ModelStorageSyncPolicyRunsPublic,
    ModelStorageSyncPolicyRunStateEnum,
    ModelStorageSyncPolicyTriggerModeEnum,
    ModelStorageSyncPolicyUpdate,
)
from gpustack.server.deps import CurrentAdminUserDep, ListParamsDep, SessionDep
from gpustack.server.model_storage_sync_policy_controller import (
    SyncPolicyDisabled,
    SyncPolicyRunConflict,
)


router = APIRouter()


@router.get("", response_model=ModelStorageSyncPoliciesPublic)
async def get_sync_policies(session: SessionDep, params: ListParamsDep):
    items = (
        await session.exec(
            select(ModelStorageSyncPolicy)
            .order_by(ModelStorageSyncPolicy.created_at.desc())
            .offset((params.page - 1) * params.perPage)
            .limit(params.perPage)
        )
    ).all()
    total = await ModelStorageSyncPolicy.count(session)
    return PaginatedList[ModelStorageSyncPolicyPublic](
        items=[ModelStorageSyncPolicyPublic.model_validate(item) for item in items],
        pagination=_pagination(params, total),
    )


@router.post("", response_model=ModelStorageSyncPolicyPublic)
async def create_sync_policy(
    session: SessionDep,
    current_user: CurrentAdminUserDep,
    policy_in: ModelStorageSyncPolicyCreate,
):
    await _active_profile_or_404(session, policy_in.profile_id)
    policy = ModelStorageSyncPolicy(
        **policy_in.model_dump(), created_by_user_id=current_user.id
    )
    _set_next_run(policy, datetime.now(timezone.utc))
    session.add(policy)
    try:
        await session.commit()
        await session.refresh(policy)
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "Conflict", "model_storage_sync_policy_conflict")
    return ModelStorageSyncPolicyPublic.model_validate(policy)


@router.get("/{id}", response_model=ModelStorageSyncPolicyPublic)
async def get_sync_policy(session: SessionDep, id: int):
    return ModelStorageSyncPolicyPublic.model_validate(
        await _policy_or_404(session, id)
    )


@router.patch("/{id}", response_model=ModelStorageSyncPolicyPublic)
async def update_sync_policy(
    session: SessionDep,
    id: int,
    policy_in: ModelStorageSyncPolicyUpdate,
):
    policy = await _policy_or_404(session, id)
    update_data = policy_in.model_dump(exclude_unset=True)
    enabled = update_data.pop("enabled", policy.enabled)
    candidate_data = {
        field: getattr(policy, field)
        for field in ModelStorageSyncPolicyCreate.model_fields
    }
    candidate_data.update(update_data)
    candidate_data["enabled"] = enabled
    try:
        candidate = ModelStorageSyncPolicyCreate.model_validate(candidate_data)
    except ValidationError as exc:
        raise InvalidException(message=str(exc.errors()[0]["msg"])) from None
    profile_requires_active = bool(
        enabled
        and (
            not policy.enabled
            or "profile_id" in update_data
            or "scope" in update_data
            or "model_file_id" in update_data
            or "worker_uuids" in update_data
        )
    )
    if profile_requires_active:
        await _active_profile_or_404(session, candidate.profile_id)
    timing_changed = bool(
        {"trigger_mode", "cron_expression", "timezone"} & update_data.keys()
    )
    was_enabled = policy.enabled
    for field, value in candidate.model_dump().items():
        setattr(policy, field, value)
    policy.enabled = enabled
    if (
        not enabled
        or policy.trigger_mode == ModelStorageSyncPolicyTriggerModeEnum.MANUAL
    ):
        policy.next_run_at = None
    elif not was_enabled or timing_changed:
        policy.next_run_at = next_window_start_utc(policy, datetime.now(timezone.utc))
    session.add(policy)
    try:
        await session.commit()
        await session.refresh(policy)
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "Conflict", "model_storage_sync_policy_conflict")
    return ModelStorageSyncPolicyPublic.model_validate(policy)


@router.delete("/{id}")
async def delete_sync_policy(session: SessionDep, id: int):
    policy = await _policy_or_404(session, id)
    active = (
        await session.exec(
            select(ModelStorageSyncPolicyRun.id).where(
                ModelStorageSyncPolicyRun.policy_id == id,
                ModelStorageSyncPolicyRun.state
                == ModelStorageSyncPolicyRunStateEnum.PENDING,
            )
        )
    ).first()
    if active is not None:
        raise HTTPException(409, "Conflict", "model_storage_sync_policy_in_use")
    await session.delete(policy)
    await session.commit()
    return {"ok": True}


@router.get("/{id}/runs", response_model=ModelStorageSyncPolicyRunsPublic)
async def get_sync_policy_runs(session: SessionDep, params: ListParamsDep, id: int):
    await _policy_or_404(session, id)
    items = (
        await session.exec(
            select(ModelStorageSyncPolicyRun)
            .where(ModelStorageSyncPolicyRun.policy_id == id)
            .order_by(ModelStorageSyncPolicyRun.created_at.desc())
            .offset((params.page - 1) * params.perPage)
            .limit(params.perPage)
        )
    ).all()
    total = (
        await session.exec(
            select(func.count(ModelStorageSyncPolicyRun.id)).where(
                ModelStorageSyncPolicyRun.policy_id == id
            )
        )
    ).one()
    return PaginatedList[ModelStorageSyncPolicyRunPublic](
        items=[ModelStorageSyncPolicyRunPublic.model_validate(item) for item in items],
        pagination=_pagination(params, total),
    )


@router.get("/{id}/runs/{run_id}", response_model=ModelStorageSyncPolicyRunPublic)
async def get_sync_policy_run(session: SessionDep, id: int, run_id: int):
    await _policy_or_404(session, id)
    run = await session.get(ModelStorageSyncPolicyRun, run_id)
    if run is None or run.policy_id != id:
        raise NotFoundException(message="model_storage_sync_policy_run_not_found")
    return ModelStorageSyncPolicyRunPublic.model_validate(run)


@router.post("/{id}/run-now", response_model=ModelStorageSyncPolicyRunPublic)
async def run_sync_policy_now(
    request: Request,
    session: SessionDep,
    current_user: CurrentAdminUserDep,
    id: int,
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1),
):
    policy = await _policy_or_404(session, id)
    controller = getattr(
        request.app.state, "model_storage_sync_policy_controller", None
    )
    if controller is None:
        raise HTTPException(
            503, "Unavailable", "model_storage_sync_policy_controller_unavailable"
        )
    try:
        run = await controller.run_now(
            session, policy, current_user.id, idempotency_key, request
        )
    except SyncPolicyRunConflict:
        raise HTTPException(
            409, "idempotency_key_reused", "idempotency_key_reused"
        ) from None
    except SyncPolicyDisabled:
        raise HTTPException(
            409, "Conflict", "model_storage_sync_policy_disabled"
        ) from None
    return ModelStorageSyncPolicyRunPublic.model_validate(run)


async def _policy_or_404(session, policy_id):
    policy = await session.get(ModelStorageSyncPolicy, policy_id)
    if policy is None:
        raise NotFoundException(message="model_storage_sync_policy_not_found")
    return policy


async def _active_profile_or_404(session, profile_id):
    profile = await session.get(ModelPreheatS3Profile, profile_id)
    if profile is None:
        raise NotFoundException(message="model_preheat_s3_profile_not_found")
    if profile.lifecycle_state != ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE:
        raise HTTPException(409, "Conflict", "s3_profile_in_maintenance")
    return profile


def _set_next_run(policy, now):
    policy.next_run_at = (
        next_window_start_utc(policy, now)
        if policy.enabled
        and policy.trigger_mode == ModelStorageSyncPolicyTriggerModeEnum.SCHEDULED
        else None
    )


def _pagination(params, total):
    return Pagination(
        page=params.page,
        perPage=params.perPage,
        total=total,
        totalPage=(total + params.perPage - 1) // params.perPage,
    )
