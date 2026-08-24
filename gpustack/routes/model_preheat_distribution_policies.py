from datetime import datetime, timezone

from fastapi import APIRouter, Request
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from gpustack.api.exceptions import HTTPException, NotFoundException
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPoliciesPublic,
    ModelPreheatDistributionPolicy,
    ModelPreheatDistributionPolicyCreate,
    ModelPreheatDistributionPolicyPublic,
    ModelPreheatDistributionPolicyUpdate,
    ModelPreheatDistributionPolicyTriggerModeEnum,
    distribution_selector_digest,
)
from gpustack.schemas.model_preheat_schedules import next_window_start_utc
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatInventoryManifestStateEnum,
)
from gpustack.schemas.model_storage_sync import (
    ModelStorageSyncTask,
    ModelStorageSyncTaskStateEnum,
)
from gpustack.server.deps import CurrentAdminUserDep, ListParamsDep, SessionDep
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity


router = APIRouter()


@router.get("", response_model=ModelPreheatDistributionPoliciesPublic)
async def get_distribution_policies(session: SessionDep, params: ListParamsDep):
    statement = (
        select(ModelPreheatDistributionPolicy)
        .order_by(ModelPreheatDistributionPolicy.created_at.desc())
        .offset((params.page - 1) * params.perPage)
        .limit(params.perPage)
    )
    items = (await session.exec(statement)).all()
    total = await ModelPreheatDistributionPolicy.count(session)
    return PaginatedList[ModelPreheatDistributionPolicyPublic](
        items=[await _public(session, item) for item in items],
        pagination=Pagination(
            page=params.page,
            perPage=params.perPage,
            total=total,
            totalPage=(total + params.perPage - 1) // params.perPage,
        ),
    )


@router.post("", response_model=ModelPreheatDistributionPolicyPublic)
async def create_distribution_policy(
    session: SessionDep,
    current_user: CurrentAdminUserDep,
    policy_in: ModelPreheatDistributionPolicyCreate,
):
    del current_user
    source_sync_task_id = None
    if policy_in.sync_task_id is not None:
        sync_task = await session.get(ModelStorageSyncTask, policy_in.sync_task_id)
        if (
            sync_task is None
            or sync_task.state != ModelStorageSyncTaskStateEnum.READY
            or not sync_task.artifact_id
        ):
            raise HTTPException(409, "Conflict", "sync_task_artifact_not_ready")
        profile_id = sync_task.profile_id
        profile_version = sync_task.profile_config_version
        request_identity = dict(sync_task.request_identity)
        request_digest = sync_task.request_digest
        source_sync_task_id = sync_task.id
        artifact = await _artifact_by_identity(
            session, profile_id, profile_version, sync_task.artifact_id
        )
    else:
        profile_id = policy_in.profile_id
        profile = await session.get(ModelPreheatS3Profile, profile_id)
        if profile is None:
            raise NotFoundException(message="model_preheat_s3_profile_not_found")
        profile_version = profile.config_version
        artifact = await _artifact_by_identity(
            session, profile_id, profile_version, policy_in.artifact_id
        )
        if artifact is not None:
            identity = ModelPreheatIdentity(
                source=artifact.source,
                model_id=artifact.model_id,
                revision=artifact.resolved_revision,
                file_patterns=tuple(artifact.include_patterns),
                exclude_patterns=tuple(artifact.exclude_patterns),
            )
            request_identity = {
                "source": artifact.source,
                "model_id": artifact.model_id,
                "requested_revision": None,
                "include_patterns": list(artifact.include_patterns),
                "exclude_patterns": list(artifact.exclude_patterns),
            }
            request_digest = identity.request_digest
    if artifact is None:
        raise HTTPException(409, "Conflict", "artifact_not_ready")
    if artifact.manifest_state != ModelPreheatInventoryManifestStateEnum.VALID:
        raise HTTPException(409, "Conflict", _artifact_unavailable_code(artifact))
    profile = await session.get(ModelPreheatS3Profile, profile_id)
    if (
        profile is None
        or profile.lifecycle_state != ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE
        or profile.config_version != profile_version
    ):
        raise HTTPException(409, "Conflict", "s3_profile_in_maintenance")
    policy = ModelPreheatDistributionPolicy(
        name=policy_in.name,
        profile_id=profile_id,
        profile_config_version=profile_version,
        request_identity=request_identity,
        request_digest=request_digest,
        target_scope=policy_in.target_scope,
        worker_selector=policy_in.worker_selector,
        gpu_selector=policy_in.gpu_selector,
        selector_digest=distribution_selector_digest(
            policy_in.worker_selector, policy_in.gpu_selector
        ),
        source_artifact_id=artifact.id,
        source_sync_task_id=source_sync_task_id,
        trigger_mode=policy_in.trigger_mode,
        cron_expression=policy_in.cron_expression,
        timezone=policy_in.timezone,
    )
    _set_next_run(policy)
    session.add(policy)
    try:
        await session.commit()
        await session.refresh(policy)
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "Conflict", "distribution_policy_conflict")
    return await _public(session, policy)


@router.get("/{id}", response_model=ModelPreheatDistributionPolicyPublic)
async def get_distribution_policy(session: SessionDep, id: int):
    return await _public(session, await _policy_or_404(session, id))


@router.patch("/{id}", response_model=ModelPreheatDistributionPolicyPublic)
async def update_distribution_policy(
    session: SessionDep,
    id: int,
    policy_in: ModelPreheatDistributionPolicyUpdate,
):
    policy = await _policy_or_404(session, id)
    update_data = policy_in.model_dump(exclude_unset=True)
    enabled = update_data.pop("enabled", policy.enabled)
    candidate_data = {
        "name": policy.name,
        "trigger_mode": policy.trigger_mode,
        "cron_expression": policy.cron_expression,
        "timezone": policy.timezone,
        "profile_id": policy.profile_id,
        "artifact_id": "bound-artifact",
        "target_scope": policy.target_scope,
        "worker_selector": policy.worker_selector,
        "gpu_selector": policy.gpu_selector,
    }
    candidate_data.update(update_data)
    try:
        candidate = ModelPreheatDistributionPolicyCreate.model_validate(candidate_data)
    except ValueError as exc:
        raise HTTPException(422, "Validation Error", str(exc)) from None
    if enabled and not policy.enabled:
        profile = await session.get(ModelPreheatS3Profile, candidate.profile_id)
        if (
            profile is None
            or profile.lifecycle_state != ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE
        ):
            raise HTTPException(409, "Conflict", "s3_profile_in_maintenance")
        if profile.config_version != policy.profile_config_version:
            raise HTTPException(409, "Conflict", "distribution_profile_version_stale")
        artifact = (
            await session.get(ModelPreheatArtifact, policy.source_artifact_id)
            if policy.source_artifact_id is not None
            else None
        )
        if (
            artifact is None
            or artifact.profile_id != policy.profile_id
            or artifact.profile_config_version != policy.profile_config_version
        ):
            raise HTTPException(409, "Conflict", "artifact_not_ready")
        if artifact.manifest_state != ModelPreheatInventoryManifestStateEnum.VALID:
            raise HTTPException(409, "Conflict", _artifact_unavailable_code(artifact))
    timing_changed = bool(
        {"trigger_mode", "cron_expression", "timezone"} & update_data.keys()
    )
    was_enabled = policy.enabled
    if enabled and not was_enabled:
        policy.profile_version_stale = False
    for field in ("name", "trigger_mode", "cron_expression", "timezone"):
        setattr(policy, field, getattr(candidate, field))
    policy.enabled = enabled
    if (
        not enabled
        or policy.trigger_mode
        != ModelPreheatDistributionPolicyTriggerModeEnum.SCHEDULED
    ):
        policy.next_run_at = None
    elif not was_enabled or timing_changed:
        policy.next_run_at = next_window_start_utc(policy, datetime.now(timezone.utc))
    session.add(policy)
    await session.commit()
    await session.refresh(policy)
    return await _public(session, policy)


@router.delete("/{id}")
async def delete_distribution_policy(session: SessionDep, id: int):
    policy = await _policy_or_404(session, id)
    try:
        await session.delete(policy)
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "Conflict", "distribution_policy_in_use") from None
    return {"ok": True}


@router.post("/{id}/reconcile", response_model=ModelPreheatDistributionPolicyPublic)
async def reconcile_distribution_policy(request: Request, session: SessionDep, id: int):
    policy = await _policy_or_404(session, id)
    reconciler = getattr(request.app.state, "model_preheat_worker_reconciler", None)
    if reconciler is None:
        raise HTTPException(503, "Unavailable", "distribution_reconciler_unavailable")
    if hasattr(reconciler, "reconcile_manual_policy"):
        await reconciler.reconcile_manual_policy(policy.id)
    else:
        await reconciler.reconcile_policy(policy.id)
    policy = await _policy_or_404(session, id, populate_existing=True)
    return await _public(session, policy)


async def _policy_or_404(session, policy_id, populate_existing=False):
    policy = await session.get(
        ModelPreheatDistributionPolicy,
        policy_id,
        populate_existing=populate_existing,
    )
    if policy is None:
        raise NotFoundException(message="model_preheat_distribution_policy_not_found")
    return policy


async def _public(session, policy):
    artifact = (
        await session.get(ModelPreheatArtifact, policy.source_artifact_id)
        if policy.source_artifact_id is not None
        else None
    )
    return ModelPreheatDistributionPolicyPublic.model_validate(
        policy,
        update={"source_artifact": artifact.artifact_id if artifact else None},
    )


async def _artifact_by_identity(session, profile_id, profile_version, artifact_id):
    if artifact_id is None:
        return None
    return (
        await session.exec(
            select(ModelPreheatArtifact).where(
                ModelPreheatArtifact.profile_id == profile_id,
                ModelPreheatArtifact.profile_config_version == profile_version,
                ModelPreheatArtifact.artifact_id == artifact_id,
            )
        )
    ).first()


def _artifact_unavailable_code(artifact):
    if artifact.manifest_state == ModelPreheatInventoryManifestStateEnum.STALE:
        return "artifact_stale"
    return "artifact_not_ready"


def _set_next_run(policy):
    policy.next_run_at = (
        next_window_start_utc(policy, datetime.now(timezone.utc))
        if policy.enabled
        and policy.trigger_mode
        == ModelPreheatDistributionPolicyTriggerModeEnum.SCHEDULED
        else None
    )
