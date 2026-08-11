from fastapi import APIRouter, Request
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from gpustack.api.exceptions import HTTPException, NotFoundException
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPoliciesPublic,
    ModelPreheatDistributionPolicy,
    ModelPreheatDistributionPolicyPublic,
    ModelPreheatDistributionPolicyUpdate,
)
from gpustack.server.deps import ListParamsDep, SessionDep


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
        items=[_public(item) for item in items],
        pagination=Pagination(
            page=params.page,
            perPage=params.perPage,
            total=total,
            totalPage=(total + params.perPage - 1) // params.perPage,
        ),
    )


@router.get("/{id}", response_model=ModelPreheatDistributionPolicyPublic)
async def get_distribution_policy(session: SessionDep, id: int):
    return _public(await _policy_or_404(session, id))


@router.patch("/{id}", response_model=ModelPreheatDistributionPolicyPublic)
async def update_distribution_policy(
    session: SessionDep,
    id: int,
    policy_in: ModelPreheatDistributionPolicyUpdate,
):
    policy = await _policy_or_404(session, id)
    update_data = policy_in.model_dump(exclude_unset=True)
    if "enabled" in update_data:
        policy.profile_version_stale = False
    for field, value in update_data.items():
        setattr(policy, field, value)
    session.add(policy)
    await session.commit()
    await session.refresh(policy)
    return _public(policy)


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
    await reconciler.reconcile_policy(policy.id)
    policy = await _policy_or_404(session, id, populate_existing=True)
    return _public(policy)


async def _policy_or_404(session, policy_id, populate_existing=False):
    policy = await session.get(
        ModelPreheatDistributionPolicy,
        policy_id,
        populate_existing=populate_existing,
    )
    if policy is None:
        raise NotFoundException(message="model_preheat_distribution_policy_not_found")
    return policy


def _public(policy):
    return ModelPreheatDistributionPolicyPublic.model_validate(policy)
