from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query
from sqlalchemy import func
from sqlmodel import col, select

from gpustack.api.exceptions import HTTPException, NotFoundException
from gpustack.schemas.common import Pagination
from gpustack.schemas.scheduler import (
    SchedulerPolicy,
    SchedulerPolicyPublic,
    SchedulerPolicyUpdate,
    SchedulingAttemptEvent,
    SchedulingAttemptEventPublic,
    SchedulingAttemptEventsPublic,
    SchedulingOutcome,
)
from gpustack.server.deps import CurrentAdminUserDep, ListParamsDep, SessionDep


router = APIRouter()


@router.get("/policies/aggregation", response_model=SchedulerPolicyPublic)
async def get_aggregation_policy(session: SessionDep):
    policy = await SchedulerPolicy.one_by_field(session, "code", "aggregation")
    if policy is None:
        raise NotFoundException(message="aggregation_scheduler_policy_not_found")
    return SchedulerPolicyPublic.model_validate(policy)


@router.put("/policies/aggregation", response_model=SchedulerPolicyPublic)
async def update_aggregation_policy(
    session: SessionDep,
    current_user: CurrentAdminUserDep,
    policy_in: SchedulerPolicyUpdate,
):
    policy = await SchedulerPolicy.one_by_field(session, "code", "aggregation")
    if policy is None:
        raise NotFoundException(message="aggregation_scheduler_policy_not_found")
    if policy.runtime_revision != policy_in.expected_revision:
        raise HTTPException(409, "Conflict", "scheduler_policy_revision_conflict")
    if abs(policy_in.target_revision - policy_in.expected_revision) != 1:
        raise HTTPException(
            400, "Bad Request", "invalid_scheduler_policy_target_revision"
        )

    policy.aggregation_rate = policy_in.aggregation_rate
    policy.enabled = policy_in.enabled
    policy.runtime_revision = policy_in.target_revision
    policy.updated_by = current_user.username
    await policy.save(session)
    return SchedulerPolicyPublic.model_validate(policy)


@router.get("/events", response_model=SchedulingAttemptEventsPublic)
async def get_scheduling_events(
    session: SessionDep,
    params: ListParamsDep,
    policy_code: Optional[str] = "aggregation",
    workload_id: Optional[str] = None,
    outcome: Optional[SchedulingOutcome] = None,
    start: Optional[datetime] = Query(default=None),
    end: Optional[datetime] = Query(default=None),
):
    if start and end and start >= end:
        raise HTTPException(400, "Bad Request", "invalid_scheduling_event_time_range")

    conditions = []
    if policy_code:
        conditions.append(SchedulingAttemptEvent.policy_code == policy_code)
    if workload_id:
        conditions.append(SchedulingAttemptEvent.workload_id == workload_id)
    if outcome:
        conditions.append(SchedulingAttemptEvent.outcome == outcome)
    if start:
        conditions.append(SchedulingAttemptEvent.occurred_at >= start)
    if end:
        conditions.append(SchedulingAttemptEvent.occurred_at < end)

    statement = select(SchedulingAttemptEvent)
    count_statement = select(func.count()).select_from(SchedulingAttemptEvent)
    for condition in conditions:
        statement = statement.where(condition)
        count_statement = count_statement.where(condition)
    statement = (
        statement.order_by(col(SchedulingAttemptEvent.occurred_at).desc())
        .offset((params.page - 1) * params.perPage)
        .limit(params.perPage)
    )
    items = (await session.exec(statement)).all()
    total = (await session.exec(count_statement)).one()
    return SchedulingAttemptEventsPublic(
        items=[SchedulingAttemptEventPublic.model_validate(item) for item in items],
        pagination=Pagination(
            page=params.page,
            perPage=params.perPage,
            total=total,
            totalPage=(total + params.perPage - 1) // params.perPage,
        ),
    )
