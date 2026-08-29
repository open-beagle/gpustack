from datetime import datetime, timezone

from fastapi import APIRouter, Header, Request
from pydantic import ValidationError
from sqlalchemy import func, or_, update
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from gpustack.api.exceptions import HTTPException, InvalidException, NotFoundException
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
)
from gpustack.schemas.model_preheat_schedules import (
    ModelPreheatSchedule,
    ModelPreheatScheduleCreate,
    ModelPreheatSchedulePublic,
    ModelPreheatScheduleRun,
    ModelPreheatScheduleRunPublic,
    ModelPreheatScheduleRunStateEnum,
    ModelPreheatScheduleTriggerModeEnum,
    ModelPreheatScheduleRunsPublic,
    ModelPreheatSchedulesPublic,
    ModelPreheatScheduleUpdate,
    next_window_start_utc,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatExecutionStateEnum,
    ModelPreheatTask,
)
from gpustack.server.deps import (
    CurrentAdminUserDep,
    ListParamsDep,
    SessionDep,
)
from gpustack.server.model_preheat_schedule_controller import (
    ScheduleConcurrencyLimit,
    ScheduleDisabled,
    ScheduleRunConflict,
)
from gpustack.server.policy_run_observability import (
    latest_runs_by_owner,
    preheat_schedule_run_observations,
)


router = APIRouter()


@router.get("", response_model=ModelPreheatSchedulesPublic)
async def get_model_preheat_schedules(session: SessionDep, params: ListParamsDep):
    statement = (
        select(ModelPreheatSchedule)
        .order_by(ModelPreheatSchedule.created_at.desc())
        .offset((params.page - 1) * params.perPage)
        .limit(params.perPage)
    )
    items = (await session.exec(statement)).all()
    latest_runs = await latest_runs_by_owner(
        session,
        ModelPreheatScheduleRun,
        ModelPreheatScheduleRun.schedule_id,
        [item.id for item in items],
    )
    observations = await preheat_schedule_run_observations(
        session, list(latest_runs.values())
    )
    total = await ModelPreheatSchedule.count(session)
    return PaginatedList[ModelPreheatSchedulePublic](
        items=[
            _public(
                item,
                latest_run=(
                    _run_public(
                        latest_runs[item.id], observations[latest_runs[item.id].id]
                    )
                    if item.id in latest_runs
                    else None
                ),
            )
            for item in items
        ],
        pagination=Pagination(
            page=params.page,
            perPage=params.perPage,
            total=total,
            totalPage=(total + params.perPage - 1) // params.perPage,
        ),
    )


@router.post("", response_model=ModelPreheatSchedulePublic)
async def create_model_preheat_schedule(
    session: SessionDep,
    current_user: CurrentAdminUserDep,
    schedule_in: ModelPreheatScheduleCreate,
):
    await _profile_or_404(session, schedule_in.s3_profile_id)
    schedule = ModelPreheatSchedule(
        **schedule_in.model_dump(),
        created_by_user_id=current_user.id,
    )
    schedule.next_window_start_utc = (
        next_window_start_utc(schedule_in, datetime.now(timezone.utc))
        if schedule.enabled
        and schedule.trigger_mode == ModelPreheatScheduleTriggerModeEnum.SCHEDULED
        else None
    )
    session.add(schedule)
    try:
        await session.commit()
        await session.refresh(schedule)
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "Conflict", "model_preheat_schedule_conflict")
    return _public(schedule)


@router.get("/{id}", response_model=ModelPreheatSchedulePublic)
async def get_model_preheat_schedule(session: SessionDep, id: int):
    schedule = await _schedule_or_404(session, id)
    latest_runs = await latest_runs_by_owner(
        session,
        ModelPreheatScheduleRun,
        ModelPreheatScheduleRun.schedule_id,
        [schedule.id],
    )
    latest_run = latest_runs.get(schedule.id)
    observations = await preheat_schedule_run_observations(
        session, [latest_run] if latest_run is not None else []
    )
    return _public(
        schedule,
        latest_run=(
            _run_public(latest_run, observations[latest_run.id])
            if latest_run is not None
            else None
        ),
    )


@router.get("/{id}/runs", response_model=ModelPreheatScheduleRunsPublic)
async def get_model_preheat_schedule_runs(
    session: SessionDep, params: ListParamsDep, id: int
):
    await _schedule_or_404(session, id)
    statement = (
        select(ModelPreheatScheduleRun)
        .where(ModelPreheatScheduleRun.schedule_id == id)
        .order_by(ModelPreheatScheduleRun.created_at.desc())
        .offset((params.page - 1) * params.perPage)
        .limit(params.perPage)
    )
    items = (await session.exec(statement)).all()
    observations = await preheat_schedule_run_observations(session, items)
    total = (
        await session.exec(
            select(func.count(ModelPreheatScheduleRun.id)).where(
                ModelPreheatScheduleRun.schedule_id == id
            )
        )
    ).one()
    return PaginatedList[ModelPreheatScheduleRunPublic](
        items=[_run_public(item, observations[item.id]) for item in items],
        pagination=Pagination(
            page=params.page,
            perPage=params.perPage,
            total=total,
            totalPage=(total + params.perPage - 1) // params.perPage,
        ),
    )


@router.get("/{id}/runs/{run_id}", response_model=ModelPreheatScheduleRunPublic)
async def get_model_preheat_schedule_run(session: SessionDep, id: int, run_id: int):
    await _schedule_or_404(session, id)
    run = await session.get(ModelPreheatScheduleRun, run_id)
    if run is None or run.schedule_id != id:
        raise NotFoundException(message="model_preheat_schedule_run_not_found")
    observations = await preheat_schedule_run_observations(
        session, [run], include_tasks=True
    )
    return _run_public(run, observations[run.id])


@router.patch("/{id}", response_model=ModelPreheatSchedulePublic)
async def update_model_preheat_schedule(
    session: SessionDep,
    id: int,
    schedule_in: ModelPreheatScheduleUpdate,
):
    schedule = await _schedule_or_404(session, id)
    update_data = schedule_in.model_dump(exclude_unset=True)
    schedule_was_enabled = schedule.enabled
    timing_changed = bool(
        {"trigger_mode", "cron_expression", "timezone"} & update_data.keys()
    )
    enabled = update_data.pop("enabled", schedule.enabled)
    candidate_data = {
        field: getattr(schedule, field)
        for field in ModelPreheatScheduleCreate.model_fields
    }
    candidate_data.update(update_data)
    try:
        candidate = ModelPreheatScheduleCreate.model_validate(candidate_data)
    except ValidationError as exc:
        raise InvalidException(message=str(exc.errors()[0]["msg"])) from None
    await _profile_or_404(session, candidate.s3_profile_id)
    for field, value in candidate.model_dump().items():
        setattr(schedule, field, value)
    schedule.enabled = enabled
    if (
        not enabled
        or schedule.trigger_mode == ModelPreheatScheduleTriggerModeEnum.MANUAL
    ):
        schedule.next_window_start_utc = None
    elif not schedule_was_enabled or timing_changed:
        schedule.next_window_start_utc = next_window_start_utc(
            schedule, datetime.now(timezone.utc)
        )
    session.add(schedule)
    try:
        await session.commit()
        await session.refresh(schedule)
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "Conflict", "model_preheat_schedule_conflict")
    return _public(schedule)


@router.delete("/{id}")
async def delete_model_preheat_schedule(session: SessionDep, id: int):
    locked = await session.exec(
        update(ModelPreheatSchedule)
        .where(ModelPreheatSchedule.id == id)
        .values(enabled=ModelPreheatSchedule.enabled)
        .execution_options(synchronize_session=False)
    )
    if locked.rowcount != 1:
        await session.rollback()
        raise NotFoundException(message="model_preheat_schedule_not_found")
    schedule = await session.get(ModelPreheatSchedule, id)
    terminal_states = [
        ModelPreheatExecutionStateEnum.READY,
        ModelPreheatExecutionStateEnum.PARTIAL,
        ModelPreheatExecutionStateEnum.ERROR,
        ModelPreheatExecutionStateEnum.CANCELED,
    ]
    active = (
        await session.exec(
            select(ModelPreheatScheduleRun.id)
            .outerjoin(
                ModelPreheatTask, ModelPreheatTask.id == ModelPreheatScheduleRun.task_id
            )
            .where(
                ModelPreheatScheduleRun.schedule_id == id,
                or_(
                    ModelPreheatScheduleRun.state.in_(
                        [
                            ModelPreheatScheduleRunStateEnum.PENDING,
                            ModelPreheatScheduleRunStateEnum.RUNNING,
                        ]
                    ),
                    (
                        ModelPreheatScheduleRun.state
                        == ModelPreheatScheduleRunStateEnum.PAUSED
                    )
                    & ModelPreheatTask.execution_state.not_in(terminal_states),
                ),
            )
        )
    ).first()
    if active is not None:
        raise HTTPException(409, "Conflict", "model_preheat_schedule_in_use")
    await session.delete(schedule)
    await session.commit()
    return {"ok": True}


@router.post("/{id}/run-now", response_model=ModelPreheatScheduleRunPublic)
async def run_model_preheat_schedule_now(
    request: Request,
    session: SessionDep,
    current_user: CurrentAdminUserDep,
    id: int,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    schedule = await _schedule_or_404(session, id)
    controller = getattr(request.app.state, "model_preheat_schedule_controller", None)
    if controller is None:
        raise HTTPException(
            503, "Unavailable", "model_preheat_schedule_controller_unavailable"
        )
    try:
        run = await controller.run_now(
            session,
            schedule,
            current_user.id,
            idempotency_key,
        )
    except ScheduleRunConflict:
        raise HTTPException(
            409, "idempotency_key_reused", "idempotency_key_reused"
        ) from None
    except ScheduleConcurrencyLimit:
        raise HTTPException(
            409, "Conflict", "model_preheat_schedule_concurrency_limit"
        ) from None
    except ScheduleDisabled as exc:
        code = str(exc)
        if code != "model_preheat_schedule_disabled":
            code = "model_preheat_schedule_disabled"
        raise HTTPException(409, "Conflict", code) from None
    observations = await preheat_schedule_run_observations(
        session, [run], include_tasks=True
    )
    return _run_public(run, observations[run.id])


async def _schedule_or_404(session, schedule_id):
    schedule = await session.get(ModelPreheatSchedule, schedule_id)
    if schedule is None:
        raise NotFoundException(message="model_preheat_schedule_not_found")
    return schedule


async def _profile_or_404(session, profile_id):
    profile = await session.get(ModelPreheatS3Profile, profile_id)
    if profile is None:
        raise NotFoundException(message="model_preheat_s3_profile_not_found")
    if profile.lifecycle_state != ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE:
        raise HTTPException(409, "Conflict", "model_preheat_s3_profile_in_maintenance")
    return profile


def _public(schedule, latest_run=None):
    return ModelPreheatSchedulePublic.model_validate(
        schedule, update={"latest_run": latest_run}
    )


def _run_public(run, observation):
    return ModelPreheatScheduleRunPublic.model_validate(
        run,
        update={
            "execution_state": observation.execution_state,
            "summary": observation.summary,
            "tasks": observation.tasks,
        },
    )
