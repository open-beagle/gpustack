import hashlib
import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import inspect, or_, update
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3ConnectivityStateEnum,
    ModelPreheatS3Profile,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatConnectivityCheckStateEnum,
    ModelPreheatS3ConnectivityCheck,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum


DEFAULT_CONNECTIVITY_TTL = timedelta(minutes=10)
_ACTIVE_CHECK_STATES = {
    ModelPreheatConnectivityCheckStateEnum.PENDING,
    ModelPreheatConnectivityCheckStateEnum.RUNNING,
}
_TERMINAL_WORKER_TASK_STATES = {
    ModelPreheatWorkerTaskStateEnum.READY,
    ModelPreheatWorkerTaskStateEnum.ERROR,
    ModelPreheatWorkerTaskStateEnum.CANCELED,
    ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED,
}


class ConnectivityCheckIdempotencyConflict(Exception):
    pass


async def create_or_reuse_connectivity_check(
    session,
    profile: ModelPreheatS3Profile,
    target_worker_uuids=None,
    *,
    idempotency_scope_key: str | None = None,
    request_hash: str | None = None,
    scope_discriminator: str | None = None,
    update_profile_pointer: bool = True,
):
    profile_identity = inspect(profile).identity
    if profile_identity is None:
        return None
    profile = await session.get(ModelPreheatS3Profile, profile_identity[0])
    if profile is None:
        return None
    if idempotency_scope_key is not None:
        idempotent_check = await _idempotent_connectivity_check(
            session, idempotency_scope_key, request_hash
        )
        if idempotent_check is not None:
            return idempotent_check
    all_registered_workers = await current_registered_workers(session)
    all_ready_workers = [
        worker
        for worker in all_registered_workers
        if worker.state == WorkerStateEnum.READY
    ]
    if target_worker_uuids is None:
        ready_workers = all_ready_workers
    else:
        requested_uuids = set(target_worker_uuids)
        ready_workers = [
            worker
            for worker in all_ready_workers
            if worker.worker_uuid in requested_uuids
        ]
    if not ready_workers:
        if all_ready_workers:
            return None
        values = {
            "connectivity_state": (
                ModelPreheatS3ConnectivityStateEnum.PARTIAL
                if all_registered_workers
                else ModelPreheatS3ConnectivityStateEnum.NO_WORKERS
            ),
        }
        conditions = [
            ModelPreheatS3Profile.id == profile.id,
            ModelPreheatS3Profile.config_version == profile.config_version,
        ]
        if update_profile_pointer:
            conditions.append(
                _profile_pointer_matches(profile.last_connectivity_check_id)
            )
            values.update(
                last_connectivity_check_id=None,
                last_connectivity_checked_at=None,
            )
        result = await session.exec(
            update(ModelPreheatS3Profile).where(*conditions).values(**values)
        )
        _expire_profile_after_failed_cas(session, profile, result.rowcount)
        await session.commit()
        await session.refresh(profile)
        return None

    active_key = _connectivity_scope_key(
        profile, ready_workers, scope_discriminator=scope_discriminator
    )
    statement = select(ModelPreheatS3ConnectivityCheck).where(
        ModelPreheatS3ConnectivityCheck.active_key == active_key
    )
    existing = (await session.exec(statement)).first()
    if existing is not None:
        changed = False
        try:
            if idempotency_scope_key is not None and existing.idempotency_key is None:
                result = await session.exec(
                    update(ModelPreheatS3ConnectivityCheck)
                    .where(
                        ModelPreheatS3ConnectivityCheck.id == existing.id,
                        ModelPreheatS3ConnectivityCheck.idempotency_key.is_(None),
                    )
                    .values(
                        idempotency_key=idempotency_scope_key,
                        request_hash=request_hash,
                    )
                    .execution_options(synchronize_session=False)
                )
                changed = result.rowcount == 1
            if (
                update_profile_pointer
                and profile.last_connectivity_check_id != existing.id
            ):
                result = await _mark_profile_checking(
                    session,
                    profile,
                    existing.id,
                    update_profile_pointer=True,
                )
                _expire_profile_after_failed_cas(session, profile, result.rowcount)
                changed = changed or result.rowcount == 1
                if result.rowcount == 0:
                    await session.refresh(profile)
            if changed:
                await session.commit()
                await session.refresh(existing)
                await session.refresh(profile)
        except IntegrityError:
            await session.rollback()
            if idempotency_scope_key is not None:
                winner = await _idempotent_connectivity_check(
                    session, idempotency_scope_key, request_hash
                )
                if winner is not None:
                    await session.refresh(profile)
                    return winner
            raise
        return existing

    now = datetime.now(timezone.utc)
    check = ModelPreheatS3ConnectivityCheck(
        profile_id=profile.id,
        profile_config_version=profile.config_version,
        idempotency_key=idempotency_scope_key,
        request_hash=request_hash,
        scope_key=active_key,
        active_key=active_key,
        state=ModelPreheatConnectivityCheckStateEnum.RUNNING,
        target_worker_uuids=[worker.worker_uuid for worker in ready_workers],
        started_at=now,
    )
    try:
        session.add(check)
        await session.flush()
        for worker in ready_workers:
            session.add(
                ModelPreheatWorkerTask(
                    connectivity_check_id=check.id,
                    worker_uuid=worker.worker_uuid,
                    worker_id=worker.id,
                    role=ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
                )
            )
        result = await _mark_profile_checking(
            session,
            profile,
            check.id,
            update_profile_pointer=update_profile_pointer,
        )
        _expire_profile_after_failed_cas(session, profile, result.rowcount)
        await session.commit()
    except IntegrityError:
        await session.rollback()
        winner = None
        if idempotency_scope_key is not None:
            winner = await _idempotent_connectivity_check(
                session, idempotency_scope_key, request_hash
            )
        if winner is None:
            winner = (await session.exec(statement)).first()
        if winner is None:
            winner = (
                await session.exec(
                    select(ModelPreheatS3ConnectivityCheck)
                    .where(ModelPreheatS3ConnectivityCheck.scope_key == active_key)
                    .order_by(ModelPreheatS3ConnectivityCheck.id.desc())
                )
            ).first()
        if winner is None:
            raise
        await session.refresh(profile)
        return winner
    await session.refresh(check)
    await session.refresh(profile)
    return check


async def aggregate_connectivity_check(session, check_id: int):
    check = await ModelPreheatS3ConnectivityCheck.one_by_id(session, check_id)
    if check is None:
        return None
    profile = await ModelPreheatS3Profile.one_by_id(session, check.profile_id)
    if profile is None:
        return check

    task_statement = (
        select(ModelPreheatWorkerTask)
        .where(
            ModelPreheatWorkerTask.connectivity_check_id == check.id,
            ModelPreheatWorkerTask.role
            == ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
        )
        .execution_options(populate_existing=True)
    )
    tasks = (await session.exec(task_statement)).all()
    tasks_by_uuid = {task.worker_uuid: task for task in tasks}
    current_registered_by_uuid = {
        worker.worker_uuid: worker
        for worker in await current_registered_workers(session)
    }
    current_ready_by_uuid = {
        worker_uuid: worker
        for worker_uuid, worker in current_registered_by_uuid.items()
        if worker.state == WorkerStateEnum.READY
    }
    current_ready_uuids = set(current_ready_by_uuid)
    target_uuids = set(check.target_worker_uuids)
    ready_uuids = {
        worker_uuid
        for worker_uuid in target_uuids
        if tasks_by_uuid.get(worker_uuid)
        and tasks_by_uuid[worker_uuid].state == ModelPreheatWorkerTaskStateEnum.READY
        and worker_uuid in current_ready_uuids
        and tasks_by_uuid[worker_uuid].worker_id
        == current_ready_by_uuid[worker_uuid].id
    }
    error_uuids = {
        worker_uuid
        for worker_uuid in target_uuids
        if tasks_by_uuid.get(worker_uuid)
        and tasks_by_uuid[worker_uuid].state
        in {
            ModelPreheatWorkerTaskStateEnum.ERROR,
            ModelPreheatWorkerTaskStateEnum.CANCELED,
            ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED,
        }
        and worker_uuid in current_ready_uuids
        and tasks_by_uuid[worker_uuid].worker_id
        == current_ready_by_uuid[worker_uuid].id
    }
    not_checked_uuids = target_uuids - ready_uuids - error_uuids
    not_checked_uuids.update(target_uuids - current_ready_uuids)
    check.success_count = len(ready_uuids)
    check.failed_count = len(error_uuids)
    check.not_checked_count = len(not_checked_uuids)

    all_tasks_terminal = all(
        tasks_by_uuid.get(worker_uuid)
        and (
            (
                worker_uuid in current_ready_by_uuid
                and tasks_by_uuid[worker_uuid].worker_id
                != current_ready_by_uuid[worker_uuid].id
            )
            or tasks_by_uuid[worker_uuid].state in _TERMINAL_WORKER_TASK_STATES
        )
        for worker_uuid in target_uuids
    )
    offline_uuids = target_uuids - current_ready_uuids
    if target_uuids and len(ready_uuids) == len(target_uuids):
        check.state = ModelPreheatConnectivityCheckStateEnum.AVAILABLE
    elif not all_tasks_terminal and not offline_uuids:
        check.state = ModelPreheatConnectivityCheckStateEnum.RUNNING
    elif error_uuids:
        check.state = (
            ModelPreheatConnectivityCheckStateEnum.PARTIAL
            if ready_uuids or not_checked_uuids
            else ModelPreheatConnectivityCheckStateEnum.UNAVAILABLE
        )
    elif not_checked_uuids:
        check.state = ModelPreheatConnectivityCheckStateEnum.PARTIAL
    else:
        check.state = ModelPreheatConnectivityCheckStateEnum.UNAVAILABLE

    if check.state not in _ACTIVE_CHECK_STATES and check.finished_at is None:
        check.finished_at = datetime.now(timezone.utc)
    if check.state not in _ACTIVE_CHECK_STATES:
        check.active_key = None
    session.add(check)

    profile_id = profile.id
    profile_config_version = check.profile_config_version
    await session.commit()
    await session.refresh(check)
    await _refresh_aggregated_profile_connectivity_state(
        session,
        profile_id,
        profile_config_version,
    )
    await session.refresh(check)
    await session.refresh(profile)
    return check


async def current_ready_workers(session) -> list[Worker]:
    return [
        worker
        for worker in await current_registered_workers(session)
        if worker.state == WorkerStateEnum.READY
    ]


async def current_registered_workers(session) -> list[Worker]:
    statement = (
        select(Worker)
        .order_by(Worker.worker_uuid, Worker.id.desc())
        .execution_options(populate_existing=True)
    )
    workers = (await session.exec(statement)).all()
    latest_by_uuid = {}
    for worker in workers:
        latest_by_uuid.setdefault(worker.worker_uuid, worker)
    return list(latest_by_uuid.values())


async def _point_profile_to_check(session, profile, check_id: int):
    return await session.exec(
        update(ModelPreheatS3Profile)
        .where(
            ModelPreheatS3Profile.id == profile.id,
            ModelPreheatS3Profile.config_version == profile.config_version,
            or_(
                ModelPreheatS3Profile.last_connectivity_check_id.is_(None),
                ModelPreheatS3Profile.last_connectivity_check_id < check_id,
            ),
        )
        .values(
            connectivity_state=ModelPreheatS3ConnectivityStateEnum.CHECKING,
            last_connectivity_check_id=check_id,
            last_connectivity_checked_at=None,
        )
    )


async def _mark_profile_checking(
    session, profile, check_id: int, *, update_profile_pointer: bool
):
    if update_profile_pointer:
        return await _point_profile_to_check(session, profile, check_id)
    return await session.exec(
        update(ModelPreheatS3Profile)
        .where(
            ModelPreheatS3Profile.id == profile.id,
            ModelPreheatS3Profile.config_version == profile.config_version,
        )
        .values(
            connectivity_state=ModelPreheatS3ConnectivityStateEnum.CHECKING,
            last_connectivity_checked_at=None,
        )
    )


def _profile_pointer_matches(check_id: int | None):
    if check_id is None:
        return ModelPreheatS3Profile.last_connectivity_check_id.is_(None)
    return ModelPreheatS3Profile.last_connectivity_check_id == check_id


def _expire_profile_after_failed_cas(session, profile, rowcount: int):
    if rowcount == 0:
        session.expire(profile)
    elif rowcount != 1:
        raise RuntimeError("unexpected_profile_connectivity_update_count")


def _connectivity_scope_key(
    profile, workers: list[Worker], *, scope_discriminator: str | None = None
) -> str:
    snapshot = sorted((worker.worker_uuid, worker.id) for worker in workers)
    payload_items = [profile.id, profile.config_version, snapshot]
    if scope_discriminator is not None:
        payload_items.append(scope_discriminator)
    payload = json.dumps(
        payload_items,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _idempotent_connectivity_check(
    session, idempotency_scope_key: str, request_hash: str | None
):
    check = (
        await session.exec(
            select(ModelPreheatS3ConnectivityCheck).where(
                ModelPreheatS3ConnectivityCheck.idempotency_key == idempotency_scope_key
            )
        )
    ).first()
    if check is not None and check.request_hash != request_hash:
        raise ConnectivityCheckIdempotencyConflict
    return check


async def _aggregate_profile_connectivity_state(
    session,
    profile_id: int,
    profile_config_version: int,
    current_registered_by_uuid: dict[str, Worker],
):
    if not current_registered_by_uuid:
        return {
            "connectivity_state": ModelPreheatS3ConnectivityStateEnum.NO_WORKERS,
            "last_connectivity_checked_at": None,
        }

    latest_results = await latest_connectivity_results_for_workers(
        session,
        profile_id,
        profile_config_version,
        list(current_registered_by_uuid.values()),
    )

    current_uuids = set(current_registered_by_uuid)
    ready_worker_uuids = {
        worker_uuid
        for worker_uuid, worker in current_registered_by_uuid.items()
        if worker.state == WorkerStateEnum.READY
    }
    offline_count = len(current_uuids - ready_worker_uuids)
    missing_uuids = ready_worker_uuids - set(latest_results)
    ready_count = sum(
        worker_uuid in ready_worker_uuids
        and task.state == ModelPreheatWorkerTaskStateEnum.READY
        for worker_uuid, (task, _) in latest_results.items()
    )
    active_count = sum(
        worker_uuid in ready_worker_uuids
        and task.state not in _TERMINAL_WORKER_TASK_STATES
        for worker_uuid, (task, _) in latest_results.items()
    )
    failed_count = sum(
        worker_uuid in ready_worker_uuids
        and task.state in _TERMINAL_WORKER_TASK_STATES
        and task.state != ModelPreheatWorkerTaskStateEnum.READY
        for worker_uuid, (task, _) in latest_results.items()
    )

    if ready_count == len(current_uuids):
        state = ModelPreheatS3ConnectivityStateEnum.AVAILABLE
    elif active_count:
        state = ModelPreheatS3ConnectivityStateEnum.CHECKING
    elif ready_count:
        state = ModelPreheatS3ConnectivityStateEnum.PARTIAL
    elif failed_count and not missing_uuids and not offline_count:
        state = ModelPreheatS3ConnectivityStateEnum.UNAVAILABLE
    elif failed_count or offline_count:
        state = ModelPreheatS3ConnectivityStateEnum.PARTIAL
    else:
        state = ModelPreheatS3ConnectivityStateEnum.STALE

    checked_at = None
    if state not in {
        ModelPreheatS3ConnectivityStateEnum.CHECKING,
        ModelPreheatS3ConnectivityStateEnum.STALE,
    }:
        finished_times = [
            result_check.finished_at
            for _, result_check in latest_results.values()
            if result_check.finished_at is not None
        ]
        if finished_times:
            checked_at = min(finished_times)
    return {
        "connectivity_state": state,
        "last_connectivity_checked_at": checked_at,
    }


async def _refresh_aggregated_profile_connectivity_state(
    session,
    profile_id: int,
    profile_config_version: int,
):
    for _ in range(3):
        profile = await session.get(ModelPreheatS3Profile, profile_id)
        if profile is None or profile.config_version != profile_config_version:
            return
        await session.refresh(profile)
        if profile.config_version != profile_config_version:
            return

        expected_pointer = profile.last_connectivity_check_id
        expected_state = profile.connectivity_state
        expected_checked_at = profile.last_connectivity_checked_at
        current_registered_by_uuid = {
            worker.worker_uuid: worker
            for worker in await current_registered_workers(session)
        }
        profile_update = await _aggregate_profile_connectivity_state(
            session,
            profile_id,
            profile_config_version,
            current_registered_by_uuid,
        )
        result = await session.exec(
            update(ModelPreheatS3Profile)
            .where(
                ModelPreheatS3Profile.id == profile_id,
                ModelPreheatS3Profile.config_version == profile_config_version,
                _profile_pointer_matches(expected_pointer),
                ModelPreheatS3Profile.connectivity_state == expected_state,
                ModelPreheatS3Profile.last_connectivity_checked_at
                == expected_checked_at,
            )
            .values(**profile_update)
            .execution_options(synchronize_session=False)
        )
        if result.rowcount == 1:
            await session.commit()
            return
        if result.rowcount != 0:
            raise RuntimeError("unexpected_profile_connectivity_update_count")
        await session.rollback()

    await session.get(ModelPreheatS3Profile, profile_id)


async def latest_connectivity_results_for_workers(
    session,
    profile_id: int,
    profile_config_version: int,
    workers: list[Worker],
):
    workers_by_uuid = {worker.worker_uuid: worker for worker in workers}
    if not workers_by_uuid:
        return {}

    statement = (
        select(ModelPreheatWorkerTask, ModelPreheatS3ConnectivityCheck)
        .join(
            ModelPreheatS3ConnectivityCheck,
            ModelPreheatWorkerTask.connectivity_check_id
            == ModelPreheatS3ConnectivityCheck.id,
        )
        .where(
            ModelPreheatS3ConnectivityCheck.profile_id == profile_id,
            ModelPreheatS3ConnectivityCheck.profile_config_version
            == profile_config_version,
            ModelPreheatWorkerTask.role
            == ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
            ModelPreheatWorkerTask.worker_uuid.in_(workers_by_uuid),
        )
        .order_by(
            ModelPreheatWorkerTask.worker_uuid,
            ModelPreheatS3ConnectivityCheck.id.desc(),
            ModelPreheatWorkerTask.id.desc(),
        )
        .execution_options(populate_existing=True)
    )
    latest_results = {}
    for task, result_check in (await session.exec(statement)).all():
        if task.worker_uuid in latest_results:
            continue
        current_worker = workers_by_uuid[task.worker_uuid]
        if task.worker_id != current_worker.id:
            continue
        latest_results[task.worker_uuid] = (task, result_check)
    return latest_results


async def mark_profile_stale_if_expired(
    session, profile: ModelPreheatS3Profile, ttl: timedelta = DEFAULT_CONNECTIVITY_TTL
) -> bool:
    profile_identity = inspect(profile).identity
    if profile_identity is None:
        return False
    profile = await session.get(ModelPreheatS3Profile, profile_identity[0])
    if profile is None:
        return False
    checked_at = profile.last_connectivity_checked_at
    if (
        profile.connectivity_state
        not in {
            ModelPreheatS3ConnectivityStateEnum.AVAILABLE,
            ModelPreheatS3ConnectivityStateEnum.PARTIAL,
            ModelPreheatS3ConnectivityStateEnum.UNAVAILABLE,
        }
        or checked_at is None
        or checked_at + ttl >= datetime.now(timezone.utc)
    ):
        return False
    result = await session.exec(
        update(ModelPreheatS3Profile)
        .where(
            ModelPreheatS3Profile.id == profile.id,
            ModelPreheatS3Profile.config_version == profile.config_version,
            _profile_pointer_matches(profile.last_connectivity_check_id),
            ModelPreheatS3Profile.connectivity_state == profile.connectivity_state,
            ModelPreheatS3Profile.last_connectivity_checked_at == checked_at,
        )
        .values(connectivity_state=ModelPreheatS3ConnectivityStateEnum.STALE)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount not in {0, 1}:
        raise RuntimeError("unexpected_profile_connectivity_update_count")
    await session.refresh(profile)
    return result.rowcount == 1


def connectivity_ttl_from_config(config) -> timedelta:
    seconds = getattr(config, "model_preheat_connectivity_ttl_seconds", 600)
    return timedelta(seconds=max(1, int(seconds)))


def worker_network_identity_changed(before: Worker, after: Worker) -> bool:
    return any(
        getattr(before, field) != getattr(after, field)
        for field in ("worker_uuid", "ip", "port", "hostname")
    )
