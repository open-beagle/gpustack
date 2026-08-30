from collections import defaultdict

from sqlalchemy import func, or_
from sqlmodel import select

from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicyRunTask,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatExecutionStateEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
)
from gpustack.schemas.model_files import ModelFile
from gpustack.schemas.model_storage_sync import ModelStorageSyncTask
from gpustack.schemas.workers import Worker
from gpustack.schemas.policy_runs import (
    PolicyRunExecutionStateEnum,
    PolicyRunObservation,
    PolicyRunSummary,
    PolicyRunTaskPublic,
)


async def distribution_run_observations(session, runs, *, include_tasks=False):
    run_ids = [run.id for run in runs if run.id is not None]
    grouped = defaultdict(list)
    if run_ids:
        rows = (
            await session.exec(
                select(
                    ModelPreheatDistributionPolicyRunTask.run_id, ModelPreheatWorkerTask
                )
                .join(
                    ModelPreheatWorkerTask,
                    ModelPreheatWorkerTask.id
                    == ModelPreheatDistributionPolicyRunTask.task_id,
                )
                .where(ModelPreheatDistributionPolicyRunTask.run_id.in_(run_ids))
                .order_by(
                    ModelPreheatDistributionPolicyRunTask.run_id,
                    ModelPreheatWorkerTask.id,
                )
            )
        ).all()
        worker_lookup = await _worker_lookup(
            session,
            worker_ids=[task.worker_id for _run_id, task in rows],
            worker_uuids=[task.worker_uuid for _run_id, task in rows],
        )
        artifact_lookup = await _artifact_lookup(
            session,
            [
                task.distribution_artifact_id
                for _run_id, task in rows
                if task.distribution_artifact_id
            ],
        )
        for run_id, task in rows:
            grouped[run_id].append(
                _preheat_worker_item(task, worker_lookup, artifact_lookup)
            )
    for run in runs:
        grouped[run.id].extend(_outcome_items(run.outcome))
    return {
        run.id: _observation(grouped[run.id], run.state, include_tasks=include_tasks)
        for run in runs
    }


async def preheat_schedule_run_observations(session, runs, *, include_tasks=False):
    task_ids = [run.task_id for run in runs if run.task_id is not None]
    parents = {}
    grouped = defaultdict(list)
    if task_ids:
        parents = {
            task.id: task
            for task in (
                await session.exec(
                    select(ModelPreheatTask).where(ModelPreheatTask.id.in_(task_ids))
                )
            ).all()
        }
        workers = (
            await session.exec(
                select(ModelPreheatWorkerTask)
                .where(ModelPreheatWorkerTask.task_id.in_(task_ids))
                .order_by(ModelPreheatWorkerTask.task_id, ModelPreheatWorkerTask.id)
            )
        ).all()
        worker_lookup = await _worker_lookup(
            session,
            worker_ids=[task.worker_id for task in workers],
            worker_uuids=[task.worker_uuid for task in workers],
        )
        artifact_lookup = await _artifact_lookup(
            session,
            [
                task.distribution_artifact_id
                for task in workers
                if task.distribution_artifact_id
            ],
        )
        for task in workers:
            grouped[task.task_id].append(
                _preheat_worker_item(task, worker_lookup, artifact_lookup)
            )
    observations = {}
    for run in runs:
        parent = parents.get(run.task_id)
        items = grouped.get(run.task_id, [])
        observation = _observation(items, run.state, include_tasks=include_tasks)
        if parent is not None:
            observation = _apply_parent(observation, parent)
        observations[run.id] = observation
    return observations


async def sync_policy_run_observations(session, runs, *, include_tasks=False):
    run_task_ids = {}
    run_outcome_items = defaultdict(list)
    all_task_ids = set()
    payload_model_file_ids = set()
    payload_worker_ids = set()
    payload_worker_uuids = set()
    for run in runs:
        task_ids = []
        payload = run.response_payload if isinstance(run.response_payload, dict) else {}
        created = payload.get("created", [])
        if not isinstance(created, list):
            created = []
        for item in created:
            task_id = item.get("task_id") if isinstance(item, dict) else None
            if isinstance(task_id, int):
                task_ids.append(task_id)
                all_task_ids.add(task_id)
        payload_items = _payload_items(payload, "skipped") + _payload_items(
            payload, "failed"
        )
        run_outcome_items[run.id].extend(payload_items)
        payload_model_file_ids.update(
            item.model_file_id for item in payload_items if item.model_file_id
        )
        payload_worker_ids.update(
            item.worker_id for item in payload_items if item.worker_id
        )
        payload_worker_uuids.update(
            item.worker_uuid for item in payload_items if item.worker_uuid
        )
        run_task_ids[run.id] = task_ids
    tasks = {}
    worker_lookup = {}
    model_file_lookup = {}
    if all_task_ids:
        sync_tasks = (
            await session.exec(
                select(ModelStorageSyncTask).where(
                    ModelStorageSyncTask.id.in_(all_task_ids)
                )
            )
        ).all()
        tasks = {task.id: task for task in sync_tasks}
        worker_lookup = await _worker_lookup(
            session,
            worker_ids=[task.worker_id for task in sync_tasks],
            worker_uuids=[task.worker_uuid for task in sync_tasks],
        )
    if payload_model_file_ids:
        model_files = (
            await session.exec(
                select(ModelFile).where(ModelFile.id.in_(payload_model_file_ids))
            )
        ).all()
        model_file_lookup = {model_file.id: model_file for model_file in model_files}
        payload_worker_ids.update(
            model_file.worker_id for model_file in model_files if model_file.worker_id
        )
    if payload_worker_ids or payload_worker_uuids:
        payload_worker_lookup = await _worker_lookup(
            session,
            worker_ids=payload_worker_ids,
            worker_uuids=payload_worker_uuids,
        )
        worker_lookup = {**payload_worker_lookup, **worker_lookup}
    for items in run_outcome_items.values():
        for item in items:
            _enrich_payload_item(item, model_file_lookup, worker_lookup)
    return {
        run.id: _observation(
            [
                _sync_item(tasks[task_id], worker_lookup)
                for task_id in run_task_ids[run.id]
                if task_id in tasks
            ]
            + [
                _missing_sync_task_item(task_id)
                for task_id in run_task_ids[run.id]
                if task_id not in tasks
            ]
            + run_outcome_items[run.id],
            run.state,
            include_tasks=include_tasks,
        )
        for run in runs
    }


async def latest_runs_by_owner(session, run_model, owner_field, owner_ids):
    """一次查询返回每个 owner 的最新 Run，供策略列表批量组装摘要。"""
    owner_ids = list(dict.fromkeys(owner_ids))
    if not owner_ids:
        return {}
    latest = (
        select(
            owner_field.label("owner_id"),
            func.max(run_model.id).label("run_id"),
        )
        .where(owner_field.in_(owner_ids))
        .group_by(owner_field)
        .subquery()
    )
    runs = (
        await session.exec(
            select(run_model).join(latest, run_model.id == latest.c.run_id)
        )
    ).all()
    return {getattr(run, owner_field.key): run for run in runs}


async def _worker_lookup(session, *, worker_ids, worker_uuids):
    ids = {worker_id for worker_id in worker_ids if worker_id is not None}
    uuids = {uuid for uuid in worker_uuids if uuid}
    if not ids and not uuids:
        return {}
    conditions = []
    if ids:
        conditions.append(Worker.id.in_(ids))
    if uuids:
        conditions.append(Worker.worker_uuid.in_(uuids))
    rows = (await session.exec(select(Worker).where(or_(*conditions)))).all()
    lookup = {}
    for worker in rows:
        lookup[("id", worker.id)] = worker
        lookup[("uuid", worker.worker_uuid)] = worker
    return lookup


async def _artifact_lookup(session, artifact_ids):
    ids = {artifact_id for artifact_id in artifact_ids if artifact_id}
    if not ids:
        return {}
    rows = (
        await session.exec(
            select(ModelPreheatArtifact).where(
                ModelPreheatArtifact.artifact_id.in_(ids)
            )
        )
    ).all()
    return {artifact.artifact_id: artifact for artifact in rows}


def _lookup_worker(lookup, worker_id, worker_uuid):
    return lookup.get(("id", worker_id)) or lookup.get(("uuid", worker_uuid))


def _preheat_worker_item(task, worker_lookup=None, artifact_lookup=None):
    worker = _lookup_worker(worker_lookup or {}, task.worker_id, task.worker_uuid)
    artifact = (artifact_lookup or {}).get(task.distribution_artifact_id)
    return PolicyRunTaskPublic(
        id=task.id,
        model_id=artifact.model_id if artifact is not None else None,
        worker_uuid=task.worker_uuid,
        worker_name=worker.name if worker is not None else None,
        worker_ip=worker.ip if worker is not None else None,
        artifact_id=task.distribution_artifact_id,
        state=task.state.value,
        progress=task.progress,
        downloaded_bytes=task.downloaded_size,
        total_bytes=task.total_size,
        error_code=task.error_code,
        state_message=task.state_message,
    )


def _sync_item(task, worker_lookup=None):
    state = task.state.value
    progress = 100 if state == "ready" else 0
    worker = _lookup_worker(worker_lookup or {}, task.worker_id, task.worker_uuid)
    return PolicyRunTaskPublic(
        id=task.id,
        model_file_id=task.model_file_id,
        model_id=task.model_id,
        worker_id=task.worker_id,
        worker_uuid=task.worker_uuid,
        worker_name=worker.name if worker is not None else None,
        worker_ip=worker.ip if worker is not None else None,
        artifact_id=task.artifact_id,
        state=state,
        progress=progress,
        total_bytes=task.total_size,
        error_code=task.error_code,
        state_message=task.state_message,
    )


def _payload_items(payload, key):
    raw_items = payload.get(key, [])
    if not isinstance(raw_items, list):
        return []
    state = "skipped" if key == "skipped" else "error"
    items = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        reason = raw.get("reason")
        reason = reason if isinstance(reason, str) and reason else None
        task_id = raw.get("task_id") if isinstance(raw.get("task_id"), int) else None
        model_file_id = (
            raw.get("model_file_id")
            if isinstance(raw.get("model_file_id"), int)
            else None
        )
        worker_id = (
            raw.get("worker_id") if isinstance(raw.get("worker_id"), int) else None
        )
        worker_uuid = (
            raw.get("worker_uuid") if isinstance(raw.get("worker_uuid"), str) else None
        )
        if not any((task_id, model_file_id, worker_id, worker_uuid, reason)):
            continue
        items.append(
            PolicyRunTaskPublic(
                id=task_id,
                model_file_id=model_file_id,
                model_id=(
                    raw.get("model_id")
                    if isinstance(raw.get("model_id"), str)
                    else None
                ),
                worker_id=worker_id,
                worker_uuid=worker_uuid,
                worker_name=(
                    raw.get("worker_name")
                    if isinstance(raw.get("worker_name"), str)
                    else None
                ),
                worker_ip=(
                    raw.get("worker_ip")
                    if isinstance(raw.get("worker_ip"), str)
                    else None
                ),
                state=state,
                error_code=reason,
                state_message=reason,
            )
        )
    return items


def _enrich_payload_item(item, model_file_lookup, worker_lookup):
    model_file = model_file_lookup.get(item.model_file_id)
    if model_file is not None:
        item.model_id = item.model_id or _model_file_display_id(model_file)
        item.worker_id = item.worker_id or model_file.worker_id
    worker = _lookup_worker(worker_lookup, item.worker_id, item.worker_uuid)
    if worker is not None:
        item.worker_uuid = item.worker_uuid or worker.worker_uuid
        item.worker_name = item.worker_name or worker.name
        item.worker_ip = item.worker_ip or worker.ip


def _model_file_display_id(model_file):
    return (
        model_file.huggingface_repo_id
        or model_file.model_scope_model_id
        or model_file.ollama_library_model_name
        or model_file.local_path
    )


def _missing_sync_task_item(task_id):
    return PolicyRunTaskPublic(
        id=task_id,
        state="error",
        error_code="model_storage_sync_task_not_found",
        state_message="model_storage_sync_task_not_found",
    )


def _outcome_items(outcome):
    if not isinstance(outcome, dict):
        return []
    return _payload_items(outcome, "skipped") + _payload_items(outcome, "failed")


def _observation(items, stored_state, *, include_tasks):
    buckets = {
        key: 0 for key in ("pending", "running", "paused", "ready", "error", "skipped")
    }
    for item in items:
        buckets[_bucket(item.state)] += 1
    summary = PolicyRunSummary(
        total=len(items),
        **buckets,
        failed=buckets["error"],
        progress=(sum(item.progress for item in items) / len(items) if items else 0),
        downloaded_bytes=sum(item.downloaded_bytes for item in items),
        total_bytes=sum(item.total_bytes for item in items),
    )
    return PolicyRunObservation(
        execution_state=_execution_state(summary, stored_state),
        summary=summary,
        tasks=list(items) if include_tasks else [],
    )


def _apply_parent(observation, parent):
    if parent.execution_state == ModelPreheatExecutionStateEnum.PAUSED:
        observation.execution_state = PolicyRunExecutionStateEnum.PAUSED
    elif parent.execution_state == ModelPreheatExecutionStateEnum.PARTIAL:
        observation.execution_state = PolicyRunExecutionStateEnum.PARTIAL_ERROR
    elif parent.execution_state == ModelPreheatExecutionStateEnum.ERROR:
        observation.execution_state = PolicyRunExecutionStateEnum.ERROR
    elif parent.execution_state == ModelPreheatExecutionStateEnum.CANCELED:
        observation.execution_state = PolicyRunExecutionStateEnum.SKIPPED
    elif not observation.summary.total:
        observation.summary = PolicyRunSummary(
            total=1,
            pending=(
                1
                if parent.execution_state == ModelPreheatExecutionStateEnum.PENDING
                else 0
            ),
            running=(
                1
                if parent.execution_state
                not in {
                    ModelPreheatExecutionStateEnum.PENDING,
                    ModelPreheatExecutionStateEnum.READY,
                }
                else 0
            ),
            ready=(
                1
                if parent.execution_state == ModelPreheatExecutionStateEnum.READY
                else 0
            ),
            progress=parent.progress,
        )
        observation.execution_state = _execution_state(
            observation.summary, parent.execution_state
        )
    return observation


def _bucket(state):
    if state in {"pending"}:
        return "pending"
    if state in {
        "running",
        "scanning",
        "publishing",
        "resolving",
        "staging",
        "distributing",
    }:
        return "running"
    if state == "paused":
        return "paused"
    if state == "ready":
        return "ready"
    if state in {"canceled", "skipped_worker_removed", "skipped"}:
        return "skipped"
    return "error"


def _execution_state(summary, stored_state):
    value = getattr(stored_state, "value", stored_state)
    if summary.total:
        if summary.paused:
            return PolicyRunExecutionStateEnum.PAUSED
        if summary.running:
            return PolicyRunExecutionStateEnum.RUNNING
        if summary.pending:
            return PolicyRunExecutionStateEnum.WAITING
        if value == "error" and not summary.ready:
            return PolicyRunExecutionStateEnum.ERROR
        if summary.error:
            return (
                PolicyRunExecutionStateEnum.PARTIAL_ERROR
                if summary.ready or summary.skipped
                else PolicyRunExecutionStateEnum.ERROR
            )
        if value == "error":
            return PolicyRunExecutionStateEnum.ERROR
        if summary.ready:
            return PolicyRunExecutionStateEnum.READY
        return PolicyRunExecutionStateEnum.SKIPPED
    if value == "ready":
        return PolicyRunExecutionStateEnum.READY
    if value == "error":
        return PolicyRunExecutionStateEnum.ERROR
    if value == "paused":
        return PolicyRunExecutionStateEnum.PAUSED
    if value == "running":
        return PolicyRunExecutionStateEnum.RUNNING
    if value in {"skipped", "canceled"}:
        return PolicyRunExecutionStateEnum.SKIPPED
    return PolicyRunExecutionStateEnum.WAITING
