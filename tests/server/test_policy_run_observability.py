import asyncio
from types import SimpleNamespace

import pytest

from gpustack.schemas.model_preheats import ModelPreheatExecutionStateEnum
from gpustack.schemas.model_storage_sync import ModelStorageSyncTaskStateEnum
from gpustack.schemas.policy_runs import PolicyRunTaskPublic
from gpustack.server.policy_run_observability import (
    _apply_parent,
    _observation,
    _sync_item,
    sync_policy_run_observations,
)


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (["pending"], "waiting"),
        (["running"], "running"),
        (["paused"], "paused"),
        (["ready"], "ready"),
        (["ready", "error"], "partial_error"),
        (["error", "error"], "error"),
        (["skipped_worker_removed"], "skipped"),
    ],
)
def test_worker_task_states_project_to_run_execution_state(states, expected):
    items = [
        PolicyRunTaskPublic(id=index, state=state, progress=0)
        for index, state in enumerate(states, start=1)
    ]

    observation = _observation(items, "ready", include_tasks=False)

    assert observation.execution_state.value == expected
    assert observation.summary.total == len(states)
    assert observation.tasks == []


def test_historical_run_without_tasks_falls_back_to_stored_state():
    observation = _observation([], "error", include_tasks=True)

    assert observation.execution_state.value == "error"
    assert observation.summary.total == 0
    assert observation.tasks == []


def test_historical_skipped_run_without_tasks_remains_skipped():
    observation = _observation([], "skipped", include_tasks=False)

    assert observation.execution_state.value == "skipped"


def test_ready_sync_task_uses_percent_progress_without_inventing_download_bytes():
    item = _sync_item(
        SimpleNamespace(
            id=1,
            model_file_id=2,
            worker_id=3,
            worker_uuid="worker-a",
            artifact_id="artifact-a",
            state=ModelStorageSyncTaskStateEnum.READY,
            total_size=1024,
            error_code=None,
            state_message=None,
        )
    )

    assert item.progress == 100
    assert item.total_bytes == 1024
    assert item.downloaded_bytes == 0


def test_malformed_legacy_sync_payload_is_ignored_without_crashing():
    run = SimpleNamespace(
        id=1,
        state="ready",
        response_payload={
            "created": "not-a-list",
            "skipped": None,
            "failed": [None, {"reason": 42}],
        },
    )

    observations = asyncio.run(
        sync_policy_run_observations(None, [run], include_tasks=True)
    )

    assert observations[1].execution_state.value == "ready"
    assert observations[1].summary.total == 0
    assert observations[1].tasks == []


@pytest.mark.parametrize(
    ("parent_state", "expected"),
    [
        (ModelPreheatExecutionStateEnum.PAUSED, "paused"),
        (ModelPreheatExecutionStateEnum.PARTIAL, "partial_error"),
        (ModelPreheatExecutionStateEnum.ERROR, "error"),
        (ModelPreheatExecutionStateEnum.CANCELED, "skipped"),
    ],
)
def test_preheat_parent_terminal_state_overrides_worker_summary(parent_state, expected):
    observation = _observation(
        [PolicyRunTaskPublic(id=1, state="ready", progress=100)],
        "ready",
        include_tasks=False,
    )

    projected = _apply_parent(
        observation,
        SimpleNamespace(execution_state=parent_state, progress=100),
    )

    assert projected.execution_state.value == expected
