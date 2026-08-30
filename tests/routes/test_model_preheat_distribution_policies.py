import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import event
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.schema import CreateTable
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.auth import get_admin_user
from gpustack.routes import model_preheat_distribution_policies
from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
    ModelPreheatDistributionPolicyRun,
    ModelPreheatDistributionPolicyRunStateEnum,
    ModelPreheatDistributionPolicyRunTask,
    ModelPreheatDistributionPolicyRunTriggerEnum,
    ModelPreheatDistributionPolicyArtifact,
    ModelPreheatWorkerObservation,
)
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.model_storage_sync import (
    ModelStorageSyncTask,
    ModelStorageSyncTaskStateEnum,
)
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatInventoryManifestStateEnum,
    ModelPreheatTask,
    ModelPreheatTargetScopeEnum,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_session
from gpustack.server.model_preheat_distribution_source import (
    resolve_distribution_sources,
)


def _test_app(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'policies.db'}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
    event.listen(
        engine.sync_engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    asyncio.run(_create_tables(engine))
    app = FastAPI()
    app.state.model_preheat_worker_reconciler = SimpleNamespace(
        reconcile_policy=lambda policy_id: _done(policy_id)
    )

    async def session_override():
        async with AsyncSession(engine) as session:
            yield session

    async def admin_override():
        return User(id=1, username="admin", is_admin=True, hashed_password="")

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_admin_user] = admin_override
    router = APIRouter(dependencies=[Depends(get_admin_user)])
    router.include_router(
        model_preheat_distribution_policies.router,
        prefix="/model-preheat-distribution-policies",
    )
    app.include_router(router, prefix="/v1")
    exceptions.register_handlers(app)
    return app, engine


async def _create_tables(engine):
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)


async def _done(value):
    return value


async def _seed(engine):
    async with AsyncSession(engine) as session:
        profile = ModelPreheatS3Profile(
            name="profile",
            endpoint="https://s3.example.com",
            bucket="models",
            access_key_encrypted={"ciphertext": "access-secret"},
            secret_key_encrypted={"ciphertext": "secret-secret"},
            encryption_key_version="v1",
        )
        session.add(profile)
        await session.flush()
        policy = ModelPreheatDistributionPolicy(
            name="模型同步",
            profile_id=profile.id,
            profile_config_version=profile.config_version,
            request_identity={
                "source": "huggingface",
                "model_id": "org/model",
                "requested_revision": "main",
                "include_patterns": [],
                "exclude_patterns": [],
            },
            request_digest="c" * 64,
            target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
            worker_selector={"worker_uuids": ["worker-a"]},
            gpu_selector={},
            selector_digest="d" * 64,
            created_by_task_id=None,
            last_reconciled_at=datetime.now(timezone.utc),
        )
        session.add(policy)
        await session.commit()
        await session.refresh(policy)
        return policy.id


def test_delete_distribution_policy_unlinks_terminal_worker_tasks(tmp_path):
    app, engine = _test_app(tmp_path)
    policy_id = asyncio.run(_seed(engine))

    async def seed_terminal_task_and_run_link():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            run = ModelPreheatDistributionPolicyRun(
                policy_id=policy_id,
                trigger=ModelPreheatDistributionPolicyRunTriggerEnum.MANUAL,
                state=ModelPreheatDistributionPolicyRunStateEnum.READY,
                window_start_utc=datetime.now(timezone.utc),
                operation_key="terminal-delete-run",
            )
            task = ModelPreheatWorkerTask(
                distribution_policy_id=policy_id,
                operation_key="terminal-delete-task",
                worker_uuid="worker-terminal",
                role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                state=ModelPreheatWorkerTaskStateEnum.READY,
                lease_owner="worker-terminal",
                lease_token_hash="terminal-lease-token",
            )
            session.add_all([run, task])
            await session.flush()
            session.add(
                ModelPreheatDistributionPolicyRunTask(run_id=run.id, task_id=task.id)
            )
            await session.commit()
            return run.id, task.id

    run_id, task_id = asyncio.run(seed_terminal_task_and_run_link())
    with TestClient(app) as client:
        response = client.delete(f"/v1/model-preheat-distribution-policies/{policy_id}")

    async def assert_deleted_and_unlinked():
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatWorkerTask, task_id)
            assert await session.get(ModelPreheatDistributionPolicy, policy_id) is None
            assert await session.get(ModelPreheatDistributionPolicyRun, run_id) is None
            assert task is not None
            assert task.distribution_policy_id is None
            assert task.lease_owner == "worker-terminal"
            assert task.lease_token_hash == "terminal-lease-token"
            assert (
                await session.get(
                    ModelPreheatDistributionPolicyRunTask,
                    (run_id, task_id),
                )
            ) is None

    asyncio.run(assert_deleted_and_unlinked())
    asyncio.run(engine.dispose())

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True}


@pytest.mark.parametrize(
    "state",
    [
        ModelPreheatWorkerTaskStateEnum.PENDING,
        ModelPreheatWorkerTaskStateEnum.RUNNING,
        ModelPreheatWorkerTaskStateEnum.PAUSED,
    ],
)
def test_delete_distribution_policy_rejects_active_worker_tasks(tmp_path, state):
    app, engine = _test_app(tmp_path)
    policy_id = asyncio.run(_seed(engine))

    async def seed_active_task():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            task = ModelPreheatWorkerTask(
                distribution_policy_id=policy_id,
                operation_key=f"active-delete-task-{state.value}",
                worker_uuid=f"worker-{state.value}",
                role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                state=state,
            )
            session.add(task)
            await session.commit()
            return task.id

    task_id = asyncio.run(seed_active_task())
    with TestClient(app) as client:
        response = client.delete(f"/v1/model-preheat-distribution-policies/{policy_id}")

    async def assert_policy_still_linked():
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatWorkerTask, task_id)
            assert (
                await session.get(ModelPreheatDistributionPolicy, policy_id) is not None
            )
            assert task is not None
            assert task.distribution_policy_id == policy_id

    asyncio.run(assert_policy_still_linked())
    asyncio.run(engine.dispose())

    assert response.status_code == 409, response.text
    assert "distribution_policy_in_use" in response.text


def test_distribution_policy_runs_list_and_detail_are_public(tmp_path):
    app, engine = _test_app(tmp_path)
    policy_id = asyncio.run(_seed(engine))

    async def seed_run():
        async with AsyncSession(engine) as session:
            run = ModelPreheatDistributionPolicyRun(
                policy_id=policy_id,
                trigger=ModelPreheatDistributionPolicyRunTriggerEnum.MANUAL,
                state=ModelPreheatDistributionPolicyRunStateEnum.ERROR,
                window_start_utc=datetime.now(timezone.utc),
                operation_key="run-public-test",
                error_code="worker_execution_failed",
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return run.id

    run_id = asyncio.run(seed_run())
    with TestClient(app) as client:
        listed = client.get("/v1/model-preheat-distribution-policies/runs")
        detail = client.get(f"/v1/model-preheat-distribution-policies/runs/{run_id}")
    asyncio.run(engine.dispose())

    assert listed.status_code == 200, listed.text
    assert listed.json()["items"][0]["id"] == run_id
    assert detail.status_code == 200, detail.text
    assert detail.json()["policy_id"] == policy_id
    assert detail.json()["policy_name"] == "模型同步"
    assert detail.json()["model_id"] == "org/model"
    assert detail.json()["error_code"] == "worker_execution_failed"


def test_distribution_run_uses_linked_worker_tasks_and_policy_exposes_latest_summary(
    tmp_path,
):
    app, engine = _test_app(tmp_path)
    policy_id = asyncio.run(_seed(engine))

    async def seed_run_and_tasks():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            run = ModelPreheatDistributionPolicyRun(
                policy_id=policy_id,
                trigger=ModelPreheatDistributionPolicyRunTriggerEnum.MANUAL,
                state=ModelPreheatDistributionPolicyRunStateEnum.READY,
                window_start_utc=datetime.now(timezone.utc),
                operation_key="observable-distribution-run",
            )
            session.add(run)
            await session.flush()
            tasks = [
                ModelPreheatWorkerTask(
                    distribution_policy_id=policy_id,
                    operation_key=f"observable-task-{index}",
                    worker_uuid=f"worker-{index}",
                    role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                    state=ModelPreheatWorkerTaskStateEnum.PENDING,
                    lease_token_hash=f"private-lease-{index}",
                )
                for index in range(2)
            ]
            session.add_all(tasks)
            await session.flush()
            session.add_all(
                [
                    ModelPreheatDistributionPolicyRunTask(
                        run_id=run.id, task_id=task.id
                    )
                    for task in tasks
                ]
            )
            await session.commit()
            return run.id, [task.id for task in tasks]

    run_id, task_ids = asyncio.run(seed_run_and_tasks())
    with TestClient(app) as client:
        policies = client.get("/v1/model-preheat-distribution-policies")
        listed = client.get("/v1/model-preheat-distribution-policies/runs")

    assert policies.status_code == 200, policies.text
    latest = policies.json()["items"][0]["latest_run"]
    assert latest["id"] == run_id
    assert latest["state"] == "ready"
    assert latest["execution_state"] == "waiting"
    assert latest["summary"]["pending"] == 2
    assert latest["tasks"] == []
    assert listed.json()["items"][0]["execution_state"] == "waiting"

    async def finish_with_partial_error():
        async with AsyncSession(engine) as session:
            first = await session.get(ModelPreheatWorkerTask, task_ids[0])
            second = await session.get(ModelPreheatWorkerTask, task_ids[1])
            first.state = ModelPreheatWorkerTaskStateEnum.READY
            first.progress = 100
            first.downloaded_size = 8
            first.total_size = 8
            second.state = ModelPreheatWorkerTaskStateEnum.ERROR
            second.error_code = "worker_download_failed"
            second.state_message = "download failed"
            session.add(first)
            session.add(second)
            await session.commit()

    asyncio.run(finish_with_partial_error())
    with TestClient(app) as client:
        detail = client.get(f"/v1/model-preheat-distribution-policies/runs/{run_id}")
    asyncio.run(engine.dispose())

    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["execution_state"] == "partial_error"
    assert body["summary"]["ready"] == 1
    assert body["summary"]["error"] == 1
    assert body["summary"]["progress"] == 50
    assert body["summary"]["downloaded_bytes"] == 8
    assert [task["id"] for task in body["tasks"]] == task_ids
    assert body["tasks"][1]["error_code"] == "worker_download_failed"
    assert "lease" not in str(body).lower()


async def _seed_artifact(
    engine, *, suffix="", artifact_id="a" * 64, model_id="org/model"
):
    async with AsyncSession(engine, expire_on_commit=False) as session:
        profile = ModelPreheatS3Profile(
            name=f"artifact-profile{suffix}",
            endpoint=f"https://s3{suffix}.example.com",
            bucket="models",
            access_key_encrypted={"ciphertext": "access-secret"},
            secret_key_encrypted={"ciphertext": "secret-secret"},
            encryption_key_version="v1",
        )
        session.add(profile)
        await session.flush()
        artifact = ModelPreheatArtifact(
            profile_id=profile.id,
            profile_config_version=profile.config_version,
            artifact_id=artifact_id,
            source="huggingface",
            model_id=model_id,
            resolved_revision="commit-1",
            include_patterns=[],
            exclude_patterns=[],
            manifest_path="models/huggingface/org/model/manifest.json",
            manifest_digest="b" * 64,
            file_count=1,
            total_size=10,
            manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
            last_verified_at=datetime.now(timezone.utc),
        )
        session.add(artifact)
        await session.commit()
        return profile.id, artifact.artifact_id


async def _seed_additional_artifact(
    engine, profile_id, artifact_id, *, model_id="org/second"
):
    async with AsyncSession(engine) as session:
        profile = await session.get(ModelPreheatS3Profile, profile_id)
        artifact = ModelPreheatArtifact(
            profile_id=profile_id,
            profile_config_version=profile.config_version,
            artifact_id=artifact_id,
            source="modelscope",
            model_id=model_id,
            resolved_revision="commit-2",
            include_patterns=[],
            exclude_patterns=[],
            manifest_path=f"models/modelscope/{model_id}/manifest.json",
            manifest_digest="f" * 64,
            file_count=1,
            total_size=20,
            manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
            last_verified_at=datetime.now(timezone.utc),
        )
        session.add(artifact)
        await session.commit()


def _create_fixed_policy(client, profile_id, artifact_id, worker_uuid="worker-a"):
    response = client.post(
        "/v1/model-preheat-distribution-policies",
        json={
            "name": f"distribution-{worker_uuid}",
            "profile_id": profile_id,
            "artifact_id": artifact_id,
            "target_scope": "selected_workers",
            "worker_selector": {"worker_uuids": [worker_uuid]},
            "gpu_selector": {},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_policy_routes_list_get_disable_and_reconcile_without_credentials(tmp_path):
    app, engine = _test_app(tmp_path)
    policy_id = asyncio.run(_seed(engine))
    with TestClient(app) as client:
        listed = client.get("/v1/model-preheat-distribution-policies")
        fetched = client.get(f"/v1/model-preheat-distribution-policies/{policy_id}")
        disabled = client.patch(
            f"/v1/model-preheat-distribution-policies/{policy_id}",
            json={"enabled": False},
        )
        reconciled = client.post(
            f"/v1/model-preheat-distribution-policies/{policy_id}/reconcile"
        )
    asyncio.run(engine.dispose())

    assert listed.status_code == 200
    assert fetched.status_code == 200
    assert disabled.status_code == 200
    assert disabled.json()["enabled"] is False
    assert fetched.json()["profile_config_version"] == 1
    assert reconciled.status_code == 200
    payload = str([listed.json(), fetched.json(), disabled.json()])
    assert "access-secret" not in payload
    assert "secret-secret" not in payload
    assert "snapshot_encrypted" not in payload


def test_policy_patch_rejects_selector_or_credential_mutation(tmp_path):
    app, engine = _test_app(tmp_path)
    policy_id = asyncio.run(_seed(engine))
    with TestClient(app) as client:
        selector = client.patch(
            f"/v1/model-preheat-distribution-policies/{policy_id}",
            json={"worker_selector": {"worker_uuids": ["other"]}},
        )
        credential = client.patch(
            f"/v1/model-preheat-distribution-policies/{policy_id}",
            json={"access_key": "credential-shaped-value"},
        )
    asyncio.run(engine.dispose())

    assert selector.status_code == 422
    assert credential.status_code == 422


def test_policy_can_be_created_from_existing_s3_artifact(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))
    with TestClient(app) as client:
        created = client.post(
            "/v1/model-preheat-distribution-policies",
            json={
                "name": "artifact-distribution",
                "profile_id": profile_id,
                "artifact_id": artifact_id,
                "target_scope": "selected_workers",
                "worker_selector": {"worker_uuids": ["worker-a"]},
                "gpu_selector": {},
            },
        )
    asyncio.run(engine.dispose())

    assert created.status_code == 200, created.text
    assert created.json()["source_artifact"] == artifact_id
    assert created.json()["source_artifact_id"] is not None
    assert created.json()["source_sync_task_id"] is None
    assert created.json()["request_identity"]["source"] == "huggingface"
    assert created.json()["trigger_mode"] == "manual"


def test_policy_can_select_multiple_artifacts_in_one_policy(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))

    async def seed_second():
        async with AsyncSession(engine) as session:
            artifact = ModelPreheatArtifact(
                profile_id=profile_id,
                profile_config_version=1,
                artifact_id="e" * 64,
                source="modelscope",
                model_id="org/second",
                resolved_revision="commit-2",
                include_patterns=[],
                exclude_patterns=[],
                manifest_path="models/modelscope/org/second/manifest.json",
                manifest_digest="f" * 64,
                file_count=1,
                total_size=20,
                manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
                last_verified_at=datetime.now(timezone.utc),
            )
            session.add(artifact)
            await session.commit()

    asyncio.run(seed_second())
    with TestClient(app) as client:
        created = client.post(
            "/v1/model-preheat-distribution-policies",
            json={
                "name": "selected-artifacts",
                "profile_id": profile_id,
                "selection_mode": "selected",
                "artifact_ids": [artifact_id, "e" * 64],
                "target_scope": "selected_workers",
                "worker_selector": {"worker_uuids": ["worker-a"]},
                "gpu_selector": {},
            },
        )
    assert created.status_code == 200, created.text
    assert created.json()["selection_mode"] == "selected"
    assert created.json()["artifact_ids"] == [artifact_id, "e" * 64]

    async def association_count():
        async with AsyncSession(engine) as session:
            return len(
                (
                    await session.exec(select(ModelPreheatDistributionPolicyArtifact))
                ).all()
            )

    assert asyncio.run(association_count()) == 2
    asyncio.run(engine.dispose())


def test_unexecuted_policy_can_switch_fixed_selected_and_all_current(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))
    second_artifact_id = "e" * 64
    asyncio.run(_seed_additional_artifact(engine, profile_id, second_artifact_id))
    with TestClient(app) as client:
        created = _create_fixed_policy(client, profile_id, artifact_id)
        selected = client.patch(
            f"/v1/model-preheat-distribution-policies/{created['id']}",
            json={
                "selection_mode": "selected",
                "artifact_ids": [artifact_id, second_artifact_id],
                "target_scope": "same_gpu_model",
                "worker_selector": {},
                "gpu_selector": {"gpu_names": ["NVIDIA A100"]},
            },
        )
        all_current = client.patch(
            f"/v1/model-preheat-distribution-policies/{created['id']}",
            json={"selection_mode": "all_current"},
        )
    asyncio.run(engine.dispose())

    assert selected.status_code == 200, selected.text
    selected_body = selected.json()
    assert selected_body["structural_editable"] is True
    assert selected_body["selection_mode"] == "selected"
    assert selected_body["source_artifact_id"] is None
    assert selected_body["artifact_ids"] == [artifact_id, second_artifact_id]
    assert selected_body["target_scope"] == "same_gpu_model"
    assert selected_body["worker_selector"] == {}
    assert selected_body["gpu_selector"] == {"gpu_names": ["NVIDIA A100"]}
    assert selected_body["request_identity"]["selection_mode"] == "selected"
    assert selected_body["request_digest"] != created["request_digest"]
    assert all_current.status_code == 200, all_current.text
    assert all_current.json()["selection_mode"] == "all_current"
    assert all_current.json()["artifact_ids"] == []


def test_unexecuted_policy_can_change_profile_and_fixed_artifact(tmp_path):
    app, engine = _test_app(tmp_path)
    first_profile_id, first_artifact_id = asyncio.run(_seed_artifact(engine))
    second_profile_id, second_artifact_id = asyncio.run(
        _seed_artifact(
            engine,
            suffix="-two",
            artifact_id="9" * 64,
            model_id="org/profile-two",
        )
    )
    with TestClient(app) as client:
        created = _create_fixed_policy(client, first_profile_id, first_artifact_id)
        updated = client.patch(
            f"/v1/model-preheat-distribution-policies/{created['id']}",
            json={
                "profile_id": second_profile_id,
                "selection_mode": "fixed",
                "artifact_id": second_artifact_id,
            },
        )
    asyncio.run(engine.dispose())

    assert updated.status_code == 200, updated.text
    body = updated.json()
    assert body["profile_id"] == second_profile_id
    assert body["source_artifact"] == second_artifact_id
    assert body["source_sync_task_id"] is None
    assert body["request_identity"]["model_id"] == "org/profile-two"


def test_executed_policy_rejects_structural_change_but_allows_basic_fields(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))
    with TestClient(app) as client:
        created = _create_fixed_policy(client, profile_id, artifact_id)

    async def add_run():
        async with AsyncSession(engine) as session:
            session.add(
                ModelPreheatDistributionPolicyRun(
                    policy_id=created["id"],
                    trigger=ModelPreheatDistributionPolicyRunTriggerEnum.MANUAL,
                    state=ModelPreheatDistributionPolicyRunStateEnum.READY,
                    window_start_utc=datetime.now(timezone.utc),
                    operation_key="executed-policy-run",
                )
            )
            await session.commit()

    asyncio.run(add_run())
    with TestClient(app) as client:
        structural = client.patch(
            f"/v1/model-preheat-distribution-policies/{created['id']}",
            json={"worker_selector": {"worker_uuids": ["worker-b"]}},
        )
        basic = client.patch(
            f"/v1/model-preheat-distribution-policies/{created['id']}",
            json={"name": "renamed", "enabled": False},
        )
        listed = client.get("/v1/model-preheat-distribution-policies")
    asyncio.run(engine.dispose())

    assert structural.status_code == 409
    assert structural.json()["message"] == "distribution_policy_already_executed"
    assert basic.status_code == 200, basic.text
    assert basic.json()["name"] == "renamed"
    assert basic.json()["enabled"] is False
    assert basic.json()["structural_editable"] is False
    assert listed.json()["items"][0]["structural_editable"] is False


def test_historical_worker_task_without_run_freezes_structural_fields(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))
    with TestClient(app) as client:
        created = _create_fixed_policy(client, profile_id, artifact_id)

    async def add_worker_task():
        async with AsyncSession(engine) as session:
            session.add(
                ModelPreheatWorkerTask(
                    distribution_policy_id=created["id"],
                    operation_key="historical-distribution-task",
                    worker_uuid="worker-a",
                    role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                )
            )
            await session.commit()

    asyncio.run(add_worker_task())
    with TestClient(app) as client:
        updated = client.patch(
            f"/v1/model-preheat-distribution-policies/{created['id']}",
            json={"target_scope": "same_gpu_model"},
        )
        fetched = client.get(f"/v1/model-preheat-distribution-policies/{created['id']}")
    asyncio.run(engine.dispose())

    assert updated.status_code == 409
    assert updated.json()["message"] == "distribution_policy_already_executed"
    assert fetched.json()["structural_editable"] is False


def test_invalid_artifact_and_unique_conflict_roll_back_structural_patch(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))
    with TestClient(app) as client:
        first = _create_fixed_policy(client, profile_id, artifact_id, "worker-a")
        second = _create_fixed_policy(client, profile_id, artifact_id, "worker-b")
        invalid = client.patch(
            f"/v1/model-preheat-distribution-policies/{second['id']}",
            json={"artifact_id": "0" * 64},
        )
        conflict = client.patch(
            f"/v1/model-preheat-distribution-policies/{second['id']}",
            json={"worker_selector": first["worker_selector"]},
        )
        fetched = client.get(f"/v1/model-preheat-distribution-policies/{second['id']}")
    asyncio.run(engine.dispose())

    assert invalid.status_code == 409
    assert invalid.json()["message"] == "artifact_not_ready"
    assert conflict.status_code == 409
    assert conflict.json()["message"] == "distribution_policy_conflict"
    assert fetched.status_code == 200
    assert fetched.json()["worker_selector"] == {"worker_uuids": ["worker-b"]}
    assert fetched.json()["source_artifact"] == artifact_id


def test_structural_patch_rejects_duplicate_selected_artifacts(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))
    with TestClient(app) as client:
        created = _create_fixed_policy(client, profile_id, artifact_id)
        updated = client.patch(
            f"/v1/model-preheat-distribution-policies/{created['id']}",
            json={
                "selection_mode": "selected",
                "artifact_ids": [artifact_id, artifact_id],
            },
        )
    asyncio.run(engine.dispose())

    assert updated.status_code == 422
    assert "duplicate_distribution_artifact" in updated.json()["message"]


def test_selector_only_edit_clears_source_sync_provenance_and_resolves_artifact(
    tmp_path,
):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))
    with TestClient(app) as client:
        created = _create_fixed_policy(client, profile_id, artifact_id)

    async def bind_sync_task():
        async with AsyncSession(engine) as session:
            worker = Worker(
                name="sync-worker",
                hostname="sync-worker",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="sync-worker-uuid",
            )
            session.add(worker)
            await session.flush()
            model_file = ModelFile(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="org/model",
                worker_id=worker.id,
                state=ModelFileStateEnum.READY,
            )
            session.add(model_file)
            await session.flush()
            sync_task = ModelStorageSyncTask(
                model_file_id=model_file.id,
                worker_id=worker.id,
                worker_uuid=worker.worker_uuid,
                profile_id=profile_id,
                profile_config_version=1,
                request_identity={"source": "modelscope", "model_id": "org/model"},
                request_digest="7" * 64,
                source="modelscope",
                model_id="org/model",
                resolved_revision="commit-1",
                credential_snapshot_encrypted={"ciphertext": "encrypted"},
                encryption_key_version="v1",
                artifact_id=artifact_id,
                state=ModelStorageSyncTaskStateEnum.READY,
            )
            session.add(sync_task)
            await session.flush()
            policy = await session.get(ModelPreheatDistributionPolicy, created["id"])
            policy.source_sync_task_id = sync_task.id
            session.add(policy)
            await session.commit()

    asyncio.run(bind_sync_task())
    with TestClient(app) as client:
        updated = client.patch(
            f"/v1/model-preheat-distribution-policies/{created['id']}",
            json={
                "worker_selector": {"worker_uuids": ["worker-b"]},
            },
        )

    async def resolve_updated():
        async with AsyncSession(engine) as session:
            policy = await session.get(ModelPreheatDistributionPolicy, created["id"])
            sources = await resolve_distribution_sources(session, policy)
            return policy.source_sync_task_id, policy.created_by_task_id, sources

    source_sync_task_id, created_by_task_id, sources = asyncio.run(resolve_updated())
    asyncio.run(engine.dispose())

    assert updated.status_code == 200, updated.text
    assert updated.json()["source_sync_task_id"] is None
    assert source_sync_task_id is None
    assert created_by_task_id is None
    assert [source.artifact.artifact_id for source in sources] == [artifact_id]


def test_artifact_selection_edit_clears_created_by_task_and_resolves_artifacts(
    tmp_path,
):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))
    second_artifact_id = "8" * 64
    asyncio.run(_seed_additional_artifact(engine, profile_id, second_artifact_id))
    with TestClient(app) as client:
        created = _create_fixed_policy(client, profile_id, artifact_id)

    async def bind_preheat_task():
        async with AsyncSession(engine) as session:
            worker = Worker(
                name="preheat-worker",
                hostname="preheat-worker",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="preheat-worker-uuid",
            )
            session.add(worker)
            await session.flush()
            task = ModelPreheatTask(
                source="huggingface",
                model_id="org/model",
                resolved_revision="commit-1",
                include_patterns=[],
                exclude_patterns=[],
                selection_digest="6" * 64,
                request_identity={"source": "huggingface", "model_id": "org/model"},
                request_digest="5" * 64,
                execution_state=ModelPreheatExecutionStateEnum.READY,
                artifact_id=artifact_id,
                seed_worker_uuid=worker.worker_uuid,
                seed_worker_id=worker.id,
                target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                target_worker_uuids=[worker.worker_uuid],
                target_worker_snapshot=[],
                s3_profile_id=profile_id,
                s3_profile_config_version=1,
                s3_profile_snapshot_encrypted={"ciphertext": "encrypted"},
                encryption_key_version="v1",
                s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
                s3_manifest_path="models/huggingface/org/model/manifest.json",
            )
            session.add(task)
            await session.flush()
            policy = await session.get(ModelPreheatDistributionPolicy, created["id"])
            policy.created_by_task_id = task.id
            session.add(policy)
            await session.commit()

    asyncio.run(bind_preheat_task())
    with TestClient(app) as client:
        updated = client.patch(
            f"/v1/model-preheat-distribution-policies/{created['id']}",
            json={
                "selection_mode": "selected",
                "artifact_ids": [artifact_id, second_artifact_id],
            },
        )

    async def resolve_updated():
        async with AsyncSession(engine) as session:
            policy = await session.get(ModelPreheatDistributionPolicy, created["id"])
            sources = await resolve_distribution_sources(session, policy)
            return policy.source_sync_task_id, policy.created_by_task_id, sources

    source_sync_task_id, created_by_task_id, sources = asyncio.run(resolve_updated())
    asyncio.run(engine.dispose())

    assert updated.status_code == 200, updated.text
    assert source_sync_task_id is None
    assert created_by_task_id is None
    assert [source.artifact.artifact_id for source in sources] == [
        artifact_id,
        second_artifact_id,
    ]


def test_all_current_policy_has_no_fixed_artifact_binding(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, _ = asyncio.run(_seed_artifact(engine))
    with TestClient(app) as client:
        created = client.post(
            "/v1/model-preheat-distribution-policies",
            json={
                "name": "all-current",
                "profile_id": profile_id,
                "selection_mode": "all_current",
                "target_scope": "selected_workers",
                "worker_selector": {"worker_uuids": ["worker-a"]},
                "gpu_selector": {},
            },
        )
    asyncio.run(engine.dispose())

    assert created.status_code == 200, created.text
    assert created.json()["selection_mode"] == "all_current"
    assert created.json()["source_artifact_id"] is None
    assert created.json()["artifact_ids"] == []


def test_policy_exposes_blocked_reason_for_stale_fixed_artifact(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))

    async def mark_stale():
        async with AsyncSession(engine) as session:
            artifact = (
                await session.exec(
                    select(ModelPreheatArtifact).where(
                        ModelPreheatArtifact.profile_id == profile_id,
                        ModelPreheatArtifact.artifact_id == artifact_id,
                    )
                )
            ).one()
            artifact.manifest_state = ModelPreheatInventoryManifestStateEnum.STALE
            session.add(artifact)
            await session.commit()

    asyncio.run(mark_stale())
    with TestClient(app) as client:
        created = client.post(
            "/v1/model-preheat-distribution-policies",
            json={
                "name": "stale-artifact",
                "profile_id": profile_id,
                "artifact_id": artifact_id,
                "target_scope": "selected_workers",
                "worker_selector": {"worker_uuids": ["worker-a"]},
                "gpu_selector": {},
            },
        )
    asyncio.run(engine.dispose())

    assert created.status_code == 409
    assert created.json()["message"] == "artifact_stale"


def test_reenable_rejects_stale_fixed_artifact(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))
    with TestClient(app) as client:
        created = client.post(
            "/v1/model-preheat-distribution-policies",
            json={
                "name": "fixed-artifact",
                "profile_id": profile_id,
                "artifact_id": artifact_id,
                "target_scope": "selected_workers",
                "worker_selector": {"worker_uuids": ["worker-a"]},
                "gpu_selector": {},
            },
        )
        policy_id = created.json()["id"]
        assert (
            client.patch(
                f"/v1/model-preheat-distribution-policies/{policy_id}",
                json={"enabled": False},
            ).status_code
            == 200
        )

    async def mark_stale():
        async with AsyncSession(engine) as session:
            artifact = (
                await session.exec(
                    select(ModelPreheatArtifact).where(
                        ModelPreheatArtifact.profile_id == profile_id,
                        ModelPreheatArtifact.artifact_id == artifact_id,
                    )
                )
            ).one()
            artifact.manifest_state = ModelPreheatInventoryManifestStateEnum.STALE
            session.add(artifact)
            await session.commit()

    asyncio.run(mark_stale())
    with TestClient(app) as client:
        enabled = client.patch(
            f"/v1/model-preheat-distribution-policies/{policy_id}",
            json={"enabled": True},
        )
    asyncio.run(engine.dispose())

    assert enabled.status_code == 409
    assert enabled.json()["message"] == "artifact_stale"


def test_reenable_rejects_profile_config_version_drift(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))
    with TestClient(app) as client:
        created = client.post(
            "/v1/model-preheat-distribution-policies",
            json={
                "name": "version-bound-artifact",
                "profile_id": profile_id,
                "artifact_id": artifact_id,
                "target_scope": "selected_workers",
                "worker_selector": {"worker_uuids": ["worker-a"]},
                "gpu_selector": {},
            },
        )
        policy_id = created.json()["id"]
        assert (
            client.patch(
                f"/v1/model-preheat-distribution-policies/{policy_id}",
                json={"enabled": False},
            ).status_code
            == 200
        )

    async def rotate_profile():
        async with AsyncSession(engine) as session:
            profile = await session.get(ModelPreheatS3Profile, profile_id)
            profile.config_version += 1
            session.add(profile)
            await session.commit()

    asyncio.run(rotate_profile())
    with TestClient(app) as client:
        enabled = client.patch(
            f"/v1/model-preheat-distribution-policies/{policy_id}",
            json={"enabled": True},
        )
    asyncio.run(engine.dispose())

    assert enabled.status_code == 409
    assert enabled.json()["message"] == "distribution_profile_version_stale"


def test_repeated_disable_preserves_profile_version_stale(tmp_path):
    app, engine = _test_app(tmp_path)
    policy_id = asyncio.run(_seed(engine))

    async def mark_stale():
        async with AsyncSession(engine) as session:
            policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
            policy.enabled = False
            policy.profile_version_stale = True
            session.add(policy)
            await session.commit()

    asyncio.run(mark_stale())
    with TestClient(app) as client:
        disabled = client.patch(
            f"/v1/model-preheat-distribution-policies/{policy_id}",
            json={"enabled": False},
        )

    async def read_policy():
        async with AsyncSession(engine) as session:
            return await session.get(ModelPreheatDistributionPolicy, policy_id)

    policy = asyncio.run(read_policy())
    asyncio.run(engine.dispose())

    assert disabled.status_code == 200
    assert policy.profile_version_stale is True


def test_policy_schema_and_successor_migration_are_portable():
    migration = (
        "gpustack/migrations/versions/"
        "2026_08_11_1500-b8c9d0e1f2a3_add_preheat_distribution_policies.py"
    )
    source = Path(migration).read_text(encoding="utf-8")
    assert 'down_revision: Union[str, None] = "a7b8c9d0e1f2"' in source
    assert "postgresql_where" not in source
    assert "CREATE UNIQUE INDEX" not in source
    for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
        for table in (
            ModelPreheatDistributionPolicy.__table__,
            ModelPreheatWorkerObservation.__table__,
        ):
            ddl = str(CreateTable(table).compile(dialect=dialect))
            assert table.name in ddl
