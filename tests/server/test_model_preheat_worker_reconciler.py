import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
    ModelPreheatDistributionPolicyRun,
    ModelPreheatDistributionPolicyRunStateEnum,
    ModelPreheatDistributionPolicyRunTask,
    ModelPreheatDistributionPolicyRunTriggerEnum,
    ModelPreheatDistributionPolicyTriggerModeEnum,
    ModelPreheatDistributionSelectionModeEnum,
    ModelPreheatDistributionWorkerSlot,
    ModelPreheatWorkerObservation,
    distribution_operation_key,
)
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatInventoryManifestStateEnum,
    ModelPreheatS3ConnectivityCheck,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.bus import Event, EventType
from gpustack.server.model_preheat_worker_reconciler import (
    ModelPreheatWorkerReconciler,
    _create_or_rebind_distribution_task,
)
from gpustack.server.model_preheat_distribution_source import (
    resolve_distribution_source,
    resolve_distribution_sources,
)


ARTIFACT_ID = "a" * 64


def test_all_current_resolves_only_current_valid_artifacts_and_freezes_tasks(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = ModelPreheatS3Profile(
                name="all-current-profile",
                endpoint="https://s3.example.com",
                bucket="models",
                access_key_encrypted={"ciphertext": "access"},
                secret_key_encrypted={"ciphertext": "secret"},
                encryption_key_version="v1",
            )
            worker = Worker(
                name="worker",
                hostname="worker",
                ip="10.0.0.2",
                port=10150,
                worker_uuid="worker-all",
                state=WorkerStateEnum.READY,
                model_storage_protocol_version=1,
            )
            session.add(profile)
            session.add(worker)
            await session.flush()
            for index, state in enumerate(
                [
                    ModelPreheatInventoryManifestStateEnum.VALID,
                    ModelPreheatInventoryManifestStateEnum.VALID,
                    ModelPreheatInventoryManifestStateEnum.STALE,
                ]
            ):
                session.add(
                    ModelPreheatArtifact(
                        profile_id=profile.id,
                        profile_config_version=profile.config_version,
                        artifact_id=str(index + 1) * 64,
                        source="huggingface",
                        model_id=f"org/model-{index}",
                        resolved_revision=f"commit-{index}",
                        include_patterns=[],
                        exclude_patterns=[],
                        manifest_path=f"models/{index}/manifest.json",
                        manifest_digest=str(index + 4) * 64,
                        file_count=1,
                        total_size=10,
                        manifest_state=state,
                        last_verified_at=datetime.now(timezone.utc),
                    )
                )
            policy = ModelPreheatDistributionPolicy(
                name="all-current",
                selection_mode=ModelPreheatDistributionSelectionModeEnum.ALL_CURRENT,
                profile_id=profile.id,
                profile_config_version=profile.config_version,
                request_identity={"selection_mode": "all_current"},
                request_digest="9" * 64,
                target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                worker_selector={"worker_uuids": [worker.worker_uuid]},
                gpu_selector={},
                selector_digest="8" * 64,
            )
            session.add(policy)
            await session.commit()
            await session.refresh(policy)
            sources = await resolve_distribution_sources(session, policy)
            for source in sources:
                await _create_or_rebind_distribution_task(
                    session, policy, source, worker, run_key="run"
                )
            await session.commit()
            tasks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.distribution_policy_id == policy.id
                    )
                )
            ).all()
        await engine.dispose()
        return sources, tasks

    sources, tasks = asyncio.run(run())
    assert [source.artifact.artifact_id for source in sources] == ["1" * 64, "2" * 64]
    assert {task.distribution_artifact_id for task in tasks} == {"1" * 64, "2" * 64}
    assert len({task.distribution_request_digest for task in tasks}) == 2


@dataclass
class FakeReadyProbe:
    result: object | None
    calls: int = 0

    async def probe(self, task):
        self.calls += 1
        return self.result


async def _database(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'worker-reconciler.db'}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return engine


async def _seed(engine, *, state=ModelPreheatExecutionStateEnum.READY):
    async with AsyncSession(engine) as session:
        profile = ModelPreheatS3Profile(
            name=f"profile-{state.value}",
            endpoint="https://s3.example.com",
            bucket="models",
            access_key_encrypted={"ciphertext": "encrypted-access"},
            secret_key_encrypted={"ciphertext": "encrypted-secret"},
            encryption_key_version="v1",
        )
        old_worker = Worker(
            name=f"old-worker-{state.value}",
            hostname="old-host",
            ip="10.0.0.1",
            port=10150,
            worker_uuid="old-uuid",
            state=WorkerStateEnum.READY,
            model_storage_protocol_version=1,
        )
        session.add(profile)
        session.add(old_worker)
        await session.flush()
        task = ModelPreheatTask(
            source="huggingface",
            model_id="org/model",
            resolved_revision="a" * 40,
            include_patterns=[],
            exclude_patterns=[],
            selection_digest="b" * 64,
            request_identity={
                "source": "huggingface",
                "model_id": "org/model",
                "requested_revision": None,
                "include_patterns": [],
                "exclude_patterns": [],
            },
            request_digest="c" * 64,
            artifact_id=(
                ARTIFACT_ID if state == ModelPreheatExecutionStateEnum.READY else None
            ),
            target_scope=ModelPreheatTargetScopeEnum.SAME_GPU_MODEL,
            target_gpu_names=["NVIDIA L40S"],
            target_worker_uuids=[old_worker.worker_uuid],
            target_worker_snapshot=[
                {"worker_uuid": old_worker.worker_uuid, "worker_id": old_worker.id}
            ],
            s3_profile_id=profile.id,
            s3_profile_config_version=profile.config_version,
            s3_profile_snapshot_encrypted={"ciphertext": "encrypted-snapshot"},
            encryption_key_version="v1",
            s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
            keep_new_workers_in_sync=True,
            execution_state=state,
            manifest_digest=(
                "d" * 64 if state == ModelPreheatExecutionStateEnum.READY else None
            ),
            s3_manifest_path=(
                f"model-storage/huggingface/org/model/{ARTIFACT_ID}/manifest.json"
                if state == ModelPreheatExecutionStateEnum.READY
                else None
            ),
        )
        session.add(task)
        await session.flush()
        task_id = task.id
        profile_id = profile.id
        await session.commit()
        return task_id, profile_id


async def _new_worker(
    session,
    *,
    worker_id_suffix="a",
    state=WorkerStateEnum.READY,
    protocol_version=1,
):
    worker = Worker(
        name=f"new-worker-{worker_id_suffix}",
        hostname=f"new-host-{worker_id_suffix}",
        ip="10.0.0.2",
        port=10150,
        worker_uuid="new-uuid",
        state=state,
        model_storage_protocol_version=protocol_version,
        labels={"gpu_names": "NVIDIA L40S"},
    )
    session.add(worker)
    await session.commit()
    await session.refresh(worker)
    return worker


def _ready_result():
    return object()


def test_ready_task_materializes_policy_once_and_non_ready_task_does_not(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        ready_id, _ = await _seed(engine)
        pending_id, _ = await _seed(
            engine, state=ModelPreheatExecutionStateEnum.DISTRIBUTING
        )
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            policies = (
                await session.exec(select(ModelPreheatDistributionPolicy))
            ).all()
        await engine.dispose()
        return ready_id, pending_id, policies

    ready_id, pending_id, policies = asyncio.run(run())
    assert len(policies) == 1
    assert policies[0].created_by_task_id == ready_id
    assert policies[0].created_by_task_id != pending_id
    assert policies[0].enabled is True


def test_preheat_source_keeps_frozen_credentials_and_trusted_local_task(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine, expire_on_commit=False) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            source = await resolve_distribution_source(session, policy)
        await engine.dispose()
        return task_id, source

    task_id, source = asyncio.run(run())
    assert source.encrypted_profile == {"ciphertext": "encrypted-snapshot"}
    assert source.preheat_task.id == task_id
    assert source.payload["artifact_id"] == ARTIFACT_ID


def test_materialization_never_rewrites_manually_sourced_artifact_policy(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, profile_id = await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine, expire_on_commit=False) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            artifact = ModelPreheatArtifact(
                profile_id=profile_id,
                profile_config_version=1,
                artifact_id=ARTIFACT_ID,
                source="huggingface",
                model_id="org/model",
                resolved_revision="a" * 40,
                include_patterns=[],
                exclude_patterns=[],
                manifest_path=(
                    f"model-storage/huggingface/org/model/{ARTIFACT_ID}/manifest.json"
                ),
                manifest_digest="d" * 64,
                file_count=1,
                total_size=10,
                manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
                last_verified_at=datetime.now(timezone.utc),
            )
            session.add(artifact)
            await session.flush()
            policy.name = "人工 Artifact 分发"
            policy.created_by_task_id = None
            policy.source_artifact_id = artifact.id
            session.add(policy)
            await session.commit()
            policy_id = policy.id
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
        await engine.dispose()
        return task_id, policy

    task_id, policy = asyncio.run(run())
    assert policy.name == "人工 Artifact 分发"
    assert policy.created_by_task_id is None
    assert policy.source_artifact_id is not None
    assert policy.source_sync_task_id is None
    assert policy.created_by_task_id != task_id


def test_manually_disabled_policy_is_not_reenabled_by_materialization(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine, expire_on_commit=False) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            policy.enabled = False
            policy.profile_version_stale = False
            session.add(policy)
            await session.commit()
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
        await engine.dispose()
        return policy

    assert asyncio.run(run()).enabled is False


def test_discovered_artifact_without_local_task_becomes_explicitly_blocked_when_stale(
    tmp_path,
):
    async def run():
        engine = await _database(tmp_path)
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = ModelPreheatS3Profile(
                name="discovered-profile",
                endpoint="https://s3.example.com",
                bucket="models",
                access_key_encrypted={"ciphertext": "access"},
                secret_key_encrypted={"ciphertext": "secret"},
                encryption_key_version="v1",
            )
            session.add(profile)
            await session.flush()
            artifact = ModelPreheatArtifact(
                profile_id=profile.id,
                profile_config_version=profile.config_version,
                artifact_id=ARTIFACT_ID,
                source="huggingface",
                model_id="org/model",
                resolved_revision="a" * 40,
                include_patterns=[],
                exclude_patterns=[],
                manifest_path=f"model-storage/huggingface/org/model/{ARTIFACT_ID}/manifest.json",
                manifest_digest="d" * 64,
                file_count=1,
                total_size=10,
                manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
                last_verified_at=datetime.now(timezone.utc),
            )
            session.add(artifact)
            await session.flush()
            policy = ModelPreheatDistributionPolicy(
                name="shared-artifact",
                profile_id=profile.id,
                profile_config_version=profile.config_version,
                request_identity={"source": "huggingface", "model_id": "org/model"},
                request_digest="c" * 64,
                target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                worker_selector={"worker_uuids": ["worker-a"]},
                gpu_selector={},
                selector_digest="d" * 64,
                source_artifact_id=artifact.id,
                trigger_mode=ModelPreheatDistributionPolicyTriggerModeEnum.MANUAL,
            )
            session.add(policy)
            await session.commit()
            artifact.manifest_state = ModelPreheatInventoryManifestStateEnum.STALE
            session.add(artifact)
            await session.commit()
            policy_id = policy.id
        reconciler = ModelPreheatWorkerReconciler(engine)
        await reconciler.reconcile_policies()
        run = await reconciler.reconcile_manual_policy(policy_id)
        async with AsyncSession(engine) as session:
            policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
        await engine.dispose()
        return policy, run

    policy, distribution_run = asyncio.run(run())
    assert policy.enabled is True
    assert policy.blocked_reason == "distribution_artifact_stale"
    assert distribution_run.state == ModelPreheatDistributionPolicyRunStateEnum.ERROR
    assert distribution_run.error_code == "distribution_artifact_stale"
    assert distribution_run.outcome["failed"] == [
        {
            "task_id": None,
            "worker_id": None,
            "worker_uuid": None,
            "reason": "distribution_artifact_stale",
        }
    ]


def test_manual_and_continuous_safety_runs_have_distinct_audit_identities(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        _, _profile_id = await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(engine)
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            policy_id = policy.id
        await reconciler.reconcile_manual_policy(policy_id)
        await reconciler.reconcile_manual_policy(policy_id)
        now = datetime.now(timezone.utc)
        await reconciler.reconcile_continuous_policy(policy_id, now)
        await reconciler.reconcile_continuous_policy(policy_id, now)
        async with AsyncSession(engine) as session:
            runs = (
                await session.exec(
                    select(ModelPreheatDistributionPolicyRun).where(
                        ModelPreheatDistributionPolicyRun.policy_id == policy_id
                    )
                )
            ).all()
        await engine.dispose()
        return runs

    runs = asyncio.run(run())
    assert len(runs) == 3
    assert [run.trigger for run in runs].count(
        ModelPreheatDistributionPolicyRunTriggerEnum.MANUAL
    ) == 2
    assert [run.trigger for run in runs].count(
        ModelPreheatDistributionPolicyRunTriggerEnum.CONTINUOUS
    ) == 1
    assert all(
        run.state == ModelPreheatDistributionPolicyRunStateEnum.ERROR for run in runs
    )
    assert all(run.error_code == "distribution_no_eligible_workers" for run in runs)
    assert all(
        run.outcome == {"created": [], "skipped": [], "failed": []} for run in runs
    )


def test_scheduled_windows_reuse_existing_active_worker_task(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(engine)
        await reconciler.reconcile_policies()
        async with AsyncSession(engine, expire_on_commit=False) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            source = await session.get(ModelPreheatTask, task_id)
            worker = (
                await session.exec(
                    select(Worker).where(Worker.worker_uuid == "old-uuid")
                )
            ).one()
            policy_id = policy.id
            worker_uuid = worker.worker_uuid
            first = await _create_or_rebind_distribution_task(
                session, policy, source, worker, run_key="scheduled-window-one"
            )
            second = await _create_or_rebind_distribution_task(
                session, policy, source, worker, run_key="scheduled-window-two"
            )
            await session.commit()
            tasks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.distribution_policy_id == policy_id,
                        ModelPreheatWorkerTask.worker_uuid == worker_uuid,
                    )
                )
            ).all()
        await engine.dispose()
        return first.id, second.id, tasks

    first_id, second_id, tasks = asyncio.run(run())
    assert first_id == second_id
    assert len(tasks) == 1
    assert tasks[0].state == ModelPreheatWorkerTaskStateEnum.PENDING


def test_distribution_worker_slot_cas_prevents_cross_session_active_tasks(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(engine)
        await reconciler.reconcile_policies()
        async with AsyncSession(engine, expire_on_commit=False) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            source = await session.get(ModelPreheatTask, task_id)
            worker = (
                await session.exec(
                    select(Worker).where(Worker.worker_uuid == "old-uuid")
                )
            ).one()
            completed = ModelPreheatWorkerTask(
                distribution_policy_id=policy.id,
                operation_key=distribution_operation_key(
                    policy.id,
                    worker.worker_uuid,
                    policy.request_digest,
                    "completed-window",
                ),
                parent_attempt=source.attempt,
                worker_uuid=worker.worker_uuid,
                worker_id=worker.id,
                role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                state=ModelPreheatWorkerTaskStateEnum.READY,
            )
            session.add(completed)
            await session.flush()
            session.add(
                ModelPreheatDistributionWorkerSlot(
                    policy_id=policy.id,
                    worker_uuid=worker.worker_uuid,
                    active_task_id=completed.id,
                    active_operation_key=completed.operation_key,
                )
            )
            await session.commit()
            policy_id = policy.id
            worker_id = worker.id
            worker_uuid = worker.worker_uuid

        start = asyncio.Event()

        async def create_for_window(run_key):
            await start.wait()
            async with AsyncSession(engine, expire_on_commit=False) as session:
                policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
                source = await session.get(ModelPreheatTask, task_id)
                worker = await session.get(Worker, worker_id)
                task = await _create_or_rebind_distribution_task(
                    session, policy, source, worker, run_key=run_key
                )
                await session.commit()
                return task.id if task is not None else None

        first = asyncio.create_task(create_for_window("scheduled-window-one"))
        second = asyncio.create_task(create_for_window("scheduled-window-two"))
        await asyncio.sleep(0)
        start.set()
        result_ids = await asyncio.gather(first, second)
        async with AsyncSession(engine) as session:
            active_tasks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.distribution_policy_id == policy_id,
                        ModelPreheatWorkerTask.worker_uuid == worker_uuid,
                        ModelPreheatWorkerTask.role
                        == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                        ModelPreheatWorkerTask.state.in_(
                            [
                                ModelPreheatWorkerTaskStateEnum.PENDING,
                                ModelPreheatWorkerTaskStateEnum.RUNNING,
                                ModelPreheatWorkerTaskStateEnum.PAUSED,
                            ]
                        ),
                    )
                )
            ).all()
        await engine.dispose()
        return result_ids, active_tasks

    result_ids, active_tasks = asyncio.run(run())
    assert len(active_tasks) == 1
    assert set(result_ids) <= {None, active_tasks[0].id}


def test_protocol_mismatch_worker_gets_no_connectivity_or_distribution(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            worker = await _new_worker(session, protocol_version=0)
            worker_uuid = worker.worker_uuid
        await reconciler.reconcile_worker(worker_uuid)
        async with AsyncSession(engine) as session:
            checks = (await session.exec(select(ModelPreheatS3ConnectivityCheck))).all()
            tasks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.worker_uuid == worker_uuid
                    )
                )
            ).all()
            observation = await session.get(ModelPreheatWorkerObservation, worker_uuid)
        await engine.dispose()
        return checks, tasks, observation

    checks, tasks, observation = asyncio.run(run())
    assert checks == []
    assert tasks == []
    assert observation.ready is False


def test_new_worker_gets_incremental_connectivity_then_idempotent_distribution(
    tmp_path,
):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        probe = FakeReadyProbe(_ready_result())
        reconciler = ModelPreheatWorkerReconciler(engine, ready_probe=probe)
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            worker = await _new_worker(session)
            worker_uuid = worker.worker_uuid
        await reconciler.handle_event(Event(EventType.CREATED, worker))
        async with AsyncSession(engine) as session:
            checks = (await session.exec(select(ModelPreheatS3ConnectivityCheck))).all()
            connectivity = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.role
                        == ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK
                    )
                )
            ).one()
            connectivity.state = ModelPreheatWorkerTaskStateEnum.READY
            session.add(connectivity)
            await session.commit()
        await reconciler.reconcile_worker(worker_uuid)
        await asyncio.gather(
            reconciler.reconcile_worker(worker_uuid),
            reconciler.reconcile_worker(worker_uuid),
        )
        async with AsyncSession(engine) as session:
            policy_id = (
                await session.exec(select(ModelPreheatDistributionPolicy.id))
            ).one()
        manual_run = await reconciler.reconcile_manual_policy(policy_id)
        async with AsyncSession(engine) as session:
            checks = (await session.exec(select(ModelPreheatS3ConnectivityCheck))).all()
            tasks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.role
                        == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE
                    )
                )
            ).all()
            links = (
                await session.exec(
                    select(ModelPreheatDistributionPolicyRunTask).where(
                        ModelPreheatDistributionPolicyRunTask.run_id == manual_run.id
                    )
                )
            ).all()
            parent = await session.get(ModelPreheatTask, task_id)
            run = await session.get(ModelPreheatDistributionPolicyRun, manual_run.id)
        await engine.dispose()
        return checks, tasks, links, parent.target_worker_uuids, probe.calls, run

    checks, tasks, links, targets, probe_calls, distribution_run = asyncio.run(run())
    assert len(checks) == 1
    assert len(tasks) == 1
    assert tasks[0].task_id is None
    assert tasks[0].distribution_policy_id is not None
    assert tasks[0].worker_uuid == "new-uuid"
    assert [link.task_id for link in links] == [tasks[0].id]
    assert targets == ["old-uuid"]
    assert probe_calls == 0
    assert distribution_run.state == ModelPreheatDistributionPolicyRunStateEnum.PENDING
    assert distribution_run.finished_at is None


def test_distribution_run_finishes_after_linked_tasks_are_terminal(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(engine)
        await reconciler.reconcile_policies()
        async with AsyncSession(engine, expire_on_commit=False) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            source = await session.get(ModelPreheatTask, task_id)
            worker = (
                await session.exec(
                    select(Worker).where(Worker.worker_uuid == "old-uuid")
                )
            ).one()
            run = ModelPreheatDistributionPolicyRun(
                policy_id=policy.id,
                trigger=ModelPreheatDistributionPolicyRunTriggerEnum.MANUAL,
                window_start_utc=datetime.now(timezone.utc),
                operation_key="manual-finish",
            )
            session.add(run)
            await session.flush()
            worker_task = await _create_or_rebind_distribution_task(
                session, policy, source, worker, run_key=run.operation_key
            )
            session.add(
                ModelPreheatDistributionPolicyRunTask(
                    run_id=run.id, task_id=worker_task.id
                )
            )
            await session.commit()
            run_id = run.id
            worker_task_id = worker_task.id
        await reconciler._settle_policy_runs()
        async with AsyncSession(engine) as session:
            running_run = await session.get(ModelPreheatDistributionPolicyRun, run_id)
            running_state = running_run.state
            running_finished_at = running_run.finished_at
            worker_task = await session.get(ModelPreheatWorkerTask, worker_task_id)
            worker_task.state = ModelPreheatWorkerTaskStateEnum.READY
            worker_task.finished_at = datetime.now(timezone.utc)
            session.add(worker_task)
            await session.commit()
        await reconciler._settle_policy_runs()
        async with AsyncSession(engine) as session:
            finished_run = await session.get(ModelPreheatDistributionPolicyRun, run_id)
            finished_state = finished_run.state
            finished_finished_at = finished_run.finished_at
        await engine.dispose()
        return running_state, running_finished_at, finished_state, finished_finished_at

    running_state, running_finished_at, finished_state, finished_finished_at = (
        asyncio.run(run())
    )
    assert running_state == ModelPreheatDistributionPolicyRunStateEnum.PENDING
    assert running_finished_at is None
    assert finished_state == ModelPreheatDistributionPolicyRunStateEnum.READY
    assert finished_finished_at is not None


def test_distribution_run_active_lease_prevents_early_settle(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(engine)
        await reconciler.reconcile_policies()
        now = datetime.now(timezone.utc)
        async with AsyncSession(engine, expire_on_commit=False) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            source = await session.get(ModelPreheatTask, task_id)
            worker = (
                await session.exec(
                    select(Worker).where(Worker.worker_uuid == "old-uuid")
                )
            ).one()
            run = ModelPreheatDistributionPolicyRun(
                policy_id=policy.id,
                trigger=ModelPreheatDistributionPolicyRunTriggerEnum.MANUAL,
                window_start_utc=now,
                operation_key="manual-active-lease",
                lease_owner="planner",
                lease_token="token",
                lease_expires_at=now + timedelta(minutes=1),
            )
            session.add(run)
            await session.flush()
            worker_task = await _create_or_rebind_distribution_task(
                session, policy, source, worker, run_key=run.operation_key
            )
            worker_task.state = ModelPreheatWorkerTaskStateEnum.READY
            worker_task.finished_at = now
            session.add(
                ModelPreheatDistributionPolicyRunTask(
                    run_id=run.id, task_id=worker_task.id
                )
            )
            await session.commit()
            run_id = run.id
        await reconciler._settle_policy_runs()
        async with AsyncSession(engine) as session:
            active_run = await session.get(ModelPreheatDistributionPolicyRun, run_id)
            active_values = (active_run.state, active_run.finished_at)
            active_run.lease_owner = None
            active_run.lease_token = None
            active_run.lease_expires_at = None
            session.add(active_run)
            await session.commit()
        await reconciler._settle_policy_runs()
        async with AsyncSession(engine) as session:
            settled_run = await session.get(ModelPreheatDistributionPolicyRun, run_id)
            settled_values = (settled_run.state, settled_run.finished_at is not None)
        await engine.dispose()
        return active_values, settled_values

    active_values, settled_values = asyncio.run(run())
    assert active_values == (ModelPreheatDistributionPolicyRunStateEnum.PENDING, None)
    assert settled_values == (ModelPreheatDistributionPolicyRunStateEnum.READY, True)


def test_distribution_run_expired_lease_can_settle_after_planner_crash(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(engine)
        await reconciler.reconcile_policies()
        now = datetime.now(timezone.utc)
        async with AsyncSession(engine, expire_on_commit=False) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            source = await session.get(ModelPreheatTask, task_id)
            worker = (
                await session.exec(
                    select(Worker).where(Worker.worker_uuid == "old-uuid")
                )
            ).one()
            run = ModelPreheatDistributionPolicyRun(
                policy_id=policy.id,
                trigger=ModelPreheatDistributionPolicyRunTriggerEnum.MANUAL,
                window_start_utc=now,
                operation_key="manual-expired-lease",
                lease_owner="crashed-planner",
                lease_token="token",
                lease_expires_at=now - timedelta(seconds=1),
            )
            session.add(run)
            await session.flush()
            worker_task = await _create_or_rebind_distribution_task(
                session, policy, source, worker, run_key=run.operation_key
            )
            worker_task.state = ModelPreheatWorkerTaskStateEnum.READY
            worker_task.finished_at = now
            session.add(
                ModelPreheatDistributionPolicyRunTask(
                    run_id=run.id, task_id=worker_task.id
                )
            )
            await session.commit()
            run_id = run.id
        await reconciler._settle_policy_runs()
        async with AsyncSession(engine) as session:
            settled = await session.get(ModelPreheatDistributionPolicyRun, run_id)
            values = (
                settled.state,
                settled.lease_owner,
                settled.lease_token,
                settled.lease_expires_at,
                settled.finished_at is not None,
            )
        await engine.dispose()
        return values

    assert asyncio.run(run()) == (
        ModelPreheatDistributionPolicyRunStateEnum.READY,
        None,
        None,
        None,
        True,
    )


def test_distribution_run_keeps_error_code_for_partial_terminal_tasks(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(engine)
        await reconciler.reconcile_policies()
        async with AsyncSession(engine, expire_on_commit=False) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            source = await session.get(ModelPreheatTask, task_id)
            worker = (
                await session.exec(
                    select(Worker).where(Worker.worker_uuid == "old-uuid")
                )
            ).one()
            run = ModelPreheatDistributionPolicyRun(
                policy_id=policy.id,
                trigger=ModelPreheatDistributionPolicyRunTriggerEnum.MANUAL,
                window_start_utc=datetime.now(timezone.utc),
                operation_key="manual-partial",
            )
            session.add(run)
            await session.flush()
            ready_task = await _create_or_rebind_distribution_task(
                session, policy, source, worker, run_key=run.operation_key
            )
            ready_task.state = ModelPreheatWorkerTaskStateEnum.READY
            ready_task.finished_at = datetime.now(timezone.utc)
            error_task = ModelPreheatWorkerTask(
                distribution_policy_id=policy.id,
                operation_key="manual-partial-error",
                worker_uuid="error-uuid",
                role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                state=ModelPreheatWorkerTaskStateEnum.ERROR,
                error_code="s3_read_failed",
                finished_at=datetime.now(timezone.utc),
            )
            session.add(error_task)
            await session.flush()
            session.add_all(
                [
                    ModelPreheatDistributionPolicyRunTask(
                        run_id=run.id, task_id=ready_task.id
                    ),
                    ModelPreheatDistributionPolicyRunTask(
                        run_id=run.id, task_id=error_task.id
                    ),
                ]
            )
            await session.commit()
            run_id = run.id
        await reconciler._settle_policy_runs()
        async with AsyncSession(engine) as session:
            run = await session.get(ModelPreheatDistributionPolicyRun, run_id)
            values = (run.state, run.error_code)
        await engine.dispose()
        return values

    assert asyncio.run(run()) == (
        ModelPreheatDistributionPolicyRunStateEnum.READY,
        "s3_read_failed",
    )


def test_maintenance_profile_does_not_create_distribution_for_existing_policy(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        _, profile_id = await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            worker = await _new_worker(session)
            worker_uuid = worker.worker_uuid
            profile = await session.get(ModelPreheatS3Profile, profile_id)
            profile.lifecycle_state = (
                ModelPreheatS3ProfileLifecycleStateEnum.MAINTENANCE
            )
            session.add(profile)
            check = ModelPreheatS3ConnectivityCheck(
                profile_id=profile.id,
                profile_config_version=profile.config_version,
                state="available",
                target_worker_uuids=[worker.worker_uuid],
            )
            session.add(check)
            await session.flush()
            session.add(
                ModelPreheatWorkerTask(
                    connectivity_check_id=check.id,
                    worker_uuid=worker.worker_uuid,
                    worker_id=worker.id,
                    role=ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
                    state=ModelPreheatWorkerTaskStateEnum.READY,
                )
            )
            await session.commit()
        await reconciler.reconcile_worker(worker_uuid)
        async with AsyncSession(engine) as session:
            tasks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.role
                        == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE
                    )
                )
            ).all()
        await engine.dispose()
        return tasks

    assert asyncio.run(run()) == []


def test_heartbeat_and_temporary_offline_do_not_create_or_skip_work(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            worker = await _new_worker(session, state=WorkerStateEnum.NOT_READY)
        await reconciler.handle_event(Event(EventType.HEARTBEAT, None))
        await reconciler.handle_event(Event(EventType.UPDATED, worker))
        async with AsyncSession(engine) as session:
            checks = (await session.exec(select(ModelPreheatS3ConnectivityCheck))).all()
            tasks = (await session.exec(select(ModelPreheatWorkerTask))).all()
        await engine.dispose()
        return checks, tasks

    checks, tasks = asyncio.run(run())
    assert checks == []
    assert tasks == []


def test_deleted_worker_is_skipped_but_disabled_policy_keeps_existing_task(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            worker = await _new_worker(session)
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            policy.enabled = False
            task = ModelPreheatWorkerTask(
                distribution_policy_id=policy.id,
                operation_key=(
                    f"{policy.id}:{worker.worker_uuid}:{policy.request_digest}"
                ),
                worker_uuid=worker.worker_uuid,
                worker_id=worker.id,
                role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                state=ModelPreheatWorkerTaskStateEnum.RUNNING,
                lease_owner=worker.worker_uuid,
                lease_token_hash="hash",
            )
            session.add(policy)
            session.add(task)
            worker_uuid = worker.worker_uuid
            worker_id = worker.id
            await session.commit()
        await reconciler._reconcile_deleted(worker_uuid, worker_id)
        async with AsyncSession(engine) as session:
            tasks = (await session.exec(select(ModelPreheatWorkerTask))).all()
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
        await engine.dispose()
        return tasks, policy

    tasks, policy = asyncio.run(run())
    assert policy.enabled is False
    assert len(tasks) == 1
    assert tasks[0].state == ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED
    assert tasks[0].lease_owner is None
    assert tasks[0].lease_token_hash is None


def test_same_uuid_reregistration_uses_latest_id_and_rechecks_connectivity(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            first = await _new_worker(session, worker_id_suffix="first")
            first_id = first.id
        await reconciler.handle_event(Event(EventType.CREATED, first))
        async with AsyncSession(engine) as session:
            first_check = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.role
                        == ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK
                    )
                )
            ).one()
            first_check.state = ModelPreheatWorkerTaskStateEnum.READY
            session.add(first_check)
            second = await _new_worker(session, worker_id_suffix="second")
            second_id = second.id
            await session.commit()
        await reconciler.reconcile_worker("new-uuid")
        async with AsyncSession(engine) as session:
            checks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.role
                        == ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK
                    )
                )
            ).all()
        await engine.dispose()
        return first_id, second_id, checks

    first_id, second_id, checks = asyncio.run(run())
    assert second_id > first_id
    assert len(checks) == 2
    assert max(task.worker_id for task in checks) == second_id


def test_policy_disabled_during_strict_ready_probe_does_not_create_work(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            worker = await _new_worker(session)
        await reconciler.handle_event(Event(EventType.CREATED, worker))
        async with AsyncSession(engine) as session:
            connectivity = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.role
                        == ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK
                    )
                )
            ).one()
            connectivity.state = ModelPreheatWorkerTaskStateEnum.READY
            session.add(connectivity)
            await session.commit()
        async with AsyncSession(engine) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            policy.enabled = False
            session.add(policy)
            await session.commit()
        await reconciler.reconcile_worker("new-uuid")
        async with AsyncSession(engine) as session:
            tasks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.role
                        == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE
                    )
                )
            ).all()
        await engine.dispose()
        return tasks

    assert asyncio.run(run()) == []


def test_network_change_rechecks_but_ordinary_ready_update_does_not(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        async with AsyncSession(engine) as session:
            worker = await _new_worker(session)
            worker_id = worker.id
        await reconciler.handle_event(Event(EventType.CREATED, worker))
        await reconciler.reconcile_worker("new-uuid")
        async with AsyncSession(engine) as session:
            unchanged_count = len(
                (await session.exec(select(ModelPreheatS3ConnectivityCheck))).all()
            )
            current = await session.get(Worker, worker_id)
            current.ip = "10.0.0.99"
            session.add(current)
            await session.commit()
        await reconciler.reconcile_worker("new-uuid")
        async with AsyncSession(engine) as session:
            changed_count = len(
                (await session.exec(select(ModelPreheatS3ConnectivityCheck))).all()
            )
        await engine.dispose()
        return unchanged_count, changed_count

    assert asyncio.run(run()) == (1, 2)


def test_deleted_old_registration_does_not_skip_new_registration_task(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            old = await _new_worker(session, worker_id_suffix="old")
            old_id = old.id
            new = await _new_worker(session, worker_id_suffix="new")
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            task = ModelPreheatWorkerTask(
                distribution_policy_id=policy.id,
                operation_key=f"new-registration-{policy.id}",
                worker_uuid=new.worker_uuid,
                worker_id=new.id,
                role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                state=ModelPreheatWorkerTaskStateEnum.RUNNING,
                lease_owner=new.worker_uuid,
                lease_token_hash="new-token",
            )
            session.add(task)
            await session.flush()
            task_id = task.id
            await session.commit()
        await reconciler.handle_event(
            Event(
                EventType.DELETED,
                SimpleNamespace(worker_uuid="new-uuid", id=old_id),
            )
        )
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatWorkerTask, task_id)
        await engine.dispose()
        return task

    task = asyncio.run(run())
    assert task.state == ModelPreheatWorkerTaskStateEnum.RUNNING
    assert task.lease_token_hash == "new-token"


def test_skipped_policy_task_rebinds_latest_same_uuid_registration(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            old = await _new_worker(session, worker_id_suffix="old")
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            old_id = old.id
            task = ModelPreheatWorkerTask(
                distribution_policy_id=policy.id,
                operation_key=distribution_operation_key(
                    policy.id, old.worker_uuid, policy.request_digest
                ),
                worker_uuid=old.worker_uuid,
                worker_id=old.id,
                role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                state=ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED,
            )
            session.add(task)
            await session.flush()
            task_id = task.id
            await session.commit()
        async with AsyncSession(engine) as session:
            new = await _new_worker(session, worker_id_suffix="new")
            new_id = new.id
        await reconciler.handle_event(
            Event(
                EventType.DELETED,
                SimpleNamespace(worker_uuid="new-uuid", id=old_id),
            )
        )
        async with AsyncSession(engine) as session:
            check_task = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.role
                        == ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
                        ModelPreheatWorkerTask.worker_id == new_id,
                    )
                )
            ).one()
            check_task.state = ModelPreheatWorkerTaskStateEnum.READY
            session.add(check_task)
            await session.commit()
        await reconciler.reconcile_worker("new-uuid")
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatWorkerTask, task_id)
        await engine.dispose()
        return new_id, task

    new_id, task = asyncio.run(run())
    assert task.worker_id == new_id
    assert task.state == ModelPreheatWorkerTaskStateEnum.PENDING


def test_profile_version_rotation_disables_old_policy_then_uses_new_ready_source(
    tmp_path,
):
    async def run():
        engine = await _database(tmp_path)
        old_task_id, profile_id = await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            profile = await session.get(ModelPreheatS3Profile, profile_id)
            profile.config_version = 2
            profile.access_key_encrypted = {"ciphertext": "new-access"}
            profile.secret_key_encrypted = {"ciphertext": "new-secret"}
            session.add(profile)
            await session.commit()
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            disabled = (
                await session.exec(select(ModelPreheatDistributionPolicy))
            ).one()
            disabled_state = disabled.enabled
            old_task = await session.get(ModelPreheatTask, old_task_id)
            new_task = ModelPreheatTask(
                **old_task.model_dump(
                    exclude={
                        "id",
                        "created_at",
                        "updated_at",
                        "deleted_at",
                        "s3_profile_config_version",
                        "s3_profile_snapshot_encrypted",
                    }
                ),
                s3_profile_config_version=2,
                s3_profile_snapshot_encrypted={"ciphertext": "new-snapshot"},
            )
            session.add(new_task)
            await session.commit()
            await session.refresh(new_task)
            new_task_id = new_task.id
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            worker = await _new_worker(session, worker_id_suffix="profile-v2")
            worker_id = worker.id
            stale_check = ModelPreheatS3ConnectivityCheck(
                profile_id=profile_id,
                profile_config_version=1,
                state="available",
                target_worker_uuids=[worker.worker_uuid],
            )
            session.add(stale_check)
            await session.flush()
            session.add(
                ModelPreheatWorkerTask(
                    connectivity_check_id=stale_check.id,
                    worker_uuid=worker.worker_uuid,
                    worker_id=worker.id,
                    role=ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
                    state=ModelPreheatWorkerTaskStateEnum.READY,
                )
            )
            await session.commit()
        async with AsyncSession(engine) as session:
            worker = await session.get(Worker, worker_id)
        await reconciler._evaluate_worker(worker)
        async with AsyncSession(engine) as session:
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            distribute_tasks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.role
                        == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE
                    )
                )
            ).all()
        await engine.dispose()
        return disabled_state, old_task_id, new_task_id, policy, distribute_tasks

    disabled_state, old_task_id, new_task_id, policy, distribute_tasks = asyncio.run(
        run()
    )
    assert disabled_state is False
    assert policy.enabled is True
    assert policy.profile_config_version == 2
    assert policy.created_by_task_id == new_task_id
    assert policy.created_by_task_id != old_task_id
    assert distribute_tasks == []


def test_same_gpu_policy_normalizes_device_and_label_names(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            worker = await _new_worker(session)
            worker.labels = {"gpu_names": "  nViDiA    l40S  "}
            session.add(worker)
            await session.commit()
        await reconciler.reconcile_worker("new-uuid")
        async with AsyncSession(engine) as session:
            connectivity = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.role
                        == ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK
                    )
                )
            ).one()
            connectivity.state = ModelPreheatWorkerTaskStateEnum.READY
            session.add(connectivity)
            await session.commit()
        await reconciler.reconcile_worker("new-uuid")
        async with AsyncSession(engine) as session:
            tasks = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.role
                        == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE
                    )
                )
            ).all()
        await engine.dispose()
        return tasks

    assert len(asyncio.run(run())) == 1


def test_connectivity_failure_does_not_commit_observation_and_periodic_recovers(
    tmp_path,
):
    async def run():
        engine = await _database(tmp_path)
        await _seed(engine)
        async with AsyncSession(engine) as session:
            worker = await _new_worker(session)

        calls = 0

        async def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected_connectivity_failure")
            from gpustack.server.model_preheat_connectivity import (
                create_or_reuse_connectivity_check,
            )

            return await create_or_reuse_connectivity_check(*args, **kwargs)

        reconciler = ModelPreheatWorkerReconciler(
            engine,
            ready_probe=FakeReadyProbe(_ready_result()),
            connectivity_creator=fail_once,
        )
        try:
            await reconciler.reconcile_worker("new-uuid")
        except RuntimeError:
            pass
        async with AsyncSession(engine) as session:
            observation_after_failure = await session.get(
                ModelPreheatWorkerObservation, "new-uuid"
            )
        await reconciler.reconcile_worker("new-uuid")
        async with AsyncSession(engine) as session:
            observation = await session.get(ModelPreheatWorkerObservation, "new-uuid")
            checks = (await session.exec(select(ModelPreheatS3ConnectivityCheck))).all()
        await engine.dispose()
        return observation_after_failure, observation, checks

    failed_observation, recovered_observation, checks = asyncio.run(run())
    assert failed_observation is None
    assert recovered_observation is not None
    assert len(checks) == 1


def test_periodic_reconcile_repairs_missing_current_profile_connectivity(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        _, profile_id = await _seed(engine)
        async with AsyncSession(engine) as session:
            worker = await _new_worker(session)
            session.add(
                ModelPreheatWorkerObservation(
                    worker_uuid=worker.worker_uuid,
                    worker_id=worker.id,
                    network_fingerprint="stale-but-present",
                    ready=True,
                )
            )
            profile = await session.get(ModelPreheatS3Profile, profile_id)
            profile.config_version = 2
            session.add(profile)
            await session.commit()
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_all()
        async with AsyncSession(engine) as session:
            checks = (await session.exec(select(ModelPreheatS3ConnectivityCheck))).all()
        await engine.dispose()
        return checks

    checks = asyncio.run(run())
    assert len(checks) >= 1
    assert any(check.profile_config_version == 2 for check in checks)


def test_distribution_error_retry_uses_classification_backoff_and_cas(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        reconciler = ModelPreheatWorkerReconciler(
            engine, ready_probe=FakeReadyProbe(_ready_result())
        )
        await reconciler.reconcile_policies()
        async with AsyncSession(engine) as session:
            worker = await _new_worker(session, worker_id_suffix="retry")
            policy = (await session.exec(select(ModelPreheatDistributionPolicy))).one()
            source = await session.get(ModelPreheatTask, task_id)
            policy_id = policy.id
            worker_id = worker.id
            retryable = ModelPreheatWorkerTask(
                distribution_policy_id=policy.id,
                operation_key=distribution_operation_key(
                    policy.id, worker.worker_uuid, policy.request_digest
                ),
                parent_attempt=source.attempt,
                worker_uuid=worker.worker_uuid,
                worker_id=worker.id,
                role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                state=ModelPreheatWorkerTaskStateEnum.ERROR,
                attempt=2,
                error_code="network_timeout",
                finished_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            )
            session.add(retryable)
            await session.commit()
            policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
            source = await session.get(ModelPreheatTask, task_id)
            worker = await session.get(Worker, worker_id)
            await _create_or_rebind_distribution_task(session, policy, source, worker)
            await session.commit()
            await session.refresh(retryable)
            retried_state = retryable.state

            retryable.state = ModelPreheatWorkerTaskStateEnum.ERROR
            retryable.error_code = "local_cache_conflict"
            retryable.finished_at = datetime.now(timezone.utc) - timedelta(days=1)
            session.add(retryable)
            await session.commit()
            policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
            source = await session.get(ModelPreheatTask, task_id)
            worker = await session.get(Worker, worker_id)
            await _create_or_rebind_distribution_task(session, policy, source, worker)
            await session.commit()
            await session.refresh(retryable)
            permanent_state = retryable.state

            retryable.error_code = "s3_throttled"
            retryable.finished_at = datetime.now(timezone.utc)
            session.add(retryable)
            await session.commit()
            policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
            source = await session.get(ModelPreheatTask, task_id)
            worker = await session.get(Worker, worker_id)
            await _create_or_rebind_distribution_task(session, policy, source, worker)
            await session.commit()
            await session.refresh(retryable)
            backoff_state = retryable.state
        await engine.dispose()
        return retried_state, permanent_state, backoff_state

    assert asyncio.run(run()) == (
        ModelPreheatWorkerTaskStateEnum.PENDING,
        ModelPreheatWorkerTaskStateEnum.ERROR,
        ModelPreheatWorkerTaskStateEnum.ERROR,
    )
