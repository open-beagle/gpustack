import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheats import (
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatCachedModel,
    ModelPreheatPublicationMarker,
    ModelPreheatDesiredStateEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
)  # noqa: F401
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.model_preheat_controller import (
    LocalInventoryProbeResult,
    ModelPreheatController,
    ReadyProbeResult,
)
from gpustack.server.model_preheat_s3_inventory import ModelPreheatS3Inventory
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
from gpustack.worker.model_preheat.s3_client import ModelPreheatS3Client


GENERATION_ID = "preheat-00000000-0000-4000-8000-000000000001"


@dataclass
class FakeReadyProbe:
    result: ReadyProbeResult | None = None
    calls: int = 0

    async def probe(self, task):
        self.calls += 1
        return self.result


@dataclass
class FakeInventoryProbe:
    states: dict[str, str]
    calls: int = 0

    async def probe(self, task, worker_uuids):
        self.calls += 1
        return {
            worker_uuid: LocalInventoryProbeResult(
                worker_uuid=worker_uuid,
                state=self.states.get(worker_uuid, "missing"),
            )
            for worker_uuid in worker_uuids
        }


class BlockingReadyProbe:
    def __init__(self, result=None):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.result = result

    async def probe(self, task):
        self.started.set()
        await self.release.wait()
        return self.result


async def _database(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'controller.db'}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return engine


async def _seed(engine, worker_uuids=("worker-a", "worker-b")):
    async with AsyncSession(engine) as session:
        workers = []
        for index, worker_uuid in enumerate(worker_uuids):
            worker = Worker(
                name=worker_uuid,
                hostname=worker_uuid,
                ip=f"127.0.0.{index + 1}",
                port=10150,
                worker_uuid=worker_uuid,
                state=WorkerStateEnum.READY,
            )
            session.add(worker)
            workers.append(worker)
        await session.flush()
        snapshot = [
            {
                "worker_uuid": worker.worker_uuid,
                "worker_id": worker.id,
                "worker_name": worker.name,
            }
            for worker in workers
        ]
        task = ModelPreheatTask(
            source="huggingface",
            model_id="org/model",
            resolved_revision="a" * 40,
            include_patterns=[],
            exclude_patterns=[],
            selection_digest="b" * 64,
            cache_key="c" * 64,
            generation_id=GENERATION_ID,
            seed_worker_uuid=worker_uuids[0],
            seed_worker_id=workers[0].id,
            target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
            target_worker_uuids=list(worker_uuids),
            target_worker_snapshot=snapshot,
            s3_profile_id=1,
            s3_profile_config_version=1,
            s3_profile_snapshot_encrypted={"ciphertext": "encrypted"},
            encryption_key_version="v1",
            s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
        )
        session.add(task)
        worker_ids = {worker.worker_uuid: worker.id for worker in workers}
        await session.commit()
        await session.refresh(task)
        return task.id, worker_ids


async def _tasks(engine, task_id):
    async with AsyncSession(engine) as session:
        return (
            await session.exec(
                select(ModelPreheatWorkerTask)
                .where(ModelPreheatWorkerTask.task_id == task_id)
                .order_by(ModelPreheatWorkerTask.id)
            )
        ).all()


def test_reconcile_creates_only_seed_before_s3_is_ready(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        controller = ModelPreheatController(
            engine,
            ready_probe=FakeReadyProbe(),
            inventory_probe=FakeInventoryProbe({}),
        )
        await controller.reconcile_task(task_id)
        children = await _tasks(engine, task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent, children

    parent, children = asyncio.run(run())
    assert [(child.role, child.worker_uuid) for child in children] == [
        (ModelPreheatWorkerTaskRoleEnum.SEED, "worker-a")
    ]
    assert children[0].parent_attempt == 1
    assert parent.execution_state == ModelPreheatExecutionStateEnum.STAGING


def test_controller_persists_publication_marker_before_seed_is_claimable(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        async with AsyncSession(engine) as session:
            session.add(
                ModelPreheatS3Profile(
                    id=1,
                    name="marker-profile",
                    endpoint="https://s3.example.com",
                    bucket="models",
                    prefix="cache",
                    access_key_encrypted={"ciphertext": "x"},
                    secret_key_encrypted={"ciphertext": "y"},
                    encryption_key_version="v1",
                    config_version=1,
                )
            )
            await session.commit()
        inventory = ModelPreheatS3Inventory(engine)
        ready_probe = FakeReadyProbe()
        controller = ModelPreheatController(
            engine,
            ready_probe=ready_probe,
            inventory_probe=FakeInventoryProbe({}),
            s3_inventory=inventory,
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            marker = (await session.exec(select(ModelPreheatPublicationMarker))).one()
            seed = (
                await session.exec(
                    select(ModelPreheatWorkerTask).where(
                        ModelPreheatWorkerTask.task_id == task_id,
                        ModelPreheatWorkerTask.role
                        == ModelPreheatWorkerTaskRoleEnum.SEED,
                    )
                )
            ).one()
            task = await session.get(ModelPreheatTask, task_id)
            identity = ModelPreheatIdentity(
                task.source,
                task.model_id,
                task.resolved_revision,
                task.include_patterns,
            )
            client = ModelPreheatS3Client(None)
            prefix = client._selection_prefix("cache", identity, task.selection_digest)
            ready = ReadyProbeResult(
                manifest_digest="e" * 64,
                generation_id=task.generation_id,
                ready_path=client._join_object_name(prefix, "ready.json"),
                manifest_path=client._join_object_name(
                    prefix,
                    "generations",
                    task.generation_id,
                    ".gpustack-manifest.json",
                ),
                cache_key=task.cache_key,
                selection_digest=task.selection_digest,
                profile_config_version=1,
                file_count=1,
                total_size=10,
            )
            initial = marker.task_id, marker.parent_attempt, marker.generation_id
            seed_reference = seed.task_id, seed.parent_attempt
            seed.state = ModelPreheatWorkerTaskStateEnum.READY
            seed.resumable_cursor = {
                "manifest_digest": ready.manifest_digest,
                "generation_id": ready.generation_id,
                "local_cache_state": "valid",
            }
            session.add(seed)
            await session.commit()
        ready_probe.result = ready
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            marker_count = len(
                (await session.exec(select(ModelPreheatPublicationMarker))).all()
            )
            cached_count = len(
                (await session.exec(select(ModelPreheatCachedModel))).all()
            )
        await engine.dispose()
        return initial, seed_reference, marker_count, cached_count

    initial, seed_reference, marker_count, cached_count = asyncio.run(run())
    assert initial == (*seed_reference, GENERATION_ID)
    assert marker_count == 0
    assert cached_count == 1


def test_strict_ready_probe_upserts_valid_inventory_before_distribution(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        async with AsyncSession(engine) as session:
            session.add(
                ModelPreheatS3Profile(
                    id=1,
                    name="controller-profile",
                    endpoint="https://s3.example.com",
                    bucket="models",
                    prefix="cache",
                    access_key_encrypted={"ciphertext": "x"},
                    secret_key_encrypted={"ciphertext": "y"},
                    encryption_key_version="v1",
                    config_version=1,
                )
            )
            await session.commit()
            task = await session.get(ModelPreheatTask, task_id)
            identity = ModelPreheatIdentity(
                task.source,
                task.model_id,
                task.resolved_revision,
                task.include_patterns,
            )
            client = ModelPreheatS3Client(None)
            prefix = client._selection_prefix("cache", identity, task.selection_digest)
            ready = ReadyProbeResult(
                manifest_digest="e" * 64,
                generation_id=task.generation_id,
                ready_path=client._join_object_name(prefix, "ready.json"),
                manifest_path=client._join_object_name(
                    prefix,
                    "generations",
                    task.generation_id,
                    ".gpustack-manifest.json",
                ),
                cache_key=task.cache_key,
                selection_digest=task.selection_digest,
                profile_config_version=1,
                file_count=1,
                total_size=10,
            )
        inventory = ModelPreheatS3Inventory(engine)
        controller = ModelPreheatController(
            engine,
            ready_probe=FakeReadyProbe(ready),
            inventory_probe=FakeInventoryProbe({}),
            s3_inventory=inventory,
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            rows = (await session.exec(select(ModelPreheatCachedModel))).all()
        await engine.dispose()
        return rows

    rows = asyncio.run(run())
    assert len(rows) == 1
    assert rows[0].manifest_state == "valid"
    assert rows[0].manifest_digest == "e" * 64


def test_controller_restores_missing_marker_for_pending_seed_after_restart(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        inventory = ModelPreheatS3Inventory(engine)
        controller = ModelPreheatController(
            engine,
            ready_probe=FakeReadyProbe(),
            inventory_probe=FakeInventoryProbe({}),
            s3_inventory=inventory,
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            marker = (await session.exec(select(ModelPreheatPublicationMarker))).one()
            await session.delete(marker)
            await session.commit()
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            marker = (await session.exec(select(ModelPreheatPublicationMarker))).one()
            seed = (await session.exec(select(ModelPreheatWorkerTask))).one()
            result = marker.task_id, marker.parent_attempt, seed.state
        await engine.dispose()
        return result

    task_id, parent_attempt, seed_state = asyncio.run(run())
    assert task_id is not None
    assert parent_attempt == 1
    assert seed_state == ModelPreheatWorkerTaskStateEnum.PENDING


def test_seed_ready_then_creates_distribution_for_current_targets(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        ready_probe = FakeReadyProbe()
        controller = ModelPreheatController(
            engine,
            ready_probe=ready_probe,
            inventory_probe=FakeInventoryProbe({}),
        )
        await controller.reconcile_task(task_id)
        ready_probe.result = ReadyProbeResult(
            manifest_digest="d" * 64,
            generation_id=GENERATION_ID,
            ready_path="model-cache/v1/ready.json",
            manifest_path="model-cache/v1/generations/id/manifest.json",
        )
        async with AsyncSession(engine) as session:
            seed = (await session.exec(select(ModelPreheatWorkerTask))).one()
            seed.state = ModelPreheatWorkerTaskStateEnum.READY
            seed.resumable_cursor = {
                "state": "ready",
                "manifest_digest": "d" * 64,
                "generation_id": GENERATION_ID,
                "local_cache_state": "valid",
                "uploaded": 1,
                "skipped": 0,
                "downloaded": 0,
                "total_size": 10,
            }
            session.add(seed)
            await session.commit()
        await controller.reconcile_task(task_id)
        children = await _tasks(engine, task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent, children

    parent, children = asyncio.run(run())
    assert [child.role for child in children].count(
        ModelPreheatWorkerTaskRoleEnum.SEED
    ) == 1
    assert {
        child.worker_uuid
        for child in children
        if child.role == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE
    } == {"worker-b"}
    assert parent.execution_state == ModelPreheatExecutionStateEnum.DISTRIBUTING
    assert parent.manifest_digest == "d" * 64


def test_ready_probe_skips_seed_and_local_hits_skip_distribution(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        ready = ReadyProbeResult(
            manifest_digest="d" * 64,
            generation_id=GENERATION_ID,
            ready_path="cache/model-cache/v1/source/model/revision/selection/ready.json",
            manifest_path="cache/model-cache/v1/source/model/revision/selection/generations/id/.gpustack-manifest.json",
        )
        controller = ModelPreheatController(
            engine,
            ready_probe=FakeReadyProbe(ready),
            inventory_probe=FakeInventoryProbe(
                {"worker-a": "valid", "worker-b": "missing"}
            ),
        )
        await controller.reconcile_task(task_id)
        children = await _tasks(engine, task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent, children

    parent, children = asyncio.run(run())
    assert [(child.role, child.worker_uuid) for child in children] == [
        (ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE, "worker-b")
    ]
    assert parent.local_cache_hit_worker_uuids == ["worker-a"]
    assert parent.execution_state == ModelPreheatExecutionStateEnum.DISTRIBUTING


def test_local_valid_worker_is_selected_as_seed_when_s3_is_missing(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        controller = ModelPreheatController(
            engine,
            ready_probe=FakeReadyProbe(),
            inventory_probe=FakeInventoryProbe({"worker-b": "valid"}),
        )
        await controller.reconcile_task(task_id)
        children = await _tasks(engine, task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent, children

    parent, children = asyncio.run(run())
    assert [(child.role, child.worker_uuid) for child in children] == [
        (ModelPreheatWorkerTaskRoleEnum.SEED, "worker-b")
    ]
    assert parent.execution_state == ModelPreheatExecutionStateEnum.STAGING
    assert parent.local_cache_hit_worker_uuids == ["worker-b"]


def test_two_controllers_reconcile_to_one_seed(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        first = ModelPreheatController(engine, FakeReadyProbe(), FakeInventoryProbe({}))
        second = ModelPreheatController(
            engine, FakeReadyProbe(), FakeInventoryProbe({})
        )
        await asyncio.gather(
            first.reconcile_task(task_id), second.reconcile_task(task_id)
        )
        children = await _tasks(engine, task_id)
        await engine.dispose()
        return children

    children = asyncio.run(run())
    assert len(children) == 1
    assert children[0].role == ModelPreheatWorkerTaskRoleEnum.SEED


def test_old_parent_attempt_results_are_not_aggregated(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, worker_ids = await _seed(engine, ("worker-a",))
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            parent.attempt = 2
            session.add(parent)
            session.add(
                ModelPreheatWorkerTask(
                    task_id=task_id,
                    parent_attempt=1,
                    worker_uuid="worker-a",
                    worker_id=worker_ids["worker-a"],
                    role=ModelPreheatWorkerTaskRoleEnum.SEED,
                    state=ModelPreheatWorkerTaskStateEnum.READY,
                )
            )
            await session.commit()
        controller = ModelPreheatController(
            engine, FakeReadyProbe(), FakeInventoryProbe({})
        )
        await controller.reconcile_task(task_id)
        children = await _tasks(engine, task_id)
        await engine.dispose()
        return children

    children = asyncio.run(run())
    assert {(child.parent_attempt, child.state) for child in children} == {
        (1, ModelPreheatWorkerTaskStateEnum.READY),
        (2, ModelPreheatWorkerTaskStateEnum.PENDING),
    }


def test_removed_seed_is_skipped_and_reselected_by_latest_worker_id(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        controller = ModelPreheatController(
            engine, FakeReadyProbe(), FakeInventoryProbe({})
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            await session.exec(delete(Worker).where(Worker.worker_uuid == "worker-a"))
            await session.commit()
        await controller.reconcile_task(task_id)
        children = await _tasks(engine, task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent, children

    parent, children = asyncio.run(run())
    assert parent.seed_worker_uuid == "worker-b"
    assert parent.removed_target_worker_uuids == ["worker-a"]
    assert any(
        child.worker_uuid == "worker-a"
        and child.state == ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED
        for child in children
    )
    assert any(
        child.worker_uuid == "worker-b"
        and child.role == ModelPreheatWorkerTaskRoleEnum.SEED
        for child in children
    )


def test_zero_current_targets_never_becomes_ready(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine, ("worker-a",))
        async with AsyncSession(engine) as session:
            await session.exec(delete(Worker))
            await session.commit()
        controller = ModelPreheatController(
            engine, FakeReadyProbe(), FakeInventoryProbe({"worker-a": "valid"})
        )
        await controller.reconcile_task(task_id)
        children = await _tasks(engine, task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent, children

    parent, children = asyncio.run(run())
    assert parent.execution_state == ModelPreheatExecutionStateEnum.ERROR
    assert parent.state_message == "no_available_targets"
    assert [(child.worker_uuid, child.state) for child in children] == [
        ("worker-a", ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED)
    ]


def test_distribution_terminal_states_aggregate_ready_or_partial(tmp_path):
    async def scenario(partial):
        engine = await _database(tmp_path / ("partial" if partial else "ready"))
        task_id, worker_ids = await _seed(engine)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            parent.execution_state = ModelPreheatExecutionStateEnum.DISTRIBUTING
            parent.manifest_digest = "d" * 64
            session.add(parent)
            session.add_all(
                [
                    ModelPreheatWorkerTask(
                        task_id=task_id,
                        parent_attempt=1,
                        worker_uuid=worker_uuid,
                        worker_id=worker_id,
                        role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                        state=(
                            ModelPreheatWorkerTaskStateEnum.ERROR
                            if partial and worker_uuid == "worker-b"
                            else ModelPreheatWorkerTaskStateEnum.READY
                        ),
                    )
                    for worker_uuid, worker_id in worker_ids.items()
                ]
            )
            await session.commit()
        controller = ModelPreheatController(
            engine, FakeReadyProbe(), FakeInventoryProbe({})
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent.execution_state

    assert asyncio.run(scenario(False)) == ModelPreheatExecutionStateEnum.READY
    assert asyncio.run(scenario(True)) == ModelPreheatExecutionStateEnum.PARTIAL


def test_same_uuid_reregistration_moves_pending_task_to_latest_worker_id(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, worker_ids = await _seed(engine, ("worker-a",))
        controller = ModelPreheatController(
            engine, FakeReadyProbe(), FakeInventoryProbe({})
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            child = (await session.exec(select(ModelPreheatWorkerTask))).one()
            child.state = ModelPreheatWorkerTaskStateEnum.RUNNING
            child.lease_owner = "worker-a"
            child.lease_token_hash = "old-token"
            child.lease_expires_at = datetime.now(timezone.utc)
            replacement = Worker(
                name="worker-a-new",
                hostname="worker-a-new",
                ip="127.0.0.9",
                port=10150,
                worker_uuid="worker-a",
                state=WorkerStateEnum.READY,
            )
            session.add_all([child, replacement])
            await session.commit()
            await session.refresh(replacement)
            replacement_id = replacement.id
        await controller.reconcile_task(task_id)
        children = await _tasks(engine, task_id)
        await engine.dispose()
        return worker_ids["worker-a"], replacement_id, children[0]

    old_id, new_id, child = asyncio.run(run())
    assert new_id > old_id
    assert child.worker_id == new_id
    assert child.state == ModelPreheatWorkerTaskStateEnum.PENDING
    assert child.lease_owner is None
    assert child.lease_token_hash is None


def test_probe_result_cannot_overwrite_parent_paused_while_probe_is_running(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        probe = BlockingReadyProbe()
        controller = ModelPreheatController(engine, probe, FakeInventoryProbe({}))
        reconcile = asyncio.create_task(controller.reconcile_task(task_id))
        await probe.started.wait()
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            parent.desired_state = ModelPreheatDesiredStateEnum.PAUSED
            parent.execution_state = ModelPreheatExecutionStateEnum.PAUSED
            parent.paused_from_state = ModelPreheatExecutionStateEnum.PENDING
            session.add(parent)
            await session.commit()
        probe.release.set()
        await reconcile
        children = await _tasks(engine, task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent, children

    parent, children = asyncio.run(run())
    assert parent.execution_state == ModelPreheatExecutionStateEnum.PAUSED
    assert children == []


def test_probe_result_cannot_create_old_attempt_child_after_retry(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        probe = BlockingReadyProbe()
        controller = ModelPreheatController(engine, probe, FakeInventoryProbe({}))
        reconcile = asyncio.create_task(controller.reconcile_task(task_id))
        await probe.started.wait()
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            parent.attempt = 2
            parent.execution_state = ModelPreheatExecutionStateEnum.PENDING
            session.add(parent)
            await session.commit()
        probe.release.set()
        await reconcile
        children = await _tasks(engine, task_id)
        await engine.dispose()
        return children

    assert asyncio.run(run()) == []


def test_latest_non_ready_registration_hides_older_ready_registration(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine, ("worker-a",))
        async with AsyncSession(engine) as session:
            session.add(
                Worker(
                    name="worker-a-new",
                    hostname="worker-a-new",
                    ip="127.0.0.9",
                    port=10150,
                    worker_uuid="worker-a",
                    state=WorkerStateEnum.NOT_READY,
                )
            )
            await session.commit()
        controller = ModelPreheatController(
            engine, FakeReadyProbe(), FakeInventoryProbe({})
        )
        await controller.reconcile_task(task_id)
        children = await _tasks(engine, task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent, children

    parent, children = asyncio.run(run())
    assert parent.execution_state == ModelPreheatExecutionStateEnum.ERROR
    assert all(
        child.state == ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED
        for child in children
    )


def test_stale_local_hits_do_not_inflate_current_target_aggregation(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, worker_ids = await _seed(engine, ("worker-a",))
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            parent.execution_state = ModelPreheatExecutionStateEnum.DISTRIBUTING
            parent.local_cache_hit_worker_uuids = ["removed-worker"]
            session.add(parent)
            session.add(
                ModelPreheatWorkerTask(
                    task_id=task_id,
                    parent_attempt=1,
                    worker_uuid="worker-a",
                    worker_id=worker_ids["worker-a"],
                    role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                    state=ModelPreheatWorkerTaskStateEnum.READY,
                )
            )
            await session.commit()
        controller = ModelPreheatController(
            engine, FakeReadyProbe(), FakeInventoryProbe({})
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent

    parent = asyncio.run(run())
    assert parent.execution_state == ModelPreheatExecutionStateEnum.READY
    assert parent.local_cache_hit_worker_uuids == []


def test_seed_ready_without_strict_ready_is_skipped_and_reselected(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        controller = ModelPreheatController(
            engine, FakeReadyProbe(), FakeInventoryProbe({})
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            seed = (await session.exec(select(ModelPreheatWorkerTask))).one()
            seed.state = ModelPreheatWorkerTaskStateEnum.READY
            seed.resumable_cursor = {
                "manifest_digest": "d" * 64,
                "generation_id": GENERATION_ID,
                "local_cache_state": "valid",
            }
            session.add(seed)
            await session.commit()
        await controller.reconcile_task(task_id)
        children = await _tasks(engine, task_id)
        await engine.dispose()
        return children

    children = asyncio.run(run())
    assert not any(
        child.role == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE for child in children
    )
    assert children[0].state == ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED
    assert children[1].role == ModelPreheatWorkerTaskRoleEnum.SEED
    assert children[1].worker_uuid == "worker-b"


def test_seed_ready_digest_mismatch_never_distributes(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        ready_probe = FakeReadyProbe()
        controller = ModelPreheatController(engine, ready_probe, FakeInventoryProbe({}))
        await controller.reconcile_task(task_id)
        ready_probe.result = ReadyProbeResult(
            manifest_digest="e" * 64,
            generation_id=GENERATION_ID,
            ready_path="model-cache/v1/ready.json",
            manifest_path="model-cache/v1/manifest.json",
        )
        async with AsyncSession(engine) as session:
            seed = (await session.exec(select(ModelPreheatWorkerTask))).one()
            seed.state = ModelPreheatWorkerTaskStateEnum.READY
            seed.resumable_cursor = {
                "manifest_digest": "d" * 64,
                "generation_id": GENERATION_ID,
                "local_cache_state": "valid",
            }
            session.add(seed)
            await session.commit()
        await controller.reconcile_task(task_id)
        children = await _tasks(engine, task_id)
        await engine.dispose()
        return children

    children = asyncio.run(run())
    assert not any(
        child.role == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE for child in children
    )


def test_error_seed_is_skipped_and_reselected(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        controller = ModelPreheatController(
            engine, FakeReadyProbe(), FakeInventoryProbe({})
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            seed = (await session.exec(select(ModelPreheatWorkerTask))).one()
            seed.state = ModelPreheatWorkerTaskStateEnum.ERROR
            seed.error_code = "network_timeout"
            session.add(seed)
            await session.commit()
        await controller.reconcile_task(task_id)
        children = await _tasks(engine, task_id)
        await engine.dispose()
        return children

    children = asyncio.run(run())
    assert children[0].state == ModelPreheatWorkerTaskStateEnum.SKIPPED_WORKER_REMOVED
    assert children[1].role == ModelPreheatWorkerTaskRoleEnum.SEED
    assert children[1].worker_uuid == "worker-b"


def test_error_seed_moves_to_latest_same_uuid_registration(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine, ("worker-a",))
        controller = ModelPreheatController(
            engine, FakeReadyProbe(), FakeInventoryProbe({})
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            seed = (await session.exec(select(ModelPreheatWorkerTask))).one()
            seed.state = ModelPreheatWorkerTaskStateEnum.ERROR
            seed.error_code = "worker_restarted"
            session.add(seed)
            replacement = Worker(
                name="worker-a-new",
                hostname="worker-a-new",
                ip="127.0.0.9",
                port=10150,
                worker_uuid="worker-a",
                state=WorkerStateEnum.READY,
            )
            session.add(replacement)
            await session.commit()
            await session.refresh(replacement)
            replacement_id = replacement.id
        await controller.reconcile_task(task_id)
        children = await _tasks(engine, task_id)
        await engine.dispose()
        return replacement_id, children

    replacement_id, children = asyncio.run(run())
    assert len(children) == 1
    assert children[0].worker_id == replacement_id
    assert children[0].state == ModelPreheatWorkerTaskStateEnum.PENDING
    assert children[0].error_code is None
