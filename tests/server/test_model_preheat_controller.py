import asyncio
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_files import ModelFile  # noqa: F401
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
)  # noqa: F401
from gpustack.schemas.model_preheats import (
    ModelPreheatDeliveryModeEnum,
    ModelPreheatDesiredStateEnum,
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.model_preheat_controller import (
    LocalInventoryProbeResult,
    ModelPreheatController,
)
from gpustack.server.bus import EventType
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity


class FakeInventoryProbe:
    def __init__(self, states):
        self.states = states
        self.probed = []

    async def probe(self, task, worker_uuids):
        self.probed.append(tuple(worker_uuids))
        return {
            worker_uuid: LocalInventoryProbeResult(
                worker_uuid,
                self.states.get(worker_uuid, "missing"),
                source=("model_file" if worker_uuid in self.states else None),
            )
            for worker_uuid in worker_uuids
        }


async def _database(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'controller.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return engine


async def _seed(engine, targets=("worker-a", "worker-b"), artifact_id=None):
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="a" * 40,
        requested_revision="master",
        file_patterns=(),
    )
    async with AsyncSession(engine) as session:
        workers = {}
        for name in ("worker-a", "worker-b", "worker-peer"):
            worker = Worker(
                name=name,
                hostname=name,
                ip="127.0.0.1",
                port=10150,
                worker_uuid=name,
                state=WorkerStateEnum.READY,
                model_storage_protocol_version=1,
            )
            session.add(worker)
            workers[name] = worker
        await session.flush()
        task = ModelPreheatTask(
            source="modelscope",
            model_id="org/model",
            requested_revision="master",
            resolved_revision="a" * 40,
            include_patterns=[],
            exclude_patterns=[],
            selection_digest="b" * 64,
            request_identity={
                "source": identity.source,
                "model_id": identity.model_path,
                "requested_revision": identity.requested_revision_path,
                "include_patterns": [],
                "exclude_patterns": [],
            },
            request_digest=identity.request_digest,
            artifact_id=artifact_id,
            seed_worker_uuid=targets[0],
            seed_worker_id=workers[targets[0]].id,
            target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
            target_worker_uuids=list(targets),
            target_worker_snapshot=[],
            s3_profile_id=1,
            s3_profile_config_version=3,
            s3_profile_snapshot_encrypted={"ciphertext": "encrypted"},
            encryption_key_version="v1",
            s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
            s3_manifest_path=(
                f"model-storage/modelscope/org/model/{artifact_id}/manifest.json"
                if artifact_id
                else None
            ),
        )
        session.add(task)
        worker_ids = {name: worker.id for name, worker in workers.items()}
        await session.commit()
        await session.refresh(task)
        return task.id, worker_ids


async def _children(engine, task_id):
    async with AsyncSession(engine) as session:
        return (
            await session.exec(
                select(ModelPreheatWorkerTask)
                .where(ModelPreheatWorkerTask.task_id == task_id)
                .order_by(ModelPreheatWorkerTask.id)
            )
        ).all()


def test_bound_s3_artifact_precedes_local_candidate(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine, artifact_id="c" * 64)
        probe = FakeInventoryProbe({"worker-b": "candidate"})
        await ModelPreheatController(engine, inventory_probe=probe).reconcile_task(
            task_id
        )
        children = await _children(engine, task_id)
        await engine.dispose()
        return children, probe.probed

    children, probed = asyncio.run(run())
    assert probed == [("worker-a", "worker-b", "worker-peer")]
    assert [(child.worker_uuid, child.role) for child in children] == [
        ("worker-a", ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE),
        ("worker-b", ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE),
    ]


def test_peer_candidate_is_seeded_once_before_public_source(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        probe = FakeInventoryProbe({"worker-peer": "candidate"})
        await ModelPreheatController(engine, inventory_probe=probe).reconcile_task(
            task_id
        )
        children = await _children(engine, task_id)
        await engine.dispose()
        return children

    children = asyncio.run(run())
    assert len(children) == 1
    assert children[0].worker_uuid == "worker-peer"
    assert children[0].role == ModelPreheatWorkerTaskRoleEnum.SEED


def test_bound_artifact_distributes_without_seed(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine, artifact_id="c" * 64)
        await ModelPreheatController(
            engine, inventory_probe=FakeInventoryProbe({})
        ).reconcile_task(task_id)
        children = await _children(engine, task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent, children

    parent, children = asyncio.run(run())
    assert parent.execution_state == ModelPreheatExecutionStateEnum.DISTRIBUTING
    assert parent.transfer_source == "s3"
    assert {child.role for child in children} == {
        ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE
    }
    assert {child.worker_uuid for child in children} == {"worker-a", "worker-b"}


def test_s3_only_bound_artifact_completes_without_workers(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine, targets=("worker-a",), artifact_id="c" * 64)
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatTask, task_id)
            task.delivery_mode = ModelPreheatDeliveryModeEnum.S3_ONLY
            task.seed_worker_uuid = None
            task.seed_worker_id = None
            task.target_worker_uuids = []
            session.add(task)
            await session.commit()
        await ModelPreheatController(engine).reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent

    parent = asyncio.run(run())
    assert parent.execution_state == ModelPreheatExecutionStateEnum.READY
    assert parent.transfer_source == "s3"


def test_all_locations_missing_uses_requested_seed_for_public_fallback(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)
        await ModelPreheatController(
            engine, inventory_probe=FakeInventoryProbe({})
        ).reconcile_task(task_id)
        children = await _children(engine, task_id)
        await engine.dispose()
        return children

    children = asyncio.run(run())
    assert [(child.worker_uuid, child.role) for child in children] == [
        ("worker-a", ModelPreheatWorkerTaskRoleEnum.SEED)
    ]


def test_inventory_probe_pause_does_not_create_stale_attempt_child(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)

        class PausingProbe(FakeInventoryProbe):
            async def probe(self, task, worker_uuids):
                async with AsyncSession(engine) as session:
                    current = await session.get(ModelPreheatTask, task.id)
                    current.desired_state = ModelPreheatDesiredStateEnum.PAUSED
                    session.add(current)
                    await session.commit()
                return await super().probe(task, worker_uuids)

        await ModelPreheatController(
            engine, inventory_probe=PausingProbe({})
        ).reconcile_task(task_id)
        children = await _children(engine, task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent, children

    parent, children = asyncio.run(run())
    assert parent.desired_state == ModelPreheatDesiredStateEnum.PAUSED
    assert parent.execution_state == ModelPreheatExecutionStateEnum.PENDING
    assert children == []


def test_inventory_probe_retry_does_not_create_old_attempt_child(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, _ = await _seed(engine)

        class RetryingProbe(FakeInventoryProbe):
            async def probe(self, task, worker_uuids):
                async with AsyncSession(engine) as session:
                    current = await session.get(ModelPreheatTask, task.id)
                    current.attempt += 1
                    session.add(current)
                    await session.commit()
                return await super().probe(task, worker_uuids)

        await ModelPreheatController(
            engine, inventory_probe=RetryingProbe({})
        ).reconcile_task(task_id)
        children = await _children(engine, task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent, children

    parent, children = asyncio.run(run())
    assert parent.attempt == 2
    assert parent.execution_state == ModelPreheatExecutionStateEnum.PENDING
    assert children == []


def test_seed_completion_uses_bound_artifact_for_distribution(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, worker_ids = await _seed(engine)
        controller = ModelPreheatController(
            engine, inventory_probe=FakeInventoryProbe({})
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            parent.artifact_id = "d" * 64
            parent.s3_manifest_path = f"prefix/{'d' * 64}/manifest.json"
            parent.manifest_digest = "e" * 64
            seed = (await session.exec(select(ModelPreheatWorkerTask))).one()
            seed.state = ModelPreheatWorkerTaskStateEnum.READY
            seed.resumable_cursor = {
                "state": "ready",
                "artifact_id": "d" * 64,
                "local_cache_state": "valid",
            }
            session.add_all([parent, seed])
            await session.commit()
        await controller.reconcile_task(task_id)
        children = await _children(engine, task_id)
        await engine.dispose()
        return children, worker_ids

    children, _ = asyncio.run(run())
    assert (
        len(
            [
                child
                for child in children
                if child.role == ModelPreheatWorkerTaskRoleEnum.SEED
            ]
        )
        == 1
    )
    assert [
        child.worker_uuid
        for child in children
        if child.role == ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE
    ] == ["worker-b"]


def test_ready_seed_reconcile_backfills_worker_model_file(tmp_path, monkeypatch):
    published = []

    async def capture_event(event_type, model_file):
        published.append((event_type, model_file.id, model_file.state))

    monkeypatch.setattr(ModelFile, "_publish_event", capture_event)

    async def run():
        engine = await _database(tmp_path)
        task_id, worker_ids = await _seed(engine)
        controller = ModelPreheatController(
            engine, inventory_probe=FakeInventoryProbe({})
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            parent.artifact_id = "d" * 64
            parent.s3_manifest_path = f"prefix/{'d' * 64}/manifest.json"
            parent.manifest_digest = "e" * 64
            seed = (await session.exec(select(ModelPreheatWorkerTask))).one()
            seed.state = ModelPreheatWorkerTaskStateEnum.READY
            seed.progress = 100
            seed.total_size = 123
            seed.resumable_cursor = {
                "state": "ready",
                "request_digest": parent.request_digest,
                "artifact_id": "d" * 64,
                "local_cache_state": "valid",
                "resolved_revision": parent.resolved_revision,
                "local_dir": "/models/org/model",
                "resolved_paths": ["/models/org/model"],
                "total_size": 123,
            }
            session.add_all([parent, seed])
            await session.commit()
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            model_file = (
                await session.exec(
                    select(ModelFile).where(ModelFile.worker_id == worker_ids["worker-a"])
                )
            ).one()
        await engine.dispose()
        return model_file

    model_file = asyncio.run(run())
    assert model_file.state.value == "ready"
    assert model_file.model_scope_model_id == "org/model"
    assert model_file.local_dir == "/models/org/model"
    assert model_file.resolved_paths == ["/models/org/model"]
    assert published == [(EventType.CREATED, model_file.id, model_file.state)]


def test_ready_seed_backfill_uses_current_worker_registration(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, worker_ids = await _seed(engine)
        controller = ModelPreheatController(
            engine, inventory_probe=FakeInventoryProbe({})
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            parent.artifact_id = "d" * 64
            parent.s3_manifest_path = f"prefix/{'d' * 64}/manifest.json"
            parent.manifest_digest = "e" * 64
            seed = (await session.exec(select(ModelPreheatWorkerTask))).one()
            seed.state = ModelPreheatWorkerTaskStateEnum.READY
            seed.progress = 100
            seed.total_size = 123
            seed.resumable_cursor = {
                "state": "ready",
                "request_digest": parent.request_digest,
                "artifact_id": "d" * 64,
                "local_cache_state": "valid",
                "resolved_revision": parent.resolved_revision,
                "local_dir": "/models/org/model",
                "resolved_paths": ["/models/org/model"],
                "total_size": 123,
            }
            replacement = Worker(
                name="worker-a-new",
                hostname="worker-a-new",
                ip="127.0.0.2",
                port=10150,
                worker_uuid="worker-a",
                state=WorkerStateEnum.READY,
                model_storage_protocol_version=1,
            )
            session.add_all([parent, seed, replacement])
            await session.commit()
            await session.refresh(replacement)
            replacement_id = replacement.id
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            model_files = (await session.exec(select(ModelFile))).all()
        await engine.dispose()
        return model_files, worker_ids["worker-a"], replacement_id

    model_files, old_worker_id, replacement_id = asyncio.run(run())
    assert len(model_files) == 1
    assert model_files[0].worker_id == replacement_id
    assert model_files[0].worker_id != old_worker_id
    assert model_files[0].worker_name_snapshot == "worker-a-new"


def test_terminal_ready_seed_backfills_worker_model_file(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, worker_ids = await _seed(engine)
        controller = ModelPreheatController(
            engine, inventory_probe=FakeInventoryProbe({})
        )
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            parent.artifact_id = "d" * 64
            parent.s3_manifest_path = f"prefix/{'d' * 64}/manifest.json"
            parent.manifest_digest = "e" * 64
            parent.execution_state = ModelPreheatExecutionStateEnum.READY
            parent.finished_at = datetime.now(timezone.utc)
            seed = (await session.exec(select(ModelPreheatWorkerTask))).one()
            seed.state = ModelPreheatWorkerTaskStateEnum.READY
            seed.progress = 100
            seed.total_size = 123
            seed.resumable_cursor = {
                "state": "ready",
                "request_digest": parent.request_digest,
                "artifact_id": "d" * 64,
                "local_cache_state": "valid",
                "resolved_revision": parent.resolved_revision,
                "local_dir": "/models/org/model",
                "resolved_paths": ["/models/org/model"],
                "total_size": 123,
            }
            session.add_all([parent, seed])
            await session.commit()
        await controller.reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            model_file = (
                await session.exec(
                    select(ModelFile).where(ModelFile.worker_id == worker_ids["worker-a"])
                )
            ).one()
        await engine.dispose()
        return model_file

    model_file = asyncio.run(run())
    assert model_file.state.value == "ready"
    assert model_file.model_scope_model_id == "org/model"
    assert model_file.local_dir == "/models/org/model"


def test_distribution_terminal_states_aggregate_partial(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        task_id, worker_ids = await _seed(engine, artifact_id="c" * 64)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
            parent.execution_state = ModelPreheatExecutionStateEnum.DISTRIBUTING
            session.add(parent)
            for index, worker_uuid in enumerate(("worker-a", "worker-b")):
                session.add(
                    ModelPreheatWorkerTask(
                        task_id=task_id,
                        worker_uuid=worker_uuid,
                        worker_id=worker_ids[worker_uuid],
                        role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                        state=(
                            ModelPreheatWorkerTaskStateEnum.READY
                            if index == 0
                            else ModelPreheatWorkerTaskStateEnum.ERROR
                        ),
                        finished_at=datetime.now(timezone.utc),
                    )
                )
            await session.commit()
        await ModelPreheatController(engine).reconcile_task(task_id)
        async with AsyncSession(engine) as session:
            parent = await session.get(ModelPreheatTask, task_id)
        await engine.dispose()
        return parent.execution_state

    assert asyncio.run(run()) == ModelPreheatExecutionStateEnum.PARTIAL
