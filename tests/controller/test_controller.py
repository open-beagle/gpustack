import asyncio
from datetime import datetime
from typing import List
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from gpustack.policies.base import ModelInstanceScore

from gpustack.schemas.links import ModelInstanceModelFileLink
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import (
    ComputedResourceClaim,
    GPUSelector,
    Model,
    ModelInstance,
    ModelPlacementOverride,
    PlacementOverrideReplicaGroup,
    ModelInstanceStateEnum,
    SourceEnum,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.controllers import (
    ModelInstanceController,
    WorkerController,
    ensure_instance_model_file,
    ensure_model_instance_file_links,
    find_scale_down_candidates,
    get_model_instance_ids_for_model_file,
    get_model_files_for_instance,
    safe_event_attr,
    sync_replicas,
)
from gpustack.server.bus import Event, EventType
from tests.fixtures.workers.fixtures import (
    linux_nvidia_19_4090_24gx2,
    linux_nvidia_2_4080_16gx2,
    linux_cpu_1,
)

from unittest.mock import patch, AsyncMock

from tests.utils.model import new_model, new_model_instance


class FakeAsyncSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class ExpiredWorkerEventData:
    def __bool__(self):
        return True

    @property
    def name(self):
        raise RuntimeError("expired worker name")


class FakeExecResult:
    def __init__(self, items):
        self.items = items

    def all(self):
        return self.items


class FakeSession:
    def __init__(self, exec_items=None):
        self.exec_items = exec_items or []
        self.added = []
        self.flushed = False

    async def exec(self, statement):
        self.statement = statement
        return FakeExecResult(self.exec_items)

    def add(self, item):
        self.added.append(item)

    async def flush(self):
        self.flushed = True


def test_model_instance_deleted_event_reconciles_parent_model():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    deleted_instance = new_model_instance(
        10,
        "test-10",
        model.id,
        state=ModelInstanceStateEnum.RUNNING,
    )
    controller = ModelInstanceController.__new__(ModelInstanceController)
    controller._engine = object()
    controller._config = AsyncMock()
    session = AsyncMock()

    sync_replicas_mock = AsyncMock()
    sync_ready_replicas_mock = AsyncMock()

    with (
        patch(
            "gpustack.server.controllers.AsyncSession",
            return_value=FakeAsyncSessionContext(session),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.one_by_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "gpustack.server.controllers.Model.one_by_id",
            new=AsyncMock(return_value=model),
        ),
        patch("gpustack.server.controllers.sync_replicas", new=sync_replicas_mock),
        patch(
            "gpustack.server.controllers.sync_ready_replicas",
            new=sync_ready_replicas_mock,
        ),
    ):
        asyncio.run(
            controller._reconcile(Event(type=EventType.DELETED, data=deleted_instance))
        )

    sync_replicas_mock.assert_awaited_once_with(session, model, controller._config)
    sync_ready_replicas_mock.assert_awaited_once_with(session, model)


def test_worker_reconcile_skips_expired_event_data_without_implicit_io():
    controller = WorkerController.__new__(WorkerController)
    controller._engine = object()
    session = AsyncMock()
    list_mock = AsyncMock()

    with (
        patch(
            "gpustack.server.controllers.AsyncSession",
            return_value=FakeAsyncSessionContext(session),
        ),
        patch("gpustack.server.controllers.ModelInstance.all_by_field", new=list_mock),
    ):
        asyncio.run(
            controller._reconcile(
                Event(type=EventType.UPDATED, data=ExpiredWorkerEventData())
            )
        )

    list_mock.assert_not_awaited()


def test_worker_reconcile_deletes_multiple_instances_after_each_commit(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'worker-instance-delete.db'}"
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    Model.__table__,
                    ModelInstance.__table__,
                    ModelFile.__table__,
                    ModelInstanceModelFileLink.__table__,
                ],
            )

        async with AsyncSession(engine) as session:
            session.add(
                Model(
                    id=3,
                    name="model-a",
                    source=SourceEnum.LOCAL_PATH,
                    local_path="/models/a",
                )
            )
            session.add_all(
                [
                    ModelInstance(
                        name="instance-1",
                        model_id=3,
                        model_name="model-a",
                        worker_name="worker-a",
                        source=SourceEnum.LOCAL_PATH,
                        local_path="/models/a",
                    ),
                    ModelInstance(
                        name="instance-2",
                        model_id=3,
                        model_name="model-a",
                        worker_name="worker-a",
                        source=SourceEnum.LOCAL_PATH,
                        local_path="/models/a",
                    ),
                ]
            )
            await session.commit()

        controller = WorkerController.__new__(WorkerController)
        controller._engine = engine
        with patch(
            "gpustack.server.services.delete_cache_by_key", new=AsyncMock()
        ):
            await controller._reconcile(
                Event(
                    type=EventType.DELETED,
                    data=Worker(
                        name="worker-a",
                        hostname="worker-a",
                        ip="127.0.0.1",
                        port=10150,
                        worker_uuid="worker-a-uuid",
                        state=WorkerStateEnum.READY,
                    ),
                ),
        )

        async with AsyncSession(engine) as session:
            instances = await ModelInstance.all_by_field(
                session, "worker_name", "worker-a"
            )
            assert instances == []
        await engine.dispose()

    asyncio.run(run())


def test_worker_deleted_event_does_not_restore_unreachable_instances(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'worker-unreachable-delete.db'}"
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    Model.__table__,
                    ModelInstance.__table__,
                    ModelFile.__table__,
                    ModelInstanceModelFileLink.__table__,
                ],
            )

        async with AsyncSession(engine) as session:
            session.add(
                Model(
                    id=3,
                    name="model-a",
                    source=SourceEnum.LOCAL_PATH,
                    local_path="/models/a",
                )
            )
            session.add(
                ModelInstance(
                    name="instance-unreachable",
                    model_id=3,
                    model_name="model-a",
                    worker_name="worker-a",
                    source=SourceEnum.LOCAL_PATH,
                    local_path="/models/a",
                    state=ModelInstanceStateEnum.UNREACHABLE,
                )
            )
            await session.commit()

        controller = WorkerController.__new__(WorkerController)
        controller._engine = engine
        with patch(
            "gpustack.server.services.delete_cache_by_key", new=AsyncMock()
        ):
            await controller._reconcile(
                Event(
                    type=EventType.DELETED,
                    data=Worker(
                        name="worker-a",
                        hostname="worker-a",
                        ip="127.0.0.1",
                        port=10150,
                        worker_uuid="worker-a-uuid",
                        state=WorkerStateEnum.READY,
                    ),
                ),
        )

        async with AsyncSession(engine) as session:
            instances = await ModelInstance.all_by_field(
                session, "worker_name", "worker-a"
            )
            assert instances == []
        await engine.dispose()

    asyncio.run(run())


def test_worker_reconcile_reloads_attached_expired_event_by_identity(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'worker.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all, tables=[Worker.__table__]
            )
        async with AsyncSession(engine, expire_on_commit=True) as real_session:
            expired = Worker(
                name="worker-a",
                hostname="worker-a",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="worker-a-uuid",
                state=WorkerStateEnum.READY,
            )
            real_session.add(expired)
            await real_session.commit()
            assert expired.model_dump() == {}

            controller = WorkerController.__new__(WorkerController)
            controller._engine = object()
            fake_session = AsyncMock()
            current = Worker(
                id=1,
                name="worker-a",
                hostname="worker-a",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="worker-a-uuid",
                state=WorkerStateEnum.READY,
            )
            reload_mock = AsyncMock(return_value=current)
            list_mock = AsyncMock(return_value=[])
            with (
                patch(
                    "gpustack.server.controllers.AsyncSession",
                    return_value=FakeAsyncSessionContext(fake_session),
                ),
                patch(
                    "gpustack.server.controllers.Worker.one_by_id",
                    new=reload_mock,
                ),
                patch(
                    "gpustack.server.controllers.ModelInstance.all_by_field",
                    new=list_mock,
                ),
            ):
                await controller._reconcile(Event(type=EventType.UPDATED, data=expired))

            reload_mock.assert_awaited_once_with(fake_session, 1)
            list_mock.assert_awaited_once_with(fake_session, "worker_name", "worker-a")
        await engine.dispose()

    asyncio.run(run())


def test_ensure_model_instance_file_links_adds_missing_links():
    session = FakeSession()
    instance = new_model_instance(376, "test-376", 80)
    model_files = [
        ModelFile(id=149, source=SourceEnum.MODEL_SCOPE, worker_id=2),
        ModelFile(id=150, source=SourceEnum.MODEL_SCOPE, worker_id=2),
    ]

    asyncio.run(ensure_model_instance_file_links(session, instance, model_files))

    assert [(link.model_instance_id, link.model_file_id) for link in session.added] == [
        (376, 149),
        (376, 150),
    ]
    assert session.flushed is True


def test_ensure_model_instance_file_links_skips_existing_links():
    session = FakeSession(
        [
            ModelInstanceModelFileLink(
                model_instance_id=376,
                model_file_id=149,
            )
        ]
    )
    instance = new_model_instance(376, "test-376", 80)
    model_files = [
        ModelFile(id=149, source=SourceEnum.MODEL_SCOPE, worker_id=2),
        ModelFile(id=150, source=SourceEnum.MODEL_SCOPE, worker_id=2),
    ]

    asyncio.run(ensure_model_instance_file_links(session, instance, model_files))

    assert [(link.model_instance_id, link.model_file_id) for link in session.added] == [
        (376, 150),
    ]


def test_get_model_instance_ids_for_model_file_reads_link_table():
    session = FakeSession(
        [
            ModelInstanceModelFileLink(model_instance_id=376, model_file_id=149),
            ModelInstanceModelFileLink(model_instance_id=None, model_file_id=149),
        ]
    )

    instance_ids = asyncio.run(get_model_instance_ids_for_model_file(session, 149))

    assert instance_ids == [376]


def test_get_model_files_for_instance_filters_same_source():
    class FakeModelFileService:
        def __init__(self, session):
            pass

        async def get_by_source_index(self, source_index):
            return [
                ModelFile(
                    id=89,
                    source=SourceEnum.MODEL_SCOPE,
                    model_scope_model_id="BAAI/bge-reranker-v2-m3",
                    worker_id=1,
                ),
                ModelFile(
                    id=113,
                    source=SourceEnum.HUGGING_FACE,
                    huggingface_repo_id="BAAI/bge-reranker-v2-m3",
                    worker_id=1,
                ),
            ]

    instance = new_model_instance(1, "bge-reranker", 6, worker_id=1)
    instance.source = SourceEnum.HUGGING_FACE
    instance.huggingface_repo_id = "BAAI/bge-reranker-v2-m3"

    with patch("gpustack.server.controllers.ModelFileService", FakeModelFileService):
        model_files = asyncio.run(get_model_files_for_instance(FakeSession(), instance))

    assert [model_file.id for model_file in model_files] == [113]


def test_ensure_instance_model_file_links_existing_files_before_sync():
    session = AsyncMock()
    instance = new_model_instance(
        376,
        "test-376",
        80,
        worker_id=2,
        state=ModelInstanceStateEnum.INITIALIZING,
    )
    model_file = ModelFile(
        id=149,
        source=SourceEnum.MODEL_SCOPE,
        worker_id=2,
        state=ModelFileStateEnum.READY,
        resolved_paths=["/var/lib/gpustack/cache/test.gguf"],
    )
    link_mock = AsyncMock()
    sync_mock = AsyncMock()

    with (
        patch(
            "gpustack.server.controllers.get_model_files_for_instance",
            new=AsyncMock(return_value=[model_file]),
        ),
        patch(
            "gpustack.server.controllers.ModelInstance.one_by_id",
            new=AsyncMock(return_value=instance),
        ),
        patch(
            "gpustack.server.controllers.ensure_model_instance_file_links",
            new=link_mock,
        ),
        patch(
            "gpustack.server.controllers.sync_instance_files_state",
            new=sync_mock,
        ),
    ):
        asyncio.run(ensure_instance_model_file(session, instance))

    link_mock.assert_awaited_once_with(session, instance, [model_file])
    sync_mock.assert_awaited_once_with(session, instance, [model_file])


def test_find_scale_down_candidates():
    w1 = linux_nvidia_19_4090_24gx2()
    w1.state = WorkerStateEnum.NOT_READY
    workers = [
        w1,
        linux_nvidia_2_4080_16gx2(),
        linux_cpu_1(),
    ]

    m = new_model(1, "test", 3, "llama3:70b")
    mis = [
        new_model_instance(
            1,
            "test-1",
            1,
            4,
            ModelInstanceStateEnum.RUNNING,
            [0, 1],
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=81,
                total_layers=81,
                ram=455165112,
                vram={0: 22912443392, 1: 22911897600},
            ),
        ),
        new_model_instance(
            2,
            "test-2",
            1,
            3,
            ModelInstanceStateEnum.RUNNING,
            [0, 1],
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=60,
                total_layers=81,
                ram=1093245112,
                vram={0: 16900820992, 1: 16900820992},
            ),
        ),
        new_model_instance(
            3,
            "test-3",
            1,
            6,
            ModelInstanceStateEnum.RUNNING,
            None,
            ComputedResourceClaim(
                is_unified_memory=False,
                offload_layers=0,
                total_layers=81,
                ram=3106511032,
            ),
        ),
    ]

    with (
        patch('sqlmodel.ext.asyncio.session.AsyncSession', AsyncMock()),
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            return_value=mis,
        ),
        patch(
            'gpustack.schemas.workers.Worker.all',
            return_value=workers,
        ),
    ):

        candidates = asyncio.run(find_scale_down_candidates(mis, m))

        expected_candidates = [
            {
                "worker_id": 4,
                "instacnce_id": 1,
                "gpu_indexes": [0, 1],
                "score": 9.538995598356342,
            },
            {
                "worker_id": 6,
                "instacnce_id": 3,
                "score": 90.1308159326069,
            },
            {
                "worker_id": 3,
                "instacnce_id": 2,
                "score": 97.3594505895714,
            },
        ]

        compare_candidates(candidates, expected_candidates)


def compare_candidates(candidates: List[ModelInstanceScore], expected_candidates):
    for i, expected in enumerate(expected_candidates):
        candidate = candidates[i]
        instance = candidate.model_instance

        if "worker_id" in expected:
            assert instance.worker_id == expected["worker_id"]

        if "instance_id" in expected:
            assert instance.id == expected["instance_id"]

        if "score" in expected:
            assert str(candidate.score)[:5] == str(expected["score"])[:5]


def test_model_event_data_is_safe_to_access_after_publish():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    event = Event(type=EventType.UPDATED, data=model.model_copy(deep=True))

    assert event.data.id == 1
    assert event.data.name == "test"


def test_convert_to_public_class_handles_model_construct_snapshot():
    from gpustack.schemas.models import ModelInstance

    instance = new_model_instance(6, "test-6", 1)
    instance.source = new_model(1, "test", huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct").source
    instance.__dict__["created_at"] = datetime.now()
    instance.__dict__["updated_at"] = datetime.now()
    snapshot = instance.__class__.model_construct(**instance.model_dump())

    public = ModelInstance._convert_to_public_class(snapshot)

    assert not hasattr(snapshot, "_sa_instance_state")
    assert public["id"] == 6
    assert public["name"] == "test-6"


def test_convert_to_public_class_uses_public_schema_for_complete_data():
    from gpustack.schemas.models import ModelInstance

    now = datetime.now()
    instance = new_model_instance(7, "test-7", 1)
    instance.source = new_model(1, "test", huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct").source
    instance.huggingface_repo_id = "Qwen/Qwen2.5-7B-Instruct"
    instance.__dict__["created_at"] = now
    instance.__dict__["updated_at"] = now

    public = ModelInstance._convert_to_public_class(instance)

    assert public.id == 7
    assert public.name == "test-7"


def test_safe_event_attr_handles_detached_orm_attribute_errors():
    class DetachedLike:
        @property
        def name(self):
            raise RuntimeError("detached")

    assert safe_event_attr(DetachedLike(), "name") == "<unknown>"


def test_sync_replicas_puts_placement_override_on_new_instances_only():
    model = new_model(1, "test", 2, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    created_instances = []

    async def mock_create(instance):
        created_instances.append(instance)
        return instance

    placement_override = ModelPlacementOverride(
        replica_groups=[
            PlacementOverrideReplicaGroup(
                gpu_selector=GPUSelector(gpu_ids=["host4090:cuda:0"])
            ),
            PlacementOverrideReplicaGroup(
                gpu_selector=GPUSelector(
                    gpu_ids=["host4080:cuda:0", "host4080:cuda:1"]
                )
            ),
        ]
    )

    with (
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            return_value=[],
        ),
        patch(
            'gpustack.server.services.ModelInstanceService.create',
            side_effect=mock_create,
        ),
    ):
        asyncio.run(
            sync_replicas(
                AsyncMock(),
                model,
                AsyncMock(),
                placement_override=placement_override,
            )
        )

    assert model.gpu_selector is None
    assert len(created_instances) == 2
    assert created_instances[0].placement_override.gpu_selector.gpu_ids == [
        "host4090:cuda:0"
    ]
    assert created_instances[1].placement_override.gpu_selector.gpu_ids == [
        "host4080:cuda:0",
        "host4080:cuda:1",
    ]


def test_sync_replicas_interprets_replica_groups_as_new_instances_only():
    model = new_model(1, "test", 3, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    existing_instances = [
        new_model_instance(1, "test-existing-1", model.id),
    ]
    created_instances = []

    async def mock_create(instance):
        created_instances.append(instance)
        return instance

    placement_override = ModelPlacementOverride(
        replica_groups=[
            PlacementOverrideReplicaGroup(
                gpu_selector=GPUSelector(gpu_ids=["host4090:cuda:0"])
            ),
            PlacementOverrideReplicaGroup(
                gpu_selector=GPUSelector(gpu_ids=["host4080:cuda:1"])
            ),
        ]
    )

    with (
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            return_value=existing_instances,
        ),
        patch(
            'gpustack.server.services.ModelInstanceService.create',
            side_effect=mock_create,
        ),
    ):
        asyncio.run(
            sync_replicas(
                AsyncMock(),
                model,
                AsyncMock(),
                placement_override=placement_override,
            )
        )

    assert len(created_instances) == 2
    assert created_instances[0].placement_override.gpu_selector.gpu_ids == [
        "host4090:cuda:0"
    ]
    assert created_instances[1].placement_override.gpu_selector.gpu_ids == [
        "host4080:cuda:1"
    ]


def test_sync_replicas_accepts_new_replica_groups_name():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    created_instances = []

    async def mock_create(instance):
        created_instances.append(instance)
        return instance

    placement_override = ModelPlacementOverride(
        new_replica_groups=[
            PlacementOverrideReplicaGroup(
                gpu_selector=GPUSelector(gpu_ids=["host4090:cuda:0"])
            )
        ]
    )

    with (
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            return_value=[],
        ),
        patch(
            'gpustack.server.services.ModelInstanceService.create',
            side_effect=mock_create,
        ),
    ):
        asyncio.run(
            sync_replicas(
                AsyncMock(),
                model,
                AsyncMock(),
                placement_override=placement_override,
            )
        )

    assert len(created_instances) == 1
    assert created_instances[0].placement_override.gpu_selector.gpu_ids == [
        "host4090:cuda:0"
    ]


def test_sync_replicas_deletes_specified_scale_in_instances_as_whole_replicas():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    instances = [
        new_model_instance(1, "test-1", model.id, 1, ModelInstanceStateEnum.RUNNING, [0, 1]),
        new_model_instance(2, "test-2", model.id, 1, ModelInstanceStateEnum.RUNNING, [2, 3]),
        new_model_instance(3, "test-3", model.id, 1, ModelInstanceStateEnum.RUNNING, [4, 5]),
    ]
    deleted_instances = []

    async def mock_delete(instance):
        deleted_instances.append(instance)
        return instance

    with (
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            return_value=instances,
        ),
        patch(
            'gpustack.server.services.ModelInstanceService.delete',
            side_effect=mock_delete,
        ),
    ):
        asyncio.run(
            sync_replicas(
                AsyncMock(),
                model,
                AsyncMock(),
                scale_in_instance_ids=[2, 3],
            )
        )

    assert [instance.id for instance in deleted_instances] == [2, 3]
    assert deleted_instances[0].gpu_indexes == [2, 3]


def test_sync_replicas_serializes_scale_down_for_same_model():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    remaining_instances = {
        1: new_model_instance(1, "test-1", model.id, 1, ModelInstanceStateEnum.RUNNING, [0]),
        2: new_model_instance(2, "test-2", model.id, 1, ModelInstanceStateEnum.RUNNING, [1]),
    }
    deleted_instance_ids = []

    async def mock_all_by_field(*args, **kwargs):
        return list(remaining_instances.values())

    async def mock_delete(instance):
        await asyncio.sleep(0.01)
        remaining_instances.pop(instance.id, None)
        deleted_instance_ids.append(instance.id)
        return instance

    async def mock_find_scale_down_candidates(instances, model):
        return [ModelInstanceScore(model_instance=instances[0], score=0)]

    async def run_concurrent_syncs():
        await asyncio.gather(
            sync_replicas(
                AsyncMock(),
                model,
                AsyncMock(),
                scale_in_instance_ids=[2],
            ),
            sync_replicas(AsyncMock(), model, AsyncMock()),
        )

    with (
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            side_effect=mock_all_by_field,
        ),
        patch(
            'gpustack.server.services.ModelInstanceService.delete',
            side_effect=mock_delete,
        ),
        patch(
            'gpustack.server.controllers.find_scale_down_candidates',
            side_effect=mock_find_scale_down_candidates,
        ),
    ):
        asyncio.run(run_concurrent_syncs())

    assert deleted_instance_ids == [2]
    assert list(remaining_instances) == [1]


def test_sync_replicas_rechecks_current_instances_before_delete():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    stale_instances = [
        new_model_instance(1, "test-1", model.id, 1, ModelInstanceStateEnum.RUNNING, [0]),
        new_model_instance(2, "test-2", model.id, 1, ModelInstanceStateEnum.RUNNING, [1]),
    ]
    current_instances = [stale_instances[0]]
    all_by_field_results = [stale_instances, current_instances]
    deleted_instances = []

    async def mock_all_by_field(*args, **kwargs):
        return all_by_field_results.pop(0)

    async def mock_find_scale_down_candidates(instances, model):
        return [ModelInstanceScore(model_instance=instances[0], score=0)]

    async def mock_delete(instance):
        deleted_instances.append(instance)
        return instance

    with (
        patch(
            'gpustack.schemas.models.ModelInstance.all_by_field',
            side_effect=mock_all_by_field,
        ),
        patch(
            'gpustack.server.controllers.find_scale_down_candidates',
            side_effect=mock_find_scale_down_candidates,
        ),
        patch(
            'gpustack.server.services.ModelInstanceService.delete',
            side_effect=mock_delete,
        ),
    ):
        asyncio.run(sync_replicas(AsyncMock(), model, AsyncMock()))

    assert deleted_instances == []


def test_sync_replicas_rejects_invalid_specified_scale_in_instances():
    model = new_model(1, "test", 1, huggingface_repo_id="Qwen/Qwen2.5-7B-Instruct")
    instances = [
        new_model_instance(1, "test-1", model.id),
        new_model_instance(2, "test-2", model.id),
        new_model_instance(3, "test-3", model.id),
    ]

    with patch(
        'gpustack.schemas.models.ModelInstance.all_by_field',
        return_value=instances,
    ):
        with pytest.raises(ValueError, match="scale_in_instance_ids"):
            asyncio.run(
                sync_replicas(
                    AsyncMock(),
                    model,
                    AsyncMock(),
                    scale_in_instance_ids=[2],
                )
            )
