import asyncio
from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.links import ModelInstanceModelFileLink
from gpustack.schemas.model_files import ModelFile
from gpustack.server import bus as bus_module
from gpustack.server.bus import Event, EventBus, EventType, event_bus
from gpustack.server.controllers import set_default_worker_selector
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstancePublic,
    SourceEnum,
)
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker, WorkerPublic, WorkerStateEnum


@pytest.mark.asyncio
async def test_publish_model_instance_event_uses_fixed_public_snapshot():
    topic = "test-model-instance-event"
    subscriber = event_bus.subscribe(topic, public_snapshot=True)
    instance = ModelInstance(
        id=7,
        name="instance-7",
        model_id=3,
        model_name="model-a",
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/a",
    )
    instance.__dict__["created_at"] = datetime.now()
    instance.__dict__["updated_at"] = datetime.now()

    try:
        await event_bus.publish(topic, Event(type=EventType.UPDATED, data=instance))
        instance.name = "mutated-after-publish"
        event = await subscriber.receive()

        assert isinstance(event.data, ModelInstancePublic)
        assert event.data is not instance
        assert event.data.name == "instance-7"
    finally:
        event_bus.unsubscribe(topic, subscriber)


@pytest.mark.asyncio
async def test_deleted_model_instance_event_uses_public_snapshot_with_timestamps(
    tmp_path,
):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'model-instance-event.db'}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(
            SQLModel.metadata.create_all,
            tables=[
                ModelInstance.__table__,
                ModelFile.__table__,
                ModelInstanceModelFileLink.__table__,
            ],
        )

    subscriber = event_bus.subscribe("modelinstance", public_snapshot=True)
    try:
        async with AsyncSession(engine, expire_on_commit=True) as session:
            instance = ModelInstance(
                name="instance-7",
                model_id=3,
                model_name="model-a",
                source=SourceEnum.LOCAL_PATH,
                local_path="/models/a",
            )
            session.add(instance)
            await session.commit()

            await instance.delete(session)

        event = await asyncio.wait_for(subscriber.receive(), timeout=1)
        assert event.type == EventType.DELETED
        assert isinstance(event.data, ModelInstancePublic)
        assert event.data.id is not None
        assert event.data.created_at is not None
        assert event.data.updated_at is not None
    finally:
        event_bus.unsubscribe("modelinstance", subscriber)
        await engine.dispose()


@pytest.mark.asyncio
async def test_deleted_model_instance_snapshot_isolates_mutable_columns(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'model-instance-snapshot.db'}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(
            SQLModel.metadata.create_all,
            tables=[
                ModelInstance.__table__,
                ModelFile.__table__,
                ModelInstanceModelFileLink.__table__,
            ],
        )

    subscriber = event_bus.subscribe("modelinstance")
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            instance = ModelInstance(
                name="instance-8",
                model_id=3,
                model_name="model-a",
                source=SourceEnum.LOCAL_PATH,
                local_path="/models/a",
                gpu_indexes=[0],
                ports=[8000],
            )
            session.add(instance)
            await session.commit()

            await instance.delete(session)
            instance.gpu_indexes.append(1)

        event = await asyncio.wait_for(subscriber.receive(), timeout=1)
        assert event.type == EventType.DELETED
        assert event.data.gpu_indexes == [0]
        assert event.data.gpu_indexes is not instance.gpu_indexes
    finally:
        event_bus.unsubscribe("modelinstance", subscriber)
        await engine.dispose()


@pytest.mark.asyncio
async def test_delete_detached_user_reloads_current_session_instance(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'user-delete.db'}"
    )
    async with engine.begin() as connection:
        await connection.run_sync(
            SQLModel.metadata.create_all, tables=[User.__table__]
        )

    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            user = User(username="detached-user", hashed_password="hashed-password")
            session.add(user)
            await session.commit()

            detached_user = await User.one_by_id(session, user.id)
            session.expunge(detached_user)
            await detached_user.delete(session)

            assert await User.one_by_id(session, user.id) is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_delete_skips_model_instance_removed_by_another_session(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'model-instance-race.db'}"
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

    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            session.add(
                Model(
                    id=3,
                    name="model-a",
                    source=SourceEnum.LOCAL_PATH,
                    local_path="/models/a",
                )
            )
            instance = ModelInstance(
                name="instance-race",
                model_id=3,
                model_name="model-a",
                source=SourceEnum.LOCAL_PATH,
                local_path="/models/a",
            )
            session.add(instance)
            await session.commit()
            instance_id = instance.id

        async with AsyncSession(engine) as first_session:
            stale_instance = await ModelInstance.one_by_id(first_session, instance_id)
            async with AsyncSession(engine) as second_session:
                current_instance = await ModelInstance.one_by_id(
                    second_session, instance_id
                )
                await current_instance.delete(second_session)

            await stale_instance.delete(first_session)
            assert await ModelInstance.one_by_id(first_session, instance_id) is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_table_model_event_snapshots_are_isolated_between_subscribers():
    topic = "test-table-model-event-shell-copy"
    first = event_bus.subscribe(topic, public_snapshot=True)
    second = event_bus.subscribe(topic, public_snapshot=True)
    instance = ModelInstance(
        id=7,
        name="instance-7",
        model_id=3,
        model_name="model-a",
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/a",
    )
    instance.__dict__["created_at"] = datetime.now()
    instance.__dict__["updated_at"] = datetime.now()

    try:
        await event_bus.publish(topic, Event(type=EventType.UPDATED, data=instance))
        first_event = await first.receive()
        second_event = await second.receive()

        assert first_event is not second_event
        assert first_event.data is not instance
        assert second_event.data is not instance
        assert first_event.data is not second_event.data

        first_event.data.name = "subscriber-mutated"

        assert second_event.data.name == "instance-7"
    finally:
        event_bus.unsubscribe(topic, first)
        event_bus.unsubscribe(topic, second)


@pytest.mark.asyncio
async def test_invalid_table_model_event_is_not_published_as_empty_object():
    topic = "test-invalid-table-model-event"
    subscriber = event_bus.subscribe(topic, public_snapshot=True)
    incomplete = ModelInstance(id=7)

    try:
        await event_bus.publish(topic, Event(type=EventType.UPDATED, data=incomplete))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(subscriber.receive(), timeout=0.01)
    finally:
        event_bus.unsubscribe(topic, subscriber)


@pytest.mark.asyncio
async def test_worker_initial_event_survives_subscription_session_end(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'worker-event.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(
            SQLModel.metadata.create_all, tables=[Worker.__table__]
        )
    async with AsyncSession(engine) as session:
        session.add(
            Worker(
                name="worker-a",
                hostname="worker-a",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="worker-a-uuid",
                state=WorkerStateEnum.READY,
            )
        )
        await session.commit()

    subscription = Worker.subscribe(engine, public_snapshot=True)
    event = await anext(subscription)
    await subscription.aclose()

    assert isinstance(event.data, WorkerPublic)
    assert event.data.name == "worker-a"
    assert event.data.state == WorkerStateEnum.READY
    await engine.dispose()


@pytest.mark.asyncio
async def test_public_snapshot_refreshes_attached_expired_worker(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'expired-event.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(
            SQLModel.metadata.create_all, tables=[Worker.__table__]
        )

    topic = "test-expired-worker-public-event"
    subscriber = event_bus.subscribe(topic, public_snapshot=True)
    try:
        async with AsyncSession(engine, expire_on_commit=True) as session:
            worker = Worker(
                name="worker-expired",
                hostname="worker-expired",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="worker-expired-uuid",
                state=WorkerStateEnum.READY,
            )
            session.add(worker)
            await session.commit()
            assert worker.model_dump() == {}

            await event_bus.publish(topic, Event(type=EventType.UPDATED, data=worker))
            event = await asyncio.wait_for(subscriber.receive(), timeout=0.1)

        assert isinstance(event.data, WorkerPublic)
        assert event.data.name == "worker-expired"
        assert event.data.worker_uuid == "worker-expired-uuid"
    finally:
        event_bus.unsubscribe(topic, subscriber)
        await engine.dispose()


@pytest.mark.asyncio
async def test_public_snapshot_refreshes_partially_expired_required_field(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'partial.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(
            SQLModel.metadata.create_all, tables=[Worker.__table__]
        )

    topic = "test-partially-expired-worker-public-event"
    subscriber = event_bus.subscribe(topic, public_snapshot=True)
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            worker = Worker(
                name="worker-partial",
                hostname="worker-partial",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="worker-partial-uuid",
                state=WorkerStateEnum.READY,
            )
            session.add(worker)
            await session.commit()
            session.expire(worker, ["name"])
            assert worker.model_dump()
            assert "name" not in worker.model_dump()

            await event_bus.publish(topic, Event(type=EventType.UPDATED, data=worker))
            event = await asyncio.wait_for(subscriber.receive(), timeout=0.1)

        assert isinstance(event.data, WorkerPublic)
        assert event.data.name == "worker-partial"
    finally:
        event_bus.unsubscribe(topic, subscriber)
        await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_publish_preserves_topic_order_for_all_subscribers(
    monkeypatch,
):
    bus = EventBus()
    topic = "test-concurrent-topic-order"
    internal = bus.subscribe(topic)
    public = bus.subscribe(topic, public_snapshot=True)
    snapshot_a_started = asyncio.Event()
    release_snapshot_a = asyncio.Event()
    original_snapshot = bus_module._snapshot_event

    async def delayed_snapshot(event):
        if event.data["sequence"] == "A":
            snapshot_a_started.set()
            await release_snapshot_a.wait()
        return await original_snapshot(event)

    monkeypatch.setattr(bus_module, "_snapshot_event", delayed_snapshot)
    publish_a = asyncio.create_task(
        bus.publish(topic, Event(EventType.UPDATED, {"sequence": "A"}))
    )
    await asyncio.wait_for(snapshot_a_started.wait(), timeout=1)
    publish_b = asyncio.create_task(
        bus.publish(topic, Event(EventType.UPDATED, {"sequence": "B"}))
    )
    await asyncio.sleep(0)
    release_snapshot_a.set()
    await asyncio.gather(publish_a, publish_b)

    internal_order = [(await internal.receive()).data["sequence"] for _ in range(2)]
    public_order = [(await public.receive()).data["sequence"] for _ in range(2)]

    assert internal_order == ["A", "B"]
    assert public_order == ["A", "B"]

    bus.unsubscribe(topic, internal)
    bus.unsubscribe(topic, public)
    assert topic not in bus._topic_publish_states


@pytest.mark.asyncio
async def test_unsubscribe_wakes_publish_blocked_by_full_queue():
    bus = EventBus()
    topic = "test-full-queue-unsubscribe"
    blocked = bus.subscribe(topic)
    blocked.queue = asyncio.Queue(maxsize=1)
    await blocked.queue.put(Event(EventType.CREATED, {"sequence": "preloaded"}))
    survivor = bus.subscribe(topic)

    publishing = asyncio.create_task(
        bus.publish(topic, Event(EventType.UPDATED, {"sequence": "A"}))
    )
    await asyncio.sleep(0)
    assert not publishing.done()

    bus.unsubscribe(topic, blocked)
    await asyncio.wait_for(publishing, timeout=1)
    assert (await survivor.receive()).data["sequence"] == "A"

    bus.unsubscribe(topic, survivor)
    replacement = bus.subscribe(topic)
    await asyncio.wait_for(
        bus.publish(topic, Event(EventType.UPDATED, {"sequence": "B"})),
        timeout=1,
    )
    assert (await replacement.receive()).data["sequence"] == "B"
    bus.unsubscribe(topic, replacement)


@pytest.mark.asyncio
async def test_worker_heartbeat_event_keeps_none_payload():
    topic = Worker.__name__.lower()
    subscriber = event_bus.subscribe(topic)
    try:
        await event_bus.publish(topic, Event(type=EventType.HEARTBEAT, data=None))
        event = await subscriber.receive()
        assert event.type == EventType.HEARTBEAT
        assert event.data is None
    finally:
        event_bus.unsubscribe(topic, subscriber)


@pytest.mark.asyncio
async def test_internal_model_event_preserves_orm_fields_for_controller():
    topic = "test-internal-model-event"
    subscriber = event_bus.subscribe(topic)
    deleted_at = datetime.now()
    model = Model(
        id=3,
        name="model-a",
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/a",
        deleted_at=deleted_at,
    )

    try:
        await event_bus.publish(topic, Event(type=EventType.DELETED, data=model))
        event = await subscriber.receive()

        assert isinstance(event.data, Model)
        assert event.data.deleted_at == deleted_at
        assert hasattr(event.data, "_sa_instance_state")
        await set_default_worker_selector(None, event.data)
    finally:
        event_bus.unsubscribe(topic, subscriber)


def test_streaming_helpers_handle_model_instance_without_orm_instrumentation():
    now = datetime.now()
    instance = ModelInstance.__new__(ModelInstance)
    instance.__dict__.update(
        {
            "id": 7,
            "name": "instance-7",
            "model_id": 3,
            "model_name": "model-a",
            "source": SourceEnum.LOCAL_PATH,
            "local_path": "/models/a",
            "created_at": now,
            "updated_at": now,
        }
    )
    event = Event(type=EventType.UPDATED, data=instance)

    assert ModelInstance._match_fields(event, {"model_id": 3}) is True
    assert ModelInstance._match_fuzzy_fields(event, {"name": "instance"}) is True

    public = ModelInstance._convert_to_public_class(instance)

    assert isinstance(public, ModelInstancePublic)
    assert public.id == 7
    assert public.created_at == now
    assert public.updated_at == now


def test_streaming_filter_errors_skip_bad_event():
    def broken_filter(_data):
        raise RuntimeError("bad event")

    assert ModelInstance._safe_filter(broken_filter, object()) is False


@pytest.mark.asyncio
async def test_publish_plain_event_still_isolates_subscribers():
    topic = "test-plain-event-copy"
    first = event_bus.subscribe(topic)
    second = event_bus.subscribe(topic)
    data = {"items": ["a"]}

    try:
        await event_bus.publish(topic, Event(type=EventType.UPDATED, data=data))
        first_event = await first.receive()
        second_event = await second.receive()

        assert first_event is not second_event
        assert first_event.data is not second_event.data

        first_event.data["items"].append("b")

        assert second_event.data == {"items": ["a"]}
    finally:
        event_bus.unsubscribe(topic, first)
        event_bus.unsubscribe(topic, second)


@pytest.mark.asyncio
async def test_publish_public_schema_event_still_isolates_subscribers():
    topic = "test-public-schema-event-copy"
    first = event_bus.subscribe(topic)
    second = event_bus.subscribe(topic)
    now = datetime.now()
    data = ModelInstancePublic(
        id=7,
        name="instance-7",
        model_id=3,
        model_name="model-a",
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/a",
        created_at=now,
        updated_at=now,
        gpu_indexes=[0],
    )

    try:
        await event_bus.publish(topic, Event(type=EventType.UPDATED, data=data))
        first_event = await first.receive()
        second_event = await second.receive()

        assert first_event.data is not second_event.data

        first_event.data.gpu_indexes.append(1)

        assert second_event.data.gpu_indexes == [0]
    finally:
        event_bus.unsubscribe(topic, first)
        event_bus.unsubscribe(topic, second)
