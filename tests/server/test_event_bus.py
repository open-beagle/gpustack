import asyncio
from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.server.bus import Event, EventType, event_bus
from gpustack.server.controllers import set_default_worker_selector
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstancePublic,
    SourceEnum,
)
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
