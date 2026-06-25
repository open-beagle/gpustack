from datetime import datetime

import pytest

from gpustack.server.bus import Event, EventType, event_bus
from gpustack.schemas.models import ModelInstance, ModelInstancePublic, SourceEnum


@pytest.mark.asyncio
async def test_publish_model_instance_event_keeps_original_orm_object():
    topic = "test-model-instance-event"
    subscriber = event_bus.subscribe(topic)
    instance = ModelInstance(
        id=7,
        name="instance-7",
        model_id=3,
        model_name="model-a",
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/a",
    )

    try:
        await event_bus.publish(topic, Event(type=EventType.UPDATED, data=instance))
        event = await subscriber.receive()

        assert event.data is instance
    finally:
        event_bus.unsubscribe(topic, subscriber)


@pytest.mark.asyncio
async def test_table_model_event_shells_are_isolated_between_subscribers():
    topic = "test-table-model-event-shell-copy"
    first = event_bus.subscribe(topic)
    second = event_bus.subscribe(topic)
    instance = ModelInstance(
        id=7,
        name="instance-7",
        model_id=3,
        model_name="model-a",
        source=SourceEnum.LOCAL_PATH,
        local_path="/models/a",
    )

    try:
        await event_bus.publish(topic, Event(type=EventType.UPDATED, data=instance))
        first_event = await first.receive()
        second_event = await second.receive()

        assert first_event is not second_event
        assert first_event.data is instance
        assert second_event.data is instance

        first_event.data = {"id": 7}

        assert second_event.data is instance
    finally:
        event_bus.unsubscribe(topic, first)
        event_bus.unsubscribe(topic, second)


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
