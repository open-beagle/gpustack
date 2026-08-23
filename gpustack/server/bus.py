import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List
from enum import Enum
import copy
import importlib
import logging

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import async_object_session


logger = logging.getLogger(__name__)


class EventType(Enum):
    CREATED = 1
    UPDATED = 2
    DELETED = 3
    UNKNOWN = 4
    HEARTBEAT = 5


@dataclass
class Event:
    type: EventType
    data: Any

    def __post_init__(self):
        if isinstance(self.type, int):
            self.type = EventType(self.type)


def event_decoder(obj):
    if "type" in obj:
        obj["type"] = EventType[obj["type"]]
    return obj


class Subscriber:
    def __init__(self, public_snapshot: bool = False):
        self.queue = asyncio.Queue(maxsize=256)
        self.public_snapshot = public_snapshot
        self._closed = asyncio.Event()

    async def enqueue(self, event: Event):
        if self._closed.is_set():
            return False
        try:
            self.queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            pass

        put_task = asyncio.create_task(self.queue.put(event))
        close_task = asyncio.create_task(self._closed.wait())
        try:
            done, _ = await asyncio.wait(
                {put_task, close_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if close_task in done or self._closed.is_set():
                return False
            await put_task
            return True
        finally:
            for task in (put_task, close_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(put_task, close_task, return_exceptions=True)

    async def receive(self) -> Any:
        return await self.queue.get()

    def close(self):
        self._closed.set()


@dataclass
class _TopicPublishState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    publishers: int = 0


class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Subscriber]] = {}
        self._topic_publish_states: Dict[str, _TopicPublishState] = {}

    def subscribe(self, topic: str, public_snapshot: bool = False) -> Subscriber:
        subscriber = Subscriber(public_snapshot=public_snapshot)
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(subscriber)
        return subscriber

    def unsubscribe(self, topic: str, subscriber: Subscriber):
        subscriber.close()
        if topic in self.subscribers:
            self.subscribers[topic].remove(subscriber)
            if not self.subscribers[topic]:
                del self.subscribers[topic]
                state = self._topic_publish_states.get(topic)
                if state is not None and state.publishers == 0:
                    del self._topic_publish_states[topic]

    async def publish(self, topic: str, event: Event):
        if topic not in self.subscribers:
            return
        state = self._topic_publish_states.setdefault(topic, _TopicPublishState())
        state.publishers += 1
        try:
            async with state.lock:
                public_snapshot = None
                public_snapshot_failed = False
                for subscriber in list(self.subscribers.get(topic, [])):
                    if not subscriber.public_snapshot:
                        data = (
                            event.data
                            if _is_table_model_event_data(event.data)
                            else copy.deepcopy(event.data)
                        )
                        await subscriber.enqueue(Event(type=event.type, data=data))
                        continue
                    if public_snapshot_failed:
                        continue
                    if public_snapshot is None:
                        try:
                            public_snapshot = await _snapshot_event(event)
                        except Exception as exc:
                            public_snapshot_failed = True
                            logger.error(
                                "Failed to snapshot event for %s: %s",
                                topic,
                                type(exc).__name__,
                            )
                            continue
                    await subscriber.enqueue(copy.deepcopy(public_snapshot))
        finally:
            state.publishers -= 1
            if (
                state.publishers == 0
                and topic not in self.subscribers
                and self._topic_publish_states.get(topic) is state
            ):
                del self._topic_publish_states[topic]


event_bus = EventBus()


async def _snapshot_event(event: Event) -> Event:
    data = event.data
    if data is None:
        return Event(type=event.type, data=None)
    if _is_table_model_event_data(data):
        state = inspect(data, raiseerr=False)
        if state is not None and (state.expired_attributes or state.unloaded):
            session = async_object_session(data)
            if session is not None:
                await session.refresh(data)
        module = importlib.import_module(type(data).__module__)
        public_class = getattr(module, f"{type(data).__name__}Public", None)
        if public_class is None:
            raise ValueError("event_public_schema_missing")
        source = data.model_dump()
        source.update(
            {
                key: value
                for key, value in getattr(data, "__dict__", {}).items()
                if not key.startswith("_")
            }
        )
        data = public_class.model_validate(source)
    elif hasattr(data, "model_copy"):
        data = data.model_copy(deep=True)
    else:
        data = copy.deepcopy(data)
    return Event(type=event.type, data=data)


def _is_table_model_event_data(data: Any) -> bool:
    return hasattr(data, "_sa_instance_state") or hasattr(type(data), "__table__")
