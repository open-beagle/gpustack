import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List
from enum import Enum
import copy
import importlib
import logging


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

    async def enqueue(self, event: Event):
        await self.queue.put(event)

    async def receive(self) -> Any:
        return await self.queue.get()


class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Subscriber]] = {}

    def subscribe(self, topic: str, public_snapshot: bool = False) -> Subscriber:
        subscriber = Subscriber(public_snapshot=public_snapshot)
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(subscriber)
        return subscriber

    def unsubscribe(self, topic: str, subscriber: Subscriber):
        if topic in self.subscribers:
            self.subscribers[topic].remove(subscriber)
            if not self.subscribers[topic]:
                del self.subscribers[topic]

    async def publish(self, topic: str, event: Event):
        if topic not in self.subscribers:
            return
        for subscriber in self.subscribers[topic]:
            if not subscriber.public_snapshot:
                data = (
                    event.data
                    if _is_table_model_event_data(event.data)
                    else copy.deepcopy(event.data)
                )
                await subscriber.enqueue(Event(type=event.type, data=data))
                continue
            try:
                snapshot = _snapshot_event(event)
            except Exception as exc:
                logger.error(
                    "Failed to snapshot event for %s: %s",
                    topic,
                    type(exc).__name__,
                )
                continue
            await subscriber.enqueue(copy.deepcopy(snapshot))


event_bus = EventBus()


def _snapshot_event(event: Event) -> Event:
    data = event.data
    if data is None:
        return Event(type=event.type, data=None)
    if _is_table_model_event_data(data):
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
