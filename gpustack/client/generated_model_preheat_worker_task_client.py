import asyncio
import json
from typing import Any, Callable, Dict, Optional

import httpx
from gpustack.api.exceptions import raise_if_response_error
from gpustack.server.bus import Event
from gpustack.schemas import *

from .generated_http_client import HTTPClient


class ModelPreheatWorkerTaskClient:
    def __init__(self, client: HTTPClient):
        self._client = client
        self._url = f"{client._base_url}/v1/model-preheat-worker-tasks"

    def list(self, params: Dict[str, Any] = None) -> ModelPreheatWorkerTasksPublic:
        response = self._client.get_httpx_client().get(self._url, params=params)
        raise_if_response_error(response)

        return ModelPreheatWorkerTasksPublic.model_validate(response.json())

    def watch(
        self,
        callback: Optional[Callable[[Event], None]] = None,
        stop_condition: Optional[Callable[[Event], bool]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        if params is None:
            params = {}
        params["watch"] = "true"

        if stop_condition is None:
            stop_condition = lambda event: False

        with self._client.get_httpx_client().stream(
            "GET", self._url, params=params, timeout=None
        ) as response:
            raise_if_response_error(response)
            for line in response.iter_lines():
                if line:
                    event_data = json.loads(line)
                    event = Event(**event_data)
                    if callback:
                        callback(event)
                    if stop_condition(event):
                        break

    async def awatch(
        self,
        callback: Optional[Callable[[Event], None]] = None,
        stop_condition: Optional[Callable[[Event], bool]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        if params is None:
            params = {}
        params["watch"] = "true"

        if stop_condition is None:
            stop_condition = lambda event: False

        async with self._client.get_async_httpx_client().stream(
            "GET",
            self._url,
            params=params,
            timeout=httpx.Timeout(connect=10, read=None, write=10, pool=10),
        ) as response:
            raise_if_response_error(response)
            lines = response.aiter_lines()
            while True:
                try:
                    line = await asyncio.wait_for(lines.__anext__(), timeout=45)
                    if line:
                        event_data = json.loads(line)
                        event = Event(**event_data)
                        if callback:
                            callback(event)
                        if stop_condition(event):
                            break
                except asyncio.TimeoutError:
                    raise Exception("watch timeout")

    def get(self, id: int) -> ModelPreheatWorkerTaskPublic:
        response = self._client.get_httpx_client().get(f"{self._url}/{id}")
        raise_if_response_error(response)
        return ModelPreheatWorkerTaskPublic.model_validate(response.json())

    def claim(self, id: int, claim: ModelPreheatWorkerTaskClaim):
        response = self._client.get_httpx_client().post(
            f"{self._url}/{id}/claim",
            content=claim.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return ModelPreheatWorkerTaskClaimed.model_validate(response.json())

    async def aclaim(self, id: int, claim: ModelPreheatWorkerTaskClaim):
        response = await self._client.get_async_httpx_client().post(
            f"{self._url}/{id}/claim",
            content=claim.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return ModelPreheatWorkerTaskClaimed.model_validate(response.json())

    def heartbeat(self, id: int, lease: ModelPreheatWorkerTaskLease):
        response = self._client.get_httpx_client().post(
            f"{self._url}/{id}/heartbeat",
            content=lease.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return ModelPreheatWorkerTaskClaimed.model_validate(response.json())

    async def aheartbeat(self, id: int, lease: ModelPreheatWorkerTaskLease):
        response = await self._client.get_async_httpx_client().post(
            f"{self._url}/{id}/heartbeat",
            content=lease.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return ModelPreheatWorkerTaskClaimed.model_validate(response.json())

    def progress(self, id: int, progress: ModelPreheatWorkerTaskProgress):
        response = self._client.get_httpx_client().patch(
            f"{self._url}/{id}/progress",
            content=progress.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return ModelPreheatWorkerTaskPublic.model_validate(response.json())

    async def aprogress(self, id: int, progress: ModelPreheatWorkerTaskProgress):
        response = await self._client.get_async_httpx_client().patch(
            f"{self._url}/{id}/progress",
            content=progress.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return ModelPreheatWorkerTaskPublic.model_validate(response.json())

    def complete(self, id: int, complete: ModelPreheatWorkerTaskComplete):
        response = self._client.get_httpx_client().post(
            f"{self._url}/{id}/complete",
            content=complete.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return ModelPreheatWorkerTaskPublic.model_validate(response.json())

    async def acomplete(self, id: int, complete: ModelPreheatWorkerTaskComplete):
        response = await self._client.get_async_httpx_client().post(
            f"{self._url}/{id}/complete",
            content=complete.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return ModelPreheatWorkerTaskPublic.model_validate(response.json())

    def fail(self, id: int, failure: ModelPreheatWorkerTaskFail):
        response = self._client.get_httpx_client().post(
            f"{self._url}/{id}/fail",
            content=failure.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return ModelPreheatWorkerTaskPublic.model_validate(response.json())

    async def afail(self, id: int, failure: ModelPreheatWorkerTaskFail):
        response = await self._client.get_async_httpx_client().post(
            f"{self._url}/{id}/fail",
            content=failure.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return ModelPreheatWorkerTaskPublic.model_validate(response.json())

    def get_execution_payload(
        self,
        id: int,
        worker_uuid: str,
        worker_id: int,
        attempt: int,
        token: str,
    ):
        response = self._client.get_httpx_client().get(
            f"{self._url}/{id}/execution-payload",
            headers={
                "X-Worker-UUID": worker_uuid,
                "X-Worker-ID": str(worker_id),
                "X-Task-Attempt": str(attempt),
                "X-Lease-Token": token,
            },
        )
        raise_if_response_error(response)
        return ModelPreheatWorkerTaskExecutionPayload.model_validate(response.json())

    async def aget_execution_payload(
        self,
        id: int,
        worker_uuid: str,
        worker_id: int,
        attempt: int,
        token: str,
    ):
        response = await self._client.get_async_httpx_client().get(
            f"{self._url}/{id}/execution-payload",
            headers={
                "X-Worker-UUID": worker_uuid,
                "X-Worker-ID": str(worker_id),
                "X-Task-Attempt": str(attempt),
                "X-Lease-Token": token,
            },
        )
        raise_if_response_error(response)
        return ModelPreheatWorkerTaskExecutionPayload.model_validate(response.json())
