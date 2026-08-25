import asyncio
import json
from typing import Any, Callable, Dict, Optional

import httpx
from gpustack.api.exceptions import raise_if_response_error
from gpustack.server.bus import Event
from gpustack.schemas.model_storage_sync import (
    ModelStorageSyncExecutionPayload,
    ModelStorageSyncTaskComplete,
    ModelStorageSyncTaskFail,
    ModelStorageSyncSourceMissing,
)

from .generated_http_client import HTTPClient


class ModelStorageSyncTaskClient:
    def __init__(self, client: HTTPClient):
        self._client = client
        self._url = f"{client._base_url}/v1/model-storage-worker-tasks"

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
                    continue

    async def aget_execution_payload(self, id: int) -> ModelStorageSyncExecutionPayload:
        response = await self._client.get_async_httpx_client().get(
            f"{self._url}/{id}/execution-payload"
        )
        raise_if_response_error(response)
        return ModelStorageSyncExecutionPayload.model_validate(response.json())

    async def acomplete(self, id: int, complete: ModelStorageSyncTaskComplete):
        response = await self._client.get_async_httpx_client().post(
            f"{self._url}/{id}/complete",
            content=complete.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)

    async def afail(self, id: int, failure: ModelStorageSyncTaskFail):
        response = await self._client.get_async_httpx_client().post(
            f"{self._url}/{id}/fail",
            content=failure.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)

    def mark_model_file_source_missing(self, id: int, expected_updated_at):
        response = self._client.get_httpx_client().post(
            f"{self._url}/model-files/{id}/source-missing",
            content=ModelStorageSyncSourceMissing(
                expected_updated_at=expected_updated_at
            ).model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
