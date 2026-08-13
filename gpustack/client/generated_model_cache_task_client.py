import asyncio
import json
from typing import Any, Callable, Dict, Optional

import httpx

from gpustack.api.exceptions import raise_if_response_error
from gpustack.schemas.model_cache import (
    ModelCacheTaskPublic,
    ModelCacheTasksPublic,
    ModelCacheTaskUpdate,
)
from gpustack.server.bus import Event


class ModelCacheTaskClient:
    def __init__(self, client):
        self._client = client
        self._url = f"{client._base_url}/v1/model-cache-tasks"

    def list(self, params: Dict[str, Any] = None) -> ModelCacheTasksPublic:
        response = self._client.get_httpx_client().get(self._url, params=params)
        raise_if_response_error(response)
        return ModelCacheTasksPublic.model_validate(response.json())

    async def awatch(
        self,
        callback: Optional[Callable[[Event], None]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        params = dict(params or {})
        params["watch"] = "true"
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
                    if line and callback:
                        callback(Event(**json.loads(line)))
                except asyncio.TimeoutError:
                    raise Exception("watch timeout")

    def update(self, id: int, update: ModelCacheTaskUpdate):
        response = self._client.get_httpx_client().put(
            f"{self._url}/{id}",
            content=update.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return ModelCacheTaskPublic.model_validate(response.json())
