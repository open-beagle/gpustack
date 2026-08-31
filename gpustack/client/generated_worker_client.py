import asyncio
import json
from typing import Any, Callable, Dict, Optional

import httpx
from gpustack.api.exceptions import raise_if_response_error
from gpustack.server.bus import Event
from gpustack.schemas import *

from .generated_http_client import HTTPClient


class WorkerClient:
    def __init__(self, client: HTTPClient):
        self._client = client
        self._url = f"{client._base_url}/v1/workers"
        self.last_model_preheat_credential = None
        self._last_model_preheat_credential_generation = -1

    def list(self, params: Dict[str, Any] = None) -> WorkersPublic:
        response = self._client.get_httpx_client().get(self._url, params=params)
        raise_if_response_error(response)

        return WorkersPublic.model_validate(response.json())

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

    def get(self, id: int) -> WorkerPublic:
        response = self._client.get_httpx_client().get(f"{self._url}/{id}")
        raise_if_response_error(response)
        return WorkerPublic.model_validate(response.json())

    def create(self, model_create: WorkerCreate, *, upgrade_proof=None):
        headers = {"Content-Type": "application/json"}
        if upgrade_proof:
            headers["X-GPUStack-Worker-Upgrade-Proof"] = upgrade_proof
        response = self._client.get_httpx_client().post(
            self._url,
            content=model_create.model_dump_json(),
            headers=headers,
        )
        raise_if_response_error(response)
        self._remember_model_preheat_credential(
            response.headers.get("X-GPUStack-Worker-Credential")
        )
        return WorkerPublic.model_validate(response.json())

    def update(
        self,
        id: int,
        model_update: WorkerUpdate,
        *,
        registration=False,
        upgrade_proof=None,
    ):
        headers = {"Content-Type": "application/json"}
        if registration:
            headers["X-GPUStack-Worker-Registration"] = "true"
        if upgrade_proof:
            headers["X-GPUStack-Worker-Upgrade-Proof"] = upgrade_proof
        response = self._client.get_httpx_client().put(
            f"{self._url}/{id}",
            content=model_update.model_dump_json(),
            headers=headers,
        )
        raise_if_response_error(response)
        if registration:
            self._remember_model_preheat_credential(
                response.headers.get("X-GPUStack-Worker-Credential")
            )
        return WorkerPublic.model_validate(response.json())

    def _remember_model_preheat_credential(self, credential):
        if not credential:
            return
        generation = _credential_generation(credential)
        if generation >= self._last_model_preheat_credential_generation:
            self.last_model_preheat_credential = credential
            self._last_model_preheat_credential_generation = generation

    def delete(self, id: int):
        response = self._client.get_httpx_client().delete(f"{self._url}/{id}")
        raise_if_response_error(response)


def _credential_generation(credential):
    parts = credential.split("_", 3) if isinstance(credential, str) else []
    if len(parts) == 4 and parts[0] == "mpw":
        try:
            return int(parts[2])
        except ValueError:
            pass
    return 0
