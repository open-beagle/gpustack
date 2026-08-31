import asyncio

import httpx
import pytest

from gpustack.api.exceptions import UnauthorizedException
from gpustack.client.generated_http_client import HTTPClient
from gpustack.client.generated_model_preheat_worker_task_client import (
    ModelPreheatWorkerTaskClient,
)
from gpustack.client.generated_model_storage_sync_task_client import (
    ModelStorageSyncTaskClient,
)


class _AsyncBytes(httpx.AsyncByteStream):
    def __init__(self, content: bytes):
        self._content = content

    async def __aiter__(self):
        yield self._content


async def _assert_watch_raises_unauthorized(client_cls):
    async def handler(request):
        return httpx.Response(
            401,
            headers={"content-type": "application/json"},
            stream=_AsyncBytes(
                b'{"code":401,"reason":"Unauthorized","message":"Unauthorized"}'
            ),
        )

    async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://testserver",
    )
    api_client = HTTPClient(base_url="http://testserver")
    api_client.set_async_httpx_client(async_client)
    try:
        with pytest.raises(UnauthorizedException, match="Unauthorized"):
            await client_cls(api_client).awatch()
    finally:
        await async_client.aclose()


def test_preheat_worker_task_watch_reads_stream_error_body():
    asyncio.run(_assert_watch_raises_unauthorized(ModelPreheatWorkerTaskClient))


def test_storage_sync_task_watch_reads_stream_error_body():
    asyncio.run(_assert_watch_raises_unauthorized(ModelStorageSyncTaskClient))
