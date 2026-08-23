from types import SimpleNamespace

from gpustack.api.exceptions import NotFoundException
from gpustack.worker.model_file_manager import (
    _complete_download_execution_with_retries,
    _retry_download_completion_ack,
)
import pytest


def test_complete_download_execution_retries_idempotently():
    calls = []

    class ModelFiles:
        def complete_download_execution(self, model_file_id, completion):
            calls.append((model_file_id, completion))
            if len(calls) < 3:
                raise RuntimeError("temporary acknowledgement failure")

        def get(self, **kwargs):
            raise AssertionError("successful retry must not require coordination read")

    clientset = SimpleNamespace(model_files=ModelFiles())
    completion = SimpleNamespace(transfer_source="s3")

    assert _complete_download_execution_with_retries(clientset, 7, completion)

    assert len(calls) == 3


def test_persistent_complete_ack_failure_never_changes_local_ready_state():
    class ModelFiles:
        def complete_download_execution(self, model_file_id, completion):
            del model_file_id, completion
            raise RuntimeError("response lost after server commit")

    clientset = SimpleNamespace(model_files=ModelFiles())

    assert not _complete_download_execution_with_retries(
        clientset, 7, SimpleNamespace(transfer_source="s3")
    )


@pytest.mark.asyncio
async def test_background_completion_ack_retries_until_idempotent_success(monkeypatch):
    calls = 0

    class ModelFiles:
        def complete_download_execution(self, model_file_id, completion):
            nonlocal calls
            del model_file_id, completion
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary failure")

    async def no_wait(_delay):
        return None

    monkeypatch.setattr("gpustack.worker.model_file_manager.asyncio.sleep", no_wait)
    clientset = SimpleNamespace(model_files=ModelFiles())

    await _retry_download_completion_ack(
        clientset, 7, SimpleNamespace(transfer_source="s3")
    )

    assert calls == 2


@pytest.mark.asyncio
async def test_background_completion_ack_stops_on_not_found(monkeypatch):
    calls = 0

    class ModelFiles:
        def complete_download_execution(self, model_file_id, completion):
            nonlocal calls
            del model_file_id, completion
            calls += 1
            raise NotFoundException(message="deleted")

    async def no_wait(_delay):
        return None

    monkeypatch.setattr("gpustack.worker.model_file_manager.asyncio.sleep", no_wait)
    clientset = SimpleNamespace(model_files=ModelFiles())

    await _retry_download_completion_ack(
        clientset, 7, SimpleNamespace(transfer_source="s3")
    )

    assert calls == 1
