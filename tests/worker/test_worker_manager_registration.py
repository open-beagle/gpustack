from types import SimpleNamespace
import os
import stat
from unittest.mock import Mock

import pytest

from gpustack.schemas.workers import Worker
from gpustack.worker.worker_manager import WorkerManager


def test_register_worker_reloads_credential_created_after_initialization(tmp_path):
    credential_path = tmp_path / "model_preheat_worker_credential"
    workers = Mock()
    existing_worker = SimpleNamespace(
        id=1,
        worker_uuid="worker-uuid",
        labels={},
    )
    workers.list.side_effect = [
        SimpleNamespace(items=[existing_worker]),
        SimpleNamespace(items=[existing_worker]),
    ]
    workers.last_model_preheat_credential = "rotated-credential"
    clientset = SimpleNamespace(
        workers=workers,
        set_model_preheat_worker_credential=Mock(),
    )
    manager = WorkerManager.__new__(WorkerManager)
    manager._preheat_credential_path = str(credential_path)
    manager._clientset = clientset
    manager._worker_name = "worker-a"
    manager._worker_uuid = "worker-uuid"

    credential_path.write_text("bootstrap-credential", encoding="utf-8")
    manager._register_worker(
        Worker(
            id=1,
            name="worker-a",
            hostname="worker-a",
            ip="127.0.0.1",
            port=10150,
            worker_uuid="worker-uuid",
        )
    )

    assert clientset.set_model_preheat_worker_credential.call_args_list[0].args == (
        "bootstrap-credential",
    )
    workers.update.assert_called_once()


def test_register_worker_keeps_existing_credentials_when_file_is_absent(tmp_path):
    workers = Mock()
    existing_worker = SimpleNamespace(
        id=1,
        worker_uuid="worker-uuid",
        labels={},
    )
    workers.list.side_effect = [
        SimpleNamespace(items=[existing_worker]),
        SimpleNamespace(items=[existing_worker]),
    ]
    workers.last_model_preheat_credential = "rotated-credential"
    clientset = SimpleNamespace(
        workers=workers,
        set_model_preheat_worker_credential=Mock(),
    )
    manager = WorkerManager.__new__(WorkerManager)
    manager._preheat_credential_path = str(
        tmp_path / "missing-model-preheat-worker-credential"
    )
    manager._clientset = clientset
    manager._worker_name = "worker-a"
    manager._worker_uuid = "worker-uuid"

    manager._register_worker(
        Worker(
            id=1,
            name="worker-a",
            hostname="worker-a",
            ip="127.0.0.1",
            port=10150,
            worker_uuid="worker-uuid",
        )
    )

    clientset.set_model_preheat_worker_credential.assert_called_once_with(
        "rotated-credential"
    )
    workers.update.assert_called_once()


def test_register_worker_keeps_previous_memory_credential_when_atomic_write_fails(
    tmp_path, monkeypatch
):
    credential_path = tmp_path / "model_preheat_worker_credential"
    workers = Mock()
    existing_worker = SimpleNamespace(
        id=1,
        worker_uuid="worker-uuid",
        labels={},
    )
    workers.list.side_effect = [
        SimpleNamespace(items=[existing_worker]),
        SimpleNamespace(items=[existing_worker]),
    ]
    workers.last_model_preheat_credential = "new-credential"
    clientset = SimpleNamespace(
        workers=workers,
        set_model_preheat_worker_credential=Mock(),
    )
    manager = WorkerManager.__new__(WorkerManager)
    manager._preheat_credential_path = str(credential_path)
    manager._clientset = clientset
    manager._worker_name = "worker-a"
    manager._worker_uuid = "worker-uuid"
    credential_path.write_text("existing-credential", encoding="utf-8")

    monkeypatch.setattr(
        manager,
        "_store_preheat_credential",
        Mock(side_effect=OSError("simulated write failure")),
    )

    with pytest.raises(OSError, match="simulated write failure"):
        manager._register_worker(
            Worker(
                id=1,
                name="worker-a",
                hostname="worker-a",
                ip="127.0.0.1",
                port=10150,
                worker_uuid="worker-uuid",
            )
        )

    assert credential_path.read_text(encoding="utf-8") == "existing-credential"
    clientset.set_model_preheat_worker_credential.assert_called_once_with(
        "existing-credential"
    )


def test_store_preheat_credential_failure_preserves_previous_file(
    tmp_path, monkeypatch
):
    credential_path = tmp_path / "model_preheat_worker_credential"
    credential_path.write_text("previous-credential", encoding="utf-8")
    os.chmod(credential_path, 0o600)
    manager = WorkerManager.__new__(WorkerManager)
    manager._preheat_credential_path = str(credential_path)

    def replace_failure(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        "gpustack.worker.preheat_credential.os.replace", replace_failure
    )

    with pytest.raises(OSError, match="simulated replace failure"):
        manager._store_preheat_credential("new-credential")

    assert credential_path.read_text(encoding="utf-8") == "previous-credential"
    assert stat.S_IMODE(os.stat(credential_path).st_mode) == 0o600
    assert not list(tmp_path.glob(".model_preheat_worker_credential.*"))


def test_store_preheat_credential_replaces_file_atomically_with_private_mode(
    tmp_path, monkeypatch
):
    credential_path = tmp_path / "model_preheat_worker_credential"
    credential_path.write_text("previous-credential", encoding="utf-8")
    manager = WorkerManager.__new__(WorkerManager)
    manager._preheat_credential_path = str(credential_path)
    original_replace = os.replace
    replace = Mock(wraps=original_replace)
    monkeypatch.setattr("gpustack.worker.preheat_credential.os.replace", replace)

    manager._store_preheat_credential("new-credential")

    assert replace.call_count == 1
    assert credential_path.read_text(encoding="utf-8") == "new-credential"
    assert stat.S_IMODE(os.stat(credential_path).st_mode) == 0o600
    assert not list(tmp_path.glob(".model_preheat_worker_credential.*"))
