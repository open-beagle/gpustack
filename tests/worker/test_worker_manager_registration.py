from types import SimpleNamespace
from unittest.mock import Mock

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
