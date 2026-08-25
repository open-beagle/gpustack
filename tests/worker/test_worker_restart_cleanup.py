import logging
from types import SimpleNamespace

from gpustack.api.exceptions import HTTPException, NotFoundException
from gpustack.worker.startup_cleanup import (
    cleanup_stale_model_instances,
    reconcile_ready_model_files,
)


class FakeModelInstancesClient:
    def __init__(self, pages):
        self.pages = pages
        self.list_params = []
        self.deleted_ids = []

    def list(self, params=None):
        self.list_params.append(params)
        items = self.pages.pop(0) if self.pages else []
        return SimpleNamespace(items=items)

    def delete(self, id):
        self.deleted_ids.append(id)


def test_worker_startup_deletes_stale_instances_for_current_worker():
    client = SimpleNamespace(
        model_instances=FakeModelInstancesClient(
            [
                [
                    SimpleNamespace(id=1, name="model-a-1"),
                    SimpleNamespace(id=2, name="model-a-2"),
                ],
                [],
            ]
        )
    )
    cleanup_stale_model_instances(client, 42, "worker-a")

    assert client.model_instances.list_params == [
        {"worker_id": 42, "page": 1, "perPage": 100},
        {"worker_id": 42, "page": 2, "perPage": 100},
    ]
    assert client.model_instances.deleted_ids == [1, 2]


def test_worker_startup_continues_after_stale_instance_was_already_deleted():
    class ConcurrentCleanupClient(FakeModelInstancesClient):
        def delete(self, id):
            self.deleted_ids.append(id)
            if id == 1:
                raise NotFoundException("模型实例不存在")

    client = SimpleNamespace(
        model_instances=ConcurrentCleanupClient(
            [
                [
                    SimpleNamespace(id=1, name="model-a-1"),
                    SimpleNamespace(id=2, name="model-a-2"),
                ],
                [],
            ]
        )
    )

    cleanup_stale_model_instances(client, 42, "worker-a")

    assert client.model_instances.deleted_ids == [1, 2]


def test_worker_startup_logs_unexpected_delete_error_and_continues(caplog):
    class FailingCleanupClient(FakeModelInstancesClient):
        def delete(self, id):
            self.deleted_ids.append(id)
            if id == 1:
                raise RuntimeError("连接中断")

    client = SimpleNamespace(
        model_instances=FailingCleanupClient(
            [
                [
                    SimpleNamespace(id=1, name="model-a-1"),
                    SimpleNamespace(id=2, name="model-a-2"),
                ],
                [],
            ]
        )
    )
    caplog.set_level(logging.ERROR, logger="gpustack.worker.startup_cleanup")

    cleanup_stale_model_instances(client, 42, "worker-a")

    assert client.model_instances.deleted_ids == [1, 2]
    assert "删除 worker worker-a 的遗留模型实例 model-a-1 失败" in caplog.text


def test_worker_startup_logs_http_error_and_continues(caplog):
    class FailingCleanupClient(FakeModelInstancesClient):
        def delete(self, id):
            self.deleted_ids.append(id)
            if id == 1:
                raise HTTPException(500, "InternalServerError", "服务端删除失败")

    client = SimpleNamespace(
        model_instances=FailingCleanupClient(
            [
                [
                    SimpleNamespace(id=1, name="model-a-1"),
                    SimpleNamespace(id=2, name="model-a-2"),
                ],
                [],
            ]
        )
    )
    caplog.set_level(logging.ERROR, logger="gpustack.worker.startup_cleanup")

    cleanup_stale_model_instances(client, 42, "worker-a")

    assert client.model_instances.deleted_ids == [1, 2]
    assert "删除 worker worker-a 的遗留模型实例 model-a-1 失败" in caplog.text


def test_worker_startup_marks_missing_ready_model_files_and_continues(tmp_path):
    existing_file = tmp_path / "model.bin"
    existing_file.write_text("ok")
    existing_dir = tmp_path / "model-dir"
    existing_dir.mkdir()

    class ModelFileRecord(SimpleNamespace):
        source = "model_scope"
        model_scope_model_id = "test/model"
        updated_at = "version"

    class ModelFilesClient:
        def __init__(self):
            self.list_params = []
            self.updated = []
            self.pages = [
                [
                    ModelFileRecord(id=1, resolved_paths=[str(existing_file)]),
                    ModelFileRecord(id=2, resolved_paths=[str(existing_dir)]),
                    ModelFileRecord(id=3, resolved_paths=[str(tmp_path / "*.missing")]),
                ],
                [ModelFileRecord(id=4, resolved_paths=[str(tmp_path / "gone")])],
                [],
            ]

        def list(self, params):
            self.list_params.append(params)
            return SimpleNamespace(items=self.pages.pop(0))

        def mark_model_file_source_missing(self, id, updated_at):
            self.updated.append((id, updated_at))
            if id == 3:
                raise RuntimeError("临时失败")

    client = ModelFilesClient()
    reconcile_ready_model_files(
        SimpleNamespace(model_files=client, model_storage_sync_tasks=client),
        42,
        "worker-a",
    )

    assert client.list_params == [
        {"worker_id": 42, "state": "ready", "page": 1, "perPage": 100},
        {"worker_id": 42, "state": "ready", "page": 2, "perPage": 100},
        {"worker_id": 42, "state": "ready", "page": 3, "perPage": 100},
    ]
    assert [item[0] for item in client.updated] == [3, 4]
