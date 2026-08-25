import logging
from types import SimpleNamespace

from gpustack.api.exceptions import HTTPException, NotFoundException
from gpustack.worker.startup_cleanup import cleanup_stale_model_instances


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
