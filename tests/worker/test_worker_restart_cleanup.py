from types import SimpleNamespace

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
