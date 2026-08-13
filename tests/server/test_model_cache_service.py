from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gpustack.server.model_cache_service import (
    ModelCacheService,
    model_object_prefix,
    validate_model_id,
)


class FakeMinio:
    def __init__(self, objects):
        self.objects = list(objects)
        self.deleted = []

    def list_objects(self, bucket, prefix, recursive=True):
        del bucket, recursive
        return [item for item in self.objects if item.object_name.startswith(prefix)]

    def remove_object(self, bucket, object_name):
        del bucket
        self.deleted.append(object_name)


def object_info(name, size=1):
    return SimpleNamespace(
        object_name=name,
        size=size,
        last_modified=datetime(2026, 8, 14, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    "model_id", ["Qwen", "Qwen/", "/model", "a/b/c", "../model", "a/..", "a\\b/c"]
)
def test_validate_model_id_rejects_unsafe_paths(model_id):
    with pytest.raises(ValueError, match="invalid_model_id"):
        validate_model_id(model_id)


def test_model_object_prefix_adds_model_organization_prefix():
    assert (
        model_object_prefix("datamodel", "Qwen/Qwen3") == "datamodel/model_Qwen/Qwen3/"
    )


def test_list_models_groups_plain_s3_directories():
    service = ModelCacheService.__new__(ModelCacheService)
    service._bucket = "bd-wind"
    service._prefix = "datamodel"
    service._client = FakeMinio(
        [
            object_info("datamodel/model_Qwen/Qwen3/config.json", 10),
            object_info("datamodel/model_Qwen/Qwen3/model.bin", 20),
            object_info("datamodel/unrelated/file", 30),
        ]
    )

    result = service.list_models()

    assert len(result.items) == 1
    assert result.items[0].model_id == "Qwen/Qwen3"
    assert result.items[0].file_count == 2
    assert result.items[0].total_size == 30


def test_delete_model_is_limited_to_computed_model_prefix():
    service = ModelCacheService.__new__(ModelCacheService)
    service._bucket = "bd-wind"
    service._prefix = "datamodel"
    service._client = FakeMinio(
        [
            object_info("datamodel/model_Qwen/Qwen3/config.json", 10),
            object_info("datamodel/model_Qwen/Other/config.json", 20),
        ]
    )

    result = service.delete_model("Qwen/Qwen3")

    assert result.deleted_file_count == 1
    assert service._client.deleted == ["datamodel/model_Qwen/Qwen3/config.json"]
