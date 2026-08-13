from collections import defaultdict
from datetime import timezone
from urllib.parse import urlparse

import urllib3
from minio import Minio

from gpustack.schemas.model_cache import (
    ModelCacheDeleteResult,
    ModelCacheFilePublic,
    ModelCacheFilesPublic,
    ModelCacheModelPublic,
    ModelCacheModelsPublic,
)
from gpustack.utils.model_cache import (
    model_object_prefix,
    safe_path_part,
    validate_model_id,
)


class ModelCacheConfigurationError(ValueError):
    pass


class ModelCacheService:
    def __init__(self, config):
        self._config = config
        self._client, self._bucket, self._prefix = _client_from_config(config)

    def s3_path(self, model_id: str) -> str:
        return f"s3://{self._bucket}/{model_object_prefix(self._prefix, model_id)}"

    def list_models(self, search: str | None = None, organization: str | None = None):
        grouped = defaultdict(lambda: {"count": 0, "size": 0, "updated_at": None})
        prefix = f"{self._prefix}/" if self._prefix else ""
        for item in self._client.list_objects(
            self._bucket, prefix=prefix, recursive=True
        ):
            relative = item.object_name[len(prefix) :]
            parts = relative.split("/", 2)
            if len(parts) < 3:
                continue
            org = parts[0]
            name = parts[1]
            if not safe_path_part(org) or not safe_path_part(name) or not parts[2]:
                continue
            model_id = f"{org}/{name}"
            if organization and org != organization:
                continue
            if search and search.lower() not in model_id.lower():
                continue
            value = grouped[model_id]
            value["count"] += 1
            value["size"] += item.size or 0
            updated_at = item.last_modified
            if updated_at and (
                value["updated_at"] is None or updated_at > value["updated_at"]
            ):
                value["updated_at"] = updated_at

        items = []
        for model_id, value in sorted(grouped.items()):
            updated_at = value["updated_at"]
            if updated_at is None:
                continue
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            items.append(
                ModelCacheModelPublic(
                    model_id=model_id,
                    s3_path=self.s3_path(model_id),
                    file_count=value["count"],
                    total_size=value["size"],
                    updated_at=updated_at,
                )
            )
        return ModelCacheModelsPublic(items=items)

    def list_files(self, model_id: str):
        prefix = model_object_prefix(self._prefix, model_id)
        items = []
        for item in self._client.list_objects(
            self._bucket, prefix=prefix, recursive=True
        ):
            if not item.object_name.startswith(prefix):
                continue
            updated_at = item.last_modified
            if updated_at is None:
                continue
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            items.append(
                ModelCacheFilePublic(
                    path=item.object_name[len(prefix) :],
                    size=item.size or 0,
                    updated_at=updated_at,
                )
            )
        if not items:
            raise ValueError("model_cache_not_found")
        return ModelCacheFilesPublic(items=items)

    def exists(self, model_id: str) -> bool:
        prefix = model_object_prefix(self._prefix, model_id)
        return (
            next(
                iter(
                    self._client.list_objects(
                        self._bucket, prefix=prefix, recursive=True
                    )
                ),
                None,
            )
            is not None
        )

    def delete_model(self, model_id: str):
        prefix = model_object_prefix(self._prefix, model_id)
        objects = list(
            self._client.list_objects(self._bucket, prefix=prefix, recursive=True)
        )
        if not objects:
            raise ValueError("model_cache_not_found")
        deleted_size = sum(item.size or 0 for item in objects)
        for item in objects:
            self._client.remove_object(self._bucket, item.object_name)
        return ModelCacheDeleteResult(
            model_id=model_id,
            deleted_file_count=len(objects),
            deleted_size=deleted_size,
        )


def _client_from_config(config):
    endpoint = (config.worker_local_s3_host or "").rstrip("/")
    prefix_uri = config.worker_local_s3_modelscope_prefix or ""
    if not endpoint or not prefix_uri.startswith("s3://"):
        raise ModelCacheConfigurationError("local_s3_not_configured")
    parsed_endpoint = urlparse(endpoint if "://" in endpoint else f"//{endpoint}")
    host = parsed_endpoint.netloc or parsed_endpoint.path
    secure = parsed_endpoint.scheme == "https" or (
        not parsed_endpoint.scheme and config.worker_local_s3_ssl
    )
    parsed_prefix = urlparse(prefix_uri)
    bucket = parsed_prefix.netloc
    prefix = parsed_prefix.path.strip("/")
    if not host or not bucket:
        raise ModelCacheConfigurationError("local_s3_not_configured")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    client = Minio(
        host,
        access_key=config.worker_local_s3_access_key,
        secret_key=config.worker_local_s3_secret_key,
        secure=secure,
        region=config.worker_local_s3_region or None,
        cert_check=False,
    )
    if config.worker_local_s3_use_virtual_hosted_style:
        client.enable_virtual_style_endpoint()
    else:
        client.disable_virtual_style_endpoint()
    return client, bucket, prefix
