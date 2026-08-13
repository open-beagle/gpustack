import asyncio
import logging
import os
import threading
from pathlib import Path
from urllib.parse import urlparse

import urllib3
from minio import Minio

from gpustack.schemas.model_cache import (
    ModelCacheTaskPublic,
    ModelCacheTaskStateEnum,
    ModelCacheTaskUpdate,
)
from gpustack.server.bus import Event, EventType


logger = logging.getLogger(__name__)


class ModelCacheTaskDeleted(Exception):
    pass


class _CancelableReader:
    def __init__(self, path: Path, cancel_event: threading.Event, on_read):
        self._file = path.open("rb")
        self._cancel_event = cancel_event
        self._on_read = on_read

    def read(self, size=-1):
        if self._cancel_event.is_set():
            raise ModelCacheTaskDeleted()
        data = self._file.read(size)
        if data:
            self._on_read(len(data))
        return data

    def close(self):
        self._file.close()


class ModelCacheManager:
    def __init__(self, worker_id: int, clientset, cfg):
        self._worker_id = worker_id
        self._client = clientset.model_cache_tasks
        self._config = cfg
        self._active: dict[int, tuple[asyncio.Task, threading.Event]] = {}

    async def watch_model_cache_tasks(self):
        while True:
            try:
                await self._client.awatch(
                    callback=self._handle_event,
                    params={"worker_id": self._worker_id},
                )
            except asyncio.CancelledError:
                for task, cancel_event in self._active.values():
                    cancel_event.set()
                    task.cancel()
                raise
            except Exception as exc:
                logger.warning(
                    "Model cache task watch disconnected: %s", type(exc).__name__
                )
                await asyncio.sleep(5)

    def _handle_event(self, event: Event):
        if not event.data:
            return
        task = ModelCacheTaskPublic.model_validate(event.data)
        if task.worker_id != self._worker_id:
            return
        if event.type == EventType.DELETED:
            active = self._active.get(task.id)
            if active:
                active[1].set()
            return
        if (
            event.type not in {EventType.CREATED, EventType.UPDATED}
            or task.state != ModelCacheTaskStateEnum.PENDING
            or task.id in self._active
        ):
            return
        cancel_event = threading.Event()
        execution = asyncio.create_task(self._execute(task, cancel_event))
        self._active[task.id] = (execution, cancel_event)
        execution.add_done_callback(lambda _: self._active.pop(task.id, None))

    async def _execute(self, task: ModelCacheTaskPublic, cancel_event: threading.Event):
        try:
            await asyncio.to_thread(self._upload, task, cancel_event)
        except ModelCacheTaskDeleted:
            return
        except Exception as exc:
            logger.error("Model cache upload failed for task %s: %s", task.id, exc)
            try:
                await asyncio.to_thread(
                    self._client.update,
                    id=task.id,
                    update=ModelCacheTaskUpdate(
                        state=ModelCacheTaskStateEnum.ERROR,
                        progress=task.progress,
                        uploaded_size=task.uploaded_size,
                        total_size=task.total_size,
                        error_message=str(exc),
                    ),
                )
            except Exception:
                logger.debug("Model cache task %s no longer exists", task.id)

    def _upload(self, task: ModelCacheTaskPublic, cancel_event: threading.Event):
        client, bucket = self._s3_client()
        files = _source_files(task.source_paths)
        total_size = sum(path.stat().st_size for path, _ in files)
        uploaded_size = 0
        last_reported = -1

        def report(bytes_read=0, *, force=False):
            nonlocal uploaded_size, last_reported
            uploaded_size += bytes_read
            progress = 100 if total_size == 0 else uploaded_size * 100 / total_size
            bucket_percent = int(progress)
            if not force and bucket_percent == last_reported:
                return
            last_reported = bucket_percent
            self._client.update(
                id=task.id,
                update=ModelCacheTaskUpdate(
                    state=ModelCacheTaskStateEnum.UPLOADING,
                    progress=progress,
                    uploaded_size=uploaded_size,
                    total_size=total_size,
                ),
            )

        try:
            report(force=True)
            for path, relative_path in files:
                if cancel_event.is_set():
                    raise ModelCacheTaskDeleted()
                object_name = f"{task.target_path.rstrip('/')}/{relative_path}"
                reader = _CancelableReader(path, cancel_event, report)
                try:
                    client.put_object(
                        bucket,
                        object_name,
                        reader,
                        length=path.stat().st_size,
                    )
                finally:
                    reader.close()
            if cancel_event.is_set():
                raise ModelCacheTaskDeleted()
            report(force=True)
            self._client.update(
                id=task.id,
                update=ModelCacheTaskUpdate(
                    state=ModelCacheTaskStateEnum.READY,
                    progress=100,
                    uploaded_size=total_size,
                    total_size=total_size,
                ),
            )
        except ModelCacheTaskDeleted:
            _delete_prefix(client, bucket, f"{task.target_path.rstrip('/')}/")
            raise
        except Exception:
            _delete_prefix(client, bucket, f"{task.target_path.rstrip('/')}/")
            raise

    def _s3_client(self):
        endpoint = (self._config.worker_local_s3_host or "").rstrip("/")
        parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}")
        host = parsed.netloc or parsed.path
        secure = parsed.scheme == "https" or (
            not parsed.scheme and self._config.worker_local_s3_ssl
        )
        prefix = urlparse(self._config.worker_local_s3_modelscope_prefix)
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        client = Minio(
            host,
            access_key=self._config.worker_local_s3_access_key,
            secret_key=self._config.worker_local_s3_secret_key,
            secure=secure,
            region=self._config.worker_local_s3_region or None,
            cert_check=False,
        )
        if self._config.worker_local_s3_use_virtual_hosted_style:
            client.enable_virtual_style_endpoint()
        else:
            client.disable_virtual_style_endpoint()
        return client, prefix.netloc


def _source_files(source_paths: list[str]):
    roots = [Path(path).absolute() for path in source_paths]
    if not roots or any(not path.exists() for path in roots):
        raise FileNotFoundError("model_cache_source_not_found")
    result = []
    for root in roots:
        if root.is_symlink():
            raise ValueError("model_cache_source_symlink")
        if root.is_file():
            result.append((root, root.name))
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            current_path = Path(current)
            if any((current_path / name).is_symlink() for name in dirs + files):
                raise ValueError("model_cache_source_symlink")
            for name in files:
                path = current_path / name
                relative = path.relative_to(root).as_posix()
                result.append((path, relative))
    result = sorted(result, key=lambda item: item[1])
    relative_paths = [relative for _, relative in result]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("model_cache_duplicate_source_path")
    return result


def _delete_prefix(client, bucket: str, prefix: str):
    for item in client.list_objects(bucket, prefix=prefix, recursive=True):
        client.remove_object(bucket, item.object_name)
