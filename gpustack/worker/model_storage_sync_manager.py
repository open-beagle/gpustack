"""Worker 侧模型同步执行器（任务 3 步骤 2）。

监听本 Worker 的模型同步任务（``worker_id`` 匹配、``state=pending``），
通过受 Worker 身份约束的 ``execution-payload`` 端点拉取一次性执行配置
（解密后的 S3 连接配置 + 可信本地源路径），扫描本地模型并调用任务 1 的统一
发布器（:meth:`ModelPreheatS3Client.publish_artifact` +
:func:`build_model_preheat_manifest`）发布 Artifact：

- 已有文件摘要一致时跳过；
- 任务取消（DELETED 事件或 cancel_event）不写 Manifest；
- 完成后 CAS 绑定 ``artifact_id``（仅从 NULL）并回写文件数/容量；
- 失败回写稳定错误码。

凭据只进入执行 payload，不进 Public schema、SSE 或日志。
"""

import asyncio
import logging
import threading
from pathlib import Path

from gpustack.schemas.model_storage_sync import (
    ModelStorageSyncTaskComplete,
    ModelStorageSyncTaskFail,
    ModelStorageSyncTaskPublic,
    ModelStorageSyncTaskStateEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    ModelPreheatIdentityError,
)
from gpustack.worker.model_preheat.manifest import (
    ModelPreheatManifestError,
    build_model_preheat_manifest,
)
from gpustack.worker.model_preheat.s3_client import (
    ModelPreheatS3Client,
    ModelPreheatS3ManifestError,
)

logger = logging.getLogger(__name__)


class ModelStorageSyncCanceled(Exception):
    """任务被取消（DELETED 事件或 cancel_event）。"""


class ModelStorageSyncManager:
    def __init__(self, worker_id: int, clientset, cfg):
        self._worker_id = worker_id
        self._client = getattr(clientset, "model_storage_sync_tasks", None)
        self._config = cfg
        self._active: dict[int, tuple[asyncio.Task, threading.Event]] = {}

    async def watch_model_storage_sync_tasks(self):
        if self._client is None:
            logger.debug("Model storage sync client unavailable; skipping sync watcher")
            return
        while True:
            try:
                await self._client.awatch(
                    callback=self._handle_event,
                    params={"worker_id": self._worker_id},
                )
            except asyncio.CancelledError:
                for _task, cancel_event in self._active.values():
                    cancel_event.set()
                    _task.cancel()
                raise
            except Exception as exc:
                logger.warning(
                    "Model storage sync watch disconnected: %s", type(exc).__name__
                )
                await asyncio.sleep(5)

    def _handle_event(self, event: Event):
        if not event.data:
            return
        task = ModelStorageSyncTaskPublic.model_validate(event.data)
        if task.worker_id != self._worker_id:
            return
        if event.type == EventType.DELETED:
            active = self._active.get(task.id)
            if active:
                active[1].set()
            return
        if event.type not in {EventType.CREATED, EventType.UPDATED}:
            return
        # 活动任务被取消（UPDATED → canceled）时也要通知执行器停止，
        # 保证“任务取消不写 Manifest”在 UPDATED 事件路径同样成立。
        if task.state == ModelStorageSyncTaskStateEnum.CANCELED:
            active = self._active.get(task.id)
            if active:
                active[1].set()
            return
        if task.state != ModelStorageSyncTaskStateEnum.PENDING or task.id in self._active:
            return
        cancel_event = threading.Event()
        execution = asyncio.create_task(self._execute(task, cancel_event))
        self._active[task.id] = (execution, cancel_event)
        execution.add_done_callback(lambda _: self._active.pop(task.id, None))

    async def _execute(self, task: ModelStorageSyncTaskPublic, cancel_event: threading.Event):
        try:
            payload = await self._client.aget_execution_payload(task.id)
            if cancel_event.is_set():
                return
            result = await asyncio.to_thread(self._publish, payload, cancel_event)
            if cancel_event.is_set():
                return
            await self._client.acomplete(
                task.id,
                ModelStorageSyncTaskComplete(
                    artifact_id=result["artifact_id"],
                    file_count=result["file_count"],
                    total_size=result["total_size"],
                ),
            )
        except ModelStorageSyncCanceled:
            # 任务取消：不写 Manifest，不回写 ready。
            return
        except Exception as exc:
            logger.error(
                "Model storage sync failed for task %s: %s", task.id, type(exc).__name__
            )
            try:
                await self._client.afail(
                    task.id,
                    ModelStorageSyncTaskFail(error_code=_stable_error_code(exc)),
                )
            except Exception:
                logger.debug("Model storage sync task %s no longer exists", task.id)

    def _publish(self, payload, cancel_event: threading.Event) -> dict:
        """扫描可信本地源路径并发布统一 Artifact（复用任务 1 发布器）。

        ``payload.profile`` 为解密后的明文 S3 连接配置（只在进程内使用）；
        ``payload.source_paths`` 为 ModelFile 的可信本地路径。已有文件摘要一致
        时跳过；取消时不写 Manifest。
        """
        identity_payload = payload.request_identity
        model_root = _model_root_from_source_paths(payload.source_paths)
        manifest = build_model_preheat_manifest(
            model_root,
            ModelPreheatIdentity(
                source=payload.source,
                model_id=payload.model_id,
                revision=payload.resolved_revision,
                requested_revision=identity_payload.get("requested_revision"),
                file_patterns=tuple(identity_payload.get("include_patterns", [])),
                exclude_patterns=tuple(identity_payload.get("exclude_patterns", [])),
            ),
            cancel_callback=lambda: _raise_canceled(cancel_event),
        )

        client = ModelPreheatS3Client.from_minio(
            endpoint=payload.profile.endpoint,
            access_key=payload.profile.access_key,
            secret_key=payload.profile.secret_key,
            secure=payload.profile.tls_enabled,
            tls_verify=payload.profile.tls_verify,
            region=payload.profile.region or None,
            use_virtual_hosted_style=payload.profile.use_virtual_hosted_style,
        )
        publish_result = client.publish_artifact(
            payload.profile.bucket,
            payload.profile.prefix,
            manifest,
            model_root,
            cancel_check=cancel_event.is_set,
        )
        if cancel_event.is_set():
            raise ModelStorageSyncCanceled()
        return {
            "artifact_id": manifest.artifact_id,
            "file_count": len(manifest.files),
            "total_size": manifest.total_size,
            "uploaded": publish_result.uploaded,
            "skipped": publish_result.skipped,
        }


def _raise_canceled(cancel_event: threading.Event):
    if cancel_event.is_set():
        raise ModelStorageSyncCanceled()


def _model_root_from_source_paths(source_paths: list[str]) -> Path:
    """可信本地模型根目录：取首个源路径（父目录），供发布器扫描。"""
    if not source_paths:
        raise ValueError("model_sync_source_not_found")
    first = Path(source_paths[0])
    return first if first.is_dir() else first.parent


def _stable_error_code(exc: Exception) -> str:
    if isinstance(exc, (ModelPreheatS3ManifestError, ModelPreheatIdentityError, ModelPreheatManifestError)):
        return "manifest_invalid"
    return "worker_execution_failed"
