"""Worker 侧模型同步执行器（任务 3 步骤 2）。

监听本 Worker 的模型同步任务（``worker_id`` 匹配、``state=pending``），
通过受 Worker 身份约束的 ``execution-payload`` 端点拉取一次性执行配置
（解密后的 S3 连接配置 + **冻结**的文件选择扫描规约），扫描本地模型并调用任务 1 的统一
发布器（:meth:`ModelPreheatS3Client.publish_artifact` +
:func:`build_model_preheat_manifest`）发布 Artifact：

- 已有文件摘要一致时跳过；
- 任务取消（DELETED 事件或 cancel_event）不写 Manifest；
- 完成后携带执行 lease token 回写 complete（CAS 绑定 ``artifact_id`` 并
  回写文件数/容量）；失败携带同一 lease token 回写稳定错误码。

凭据只进入执行 payload，不进 Public schema、SSE 或日志。
"""

import asyncio
import hashlib
import logging
import threading

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
        payload = None
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
                    request_digest=payload.request_digest,
                    lease_token=payload.lease_token,
                    file_count=result["file_count"],
                    total_size=result["total_size"],
                    manifest_digest=result["manifest_digest"],
                    manifest_path=result.get("manifest_path"),
                ),
            )
        except ModelStorageSyncCanceled:
            # 任务取消：不写 Manifest，不回写 ready。
            return
        except Exception as exc:
            logger.error(
                "Model storage sync failed for task %s: %s", task.id, type(exc).__name__
            )
            # 执行 payload 都拉取失败时没有可用 lease：失败回写无法通过
            # lease 校验（Server 稳定 409），只记录日志，不重复调用。
            if payload is None:
                return
            try:
                await self._client.afail(
                    task.id,
                    ModelStorageSyncTaskFail(
                        lease_token=payload.lease_token,
                        error_code=_stable_error_code(exc),
                    ),
                )
            except Exception:
                logger.debug("Model storage sync task %s no longer exists", task.id)


    def _publish(self, payload, cancel_event: threading.Event) -> dict:
        """扫描可信本地源路径并发布统一 Artifact（复用任务 1 发布器）。

        ``payload.profile`` 为解密后的明文 S3 连接配置（只在进程内使用）；
        扫描 root 与 include_patterns 一律取执行 payload 中**任务创建时
        冻结**的 ``scan_spec``（与 ``request_identity.include_patterns``
        一致）：不漏文件也不发布无关邻居文件，且不依赖 Worker 本地路径
        布局猜测（无 suffix 猜测、无 is_dir 探测）。已有文件摘要一致时
        跳过；取消时不写 Manifest。
        """
        identity_payload = payload.request_identity
        if not _scan_spec_root_matches_source_paths(payload):
            raise ValueError("model_sync_source_not_found")
        model_root, _frozen_patterns = _scan_spec_from_payload(payload)
        # patterns 以任务创建时固定的 request_identity 为准（与实际文件选择
        # 一致，规范化形态）；冻结 scan_spec 仅用于确定扫描 root。
        effective_include = tuple(identity_payload.get("include_patterns", []))
        manifest = build_model_preheat_manifest(
            model_root,
            ModelPreheatIdentity(
                source=payload.source,
                model_id=payload.model_id,
                revision=payload.resolved_revision,
                requested_revision=identity_payload.get("requested_revision"),
                file_patterns=effective_include,
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
        manifest_bytes = manifest.to_artifact_json_bytes()
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        return {
            "artifact_id": manifest.artifact_id,
            "file_count": len(manifest.files),
            "total_size": manifest.total_size,
            "uploaded": publish_result.uploaded,
            "skipped": publish_result.skipped,
            "manifest_digest": manifest_digest,
            "manifest_path": client.artifact_manifest_object(
                payload.profile.prefix, manifest
            ),
        }


def _raise_canceled(cancel_event: threading.Event):
    if cancel_event.is_set():
        raise ModelStorageSyncCanceled()


def _scan_spec_from_payload(payload) -> tuple[str, tuple[str, ...]]:
    """从执行 payload 读取**冻结**的扫描规约（root + include_patterns）。

    执行文件选择在任务创建时冻结（Server 与 Worker 用同一
    :func:`gpustack.server.model_storage_scan_spec.compute_scan_spec` 计算）；
    Worker 只消费 payload 中的冻结值，**不重读当前 ModelFile、不重算规约**
    （ModelFile 后续被修改/重新下载不改变本任务的文件选择）。规约缺失或不
    合法时稳定失败（不退回任何猜测路径）。
    """
    scan_spec = payload.scan_spec or {}
    root = (scan_spec.get("root") or "").strip()
    include_patterns = tuple(scan_spec.get("include_patterns") or [])
    if not root or not root.startswith("/"):
        # root 是任务创建时由 compute_scan_spec 冻结的 POSIX 路径（下载器
        # 约定为绝对路径布局）；缺失/非法即稳定失败。
        raise ValueError("model_sync_source_not_found")
    return root, include_patterns


def _scan_spec_root_matches_source_paths(payload) -> bool:
    """一致性校验：payload 的 source_paths 必须全部位于冻结 root 之下。

    防止 payload 损坏/串任务时扫描 root 遗漏所选路径（与创建侧
    compute_scan_spec 的一致性保证同语义）。
    """
    from pathlib import PurePosixPath

    try:
        root, _ = _scan_spec_from_payload(payload)
        root_path = PurePosixPath(root)
        source_paths = list(payload.source_paths or [])
        if not source_paths:
            return False
        for path in source_paths:
            PurePosixPath(str(path).rstrip("/")).relative_to(root_path)
        return True
    except (ValueError, TypeError):
        return False


def _stable_error_code(exc: Exception) -> str:
    if isinstance(exc, (ModelPreheatS3ManifestError, ModelPreheatIdentityError, ModelPreheatManifestError)):
        return "manifest_invalid"
    return "worker_execution_failed"
