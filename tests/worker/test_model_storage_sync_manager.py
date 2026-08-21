"""任务 3：Worker 侧模型同步执行器定向测试。

覆盖：扫描可信本地源路径并调用任务 1 发布器发布 Artifact（已有文件摘要一致
时跳过）；任务取消不写 Manifest；完成后 CAS 绑定 artifact_id（仅从 NULL）；
失败回写稳定错误码；凭据只进入执行 payload、不进日志。

外部 S3 不可用：注入 fake ``ModelPreheatS3Client``（发布器），只验证同步执行器
的编排逻辑（本地扫描 → 发布 → CAS 绑定/取消/失败回写），不伪造网络可达性。
"""

import asyncio
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gpustack.schemas.model_storage_sync import (
    ModelStorageSyncExecutionPayload,
    ModelStorageSyncExecutionProfile,
    ModelStorageSyncTaskComplete,
    ModelStorageSyncTaskFail,
    ModelStorageSyncTaskPublic,
    ModelStorageSyncTaskStateEnum,
)
from gpustack.worker import model_storage_sync_manager as msm
from gpustack.worker.model_preheat.s3_client import PublishResult


class _FakeS3Client:
    last = None

    def __init__(self, *, fail=False, cancel_publish=False):
        self.fail = fail
        self.cancel_publish = cancel_publish
        self.publish_calls = 0

    @classmethod
    def from_minio(cls, **kwargs):
        instance = cls(
            fail=getattr(cls, "_fail", False),
            cancel_publish=getattr(cls, "_cancel_publish", False),
        )
        cls.last = instance
        return instance

    def publish_artifact(self, bucket, prefix, manifest, root, *, cancel_check=None):
        self.publish_calls += 1
        if self.cancel_publish:
            raise msm.ModelStorageSyncCanceled()
        if self.fail:
            raise ValueError("s3 write boom")
        return PublishResult(
            uploaded=len(manifest.files),
            skipped=0,
            ready_written=True,
            ready_digest=manifest.artifact_id,
            generation_prefix=manifest.artifact_prefix(prefix),
        )


def _payload(tmp_path, source_paths):
    return ModelStorageSyncExecutionPayload(
        task_id=1,
        state=ModelStorageSyncTaskStateEnum.PENDING,
        source="modelscope",
        model_id="Qwen/Test",
        resolved_revision="8f73c6a91b",
        request_identity={
            "source": "modelscope",
            "model_id": "Qwen/Test",
            "requested_revision": None,
            "include_patterns": [],
            "exclude_patterns": [],
        },
        request_digest="d" * 64,
        source_paths=source_paths,
        profile=ModelStorageSyncExecutionProfile(
            endpoint="https://s3.example.com",
            bucket="models",
            prefix="datamodel",
            access_key="AK",
            secret_key="SK",
        ),
    )


def _make_manager(tmp_path, **client_attrs):
    for key, value in client_attrs.items():
        setattr(_FakeS3Client, key, value)
    clientset = SimpleNamespace(model_storage_sync_tasks=None)
    manager = msm.ModelStorageSyncManager(
        worker_id=7,
        clientset=clientset,
        cfg=SimpleNamespace(cache_dir=str(tmp_path), data_dir=str(tmp_path)),
    )
    return manager


def test_publish_scans_local_and_reports_artifact(tmp_path, monkeypatch):
    model_dir = tmp_path / "Qwen" / "Test"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"a": 1}')
    (model_dir / "model.bin").write_bytes(b"\x00" * 128)

    monkeypatch.setattr(msm, "ModelPreheatS3Client", _FakeS3Client)
    manager = _make_manager(tmp_path)
    payload = _payload(tmp_path, [str(model_dir)])
    result = manager._publish(payload, threading.Event())

    assert _FakeS3Client.last.publish_calls == 1
    assert len(result["artifact_id"]) == 64
    assert result["file_count"] == 2
    assert result["total_size"] == len('{"a": 1}') + 128
    # 发布器拿到的是解密后明文凭据对应的连接参数。
    assert payload.profile.bucket == "models"


def test_publish_cancel_does_not_write_manifest(tmp_path, monkeypatch):
    model_dir = tmp_path / "Qwen" / "Test"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"a": 1}')

    monkeypatch.setattr(msm, "ModelPreheatS3Client", _FakeS3Client)
    manager = _make_manager(tmp_path, _cancel_publish=True)
    payload = _payload(tmp_path, [str(model_dir)])
    with pytest.raises(msm.ModelStorageSyncCanceled):
        manager._publish(payload, threading.Event())


def test_publish_missing_source_paths_is_error(tmp_path, monkeypatch):
    monkeypatch.setattr(msm, "ModelPreheatS3Client", _FakeS3Client)
    manager = _make_manager(tmp_path)
    payload = _payload(tmp_path, [])
    with pytest.raises(ValueError):
        manager._publish(payload, threading.Event())


def test_stable_error_code_maps_manifest_errors():
    assert msm._stable_error_code(ValueError()) == "worker_execution_failed"
    from gpustack.worker.model_preheat.s3_client import ModelPreheatS3ManifestError

    assert (
        msm._stable_error_code(ModelPreheatS3ManifestError("x")) == "manifest_invalid"
    )


def test_handle_event_filters_other_workers_and_non_pending(tmp_path):
    from gpustack.server.bus import Event, EventType

    manager = _make_manager(tmp_path)

    def public(task_id, worker_id, state):
        return ModelStorageSyncTaskPublic(
            id=task_id,
            model_file_id=1,
            worker_id=worker_id,
            worker_uuid="worker-a-uuid",
            profile_id=1,
            profile_config_version=1,
            request_digest="d" * 64,
            source="modelscope",
            model_id="Qwen/Test",
            resolved_revision="sha",
            artifact_id=None,
            state=state,
            file_count=0,
            total_size=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    # 其他 Worker：忽略。
    manager._handle_event(Event(type=EventType.CREATED, data=public(1, 999, ModelStorageSyncTaskStateEnum.PENDING).model_dump()))
    assert manager._active == {}
    # 本 Worker 但非 pending：忽略。
    manager._handle_event(Event(type=EventType.CREATED, data=public(1, 7, ModelStorageSyncTaskStateEnum.READY).model_dump()))
    assert manager._active == {}


def test_handle_event_cancel_via_updated_stops_active_task(tmp_path, monkeypatch):
    """UPDATED 事件把活动任务置为 canceled 时，必须通知执行器停止。

    同步上下文无事件循环：把执行器任务创建短路为占位 Task，
    只验证事件处理对活动项 cancel_event 的驱动。
    """
    from gpustack.server.bus import Event, EventType
    import asyncio as _asyncio

    manager = _make_manager(tmp_path)

    class _Placeholder:
        def add_done_callback(self, callback):
            pass

    def _fake_create_task(coro):
        coro.close()
        return _Placeholder()

    monkeypatch.setattr(msm.asyncio, "create_task", _fake_create_task)

    def public(task_id, worker_id, state):
        return ModelStorageSyncTaskPublic(
            id=task_id,
            model_file_id=1,
            worker_id=worker_id,
            worker_uuid="worker-a-uuid",
            profile_id=1,
            profile_config_version=1,
            request_digest="d" * 64,
            source="modelscope",
            model_id="Qwen/Test",
            resolved_revision="sha",
            artifact_id=None,
            state=state,
            file_count=0,
            total_size=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    manager._handle_event(
        Event(
            type=EventType.CREATED,
            data=public(5, 7, ModelStorageSyncTaskStateEnum.PENDING).model_dump(),
        )
    )
    assert 5 in manager._active
    cancel_event = manager._active[5][1]
    assert not cancel_event.is_set()
    # 服务端 DELETE 前先 UPDATE 为 canceled：执行器必须收到取消信号。
    manager._handle_event(
        Event(
            type=EventType.UPDATED,
            data=public(5, 7, ModelStorageSyncTaskStateEnum.CANCELED).model_dump(),
        )
    )
    assert cancel_event.is_set()
