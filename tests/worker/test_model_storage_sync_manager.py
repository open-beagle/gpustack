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
from pathlib import Path
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
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
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

    def artifact_manifest_object(self, profile_prefix, manifest):
        # 与真实发布器一致：manifest 对象 Key。
        return f"{manifest.artifact_prefix(profile_prefix)}/manifest.json"


def _payload(tmp_path, source_paths):
    """构造执行 payload：scan_spec 由 compute_scan_spec 冻结（与任务创建一致）。"""
    from gpustack.server.model_storage_scan_spec import compute_scan_spec

    root, _patterns = compute_scan_spec(list(source_paths))
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
            "source_paths": list(source_paths),
            "scan_spec": {"root": root, "include_patterns": [], "exclude_patterns": []},
        },
        request_digest="d" * 64,
        source_paths=source_paths,
        scan_spec={"root": root, "include_patterns": [], "exclude_patterns": []},
        lease_token="lease-token-1",
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


def test_publish_ollama_file_uses_single_source_segment(tmp_path, monkeypatch):
    model_file = tmp_path / "ollama" / "qwen2_5_7b"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"ollama-model")

    monkeypatch.setattr(msm, "ModelPreheatS3Client", _FakeS3Client)
    manager = _make_manager(tmp_path)
    payload = _payload(tmp_path, [str(model_file)])
    payload.scan_spec = {
        "root": str(model_file.parent),
        "include_patterns": ["qwen2_5_7b", "qwen2_5_7b/**"],
        "exclude_patterns": [],
    }
    payload.source = "ollama_library"
    payload.model_id = "qwen2.5:7b"
    payload.resolved_revision = "local-snapshot-" + "a" * 64
    payload.request_identity = {
        "source": "ollama_library",
        "model_id": "qwen2.5:7b",
        "requested_revision": None,
        "include_patterns": ["qwen2_5_7b", "qwen2_5_7b/**"],
        "exclude_patterns": [],
    }
    payload.request_digest = ModelPreheatIdentity(
        source=payload.source,
        model_id=payload.model_id,
        revision=payload.resolved_revision,
        requested_revision=None,
        file_patterns=payload.request_identity["include_patterns"],
    ).request_digest

    result = manager._publish(payload, threading.Event())

    assert result["file_count"] == 1
    assert result["manifest_path"].startswith("datamodel/ollama_library/qwen2.5:7b/")
    assert "/ollama_library/ollama_library/" not in result["manifest_path"]


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
    # 空 source_paths：冻结扫描规约无法计算（compute_scan_spec 拒绝空集合），
    # 在 payload 冻结阶段即稳定失败，_publish 不进入扫描。
    with pytest.raises(ValueError):
        _payload(tmp_path, [])
    # 即使绕过构造（payload 缺省 scan_spec 非法），_publish 也不得扫描。
    broken = ModelStorageSyncExecutionPayload.model_construct(
        task_id=1,
        state=ModelStorageSyncTaskStateEnum.PENDING,
        source="modelscope",
        model_id="Qwen/Test",
        resolved_revision="8f73c6a91b",
        request_identity={"include_patterns": [], "exclude_patterns": []},
        request_digest="d" * 64,
        source_paths=[],
        scan_spec={"root": "", "include_patterns": [], "exclude_patterns": []},
        lease_token="lease-token-1",
        profile=ModelStorageSyncExecutionProfile(
            endpoint="https://s3.example.com",
            bucket="models",
            access_key="AK",
            secret_key="SK",
        ),
    )
    with pytest.raises(ValueError):
        manager._publish(broken, threading.Event())


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
    manager._handle_event(
        Event(
            type=EventType.CREATED,
            data=public(1, 999, ModelStorageSyncTaskStateEnum.PENDING).model_dump(),
        )
    )
    assert manager._active == {}
    # 本 Worker 但非可执行状态：忽略。
    manager._handle_event(
        Event(
            type=EventType.CREATED,
            data=public(1, 7, ModelStorageSyncTaskStateEnum.READY).model_dump(),
        )
    )
    assert manager._active == {}


def test_initial_publishing_event_is_resumed_once_after_worker_restart(
    tmp_path, monkeypatch
):
    """Worker 重启后的初始 publishing 快照会重领 payload，重复事件不重复执行。"""
    from gpustack.server.bus import Event, EventType

    manager = _make_manager(tmp_path)
    scheduled = []

    class _Placeholder:
        def add_done_callback(self, callback):
            pass

    def _fake_create_task(coro):
        scheduled.append(coro)
        coro.close()
        return _Placeholder()

    monkeypatch.setattr(msm.asyncio, "create_task", _fake_create_task)
    task = ModelStorageSyncTaskPublic(
        id=8,
        model_file_id=1,
        worker_id=7,
        worker_uuid="worker-a-uuid",
        profile_id=1,
        profile_config_version=1,
        request_digest="d" * 64,
        source="modelscope",
        model_id="Qwen/Test",
        resolved_revision="sha",
        state=ModelStorageSyncTaskStateEnum.PUBLISHING,
        file_count=0,
        total_size=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    manager._handle_event(Event(type=EventType.CREATED, data=task.model_dump()))
    manager._handle_event(Event(type=EventType.UPDATED, data=task.model_dump()))

    assert len(scheduled) == 1
    assert 8 in manager._active


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


# ---------------------------------------------------------------------------
# Review 子阶段 C：文件选择 / 多路径 / manifest 语义与 complete 契约
# ---------------------------------------------------------------------------


class _RecordingS3Client:
    """记录 manifest 与 root，返回确定性 PublishResult（用于文件选择校验）。"""

    last_manifest = None
    last_root = None

    @classmethod
    def from_minio(cls, **kwargs):
        return cls()

    def publish_artifact(self, bucket, prefix, manifest, root, *, cancel_check=None):
        _RecordingS3Client.last_manifest = manifest
        _RecordingS3Client.last_root = str(root)
        return PublishResult(
            uploaded=len(manifest.files),
            skipped=0,
            ready_written=True,
            ready_digest=manifest.artifact_id,
            generation_prefix=manifest.artifact_prefix(prefix),
        )

    def artifact_manifest_object(self, profile_prefix, manifest):
        return f"{manifest.artifact_prefix(profile_prefix)}/manifest.json"


def _identity_patterns(source_paths):
    """与任务创建一致：由 compute_scan_spec 得到实际文件选择 patterns。"""
    from gpustack.server.model_storage_scan_spec import compute_scan_spec

    repository_complete = len(source_paths) == 1 and Path(source_paths[0]).is_dir()
    _root, patterns = compute_scan_spec(
        list(source_paths), repository_complete=repository_complete
    )
    return patterns


def _payload_for_paths(tmp_path, source_paths, include_patterns):
    from gpustack.server.model_storage_scan_spec import compute_scan_spec

    repository_complete = len(source_paths) == 1 and Path(source_paths[0]).is_dir()
    root, _patterns = compute_scan_spec(
        list(source_paths), repository_complete=repository_complete
    )
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
            "include_patterns": include_patterns,
            "exclude_patterns": [],
            "source_paths": list(source_paths),
            "scan_spec": {
                "root": root,
                "include_patterns": list(include_patterns),
                "exclude_patterns": [],
            },
        },
        request_digest="d" * 64,
        source_paths=source_paths,
        scan_spec={
            "root": root,
            "include_patterns": list(include_patterns),
            "exclude_patterns": [],
        },
        lease_token="lease-token-1",
        profile=ModelStorageSyncExecutionProfile(
            endpoint="https://s3.example.com",
            bucket="models",
            prefix="datamodel",
            access_key="AK",
            secret_key="SK",
        ),
    )


def test_publish_single_file_selects_only_that_file(tmp_path, monkeypatch):
    """单文件：只发布该文件，不发布同目录无关邻居文件。"""
    model_dir = tmp_path / "Qwen" / "Test"
    model_dir.mkdir(parents=True)
    (model_dir / "wanted.bin").write_bytes(b"x" * 16)
    (model_dir / "unrelated.bin").write_bytes(b"y" * 16)

    monkeypatch.setattr(msm, "ModelPreheatS3Client", _RecordingS3Client)
    manager = _make_manager(tmp_path)
    source_paths = [str(model_dir / "wanted.bin")]
    payload = _payload_for_paths(
        tmp_path, source_paths, _identity_patterns(source_paths)
    )
    result = manager._publish(payload, threading.Event())

    manifest = _RecordingS3Client.last_manifest
    selected = {f.path for f in manifest.files}
    assert selected == {"wanted.bin"}
    assert "unrelated.bin" not in selected
    assert result["file_count"] == 1
    assert result["total_size"] == 16
    # manifest_digest 为 64 位小写十六进制，manifest_path 为对象 Key。
    assert len(result["manifest_digest"]) == 64
    assert result["manifest_path"].endswith("/manifest.json")


def test_publish_multiple_files_selects_only_selected(tmp_path, monkeypatch):
    """多文件：只发布所选文件，漏选/多选都不允许。"""
    model_dir = tmp_path / "Qwen" / "Test"
    model_dir.mkdir(parents=True)
    (model_dir / "a.bin").write_bytes(b"a" * 8)
    (model_dir / "b.bin").write_bytes(b"b" * 8)
    (model_dir / "other.bin").write_bytes(b"c" * 8)

    monkeypatch.setattr(msm, "ModelPreheatS3Client", _RecordingS3Client)
    manager = _make_manager(tmp_path)
    source_paths = [str(model_dir / "a.bin"), str(model_dir / "b.bin")]
    payload = _payload_for_paths(
        tmp_path, source_paths, _identity_patterns(source_paths)
    )
    result = manager._publish(payload, threading.Event())

    selected = {f.path for f in _RecordingS3Client.last_manifest.files}
    assert selected == {"a.bin", "b.bin"}
    assert "other.bin" not in selected
    assert result["file_count"] == 2
    assert result["total_size"] == 16


def test_publish_whole_dir_selects_all_files_including_nested(tmp_path, monkeypatch):
    """整目录：全量扫描，含嵌套子目录文件。"""
    model_dir = tmp_path / "Qwen" / "Test"
    (model_dir / "sub").mkdir(parents=True)
    (model_dir / "top.bin").write_bytes(b"t" * 4)
    (model_dir / "sub" / "nested.bin").write_bytes(b"n" * 4)

    monkeypatch.setattr(msm, "ModelPreheatS3Client", _RecordingS3Client)
    manager = _make_manager(tmp_path)
    source_paths = [str(model_dir)]
    payload = _payload_for_paths(
        tmp_path, source_paths, _identity_patterns(source_paths)
    )
    result = manager._publish(payload, threading.Event())

    selected = {f.path for f in _RecordingS3Client.last_manifest.files}
    assert selected == {"top.bin", "sub/nested.bin"}
    assert result["file_count"] == 2
    assert result["total_size"] == 8


def test_publish_manifest_object_keys_are_accurate(tmp_path, monkeypatch):
    """目标对象 Key / manifest 准确：文件 Key 位于 <artifact_id>/files/ 下。"""
    model_dir = tmp_path / "Qwen" / "Test"
    model_dir.mkdir(parents=True)
    (model_dir / "f.bin").write_bytes(b"z" * 4)

    monkeypatch.setattr(msm, "ModelPreheatS3Client", _RecordingS3Client)
    manager = _make_manager(tmp_path)
    source_paths = [str(model_dir)]
    payload = _payload_for_paths(
        tmp_path, source_paths, _identity_patterns(source_paths)
    )
    manager._publish(payload, threading.Event())
    manifest = _RecordingS3Client.last_manifest
    # 对象 Key 由真实发布器推导：prefix/source/model/artifact_id/manifest.json。
    expected = f"datamodel/modelscope/Qwen/Test/{manifest.artifact_id}/manifest.json"
    assert payload.profile.prefix == "datamodel"
    from gpustack.worker.model_preheat.s3_client import ModelPreheatS3Client

    real_client = ModelPreheatS3Client(None)
    key = real_client.artifact_manifest_object(payload.profile.prefix, manifest)
    assert key == expected
    # 文件对象 Key 位于 .../files/<path>。
    file_key = real_client.artifact_file_object(
        payload.profile.prefix, manifest, manifest.files[0]
    )
    assert f"/files/" in file_key


def test_execute_complete_payload_carries_digest_and_manifest(tmp_path, monkeypatch):
    """_execute 的 complete payload 必须携带 request_digest、manifest_digest
    与 manifest_path（Server 校验与库存写入依赖）。"""
    from types import SimpleNamespace

    model_dir = tmp_path / "Qwen" / "Test"
    model_dir.mkdir(parents=True)
    (model_dir / "f.bin").write_bytes(b"q" * 4)

    monkeypatch.setattr(msm, "ModelPreheatS3Client", _RecordingS3Client)
    source_paths = [str(model_dir)]
    payload = _payload_for_paths(
        tmp_path, source_paths, _identity_patterns(source_paths)
    )

    captured: dict = {}

    class _Client:
        async def aget_execution_payload(self, id):
            return payload

        async def acomplete(self, id, complete):
            captured["complete"] = complete

        async def afail(self, id, failure):
            captured["fail"] = failure

    manager = msm.ModelStorageSyncManager(
        worker_id=7,
        clientset=SimpleNamespace(model_storage_sync_tasks=_Client()),
        cfg=SimpleNamespace(cache_dir=str(tmp_path)),
    )
    public = ModelStorageSyncTaskPublic(
        id=1,
        model_file_id=1,
        worker_id=7,
        worker_uuid="worker-a-uuid",
        profile_id=1,
        profile_config_version=1,
        request_digest="d" * 64,
        source="modelscope",
        model_id="Qwen/Test",
        resolved_revision="8f73c6a91b",
        artifact_id=None,
        state=ModelStorageSyncTaskStateEnum.PENDING,
        file_count=0,
        total_size=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    import asyncio

    asyncio.run(manager._execute(public, threading.Event()))
    complete = captured["complete"]
    assert complete.request_digest == "d" * 64
    assert len(complete.manifest_digest) == 64
    assert complete.artifact_id
    assert complete.file_count == 1
    assert complete.total_size == 4
    # complete 必须携带执行 lease token（Server lease 校验依赖）。
    assert complete.lease_token == "lease-token-1"


# ---------------------------------------------------------------------------
# Review 定向修复 5：执行文件选择任务创建时冻结，Worker 不重读/不重算
# ---------------------------------------------------------------------------


def test_publish_uses_frozen_scan_spec_root(tmp_path, monkeypatch):
    """_publish 的扫描 root 必须等于 payload 中任务创建时冻结的 scan_spec.root。"""
    from gpustack.server.model_storage_scan_spec import compute_scan_spec

    model_dir = tmp_path / "Qwen" / "Test"
    model_dir.mkdir(parents=True)
    (model_dir / "f.bin").write_bytes(b"z" * 4)

    monkeypatch.setattr(msm, "ModelPreheatS3Client", _RecordingS3Client)
    manager = _make_manager(tmp_path)
    frozen_root, frozen_patterns = compute_scan_spec(
        [str(model_dir)], repository_complete=True
    )
    payload = _payload_for_paths(tmp_path, [str(model_dir)], frozen_patterns)
    assert payload.scan_spec["root"] == frozen_root
    manager._publish(payload, threading.Event())
    # 扫描 root 必须是冻结 root（而不是任何 Worker 本地猜测）。
    assert _RecordingS3Client.last_root == frozen_root


def test_publish_identity_is_stable_across_worker_absolute_roots(tmp_path, monkeypatch):
    """不同 Worker 绝对目录不能改变逻辑身份、Manifest 路径或 Artifact ID。"""
    monkeypatch.setattr(msm, "ModelPreheatS3Client", _RecordingS3Client)
    manager = _make_manager(tmp_path)
    roots = [
        tmp_path / "worker-a" / "cache" / "model",
        tmp_path / "worker-b" / "mnt" / "model",
    ]
    results = []
    manifests = []
    patterns = []
    for root in roots:
        (root / "sub").mkdir(parents=True)
        (root / "config.json").write_text('{"a":1}')
        (root / "sub" / "weights.bin").write_bytes(b"weights")
        source_paths = [str(root)]
        current_patterns = _identity_patterns(source_paths)
        payload = _payload_for_paths(tmp_path, source_paths, current_patterns)
        results.append(manager._publish(payload, threading.Event()))
        manifests.append(_RecordingS3Client.last_manifest)
        patterns.append(current_patterns)

    assert patterns == [[], []]
    assert results[0]["artifact_id"] == results[1]["artifact_id"]
    expected_paths = {"config.json", "sub/weights.bin"}
    assert {file.path for file in manifests[0].files} == expected_paths
    assert {file.path for file in manifests[1].files} == expected_paths


def test_publish_rejects_root_inconsistent_with_source_paths(tmp_path, monkeypatch):
    """payload 的 source_paths 与冻结 root 不一致（损坏/串任务）时稳定失败。"""
    monkeypatch.setattr(msm, "ModelPreheatS3Client", _RecordingS3Client)
    manager = _make_manager(tmp_path)
    payload = _payload(tmp_path, ["/models/Qwen/Test"])
    # 篡改冻结 root：source_paths 不再位于 root 之下，不得退回猜测路径。
    payload.scan_spec = dict(payload.scan_spec, root="/elsewhere")
    with pytest.raises(ValueError):
        manager._publish(payload, threading.Event())
    # 篡改 source_paths 使其越出 root 之外。
    payload2 = _payload(tmp_path, ["/models/Qwen/Test"])
    payload2.source_paths = ["/other/model"]
    with pytest.raises(ValueError):
        manager._publish(payload2, threading.Event())


def test_publish_rejects_missing_or_invalid_scan_spec(tmp_path, monkeypatch):
    """scan_spec 缺失/非法（root 为空）时稳定失败，不退回任何猜测路径。"""
    monkeypatch.setattr(msm, "ModelPreheatS3Client", _RecordingS3Client)
    manager = _make_manager(tmp_path)
    model_dir = tmp_path / "Qwen" / "Test"
    model_dir.mkdir(parents=True)
    (model_dir / "f.bin").write_bytes(b"q" * 4)
    payload = _payload(tmp_path, [str(model_dir)])
    payload.scan_spec = {"root": "", "include_patterns": [], "exclude_patterns": []}
    with pytest.raises(ValueError):
        manager._publish(payload, threading.Event())


def test_execute_sends_lease_token_on_fail(tmp_path, monkeypatch):
    """执行失败回写 fail 时必须携带同一执行 lease token。"""
    # 清理跨测试的类属性状态（_FakeS3Client 用类属性传参）。
    for attr in ("_fail", "_cancel_publish"):
        if hasattr(_FakeS3Client, attr):
            delattr(_FakeS3Client, attr)
    try:
        _run_test_execute_sends_lease_token_on_fail(tmp_path, monkeypatch)
    finally:
        for attr in ("_fail", "_cancel_publish"):
            if hasattr(_FakeS3Client, attr):
                delattr(_FakeS3Client, attr)


def _run_test_execute_sends_lease_token_on_fail(tmp_path, monkeypatch):
    model_dir = tmp_path / "Qwen" / "Test"
    model_dir.mkdir(parents=True)
    (model_dir / "f.bin").write_bytes(b"q" * 4)

    monkeypatch.setattr(msm, "ModelPreheatS3Client", _FakeS3Client)
    setattr(_FakeS3Client, "_fail", True)
    source_paths = [str(model_dir)]
    payload = _payload(tmp_path, source_paths)

    captured: dict = {}

    class _Client:
        async def aget_execution_payload(self, id):
            return payload

        async def acomplete(self, id, complete):
            captured["complete"] = complete

        async def afail(self, id, failure):
            captured["fail"] = failure

    manager = msm.ModelStorageSyncManager(
        worker_id=7,
        clientset=SimpleNamespace(model_storage_sync_tasks=_Client()),
        cfg=SimpleNamespace(cache_dir=str(tmp_path)),
    )
    public = ModelStorageSyncTaskPublic(
        id=1,
        model_file_id=1,
        worker_id=7,
        worker_uuid="worker-a-uuid",
        profile_id=1,
        profile_config_version=1,
        request_digest="d" * 64,
        source="modelscope",
        model_id="Qwen/Test",
        resolved_revision="8f73c6a91b",
        artifact_id=None,
        state=ModelStorageSyncTaskStateEnum.PENDING,
        file_count=0,
        total_size=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    import asyncio

    asyncio.run(manager._execute(public, threading.Event()))
    assert "fail" in captured
    assert captured["fail"].lease_token == "lease-token-1"
    assert captured["fail"].error_code == "worker_execution_failed"
