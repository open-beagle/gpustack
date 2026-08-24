from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gpustack.schemas.models import (
    ModelInstance,
    ModelInstanceStateEnum,
    SourceEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker import serve_manager


def _instance(**overrides):
    values = {
        "id": 1,
        "name": "ollama-fUK4a",
        "model_id": 1,
        "model_name": "qwen",
        "worker_id": 1,
        "source": SourceEnum.LOCAL_PATH,
        "local_path": "/models/qwen",
        "state": ModelInstanceStateEnum.ERROR,
        "restart_count": 0,
        "updated_at": datetime.now(timezone.utc) - timedelta(minutes=10),
    }
    values.update(overrides)
    return ModelInstance(**values)


def _manager(tmp_path):
    return serve_manager.ServeManager(
        worker_id=1,
        clientset=SimpleNamespace(),
        cfg=SimpleNamespace(
            log_dir=str(tmp_path / "logs"),
            cache_dir=str(tmp_path / "cache"),
            service_port_range=None,
        ),
    )


def test_auto_restart_stops_after_consecutive_failure_limit(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    manager._clientset = SimpleNamespace(
        models=SimpleNamespace(
            get=lambda model_id: SimpleNamespace(restart_on_error=True)
        )
    )
    updates = []
    monkeypatch.setattr(
        manager,
        "_update_model_instance",
        lambda instance_id, **kwargs: updates.append((instance_id, kwargs)),
    )
    instance = _instance(restart_count=5)
    manager._error_model_instances[instance.id] = instance

    manager._restart_error_instance(instance)

    assert updates == [
        (
            instance.id,
            {
                "state": ModelInstanceStateEnum.ERROR,
                "state_message": "连续自动重启已达到上限（5 次），已停止自动重启。请检查服务日志后手动重新部署。",
            },
        )
    ]
    assert manager._error_model_instances == {}

    # 同一条已终止的错误事件不得再次写库或重复打日志。
    manager._restart_error_instance(instance)
    assert len(updates) == 1


def test_auto_restart_limit_retries_after_state_update_failure(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    manager._clientset = SimpleNamespace(
        models=SimpleNamespace(
            get=lambda model_id: SimpleNamespace(restart_on_error=True)
        )
    )
    instance = _instance(restart_count=5)
    manager._error_model_instances[instance.id] = instance
    updates = []

    def fail_once(instance_id, **kwargs):
        updates.append((instance_id, kwargs))
        if len(updates) == 1:
            raise RuntimeError("temporary API failure")

    monkeypatch.setattr(manager, "_update_model_instance", fail_once)

    manager._restart_error_instance(instance)
    assert manager._error_model_instances == {instance.id: instance}
    assert manager._restart_limit_reported_model_instances == set()

    manager._restart_error_instance(instance)
    assert len(updates) == 2
    assert manager._error_model_instances == {}
    assert manager._restart_limit_reported_model_instances == {instance.id}


def test_disabling_auto_restart_removes_already_queued_instance(tmp_path):
    manager = _manager(tmp_path)
    instance = _instance()
    manager._error_model_instances[instance.id] = instance
    manager._model_cache_by_instance[instance.id] = SimpleNamespace(
        restart_on_error=True
    )
    manager._clientset = SimpleNamespace(
        models=SimpleNamespace(
            get=lambda model_id: SimpleNamespace(restart_on_error=False)
        )
    )

    manager._restart_error_instance(instance)

    assert manager._error_model_instances == {}
    assert manager._model_cache_by_instance[instance.id].restart_on_error is False


def test_running_event_resets_upgrade_leftover_restart_counter(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    instance = _instance(
        state=ModelInstanceStateEnum.RUNNING,
        restart_count=87,
        last_restart_time=datetime.now(timezone.utc),
    )
    manager._serving_model_instances[instance.id] = SimpleNamespace(
        is_alive=lambda: True
    )
    updates = []
    monkeypatch.setattr(
        manager,
        "_update_model_instance",
        lambda instance_id, **kwargs: updates.append((instance_id, kwargs)),
    )
    monkeypatch.setattr(
        serve_manager.logger, "trace", lambda message: None, raising=False
    )

    manager._handle_model_instance_event(
        Event(type=EventType.UPDATED, data=instance.model_dump())
    )

    assert updates == [
        (
            instance.id,
            {"restart_count": 0, "last_restart_time": None},
        )
    ]


def test_deleting_running_instance_does_not_reset_restart_counter(
    tmp_path, monkeypatch
):
    manager = _manager(tmp_path)
    instance = _instance(
        state=ModelInstanceStateEnum.RUNNING,
        restart_count=87,
        last_restart_time=datetime.now(timezone.utc),
    )
    manager._serving_model_instances[instance.id] = SimpleNamespace(
        is_alive=lambda: True
    )
    updates = []
    stopped = []
    monkeypatch.setattr(
        manager,
        "_update_model_instance",
        lambda instance_id, **kwargs: updates.append((instance_id, kwargs)),
    )
    monkeypatch.setattr(manager, "_stop_model_instance", lambda mi: stopped.append(mi))
    monkeypatch.setattr(
        serve_manager.logger, "trace", lambda message: None, raising=False
    )

    manager._handle_model_instance_event(
        Event(type=EventType.DELETED, data=instance.model_dump())
    )

    assert updates == []
    assert stopped == [instance]


def test_running_resets_consecutive_restart_counter(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    instance = _instance(
        state=ModelInstanceStateEnum.INITIALIZING,
        restart_count=3,
        last_restart_time=datetime.now(timezone.utc),
    )
    manager._serving_model_instances[instance.id] = SimpleNamespace(
        is_alive=lambda: True
    )
    manager._starting_model_instances[instance.id] = instance
    manager._model_cache_by_instance[instance.id] = SimpleNamespace()
    manager._clientset = SimpleNamespace(
        model_instances=SimpleNamespace(get=lambda id: instance)
    )
    updates = []
    monkeypatch.setattr(serve_manager, "get_backend", lambda model: "llama.cpp")
    monkeypatch.setattr(serve_manager, "is_ready", lambda backend, model: True)
    monkeypatch.setattr(
        manager,
        "_update_model_instance",
        lambda instance_id, **kwargs: updates.append((instance_id, kwargs)),
    )

    manager.health_check_serving_instances()

    assert updates == [
        (
            instance.id,
            {
                "state": ModelInstanceStateEnum.RUNNING,
                "state_message": "",
                "placement_override": None,
                "restart_count": 0,
                "last_restart_time": None,
            },
        )
    ]
