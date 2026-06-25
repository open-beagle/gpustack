from pathlib import Path


def test_server_core_background_tasks_keep_failure_propagation():
    source = Path("gpustack/server/server.py").read_text()

    assert "def _create_async_task(self, coro)" in source
    assert "asyncio.create_task(coro)" in source
    assert "async def _run_background_task(self, coro)" not in source


def test_server_long_running_watchers_are_restartable():
    source = Path("gpustack/server/server.py").read_text()

    assert "async def _run_restartable_background_task" in source
    assert "await asyncio.sleep(5)" in source
    assert "self._create_async_task(scheduler.start())" in source
    assert (
        "self._create_restartable_async_task(model_controller.start, \"model controller\")"
        in source
    )
    assert "model_instance_controller.start" in source
    assert "\"model instance controller\"" in source


def test_restartable_background_task_backs_off_after_normal_return():
    source = Path("gpustack/server/server.py").read_text()

    assert "await task_factory()\n                await asyncio.sleep(5)" in source
