from pathlib import Path


def test_worker_background_tasks_keep_failure_propagation():
    source = Path("gpustack/worker/worker.py").read_text()

    assert "def _create_async_task(self, coro)" in source
    assert "asyncio.create_task(coro)" in source
    assert "async def _run_background_task(self, coro)" not in source


def test_worker_watchers_have_internal_retry_loops():
    serve_manager_source = Path("gpustack/worker/serve_manager.py").read_text()
    model_file_manager_source = Path("gpustack/worker/model_file_manager.py").read_text()

    assert "async def watch_model_instances(self)" in serve_manager_source
    assert "async def monitor_error_instances(self)" in serve_manager_source
    assert "await asyncio.sleep(5)" in serve_manager_source
    assert "async def watch_model_files(self)" in model_file_manager_source
    assert "await asyncio.sleep(5)" in model_file_manager_source
