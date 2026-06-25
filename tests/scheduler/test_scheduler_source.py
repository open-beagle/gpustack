from pathlib import Path


def test_scheduler_cycle_task_is_supervised():
    source = Path("gpustack/scheduler/scheduler.py").read_text()

    assert "async def _run_schedule_cycle(self)" in source
    assert "logger.error(f\"Scheduler cycle failed: {e}\")" in source
    assert "asyncio.create_task(self._run_schedule_cycle())" in source


def test_scheduler_cycle_restarts_after_unexpected_failure():
    source = Path("gpustack/scheduler/scheduler.py").read_text()

    assert "while True:" in source
    assert "await self._schedule_cycle()" in source
    assert "await asyncio.sleep(5)" in source


def test_scheduler_event_trigger_restarts_after_unexpected_failure():
    source = Path("gpustack/scheduler/scheduler.py").read_text()

    assert "async def _run_event_trigger(self)" in source
    assert "async for event in ModelInstance.subscribe(self._engine)" in source
    assert "logger.error(f\"Scheduler event trigger failed: {e}\")" in source
