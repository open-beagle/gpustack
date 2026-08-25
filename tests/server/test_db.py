import aiosqlite
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from gpustack.server.db import listen_events


def test_aiosqlite_supports_force_termination():
    assert callable(getattr(aiosqlite.Connection, "stop", None))


@pytest.mark.asyncio
async def test_sqlite_engine_dispose_closes_driver_connection_once(monkeypatch):
    close_calls = 0
    original_close = aiosqlite.Connection.close

    async def tracked_close(connection):
        nonlocal close_calls
        close_calls += 1
        await original_close(connection)

    monkeypatch.setattr(aiosqlite.Connection, "close", tracked_close)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    listen_events(engine)

    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    await engine.dispose()

    assert close_calls == 1
