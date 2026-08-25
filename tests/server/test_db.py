import aiosqlite
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from gpustack.server import db
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


@pytest.mark.asyncio
async def test_init_db_configures_resilient_connection_pool(monkeypatch):
    captured = {}
    engine = object()

    def fake_create_async_engine(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return engine

    async def fake_create_db_and_tables(actual_engine):
        assert actual_engine is engine

    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "create_async_engine", fake_create_async_engine)
    monkeypatch.setattr(db, "create_db_and_tables", fake_create_db_and_tables)
    monkeypatch.setattr(db, "listen_events", lambda actual_engine: None)

    await db.init_db("postgresql://gpustack:test@database/gpustack")

    assert captured["url"].startswith("postgresql+asyncpg://")
    assert captured["kwargs"]["pool_size"] == db.DB_POOL_SIZE
    assert captured["kwargs"]["max_overflow"] == db.DB_MAX_OVERFLOW
    assert captured["kwargs"]["pool_timeout"] == db.DB_POOL_TIMEOUT
    assert captured["kwargs"]["pool_recycle"] == db.DB_POOL_RECYCLE
    assert captured["kwargs"]["pool_pre_ping"] is True
    assert captured["kwargs"]["pool_use_lifo"] is True
