import asyncio
import importlib.util
import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import create_engine, inspect
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateTable
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.auth import get_admin_user
from gpustack.config.config import Config
from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
)
from gpustack.schemas.model_preheats import ModelPreheatCreate, ModelPreheatTask
from gpustack.schemas.users import User
from gpustack.server.db import get_session
from gpustack.worker.downloaders import _preheat_file_selected


def _load_schedule_migrations(prefix):
    def load(name, filename):
        path = Path("gpustack/migrations/versions") / filename
        spec = importlib.util.spec_from_file_location(name, path)
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        return migration

    return (
        load(
            f"{prefix}_initial",
            "2026_08_12_1000-11caac4ba6d4_add_model_preheat_schedules.py",
        ),
        load(
            f"{prefix}_successor",
            "2026_08_12_1800-b0307846729c_fix_model_preheat_schedule_ddl.py",
        ),
    )


def _seed_schedule_migration_source(connection, initial, operation_keys):
    connection.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY)")
    connection.exec_driver_sql(
        "CREATE TABLE model_preheat_s3_profiles (id INTEGER PRIMARY KEY)"
    )
    connection.exec_driver_sql(
        "CREATE TABLE model_preheat_tasks "
        "(id INTEGER PRIMARY KEY, schedule_id INTEGER)"
    )
    initial.upgrade()
    connection.exec_driver_sql(
        "CREATE TABLE model_preheat_worker_tasks ("
        "id INTEGER PRIMARY KEY, task_id INTEGER, payload TEXT, "
        "FOREIGN KEY(task_id) REFERENCES model_preheat_tasks(id) "
        "ON DELETE CASCADE)"
    )
    connection.exec_driver_sql(
        "CREATE TABLE model_preheat_distribution_policies ("
        "id INTEGER PRIMARY KEY, created_by_task_id INTEGER, payload TEXT, "
        "FOREIGN KEY(created_by_task_id) REFERENCES model_preheat_tasks(id) "
        "ON DELETE RESTRICT)"
    )
    connection.exec_driver_sql("INSERT INTO users (id) VALUES (1)")
    connection.exec_driver_sql("INSERT INTO model_preheat_s3_profiles (id) VALUES (1)")
    connection.exec_driver_sql(
        "INSERT INTO model_preheat_schedules ("
        "id, name, enabled, cron_expression, timezone, "
        "window_duration_minutes, max_concurrency, source, model_id, "
        "revision, include_patterns, exclude_patterns, target_scope, "
        "target_worker_uuids, seed_worker_uuid, s3_profile_id, "
        "s3_backfill_policy, keep_new_workers_in_sync, created_by_user_id, "
        "next_window_start_utc, last_window_start_utc, deleted_at, "
        "created_at, updated_at) VALUES ("
        "1, 'nightly', 1, '0 1 * * *', 'UTC', 60, 2, "
        "'huggingface', 'org/model', 'revision', '[]', '[]', "
        "'selected_workers', '[\"worker-a\"]', NULL, 1, "
        "'when_missing', 0, 1, NULL, NULL, NULL, "
        "'2026-08-12 00:00:00', '2026-08-12 00:00:00')"
    )
    connection.exec_driver_sql(
        "INSERT INTO model_preheat_tasks (id, schedule_id) VALUES (1, 1)"
    )
    connection.exec_driver_sql(
        "INSERT INTO model_preheat_worker_tasks "
        "(id, task_id, payload) VALUES (1, 1, 'cascade-row')"
    )
    connection.exec_driver_sql(
        "INSERT INTO model_preheat_distribution_policies "
        "(id, created_by_task_id, payload) VALUES (1, 1, 'restrict-row')"
    )
    for index, operation_key in enumerate(operation_keys, start=1):
        hour = index - 1
        connection.exec_driver_sql(
            "INSERT INTO model_preheat_schedule_runs ("
            "id, schedule_id, window_start_utc, window_end_utc, `trigger`, "
            "state, operation_key, slot, task_id, created_by_user_id, "
            "error_code, started_at, finished_at, deleted_at, created_at, "
            "updated_at) VALUES (?, 1, ?, ?, 'scheduled', 'running', ?, ?, 1, 1, "
            "NULL, ?, NULL, NULL, ?, ?)",
            (
                index,
                f"2026-08-12 {hour:02d}:00:00",
                f"2026-08-12 {hour:02d}:30:00",
                operation_key,
                index - 1,
                f"2026-08-12 {hour:02d}:00:00",
                f"2026-08-12 {hour:02d}:00:00",
                f"2026-08-12 {hour:02d}:00:00",
            ),
        )


def _schedule_migration_state(connection):
    inspector = inspect(connection)
    unique_constraints = {
        constraint["name"]: constraint["column_names"]
        for constraint in inspector.get_unique_constraints(
            "model_preheat_schedule_runs"
        )
    }
    return {
        "schedule_bandwidth": "bandwidth_limit_mbps"
        in {
            column["name"]
            for column in inspector.get_columns("model_preheat_schedules")
        },
        "task_bandwidth": "bandwidth_limit_mbps"
        in {column["name"] for column in inspector.get_columns("model_preheat_tasks")},
        "task_schedule_fk": any(
            foreign_key["referred_table"] == "model_preheat_schedules"
            for foreign_key in inspector.get_foreign_keys("model_preheat_tasks")
        ),
        "operation_key_nullable": next(
            column["nullable"]
            for column in inspector.get_columns("model_preheat_schedule_runs")
            if column["name"] == "operation_key"
        ),
        "window_unique": unique_constraints["uix_preheat_schedule_window"],
    }


def _expected_schedule_migration_state(upgraded):
    return {
        "schedule_bandwidth": upgraded,
        "task_bandwidth": upgraded,
        "task_schedule_fk": upgraded,
        "operation_key_nullable": not upgraded,
        "window_unique": [
            "schedule_id",
            "window_start_utc",
            *(["operation_key"] if upgraded else []),
        ],
    }


def _assert_schedule_migration_data_preserved(connection, expected_run_count=1):
    assert connection.exec_driver_sql(
        "SELECT task_id, payload FROM model_preheat_worker_tasks"
    ).all() == [(1, "cascade-row")]
    assert connection.exec_driver_sql(
        "SELECT created_by_task_id, payload FROM model_preheat_distribution_policies"
    ).all() == [(1, "restrict-row")]
    assert (
        connection.exec_driver_sql(
            "SELECT task_id FROM model_preheat_schedule_runs ORDER BY id"
        ).all()
        == [(1,)] * expected_run_count
    )
    assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []


def _assert_no_alembic_temp_tables(connection):
    assert (
        connection.exec_driver_sql(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name LIKE '_alembic_tmp_%'"
        ).all()
        == []
    )


def test_schedule_window_calculation_handles_cross_day_and_dst():
    from gpustack.schemas.model_preheat_schedules import (
        ModelPreheatScheduleCreate,
        next_window_start_utc,
        window_end_utc,
    )

    spring = ModelPreheatScheduleCreate(
        name="spring",
        cron_expression="30 2 * * *",
        timezone="America/New_York",
        window_duration_minutes=180,
        source="huggingface",
        model_id="org/model",
        s3_profile_id=1,
        target_worker_uuids=["worker-a"],
    )
    spring_start = next_window_start_utc(
        spring,
        datetime(2026, 3, 8, 6, 0, tzinfo=timezone.utc),
    )
    assert spring_start == datetime(2026, 3, 9, 6, 30, tzinfo=timezone.utc)
    assert window_end_utc(spring_start, spring.window_duration_minutes) == datetime(
        2026, 3, 9, 9, 30, tzinfo=timezone.utc
    )

    autumn = spring.model_copy(
        update={"cron_expression": "30 1 * * *", "window_duration_minutes": 60}
    )
    first_fold = next_window_start_utc(
        autumn,
        datetime(2026, 11, 1, 4, 0, tzinfo=timezone.utc),
    )
    after_first_fold = next_window_start_utc(
        autumn,
        datetime(2026, 11, 1, 5, 31, tzinfo=timezone.utc),
    )
    assert first_fold == datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc)
    assert after_first_fold == datetime(2026, 11, 2, 6, 30, tzinfo=timezone.utc)

    cross_day = spring.model_copy(update={"window_duration_minutes": 240})
    assert window_end_utc(
        datetime(2026, 8, 14, 23, 0, tzinfo=timezone.utc),
        cross_day.window_duration_minutes,
    ) == datetime(2026, 8, 15, 3, 0, tzinfo=timezone.utc)


def test_schedule_input_rejects_invalid_time_and_concurrency():
    from gpustack.schemas.model_preheat_schedules import ModelPreheatScheduleCreate

    base = {
        "name": "nightly",
        "cron_expression": "0 1 * * *",
        "timezone": "UTC",
        "window_duration_minutes": 60,
        "source": "modelscope",
        "model_id": "Qwen/Test",
        "s3_profile_id": 1,
        "target_worker_uuids": ["worker-a"],
    }
    for update in (
        {"cron_expression": "not-a-cron"},
        {"timezone": "Mars/Olympus"},
        {"window_duration_minutes": 0},
        {"max_concurrency": 0},
    ):
        with pytest.raises(ValueError):
            ModelPreheatScheduleCreate(**(base | update))

    manual = ModelPreheatScheduleCreate(
        **(
            base
            | {
                "name": "manual",
                "trigger_mode": "manual",
                "cron_expression": None,
            }
        )
    )
    assert manual.trigger_mode.value == "manual"
    assert manual.cron_expression is None
    with pytest.raises(ValueError, match="cron_expression_required"):
        ModelPreheatScheduleCreate(
            **(base | {"cron_expression": None, "trigger_mode": "scheduled"})
        )


def test_model_preheat_feature_flag_defaults_off_and_accepts_explicit_enable(tmp_path):
    assert Config(data_dir=str(tmp_path)).model_preheat_enabled is False
    assert (
        Config(data_dir=str(tmp_path), model_preheat_enabled=True).model_preheat_enabled
        is True
    )


def test_schedule_accepts_optional_bandwidth_limit():
    from gpustack.schemas.model_preheat_schedules import ModelPreheatScheduleCreate

    schedule = ModelPreheatScheduleCreate(
        name="limited",
        cron_expression="0 1 * * *",
        timezone="UTC",
        window_duration_minutes=60,
        source="huggingface",
        model_id="org/model",
        s3_profile_id=1,
        target_worker_uuids=["worker-a"],
        bandwidth_limit_mbps=25,
    )
    assert schedule.bandwidth_limit_mbps == 25
    with pytest.raises(ValueError):
        ModelPreheatScheduleCreate(
            **(schedule.model_dump() | {"bandwidth_limit_mbps": 0})
        )


def test_schedule_schema_and_migration_are_portable():
    from gpustack.schemas.model_preheat_schedules import (
        ModelPreheatSchedule,
        ModelPreheatScheduleRun,
    )
    from gpustack.schemas.model_preheats import ModelPreheatTask

    assert ModelPreheatS3Profile.__tablename__ == "model_preheat_s3_profiles"

    migration = Path(
        "gpustack/migrations/versions/"
        "2026_08_12_1000-11caac4ba6d4_add_model_preheat_schedules.py"
    )
    source = migration.read_text(encoding="utf-8")
    assert 'down_revision: Union[str, None] = "f2a3b4c5d6e7"' in source
    assert "postgresql_where" not in source
    assert "advisory" not in source.lower()
    assert "CREATE UNIQUE INDEX" not in source
    unique_constraints = {
        constraint.name: [column.name for column in constraint.columns]
        for constraint in ModelPreheatScheduleRun.__table__.constraints
        if constraint.name
    }
    assert unique_constraints["uix_preheat_schedule_window"] == [
        "schedule_id",
        "window_start_utc",
        "operation_key",
    ]
    assert ModelPreheatScheduleRun.__table__.c.operation_key.nullable is False
    assert ModelPreheatSchedule.__table__.c.cron_expression.nullable is True
    assert ModelPreheatSchedule.__table__.c.trigger_mode.nullable is False
    assert "uix_preheat_schedule_run_operation" in unique_constraints
    assert "uix_preheat_schedule_slot" in unique_constraints
    task_schedule_fk = next(
        foreign_key
        for foreign_key in ModelPreheatTask.__table__.foreign_keys
        if foreign_key.parent.name == "schedule_id"
    )
    assert task_schedule_fk.target_fullname == "model_preheat_schedules.id"
    assert task_schedule_fk.ondelete == "SET NULL"
    for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
        for table in (
            ModelPreheatSchedule.__table__,
            ModelPreheatScheduleRun.__table__,
        ):
            ddl = str(CreateTable(table).compile(dialect=dialect))
            assert table.name in ddl
        run_ddl = " ".join(
            str(CreateTable(ModelPreheatScheduleRun.__table__).compile(dialect=dialect))
            .replace("`", "")
            .replace('"', "")
            .split()
        )
        assert "operation_key VARCHAR(64) NOT NULL" in run_ddl


def test_schedule_trigger_mode_migration_preserves_scheduled_and_allows_manual(
    tmp_path,
):
    migration_path = Path(
        "gpustack/migrations/versions/"
        "2026_08_23_1800-e9f0a1b2c3d4_add_schedule_trigger_mode.py"
    )
    spec = importlib.util.spec_from_file_location(
        "schedule_trigger_mode_migration", migration_path
    )
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine(f"sqlite:///{tmp_path / 'trigger-mode.db'}")

    with engine.connect() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE model_preheat_schedules ("
            "id INTEGER PRIMARY KEY, cron_expression VARCHAR(255) NOT NULL)"
        )
        connection.exec_driver_sql(
            "INSERT INTO model_preheat_schedules "
            "(id, cron_expression) VALUES (1, '0 1 * * *')"
        )
        context = MigrationContext.configure(connection)
        migration.op = Operations(context)
        migration.upgrade()

        columns = {
            column["name"]: column
            for column in inspect(connection).get_columns("model_preheat_schedules")
        }
        assert columns["cron_expression"]["nullable"] is True
        assert columns["trigger_mode"]["nullable"] is False
        assert (
            connection.exec_driver_sql(
                "SELECT trigger_mode FROM model_preheat_schedules WHERE id = 1"
            ).scalar_one()
            == "scheduled"
        )
        connection.exec_driver_sql(
            "INSERT INTO model_preheat_schedules "
            "(id, trigger_mode, cron_expression) VALUES (2, 'manual', NULL)"
        )

        migration.downgrade()
        columns = {
            column["name"]: column
            for column in inspect(connection).get_columns("model_preheat_schedules")
        }
        assert "trigger_mode" not in columns
        assert columns["cron_expression"]["nullable"] is False
        assert (
            connection.exec_driver_sql(
                "SELECT COUNT(*) FROM model_preheat_schedules "
                "WHERE cron_expression IS NULL"
            ).scalar_one()
            == 0
        )


def test_schedule_migration_executes_on_sqlite_and_compiles_for_server_dialects(
    tmp_path,
):
    old_migration_path = Path(
        "gpustack/migrations/versions/"
        "2026_08_12_1000-11caac4ba6d4_add_model_preheat_schedules.py"
    )
    successor_path = Path(
        "gpustack/migrations/versions/"
        "2026_08_12_1800-b0307846729c_fix_model_preheat_schedule_ddl.py"
    )

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        return migration

    old_migration = load("task13_old_migration", old_migration_path)
    successor = load("task13_successor_migration", successor_path)
    assert successor.down_revision == "11caac4ba6d4"

    engine = create_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        connection.commit()
        context = MigrationContext.configure(
            connection,
            opts={"transactional_ddl": True},
        )
        operations = Operations(context)
        old_migration.op = operations
        successor.op = operations
        with context.begin_transaction():
            connection.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            connection.exec_driver_sql(
                "CREATE TABLE model_preheat_s3_profiles (id INTEGER PRIMARY KEY)"
            )
            connection.exec_driver_sql(
                "CREATE TABLE model_preheat_tasks "
                "(id INTEGER PRIMARY KEY, schedule_id INTEGER)"
            )
            old_migration.upgrade()
            inspector = inspect(connection)
            assert "model_preheat_schedules" in inspector.get_table_names()
            assert "model_preheat_schedule_runs" in inspector.get_table_names()
            assert "bandwidth_limit_mbps" not in {
                column["name"]
                for column in inspector.get_columns("model_preheat_schedules")
            }
            assert "bandwidth_limit_mbps" not in {
                column["name"]
                for column in inspector.get_columns("model_preheat_tasks")
            }

            successor.upgrade()
            inspector = inspect(connection)
            assert "bandwidth_limit_mbps" in {
                column["name"]
                for column in inspector.get_columns("model_preheat_schedules")
            }
            assert "bandwidth_limit_mbps" in {
                column["name"]
                for column in inspector.get_columns("model_preheat_tasks")
            }
            task_fk = inspector.get_foreign_keys("model_preheat_tasks")[0]
            assert task_fk["referred_table"] == "model_preheat_schedules"
            assert task_fk["options"]["ondelete"] == "SET NULL"
            successor.downgrade()
            inspector = inspect(connection)
            assert "model_preheat_schedules" in inspector.get_table_names()
            assert "bandwidth_limit_mbps" not in {
                column["name"]
                for column in inspector.get_columns("model_preheat_schedules")
            }
            assert "bandwidth_limit_mbps" not in {
                column["name"]
                for column in inspector.get_columns("model_preheat_tasks")
            }
            assert not inspector.get_foreign_keys("model_preheat_tasks")
            old_migration.downgrade()
            assert (
                "model_preheat_schedules" not in inspect(connection).get_table_names()
            )

    for dialect_name in ("sqlite", "postgresql", "mysql"):
        output = io.StringIO()
        context = MigrationContext.configure(
            dialect_name=dialect_name,
            opts={"as_sql": True, "output_buffer": output},
        )
        successor.op = Operations(context)
        successor.upgrade()
        successor.downgrade()
        ddl = output.getvalue()
        assert "bandwidth_limit_mbps" in ddl
        assert "model_preheat_tasks" in ddl


def test_schedule_migration_preserves_sqlite_task_foreign_key_dependents(tmp_path):
    def load(name, filename):
        path = Path("gpustack/migrations/versions") / filename
        spec = importlib.util.spec_from_file_location(name, path)
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        return migration

    initial = load(
        "task13_fk_initial",
        "2026_08_12_1000-11caac4ba6d4_add_model_preheat_schedules.py",
    )
    successor = load(
        "task13_fk_successor",
        "2026_08_12_1800-b0307846729c_fix_model_preheat_schedule_ddl.py",
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'migration-fk-data.db'}")

    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        connection.commit()
        context = MigrationContext.configure(
            connection,
            opts={"transactional_ddl": True},
        )
        operations = Operations(context)
        initial.op = operations
        successor.op = operations

        with context.begin_transaction():
            connection.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY)")
            connection.exec_driver_sql(
                "CREATE TABLE model_preheat_s3_profiles (id INTEGER PRIMARY KEY)"
            )
            connection.exec_driver_sql(
                "CREATE TABLE model_preheat_tasks "
                "(id INTEGER PRIMARY KEY, schedule_id INTEGER)"
            )
            initial.upgrade()
            connection.exec_driver_sql(
                "CREATE TABLE model_preheat_worker_tasks ("
                "id INTEGER PRIMARY KEY, task_id INTEGER, payload TEXT, "
                "FOREIGN KEY(task_id) REFERENCES model_preheat_tasks(id) "
                "ON DELETE CASCADE)"
            )
            connection.exec_driver_sql(
                "CREATE TABLE model_preheat_distribution_policies ("
                "id INTEGER PRIMARY KEY, created_by_task_id INTEGER, payload TEXT, "
                "FOREIGN KEY(created_by_task_id) REFERENCES model_preheat_tasks(id) "
                "ON DELETE RESTRICT)"
            )
            connection.exec_driver_sql("INSERT INTO users (id) VALUES (1)")
            connection.exec_driver_sql(
                "INSERT INTO model_preheat_s3_profiles (id) VALUES (1)"
            )
            connection.exec_driver_sql(
                "INSERT INTO model_preheat_schedules ("
                "id, name, enabled, cron_expression, timezone, "
                "window_duration_minutes, max_concurrency, source, model_id, "
                "revision, include_patterns, exclude_patterns, target_scope, "
                "target_worker_uuids, seed_worker_uuid, s3_profile_id, "
                "s3_backfill_policy, keep_new_workers_in_sync, created_by_user_id, "
                "next_window_start_utc, last_window_start_utc, deleted_at, "
                "created_at, updated_at) VALUES ("
                "1, 'nightly', 1, '0 1 * * *', 'UTC', 60, 1, "
                "'huggingface', 'org/model', 'revision', '[]', '[]', "
                "'selected_workers', '[\"worker-a\"]', NULL, 1, "
                "'when_missing', 0, 1, NULL, NULL, NULL, "
                "'2026-08-12 00:00:00', '2026-08-12 00:00:00')"
            )
            connection.exec_driver_sql(
                "INSERT INTO model_preheat_tasks (id, schedule_id) VALUES (1, 1)"
            )
            connection.exec_driver_sql(
                "INSERT INTO model_preheat_worker_tasks "
                "(id, task_id, payload) VALUES (1, 1, 'cascade-row')"
            )
            connection.exec_driver_sql(
                "INSERT INTO model_preheat_schedule_runs ("
                "id, schedule_id, window_start_utc, window_end_utc, `trigger`, "
                "state, operation_key, slot, task_id, created_by_user_id, "
                "error_code, started_at, finished_at, deleted_at, created_at, "
                "updated_at) VALUES ("
                "1, 1, '2026-08-12 00:00:00', '2026-08-12 01:00:00', "
                "'scheduled', 'running', 'scheduled-operation', 0, 1, 1, "
                "NULL, '2026-08-12 00:00:00', NULL, NULL, "
                "'2026-08-12 00:00:00', '2026-08-12 00:00:00')"
            )
            connection.exec_driver_sql(
                "INSERT INTO model_preheat_distribution_policies "
                "(id, created_by_task_id, payload) VALUES (1, 1, 'restrict-row')"
            )
            successor.upgrade()

        def assert_references_preserved():
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
            assert connection.exec_driver_sql(
                "SELECT task_id, payload FROM model_preheat_worker_tasks"
            ).one() == (1, "cascade-row")
            assert (
                connection.exec_driver_sql(
                    "SELECT task_id FROM model_preheat_schedule_runs"
                ).scalar_one()
                == 1
            )
            assert connection.exec_driver_sql(
                "SELECT created_by_task_id, payload "
                "FROM model_preheat_distribution_policies"
            ).one() == (1, "restrict-row")
            assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []

        assert_references_preserved()
        connection.commit()
        with context.begin_transaction():
            successor.downgrade()
        assert_references_preserved()


def test_schedule_migration_restores_sqlite_foreign_keys_when_batch_fails(
    tmp_path, monkeypatch
):
    path = Path(
        "gpustack/migrations/versions/"
        "2026_08_12_1800-b0307846729c_fix_model_preheat_schedule_ddl.py"
    )
    spec = importlib.util.spec_from_file_location("task13_fk_failure", path)
    successor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(successor)
    engine = create_engine(f"sqlite:///{tmp_path / 'migration-fk-failure.db'}")

    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        connection.commit()
        context = MigrationContext.configure(
            connection,
            opts={"transactional_ddl": True},
        )
        operations = Operations(context)
        successor.op = operations

        def failing_batch(*args, **kwargs):
            del args, kwargs
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 0
            raise RuntimeError("injected_batch_failure")

        monkeypatch.setattr(operations, "batch_alter_table", failing_batch)
        with pytest.raises(RuntimeError, match="injected_batch_failure"):
            with context.begin_transaction():
                connection.exec_driver_sql(
                    "CREATE TABLE model_preheat_schedules " "(id INTEGER PRIMARY KEY)"
                )
                connection.exec_driver_sql(
                    "CREATE TABLE model_preheat_tasks "
                    "(id INTEGER PRIMARY KEY, schedule_id INTEGER)"
                )
                connection.exec_driver_sql(
                    "CREATE TABLE model_preheat_schedule_runs "
                    "(id INTEGER PRIMARY KEY)"
                )
                successor.upgrade()

        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
def test_schedule_migration_second_batch_failure_is_directly_retryable(
    tmp_path, direction
):
    initial, successor = _load_schedule_migrations(f"task13_{direction}_batch_retry")
    engine = create_engine(f"sqlite:///{tmp_path / f'{direction}-batch-retry.db'}")

    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        connection.commit()
        context = MigrationContext.configure(
            connection,
            opts={"transactional_ddl": True},
        )
        operations = Operations(context)
        initial.op = operations
        successor.op = operations
        with context.begin_transaction():
            _seed_schedule_migration_source(
                connection, initial, ["scheduled-operation"]
            )
        if direction == "downgrade":
            with context.begin_transaction():
                successor.upgrade()

        target_table = (
            "_alembic_tmp_model_preheat_schedule_runs"
            if direction == "upgrade"
            else "_alembic_tmp_model_preheat_tasks"
        )
        failure_injected = False

        def fail_second_batch_insert(
            conn, cursor, statement, parameters, context, executemany
        ):
            del conn, cursor, parameters, context, executemany
            nonlocal failure_injected
            normalized = statement.lower().replace('"', "").replace("`", "")
            if f"insert into {target_table}" in normalized:
                failure_injected = True
                raise RuntimeError(f"injected_{direction}_second_batch_failure")

        sa.event.listen(connection, "before_cursor_execute", fail_second_batch_insert)
        try:
            with pytest.raises(
                RuntimeError, match=f"injected_{direction}_second_batch_failure"
            ):
                with context.begin_transaction():
                    getattr(successor, direction)()
        finally:
            sa.event.remove(
                connection, "before_cursor_execute", fail_second_batch_insert
            )

        assert failure_injected
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        _assert_no_alembic_temp_tables(connection)
        _assert_schedule_migration_data_preserved(connection)
        assert _schedule_migration_state(
            connection
        ) == _expected_schedule_migration_state(upgraded=direction == "downgrade")

        with context.begin_transaction():
            getattr(successor, direction)()

        expected_upgraded = direction == "upgrade"
        assert _schedule_migration_state(
            connection
        ) == _expected_schedule_migration_state(expected_upgraded)
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        _assert_no_alembic_temp_tables(connection)
        _assert_schedule_migration_data_preserved(connection)


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
def test_schedule_migration_foreign_key_check_failure_is_directly_retryable(
    tmp_path, monkeypatch, direction
):
    initial, successor = _load_schedule_migrations(f"task13_{direction}_fk_retry")
    engine = create_engine(f"sqlite:///{tmp_path / f'{direction}-fk-retry.db'}")

    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        connection.commit()
        context = MigrationContext.configure(
            connection,
            opts={"transactional_ddl": True},
        )
        operations = Operations(context)
        initial.op = operations
        successor.op = operations
        with context.begin_transaction():
            _seed_schedule_migration_source(
                connection, initial, ["scheduled-operation"]
            )
        if direction == "downgrade":
            with context.begin_transaction():
                successor.upgrade()

        step_name = f"_{direction}_sqlite"
        migration_step = getattr(successor, step_name)

        def inject_orphan_before_check():
            migration_step()
            connection.exec_driver_sql(
                "INSERT INTO model_preheat_worker_tasks "
                "(id, task_id, payload) VALUES (2, 999, 'injected-orphan')"
            )

        monkeypatch.setattr(successor, step_name, inject_orphan_before_check)
        with pytest.raises(RuntimeError, match="sqlite_foreign_key_check_failed"):
            with context.begin_transaction():
                getattr(successor, direction)()
        monkeypatch.setattr(successor, step_name, migration_step)

        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        _assert_no_alembic_temp_tables(connection)
        _assert_schedule_migration_data_preserved(connection)
        assert _schedule_migration_state(
            connection
        ) == _expected_schedule_migration_state(upgraded=direction == "downgrade")

        with context.begin_transaction():
            getattr(successor, direction)()

        expected_upgraded = direction == "upgrade"
        assert _schedule_migration_state(
            connection
        ) == _expected_schedule_migration_state(expected_upgraded)
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        _assert_no_alembic_temp_tables(connection)
        _assert_schedule_migration_data_preserved(connection)


def test_schedule_migration_backfills_and_rejects_null_operation_keys(tmp_path):
    initial, successor = _load_schedule_migrations("task13_operation_key_not_null")
    engine = create_engine(f"sqlite:///{tmp_path / 'operation-key-not-null.db'}")

    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        connection.commit()
        context = MigrationContext.configure(
            connection,
            opts={"transactional_ddl": True},
        )
        operations = Operations(context)
        initial.op = operations
        successor.op = operations
        with context.begin_transaction():
            _seed_schedule_migration_source(connection, initial, [None, None])
            successor.upgrade()

        operation_keys = (
            connection.exec_driver_sql(
                "SELECT operation_key FROM model_preheat_schedule_runs ORDER BY id"
            )
            .scalars()
            .all()
        )
        assert len(operation_keys) == 2
        assert all(operation_keys)
        assert len(set(operation_keys)) == 2
        assert all(len(operation_key) <= 64 for operation_key in operation_keys)
        assert _schedule_migration_state(connection)["operation_key_nullable"] is False

        with pytest.raises(sa.exc.IntegrityError):
            connection.exec_driver_sql(
                "INSERT INTO model_preheat_schedule_runs ("
                "id, schedule_id, window_start_utc, window_end_utc, `trigger`, "
                "state, operation_key, slot, task_id, created_by_user_id, "
                "error_code, started_at, finished_at, deleted_at, created_at, "
                "updated_at) VALUES ("
                "3, 1, '2026-08-12 02:00:00', '2026-08-12 02:30:00', "
                "'manual', 'running', NULL, NULL, 1, 1, NULL, "
                "'2026-08-12 02:00:00', NULL, NULL, "
                "'2026-08-12 02:00:00', '2026-08-12 02:00:00')"
            )
        connection.rollback()
        _assert_schedule_migration_data_preserved(connection, expected_run_count=2)


def test_schedule_migration_joins_existing_sqlite_transaction_with_foreign_keys_off(
    tmp_path,
):
    initial, successor = _load_schedule_migrations("task13_existing_transaction")
    engine = create_engine(f"sqlite:///{tmp_path / 'existing-transaction.db'}")

    with engine.connect() as connection:
        context = MigrationContext.configure(
            connection,
            opts={"transactional_ddl": True},
        )
        operations = Operations(context)
        initial.op = operations
        successor.op = operations

        with context.begin_transaction():
            _seed_schedule_migration_source(connection, initial, [None])
            assert connection.connection.driver_connection.in_transaction
            successor.upgrade()

        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 0
        assert _schedule_migration_state(
            connection
        ) == _expected_schedule_migration_state(upgraded=True)
        operation_key = connection.exec_driver_sql(
            "SELECT operation_key FROM model_preheat_schedule_runs"
        ).scalar_one()
        assert operation_key
        _assert_no_alembic_temp_tables(connection)
        _assert_schedule_migration_data_preserved(connection)


def test_sqlite_offline_migration_preserves_existing_task_defaults(tmp_path):
    successor_path = Path(
        "gpustack/migrations/versions/"
        "2026_08_12_1800-b0307846729c_fix_model_preheat_schedule_ddl.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task13_offline_defaults", successor_path
    )
    successor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(successor)
    database_path = tmp_path / "offline-defaults.db"
    engine = create_engine(f"sqlite:///{database_path}")
    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        connection.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY)")
        connection.exec_driver_sql(
            "CREATE TABLE model_preheat_s3_profiles (id INTEGER PRIMARY KEY)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE model_preheat_schedules " "(id INTEGER PRIMARY KEY)"
        )
        old_table = successor._model_preheat_tasks_table(False)
        sa.Table(
            "users",
            old_table.metadata,
            sa.Column("id", sa.Integer(), primary_key=True),
        )
        sa.Table(
            "model_preheat_s3_profiles",
            old_table.metadata,
            sa.Column("id", sa.Integer(), primary_key=True),
        )
        old_table.c.desired_state.server_default = sa.DefaultClause("running")
        old_table.c.execution_state.server_default = sa.DefaultClause("pending")
        old_table.c.keep_new_workers_in_sync.server_default = sa.DefaultClause(
            sa.false()
        )
        old_table.create(connection)
        old_run_table = successor._model_preheat_schedule_runs_table(False)
        for table_name in (
            "users",
            "model_preheat_schedules",
            "model_preheat_tasks",
        ):
            sa.Table(
                table_name,
                old_run_table.metadata,
                sa.Column("id", sa.Integer(), primary_key=True),
            )
        old_run_table.create(connection)

    def offline_sql(operation):
        output = io.StringIO()
        context = MigrationContext.configure(
            dialect_name="sqlite",
            opts={"as_sql": True, "output_buffer": output},
        )
        successor.op = Operations(context)
        operation()
        return output.getvalue()

    def defaults():
        with sqlite3.connect(database_path) as connection:
            return {
                row[1]: row[4]
                for row in connection.execute("PRAGMA table_info(model_preheat_tasks)")
            }

    expected = {
        "desired_state": "'running'",
        "execution_state": "'pending'",
        "keep_new_workers_in_sync": "0",
    }
    assert {key: defaults()[key] for key in expected} == expected

    with sqlite3.connect(database_path) as connection:
        connection.executescript(offline_sql(successor.upgrade))
    assert {key: defaults()[key] for key in expected} == expected

    with sqlite3.connect(database_path) as connection:
        connection.executescript(offline_sql(successor.downgrade))
    assert {key: defaults()[key] for key in expected} == expected


async def _create_tables(engine):
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)


async def _drop_tables(engine):
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.drop_all)


def _test_app(tmp_path):
    from gpustack.routes import model_preheat_schedules
    from gpustack.server.model_preheat_schedule_controller import (
        ModelPreheatScheduleController,
    )

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'schedules.db'}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
    asyncio.run(_create_tables(engine))

    created_tasks = 0

    async def create_task(session, schedule, created_by_user_id):
        nonlocal created_tasks
        created_tasks += 1
        task = ModelPreheatTask(
            source=schedule.source,
            model_id=schedule.model_id,
            requested_revision=schedule.revision,
            resolved_revision=schedule.revision or "test-revision",
            include_patterns=schedule.include_patterns,
            exclude_patterns=schedule.exclude_patterns,
            selection_digest="test-selection-digest",
            request_identity={
                "source": schedule.source,
                "model_id": schedule.model_id,
                "requested_revision": schedule.revision,
                "include_patterns": schedule.include_patterns,
                "exclude_patterns": schedule.exclude_patterns,
            },
            request_digest=f"{created_tasks:064d}",
            target_scope=schedule.target_scope,
            target_worker_uuids=schedule.target_worker_uuids,
            target_worker_snapshot=[],
            s3_profile_id=schedule.s3_profile_id,
            s3_profile_config_version=1,
            s3_profile_snapshot_encrypted={"ciphertext": "encrypted"},
            encryption_key_version="v1",
            s3_backfill_policy=schedule.s3_backfill_policy,
            schedule_id=schedule.id,
            created_by_user_id=created_by_user_id,
        )
        session.add(task)
        await session.flush()
        return task

    app = FastAPI()
    app.state.model_preheat_schedule_controller = ModelPreheatScheduleController(
        engine,
        task_creator=create_task,
    )

    async def session_override():
        async with AsyncSession(engine) as session:
            yield session

    async def admin_user_override():
        return User(id=1, username="admin", is_admin=True, hashed_password="")

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_admin_user] = admin_user_override
    admin_router = APIRouter(dependencies=[Depends(get_admin_user)])
    admin_router.include_router(
        model_preheat_schedules.router,
        prefix="/model-preheat-schedules",
    )
    app.include_router(admin_router, prefix="/v1")
    exceptions.register_handlers(app)
    return app, engine


async def _seed_profile(engine):
    async with AsyncSession(engine) as session:
        profile = ModelPreheatS3Profile(
            name="schedule-profile",
            endpoint="https://s3.example.com",
            bucket="models",
            access_key_encrypted={"ciphertext": "must-not-leak"},
            secret_key_encrypted={"ciphertext": "must-not-leak"},
            encryption_key_version="v1",
        )
        session.add(profile)
        await session.commit()
        await session.refresh(profile)
        return profile.id


def _payload(profile_id, **overrides):
    payload = {
        "name": "nightly",
        "cron_expression": "0 1 * * *",
        "timezone": "Asia/Shanghai",
        "window_duration_minutes": 120,
        "max_concurrency": 1,
        "source": "modelscope",
        "model_id": "Qwen/Test",
        "revision": "commit-123",
        "include_patterns": ["config.json"],
        "exclude_patterns": [],
        "target_scope": "selected_workers",
        "target_worker_uuids": ["worker-a"],
        "s3_profile_id": profile_id,
        "s3_backfill_policy": "when_missing",
    }
    payload.update(overrides)
    return payload


def test_schedule_create_rejects_maintenance_profile(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id = asyncio.run(_seed_profile(engine))

    async def maintain():
        async with AsyncSession(engine) as session:
            profile = await session.get(ModelPreheatS3Profile, profile_id)
            profile.lifecycle_state = (
                ModelPreheatS3ProfileLifecycleStateEnum.MAINTENANCE
            )
            session.add(profile)
            await session.commit()

    asyncio.run(maintain())
    with TestClient(app) as client:
        response = client.post("/v1/model-preheat-schedules", json=_payload(profile_id))

    assert response.status_code == 409
    assert response.json()["message"] == "model_preheat_s3_profile_in_maintenance"
    asyncio.run(engine.dispose())


def test_schedule_admin_crud_validates_profile_and_hides_internal_keys(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id = asyncio.run(_seed_profile(engine))

    with TestClient(app) as client:
        missing_profile = client.post(
            "/v1/model-preheat-schedules", json=_payload(profile_id + 999)
        )
        created = client.post("/v1/model-preheat-schedules", json=_payload(profile_id))
        schedule_id = created.json()["id"]
        listed = client.get("/v1/model-preheat-schedules")
        fetched = client.get(f"/v1/model-preheat-schedules/{schedule_id}")
        renamed = client.patch(
            f"/v1/model-preheat-schedules/{schedule_id}",
            json={"name": "nightly-renamed", "window_duration_minutes": 150},
        )
        updated = client.patch(
            f"/v1/model-preheat-schedules/{schedule_id}",
            json={"enabled": False, "window_duration_minutes": 180},
        )
        deleted = client.delete(f"/v1/model-preheat-schedules/{schedule_id}")
        missing = client.get(f"/v1/model-preheat-schedules/{schedule_id}")

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert missing_profile.status_code == 404
    assert created.status_code == 200, created.text
    assert listed.status_code == 200, listed.text
    assert listed.json()["pagination"]["total"] == 1
    assert fetched.json()["name"] == "nightly"
    assert renamed.json()["name"] == "nightly-renamed"
    assert (
        renamed.json()["next_window_start_utc"]
        == created.json()["next_window_start_utc"]
    )
    assert updated.json()["enabled"] is False
    assert updated.json()["window_duration_minutes"] == 180
    assert updated.json()["next_window_start_utc"] is None
    assert "operation_key" not in str(created.json())
    assert "must-not-leak" not in str(created.json())
    assert deleted.status_code == 200
    assert missing.status_code == 404


def test_manual_schedule_can_be_saved_without_cron_and_run_now(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id = asyncio.run(_seed_profile(engine))

    with TestClient(app) as client:
        created = client.post(
            "/v1/model-preheat-schedules",
            json=_payload(
                profile_id,
                name="manual-policy",
                trigger_mode="manual",
                cron_expression=None,
            ),
        )
        missing_cron = client.patch(
            f"/v1/model-preheat-schedules/{created.json()['id']}",
            json={"trigger_mode": "scheduled"},
        )
        scheduled = client.patch(
            f"/v1/model-preheat-schedules/{created.json()['id']}",
            json={"trigger_mode": "scheduled", "cron_expression": "0 2 * * *"},
        )
        manual_again = client.patch(
            f"/v1/model-preheat-schedules/{created.json()['id']}",
            json={"trigger_mode": "manual"},
        )
        run = client.post(
            f"/v1/model-preheat-schedules/{created.json()['id']}/run-now",
            headers={"Idempotency-Key": "manual-run"},
        )

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert created.status_code == 200, created.text
    assert created.json()["trigger_mode"] == "manual"
    assert created.json()["cron_expression"] is None
    assert created.json()["next_window_start_utc"] is None
    assert missing_cron.status_code == 422
    assert scheduled.status_code == 200, scheduled.text
    assert scheduled.json()["next_window_start_utc"] is not None
    assert manual_again.status_code == 200, manual_again.text
    assert manual_again.json()["cron_expression"] is None
    assert manual_again.json()["next_window_start_utc"] is None
    assert run.status_code == 200, run.text
    assert run.json()["trigger"] == "manual"


def test_schedule_patterns_survive_create_unrelated_patch_and_worker_matching(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id = asyncio.run(_seed_profile(engine))
    include_patterns = [
        "weights/my model?.safetensors",
        "配置/[0-9].json",
        "dir with space/*.txt",
    ]
    exclude_patterns = ["配置/[5-9].json"]

    with TestClient(app) as client:
        created = client.post(
            "/v1/model-preheat-schedules",
            json=_payload(
                profile_id,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            ),
        )
        schedule_id = created.json()["id"]
        patched = client.patch(
            f"/v1/model-preheat-schedules/{schedule_id}",
            json={"name": "nightly-renamed"},
        )

    expected_include = [
        "配置/[0-9].json",
        "dir with space/*.txt",
        "weights/my model?.safetensors",
    ]
    assert created.status_code == 200, created.text
    assert created.json()["include_patterns"] == expected_include
    assert created.json()["exclude_patterns"] == exclude_patterns
    assert patched.status_code == 200, patched.text
    assert patched.json()["include_patterns"] == expected_include
    assert patched.json()["exclude_patterns"] == exclude_patterns

    task_input = ModelPreheatCreate(
        source=patched.json()["source"],
        model_id=patched.json()["model_id"],
        revision=patched.json()["revision"],
        include_patterns=patched.json()["include_patterns"],
        exclude_patterns=patched.json()["exclude_patterns"],
        target_worker_ids=[1],
        s3_profile_id=profile_id,
    )
    assert task_input.include_patterns == [
        "配置/[0-9].json",
        "dir with space/*.txt",
        "weights/my model?.safetensors",
    ]
    assert task_input.exclude_patterns == ["配置/[5-9].json"]
    assert _preheat_file_selected(
        "weights/my model1.safetensors",
        task_input.include_patterns,
        task_input.exclude_patterns,
    )
    assert _preheat_file_selected(
        "配置/3.json", task_input.include_patterns, task_input.exclude_patterns
    )
    assert not _preheat_file_selected(
        "配置/7.json", task_input.include_patterns, task_input.exclude_patterns
    )
    assert _preheat_file_selected(
        "dir with space/readme.txt",
        task_input.include_patterns,
        task_input.exclude_patterns,
    )

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())


def test_run_now_is_idempotent_and_active_run_blocks_delete(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id = asyncio.run(_seed_profile(engine))

    with TestClient(app) as client:
        first_schedule = client.post(
            "/v1/model-preheat-schedules", json=_payload(profile_id)
        ).json()
        second_schedule = client.post(
            "/v1/model-preheat-schedules",
            json=_payload(profile_id, name="nightly-2"),
        ).json()
        first = client.post(
            f"/v1/model-preheat-schedules/{first_schedule['id']}/run-now",
            headers={"Idempotency-Key": "run-key"},
        )
        replay = client.post(
            f"/v1/model-preheat-schedules/{first_schedule['id']}/run-now",
            headers={"Idempotency-Key": "run-key"},
        )
        conflict = client.post(
            f"/v1/model-preheat-schedules/{second_schedule['id']}/run-now",
            headers={"Idempotency-Key": "run-key"},
        )
        delete_active = client.delete(
            f"/v1/model-preheat-schedules/{first_schedule['id']}"
        )

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert first.status_code == 200, first.text
    assert replay.status_code == 200, replay.text
    assert replay.json()["id"] == first.json()["id"]
    assert replay.json()["task_id"] == first.json()["task_id"]
    assert "operation_key" not in first.json()
    assert conflict.status_code == 409
    assert conflict.json()["reason"] == "idempotency_key_reused"
    assert delete_active.status_code == 409


def test_schedule_run_list_and_get_are_admin_queryable(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id = asyncio.run(_seed_profile(engine))

    with TestClient(app) as client:
        schedule = client.post(
            "/v1/model-preheat-schedules", json=_payload(profile_id)
        ).json()
        run = client.post(
            f"/v1/model-preheat-schedules/{schedule['id']}/run-now",
            headers={"Idempotency-Key": "query-run"},
        ).json()
        listed = client.get(f"/v1/model-preheat-schedules/{schedule['id']}/runs")
        fetched = client.get(
            f"/v1/model-preheat-schedules/{schedule['id']}/runs/{run['id']}"
        )

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert listed.status_code == 200, listed.text
    assert listed.json()["pagination"]["total"] == 1
    assert listed.json()["items"][0]["id"] == run["id"]
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["id"] == run["id"]
    assert "operation_key" not in str(listed.json())
