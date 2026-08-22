from alembic import command
from alembic.config import Config
import sqlalchemy as sa

from gpustack.migrations.validate import validate_revision_graph


def test_alembic_revision_graph_has_one_resolvable_head():
    # 任务 4 固定 Profile 引用保护后，唯一 head 前移到引用保护 revision。
    assert validate_revision_graph() == "a5b6c7d8e9f0"


def test_alembic_upgrades_existing_model_cache_head_to_schedule_head(tmp_path):
    database_path = tmp_path / "migration.db"
    engine = sa.create_engine(f"sqlite:///{database_path}")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY)")
        connection.exec_driver_sql(
            "CREATE TABLE model_preheat_s3_profiles (id INTEGER PRIMARY KEY)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE model_preheat_tasks "
            "(id INTEGER PRIMARY KEY, schedule_id INTEGER)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE alembic_version "
            "(version_num VARCHAR(32) NOT NULL PRIMARY KEY)"
        )
        connection.exec_driver_sql(
            "INSERT INTO alembic_version (version_num) VALUES (?)",
            ("f2a3b4c5d6e7",),
        )

    config = Config()
    config.set_main_option("script_location", "gpustack/migrations")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    # 只验证到 schedule head；收敛 migration c1d2e3f4a5b6 的完整前置
    # 与 DDL 由 tests/migrations/test_model_storage_migration.py 单独覆盖。
    command.upgrade(config, "b0307846729c")

    with engine.connect() as connection:
        assert (
            connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one()
            == "b0307846729c"
        )
        tables = set(sa.inspect(connection).get_table_names())
        assert "model_preheat_schedules" in tables
        assert "model_preheat_schedule_runs" in tables
