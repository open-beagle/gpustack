"""任务 3 子阶段 B：活动同步任务数据库级去重槽 migration 定向测试。

覆盖：
- 新增 head ``d2e3f4a5b6c7``（真实 Alembic head 为收敛 migration
  ``c1d2e3f4a5b6`` 之后一级）；
- ``model_storage_sync_task_dedupe_slots`` 表结构与 SQLModel 一致，
  ``dedupe_key`` 唯一约束在三库（SQLite/PostgreSQL/MySQL）DDL 上均可渲染；
- 从收敛 head 升级到新 head 建表、降级回收敛 head 删表；
- 唯一约束数据库级生效（SQLite 实库插入重复键失败）；
- 任务 + 槽位同事务提交/回滚原子性：槽位唯一冲突时任务一并回滚，不留遗留。
"""

import importlib.util
from pathlib import Path

import sqlalchemy as sa
import pytest
from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable

from gpustack.migrations.validate import validate_revision_graph

UNIFY_REVISION = "c1d2e3f4a5b6"
DEDUPE_REVISION = "d2e3f4a5b6c7"
LEASE_REVISION = "e3f4a5b6c7d8"

MIGRATION_ROOT = (
    Path(__file__).resolve().parents[2] / "gpustack/migrations/versions"
)


def _alembic_config(tmp_path: Path, name: str = "dedupe.db") -> Config:
    database_path = tmp_path / name
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "gpustack/migrations"),
    )
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    return config


def _upgrade_from_unify_head(tmp_path: Path):
    """从收敛 head ``c1d2e3f4a5b6``（表已存在）升级到新 head。"""
    config = _alembic_config(tmp_path)
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    with engine.begin() as connection:
        # 收敛 head 状态的最小前置：同步任务表（dedupe 表外键目标）。
        connection.exec_driver_sql(
            "CREATE TABLE model_storage_sync_tasks ("
            " id INTEGER PRIMARY KEY, model_file_id INTEGER NOT NULL, "
            " worker_id INTEGER NOT NULL, worker_uuid VARCHAR(255) NOT NULL, "
            " profile_id INTEGER NOT NULL, profile_config_version INTEGER NOT NULL, "
            " request_identity JSON NOT NULL, request_digest VARCHAR(64) NOT NULL, "
            " source VARCHAR(32) NOT NULL, model_id VARCHAR(1024) NOT NULL, "
            " resolved_revision VARCHAR(1024) NOT NULL, "
            " credential_snapshot_encrypted JSON NOT NULL, "
            " encryption_key_version VARCHAR(255) NOT NULL, "
            " artifact_id VARCHAR(64), state VARCHAR(32) NOT NULL DEFAULT 'pending', "
            " state_message TEXT, error_code VARCHAR(64), file_count INTEGER NOT NULL, "
            " total_size BIGINT NOT NULL, transfer_source VARCHAR(32), "
            " transfer_profile_id INTEGER, source_worker_id INTEGER, "
            " created_by_user_id INTEGER, started_at DATETIME, finished_at DATETIME, "
            " created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, "
            " deleted_at DATETIME)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)"
        )
        connection.exec_driver_sql(
            "INSERT INTO alembic_version (version_num) VALUES (?)",
            (UNIFY_REVISION,),
        )
    command.upgrade(config, "head")
    return config, engine


def test_new_head_is_dedupe_revision():
    # 任务 3 定向复审后追加 lease migration：唯一 head 前移到 lease revision。
    assert validate_revision_graph() == LEASE_REVISION


def test_lease_revision_chains_from_dedupe_revision():
    """lease migration 紧跟 dedupe head，不产生多 head、不改历史。"""
    scripts = _load_scripts()
    rev = scripts.get_revision(LEASE_REVISION)
    assert rev is not None
    assert rev.down_revision == DEDUPE_REVISION


def _load_scripts():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "gpustack/migrations"),
    )
    return ScriptDirectory.from_config(config)


def _load_lease_migration_module():
    path = (
        MIGRATION_ROOT
        / f"2026_08_22_1000-{LEASE_REVISION}_add_model_storage_sync_lease.py"
    )
    spec = importlib.util.spec_from_file_location("_sync_lease", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "dialect",
    [sqlite.dialect(), postgresql.dialect(), mysql.dialect()],
    ids=["sqlite", "postgresql", "mysql"],
)
def test_lease_migration_ddl_renders_on_all_supported_dialects(dialect):
    """lease migration（3 个可空列）在三库 DDL 上均可渲染。"""
    from alembic import op

    import io

    module = _load_lease_migration_module()
    buffer = io.StringIO()
    context = MigrationContext.configure(
        _DialectConnection(dialect),
        opts={"as_sql": True, "output_buffer": buffer},
    )
    with op.Operations.context(context):
        module.upgrade()
    sql = buffer.getvalue()
    assert sql.strip(), f"upgrade DDL rendered empty for {dialect.name}"
    for column in (
        "lease_token_encrypted",
        "manifest_digest",
        "manifest_path",
    ):
        assert column in sql
    assert "model_storage_sync_tasks" in sql


def test_upgrade_to_lease_head_adds_columns_and_is_idempotent_ish(tmp_path):
    """从 dedupe head 升级到 lease head：新增三列，且三列均可空（历史数据兼容）。"""
    config = _alembic_config(tmp_path, name="lease.db")
    database_path = tmp_path / "lease.db"
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    # 预置 dedupe head 的最小 tasks 表。
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE model_storage_sync_tasks ("
            " id INTEGER PRIMARY KEY, model_file_id INTEGER NOT NULL, "
            " worker_id INTEGER NOT NULL, worker_uuid VARCHAR(255) NOT NULL, "
            " profile_id INTEGER NOT NULL, profile_config_version INTEGER NOT NULL, "
            " request_identity JSON NOT NULL, request_digest VARCHAR(64) NOT NULL, "
            " source VARCHAR(32) NOT NULL, model_id VARCHAR(1024) NOT NULL, "
            " resolved_revision VARCHAR(1024) NOT NULL, "
            " credential_snapshot_encrypted JSON NOT NULL, "
            " encryption_key_version VARCHAR(255) NOT NULL, "
            " artifact_id VARCHAR(64), state VARCHAR(32) NOT NULL DEFAULT 'pending', "
            " state_message TEXT, error_code VARCHAR(64), file_count INTEGER NOT NULL, "
            " total_size BIGINT NOT NULL, transfer_source VARCHAR(32), "
            " transfer_profile_id INTEGER, source_worker_id INTEGER, "
            " created_by_user_id INTEGER, started_at DATETIME, finished_at DATETIME, "
            " created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, "
            " deleted_at DATETIME)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)"
        )
        connection.exec_driver_sql(
            "INSERT INTO alembic_version (version_num) VALUES (?)",
            (DEDUPE_REVISION,),
        )
    command.upgrade(config, "head")
    insp = sa.inspect(engine)
    columns = {
        c["name"]: c
        for c in insp.get_columns("model_storage_sync_tasks")
    }
    for column in (
        "lease_token_encrypted",
        "manifest_digest",
        "manifest_path",
    ):
        assert column in columns
        # 可空：历史任务（三列 NULL）兼容，不破坏既有数据。
        assert columns[column]["nullable"] is True
    with engine.connect() as connection:
        assert (
            connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one()
            == LEASE_REVISION
        )
    engine.dispose()


def test_upgrade_from_unify_head_creates_slot_table(tmp_path):
    _, engine = _upgrade_from_unify_head(tmp_path)
    insp = sa.inspect(engine)
    assert "model_storage_sync_task_dedupe_slots" in insp.get_table_names()
    columns = {c["name"] for c in insp.get_columns("model_storage_sync_task_dedupe_slots")}
    assert {"dedupe_key", "task_id", "created_at", "updated_at", "deleted_at"} <= columns
    unique = {
        frozenset(c["column_names"])
        for c in insp.get_unique_constraints("model_storage_sync_task_dedupe_slots")
    }
    # 唯一约束也体现在索引中（SQLite 实现路径）。
    index_columns = {
        frozenset(ix["column_names"])
        for ix in insp.get_indexes("model_storage_sync_task_dedupe_slots")
        if ix["unique"]
    }
    assert frozenset(["dedupe_key"]) in (unique | index_columns)
    with engine.connect() as connection:
        assert (
            connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one()
            == LEASE_REVISION
        )


def test_dedupe_key_unique_constraint_enforced_at_database_level(tmp_path):
    """SQLite 实库：重复 dedupe_key 插入必须失败（数据库级保证）。"""
    _, engine = _upgrade_from_unify_head(tmp_path)
    with engine.begin() as connection:
        now = "2026-08-20 00:00:00"
        connection.exec_driver_sql(
            "INSERT INTO model_storage_sync_task_dedupe_slots "
            "(dedupe_key, task_id, created_at, updated_at) VALUES (?, 1, ?, ?)",
            ("msync:1:1", now, now),
        )
        with pytest.raises(IntegrityError):
            connection.exec_driver_sql(
                "INSERT INTO model_storage_sync_task_dedupe_slots "
                "(dedupe_key, task_id, created_at, updated_at) VALUES (?, 2, ?, ?)",
                ("msync:1:1", now, now),
            )


def test_downgrade_to_unify_head_drops_slot_table(tmp_path):
    config, engine = _upgrade_from_unify_head(tmp_path)
    command.downgrade(config, UNIFY_REVISION)
    with engine.connect() as connection:
        tables = set(sa.inspect(connection).get_table_names())
        assert "model_storage_sync_task_dedupe_slots" not in tables
        assert (
            connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one()
            == UNIFY_REVISION
        )


def _load_dedupe_migration_module():
    path = (
        MIGRATION_ROOT
        / f"2026_08_20_1100-{DEDUPE_REVISION}_add_model_storage_sync_dedupe_slots.py"
    )
    spec = importlib.util.spec_from_file_location("_dedupe_slots", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DialectConnection:
    """暴露 dialect 的最小连接替身，供 Alembic 离线 DDL 渲染。"""

    def __init__(self, dialect):
        self.dialect = dialect


@pytest.mark.parametrize(
    "dialect",
    [sqlite.dialect(), postgresql.dialect(), mysql.dialect()],
    ids=["sqlite", "postgresql", "mysql"],
)
def test_dedupe_migration_ddl_renders_on_all_supported_dialects(dialect):
    from alembic import op

    import io

    module = _load_dedupe_migration_module()
    buffer = io.StringIO()
    context = MigrationContext.configure(
        _DialectConnection(dialect),
        opts={"as_sql": True, "output_buffer": buffer},
    )
    with op.Operations.context(context):
        module.upgrade()
    sql = buffer.getvalue()
    assert sql.strip(), f"upgrade DDL rendered empty for {dialect.name}"
    assert "model_storage_sync_task_dedupe_slots" in sql
    assert "dedupe_key" in sql
    assert "uix_model_storage_sync_dedupe_key" in sql


def test_slot_table_schema_compiles_on_all_supported_dialects():
    from gpustack.schemas import model_storage_sync  # noqa: F401

    table = model_storage_sync.ModelStorageSyncTaskDedupeSlot.__table__
    for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
        compiled = str(CreateTable(table).compile(dialect=dialect))
        assert "model_storage_sync_task_dedupe_slots" in compiled


def test_task_and_slot_commit_is_atomic_on_unique_conflict(tmp_path):
    """回滚语义：任务 flush 后，槽位唯一冲突必须让任务一并回滚。

    模拟并发竞争在数据库层：事务内先 flush 任务 T1，再插入与既有已提交
    槽位同键的槽位（触发唯一冲突）；回滚后 T1 不得残留（不留遗留任务）。
    """
    config = _alembic_config(tmp_path)
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE model_storage_sync_tasks ("
            " id INTEGER PRIMARY KEY, model_file_id INTEGER NOT NULL, "
            " worker_id INTEGER NOT NULL, worker_uuid VARCHAR(255) NOT NULL, "
            " profile_id INTEGER NOT NULL, profile_config_version INTEGER NOT NULL, "
            " request_identity JSON NOT NULL, request_digest VARCHAR(64) NOT NULL, "
            " source VARCHAR(32) NOT NULL, model_id VARCHAR(1024) NOT NULL, "
            " resolved_revision VARCHAR(1024) NOT NULL, "
            " credential_snapshot_encrypted JSON NOT NULL, "
            " encryption_key_version VARCHAR(255) NOT NULL, "
            " artifact_id VARCHAR(64), state VARCHAR(32) NOT NULL DEFAULT 'pending', "
            " state_message TEXT, error_code VARCHAR(64), file_count INTEGER NOT NULL, "
            " total_size BIGINT NOT NULL, transfer_source VARCHAR(32), "
            " transfer_profile_id INTEGER, source_worker_id INTEGER, "
            " created_by_user_id INTEGER, started_at DATETIME, finished_at DATETIME, "
            " created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, "
            " deleted_at DATETIME)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE model_storage_sync_task_dedupe_slots ("
            " id INTEGER PRIMARY KEY, dedupe_key VARCHAR(64) NOT NULL, "
            " task_id INTEGER, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, "
            " deleted_at DATETIME, "
            " CONSTRAINT uix_model_storage_sync_dedupe_key UNIQUE (dedupe_key))"
        )

    # 既有已提交的槽位（先创建者）。
    with engine.begin() as connection:
        now = "2026-08-20 00:00:00"
        connection.exec_driver_sql(
            "INSERT INTO model_storage_sync_task_dedupe_slots "
            "(dedupe_key, task_id, created_at, updated_at) VALUES ('msync:1:1', 1, ?, ?)",
            (now, now),
        )

    # 用同步引擎 + 原生连接事务模拟同一原子事务（ORM 与原生同语义）。
    task_inserted = False
    with engine.connect() as connection:
        try:
            connection.exec_driver_sql(
                "INSERT INTO model_storage_sync_tasks (id, model_file_id, worker_id, "
                "worker_uuid, profile_id, profile_config_version, request_identity, "
                "request_digest, source, model_id, resolved_revision, "
                "credential_snapshot_encrypted, encryption_key_version, state, "
                "file_count, total_size, created_at, updated_at) VALUES "
                "(2, 1, 1, 'w', 1, 1, '{}', ?, 'modelscope', 'm', 'r', '{}', 'v1', "
                "'pending', 0, 0, ?, ?)",
                ("d" * 64, "2026-08-20 00:00:00", "2026-08-20 00:00:00"),
            )
            task_inserted = True
            # 与既有已提交槽位同键：唯一冲突。
            connection.exec_driver_sql(
                "INSERT INTO model_storage_sync_task_dedupe_slots "
                "(dedupe_key, task_id, created_at, updated_at) VALUES "
                "('msync:1:1', 2, ?, ?)",
                ("2026-08-20 00:00:00", "2026-08-20 00:00:00"),
            )
            connection.commit()
        except IntegrityError:
            connection.rollback()
            assert task_inserted, "任务必须先 flush 才能进入槽位冲突路径"

    with engine.connect() as connection:
        assert (
            connection.exec_driver_sql(
                "SELECT count(*) FROM model_storage_sync_tasks"
            ).scalar_one()
            == 0
        ), "槽位唯一冲突后任务必须整体回滚，不得残留"
        assert (
            connection.exec_driver_sql(
                "SELECT count(*) FROM model_storage_sync_task_dedupe_slots"
            ).scalar_one()
            == 1
        ), "既有已提交槽位不受回滚影响"
