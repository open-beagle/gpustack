"""S3 Profile 生命周期 migration 的真实基线升级与跨数据库 DDL 验证。"""

from datetime import datetime, timezone
from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable


BASE_REVISION = "b6c7d8e9f0a1"
LIFECYCLE_REVISION = "c7d8e9f0a1b2"


def _config(tmp_path: Path) -> tuple[Config, sa.Engine]:
    database_path = tmp_path / "profile-lifecycle.db"
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "gpustack/migrations"),
    )
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    return config, sa.create_engine(config.get_main_option("sqlalchemy.url"))


def _insert_profile(connection, profile_id, name, endpoint, bucket, **values):
    now = datetime.now(timezone.utc)
    connection.execute(
        sa.text(
            "INSERT INTO model_preheat_s3_profiles "
            "(id, name, endpoint, bucket, access_key_encrypted, "
            "secret_key_encrypted, encryption_key_version, system_managed, "
            "default_slot, created_at, updated_at) "
            "VALUES (:id, :name, :endpoint, :bucket, '{}', '{}', 'v1', "
            ":system_managed, :default_slot, :now, :now)"
        ),
        {
            "id": profile_id,
            "name": name,
            "endpoint": endpoint,
            "bucket": bucket,
            "system_managed": values.get("system_managed", False),
            "default_slot": values.get("default_slot"),
            "now": now,
        },
    )


def _seed_b6_data(connection):
    # 默认优先于手工；没有默认时手工优先；同类型最终按稳定 ID 选择。
    _insert_profile(
        connection,
        1,
        "default-system",
        "https://S3.EXAMPLE.com:443",
        "models-a",
        system_managed=True,
        default_slot="global",
    )
    _insert_profile(connection, 2, "manual-a", "https://s3.example.com", "MODELS-A")
    _insert_profile(
        connection,
        3,
        "system-b",
        "http://s3.example.com:80",
        "models-b",
        system_managed=True,
    )
    _insert_profile(connection, 4, "manual-b", "http://s3.example.com", "models-b")
    _insert_profile(
        connection, 5, "manual-c-first", "https://other.example.com", "models-c"
    )
    _insert_profile(
        connection, 6, "manual-c-second", "https://other.example.com", "models-c"
    )
    _insert_profile(connection, 7, "invalid-port-text", "https://broken:abc", "bad-a")
    _insert_profile(
        connection, 8, "invalid-port-range", "https://broken:99999", "bad-b"
    )
    _insert_profile(
        connection, 9, "invalid-userinfo", "https://user@s3.example.com", "bad-c"
    )
    _insert_profile(
        connection, 10, "invalid-path", "https://s3.example.com/path", "bad-d"
    )
    _insert_profile(
        connection, 11, "invalid-query", "https://s3.example.com?query=1", "bad-e"
    )
    _insert_profile(
        connection, 12, "invalid-fragment", "https://s3.example.com#part", "bad-f"
    )

    now = datetime.now(timezone.utc)
    connection.execute(
        sa.text(
            "INSERT INTO model_preheat_artifacts "
            "(id, profile_id, profile_config_version, artifact_id, source, model_id, "
            "resolved_revision, include_patterns, exclude_patterns, manifest_path, "
            "manifest_digest, file_count, total_size, manifest_state, last_verified_at, "
            "created_at, updated_at) VALUES "
            "(1, 1, 1, :digest, 'huggingface', 'org/model', 'rev', '[]', '[]', "
            "'manifest.json', :digest, 1, 1, 'valid', :now, :now, :now)"
        ),
        {"digest": "a" * 64, "now": now},
    )
    connection.execute(
        sa.text(
            "INSERT INTO model_storage_sync_tasks "
            "(id, model_file_id, worker_id, worker_uuid, profile_id, "
            "profile_config_version, request_identity, request_digest, source, model_id, "
            "resolved_revision, credential_snapshot_encrypted, encryption_key_version, "
            "started_at, created_at, updated_at) VALUES "
            "(1, 1, 1, 'worker', 3, 1, '{}', :digest, 'modelscope', 'org/model', "
            "'rev', '{}', 'v1', :now, :now, :now)"
        ),
        {"digest": "b" * 64, "now": now},
    )
    connection.execute(
        sa.text(
            "INSERT INTO model_preheat_tasks "
            "(id, source, model_id, resolved_revision, include_patterns, exclude_patterns, "
            "selection_digest, request_identity, request_digest, target_scope, "
            "target_worker_uuids, target_worker_snapshot, s3_profile_id, "
            "s3_profile_config_version, s3_profile_snapshot_encrypted, "
            "encryption_key_version, s3_backfill_policy, created_at, updated_at) VALUES "
            "(1, 'huggingface', 'org/model', 'rev', '[]', '[]', :digest, '{}', :digest, "
            "'seed_worker', '[]', '[]', 4, 1, '{}', 'v1', 'when_missing', :now, :now)"
        ),
        {"digest": "c" * 64, "now": now},
    )
    connection.execute(
        sa.text(
            "INSERT INTO model_preheat_worker_tasks "
            "(id, task_id, worker_uuid, role, attempt, created_at, updated_at) "
            "VALUES (1, 1, 'worker', 'seed', 1, :now, :now)"
        ),
        {"now": now},
    )
    # 已领取下载是使用证据；未领取的 profile 2 不应被误判为 used。
    for execution_id, profile_id, claimed_at in ((1, 5, now), (2, 2, None)):
        connection.execute(
            sa.text(
                "INSERT INTO model_file_download_executions "
                "(id, model_file_id, request_identity, request_digest, target_worker_id, "
                "target_worker_uuid, default_profile_id, claimed_at, created_at, updated_at) "
                "VALUES (:id, :model_file_id, '{}', :digest, 1, 'worker', :profile_id, "
                ":claimed_at, :now, :now)"
            ),
            {
                "id": execution_id,
                "model_file_id": execution_id,
                "digest": str(execution_id) * 64,
                "profile_id": profile_id,
                "claimed_at": claimed_at,
                "now": now,
            },
        )


def test_real_b6_sqlite_upgrade_backfills_and_downgrades(tmp_path):
    config, engine = _config(tmp_path)
    command.upgrade(config, BASE_REVISION)
    with engine.begin() as connection:
        _seed_b6_data(connection)

    command.upgrade(config, LIFECYCLE_REVISION)
    inspector = sa.inspect(engine)
    columns = {
        column["name"]: column
        for column in inspector.get_columns("model_preheat_s3_profiles")
    }
    assert columns["active_storage_key"]["type"].length == 64
    assert columns["active_storage_key"]["nullable"] is True
    assert columns["ever_used_at"]["nullable"] is True
    unique_columns = {
        tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("model_preheat_s3_profiles")
    }
    assert ("active_storage_key",) in unique_columns

    with engine.connect() as connection:
        rows = (
            connection.execute(
                sa.text(
                    "SELECT id, lifecycle_state, active_storage_key, default_slot, ever_used_at "
                    "FROM model_preheat_s3_profiles ORDER BY id"
                )
            )
            .mappings()
            .all()
        )
    assert [(row["id"], row["lifecycle_state"]) for row in rows] == [
        (1, "active"),
        (2, "maintenance"),
        (3, "maintenance"),
        (4, "active"),
        (5, "active"),
        (6, "maintenance"),
        (7, "maintenance"),
        (8, "maintenance"),
        (9, "maintenance"),
        (10, "maintenance"),
        (11, "maintenance"),
        (12, "maintenance"),
    ]
    assert rows[1]["default_slot"] is None
    assert rows[0]["active_storage_key"] is not None
    assert rows[1]["active_storage_key"] is None
    assert {row["id"] for row in rows if row["ever_used_at"] is not None} == {
        1,
        3,
        4,
        5,
    }

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                sa.text(
                    "UPDATE model_preheat_s3_profiles SET lifecycle_state='active', "
                    "active_storage_key=:key WHERE id=2"
                ),
                {"key": rows[0]["active_storage_key"]},
            )

    command.downgrade(config, BASE_REVISION)
    inspector = sa.inspect(engine)
    assert {
        "lifecycle_state",
        "active_storage_key",
        "ever_used_at",
    }.isdisjoint(
        {
            column["name"]
            for column in inspector.get_columns("model_preheat_s3_profiles")
        }
    )
    with engine.connect() as connection:
        assert (
            connection.execute(
                sa.text("SELECT COUNT(*) FROM model_preheat_s3_profiles")
            ).scalar_one()
            == 12
        )
        assert (
            connection.execute(
                sa.text("SELECT version_num FROM alembic_version")
            ).scalar_one()
            == BASE_REVISION
        )
    engine.dispose()


@pytest.mark.parametrize(
    "dialect",
    [sqlite.dialect(), postgresql.dialect(), mysql.dialect()],
    ids=["sqlite", "postgresql", "mysql"],
)
def test_lifecycle_columns_and_unique_constraint_compile_for_supported_databases(
    dialect,
):
    metadata = sa.MetaData()
    table = sa.Table(
        "model_preheat_s3_profiles",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "lifecycle_state", sa.String(32), nullable=False, server_default="active"
        ),
        sa.Column("active_storage_key", sa.String(64), nullable=True),
        sa.Column("ever_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "active_storage_key",
            name="uix_model_preheat_s3_profiles_active_storage_key",
        ),
    )
    ddl = str(CreateTable(table).compile(dialect=dialect))
    assert "active_storage_key" in ddl
    assert "VARCHAR(64)" in ddl.upper()
    assert "ever_used_at" in ddl
    assert "uix_model_preheat_s3_profiles_active_storage_key" in ddl
