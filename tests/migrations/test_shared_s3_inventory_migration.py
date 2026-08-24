import importlib
import io

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION_MODULE = (
    "gpustack.migrations.versions."
    "2026_08_24_1300-3b4c5d6e7f8_add_shared_s3_inventory_discovery"
)


def _migration():
    return importlib.import_module(MIGRATION_MODULE)


def test_shared_s3_inventory_migration_adds_profile_refresh_state():
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    sa.Table(
        "model_preheat_s3_profiles",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
    )
    sa.Table(
        "model_preheat_artifacts",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("profile_config_version", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("artifact_id", sa.String(), nullable=False),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = _migration()
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        columns = {
            column["name"]
            for column in sa.inspect(connection).get_columns("model_preheat_s3_profiles")
        }
        indexes = sa.inspect(connection).get_indexes("model_preheat_artifacts")

    assert migration.revision == "3b4c5d6e7f8"
    assert migration.down_revision == "2a3b4c5d6e7f"
    assert {
        "inventory_refresh_interval_seconds",
        "inventory_last_attempt_at",
        "inventory_last_success_at",
        "inventory_last_scan_count",
        "inventory_last_error_code",
        "inventory_refresh_owner",
        "inventory_refresh_config_version",
        "inventory_refresh_lease_expires_at",
    } <= columns
    assert any(
        index["name"] == "ix_preheat_artifact_profile_version_source"
        for index in indexes
    )


def test_shared_s3_inventory_migration_renders_supported_dialects():
    migration = _migration()
    for dialect in ("sqlite", "postgresql", "mysql"):
        output = io.StringIO()
        context = MigrationContext.configure(
            url=f"{dialect}://", opts={"as_sql": True, "output_buffer": output}
        )
        migration.op = Operations(context)
        migration.upgrade()
        ddl = output.getvalue()
        assert "inventory_last_success_at" in ddl
        assert "inventory_refresh_owner" in ddl
        assert "ix_preheat_artifact_profile_version_source" in ddl
        assert "model_id" not in ddl
