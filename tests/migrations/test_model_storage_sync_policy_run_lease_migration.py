import importlib
import io

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION_MODULE = (
    "gpustack.migrations.versions."
    "2026_08_24_1230-2a3b4c5d6e7f_add_sync_policy_run_lease"
)


def _migration():
    return importlib.import_module(MIGRATION_MODULE)


def test_sync_policy_run_lease_migration_adds_recovery_columns_on_sqlite():
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    sa.Table(
        "model_storage_sync_policy_runs",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
    )
    sa.Table(
        "model_preheat_idempotency_records",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
    )
    sa.Table("users", metadata, sa.Column("id", sa.Integer(), primary_key=True))
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = _migration()
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        columns = {
            column["name"]
            for column in sa.inspect(connection).get_columns(
                "model_storage_sync_policy_runs"
            )
        }
        idempotency_columns = {
            column["name"]
            for column in sa.inspect(connection).get_columns(
                "model_preheat_idempotency_records"
            )
        }
        run_foreign_keys = sa.inspect(connection).get_foreign_keys(
            "model_storage_sync_policy_runs"
        )

    assert migration.down_revision == "1a2b3c4d5e6f"
    assert {
        "attempt",
        "lease_owner",
        "lease_token",
        "lease_expires_at",
        "started_at",
        "execution_user_id",
    } <= columns
    assert {"batch_lease_token", "batch_lease_expires_at"} <= idempotency_columns
    assert any(
        foreign_key["constrained_columns"] == ["execution_user_id"]
        and foreign_key["referred_table"] == "users"
        and foreign_key["referred_columns"] == ["id"]
        and foreign_key["options"].get("ondelete") == "RESTRICT"
        for foreign_key in run_foreign_keys
    )


def test_sync_policy_run_lease_migration_ddl_renders_for_supported_databases():
    migration = _migration()
    for dialect in ("sqlite", "postgresql", "mysql"):
        output = io.StringIO()
        context = MigrationContext.configure(
            url=f"{dialect}://",
            opts={"as_sql": True, "output_buffer": output},
        )
        migration.op = Operations(context)
        migration.upgrade()
        ddl = output.getvalue()
        assert "model_storage_sync_policy_runs" in ddl
        assert "lease_owner" in ddl
        assert "model_preheat_idempotency_records" in ddl
        assert "batch_lease_token" in ddl
        if dialect == "sqlite":
            assert "uix_storage_sync_policy_operation" in ddl
            assert "uix_storage_sync_policy_window" in ddl
            assert "model_storage_sync_policies" in ddl
            assert "ON DELETE CASCADE" in ddl
            assert "ON DELETE SET NULL" in ddl
