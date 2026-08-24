import importlib
import io

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION_MODULE = (
    "gpustack.migrations.versions."
    "2026_08_24_1100-0a1b2c3d4e5f_add_model_storage_sync_policies"
)


def _migration():
    return importlib.import_module(MIGRATION_MODULE)


def test_sync_policy_migration_creates_independent_policy_tables_on_sqlite():
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    for table_name in (
        "model_preheat_s3_profiles",
        "model_files",
        "users",
    ):
        sa.Table(table_name, metadata, sa.Column("id", sa.Integer(), primary_key=True))
    metadata.create_all(engine)

    with engine.begin() as connection:
        migration = _migration()
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        tables = set(sa.inspect(connection).get_table_names())
        run_constraints = {
            item["name"]
            for item in sa.inspect(connection).get_unique_constraints(
                "model_storage_sync_policy_runs"
            )
        }

    assert migration.down_revision == "f0a1b2c3d4e5"
    assert "model_storage_sync_policies" in tables
    assert "model_storage_sync_policy_runs" in tables
    assert run_constraints == {
        "uix_storage_sync_policy_operation",
        "uix_storage_sync_policy_window",
    }


def test_sync_policy_migration_ddl_renders_for_supported_databases():
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
        assert "model_storage_sync_policies" in ddl
        assert "model_storage_sync_policy_runs" in ddl
