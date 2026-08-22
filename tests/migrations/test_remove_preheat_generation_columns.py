import importlib
import io

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION_MODULE = (
    "gpustack.migrations.versions."
    "2026_08_22_1400-b6c7d8e9f0a1_remove_preheat_generation_columns"
)


def _migration():
    return importlib.import_module(MIGRATION_MODULE)


def _run_online(connection, direction):
    migration = _migration()
    context = MigrationContext.configure(connection)
    migration.op = Operations(context)
    getattr(migration, direction)()


def test_revision_extends_current_model_storage_head():
    migration = _migration()

    assert migration.revision == "b6c7d8e9f0a1"
    assert migration.down_revision == "a5b6c7d8e9f0"


def test_sqlite_upgrade_and_downgrade_remove_and_restore_generation_columns():
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    sa.Table(
        "model_preheat_tasks",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("request_digest", sa.String(length=64), nullable=False),
        sa.Column("generation_id", sa.String(length=256), nullable=False),
        sa.Column("s3_ready_path", sa.String(length=255), nullable=True),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO model_preheat_tasks "
                "(id, request_digest, generation_id, s3_ready_path) "
                "VALUES (1, :digest, 'preheat-old', 'ready.json')"
            ),
            {"digest": "d" * 64},
        )
        _run_online(connection, "upgrade")
        upgraded = {
            column["name"]
            for column in sa.inspect(connection).get_columns("model_preheat_tasks")
        }
        row = connection.execute(
            sa.text("SELECT id, request_digest FROM model_preheat_tasks")
        ).one()
        _run_online(connection, "downgrade")
        downgraded = {
            column["name"]
            for column in sa.inspect(connection).get_columns("model_preheat_tasks")
        }
        restored = connection.execute(
            sa.text("SELECT generation_id, s3_ready_path FROM model_preheat_tasks")
        ).one()

    assert upgraded == {"id", "request_digest"}
    assert row == (1, "d" * 64)
    assert downgraded == {
        "id",
        "request_digest",
        "generation_id",
        "s3_ready_path",
    }
    assert restored == ("legacy", None)


def test_postgresql_and_mysql_ddl_compile_for_upgrade_and_downgrade():
    migration = _migration()

    for dialect in ("postgresql", "mysql"):
        rendered = []
        for direction in ("upgrade", "downgrade"):
            output = io.StringIO()
            context = MigrationContext.configure(
                url=f"{dialect}://",
                opts={"as_sql": True, "output_buffer": output},
            )
            migration.op = Operations(context)
            getattr(migration, direction)()
            rendered.append(output.getvalue())

        ddl = "\n".join(rendered)
        assert "generation_id" in ddl
        assert "s3_ready_path" in ddl
        assert "DROP COLUMN" in ddl
        assert "ADD COLUMN" in ddl
