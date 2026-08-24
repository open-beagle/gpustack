import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION_MODULE = (
    "gpustack.migrations.versions."
    "2026_08_24_1000-f0a1b2c3d4e5_add_model_file_worker_snapshot"
)


def _migration():
    return importlib.import_module(MIGRATION_MODULE)


def test_worker_snapshot_migration_backfills_only_existing_workers():
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    sa.Table(
        "workers",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("worker_uuid", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
    )
    sa.Table(
        "model_files",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("worker_id", sa.Integer(), nullable=True),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO workers (id, worker_uuid, name) "
                "VALUES (1, 'worker-uuid', 'worker-a')"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO model_files (id, worker_id) " "VALUES (1, 1), (2, 999)"
            )
        )
        migration = _migration()
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        rows = connection.execute(
            sa.text(
                "SELECT id, worker_id, worker_uuid_snapshot, worker_name_snapshot "
                "FROM model_files ORDER BY id"
            )
        ).all()

    assert migration.down_revision == "e9f0a1b2c3d4"
    assert rows == [
        (1, 1, "worker-uuid", "worker-a"),
        (2, 999, None, None),
    ]
