import importlib

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION_MODULE = (
    "gpustack.migrations.versions."
    "2026_08_24_1200-1a2b3c4d5e6f_bind_distribution_artifact_source"
)


def test_distribution_source_migration_backfills_existing_task_artifact():
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    policies = sa.Table(
        "model_preheat_distribution_policies",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("profile_config_version", sa.Integer(), nullable=False),
        sa.Column("created_by_task_id", sa.Integer(), nullable=True),
    )
    artifacts = sa.Table(
        "model_preheat_artifacts",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("profile_config_version", sa.Integer(), nullable=False),
        sa.Column("created_by_task_id", sa.Integer(), nullable=True),
    )
    sa.Table(
        "model_storage_sync_tasks",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            policies.insert().values(
                id=1,
                profile_id=2,
                profile_config_version=3,
                created_by_task_id=4,
            )
        )
        connection.execute(
            artifacts.insert().values(
                id=5,
                profile_id=2,
                profile_config_version=3,
                created_by_task_id=4,
            )
        )
        migration = importlib.import_module(MIGRATION_MODULE)
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        row = connection.execute(
            sa.text(
                "SELECT source_artifact_id, source_sync_task_id "
                "FROM model_preheat_distribution_policies WHERE id = 1"
            )
        ).one()

    assert migration.down_revision == "0a1b2c3d4e5f"
    assert row == (5, None)
