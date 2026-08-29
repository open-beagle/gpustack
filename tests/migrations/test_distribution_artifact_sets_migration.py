from pathlib import Path

from alembic import command
from alembic.config import Config
import sqlalchemy as sa
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.schema import CreateTable

from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicyArtifact,
    ModelPreheatDistributionWorkerSlot,
)


BASE_REVISION = "a3b4c5d6e7f8"
SET_REVISION = "b4c5d6e7f8a9"


def _config(tmp_path):
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "gpustack/migrations"),
    )
    config.set_main_option("sqlalchemy.url", f"sqlite:///{tmp_path / 'sets.db'}")
    return config


def test_distribution_artifact_sets_migration_is_linear_and_reversible(tmp_path):
    config = _config(tmp_path)
    command.upgrade(config, BASE_REVISION)
    command.upgrade(config, SET_REVISION)
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    inspector = sa.inspect(engine)
    assert "model_preheat_distribution_policy_artifacts" in inspector.get_table_names()
    assert "selection_mode" in {
        column["name"]
        for column in inspector.get_columns("model_preheat_distribution_policies")
    }
    worker_columns = {
        column["name"] for column in inspector.get_columns("model_preheat_worker_tasks")
    }
    assert {"distribution_artifact_id", "distribution_request_digest"} <= worker_columns
    command.downgrade(config, BASE_REVISION)
    inspector = sa.inspect(engine)
    assert (
        "model_preheat_distribution_policy_artifacts" not in inspector.get_table_names()
    )
    engine.dispose()


def test_distribution_artifact_set_tables_compile_for_supported_databases():
    for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
        for table in (
            ModelPreheatDistributionPolicyArtifact.__table__,
            ModelPreheatDistributionWorkerSlot.__table__,
        ):
            ddl = str(CreateTable(table).compile(dialect=dialect))
            assert table.name in ddl
