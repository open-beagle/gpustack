from pathlib import Path

from alembic import command
from alembic.config import Config
import sqlalchemy as sa
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.schema import CreateTable

from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicyRun,
    ModelPreheatDistributionPolicyRunTask,
)


BASE_REVISION = "b4c5d6e7f8a9"
OBSERVABILITY_REVISION = "c5d6e7f8a9b0"


def _config(tmp_path):
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "gpustack/migrations"),
    )
    config.set_main_option(
        "sqlalchemy.url", f"sqlite:///{tmp_path / 'run-observability.db'}"
    )
    return config


def test_distribution_run_observability_migration_is_linear_and_reversible(tmp_path):
    config = _config(tmp_path)
    command.upgrade(config, BASE_REVISION)
    command.upgrade(config, OBSERVABILITY_REVISION)
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    inspector = sa.inspect(engine)
    table = "model_preheat_distribution_policy_run_tasks"

    assert table in inspector.get_table_names()
    assert "outcome" in {
        column["name"]
        for column in inspector.get_columns("model_preheat_distribution_policy_runs")
    }
    assert {column["name"] for column in inspector.get_columns(table)} == {
        "run_id",
        "task_id",
    }
    assert {"run_id", "task_id"} == set(
        inspector.get_pk_constraint(table)["constrained_columns"]
    )
    assert "ix_distribution_run_task_task" in {
        index["name"] for index in inspector.get_indexes(table)
    }
    foreign_keys = inspector.get_foreign_keys(table)
    assert any(
        foreign_key["constrained_columns"] == ["run_id"]
        and foreign_key["referred_table"] == "model_preheat_distribution_policy_runs"
        and foreign_key["options"].get("ondelete") == "CASCADE"
        for foreign_key in foreign_keys
    )
    assert any(
        foreign_key["constrained_columns"] == ["task_id"]
        and foreign_key["referred_table"] == "model_preheat_worker_tasks"
        and foreign_key["options"].get("ondelete") == "RESTRICT"
        for foreign_key in foreign_keys
    )

    command.downgrade(config, BASE_REVISION)
    assert table not in sa.inspect(engine).get_table_names()
    assert "outcome" not in {
        column["name"]
        for column in sa.inspect(engine).get_columns(
            "model_preheat_distribution_policy_runs"
        )
    }
    engine.dispose()


def test_distribution_run_task_table_compiles_for_supported_databases():
    for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
        link_table = ModelPreheatDistributionPolicyRunTask.__table__
        link_ddl = str(CreateTable(link_table).compile(dialect=dialect))
        assert link_table.name in link_ddl
        assert "run_id" in link_ddl
        assert "task_id" in link_ddl
        run_ddl = str(
            CreateTable(ModelPreheatDistributionPolicyRun.__table__).compile(
                dialect=dialect
            )
        )
        assert "outcome" in run_ddl
