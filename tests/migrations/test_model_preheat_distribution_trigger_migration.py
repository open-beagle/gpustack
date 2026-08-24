from pathlib import Path

from alembic import command
from alembic.config import Config
import sqlalchemy as sa
from sqlmodel import Session

from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
    ModelPreheatDistributionPolicyTriggerModeEnum,
)


BASE_REVISION = "4c5d6e7f8a9"
TRIGGER_REVISION = "5d6e7f8a9b0"


def _config(tmp_path):
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "gpustack/migrations"),
    )
    config.set_main_option("sqlalchemy.url", f"sqlite:///{tmp_path / 'trigger.db'}")
    return config


def test_distribution_trigger_migration_is_linear_from_shared_s3_head():
    versions = Path("gpustack/migrations/versions")
    source = next(
        versions.glob("*_add_model_preheat_distribution_triggers.py")
    ).read_text(encoding="utf-8")
    assert 'down_revision: Union[str, None] = "4c5d6e7f8a9"' in source
    assert "model_preheat_distribution_policy_runs" in source
    assert "postgresql_where" not in source


def test_distribution_trigger_upgrade_and_downgrade(tmp_path):
    config = _config(tmp_path)
    command.upgrade(config, BASE_REVISION)
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    now = "2026-08-24 00:00:00"
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "INSERT INTO model_preheat_s3_profiles "
                "(id, name, endpoint, bucket, access_key_encrypted, secret_key_encrypted, "
                "encryption_key_version, config_version, created_at, updated_at) "
                "VALUES (1, 's3', 'https://s3.example.com', 'models', '{}', '{}', 'v1', 1, :now, :now)"
            ),
            {"now": now},
        )
        connection.execute(
            sa.text(
                "INSERT INTO model_preheat_distribution_policies "
                "(id, name, enabled, profile_version_stale, profile_id, profile_config_version, "
                "request_identity, request_digest, target_scope, worker_selector, gpu_selector, "
                "selector_digest, created_at, updated_at) "
                "VALUES (1, 'existing', 1, 0, 1, 1, '{}', :digest, 'SELECTED_WORKERS', "
                "'{\"worker_uuids\":[\"worker-a\"]}', '{}', :selector, :now, :now)"
            ),
            {"digest": "a" * 64, "selector": "b" * 64, "now": now},
        )
    command.upgrade(config, TRIGGER_REVISION)
    with Session(engine) as session:
        policy = session.get(ModelPreheatDistributionPolicy, 1)
    assert (
        policy.trigger_mode == ModelPreheatDistributionPolicyTriggerModeEnum.CONTINUOUS
    )
    inspector = sa.inspect(engine)
    columns = {
        column["name"]
        for column in inspector.get_columns("model_preheat_distribution_policies")
    }
    assert {
        "trigger_mode",
        "cron_expression",
        "timezone",
        "next_run_at",
        "last_run_at",
        "blocked_reason",
    } <= columns
    assert "model_preheat_distribution_policy_runs" in inspector.get_table_names()
    assert "model_preheat_distribution_worker_slots" in inspector.get_table_names()

    command.downgrade(config, BASE_REVISION)
    inspector = sa.inspect(engine)
    columns = {
        column["name"]
        for column in inspector.get_columns("model_preheat_distribution_policies")
    }
    assert "trigger_mode" not in columns
    assert "model_preheat_distribution_policy_runs" not in inspector.get_table_names()
    assert "model_preheat_distribution_worker_slots" not in inspector.get_table_names()
