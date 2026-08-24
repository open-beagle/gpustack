from pathlib import Path

from alembic import command
from alembic.config import Config
import sqlalchemy as sa


BASE_REVISION = "3b4c5d6e7f8"
DELIVERY_MODE_REVISION = "4c5d6e7f8a9"


def _config(tmp_path):
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "gpustack/migrations"),
    )
    config.set_main_option("sqlalchemy.url", f"sqlite:///{tmp_path / 'delivery.db'}")
    return config


def test_delivery_mode_upgrade_backfills_and_downgrades(tmp_path):
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
                "INSERT INTO model_preheat_tasks "
                "(id, source, model_id, resolved_revision, include_patterns, exclude_patterns, "
                "selection_digest, request_identity, request_digest, target_scope, target_worker_uuids, "
                "target_worker_snapshot, s3_profile_id, s3_profile_config_version, "
                "s3_profile_snapshot_encrypted, encryption_key_version, s3_backfill_policy, created_at, updated_at) "
                "VALUES (1, 'huggingface', 'org/model', 'rev', '[]', '[]', :digest, '{}', :digest, "
                "'selected_workers', '[]', '[]', 1, 1, '{}', 'v1', 'when_missing', :now, :now)"
            ),
            {"digest": "a" * 64, "now": now},
        )

    command.upgrade(config, DELIVERY_MODE_REVISION)
    inspector = sa.inspect(engine)
    for table in ("model_preheat_tasks", "model_preheat_schedules"):
        columns = {column["name"]: column for column in inspector.get_columns(table)}
        assert columns["delivery_mode"]["nullable"] is False
        assert columns["connectivity_failure_override"]["nullable"] is False
    with engine.connect() as connection:
        assert connection.execute(
            sa.text("SELECT delivery_mode, connectivity_failure_override FROM model_preheat_tasks")
        ).one() == ("s3_and_workers", 0)

    command.downgrade(config, BASE_REVISION)
    columns = {
        column["name"]
        for column in sa.inspect(engine).get_columns("model_preheat_tasks")
    }
    assert "delivery_mode" not in columns
    assert "connectivity_failure_override" not in columns
