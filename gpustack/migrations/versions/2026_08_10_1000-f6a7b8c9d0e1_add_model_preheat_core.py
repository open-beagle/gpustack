"""新增模型预热核心表

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-08-10 10:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_preheat_s3_profiles",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("endpoint", sa.String(length=255), nullable=False),
        sa.Column("bucket", sa.String(length=255), nullable=False),
        sa.Column("prefix", sa.String(length=255), nullable=False, server_default=""),
        sa.Column(
            "tls_enabled", sa.Boolean(), nullable=False, server_default=sa.true()
        ),
        sa.Column("tls_verify", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("region", sa.String(length=255), nullable=True),
        sa.Column(
            "use_virtual_hosted_style",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column(
            "is_default", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("access_key_encrypted", sa.JSON(), nullable=False),
        sa.Column("secret_key_encrypted", sa.JSON(), nullable=False),
        sa.Column("encryption_key_version", sa.String(length=255), nullable=False),
        sa.Column("config_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "connectivity_state",
            sa.String(length=255),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("last_connectivity_check_id", sa.Integer(), nullable=True),
        sa.Column("last_connectivity_checked_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.UniqueConstraint("name", name="uix_model_preheat_s3_profiles_name"),
    )
    op.create_table(
        "model_preheat_tasks",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("source", sa.String(length=255), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False),
        sa.Column("requested_revision", sa.String(length=255), nullable=True),
        sa.Column("resolved_revision", sa.String(length=255), nullable=False),
        sa.Column("include_patterns", sa.JSON(), nullable=False),
        sa.Column("exclude_patterns", sa.JSON(), nullable=False),
        sa.Column("selection_digest", sa.String(length=255), nullable=False),
        sa.Column("cache_key", sa.String(length=255), nullable=False),
        sa.Column("generation_id", sa.String(length=255), nullable=False),
        sa.Column(
            "desired_state",
            sa.String(length=255),
            nullable=False,
            server_default="running",
        ),
        sa.Column(
            "execution_state",
            sa.String(length=255),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("state_message", sa.Text(), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("seed_worker_uuid", sa.String(length=255), nullable=True),
        sa.Column("seed_worker_id", sa.Integer(), nullable=True),
        sa.Column("seed_source", sa.String(length=255), nullable=True),
        sa.Column("target_scope", sa.String(length=255), nullable=False),
        sa.Column("target_gpu_names", sa.JSON(), nullable=True),
        sa.Column("target_worker_uuids", sa.JSON(), nullable=False),
        sa.Column("target_worker_snapshot", sa.JSON(), nullable=False),
        sa.Column("local_cache_hit_worker_uuids", sa.JSON(), nullable=True),
        sa.Column("removed_target_worker_uuids", sa.JSON(), nullable=True),
        sa.Column("s3_profile_id", sa.Integer(), nullable=False),
        sa.Column("s3_profile_config_version", sa.Integer(), nullable=False),
        sa.Column("s3_profile_snapshot_encrypted", sa.JSON(), nullable=False),
        sa.Column("encryption_key_version", sa.String(length=255), nullable=False),
        sa.Column("s3_backfill_policy", sa.String(length=255), nullable=False),
        sa.Column("s3_ready_path", sa.String(length=255), nullable=True),
        sa.Column("s3_manifest_path", sa.String(length=255), nullable=True),
        sa.Column("manifest_digest", sa.String(length=255), nullable=True),
        sa.Column(
            "keep_new_workers_in_sync",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("schedule_id", sa.Integer(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("started_at", UTCDateTime(), nullable=True),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["s3_profile_id"], ["model_preheat_s3_profiles.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"], ondelete="SET NULL"
        ),
    )
    op.create_table(
        "model_preheat_s3_connectivity_checks",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("profile_config_version", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("request_hash", sa.String(length=255), nullable=True),
        sa.Column("scope_key", sa.String(length=255), nullable=True),
        sa.Column("active_key", sa.String(length=255), nullable=True),
        sa.Column(
            "state", sa.String(length=255), nullable=False, server_default="pending"
        ),
        sa.Column("target_worker_uuids", sa.JSON(), nullable=False),
        sa.Column("success_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "not_checked_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("started_at", UTCDateTime(), nullable=True),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["profile_id"], ["model_preheat_s3_profiles.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("active_key", name="uix_preheat_connectivity_active"),
        sa.UniqueConstraint(
            "idempotency_key", name="uix_preheat_connectivity_idempotency"
        ),
    )
    op.create_table(
        "model_preheat_worker_tasks",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("connectivity_check_id", sa.Integer(), nullable=True),
        sa.Column("worker_uuid", sa.String(length=255), nullable=False),
        sa.Column("worker_id", sa.Integer(), nullable=True),
        sa.Column("role", sa.String(length=255), nullable=False),
        sa.Column(
            "state", sa.String(length=255), nullable=False, server_default="pending"
        ),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("lease_owner", sa.String(length=255), nullable=True),
        sa.Column("lease_token_hash", sa.String(length=255), nullable=True),
        sa.Column("lease_expires_at", UTCDateTime(), nullable=True),
        sa.Column("last_heartbeat_at", UTCDateTime(), nullable=True),
        sa.Column("state_message", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=255), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("local_staging_dir", sa.Text(), nullable=True),
        sa.Column(
            "downloaded_size", sa.BigInteger(), nullable=False, server_default="0"
        ),
        sa.Column("total_size", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("resumable_cursor", sa.JSON(), nullable=True),
        sa.Column("started_at", UTCDateTime(), nullable=True),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["task_id"], ["model_preheat_tasks.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["connectivity_check_id"],
            ["model_preheat_s3_connectivity_checks.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "task_id", "worker_uuid", "role", name="uix_preheat_task_worker_role"
        ),
        sa.UniqueConstraint(
            "connectivity_check_id",
            "worker_uuid",
            "role",
            name="uix_preheat_check_worker_role",
        ),
    )
    op.create_index(
        "ix_preheat_worker_uuid_state",
        "model_preheat_worker_tasks",
        ["worker_uuid", "state"],
        unique=False,
    )
    op.create_table(
        "model_preheat_idempotency_records",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("operation", sa.String(length=255), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=255), nullable=False),
        sa.Column("resource_type", sa.String(length=255), nullable=False),
        sa.Column("resource_id", sa.Integer(), nullable=False),
        sa.Column(
            "response_status", sa.Integer(), nullable=False, server_default="200"
        ),
        sa.Column("expires_at", UTCDateTime(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.UniqueConstraint(
            "user_id", "operation", "idempotency_key", name="uix_preheat_idempotency"
        ),
    )
    op.create_table(
        "model_preheat_task_locks",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("operation_key", sa.String(length=255), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("lease_expires_at", UTCDateTime(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["task_id"], ["model_preheat_tasks.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("operation_key", name="uix_preheat_operation"),
    )
    op.create_table(
        "model_preheat_publish_locks",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("s3_profile_id", sa.Integer(), nullable=False),
        sa.Column("cache_key", sa.String(length=255), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("lease_expires_at", UTCDateTime(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["s3_profile_id"], ["model_preheat_s3_profiles.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["task_id"], ["model_preheat_tasks.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("s3_profile_id", "cache_key", name="uix_preheat_publish"),
    )


def downgrade() -> None:
    op.drop_table("model_preheat_publish_locks")
    op.drop_table("model_preheat_task_locks")
    op.drop_table("model_preheat_idempotency_records")
    op.drop_table("model_preheat_worker_tasks")
    op.drop_table("model_preheat_s3_connectivity_checks")
    op.drop_table("model_preheat_tasks")
    op.drop_table("model_preheat_s3_profiles")
