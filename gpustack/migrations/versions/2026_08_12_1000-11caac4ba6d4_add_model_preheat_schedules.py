"""add model preheat schedules

Revision ID: 11caac4ba6d4
Revises: f2a3b4c5d6e7
Create Date: 2026-08-12 10:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


revision: str = "11caac4ba6d4"
down_revision: Union[str, None] = "f2a3b4c5d6e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_preheat_schedules",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("cron_expression", sa.String(length=255), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("window_duration_minutes", sa.Integer(), nullable=False),
        sa.Column("max_concurrency", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("model_id", sa.String(length=512), nullable=False),
        sa.Column("revision", sa.String(length=512), nullable=True),
        sa.Column("include_patterns", sa.JSON(), nullable=False),
        sa.Column("exclude_patterns", sa.JSON(), nullable=False),
        sa.Column("target_scope", sa.String(length=64), nullable=False),
        sa.Column("target_worker_uuids", sa.JSON(), nullable=False),
        sa.Column("seed_worker_uuid", sa.String(length=255), nullable=True),
        sa.Column("s3_profile_id", sa.Integer(), nullable=False),
        sa.Column("s3_backfill_policy", sa.String(length=64), nullable=False),
        sa.Column("keep_new_workers_in_sync", sa.Boolean(), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("next_window_start_utc", UTCDateTime(), nullable=True),
        sa.Column("last_window_start_utc", UTCDateTime(), nullable=True),
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["s3_profile_id"],
            ["model_preheat_s3_profiles.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uix_preheat_schedule_name"),
    )
    op.create_table(
        "model_preheat_schedule_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("schedule_id", sa.Integer(), nullable=False),
        sa.Column("window_start_utc", UTCDateTime(), nullable=False),
        sa.Column("window_end_utc", UTCDateTime(), nullable=False),
        sa.Column("trigger", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=64), nullable=False),
        sa.Column("operation_key", sa.String(length=64), nullable=True),
        sa.Column("slot", sa.Integer(), nullable=True),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=255), nullable=True),
        sa.Column("started_at", UTCDateTime(), nullable=True),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["schedule_id"], ["model_preheat_schedules.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["task_id"], ["model_preheat_tasks.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "schedule_id", "window_start_utc", name="uix_preheat_schedule_window"
        ),
        sa.UniqueConstraint(
            "operation_key", name="uix_preheat_schedule_run_operation"
        ),
        sa.UniqueConstraint(
            "schedule_id", "slot", name="uix_preheat_schedule_slot"
        ),
    )


def downgrade() -> None:
    op.drop_table("model_preheat_schedule_runs")
    op.drop_table("model_preheat_schedules")
