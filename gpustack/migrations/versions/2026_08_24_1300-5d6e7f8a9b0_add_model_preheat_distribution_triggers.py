"""增加固定 Artifact 分发策略触发方式与定时运行租约。

Revision ID: 5d6e7f8a9b0
Revises: 4c5d6e7f8a9
Create Date: 2026-08-24 13:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "5d6e7f8a9b0"
down_revision: Union[str, None] = "4c5d6e7f8a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_preheat_distribution_policies") as batch_op:
        batch_op.add_column(
            sa.Column(
                "trigger_mode",
                sa.String(length=32),
                nullable=False,
                server_default="continuous",
            )
        )
        batch_op.add_column(sa.Column("cron_expression", sa.String(length=255), nullable=True))
        batch_op.add_column(
            sa.Column("timezone", sa.String(length=64), nullable=False, server_default="UTC")
        )
        batch_op.add_column(sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("blocked_reason", sa.String(length=255), nullable=True))

    op.create_table(
        "model_preheat_distribution_policy_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("policy_id", sa.Integer(), nullable=False),
        sa.Column("trigger", sa.String(length=32), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("window_start_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("operation_key", sa.String(length=64), nullable=False),
        sa.Column("lease_owner", sa.String(length=64), nullable=True),
        sa.Column("lease_token", sa.String(length=64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=255), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["policy_id"], ["model_preheat_distribution_policies.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("operation_key", name="uix_distribution_policy_operation"),
        sa.UniqueConstraint("policy_id", "window_start_utc", name="uix_distribution_policy_window"),
    )
    op.create_table(
        "model_preheat_distribution_worker_slots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("policy_id", sa.Integer(), nullable=False),
        sa.Column("worker_uuid", sa.String(length=255), nullable=False),
        sa.Column("active_task_id", sa.Integer(), nullable=True),
        sa.Column("active_operation_key", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["policy_id"], ["model_preheat_distribution_policies.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "policy_id", "worker_uuid", name="uix_distribution_policy_worker_slot"
        ),
    )


def downgrade() -> None:
    op.drop_table("model_preheat_distribution_worker_slots")
    op.drop_table("model_preheat_distribution_policy_runs")
    with op.batch_alter_table("model_preheat_distribution_policies") as batch_op:
        batch_op.drop_column("blocked_reason")
        batch_op.drop_column("last_run_at")
        batch_op.drop_column("next_run_at")
        batch_op.drop_column("timezone")
        batch_op.drop_column("cron_expression")
        batch_op.drop_column("trigger_mode")
