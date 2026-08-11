"""添加模型预热持久分发策略

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-08-11 15:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


revision: str = "b8c9d0e1f2a3"
down_revision: Union[str, None] = "a7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_preheat_distribution_policies",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "profile_version_stale",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("profile_config_version", sa.Integer(), nullable=False),
        sa.Column("cache_key", sa.String(length=255), nullable=False),
        sa.Column("target_scope", sa.String(length=255), nullable=False),
        sa.Column("worker_selector", sa.JSON(), nullable=False),
        sa.Column("gpu_selector", sa.JSON(), nullable=False),
        sa.Column("selector_digest", sa.String(length=255), nullable=False),
        sa.Column("created_by_task_id", sa.Integer(), nullable=True),
        sa.Column("last_reconciled_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["profile_id"], ["model_preheat_s3_profiles.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["created_by_task_id"], ["model_preheat_tasks.id"], ondelete="RESTRICT"
        ),
        sa.UniqueConstraint(
            "profile_id",
            "cache_key",
            "target_scope",
            "selector_digest",
            name="uix_preheat_distribution_policy_selector",
        ),
    )
    op.create_table(
        "model_preheat_worker_observations",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("worker_uuid", sa.String(length=255), primary_key=True),
        sa.Column("worker_id", sa.Integer(), nullable=False),
        sa.Column("network_fingerprint", sa.String(length=255), nullable=False),
        sa.Column("ready", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
    )
    with op.batch_alter_table("model_preheat_worker_tasks") as batch_op:
        batch_op.add_column(
            sa.Column("distribution_policy_id", sa.Integer(), nullable=True)
        )
        batch_op.add_column(sa.Column("operation_key", sa.String(255), nullable=True))
        batch_op.create_foreign_key(
            "fk_preheat_worker_task_distribution_policy",
            "model_preheat_distribution_policies",
            ["distribution_policy_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_unique_constraint(
            "uix_preheat_worker_distribution_operation", ["operation_key"]
        )


def downgrade() -> None:
    with op.batch_alter_table("model_preheat_worker_tasks") as batch_op:
        batch_op.drop_constraint(
            "uix_preheat_worker_distribution_operation", type_="unique"
        )
        batch_op.drop_constraint(
            "fk_preheat_worker_task_distribution_policy", type_="foreignkey"
        )
        batch_op.drop_column("operation_key")
        batch_op.drop_column("distribution_policy_id")
    op.drop_table("model_preheat_worker_observations")
    op.drop_table("model_preheat_distribution_policies")
