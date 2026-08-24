"""为持续分发策略绑定实际 S3 Artifact 来源

Revision ID: 1a2b3c4d5e6f
Revises: 0a1b2c3d4e5f
Create Date: 2026-08-24 12:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "1a2b3c4d5e6f"
down_revision: Union[str, None] = "0a1b2c3d4e5f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_preheat_distribution_policies") as batch_op:
        batch_op.add_column(
            sa.Column("source_artifact_id", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("source_sync_task_id", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_distribution_policy_source_artifact",
            "model_preheat_artifacts",
            ["source_artifact_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_foreign_key(
            "fk_distribution_policy_source_sync_task",
            "model_storage_sync_tasks",
            ["source_sync_task_id"],
            ["id"],
            ondelete="SET NULL",
        )

    op.execute(
        sa.text(
            "UPDATE model_preheat_distribution_policies SET source_artifact_id = ("
            "SELECT model_preheat_artifacts.id FROM model_preheat_artifacts "
            "WHERE model_preheat_artifacts.profile_id = "
            "model_preheat_distribution_policies.profile_id "
            "AND model_preheat_artifacts.profile_config_version = "
            "model_preheat_distribution_policies.profile_config_version "
            "AND model_preheat_artifacts.created_by_task_id = "
            "model_preheat_distribution_policies.created_by_task_id"
            ") WHERE created_by_task_id IS NOT NULL"
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("model_preheat_distribution_policies") as batch_op:
        batch_op.drop_constraint(
            "fk_distribution_policy_source_sync_task", type_="foreignkey"
        )
        batch_op.drop_constraint(
            "fk_distribution_policy_source_artifact", type_="foreignkey"
        )
        batch_op.drop_column("source_sync_task_id")
        batch_op.drop_column("source_artifact_id")
