"""为分发策略增加 Artifact 集合选择与不可变子任务绑定

Revision ID: b4c5d6e7f8a9
Revises: a3b4c5d6e7f8
Create Date: 2026-08-29 21:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b4c5d6e7f8a9"
down_revision: Union[str, None] = "a3b4c5d6e7f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_preheat_distribution_policies") as batch_op:
        batch_op.add_column(
            sa.Column(
                "selection_mode",
                sa.String(32),
                nullable=False,
                server_default="fixed",
            )
        )

    op.create_table(
        "model_preheat_distribution_policy_artifacts",
        sa.Column("policy_id", sa.Integer(), nullable=False),
        sa.Column("artifact_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["policy_id"],
            ["model_preheat_distribution_policies.id"],
            name="fk_distribution_selected_policy",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["artifact_id"],
            ["model_preheat_artifacts.id"],
            name="fk_distribution_selected_artifact",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("policy_id", "artifact_id"),
        sa.UniqueConstraint(
            "policy_id",
            "artifact_id",
            name="uix_distribution_policy_selected_artifact",
        ),
    )

    with op.batch_alter_table("model_preheat_worker_tasks") as batch_op:
        batch_op.add_column(
            sa.Column("distribution_artifact_id", sa.String(64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("distribution_request_digest", sa.String(64), nullable=True)
        )
    op.execute(
        sa.text(
            "UPDATE model_preheat_worker_tasks SET distribution_artifact_id = ("
            "SELECT model_preheat_artifacts.artifact_id FROM model_preheat_artifacts "
            "JOIN model_preheat_distribution_policies ON "
            "model_preheat_distribution_policies.source_artifact_id = model_preheat_artifacts.id "
            "WHERE model_preheat_distribution_policies.id = "
            "model_preheat_worker_tasks.distribution_policy_id), "
            "distribution_request_digest = ("
            "SELECT request_digest FROM model_preheat_distribution_policies "
            "WHERE model_preheat_distribution_policies.id = "
            "model_preheat_worker_tasks.distribution_policy_id) "
            "WHERE distribution_policy_id IS NOT NULL"
        )
    )

    with op.batch_alter_table("model_preheat_distribution_worker_slots") as batch_op:
        batch_op.add_column(sa.Column("artifact_id", sa.String(64), nullable=True))
    op.execute(
        sa.text(
            "UPDATE model_preheat_distribution_worker_slots SET artifact_id = COALESCE(("
            "SELECT model_preheat_artifacts.artifact_id FROM model_preheat_artifacts "
            "JOIN model_preheat_distribution_policies ON "
            "model_preheat_distribution_policies.source_artifact_id = model_preheat_artifacts.id "
            "WHERE model_preheat_distribution_policies.id = "
            "model_preheat_distribution_worker_slots.policy_id), ("
            "SELECT request_digest FROM model_preheat_distribution_policies "
            "WHERE model_preheat_distribution_policies.id = "
            "model_preheat_distribution_worker_slots.policy_id))"
        )
    )
    with op.batch_alter_table("model_preheat_distribution_worker_slots") as batch_op:
        batch_op.drop_constraint(
            "uix_distribution_policy_worker_slot", type_="unique"
        )
        batch_op.alter_column(
            "artifact_id", existing_type=sa.String(64), nullable=False
        )
        batch_op.create_unique_constraint(
            "uix_distribution_policy_artifact_worker_slot",
            ["policy_id", "artifact_id", "worker_uuid"],
        )


def downgrade() -> None:
    op.execute(
        sa.text(
            "DELETE FROM model_preheat_distribution_worker_slots WHERE id NOT IN ("
            "SELECT kept.id FROM (SELECT MIN(id) AS id FROM "
            "model_preheat_distribution_worker_slots GROUP BY policy_id, worker_uuid) "
            "AS kept)"
        )
    )
    with op.batch_alter_table("model_preheat_distribution_worker_slots") as batch_op:
        batch_op.drop_constraint(
            "uix_distribution_policy_artifact_worker_slot", type_="unique"
        )
        batch_op.drop_column("artifact_id")
        batch_op.create_unique_constraint(
            "uix_distribution_policy_worker_slot", ["policy_id", "worker_uuid"]
        )
    with op.batch_alter_table("model_preheat_worker_tasks") as batch_op:
        batch_op.drop_column("distribution_request_digest")
        batch_op.drop_column("distribution_artifact_id")
    op.drop_table("model_preheat_distribution_policy_artifacts")
    with op.batch_alter_table("model_preheat_distribution_policies") as batch_op:
        batch_op.drop_column("selection_mode")
