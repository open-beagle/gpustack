"""关联分发策略 Run 与实际 Worker 子任务

Revision ID: c5d6e7f8a9b0
Revises: b4c5d6e7f8a9
Create Date: 2026-08-29 22:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c5d6e7f8a9b0"
down_revision: Union[str, None] = "b4c5d6e7f8a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_preheat_distribution_policy_runs") as batch_op:
        batch_op.add_column(sa.Column("outcome", sa.JSON(), nullable=True))
    op.create_table(
        "model_preheat_distribution_policy_run_tasks",
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["model_preheat_distribution_policy_runs.id"],
            name="fk_distribution_run_task_run",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["model_preheat_worker_tasks.id"],
            name="fk_distribution_run_task_task",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("run_id", "task_id"),
    )
    op.create_index(
        "ix_distribution_run_task_task",
        "model_preheat_distribution_policy_run_tasks",
        ["task_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_distribution_run_task_task",
        table_name="model_preheat_distribution_policy_run_tasks",
    )
    op.drop_table("model_preheat_distribution_policy_run_tasks")
    with op.batch_alter_table("model_preheat_distribution_policy_runs") as batch_op:
        batch_op.drop_column("outcome")
