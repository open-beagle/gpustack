"""添加模型预热父任务尝试关联

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-08-11 11:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_preheat_tasks") as batch_op:
        batch_op.add_column(
            sa.Column("paused_from_state", sa.String(length=255), nullable=True)
        )

    with op.batch_alter_table("model_preheat_worker_tasks") as batch_op:
        batch_op.add_column(
            sa.Column("parent_attempt", sa.Integer(), nullable=True)
        )

    op.execute(
        sa.text(
            """
            UPDATE model_preheat_worker_tasks
            SET parent_attempt = COALESCE(
                (SELECT attempt FROM model_preheat_tasks
                 WHERE model_preheat_tasks.id = model_preheat_worker_tasks.task_id),
                1
            )
            """
        )
    )

    with op.batch_alter_table("model_preheat_worker_tasks") as batch_op:
        batch_op.alter_column(
            "parent_attempt",
            existing_type=sa.Integer(),
            nullable=False,
            server_default="1",
        )
        batch_op.drop_constraint(
            "uix_preheat_task_worker_role", type_="unique"
        )
        batch_op.create_unique_constraint(
            "uix_preheat_task_attempt_worker_role",
            ["task_id", "parent_attempt", "worker_uuid", "role"],
        )


def downgrade() -> None:
    with op.batch_alter_table("model_preheat_worker_tasks") as batch_op:
        batch_op.drop_constraint(
            "uix_preheat_task_attempt_worker_role", type_="unique"
        )
        batch_op.create_unique_constraint(
            "uix_preheat_task_worker_role", ["task_id", "worker_uuid", "role"]
        )
        batch_op.drop_column("parent_attempt")

    with op.batch_alter_table("model_preheat_tasks") as batch_op:
        batch_op.drop_column("paused_from_state")
