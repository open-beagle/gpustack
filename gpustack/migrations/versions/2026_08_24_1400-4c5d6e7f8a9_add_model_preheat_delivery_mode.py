"""为预热任务和计划增加交付模式

Revision ID: 4c5d6e7f8a9
Revises: 3b4c5d6e7f8
Create Date: 2026-08-24 14:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "4c5d6e7f8a9"
down_revision: Union[str, None] = "3b4c5d6e7f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table in ("model_preheat_tasks", "model_preheat_schedules"):
        with op.batch_alter_table(table, recreate="never") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "delivery_mode",
                    sa.String(length=32),
                    nullable=False,
                    server_default=sa.text("'s3_and_workers'"),
                )
            )
            batch_op.add_column(
                sa.Column(
                    "connectivity_failure_override",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.false(),
                )
            )


def downgrade() -> None:
    for table in ("model_preheat_schedules", "model_preheat_tasks"):
        with op.batch_alter_table(table, recreate="never") as batch_op:
            batch_op.drop_column("connectivity_failure_override")
            batch_op.drop_column("delivery_mode")
