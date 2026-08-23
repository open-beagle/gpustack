"""为模型预热 Schedule 增加显式触发模式

Revision ID: e9f0a1b2c3d4
Revises: d8e9f0a1b2c3
Create Date: 2026-08-23 18:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e9f0a1b2c3d4"
down_revision: Union[str, None] = "d8e9f0a1b2c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_preheat_schedules") as batch_op:
        batch_op.add_column(
            sa.Column(
                "trigger_mode",
                sa.String(length=32),
                nullable=False,
                server_default="scheduled",
            )
        )
        batch_op.alter_column(
            "cron_expression",
            existing_type=sa.String(length=255),
            nullable=True,
        )


def downgrade() -> None:
    # 旧版本没有手动模式；为保证结构可逆，只在 downgrade 时为手动记录补一个
    # 合法 Cron。前向运行从不使用该值模拟“手动”。
    op.execute(
        sa.text(
            "UPDATE model_preheat_schedules "
            "SET cron_expression = '0 0 * * *' "
            "WHERE cron_expression IS NULL"
        )
    )
    with op.batch_alter_table("model_preheat_schedules") as batch_op:
        batch_op.alter_column(
            "cron_expression",
            existing_type=sa.String(length=255),
            nullable=False,
        )
        batch_op.drop_column("trigger_mode")
