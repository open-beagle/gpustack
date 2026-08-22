"""remove preheat generation columns

Revision ID: b6c7d8e9f0a1
Revises: a5b6c7d8e9f0
Create Date: 2026-08-22 14:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "b6c7d8e9f0a1"
down_revision: Union[str, None] = "a5b6c7d8e9f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_preheat_tasks") as batch_op:
        batch_op.drop_column("s3_ready_path")
        batch_op.drop_column("generation_id")


def downgrade() -> None:
    with op.batch_alter_table("model_preheat_tasks") as batch_op:
        batch_op.add_column(
            sa.Column(
                "generation_id",
                sa.String(length=256),
                nullable=False,
                server_default="legacy",
            )
        )
        batch_op.add_column(
            sa.Column("s3_ready_path", sa.String(length=255), nullable=True)
        )
