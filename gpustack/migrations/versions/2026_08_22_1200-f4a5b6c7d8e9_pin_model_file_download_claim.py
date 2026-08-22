"""pin model file download claim

Revision ID: f4a5b6c7d8e9
Revises: e3f4a5b6c7d8
Create Date: 2026-08-22 12:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "f4a5b6c7d8e9"
down_revision: Union[str, None] = "e3f4a5b6c7d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_file_download_executions") as batch_op:
        batch_op.add_column(
            sa.Column("resolved_revision", sa.String(length=1024), nullable=True)
        )
        batch_op.add_column(
            sa.Column("artifact_id", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(sa.Column("manifest_path", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("artifact_total_size", sa.BigInteger(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("model_file_download_executions") as batch_op:
        batch_op.drop_column("artifact_total_size")
        batch_op.drop_column("manifest_path")
        batch_op.drop_column("artifact_id")
        batch_op.drop_column("resolved_revision")
