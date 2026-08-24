"""为 ModelFile 增加历史 Worker 身份快照

Revision ID: f0a1b2c3d4e5
Revises: e9f0a1b2c3d4
Create Date: 2026-08-24 10:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f0a1b2c3d4e5"
down_revision: Union[str, None] = "e9f0a1b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_files") as batch_op:
        batch_op.add_column(
            sa.Column("worker_uuid_snapshot", sa.String(length=255), nullable=True)
        )
        batch_op.add_column(
            sa.Column("worker_name_snapshot", sa.String(length=255), nullable=True)
        )

    op.execute(
        sa.text(
            "UPDATE model_files SET "
            "worker_uuid_snapshot = (SELECT workers.worker_uuid FROM workers "
            "WHERE workers.id = model_files.worker_id), "
            "worker_name_snapshot = (SELECT workers.name FROM workers "
            "WHERE workers.id = model_files.worker_id) "
            "WHERE EXISTS (SELECT 1 FROM workers "
            "WHERE workers.id = model_files.worker_id)"
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("model_files") as batch_op:
        batch_op.drop_column("worker_name_snapshot")
        batch_op.drop_column("worker_uuid_snapshot")
