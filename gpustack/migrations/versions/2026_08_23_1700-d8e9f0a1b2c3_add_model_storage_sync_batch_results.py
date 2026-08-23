"""持久化批量模型同步的幂等响应

Revision ID: d8e9f0a1b2c3
Revises: c7d8e9f0a1b2
Create Date: 2026-08-23 17:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d8e9f0a1b2c3"
down_revision: Union[str, None] = "c7d8e9f0a1b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_storage_sync_batch_results",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("idempotency_record_id", sa.Integer(), nullable=False),
        sa.Column("response_payload", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["idempotency_record_id"],
            ["model_preheat_idempotency_records.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "idempotency_record_id",
            name="uix_model_storage_sync_batch_result_idempotency",
        ),
    )


def downgrade() -> None:
    op.drop_table("model_storage_sync_batch_results")
