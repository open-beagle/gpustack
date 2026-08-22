"""protect download execution profile

Revision ID: a5b6c7d8e9f0
Revises: f4a5b6c7d8e9
Create Date: 2026-08-22 13:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "a5b6c7d8e9f0"
down_revision: Union[str, None] = "f4a5b6c7d8e9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_file_download_execution_profile_pins",
        sa.Column("execution_id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["execution_id"],
            ["model_file_download_executions.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["profile_id"],
            ["model_preheat_s3_profiles.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("execution_id"),
    )
    op.create_index(
        "ix_model_file_download_execution_profile_pins_profile_id",
        "model_file_download_execution_profile_pins",
        ["profile_id"],
        unique=False,
    )
    op.execute(
        sa.text(
            "INSERT INTO model_file_download_execution_profile_pins "
            "(execution_id, profile_id) "
            "SELECT id, default_profile_id FROM model_file_download_executions "
            "WHERE default_profile_id IS NOT NULL"
        )
    )


def downgrade() -> None:
    op.drop_index(
        "ix_model_file_download_execution_profile_pins_profile_id",
        table_name="model_file_download_execution_profile_pins",
    )
    op.drop_table("model_file_download_execution_profile_pins")
