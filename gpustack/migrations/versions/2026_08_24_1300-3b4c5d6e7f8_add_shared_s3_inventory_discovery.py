"""为共享 S3 库存发现增加刷新状态

Revision ID: 3b4c5d6e7f8
Revises: 2a3b4c5d6e7f
Create Date: 2026-08-24 13:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


revision: str = "3b4c5d6e7f8"
down_revision: Union[str, None] = "2a3b4c5d6e7f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table(
        "model_preheat_s3_profiles", recreate="never"
    ) as batch_op:
        batch_op.add_column(
            sa.Column("inventory_refresh_interval_seconds", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("inventory_last_attempt_at", UTCDateTime(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("inventory_last_success_at", UTCDateTime(), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "inventory_last_scan_count",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch_op.add_column(
            sa.Column("inventory_last_error_code", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("inventory_refresh_owner", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("inventory_refresh_config_version", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("inventory_refresh_lease_expires_at", UTCDateTime(), nullable=True)
        )
    op.create_index(
        "ix_preheat_artifact_profile_version_source",
        "model_preheat_artifacts",
        ["profile_id", "profile_config_version", "source", "artifact_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_preheat_artifact_profile_version_source",
        table_name="model_preheat_artifacts",
    )
    with op.batch_alter_table(
        "model_preheat_s3_profiles", recreate="never"
    ) as batch_op:
        batch_op.drop_column("inventory_refresh_lease_expires_at")
        batch_op.drop_column("inventory_refresh_config_version")
        batch_op.drop_column("inventory_refresh_owner")
        batch_op.drop_column("inventory_last_error_code")
        batch_op.drop_column("inventory_last_scan_count")
        batch_op.drop_column("inventory_last_success_at")
        batch_op.drop_column("inventory_last_attempt_at")
        batch_op.drop_column("inventory_refresh_interval_seconds")
