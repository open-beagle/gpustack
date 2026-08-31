"""为 Worker 凭据轮换增加待确认凭据

Revision ID: e7f8a9b0c1d2
Revises: d6e7f8a9b0c1
Create Date: 2026-08-31 20:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


revision: str = "e7f8a9b0c1d2"
down_revision: Union[str, None] = "d6e7f8a9b0c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_preheat_worker_pending_credentials",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("identity_id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", UTCDateTime(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["identity_id"],
            ["model_preheat_worker_identities.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "identity_id",
            "token_hash",
            name="uix_preheat_worker_pending_credential_token",
        ),
    )
    op.create_index(
        "ix_preheat_worker_pending_credential_identity",
        "model_preheat_worker_pending_credentials",
        ["identity_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_preheat_worker_pending_credential_identity",
        table_name="model_preheat_worker_pending_credentials",
    )
    op.drop_table("model_preheat_worker_pending_credentials")
