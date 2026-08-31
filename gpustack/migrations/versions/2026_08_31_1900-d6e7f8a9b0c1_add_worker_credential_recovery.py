"""为 Worker 凭据轮换增加仅注册恢复凭据

Revision ID: d6e7f8a9b0c1
Revises: c5d6e7f8a9b0
Create Date: 2026-08-31 19:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


revision: str = "d6e7f8a9b0c1"
down_revision: Union[str, None] = "c5d6e7f8a9b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_preheat_worker_identities") as batch_op:
        batch_op.add_column(
            sa.Column("registration_recovery_token_hash", sa.String(64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("registration_recovery_issued_at", UTCDateTime(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("model_preheat_worker_identities") as batch_op:
        batch_op.drop_column("registration_recovery_issued_at")
        batch_op.drop_column("registration_recovery_token_hash")
