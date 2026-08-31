"""为旧 Worker 直覆盖升级增加一次 proof 窗口

Revision ID: f9b0c1d2e3f4
Revises: f8a9b0c1d2e3
Create Date: 2026-08-31 22:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


revision: str = "f9b0c1d2e3f4"
down_revision: Union[str, None] = "f8a9b0c1d2e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("model_preheat_worker_identities") as batch_op:
        batch_op.add_column(
            sa.Column("upgrade_proof_hash", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("upgrade_proof_window_started_at", UTCDateTime(), nullable=True)
        )
    identities = sa.table(
        "model_preheat_worker_identities",
        sa.column("id", sa.Integer()),
        sa.column("token_hash", sa.String(length=64)),
        sa.column("registration_recovery_token_hash", sa.String(length=64)),
        sa.column("bootstrap_required", sa.Boolean()),
        sa.column("revoked_at", UTCDateTime()),
        sa.column("upgrade_proof_window_started_at", UTCDateTime()),
    )
    pending = sa.table(
        "model_preheat_worker_pending_credentials",
        sa.column("identity_id", sa.Integer()),
    )
    op.execute(
        identities.update()
        .where(
            identities.c.bootstrap_required.is_(True),
            identities.c.token_hash.is_(None),
            identities.c.registration_recovery_token_hash.is_(None),
            identities.c.revoked_at.is_(None),
            ~sa.exists(
                sa.select(pending.c.identity_id).where(
                    pending.c.identity_id == identities.c.id
                )
            ),
        )
        .values(upgrade_proof_window_started_at=sa.func.now())
    )


def downgrade() -> None:
    with op.batch_alter_table("model_preheat_worker_identities") as batch_op:
        batch_op.drop_column("upgrade_proof_window_started_at")
        batch_op.drop_column("upgrade_proof_hash")
