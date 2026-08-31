"""为待确认 Worker 凭据增加签发代次

Revision ID: f8a9b0c1d2e3
Revises: e7f8a9b0c1d2
Create Date: 2026-08-31 21:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f8a9b0c1d2e3"
down_revision: Union[str, None] = "e7f8a9b0c1d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # e7 已产生但尚未确认的候选没有可信签发代次。以 -1 回填使其不能
    # 匹配现有 identity.token_version，管理员或恢复凭据可安全重新签发。
    with op.batch_alter_table("model_preheat_worker_pending_credentials") as batch_op:
        batch_op.add_column(
            sa.Column(
                "identity_token_version",
                sa.Integer(),
                nullable=False,
                server_default="-1",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("model_preheat_worker_pending_credentials") as batch_op:
        batch_op.drop_column("identity_token_version")
