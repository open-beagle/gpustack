"""统一模型存储任务 3 Review 修复：同步任务执行 lease 与完成结果固定字段

在 ``model_storage_sync_tasks`` 上新增三列（全部可空，不改历史数据语义）：

- ``lease_token_encrypted``：一次性执行 lease token 的 AES-GCM 加密快照
  （JSON）。明文只进入受 Worker 身份约束的执行 payload；complete/fail 必须
  回传该 lease，Server 解密快照后做恒定时间比较。lease 不发明文入库。
- ``manifest_digest`` / ``manifest_path``：任务完成时固定的发布结果
  （Manifest SHA-256 与对象 Key），等价重放幂等判定使用：同一已完成执行的
  重放必须与这些固定值一致，不同 artifact/过期执行稳定冲突。

历史任务（三列为 NULL）在 complete/fail 时因 lease 快照缺失被稳定拒绝
（拒绝优于误放行）；重放语义只对创建时即带 lease 的任务成立。

Revision ID: e3f4a5b6c7d8
Revises: d2e3f4a5b6c7
Create Date: 2026-08-22 10:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e3f4a5b6c7d8"
down_revision: Union[str, None] = "d2e3f4a5b6c7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_storage_sync_tasks",
        sa.Column("lease_token_encrypted", sa.JSON(), nullable=True),
    )
    op.add_column(
        "model_storage_sync_tasks",
        sa.Column("manifest_digest", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "model_storage_sync_tasks",
        sa.Column("manifest_path", sa.String(length=1024), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_storage_sync_tasks", "manifest_path")
    op.drop_column("model_storage_sync_tasks", "manifest_digest")
    op.drop_column("model_storage_sync_tasks", "lease_token_encrypted")
