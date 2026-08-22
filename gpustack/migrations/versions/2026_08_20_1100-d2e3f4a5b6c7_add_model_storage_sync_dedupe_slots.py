"""统一模型存储任务 3：活动同步任务数据库级去重槽

新增 ``model_storage_sync_task_dedupe_slots`` 表：同一
``(model_file_id, profile_id)`` 的活动同步任务（pending/scanning/publishing）
在同一事务中占用 ``dedupe_key`` 唯一槽位，终态时在同一事务中释放为 NULL。
``dedupe_key`` 唯一约束是三库（SQLite/PostgreSQL/MySQL）通用的数据库级并发
保证：并发创建时后到者得到唯一冲突并整体回滚，不产生重复任务或遗留任务。

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
Create Date: 2026-08-20 11:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


revision: str = "d2e3f4a5b6c7"
down_revision: Union[str, None] = "c1d2e3f4a5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_storage_sync_task_dedupe_slots",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("dedupe_key", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            UTCDateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            UTCDateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["task_id"], ["model_storage_sync_tasks.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "dedupe_key", name="uix_model_storage_sync_dedupe_key"
        ),
    )


def downgrade() -> None:
    op.drop_table("model_storage_sync_task_dedupe_slots")
