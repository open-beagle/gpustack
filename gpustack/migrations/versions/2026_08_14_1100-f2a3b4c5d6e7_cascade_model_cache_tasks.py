"""cascade model cache tasks when deleting model files

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-08-14 11:00:00.000000
"""

from typing import Sequence, Union

from alembic import op


revision: str = "f2a3b4c5d6e7"
down_revision: Union[str, None] = "e1f2a3b4c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_naming_convention = {
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"
}


def upgrade() -> None:
    with op.batch_alter_table(
        "model_cache_tasks", naming_convention=_naming_convention
    ) as batch_op:
        batch_op.drop_constraint(
            "fk_model_cache_tasks_model_file_id_model_files", type_="foreignkey"
        )
        batch_op.create_foreign_key(
            "fk_model_cache_tasks_model_file_id_model_files",
            "model_files",
            ["model_file_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    with op.batch_alter_table(
        "model_cache_tasks", naming_convention=_naming_convention
    ) as batch_op:
        batch_op.drop_constraint(
            "fk_model_cache_tasks_model_file_id_model_files", type_="foreignkey"
        )
        batch_op.create_foreign_key(
            "fk_model_cache_tasks_model_file_id_model_files",
            "model_files",
            ["model_file_id"],
            ["id"],
            ondelete="RESTRICT",
        )
