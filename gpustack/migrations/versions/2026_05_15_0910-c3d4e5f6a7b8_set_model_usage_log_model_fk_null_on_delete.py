"""set model usage log model fk null on delete

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-05-15 09:10:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

naming_convention = {
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
}


def upgrade() -> None:
    with op.batch_alter_table(
        'model_usage_logs', naming_convention=naming_convention
    ) as batch_op:
        batch_op.drop_constraint(
            'fk_model_usage_logs_model_id_models', type_='foreignkey'
        )
        batch_op.create_foreign_key(
            'fk_model_usage_logs_model_id_models',
            'models',
            ['model_id'],
            ['id'],
            ondelete='SET NULL',
        )


def downgrade() -> None:
    with op.batch_alter_table(
        'model_usage_logs', naming_convention=naming_convention
    ) as batch_op:
        batch_op.drop_constraint(
            'fk_model_usage_logs_model_id_models', type_='foreignkey'
        )
        batch_op.create_foreign_key(
            'fk_model_usage_logs_model_id_models',
            'models',
            ['model_id'],
            ['id'],
        )
