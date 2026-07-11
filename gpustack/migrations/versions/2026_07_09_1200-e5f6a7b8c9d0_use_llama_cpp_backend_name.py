"""use llama.cpp backend name

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-07-09 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE models SET backend = 'llama.cpp' WHERE backend = 'llama-cpp'")


def downgrade() -> None:
    op.execute("UPDATE models SET backend = 'llama-cpp' WHERE backend = 'llama.cpp'")
