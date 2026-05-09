"""add model usage stats

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-05-09 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def operation_enum_type():
    values = (
        'COMPLETION',
        'CHAT_COMPLETION',
        'EMBEDDING',
        'RERANK',
        'IMAGE_GENERATION',
        'AUDIO_SPEECH',
        'AUDIO_TRANSCRIPTION',
    )
    if op.get_bind().dialect.name == 'postgresql':
        return postgresql.ENUM(*values, name='operationenum', create_type=False)
    return sa.Enum(*values, name='operationenum')


def create_stat_table(table_name: str, include_hour: bool):
    columns = [
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
    ]
    if include_hour:
        columns.append(sa.Column('hour', sa.Integer(), nullable=False))
    columns.extend(
        [
            sa.Column('api_key_id', sa.Integer(), nullable=False),
            sa.Column('api_key_access_key', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
            sa.Column('model_id', sa.Integer(), nullable=False),
            sa.Column('model_name', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
            sa.Column('source_ip', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column('operation', operation_enum_type(), nullable=False),
            sa.Column('worker_id', sa.Integer(), nullable=False),
            sa.Column('worker_name', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
            sa.Column('request_count', sa.Integer(), nullable=False),
            sa.Column('success_count', sa.Integer(), nullable=False),
            sa.Column('failure_count', sa.Integer(), nullable=False),
            sa.Column('prompt_token_count', sa.Integer(), nullable=False),
            sa.Column('completion_token_count', sa.Integer(), nullable=False),
            sa.Column('total_token_count', sa.Integer(), nullable=False),
            sa.Column('duration_ms_sum', sa.Integer(), nullable=False),
            sa.Column('last_call_time', gpustack.schemas.common.UTCDateTime(), nullable=False),
            sa.PrimaryKeyConstraint('id'),
        ]
    )
    op.create_table(table_name, *columns)


def create_stat_indexes(table_name: str, include_hour: bool):
    op.create_index(op.f(f'ix_{table_name}_api_key_id'), table_name, ['api_key_id'], unique=False)
    op.create_index(op.f(f'ix_{table_name}_date'), table_name, ['date'], unique=False)
    if include_hour:
        op.create_index(op.f(f'ix_{table_name}_hour'), table_name, ['hour'], unique=False)
    op.create_index(op.f(f'ix_{table_name}_last_call_time'), table_name, ['last_call_time'], unique=False)
    op.create_index(op.f(f'ix_{table_name}_model_id'), table_name, ['model_id'], unique=False)
    op.create_index(op.f(f'ix_{table_name}_model_name'), table_name, ['model_name'], unique=False)
    op.create_index(op.f(f'ix_{table_name}_operation'), table_name, ['operation'], unique=False)
    op.create_index(op.f(f'ix_{table_name}_source_ip'), table_name, ['source_ip'], unique=False)
    op.create_index(op.f(f'ix_{table_name}_worker_id'), table_name, ['worker_id'], unique=False)
    unique_columns = ['date']
    if include_hour:
        unique_columns.append('hour')
    unique_columns.extend(['api_key_id', 'model_id', 'source_ip', 'operation', 'worker_id'])
    op.create_index(f'uix_{table_name}_dimensions', table_name, unique_columns, unique=True)


def upgrade() -> None:
    create_stat_table('model_usage_hourly_stats', include_hour=True)
    create_stat_indexes('model_usage_hourly_stats', include_hour=True)
    create_stat_table('model_usage_daily_stats', include_hour=False)
    create_stat_indexes('model_usage_daily_stats', include_hour=False)


def drop_stat_table(table_name: str, include_hour: bool):
    op.drop_index(f'uix_{table_name}_dimensions', table_name=table_name)
    op.drop_index(op.f(f'ix_{table_name}_worker_id'), table_name=table_name)
    op.drop_index(op.f(f'ix_{table_name}_source_ip'), table_name=table_name)
    op.drop_index(op.f(f'ix_{table_name}_operation'), table_name=table_name)
    op.drop_index(op.f(f'ix_{table_name}_model_name'), table_name=table_name)
    op.drop_index(op.f(f'ix_{table_name}_model_id'), table_name=table_name)
    op.drop_index(op.f(f'ix_{table_name}_last_call_time'), table_name=table_name)
    if include_hour:
        op.drop_index(op.f(f'ix_{table_name}_hour'), table_name=table_name)
    op.drop_index(op.f(f'ix_{table_name}_date'), table_name=table_name)
    op.drop_index(op.f(f'ix_{table_name}_api_key_id'), table_name=table_name)
    op.drop_table(table_name)


def downgrade() -> None:
    drop_stat_table('model_usage_daily_stats', include_hour=False)
    drop_stat_table('model_usage_hourly_stats', include_hour=True)
