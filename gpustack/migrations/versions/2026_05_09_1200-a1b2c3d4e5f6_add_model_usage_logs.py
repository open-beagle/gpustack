"""add model usage logs

Revision ID: a1b2c3d4e5f6
Revises: d19176de3b74
Create Date: 2026-05-09 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'd19176de3b74'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'model_usage_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('request_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('call_time', gpustack.schemas.common.UTCDateTime(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('hour', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('api_key_id', sa.Integer(), nullable=True),
        sa.Column('api_key_access_key', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('model_id', sa.Integer(), nullable=True),
        sa.Column('model_name', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column(
            'operation',
            sa.Enum(
                'COMPLETION',
                'CHAT_COMPLETION',
                'EMBEDDING',
                'RERANK',
                'IMAGE_GENERATION',
                'AUDIO_SPEECH',
                'AUDIO_TRANSCRIPTION',
                name='operationenum',
            ),
            nullable=False,
        ),
        sa.Column('source_ip', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('raw_forwarded_for', sa.Text(), nullable=True),
        sa.Column('prompt_token_count', sa.Integer(), nullable=False),
        sa.Column('completion_token_count', sa.Integer(), nullable=False),
        sa.Column('total_token_count', sa.Integer(), nullable=False),
        sa.Column('usage_available', sa.Boolean(), nullable=False),
        sa.Column('status_code', sa.Integer(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('ttft_ms', sa.Integer(), nullable=True),
        sa.Column('tokens_per_second', sa.Float(), nullable=True),
        sa.Column('error_code', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('error_type', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('worker_id', sa.Integer(), nullable=True),
        sa.Column('worker_name', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('worker_ip', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('model_instance_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['api_key_id'], ['api_keys.id']),
        sa.ForeignKeyConstraint(['model_id'], ['models.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_model_usage_logs_api_key_id'), 'model_usage_logs', ['api_key_id'], unique=False)
    op.create_index(op.f('ix_model_usage_logs_call_time'), 'model_usage_logs', ['call_time'], unique=False)
    op.create_index(op.f('ix_model_usage_logs_date'), 'model_usage_logs', ['date'], unique=False)
    op.create_index(op.f('ix_model_usage_logs_hour'), 'model_usage_logs', ['hour'], unique=False)
    op.create_index(op.f('ix_model_usage_logs_model_id'), 'model_usage_logs', ['model_id'], unique=False)
    op.create_index(op.f('ix_model_usage_logs_model_instance_id'), 'model_usage_logs', ['model_instance_id'], unique=False)
    op.create_index(op.f('ix_model_usage_logs_model_name'), 'model_usage_logs', ['model_name'], unique=False)
    op.create_index(op.f('ix_model_usage_logs_operation'), 'model_usage_logs', ['operation'], unique=False)
    op.create_index(op.f('ix_model_usage_logs_request_id'), 'model_usage_logs', ['request_id'], unique=False)
    op.create_index(op.f('ix_model_usage_logs_source_ip'), 'model_usage_logs', ['source_ip'], unique=False)
    op.create_index(op.f('ix_model_usage_logs_status_code'), 'model_usage_logs', ['status_code'], unique=False)
    op.create_index(op.f('ix_model_usage_logs_success'), 'model_usage_logs', ['success'], unique=False)
    op.create_index(op.f('ix_model_usage_logs_user_id'), 'model_usage_logs', ['user_id'], unique=False)
    op.create_index(op.f('ix_model_usage_logs_worker_id'), 'model_usage_logs', ['worker_id'], unique=False)
    op.create_index('ix_model_usage_logs_date_api_key_id', 'model_usage_logs', ['date', 'api_key_id'], unique=False)
    op.create_index('ix_model_usage_logs_date_hour', 'model_usage_logs', ['date', 'hour'], unique=False)
    op.create_index('ix_model_usage_logs_date_model_id', 'model_usage_logs', ['date', 'model_id'], unique=False)
    op.create_index('ix_model_usage_logs_date_source_ip', 'model_usage_logs', ['date', 'source_ip'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_model_usage_logs_date_source_ip', table_name='model_usage_logs')
    op.drop_index('ix_model_usage_logs_date_model_id', table_name='model_usage_logs')
    op.drop_index('ix_model_usage_logs_date_hour', table_name='model_usage_logs')
    op.drop_index('ix_model_usage_logs_date_api_key_id', table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_worker_id'), table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_user_id'), table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_success'), table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_status_code'), table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_source_ip'), table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_request_id'), table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_operation'), table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_model_name'), table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_model_instance_id'), table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_model_id'), table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_hour'), table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_date'), table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_call_time'), table_name='model_usage_logs')
    op.drop_index(op.f('ix_model_usage_logs_api_key_id'), table_name='model_usage_logs')
    op.drop_table('model_usage_logs')
