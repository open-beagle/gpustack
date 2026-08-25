"""add scheduler policy and immutable scheduling attempt events

Revision ID: a3b4c5d6e7f8
Revises: 5d6e7f8a9b0
Create Date: 2026-08-25 10:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a3b4c5d6e7f8"
down_revision: Union[str, None] = "5d6e7f8a9b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "scheduler_policies",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("code", sa.String(64), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("algorithm", sa.String(64), nullable=False),
        sa.Column("aggregation_rate", sa.Float(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("runtime_revision", sa.BigInteger(), nullable=False),
        sa.Column("updated_by", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint("aggregation_rate > 0 AND aggregation_rate <= 100", name="ck_scheduler_policy_rate"),
    )
    op.create_table(
        "scheduling_attempt_events",
        sa.Column("event_id", sa.String(36), primary_key=True),
        sa.Column("workload_id", sa.String(255), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("policy_code", sa.String(64), nullable=False),
        sa.Column("policy_revision", sa.BigInteger(), nullable=False),
        sa.Column("requested_replicas", sa.Integer(), nullable=False),
        sa.Column("requested_resources", sa.JSON(), nullable=False),
        sa.Column("candidate_targets", sa.JSON(), nullable=False),
        sa.Column("selected_targets", sa.JSON(), nullable=False),
        sa.Column("outcome", sa.String(16), nullable=False),
        sa.Column("reason_code", sa.String(128), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("latency_ms", sa.BigInteger(), nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column("occurred_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("workload_id", "attempt_no", name="uix_scheduling_attempt_workload_attempt"),
        sa.CheckConstraint("outcome IN ('success', 'failed')", name="ck_scheduling_attempt_outcome"),
        sa.CheckConstraint("latency_ms >= 0", name="ck_scheduling_attempt_latency"),
    )
    op.create_index("ix_scheduling_attempt_policy_occurred", "scheduling_attempt_events", ["policy_code", "occurred_at"])
    op.create_index("ix_scheduling_attempt_workload", "scheduling_attempt_events", ["workload_id"])
    op.execute(
        "INSERT INTO scheduler_policies "
        "(code, name, algorithm, aggregation_rate, enabled, runtime_revision, updated_by, created_at, updated_at) "
        "VALUES ('aggregation', 'Aggregation priority scheduling policy', 'MostAllocated', 100, TRUE, 1, 'system', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
    )


def downgrade() -> None:
    op.drop_index("ix_scheduling_attempt_workload", table_name="scheduling_attempt_events")
    op.drop_index("ix_scheduling_attempt_policy_occurred", table_name="scheduling_attempt_events")
    op.drop_table("scheduling_attempt_events")
    op.drop_table("scheduler_policies")
