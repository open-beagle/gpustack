"""添加模型预热 worker 身份和扫描版本保护

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-08-11 21:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


revision: str = "d0e1f2a3b4c5"
down_revision: Union[str, None] = "c9d0e1f2a3b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_preheat_worker_uuid_state",
        "model_preheat_worker_tasks",
        ["worker_uuid", "state"],
        unique=False,
    )
    op.add_column(
        "model_preheat_inventory_jobs",
        sa.Column("scan_started_at", UTCDateTime(), nullable=True),
    )
    op.add_column(
        "model_preheat_cached_models",
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
    )
    op.create_table(
        "model_preheat_worker_identities",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("worker_id", sa.Integer(), nullable=False),
        sa.Column("worker_uuid", sa.String(length=256), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=True),
        sa.Column("token_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "bootstrap_required", sa.Boolean(), nullable=False, server_default=sa.true()
        ),
        sa.Column("expires_at", UTCDateTime(), nullable=True),
        sa.Column("revoked_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(["worker_id"], ["workers.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "worker_id", name="uix_preheat_worker_identity_worker"
        ),
    )
    op.create_index(
        "ix_preheat_worker_identity_uuid",
        "model_preheat_worker_identities",
        ["worker_uuid"],
        unique=False,
    )
    op.execute(_bootstrap_existing_workers_statement())
    op.create_table(
        "model_preheat_inventory_scan_snapshots",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("cached_model_id", sa.Integer(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"], ["model_preheat_inventory_jobs.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["cached_model_id"],
            ["model_preheat_cached_models.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "job_id",
            "cached_model_id",
            name="uix_preheat_inventory_scan_snapshot_row",
        ),
    )
    op.create_index(
        "ix_preheat_inventory_scan_snapshot_job",
        "model_preheat_inventory_scan_snapshots",
        ["job_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_preheat_inventory_scan_snapshot_job",
        table_name="model_preheat_inventory_scan_snapshots",
    )
    op.drop_table("model_preheat_inventory_scan_snapshots")
    op.drop_index(
        "ix_preheat_worker_identity_uuid",
        table_name="model_preheat_worker_identities",
    )
    op.drop_table("model_preheat_worker_identities")
    op.drop_column("model_preheat_cached_models", "revision")
    op.drop_column("model_preheat_inventory_jobs", "scan_started_at")
    op.drop_index(
        "ix_preheat_worker_uuid_state",
        table_name="model_preheat_worker_tasks",
    )


def _bootstrap_existing_workers_statement():
    workers = sa.table(
        "workers",
        sa.column("id", sa.Integer()),
        sa.column("worker_uuid", sa.String(length=256)),
    )
    identities = sa.table(
        "model_preheat_worker_identities",
        sa.column("worker_id", sa.Integer()),
        sa.column("worker_uuid", sa.String(length=256)),
        sa.column("token_hash", sa.String(length=64)),
        sa.column("token_version", sa.Integer()),
        sa.column("bootstrap_required", sa.Boolean()),
        sa.column("expires_at", UTCDateTime()),
        sa.column("revoked_at", UTCDateTime()),
        sa.column("created_at", UTCDateTime()),
        sa.column("updated_at", UTCDateTime()),
    )
    columns = [
        "worker_id",
        "worker_uuid",
        "token_hash",
        "token_version",
        "bootstrap_required",
        "expires_at",
        "revoked_at",
        "created_at",
        "updated_at",
    ]
    values = sa.select(
        workers.c.id,
        workers.c.worker_uuid,
        sa.null(),
        sa.literal(0),
        sa.true(),
        sa.null(),
        sa.null(),
        sa.func.now(),
        sa.func.now(),
    )
    return sa.insert(identities).from_select(columns, values)
