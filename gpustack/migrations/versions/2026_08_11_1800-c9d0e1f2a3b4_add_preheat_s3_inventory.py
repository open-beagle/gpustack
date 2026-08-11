"""添加模型预热 S3 库存和刷新作业

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-08-11 18:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


revision: str = "c9d0e1f2a3b4"
down_revision: Union[str, None] = "b8c9d0e1f2a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_preheat_cached_models",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("profile_config_version", sa.Integer(), nullable=False),
        sa.Column("cache_key", sa.String(length=256), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("model_id", sa.String(length=1024), nullable=False),
        sa.Column("resolved_revision", sa.String(length=1024), nullable=False),
        sa.Column("include_patterns", sa.JSON(), nullable=False),
        sa.Column("exclude_patterns", sa.JSON(), nullable=False),
        sa.Column("generation_id", sa.String(length=256), nullable=False),
        sa.Column("ready_path", sa.Text(), nullable=False),
        sa.Column("manifest_path", sa.Text(), nullable=False),
        sa.Column("manifest_digest", sa.String(length=64), nullable=False),
        sa.Column("file_count", sa.Integer(), nullable=False),
        sa.Column("total_size", sa.BigInteger(), nullable=False),
        sa.Column("manifest_state", sa.String(length=16), nullable=False),
        sa.Column("last_verified_at", UTCDateTime(), nullable=False),
        sa.Column("created_by_task_id", sa.Integer(), nullable=True),
        sa.Column("source_parent_attempt", sa.Integer(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["profile_id"], ["model_preheat_s3_profiles.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["created_by_task_id"], ["model_preheat_tasks.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint(
            "profile_id",
            "cache_key",
            name="uix_preheat_cached_model_profile_key",
        ),
    )
    op.create_index(
        "ix_preheat_cached_model_profile_state_key",
        "model_preheat_cached_models",
        ["profile_id", "manifest_state", "cache_key"],
    )
    op.create_table(
        "model_preheat_inventory_jobs",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("profile_config_version", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("active_key", sa.String(length=255), nullable=True),
        sa.Column("claim_token", sa.String(length=64), nullable=True),
        sa.Column("cursor", sa.JSON(), nullable=True),
        sa.Column("scanned_count", sa.Integer(), nullable=False),
        sa.Column("valid_count", sa.Integer(), nullable=False),
        sa.Column("invalid_count", sa.Integer(), nullable=False),
        sa.Column("orphan_count", sa.Integer(), nullable=False),
        sa.Column("deleted_count", sa.Integer(), nullable=False),
        sa.Column("skipped_count", sa.Integer(), nullable=False),
        sa.Column("failed_count", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", UTCDateTime(), nullable=True),
        sa.Column("lease_expires_at", UTCDateTime(), nullable=True),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["profile_id"], ["model_preheat_s3_profiles.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("active_key", name="uix_preheat_inventory_job_active"),
    )
    op.create_index(
        "ix_preheat_inventory_job_profile_created",
        "model_preheat_inventory_jobs",
        ["profile_id", "created_at"],
    )
    op.create_table(
        "model_preheat_inventory_generations",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("generation_key", sa.String(length=64), nullable=False),
        sa.Column("selection_key", sa.String(length=256), nullable=False),
        sa.Column("cache_key", sa.String(length=256), nullable=True),
        sa.Column("generation_path", sa.Text(), nullable=False),
        sa.Column("ready_path", sa.Text(), nullable=False),
        sa.Column("ready_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("ready_generation_path", sa.Text(), nullable=True),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("first_seen_at", UTCDateTime(), nullable=False),
        sa.Column("last_seen_at", UTCDateTime(), nullable=False),
        sa.Column("orphaned_at", UTCDateTime(), nullable=True),
        sa.Column("deleted_at_s3", UTCDateTime(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["profile_id"], ["model_preheat_s3_profiles.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "profile_id",
            "generation_key",
            name="uix_preheat_inventory_generation_path",
        ),
    )
    op.create_index(
        "ix_preheat_inventory_generation_gc",
        "model_preheat_inventory_generations",
        ["profile_id", "state", "orphaned_at"],
    )
    op.create_table(
        "model_preheat_inventory_selection_locks",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("selection_key", sa.String(length=256), nullable=False),
        sa.Column("owner_token", sa.String(length=64), nullable=False),
        sa.Column("operation", sa.String(length=16), nullable=False),
        sa.Column("lease_expires_at", UTCDateTime(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["profile_id"], ["model_preheat_s3_profiles.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "profile_id",
            "selection_key",
            name="uix_preheat_inventory_selection_lock",
        ),
    )
    op.create_table(
        "model_preheat_publication_markers",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("selection_key", sa.String(length=256), nullable=False),
        sa.Column("generation_id", sa.String(length=256), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("parent_attempt", sa.Integer(), nullable=False),
        sa.Column("profile_config_version", sa.Integer(), nullable=False),
        sa.Column("terminated_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["profile_id"], ["model_preheat_s3_profiles.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["task_id"], ["model_preheat_tasks.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint(
            "profile_id",
            "selection_key",
            "generation_id",
            name="uix_preheat_publication_marker_generation",
        ),
    )
    op.create_index(
        "ix_preheat_publication_marker_task_attempt",
        "model_preheat_publication_markers",
        ["task_id", "parent_attempt"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_preheat_publication_marker_task_attempt",
        table_name="model_preheat_publication_markers",
    )
    op.drop_table("model_preheat_publication_markers")
    op.drop_table("model_preheat_inventory_selection_locks")
    op.drop_index(
        "ix_preheat_inventory_generation_gc",
        table_name="model_preheat_inventory_generations",
    )
    op.drop_table("model_preheat_inventory_generations")
    op.drop_index(
        "ix_preheat_inventory_job_profile_created",
        table_name="model_preheat_inventory_jobs",
    )
    op.drop_table("model_preheat_inventory_jobs")
    op.drop_index(
        "ix_preheat_cached_model_profile_state_key",
        table_name="model_preheat_cached_models",
    )
    op.drop_table("model_preheat_cached_models")
