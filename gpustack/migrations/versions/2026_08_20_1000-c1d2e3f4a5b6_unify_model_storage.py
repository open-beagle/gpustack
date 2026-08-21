"""收敛模型存储数据与统一 Artifact

Revision ID: c1d2e3f4a5b6
Revises: b0307846729c
Create Date: 2026-08-20 10:00:00.000000
"""

import hashlib
import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime
from gpustack.worker.model_preheat.identity import (
    encode_path,
    normalize_source,
)


revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, None] = "b0307846729c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _request_identity(
    source, model_id, requested_revision, include_patterns, exclude_patterns
) -> dict:
    # 存量任务由旧 ``ModelPreheatCreate`` 入库，其 validator 已对
    # include/exclude_patterns 做 ``encode_path`` + 排序（存量为**已编码**值），
    # 而 source 已规范化、model_id 与 requested_revision 保留**原始值**（仅校验
    # 可编码，不编码存储）。因此回填时必须：source 规范化、model_id 与
    # requested_revision 按运行时 ``encode_path`` 编码，但对 patterns **不得二次
    # 编码**（存量已是编码值，二次编码会把 ``%20`` 变成 ``%2520``），仅排序去重。
    # 结果与运行时 ``ModelPreheatIdentity``（对同一**原始**请求）的
    # ``request_digest`` 完全一致。
    return {
        "source": normalize_source(source),
        "model_id": encode_path(model_id),
        "requested_revision": (
            encode_path(requested_revision) if requested_revision else None
        ),
        "include_patterns": sorted(include_patterns or []),
        "exclude_patterns": sorted(exclude_patterns or []),
    }


def _request_digest(identity: dict) -> str:
    # 与运行时 ``_canonical_sha256`` 完全一致：ensure_ascii=False、sort_keys、
    # 紧凑分隔符后的 SHA-256。身份 dict 已按运行时规范（source 规范化、
    # model_id/revision 编码、patterns 存量编码值排序）构造，等价于
    # ``ModelPreheatIdentity`` 对同一原始请求的 ``request_digest``。
    payload = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def upgrade() -> None:
    _upgrade_profiles()
    _upgrade_model_files()
    _upgrade_workers()
    _upgrade_tasks()
    _upgrade_policies()
    _drop_old_tables()
    _create_new_tables()


def _upgrade_profiles() -> None:
    with op.batch_alter_table("model_preheat_s3_profiles") as batch_op:
        batch_op.add_column(
            sa.Column(
                "provisioning_source",
                sa.String(length=32),
                nullable=False,
                server_default="manual",
            )
        )
        batch_op.add_column(sa.Column("provisioning_key", sa.String(length=255), nullable=True))
        batch_op.add_column(
            sa.Column(
                "system_managed", sa.Boolean(), nullable=False, server_default=sa.false()
            )
        )
        batch_op.add_column(sa.Column("default_slot", sa.String(length=32), nullable=True))
        batch_op.add_column(
            sa.Column(
                "source_fallback_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )
        # 持久层删除独立 is_default；Public API 由 default_slot 派生。
        batch_op.drop_column("is_default")
        # 普通唯一约束保证 SQLite/PostgreSQL/MySQL 最多一个系统 Profile 与一个默认 Profile；
        # 多个 NULL 允许并存。
        batch_op.create_unique_constraint(
            "uix_model_preheat_s3_profiles_provisioning_key", ["provisioning_key"]
        )
        batch_op.create_unique_constraint(
            "uix_model_preheat_s3_profiles_default_slot", ["default_slot"]
        )


def _upgrade_model_files() -> None:
    op.add_column(
        "model_files", sa.Column("requested_revision", sa.String(length=1024), nullable=True)
    )
    op.add_column(
        "model_files", sa.Column("resolved_revision", sa.String(length=1024), nullable=True)
    )


def _upgrade_workers() -> None:
    op.add_column(
        "workers",
        sa.Column(
            "model_storage_protocol_version",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def _upgrade_tasks() -> None:
    with op.batch_alter_table("model_preheat_tasks") as batch_op:
        # 先以可空加入，回填后收敛为不可空。
        batch_op.add_column(
            sa.Column("request_identity", sa.JSON(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("request_digest", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("artifact_id", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("transfer_source", sa.String(length=32), nullable=True)
        )
        batch_op.add_column(
            sa.Column("transfer_profile_id", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("source_worker_id", sa.Integer(), nullable=True)
        )
        batch_op.drop_column("cache_key")
    _backfill_tasks_request_identity()
    with op.batch_alter_table("model_preheat_tasks") as batch_op:
        batch_op.alter_column("request_identity", existing_type=sa.JSON(), nullable=False)
        batch_op.alter_column(
            "request_digest",
            existing_type=sa.String(length=64),
            nullable=False,
        )


def _upgrade_policies() -> None:
    with op.batch_alter_table(
        "model_preheat_distribution_policies"
    ) as batch_op:
        batch_op.add_column(sa.Column("request_identity", sa.JSON(), nullable=True))
        batch_op.add_column(
            sa.Column("request_digest", sa.String(length=64), nullable=True)
        )
    # 先回填 request_digest（读取旧 cache_key 生成每行唯一且确定的摘要），
    # 再删除 cache_key 并重建唯一约束。旧唯一约束为
    # (profile_id, cache_key, target_scope, selector_digest)，因此把 cache_key 纳入
    # 摘要可保证新约束 (profile_id, request_digest, target_scope, selector_digest)
    # 在同一 (profile_id, target_scope, selector_digest) 组内仍然每行唯一。
    _backfill_policies_request_identity()
    with op.batch_alter_table(
        "model_preheat_distribution_policies"
    ) as batch_op:
        batch_op.drop_column("cache_key")
        batch_op.drop_constraint(
            "uix_preheat_distribution_policy_selector", type_="unique"
        )
        batch_op.create_unique_constraint(
            "uix_preheat_distribution_policy_selector",
            ["profile_id", "request_digest", "target_scope", "selector_digest"],
        )
    with op.batch_alter_table(
        "model_preheat_distribution_policies"
    ) as batch_op:
        batch_op.alter_column("request_identity", existing_type=sa.JSON(), nullable=False)
        batch_op.alter_column(
            "request_digest",
            existing_type=sa.String(length=64),
            nullable=False,
        )


def _backfill_tasks_request_identity() -> None:
    bind = op.get_bind()
    table = sa.table(
        "model_preheat_tasks",
        sa.column("id", sa.Integer()),
        sa.column("source", sa.String()),
        sa.column("model_id", sa.String()),
        sa.column("requested_revision", sa.String()),
        sa.column("include_patterns", sa.JSON()),
        sa.column("exclude_patterns", sa.JSON()),
        sa.column("request_identity", sa.JSON()),
        sa.column("request_digest", sa.String(length=64)),
    )

    rows = bind.execute(
        table.select().where(
            sa.or_(
                table.c.request_identity.is_(None),
                table.c.request_digest.is_(None),
            )
        )
    ).all()
    for row in rows:
        identity = _request_identity(
            row.source,
            row.model_id,
            row.requested_revision,
            row.include_patterns,
            row.exclude_patterns,
        )
        bind.execute(
            table.update()
            .where(table.c.id == row.id)
            .values(
                request_identity=identity,
                request_digest=_request_digest(identity),
            )
        )


def _backfill_policies_request_identity() -> None:
    bind = op.get_bind()
    table = sa.table(
        "model_preheat_distribution_policies",
        sa.column("id", sa.Integer()),
        sa.column("cache_key", sa.String()),
        sa.column("request_identity", sa.JSON()),
        sa.column("request_digest", sa.String(length=64)),
    )
    rows = bind.execute(
        table.select()
        .where(
            sa.or_(
                table.c.request_identity.is_(None),
                table.c.request_digest.is_(None),
            )
        )
    ).all()
    for row in rows:
        # 旧策略仅有 cache_key，无法重建规范 request identity，故身份写空对象；
        # request_digest 必须“每行唯一且确定”：把旧 cache_key 与稳定行 id 纳入
        # 规范化摘要，既保持同一 (profile_id, target_scope, selector_digest) 组内
        # 每行唯一（等价旧唯一约束），又对同一行确定可复现。
        digest = hashlib.sha256(
            json.dumps(
                {
                    "legacy": "model_preheat_distribution_policy",
                    "row_id": row.id,
                    "cache_key": row.cache_key or "",
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        bind.execute(
            table.update()
            .where(table.c.id == row.id)
            .values(request_identity={}, request_digest=digest)
        )


def _drop_old_tables() -> None:
    # 按依赖顺序删除旧 cache / inventory / generation / selection lock / publication marker。
    op.drop_index(
        "ix_preheat_inventory_scan_snapshot_job",
        table_name="model_preheat_inventory_scan_snapshots",
    )
    op.drop_table("model_preheat_inventory_scan_snapshots")
    op.drop_index(
        "ix_preheat_publication_marker_task_attempt",
        table_name="model_preheat_publication_markers",
    )
    op.drop_table("model_preheat_publication_markers")
    op.drop_index(
        "ix_preheat_inventory_generation_gc",
        table_name="model_preheat_inventory_generations",
    )
    op.drop_table("model_preheat_inventory_generations")
    op.drop_table("model_preheat_inventory_selection_locks")
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
    op.drop_table("model_preheat_publish_locks")
    op.drop_index("ix_model_cache_tasks_model_id", table_name="model_cache_tasks")
    op.drop_table("model_cache_tasks")


def _create_new_tables() -> None:
    op.create_table(
        "model_preheat_artifacts",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("profile_config_version", sa.Integer(), nullable=False),
        sa.Column("artifact_id", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("model_id", sa.String(length=1024), nullable=False),
        sa.Column("resolved_revision", sa.String(length=1024), nullable=False),
        sa.Column("include_patterns", sa.JSON(), nullable=False),
        sa.Column("exclude_patterns", sa.JSON(), nullable=False),
        sa.Column("manifest_path", sa.Text(), nullable=False),
        sa.Column("manifest_digest", sa.String(length=64), nullable=False),
        sa.Column("file_count", sa.Integer(), nullable=False),
        sa.Column("total_size", sa.BigInteger(), nullable=False),
        sa.Column("manifest_state", sa.String(length=16), nullable=False),
        sa.Column("last_verified_at", UTCDateTime(), nullable=False),
        sa.Column("created_by_task_id", sa.Integer(), nullable=True),
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
            "profile_config_version",
            "artifact_id",
            name="uix_preheat_artifact_profile_version_artifact",
        ),
    )
    op.create_index(
        "ix_preheat_artifact_profile_state_version",
        "model_preheat_artifacts",
        ["profile_id", "profile_config_version", "manifest_state"],
    )
    op.create_table(
        "model_storage_sync_tasks",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("model_file_id", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.Integer(), nullable=False),
        sa.Column("worker_uuid", sa.String(length=255), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("profile_config_version", sa.Integer(), nullable=False),
        sa.Column("request_identity", sa.JSON(), nullable=False),
        sa.Column("request_digest", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("model_id", sa.String(length=1024), nullable=False),
        sa.Column("resolved_revision", sa.String(length=1024), nullable=False),
        sa.Column("credential_snapshot_encrypted", sa.JSON(), nullable=False),
        sa.Column("encryption_key_version", sa.String(length=255), nullable=False),
        sa.Column("artifact_id", sa.String(length=64), nullable=True),
        sa.Column("state", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("state_message", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("file_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_size", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("transfer_source", sa.String(length=32), nullable=True),
        sa.Column("transfer_profile_id", sa.Integer(), nullable=True),
        sa.Column("source_worker_id", sa.Integer(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("started_at", UTCDateTime(), nullable=True),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_file_id"], ["model_files.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["worker_id"], ["workers.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["profile_id"], ["model_preheat_s3_profiles.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"], ondelete="SET NULL"
        ),
    )
    op.create_index(
        "ix_model_storage_sync_model_file_profile",
        "model_storage_sync_tasks",
        ["model_file_id", "profile_id"],
    )
    op.create_index(
        "ix_model_storage_sync_state", "model_storage_sync_tasks", ["state"]
    )
    op.create_table(
        "model_file_download_executions",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("model_file_id", sa.Integer(), nullable=False),
        sa.Column("request_identity", sa.JSON(), nullable=False),
        sa.Column("request_digest", sa.String(length=64), nullable=False),
        sa.Column("target_worker_id", sa.Integer(), nullable=False),
        sa.Column("target_worker_uuid", sa.String(length=255), nullable=False),
        sa.Column("default_profile_id", sa.Integer(), nullable=True),
        sa.Column("default_profile_config_version", sa.Integer(), nullable=True),
        sa.Column("credential_snapshot_encrypted", sa.JSON(), nullable=True),
        sa.Column("encryption_key_version", sa.String(length=255), nullable=True),
        sa.Column(
            "state",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("claimed_by_worker_uuid", sa.String(length=255), nullable=True),
        sa.Column("claimed_at", UTCDateTime(), nullable=True),
        sa.Column("transfer_source", sa.String(length=32), nullable=True),
        sa.Column("transfer_profile_id", sa.Integer(), nullable=True),
        sa.Column("source_worker_id", sa.Integer(), nullable=True),
        sa.Column("state_message", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_file_id"], ["model_files.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["target_worker_id"], ["workers.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["default_profile_id"], ["model_preheat_s3_profiles.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint(
            "model_file_id", name="uix_model_file_download_execution_model_file"
        ),
    )


def downgrade() -> None:
    # 仅恢复空表结构，不回填旧数据。
    _drop_new_tables()
    _recreate_old_tables()
    _downgrade_policies()
    _downgrade_tasks()
    _downgrade_workers()
    _downgrade_model_files()
    _downgrade_profiles()


def _drop_new_tables() -> None:
    op.drop_table("model_file_download_executions")
    op.drop_index(
        "ix_model_storage_sync_state", table_name="model_storage_sync_tasks"
    )
    op.drop_index(
        "ix_model_storage_sync_model_file_profile", table_name="model_storage_sync_tasks"
    )
    op.drop_table("model_storage_sync_tasks")
    op.drop_index(
        "ix_preheat_artifact_profile_state_version",
        table_name="model_preheat_artifacts",
    )
    op.drop_table("model_preheat_artifacts")


def _recreate_old_tables() -> None:
    op.create_table(
        "model_preheat_publish_locks",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("s3_profile_id", sa.Integer(), nullable=False),
        sa.Column("cache_key", sa.String(length=255), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("lease_expires_at", UTCDateTime(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["s3_profile_id"], ["model_preheat_s3_profiles.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["task_id"], ["model_preheat_tasks.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("s3_profile_id", "cache_key", name="uix_preheat_publish"),
    )
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
            "profile_id", "cache_key", name="uix_preheat_cached_model_profile_key"
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
            "job_id", "cached_model_id", name="uix_preheat_inventory_scan_snapshot_row"
        ),
    )
    op.create_index(
        "ix_preheat_inventory_scan_snapshot_job",
        "model_preheat_inventory_scan_snapshots",
        ["job_id"],
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
            "profile_id", "selection_key", name="uix_preheat_inventory_selection_lock"
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
    op.create_table(
        "model_cache_tasks",
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("model_file_id", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.Integer(), nullable=False),
        # downgrade 重建旧表时使用显式长度，保证 MySQL DDL 有效（VARCHAR 需要长度）。
        sa.Column("model_id", sa.String(length=255), nullable=False),
        sa.Column("target_path", sa.Text(), nullable=False),
        sa.Column("source_paths", sa.JSON(), nullable=False),
        sa.Column("state", sa.String(length=64), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("uploaded_size", sa.BigInteger(), nullable=False),
        sa.Column("total_size", sa.BigInteger(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_file_id"], ["model_files.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["worker_id"], ["workers.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"], ondelete="SET NULL"
        ),
    )
    op.create_index("ix_model_cache_tasks_model_id", "model_cache_tasks", ["model_id"])


def _downgrade_policies() -> None:
    with op.batch_alter_table(
        "model_preheat_distribution_policies"
    ) as batch_op:
        batch_op.drop_constraint(
            "uix_preheat_distribution_policy_selector", type_="unique"
        )
        batch_op.create_unique_constraint(
            "uix_preheat_distribution_policy_selector",
            ["profile_id", "cache_key", "target_scope", "selector_digest"],
        )
        # 旧 cache_key 无来源，回退时用 server_default 空串保持 NOT NULL，仅恢复结构。
        batch_op.add_column(
            sa.Column(
                "cache_key",
                sa.String(length=255),
                nullable=False,
                server_default="",
            )
        )
        batch_op.drop_column("request_identity")
        batch_op.drop_column("request_digest")


def _downgrade_tasks() -> None:
    with op.batch_alter_table("model_preheat_tasks") as batch_op:
        batch_op.drop_column("transfer_source")
        batch_op.drop_column("transfer_profile_id")
        batch_op.drop_column("source_worker_id")
        batch_op.drop_column("artifact_id")
        batch_op.add_column(
            sa.Column(
                "cache_key",
                sa.String(length=255),
                nullable=False,
                server_default="",
            )
        )
        batch_op.drop_column("request_identity")
        batch_op.drop_column("request_digest")


def _downgrade_workers() -> None:
    op.drop_column("workers", "model_storage_protocol_version")


def _downgrade_model_files() -> None:
    op.drop_column("model_files", "requested_revision")
    op.drop_column("model_files", "resolved_revision")


def _downgrade_profiles() -> None:
    with op.batch_alter_table("model_preheat_s3_profiles") as batch_op:
        batch_op.drop_constraint(
            "uix_model_preheat_s3_profiles_default_slot", type_="unique"
        )
        batch_op.drop_constraint(
            "uix_model_preheat_s3_profiles_provisioning_key", type_="unique"
        )
        batch_op.drop_column("source_fallback_enabled")
        batch_op.drop_column("default_slot")
        batch_op.drop_column("system_managed")
        batch_op.drop_column("provisioning_key")
        batch_op.drop_column("provisioning_source")
        batch_op.add_column(
            sa.Column(
                "is_default", sa.Boolean(), nullable=False, server_default=sa.false()
            )
        )
