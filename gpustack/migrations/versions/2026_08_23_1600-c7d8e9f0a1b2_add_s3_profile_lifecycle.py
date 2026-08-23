"""add S3 profile lifecycle and active storage key

Revision ID: c7d8e9f0a1b2
Revises: b6c7d8e9f0a1
Create Date: 2026-08-23 16:00:00
"""

from datetime import datetime, timezone
import hashlib
from typing import Sequence, Union
from urllib.parse import urlparse

import sqlalchemy as sa
from alembic import op


revision: str = "c7d8e9f0a1b2"
down_revision: Union[str, None] = "b6c7d8e9f0a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _storage_key(endpoint: str, bucket: str) -> str:
    parsed = urlparse(endpoint)
    port = parsed.port
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("invalid_endpoint")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("invalid_endpoint_port")
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid_endpoint_format")
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if (scheme, port) in {("http", 80), ("https", 443)}:
        port = None
    normalized_endpoint = f"{scheme}://{host}"
    if port is not None:
        normalized_endpoint = f"{normalized_endpoint}:{port}"
    location = f"{normalized_endpoint}|{(bucket or '').strip().lower()}"
    return hashlib.sha256(location.encode("utf-8")).hexdigest()


def upgrade() -> None:
    with op.batch_alter_table("model_preheat_s3_profiles") as batch_op:
        batch_op.add_column(
            sa.Column("lifecycle_state", sa.String(length=32), nullable=False, server_default="active")
        )
        batch_op.add_column(sa.Column("active_storage_key", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("ever_used_at", sa.DateTime(timezone=True), nullable=True))

    bind = op.get_bind()
    profiles = sa.table(
        "model_preheat_s3_profiles",
        sa.column("id", sa.Integer),
        sa.column("endpoint", sa.String),
        sa.column("bucket", sa.String),
        sa.column("default_slot", sa.String),
        sa.column("system_managed", sa.Boolean),
        sa.column("lifecycle_state", sa.String(length=32)),
        sa.column("active_storage_key", sa.String(length=64)),
        sa.column("ever_used_at", sa.DateTime(timezone=True)),
    )
    rows = bind.execute(
        sa.select(
            profiles.c.id,
            profiles.c.endpoint,
            profiles.c.bucket,
            profiles.c.default_slot,
            profiles.c.system_managed,
        ).order_by(profiles.c.id)
    ).mappings()
    grouped = {}
    for row in rows:
        try:
            storage_key = _storage_key(row["endpoint"], row["bucket"])
        except (TypeError, ValueError):
            bind.execute(
                profiles.update()
                .where(profiles.c.id == row["id"])
                .values(
                    lifecycle_state="maintenance",
                    active_storage_key=None,
                    default_slot=None,
                )
            )
            continue
        grouped.setdefault(storage_key, []).append(row)

    # 存量重复不得阻断升级：默认手工优先，其余按稳定 ID 保留一个 active。
    for storage_key, candidates in grouped.items():
        winner = min(
            candidates,
            key=lambda row: (
                0 if row["default_slot"] == "global" else 1,
                0 if not row["system_managed"] else 1,
                row["id"],
            ),
        )
        for row in candidates:
            values = {
                "lifecycle_state": "active" if row["id"] == winner["id"] else "maintenance",
                "active_storage_key": storage_key if row["id"] == winner["id"] else None,
            }
            if row["id"] != winner["id"]:
                values["default_slot"] = None
            bind.execute(
                profiles.update().where(profiles.c.id == row["id"]).values(**values)
            )

    # 仅回填可证明的真实存储使用；连通性记录和未领取下载执行不计入。
    now = datetime.now(timezone.utc)
    used_profile_ids = set()
    evidence_queries = [
        "SELECT DISTINCT profile_id FROM model_preheat_artifacts",
        "SELECT DISTINCT profile_id FROM model_storage_sync_tasks "
        "WHERE started_at IS NOT NULL",
        "SELECT DISTINCT preheat.s3_profile_id "
        "FROM model_preheat_tasks AS preheat "
        "JOIN model_preheat_worker_tasks AS worker_task "
        "ON worker_task.task_id = preheat.id "
        "WHERE preheat.s3_profile_id IS NOT NULL "
        "AND (worker_task.started_at IS NOT NULL OR worker_task.attempt > 0)",
        "SELECT DISTINCT default_profile_id FROM model_file_download_executions "
        "WHERE default_profile_id IS NOT NULL AND claimed_at IS NOT NULL",
    ]
    for statement in evidence_queries:
        for profile_id, in bind.execute(sa.text(statement)):
            used_profile_ids.add(profile_id)
    if used_profile_ids:
        bind.execute(
            profiles.update()
            .where(profiles.c.id.in_(used_profile_ids))
            .values(ever_used_at=now)
        )

    with op.batch_alter_table("model_preheat_s3_profiles") as batch_op:
        batch_op.create_unique_constraint(
            "uix_model_preheat_s3_profiles_active_storage_key",
            ["active_storage_key"],
        )


def downgrade() -> None:
    with op.batch_alter_table("model_preheat_s3_profiles") as batch_op:
        batch_op.drop_constraint(
            "uix_model_preheat_s3_profiles_active_storage_key", type_="unique"
        )
        batch_op.drop_column("ever_used_at")
        batch_op.drop_column("active_storage_key")
        batch_op.drop_column("lifecycle_state")
