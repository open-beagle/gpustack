"""修复模型预热调度增量 DDL

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-08-12 18:00:00.000000
"""

from contextlib import contextmanager
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.schemas.common import UTCDateTime


revision: str = "f2a3b4c5d6e7"
down_revision: Union[str, None] = "e1f2a3b4c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


TASK_SCHEDULE_FK = "fk_model_preheat_tasks_schedule_id_model_preheat_schedules"
SCHEDULE_WINDOW_UNIQUE = "uix_preheat_schedule_window"


def upgrade() -> None:
    if op.get_context().dialect.name == "sqlite":
        with _sqlite_foreign_keys_suspended():
            _upgrade_sqlite()
        return

    _upgrade_server_database()


def _upgrade_sqlite() -> None:
    op.add_column(
        "model_preheat_schedules",
        sa.Column("bandwidth_limit_mbps", sa.Integer(), nullable=True),
    )
    with op.batch_alter_table(
        "model_preheat_tasks",
        copy_from=(
            _model_preheat_tasks_table(False) if op.get_context().as_sql else None
        ),
    ) as batch_op:
        batch_op.add_column(
            sa.Column("bandwidth_limit_mbps", sa.Integer(), nullable=True),
        )
        batch_op.create_foreign_key(
            TASK_SCHEDULE_FK,
            "model_preheat_schedules",
            ["schedule_id"],
            ["id"],
            ondelete="SET NULL",
        )
    _backfill_operation_keys()
    with op.batch_alter_table(
        "model_preheat_schedule_runs",
        copy_from=(
            _model_preheat_schedule_runs_table(False)
            if op.get_context().as_sql
            else None
        ),
    ) as batch_op:
        batch_op.alter_column(
            "operation_key",
            existing_type=sa.String(length=64),
            nullable=False,
        )
        batch_op.drop_constraint(SCHEDULE_WINDOW_UNIQUE, type_="unique")
        batch_op.create_unique_constraint(
            SCHEDULE_WINDOW_UNIQUE,
            ["schedule_id", "window_start_utc", "operation_key"],
        )


def _upgrade_server_database() -> None:
    op.add_column(
        "model_preheat_schedules",
        sa.Column("bandwidth_limit_mbps", sa.Integer(), nullable=True),
    )
    op.add_column(
        "model_preheat_tasks",
        sa.Column("bandwidth_limit_mbps", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        TASK_SCHEDULE_FK,
        "model_preheat_tasks",
        "model_preheat_schedules",
        ["schedule_id"],
        ["id"],
        ondelete="SET NULL",
    )
    _backfill_operation_keys()
    op.alter_column(
        "model_preheat_schedule_runs",
        "operation_key",
        existing_type=sa.String(length=64),
        nullable=False,
    )
    op.drop_constraint(
        SCHEDULE_WINDOW_UNIQUE,
        "model_preheat_schedule_runs",
        type_="unique",
    )
    op.create_unique_constraint(
        SCHEDULE_WINDOW_UNIQUE,
        "model_preheat_schedule_runs",
        ["schedule_id", "window_start_utc", "operation_key"],
    )


def downgrade() -> None:
    if op.get_context().dialect.name == "sqlite":
        with _sqlite_foreign_keys_suspended():
            _downgrade_sqlite()
        return

    _downgrade_server_database()


def _downgrade_sqlite() -> None:
    with op.batch_alter_table(
        "model_preheat_schedule_runs",
        copy_from=(
            _model_preheat_schedule_runs_table(True)
            if op.get_context().as_sql
            else None
        ),
    ) as batch_op:
        batch_op.alter_column(
            "operation_key",
            existing_type=sa.String(length=64),
            nullable=True,
        )
        batch_op.drop_constraint(SCHEDULE_WINDOW_UNIQUE, type_="unique")
        batch_op.create_unique_constraint(
            SCHEDULE_WINDOW_UNIQUE,
            ["schedule_id", "window_start_utc"],
        )
    with op.batch_alter_table(
        "model_preheat_tasks",
        copy_from=_model_preheat_tasks_table(True) if op.get_context().as_sql else None,
    ) as batch_op:
        batch_op.drop_constraint(TASK_SCHEDULE_FK, type_="foreignkey")
        batch_op.drop_column("bandwidth_limit_mbps")
    op.drop_column("model_preheat_schedules", "bandwidth_limit_mbps")


def _downgrade_server_database() -> None:
    op.alter_column(
        "model_preheat_schedule_runs",
        "operation_key",
        existing_type=sa.String(length=64),
        nullable=True,
    )
    op.drop_constraint(
        SCHEDULE_WINDOW_UNIQUE,
        "model_preheat_schedule_runs",
        type_="unique",
    )
    op.create_unique_constraint(
        SCHEDULE_WINDOW_UNIQUE,
        "model_preheat_schedule_runs",
        ["schedule_id", "window_start_utc"],
    )
    op.drop_constraint(
        TASK_SCHEDULE_FK,
        "model_preheat_tasks",
        type_="foreignkey",
    )
    op.drop_column("model_preheat_tasks", "bandwidth_limit_mbps")
    op.drop_column("model_preheat_schedules", "bandwidth_limit_mbps")


def _backfill_operation_keys() -> None:
    schedule_runs = sa.table(
        "model_preheat_schedule_runs",
        sa.column("id", sa.Integer()),
        sa.column("operation_key", sa.String(length=64)),
    )
    legacy_key = sa.literal("legacy-null-run:") + sa.cast(
        schedule_runs.c.id, sa.String(length=32)
    )
    op.execute(
        schedule_runs.update()
        .where(schedule_runs.c.operation_key.is_(None))
        .values(operation_key=legacy_key)
    )


@contextmanager
def _sqlite_foreign_keys_suspended():
    context = op.get_context()
    if context.as_sql:
        op.execute("PRAGMA foreign_keys=OFF")
        try:
            yield
        finally:
            op.execute("PRAGMA foreign_keys=ON")
        return

    bind = op.get_bind()
    foreign_keys_enabled = bool(
        bind.exec_driver_sql("PRAGMA foreign_keys").scalar_one()
    )
    if context._transaction is None:
        raise RuntimeError("sqlite_migration_transaction_required")

    if foreign_keys_enabled:
        with context.autocommit_block():
            op.get_bind().exec_driver_sql("PRAGMA foreign_keys=OFF")

    # Python sqlite3 的 legacy transaction control 不会为 DDL 自动 BEGIN。
    # SAVEPOINT 既能开启物理事务，也能安全加入 Alembic 已开启的事务。
    savepoint = op.get_bind().begin_nested()
    try:
        yield
        if foreign_keys_enabled:
            violation = (
                op.get_bind().exec_driver_sql("PRAGMA foreign_key_check").first()
            )
            if violation is not None:
                raise RuntimeError(f"sqlite_foreign_key_check_failed: {violation}")
    except BaseException:
        savepoint.rollback()
        if foreign_keys_enabled:
            with context.autocommit_block():
                op.get_bind().exec_driver_sql("PRAGMA foreign_keys=ON")
        raise
    else:
        savepoint.commit()
        if foreign_keys_enabled:
            with context.autocommit_block():
                op.get_bind().exec_driver_sql("PRAGMA foreign_keys=ON")


def _model_preheat_tasks_table(include_changes: bool) -> sa.Table:
    columns = [
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("source", sa.String(length=255), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False),
        sa.Column("requested_revision", sa.String(length=255), nullable=True),
        sa.Column("resolved_revision", sa.String(length=255), nullable=False),
        sa.Column("include_patterns", sa.JSON(), nullable=False),
        sa.Column("exclude_patterns", sa.JSON(), nullable=False),
        sa.Column("selection_digest", sa.String(length=255), nullable=False),
        sa.Column("cache_key", sa.String(length=255), nullable=False),
        sa.Column("generation_id", sa.String(length=255), nullable=False),
        sa.Column(
            "desired_state",
            sa.String(length=255),
            nullable=False,
            server_default="running",
        ),
        sa.Column(
            "execution_state",
            sa.String(length=255),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("paused_from_state", sa.String(length=255), nullable=True),
        sa.Column("state_message", sa.Text(), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("seed_worker_uuid", sa.String(length=255), nullable=True),
        sa.Column("seed_worker_id", sa.Integer(), nullable=True),
        sa.Column("seed_source", sa.String(length=255), nullable=True),
        sa.Column("target_scope", sa.String(length=255), nullable=False),
        sa.Column("target_gpu_names", sa.JSON(), nullable=True),
        sa.Column("target_worker_uuids", sa.JSON(), nullable=False),
        sa.Column("target_worker_snapshot", sa.JSON(), nullable=False),
        sa.Column("local_cache_hit_worker_uuids", sa.JSON(), nullable=True),
        sa.Column("removed_target_worker_uuids", sa.JSON(), nullable=True),
        sa.Column("s3_profile_id", sa.Integer(), nullable=False),
        sa.Column("s3_profile_config_version", sa.Integer(), nullable=False),
        sa.Column("s3_profile_snapshot_encrypted", sa.JSON(), nullable=False),
        sa.Column("encryption_key_version", sa.String(length=255), nullable=False),
        sa.Column("s3_backfill_policy", sa.String(length=255), nullable=False),
        sa.Column("s3_ready_path", sa.String(length=255), nullable=True),
        sa.Column("s3_manifest_path", sa.String(length=255), nullable=True),
        sa.Column("manifest_digest", sa.String(length=255), nullable=True),
        sa.Column(
            "keep_new_workers_in_sync",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("schedule_id", sa.Integer(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("started_at", UTCDateTime(), nullable=True),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["s3_profile_id"],
            ["model_preheat_s3_profiles.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"], ondelete="SET NULL"
        ),
    ]
    if include_changes:
        columns.extend(
            [
                sa.Column("bandwidth_limit_mbps", sa.Integer(), nullable=True),
                sa.ForeignKeyConstraint(
                    ["schedule_id"],
                    ["model_preheat_schedules.id"],
                    name=TASK_SCHEDULE_FK,
                    ondelete="SET NULL",
                ),
            ]
        )
    return sa.Table("model_preheat_tasks", sa.MetaData(), *columns)


def _model_preheat_schedule_runs_table(include_changes: bool) -> sa.Table:
    window_columns = ["schedule_id", "window_start_utc"]
    if include_changes:
        window_columns.append("operation_key")
    return sa.Table(
        "model_preheat_schedule_runs",
        sa.MetaData(),
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("schedule_id", sa.Integer(), nullable=False),
        sa.Column("window_start_utc", UTCDateTime(), nullable=False),
        sa.Column("window_end_utc", UTCDateTime(), nullable=False),
        sa.Column("trigger", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=64), nullable=False),
        sa.Column(
            "operation_key",
            sa.String(length=64),
            nullable=not include_changes,
        ),
        sa.Column("slot", sa.Integer(), nullable=True),
        sa.Column("task_id", sa.Integer(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=255), nullable=True),
        sa.Column("started_at", UTCDateTime(), nullable=True),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("deleted_at", UTCDateTime(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["schedule_id"], ["model_preheat_schedules.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["task_id"], ["model_preheat_tasks.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint(*window_columns, name=SCHEDULE_WINDOW_UNIQUE),
        sa.UniqueConstraint("operation_key", name="uix_preheat_schedule_run_operation"),
        sa.UniqueConstraint("schedule_id", "slot", name="uix_preheat_schedule_slot"),
    )
