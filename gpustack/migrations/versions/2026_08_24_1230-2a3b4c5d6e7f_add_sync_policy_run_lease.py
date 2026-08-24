"""为模型同步策略运行记录增加跨 Server 租约

Revision ID: 2a3b4c5d6e7f
Revises: 1a2b3c4d5e6f
Create Date: 2026-08-24 12:30:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "2a3b4c5d6e7f"
down_revision: Union[str, None] = "1a2b3c4d5e6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table(
        "model_storage_sync_policy_runs", **_run_batch_options()
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "attempt",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch_op.add_column(sa.Column("lease_owner", sa.String(length=64)))
        batch_op.add_column(sa.Column("lease_token", sa.String(length=64)))
        batch_op.add_column(
            sa.Column("lease_expires_at", sa.DateTime(timezone=True))
        )
        batch_op.add_column(sa.Column("started_at", sa.DateTime(timezone=True)))
        batch_op.add_column(
            sa.Column("execution_user_id", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_storage_sync_policy_run_execution_user",
            "users",
            ["execution_user_id"],
            ["id"],
            ondelete="RESTRICT",
        )

    with op.batch_alter_table(
        "model_preheat_idempotency_records", recreate="never"
    ) as batch_op:
        batch_op.add_column(sa.Column("batch_lease_token", sa.String(length=64)))
        batch_op.add_column(
            sa.Column("batch_lease_expires_at", sa.DateTime(timezone=True))
        )


def downgrade() -> None:
    with op.batch_alter_table(
        "model_storage_sync_policy_runs", **_run_batch_options()
    ) as batch_op:
        batch_op.drop_column("started_at")
        batch_op.drop_column("lease_expires_at")
        batch_op.drop_column("lease_token")
        batch_op.drop_column("lease_owner")
        batch_op.drop_constraint(
            "fk_storage_sync_policy_run_execution_user", type_="foreignkey"
        )
        batch_op.drop_column("execution_user_id")
        batch_op.drop_column("attempt")

    with op.batch_alter_table(
        "model_preheat_idempotency_records", recreate="never"
    ) as batch_op:
        batch_op.drop_column("batch_lease_expires_at")
        batch_op.drop_column("batch_lease_token")


def _run_batch_options():
    context = op.get_context()
    if context.dialect.name != "sqlite":
        return {"recreate": "never"}
    options = {"recreate": "always"}
    if context.as_sql:
        options["copy_from"] = sa.Table(
            "model_storage_sync_policy_runs",
            sa.MetaData(),
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "policy_id",
                sa.Integer(),
                sa.ForeignKey(
                    "model_storage_sync_policies.id", ondelete="CASCADE"
                ),
                nullable=False,
            ),
            sa.Column("trigger", sa.String(length=32), nullable=False),
            sa.Column("state", sa.String(length=32), nullable=False),
            sa.Column("window_start_utc", sa.DateTime(timezone=True), nullable=False),
            sa.Column("operation_key", sa.String(length=64), nullable=False),
            sa.Column("request_hash", sa.String(length=64), nullable=False),
            sa.Column("response_payload", sa.JSON(), nullable=True),
            sa.Column("error_code", sa.String(length=255), nullable=True),
            sa.Column(
                "created_by_user_id",
                sa.Integer(),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint(
                "operation_key", name="uix_storage_sync_policy_operation"
            ),
            sa.UniqueConstraint(
                "policy_id",
                "window_start_utc",
                name="uix_storage_sync_policy_window",
            ),
        )
    return options
