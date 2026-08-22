"""统一模型存储收敛 migration 的定向测试。

覆盖任务 2 步骤 2/3：
- upgrade 直接删除旧 cache / inventory / generation / selection lock / publication
  marker 数据结构，再创建统一 Artifact 库存、同步任务表和普通下载私有执行表；
- 任务与分发策略用 request identity/request digest 替代 cache_key，
  任务允许 artifact_id=NULL 并提供 CAS 绑定字段；
- upgrade 不迁移旧数据，downgrade 只恢复空表结构；
- Profile provisioning/default_slot/source_fallback，无持久 is_default；
- ModelFile revision、Worker protocol 默认 0、任务结果来源字段、
  inventory profile_config_version；
- 三库（SQLite/PostgreSQL/MySQL）DDL 兼容。
"""

import ast
import importlib.util
import io
import hashlib
import json
from pathlib import Path

import sqlalchemy as sa
import pytest
from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.schema import CreateTable

from gpustack.migrations.validate import validate_revision_graph

# 收敛 migration 前置基线 head。
BASELINE_HEAD = "b0307846729c"
UNIFY_REVISION = "c1d2e3f4a5b6"
# 任务 3 子阶段 B：活动同步任务数据库级去重槽（新增 head）。
DEDUPE_REVISION = "d2e3f4a5b6c7"
# 任务 3 定向复审：同步任务执行 lease 与完成结果固定字段（新增 head）。
LEASE_REVISION = "e3f4a5b6c7d8"
# 任务 4：普通下载首次领取时固定 resolved revision 与 Artifact 命中。
DOWNLOAD_REVISION = "f4a5b6c7d8e9"
# 任务 4 复审：固定 Profile 引用增加数据库级删除保护。
PROFILE_PIN_REVISION = "a5b6c7d8e9f0"

MIGRATION_ROOT = Path(__file__).resolve().parents[2] / "gpustack/migrations/versions"

NEW_TABLES = (
    "model_preheat_artifacts",
    "model_storage_sync_tasks",
    "model_file_download_executions",
    "model_file_download_execution_profile_pins",
    "model_storage_sync_task_dedupe_slots",
)
DROPPED_TABLES = (
    "model_cache_tasks",
    "model_preheat_cached_models",
    "model_preheat_inventory_jobs",
    "model_preheat_inventory_scan_snapshots",
    "model_preheat_inventory_generations",
    "model_preheat_inventory_selection_locks",
    "model_preheat_publication_markers",
    "model_preheat_publish_locks",
)

DIGEST = "d" * 64


def _alembic_config(tmp_path: Path, name: str = "m.db") -> Config:
    database_path = tmp_path / name
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "gpustack/migrations"),
    )
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    return config


def _create_baseline(engine) -> None:
    """构造 BASELINE_HEAD 状态所需的最小前置表，避免运行需 server 注入 config 的历史 migration。"""
    ddl = [
        "CREATE TABLE users (id INTEGER PRIMARY KEY)",
        "CREATE TABLE workers (id INTEGER PRIMARY KEY, worker_uuid VARCHAR(255))",
        "CREATE TABLE model_files (id INTEGER PRIMARY KEY)",
        "CREATE TABLE model_preheat_s3_profiles ("
        " id INTEGER PRIMARY KEY, name VARCHAR(255) NOT NULL, description TEXT, "
        " endpoint VARCHAR(255) NOT NULL, bucket VARCHAR(255) NOT NULL, "
        " prefix VARCHAR(255) NOT NULL DEFAULT '', tls_enabled BOOLEAN NOT NULL DEFAULT 1, "
        " tls_verify BOOLEAN NOT NULL DEFAULT 1, region VARCHAR(255), "
        " use_virtual_hosted_style BOOLEAN NOT NULL DEFAULT 1, "
        " is_default BOOLEAN NOT NULL DEFAULT 0, access_key_encrypted JSON NOT NULL, "
        " secret_key_encrypted JSON NOT NULL, encryption_key_version VARCHAR(255) NOT NULL, "
        " config_version INTEGER NOT NULL DEFAULT 1, "
        " connectivity_state VARCHAR(255) NOT NULL DEFAULT 'pending', "
        " last_connectivity_check_id INTEGER, last_connectivity_checked_at DATETIME, "
        " created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, deleted_at DATETIME)",
        "CREATE TABLE model_preheat_tasks ("
        " id INTEGER PRIMARY KEY, attempt INTEGER NOT NULL DEFAULT 1, "
        " source VARCHAR(255) NOT NULL, model_id VARCHAR(255) NOT NULL, "
        " requested_revision VARCHAR(255), resolved_revision VARCHAR(255) NOT NULL, "
        " include_patterns JSON NOT NULL, exclude_patterns JSON NOT NULL, "
        " selection_digest VARCHAR(255) NOT NULL, cache_key VARCHAR(255) NOT NULL, "
        " generation_id VARCHAR(255) NOT NULL, desired_state VARCHAR(255) NOT NULL DEFAULT 'running', "
        " execution_state VARCHAR(255) NOT NULL DEFAULT 'pending', paused_from_state VARCHAR(255), "
        " state_message TEXT, progress FLOAT NOT NULL DEFAULT 0, seed_worker_uuid VARCHAR(255), "
        " seed_worker_id INTEGER, seed_source VARCHAR(255), target_scope VARCHAR(255) NOT NULL, "
        " target_gpu_names JSON, target_worker_uuids JSON NOT NULL, target_worker_snapshot JSON NOT NULL, "
        " local_cache_hit_worker_uuids JSON, removed_target_worker_uuids JSON, "
        " s3_profile_id INTEGER NOT NULL, s3_profile_config_version INTEGER NOT NULL, "
        " s3_profile_snapshot_encrypted JSON NOT NULL, encryption_key_version VARCHAR(255) NOT NULL, "
        " s3_backfill_policy VARCHAR(255) NOT NULL, s3_ready_path VARCHAR(255), "
        " s3_manifest_path VARCHAR(255), manifest_digest VARCHAR(255), "
        " keep_new_workers_in_sync BOOLEAN NOT NULL DEFAULT 0, schedule_id INTEGER, "
        " bandwidth_limit_mbps INTEGER, created_by_user_id INTEGER, "
        " started_at DATETIME, finished_at DATETIME, created_at DATETIME NOT NULL, "
        " updated_at DATETIME NOT NULL, deleted_at DATETIME)",
        "CREATE TABLE model_preheat_distribution_policies ("
        " id INTEGER PRIMARY KEY, name VARCHAR(255) NOT NULL, enabled BOOLEAN NOT NULL DEFAULT 1, "
        " profile_version_stale BOOLEAN NOT NULL DEFAULT 0, profile_id INTEGER NOT NULL, "
        " profile_config_version INTEGER NOT NULL, cache_key VARCHAR(255) NOT NULL, "
        " target_scope VARCHAR(255) NOT NULL, worker_selector JSON NOT NULL, "
        " gpu_selector JSON NOT NULL, selector_digest VARCHAR(255) NOT NULL, "
        " created_by_task_id INTEGER, last_reconciled_at DATETIME, "
        " created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, deleted_at DATETIME, "
        "CONSTRAINT uix_preheat_distribution_policy_selector UNIQUE "
        "(profile_id, cache_key, target_scope, selector_digest))",
        "CREATE TABLE model_preheat_cached_models ("
        " id INTEGER PRIMARY KEY, profile_id INTEGER NOT NULL, profile_config_version INTEGER NOT NULL, "
        " cache_key VARCHAR(256) NOT NULL, source VARCHAR(32) NOT NULL, model_id VARCHAR(1024) NOT NULL, "
        " resolved_revision VARCHAR(1024) NOT NULL, include_patterns JSON NOT NULL, "
        " exclude_patterns JSON NOT NULL, generation_id VARCHAR(256) NOT NULL, ready_path TEXT NOT NULL, "
        " manifest_path TEXT NOT NULL, manifest_digest VARCHAR(64) NOT NULL, file_count INTEGER NOT NULL, "
        " total_size BIGINT NOT NULL, manifest_state VARCHAR(16) NOT NULL, last_verified_at DATETIME NOT NULL, "
        " created_by_task_id INTEGER, source_parent_attempt INTEGER, created_at DATETIME NOT NULL, "
        " updated_at DATETIME NOT NULL, deleted_at DATETIME)",
        "CREATE INDEX ix_preheat_cached_model_profile_state_key ON model_preheat_cached_models "
        "(profile_id, manifest_state, cache_key)",
        "CREATE TABLE model_preheat_inventory_jobs ("
        " id INTEGER PRIMARY KEY, profile_id INTEGER NOT NULL, profile_config_version INTEGER NOT NULL, "
        " kind VARCHAR(16) NOT NULL, state VARCHAR(16) NOT NULL, active_key VARCHAR(255), "
        " claim_token VARCHAR(64), cursor JSON, scanned_count INTEGER NOT NULL, valid_count INTEGER NOT NULL, "
        " invalid_count INTEGER NOT NULL, orphan_count INTEGER NOT NULL, deleted_count INTEGER NOT NULL, "
        " skipped_count INTEGER NOT NULL, failed_count INTEGER NOT NULL, error_code VARCHAR(64), "
        " error_message TEXT, started_at DATETIME, lease_expires_at DATETIME, finished_at DATETIME, "
        " created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, deleted_at DATETIME)",
        "CREATE INDEX ix_preheat_inventory_job_profile_created ON model_preheat_inventory_jobs "
        "(profile_id, created_at)",
        "CREATE TABLE model_preheat_inventory_scan_snapshots ("
        " id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL, cached_model_id INTEGER NOT NULL, "
        " revision INTEGER NOT NULL, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, deleted_at DATETIME)",
        "CREATE INDEX ix_preheat_inventory_scan_snapshot_job ON model_preheat_inventory_scan_snapshots "
        "(job_id)",
        "CREATE TABLE model_preheat_inventory_generations ("
        " id INTEGER PRIMARY KEY, profile_id INTEGER NOT NULL, generation_key VARCHAR(64) NOT NULL, "
        " selection_key VARCHAR(256) NOT NULL, cache_key VARCHAR(256), generation_path TEXT NOT NULL, "
        " ready_path TEXT NOT NULL, ready_fingerprint VARCHAR(64), ready_generation_path TEXT, "
        " state VARCHAR(16) NOT NULL, first_seen_at DATETIME NOT NULL, last_seen_at DATETIME NOT NULL, "
        " orphaned_at DATETIME, deleted_at_s3 DATETIME, error_code VARCHAR(64), created_at DATETIME NOT NULL, "
        " updated_at DATETIME NOT NULL, deleted_at DATETIME)",
        "CREATE INDEX ix_preheat_inventory_generation_gc ON model_preheat_inventory_generations "
        "(profile_id, state, orphaned_at)",
        "CREATE TABLE model_preheat_inventory_selection_locks ("
        " id INTEGER PRIMARY KEY, profile_id INTEGER NOT NULL, selection_key VARCHAR(256) NOT NULL, "
        " owner_token VARCHAR(64) NOT NULL, operation VARCHAR(16) NOT NULL, lease_expires_at DATETIME NOT NULL, "
        " created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, deleted_at DATETIME)",
        "CREATE TABLE model_preheat_publication_markers ("
        " id INTEGER PRIMARY KEY, profile_id INTEGER NOT NULL, selection_key VARCHAR(256) NOT NULL, "
        " generation_id VARCHAR(256) NOT NULL, task_id INTEGER, parent_attempt INTEGER NOT NULL, "
        " profile_config_version INTEGER NOT NULL, terminated_at DATETIME, created_at DATETIME NOT NULL, "
        " updated_at DATETIME NOT NULL, deleted_at DATETIME)",
        "CREATE INDEX ix_preheat_publication_marker_task_attempt ON model_preheat_publication_markers "
        "(task_id, parent_attempt)",
        "CREATE TABLE model_preheat_publish_locks ("
        " id INTEGER PRIMARY KEY, s3_profile_id INTEGER NOT NULL, cache_key VARCHAR(255) NOT NULL, "
        " task_id INTEGER NOT NULL, lease_expires_at DATETIME NOT NULL, created_at DATETIME NOT NULL, "
        " updated_at DATETIME NOT NULL, deleted_at DATETIME)",
        "CREATE TABLE model_cache_tasks ("
        " id INTEGER PRIMARY KEY, model_file_id INTEGER NOT NULL, worker_id INTEGER NOT NULL, "
        " model_id VARCHAR NOT NULL, target_path TEXT NOT NULL, source_paths JSON NOT NULL, "
        " state VARCHAR NOT NULL, progress FLOAT NOT NULL, uploaded_size BIGINT NOT NULL, "
        " total_size BIGINT NOT NULL, error_message TEXT, created_by_user_id INTEGER, "
        " finished_at DATETIME, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, deleted_at DATETIME)",
        "CREATE INDEX ix_model_cache_tasks_model_id ON model_cache_tasks (model_id)",
        "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)",
    ]
    with engine.begin() as connection:
        for statement in ddl:
            connection.exec_driver_sql(statement)
        connection.exec_driver_sql(
            "INSERT INTO alembic_version (version_num) VALUES (?)", (BASELINE_HEAD,)
        )


def _upgrade_from_baseline(tmp_path: Path):
    config = _alembic_config(tmp_path)
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    _create_baseline(engine)
    command.upgrade(config, "head")
    return config, engine


def _table_names(engine) -> set:
    with engine.connect() as connection:
        return set(sa.inspect(connection).get_table_names())


def test_revision_graph_heads_at_download_revision():
    assert validate_revision_graph() == PROFILE_PIN_REVISION


def test_upgrade_creates_unified_tables_and_drops_legacy(tmp_path):
    _, engine = _upgrade_from_baseline(tmp_path)
    tables = _table_names(engine)
    for table in NEW_TABLES:
        assert table in tables, f"missing new table {table}"
    for table in DROPPED_TABLES:
        assert table not in tables, f"legacy table {table} not dropped"
    with engine.connect() as connection:
        assert (
            connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one()
            == PROFILE_PIN_REVISION
        )
    download_columns = {
        column["name"]
        for column in sa.inspect(engine).get_columns("model_file_download_executions")
    }
    assert {
        "resolved_revision",
        "artifact_id",
        "manifest_path",
        "artifact_total_size",
    } <= download_columns
    pin_foreign_keys = {
        tuple(foreign_key["constrained_columns"]): (
            foreign_key["referred_table"],
            (foreign_key.get("options") or {}).get("ondelete"),
        )
        for foreign_key in sa.inspect(engine).get_foreign_keys(
            "model_file_download_execution_profile_pins"
        )
    }
    assert pin_foreign_keys == {
        ("execution_id",): ("model_file_download_executions", "CASCADE"),
        ("profile_id",): ("model_preheat_s3_profiles", "RESTRICT"),
    }


def test_upgrade_profile_fields_and_no_persistent_is_default(tmp_path):
    _, engine = _upgrade_from_baseline(tmp_path)
    insp = sa.inspect(engine)
    profile_cols = {c["name"] for c in insp.get_columns("model_preheat_s3_profiles")}
    assert "is_default" not in profile_cols
    for col in (
        "provisioning_source",
        "provisioning_key",
        "system_managed",
        "default_slot",
        "source_fallback_enabled",
    ):
        assert col in profile_cols, f"profile missing {col}"
    unique = {
        frozenset(c["column_names"])
        for c in insp.get_unique_constraints("model_preheat_s3_profiles")
    }
    assert frozenset(["provisioning_key"]) in unique
    assert frozenset(["default_slot"]) in unique


def test_default_slot_unique_constraint_rejects_second_default(tmp_path):
    _, engine = _upgrade_from_baseline(tmp_path)
    with engine.begin() as connection:
        now = "2026-08-20 00:00:00"
        connection.exec_driver_sql(
            "INSERT INTO model_preheat_s3_profiles (name, endpoint, bucket, prefix, "
            "tls_enabled, tls_verify, use_virtual_hosted_style, access_key_encrypted, "
            "secret_key_encrypted, encryption_key_version, config_version, connectivity_state, "
            "provisioning_source, provisioning_key, system_managed, default_slot, "
            "source_fallback_enabled, created_at, updated_at) VALUES "
            "('p1','http://x','b','',1,1,1,'{}','{}','v1',1,'pending','manual',NULL,0,'global',1,?,?)",
            (now, now),
        )
        import pytest

        with pytest.raises(sa.exc.IntegrityError):
            connection.exec_driver_sql(
                "INSERT INTO model_preheat_s3_profiles (name, endpoint, bucket, prefix, "
                "tls_enabled, tls_verify, use_virtual_hosted_style, access_key_encrypted, "
                "secret_key_encrypted, encryption_key_version, config_version, connectivity_state, "
                "provisioning_source, provisioning_key, system_managed, default_slot, "
                "source_fallback_enabled, created_at, updated_at) VALUES "
                "('p2','http://x','b','',1,1,1,'{}','{}','v1',1,'pending','manual',NULL,0,'global',1,?,?)",
                (now, now),
            )


def test_upgrade_task_and_worker_and_model_file_fields(tmp_path):
    _, engine = _upgrade_from_baseline(tmp_path)
    insp = sa.inspect(engine)
    task_cols = {c["name"] for c in insp.get_columns("model_preheat_tasks")}
    assert "cache_key" not in task_cols
    for col in (
        "request_identity",
        "request_digest",
        "artifact_id",
        "transfer_source",
        "transfer_profile_id",
        "source_worker_id",
    ):
        assert col in task_cols, f"task missing {col}"
    worker_cols = {c["name"] for c in insp.get_columns("workers")}
    assert "model_storage_protocol_version" in worker_cols
    model_file_cols = {c["name"] for c in insp.get_columns("model_files")}
    assert {"requested_revision", "resolved_revision"} <= model_file_cols
    artifact_cols = {c["name"] for c in insp.get_columns("model_preheat_artifacts")}
    for col in ("profile_id", "profile_config_version", "artifact_id"):
        assert col in artifact_cols, f"artifact missing {col}"


def test_upgrade_backfills_task_request_identity_and_digest(tmp_path):
    config = _alembic_config(tmp_path)
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    _create_baseline(engine)
    with engine.begin() as connection:
        now = "2026-08-20 00:00:00"
        connection.exec_driver_sql(
            "INSERT INTO model_preheat_tasks (id, source, model_id, requested_revision, "
            "resolved_revision, include_patterns, exclude_patterns, selection_digest, cache_key, "
            "generation_id, target_scope, target_worker_uuids, target_worker_snapshot, "
            "s3_profile_snapshot_encrypted, encryption_key_version, s3_backfill_policy, "
            "s3_profile_id, s3_profile_config_version, created_at, updated_at) VALUES "
            "(1,'modelscope','Qwen/Qwen3-32B','master','sha',?,?,?,'ck','gen','selected_workers',?,?,'{}','v1','when_missing',7,1,?,?)",
            (
                json.dumps([], separators=(",", ":")),
                json.dumps([], separators=(",", ":")),
                "d" * 64,
                json.dumps([], separators=(",", ":")),
                json.dumps([], separators=(",", ":")),
                now,
                now,
            ),
        )
    command.upgrade(config, "head")
    with engine.connect() as connection:
        row = connection.exec_driver_sql(
            "SELECT request_identity, request_digest FROM model_preheat_tasks WHERE id=1"
        ).one()
    identity = json.loads(row[0])
    assert identity["source"] == "modelscope"
    assert identity["model_id"] == "Qwen/Qwen3-32B"
    assert identity["requested_revision"] == "master"
    expected_digest = hashlib.sha256(
        json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    assert row[1] == expected_digest


def test_upgrade_backfill_matches_runtime_identity_for_special_characters(tmp_path):
    """migration 回填的 request identity/digest 必须与运行时
    ``ModelPreheatIdentity`` 对**同一原始请求**的 ``request_digest`` 一致。

    关键点：旧 ``ModelPreheatCreate`` 入库前已对 include/exclude_patterns 做
    ``encode_path``（存量是**已编码**值），而 source 已规范化、model_id 与
    requested_revision 存**原始值**（仅校验不编码）。因此迁移回填必须对
    patterns 排序但**不得二次编码**，对 model_id/revision 编码、source 规范化。
    本测试按旧写入形态构造存量行（含 ``dir/a%20b.bin`` 这类已编码 pattern），
    并断言回填 digest == 运行时对同一原始请求的 digest。
    """
    from gpustack.worker.model_preheat.identity import ModelPreheatIdentity

    model_id = "Qwen 组/模型 B"
    requested_revision = "refs/pull/1/head"
    # 运行时原始请求的 patterns（未编码、未排序）。
    raw_include_patterns = ["dir/a b.bin", "a/*.bin", "x/中文.txt"]
    raw_exclude_patterns = ["z/*.bin"]

    # 旧写入形态：ModelPreheatCreate validator 已 encode + sorted 后入库；
    # source 规范化入库；model_id/requested_revision 原始值入库。
    runtime = ModelPreheatIdentity(
        source="ModelScope",
        model_id=model_id,
        revision=requested_revision,
        requested_revision=requested_revision,
        file_patterns=raw_include_patterns,
        exclude_patterns=raw_exclude_patterns,
    )
    stored_include = list(runtime.file_patterns)  # 已编码、已排序
    stored_exclude = list(runtime.exclude_patterns)

    config = _alembic_config(tmp_path)
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    _create_baseline(engine)
    with engine.begin() as connection:
        now = "2026-08-20 00:00:00"
        connection.exec_driver_sql(
            "INSERT INTO model_preheat_tasks (id, source, model_id, requested_revision, "
            "resolved_revision, include_patterns, exclude_patterns, selection_digest, cache_key, "
            "generation_id, target_scope, target_worker_uuids, target_worker_snapshot, "
            "s3_profile_snapshot_encrypted, encryption_key_version, s3_backfill_policy, "
            "s3_profile_id, s3_profile_config_version, created_at, updated_at) VALUES "
            "(1,?,?,'refs/pull/1/head','sha',?,?,?,'ck','gen',"
            "'selected_workers',?,?,'{}','v1','when_missing',7,1,?,?)",
            (
                runtime.source,
                model_id,
                json.dumps(stored_include, separators=(",", ":")),
                json.dumps(stored_exclude, separators=(",", ":")),
                "d" * 64,
                json.dumps([], separators=(",", ":")),
                json.dumps([], separators=(",", ":")),
                now,
                now,
            ),
        )
    command.upgrade(config, "head")
    with engine.connect() as connection:
        row = connection.exec_driver_sql(
            "SELECT request_identity, request_digest FROM model_preheat_tasks WHERE id=1"
        ).one()
    stored_identity = json.loads(row[0])

    assert stored_identity == {
        "source": runtime.source,
        "model_id": runtime.model_path,
        "requested_revision": runtime.requested_revision_path,
        "include_patterns": list(runtime.file_patterns),
        "exclude_patterns": list(runtime.exclude_patterns),
    }
    # 特殊字符等价摘要：迁移回填 digest 必须等于运行时对同一原始请求的 digest。
    assert row[1] == runtime.request_digest
    # patterns 不得二次编码：存量 ``dir/a%20b.bin`` 保持原样（不能变成
    # ``dir/a%2520b.bin``）。
    assert "dir/a%20b.bin" in stored_identity["include_patterns"]
    assert "dir/a%2520b.bin" not in stored_identity["include_patterns"]
    # model_id 中文/空格被运行时规则 percent-encode。
    assert "Qwen%20%E7%BB%84" in stored_identity["model_id"]
    assert stored_identity["include_patterns"] == sorted(
        stored_identity["include_patterns"]
    )


def test_upgrade_does_not_migrate_legacy_inventory_data(tmp_path):
    config = _alembic_config(tmp_path)
    engine = sa.create_engine(config.get_main_option("sqlalchemy.url"))
    _create_baseline(engine)
    with engine.begin() as connection:
        now = "2026-08-20 00:00:00"
        connection.exec_driver_sql(
            "INSERT INTO model_preheat_cached_models (id, profile_id, profile_config_version, "
            "cache_key, source, model_id, resolved_revision, include_patterns, exclude_patterns, "
            "generation_id, ready_path, manifest_path, manifest_digest, file_count, total_size, "
            "manifest_state, last_verified_at, created_at, updated_at) VALUES "
            "(?,7,1,'ck','modelscope','Qwen/Qwen3-32B','sha','[]','[]','gen','r','m',?,1,1024,"
            "'valid',?,?,?)",
            (1, DIGEST, now, now, now),
        )
    with engine.connect() as connection:
        assert (
            connection.exec_driver_sql(
                "SELECT count(*) FROM model_preheat_cached_models"
            ).scalar_one()
            == 1
        )
    command.upgrade(config, "head")
    with engine.connect() as connection:
        # 旧 inventory 行随表删除而消失，不迁移到统一库存。
        assert "model_preheat_cached_models" not in _table_names(engine)
        artifact_rows = connection.exec_driver_sql(
            "SELECT count(*) FROM model_preheat_artifacts"
        ).scalar_one()
    assert artifact_rows == 0


def test_downgrade_restores_empty_structure(tmp_path):
    config, engine = _upgrade_from_baseline(tmp_path)
    with engine.begin() as connection:
        now = "2026-08-20 00:00:00"
        connection.exec_driver_sql(
            "INSERT INTO model_storage_sync_tasks (id, model_file_id, worker_id, worker_uuid, "
            "profile_id, profile_config_version, request_identity, request_digest, source, "
            "model_id, resolved_revision, credential_snapshot_encrypted, encryption_key_version, "
            "state, file_count, total_size, created_at, updated_at) VALUES "
            "(1,1,1,'w',7,1,'{}',?,'modelscope','Qwen/Qwen3-32B','sha','{}','v1','pending',1,1,?,?)",
            (DIGEST, now, now),
        )
    with engine.connect() as connection:
        assert (
            connection.exec_driver_sql(
                "SELECT count(*) FROM model_storage_sync_tasks"
            ).scalar_one()
            == 1
        )
    command.downgrade(config, BASELINE_HEAD)
    tables = _table_names(engine)
    for table in DROPPED_TABLES:
        assert table in tables, f"downgrade should restore {table}"
    for table in NEW_TABLES:
        assert table not in tables, f"downgrade should drop {table}"
    with engine.connect() as connection:
        version = connection.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one()
    assert version == BASELINE_HEAD
    profile_cols = {
        c["name"] for c in sa.inspect(engine).get_columns("model_preheat_s3_profiles")
    }
    assert "is_default" in profile_cols
    task_cols = {
        c["name"] for c in sa.inspect(engine).get_columns("model_preheat_tasks")
    }
    assert "cache_key" in task_cols
    assert "request_digest" not in task_cols


def _migration_columns() -> dict:
    files = (
        MIGRATION_ROOT / f"2026_08_20_1000-{UNIFY_REVISION}_unify_model_storage.py",
        MIGRATION_ROOT
        / f"2026_08_20_1100-{DEDUPE_REVISION}_add_model_storage_sync_dedupe_slots.py",
        MIGRATION_ROOT
        / f"2026_08_22_1000-{LEASE_REVISION}_add_model_storage_sync_lease.py",
        MIGRATION_ROOT
        / f"2026_08_22_1200-{DOWNLOAD_REVISION}_pin_model_file_download_claim.py",
        MIGRATION_ROOT
        / f"2026_08_22_1300-{PROFILE_PIN_REVISION}_protect_download_execution_profile.py",
    )
    # 多文件对同一表的列定义需**合并**（create_table 全量 + add_column 增量），
    # 而不是后者覆盖前者。
    merged: dict = {}
    for path in files:
        for name, columns in _migration_file_columns(path).items():
            merged.setdefault(name, set()).update(columns)
    # 任务 4 使用 batch_alter_table；AST 通用解析器无法从 batch_op.add_column
    # 的局部变量反推出表名，在这里显式合并其固定列集合。
    merged.setdefault("model_file_download_executions", set()).update(
        {"resolved_revision", "artifact_id", "manifest_path", "artifact_total_size"}
    )
    return merged


def _migration_file_columns(path: Path) -> dict:
    tree = ast.parse(path.read_text())
    columns: dict = {}
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Attribute) or not call.args:
            continue
        if call.func.attr == "create_table" and isinstance(call.args[0], ast.Constant):
            columns[call.args[0].value] = {
                arg.args[0].value
                for arg in call.args[1:]
                if isinstance(arg, ast.Call)
                and isinstance(arg.func, ast.Attribute)
                and arg.func.attr == "Column"
                and arg.args
                and isinstance(arg.args[0], ast.Constant)
            }
        # op.add_column("table", sa.Column("col", ...))：增量加列 migration。
        elif call.func.attr == "add_column" and isinstance(call.args[0], ast.Constant):
            table = call.args[0].value
            new_columns = set()
            for arg in call.args[1:]:
                if (
                    isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Attribute)
                    and arg.func.attr == "Column"
                    and arg.args
                    and isinstance(arg.args[0], ast.Constant)
                ):
                    new_columns.add(arg.args[0].value)
            columns.setdefault(table, set()).update(new_columns)
    return columns


def test_new_tables_match_schema_on_all_supported_dialects():
    from gpustack.schemas import model_files  # noqa: F401
    from gpustack.schemas import model_file_download_executions as download
    from gpustack.schemas import model_preheat_s3_profiles  # noqa: F401
    from gpustack.schemas import model_preheats
    from gpustack.schemas import model_storage_sync
    from gpustack.schemas import workers  # noqa: F401

    migration_columns = _migration_columns()
    models = (
        model_preheats.ModelPreheatArtifact,
        model_storage_sync.ModelStorageSyncTask,
        model_storage_sync.ModelStorageSyncTaskDedupeSlot,
        download.ModelFileDownloadExecution,
        download.ModelFileDownloadExecutionProfilePin,
    )
    for model in models:
        assert migration_columns[model.__tablename__] == set(
            model.__table__.columns.keys()
        ), f"column mismatch for {model.__tablename__}"
        for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
            assert str(CreateTable(model.__table__).compile(dialect=dialect))


def test_altered_tables_compile_on_all_supported_dialects():
    from gpustack.schemas import model_files
    from gpustack.schemas import model_preheat_distribution_policies
    from gpustack.schemas import model_preheat_s3_profiles
    from gpustack.schemas import model_preheats
    from gpustack.schemas import workers

    models = (
        model_files.ModelFile,
        model_preheat_s3_profiles.ModelPreheatS3Profile,
        model_preheats.ModelPreheatTask,
        model_preheat_distribution_policies.ModelPreheatDistributionPolicy,
        workers.Worker,
    )
    for model in models:
        for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
            assert str(CreateTable(model.__table__).compile(dialect=dialect))


def test_task_and_policy_use_request_identity_not_cache_key():
    from gpustack.schemas import model_preheat_distribution_policies
    from gpustack.schemas import model_preheats

    task_cols = model_preheats.ModelPreheatTask.__table__.columns
    assert "cache_key" not in task_cols
    assert "request_identity" in task_cols and "request_digest" in task_cols
    assert task_cols["artifact_id"].nullable

    policy_cols = (
        model_preheat_distribution_policies.ModelPreheatDistributionPolicy.__table__.columns
    )
    assert "cache_key" not in policy_cols
    assert "request_identity" in policy_cols and "request_digest" in policy_cols


# ---------------------------------------------------------------------------
# Alembic 迁移路径（而非仅最终 SQLModel 结构）的 PG/MySQL DDL 渲染校验。
# 无需外部实库：用 Alembic MigrationContext 按方言离线渲染本收敛 migration 的
# upgrade/downgrade DDL 操作。数据回填（backfill）在 SQLite 在线路径已由其它用例
# 验证；此处数据操作置空，专注校验跨库 DDL 路径可编译（例如 MySQL 要求 VARCHAR
# 必须有长度，downgrade 重建的 model_cache_tasks 若缺长度会在此暴露）。
# ---------------------------------------------------------------------------
def _load_unify_migration_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "gpustack/migrations/versions/2026_08_20_1000-c1d2e3f4a5b6_unify_model_storage.py"
    )
    spec = importlib.util.spec_from_file_location("_unify_model_storage", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_profile_pin_migration_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "gpustack/migrations/versions/2026_08_22_1300-a5b6c7d8e9f0_protect_download_execution_profile.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_protect_download_execution_profile", str(path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DialectConnection:
    """暴露 dialect 的最小连接替身，供 Alembic 离线 DDL 渲染。"""

    def __init__(self, dialect):
        self.dialect = dialect


def _render_migration_ddl(dialect, migration_fn):
    from alembic import op

    buffer = io.StringIO()
    context = MigrationContext.configure(
        _DialectConnection(dialect),
        opts={"as_sql": True, "output_buffer": buffer},
    )
    with op.Operations.context(context):
        migration_fn()
    return buffer.getvalue()


@pytest.mark.parametrize(
    "dialect",
    [postgresql.dialect(), mysql.dialect()],
    ids=["postgresql", "mysql"],
)
def test_migration_ddl_path_renders_on_all_supported_dialects(dialect):
    module = _load_unify_migration_module()
    # 数据回填在 SQLite 在线路径已验证；此处置空以专注 DDL 路径渲染。
    original_tasks = module._backfill_tasks_request_identity
    original_policies = module._backfill_policies_request_identity
    module._backfill_tasks_request_identity = lambda: None
    module._backfill_policies_request_identity = lambda: None
    try:
        upgrade_sql = _render_migration_ddl(dialect, module.upgrade)
        downgrade_sql = _render_migration_ddl(dialect, module.downgrade)
    finally:
        module._backfill_tasks_request_identity = original_tasks
        module._backfill_policies_request_identity = original_policies

    assert upgrade_sql.strip(), "upgrade DDL rendered empty"
    assert downgrade_sql.strip(), "downgrade DDL rendered empty"
    combined = upgrade_sql + "\n" + downgrade_sql
    for probe in (
        "model_preheat_artifacts",
        "model_storage_sync_tasks",
        "model_file_download_executions",
        "model_preheat_cached_models",  # downgrade 重建旧表
        "request_identity",
        "default_slot",
        "model_storage_protocol_version",
    ):
        assert probe in combined, f"expected {probe!r} in {dialect.name} migration DDL"


@pytest.mark.parametrize(
    "dialect",
    [postgresql.dialect(), mysql.dialect()],
    ids=["postgresql", "mysql"],
)
def test_profile_pin_migration_ddl_renders_on_all_supported_dialects(dialect):
    module = _load_profile_pin_migration_module()
    combined = (
        _render_migration_ddl(dialect, module.upgrade)
        + "\n"
        + (_render_migration_ddl(dialect, module.downgrade))
    )
    assert "model_file_download_execution_profile_pins" in combined
    assert "ON DELETE RESTRICT" in combined
    assert "ON DELETE CASCADE" in combined
