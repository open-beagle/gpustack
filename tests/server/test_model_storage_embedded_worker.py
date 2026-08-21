"""任务 2 步骤 4：升级后 embedded Worker 一次性凭据修复的定向测试。

覆盖：
- “旧数据库 + embedded Worker + 无凭据文件”自动恢复：bootstrap_required 的既有
  Worker 在本地无凭据文件时签发一次性凭据并原子写入 0600 文件；
- 重启幂等：本地已有凭据文件时不重复签发/覆盖；
- 远程 Worker 身份隔离不放宽：bootstrap_required 为 False（已有凭据）或
  本地已有凭据文件时，不覆盖既有凭据；不匹配的 UUID 不签发。
"""

import asyncio
import os
import stat
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

import pytest
import gpustack.server.model_preheat_worker_identity as worker_identity_module

# 导入以注册完整 metadata（Worker / ModelPreheatWorkerIdentity 及其依赖表）。
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile  # noqa: F401
from gpustack.schemas.workers import Worker
from gpustack.schemas.model_preheats import ModelPreheatWorkerIdentity
from gpustack.server.model_preheat_worker_identity import (
    issue_embedded_worker_credential_file,
    validate_model_preheat_worker_credential,
)


@asynccontextmanager
async def _session(tmp_path, name="worker.db"):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/{name}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
    await engine.dispose()


async def _worker_and_identity(session, worker_id, worker_uuid, bootstrap_required):
    worker = Worker(
        name=f"worker-{worker_id}",
        hostname=f"worker-{worker_id}",
        ip="127.0.0.1",
        port=10150,
        worker_uuid=worker_uuid,
    )
    session.add(worker)
    await session.flush()
    identity = ModelPreheatWorkerIdentity(
        worker_id=worker.id,
        worker_uuid=worker_uuid,
        bootstrap_required=bootstrap_required,
    )
    session.add(identity)
    await session.commit()
    return worker, identity


def test_embedded_worker_recovers_missing_credential_file(tmp_path):
    async def run():
        credential_path = str(tmp_path / "model_preheat_worker_credential")
        assert not os.path.exists(credential_path)
        async with _session(tmp_path) as session:
            await _worker_and_identity(
                session, 1, "uuid-1", bootstrap_required=True
            )
            written = await issue_embedded_worker_credential_file(
                session, "uuid-1", credential_path
            )
            assert written is True
            assert os.path.exists(credential_path)
            mode = stat.S_IMODE(os.stat(credential_path).st_mode)
            assert mode == 0o600
            token = open(credential_path).read().strip()
            valid = await validate_model_preheat_worker_credential(
                session, token, "uuid-1"
            )
            assert valid is not None

    asyncio.run(run())


def test_embedded_worker_does_not_overwrite_existing_credential_file(tmp_path):
    async def run():
        credential_path = str(tmp_path / "model_preheat_worker_credential")
        # 本地已存在凭据文件（例如上一次注册已轮换），不得覆盖。
        with open(credential_path, "w") as file:
            file.write("existing-credential-token")
        async with _session(tmp_path) as session:
            await _worker_and_identity(
                session, 1, "uuid-1", bootstrap_required=True
            )
            written = await issue_embedded_worker_credential_file(
                session, "uuid-1", credential_path
            )
            assert written is False
            assert open(credential_path).read().strip() == "existing-credential-token"

    asyncio.run(run())


def test_embedded_worker_skips_when_not_bootstrap_required(tmp_path):
    async def run():
        credential_path = str(tmp_path / "model_preheat_worker_credential")
        async with _session(tmp_path) as session:
            await _worker_and_identity(
                session, 1, "uuid-1", bootstrap_required=False
            )
            written = await issue_embedded_worker_credential_file(
                session, "uuid-1", credential_path
            )
            assert written is False
            assert not os.path.exists(credential_path)

    asyncio.run(run())


def test_embedded_worker_does_not_issue_for_unknown_uuid(tmp_path):
    async def run():
        credential_path = str(tmp_path / "model_preheat_worker_credential")
        async with _session(tmp_path) as session:
            await _worker_and_identity(
                session, 1, "uuid-1", bootstrap_required=True
            )
            # 不匹配的 UUID（例如试图接管另一节点的既有身份）不得签发。
            written = await issue_embedded_worker_credential_file(
                session, "uuid-unknown", credential_path
            )
            assert written is False
            assert not os.path.exists(credential_path)

    asyncio.run(run())


def test_embedded_worker_write_failure_is_recoverable(tmp_path, monkeypatch):
    """写盘失败必须可恢复：回滚身份变更（bootstrap_required 复原），
    下次启动可重试，而不是留下 DB 已有凭据但无文件的不可恢复状态。"""

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        worker_identity_module, "_write_credential_file", _boom
    )

    async def run():
        credential_path = str(tmp_path / "model_preheat_worker_credential")
        async with _session(tmp_path) as session:
            worker, identity = await _worker_and_identity(
                session, 1, "uuid-1", bootstrap_required=True
            )
            identity_id = identity.id
            written = await issue_embedded_worker_credential_file(
                session, "uuid-1", credential_path
            )
            assert written is False
            assert not os.path.exists(credential_path)
            # 身份变更已回滚：仍是 bootstrap_required，可再次触发恢复。
            refreshed = (
                await session.exec(
                    select(ModelPreheatWorkerIdentity).where(
                        ModelPreheatWorkerIdentity.id == identity_id
                    )
                )
            ).one()
            assert refreshed.bootstrap_required is True
            assert refreshed.token_hash is None

    asyncio.run(run())


def test_embedded_worker_picks_latest_worker_on_duplicate_uuid(tmp_path):
    """重复 worker_uuid 时取最新（最大 worker.id）的记录绑定凭据。"""

    async def run():
        credential_path = str(tmp_path / "model_preheat_worker_credential")
        async with _session(tmp_path) as session:
            old_worker, old_identity = await _worker_and_identity(
                session, 1, "uuid-dup", bootstrap_required=True
            )
            # 模拟 UUID 被复用：新建一条更大的 worker.id 记录及其身份。
            new_worker = Worker(
                name="worker-new",
                hostname="worker-new",
                ip="127.0.0.1",
                port=10151,
                worker_uuid="uuid-dup",
            )
            session.add(new_worker)
            await session.flush()
            new_identity = ModelPreheatWorkerIdentity(
                worker_id=new_worker.id,
                worker_uuid="uuid-dup",
                bootstrap_required=True,
            )
            session.add(new_identity)
            assert new_worker.id > old_worker.id
            await session.commit()

            written = await issue_embedded_worker_credential_file(
                session, "uuid-dup", credential_path
            )
            assert written is True
            assert os.path.exists(credential_path)
            # 凭据应绑定到最新（最大 id）Worker 的身份。
            new_ref = await session.get(ModelPreheatWorkerIdentity, new_identity.id)
            old_ref = await session.get(ModelPreheatWorkerIdentity, old_identity.id)
            assert new_ref.bootstrap_required is False
            assert new_ref.token_hash is not None
            token = open(credential_path).read().strip()
            valid_new = await validate_model_preheat_worker_credential(
                session, token, "uuid-dup"
            )
            assert valid_new is not None
            # 旧记录身份不应被当作当前凭据持有者。
            assert valid_new.worker_id == new_worker.id

    asyncio.run(run())


def test_embedded_worker_db_commit_failure_cleans_only_fresh_credential_file(
    tmp_path, monkeypatch
):
    """写盘成功但 DB commit 失败：必须删除仅本次创建的凭据文件并回滚，
    使重启可重签；绝不能删除调用前已存在的凭据文件。"""

    async def run():
        credential_path = str(tmp_path / "model_preheat_worker_credential")
        assert not os.path.exists(credential_path)
        async with _session(tmp_path) as session:
            await _worker_and_identity(
                session, 1, "uuid-1", bootstrap_required=True
            )

            commit_calls = {"n": 0}

            async def failing_commit():
                commit_calls["n"] += 1
                if commit_calls["n"] == 1:
                    raise RuntimeError("simulated db commit failure")
                await _original_commit()

            _original_commit = session.commit
            monkeypatch.setattr(session, "commit", failing_commit)
            written = await issue_embedded_worker_credential_file(
                session, "uuid-1", credential_path
            )
            assert written is False
            # 本次创建但无效的凭据文件必须被删除，重启可重签。
            assert not os.path.exists(credential_path)
            refreshed = (
                await session.exec(
                    select(ModelPreheatWorkerIdentity).where(
                        ModelPreheatWorkerIdentity.worker_id == 1
                    )
                )
            ).one()
            assert refreshed.bootstrap_required is True
            assert refreshed.token_hash is None

            # 重启重签仍然成功。
            written = await issue_embedded_worker_credential_file(
                session, "uuid-1", credential_path
            )
            assert written is True
            assert os.path.exists(credential_path)

    asyncio.run(run())
