"""任务 2 步骤 4：``worker-local-s3-*`` 系统 Profile 引导的定向测试。

覆盖：
- 幂等创建（provisioning_key=worker_local_s3、system、默认 URI bucket/prefix、
  fallback 映射）与重启不重复创建；
- 连接/凭据变化递增 config_version 并重置连通性为 pending；
- 仅当系统中当前没有默认 Profile 时占 global；UI 选择手工默认后重启不抢回；
- default_slot 唯一约束在 SQLite 并发下保证最多一个默认 Profile；
- 未完整配置 worker-local-s3-* 时不引导。
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
import pytest
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
    generate_model_preheat_credential_key,
)
from gpustack.schemas.model_preheat_s3_profiles import (
    DEFAULT_SLOT_GLOBAL,
    ModelPreheatS3ConnectivityStateEnum,
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
    ModelPreheatS3ProvisioningSourceEnum,
    model_preheat_s3_storage_key,
)
from gpustack.server.model_storage_bootstrap import (
    DEFAULT_LOCAL_S3_URI,
    LOCAL_S3_PROFILE_NAME,
    bootstrap_worker_local_s3_profile,
    parse_local_s3_target,
)


def _config(**overrides):
    base = dict(
        worker_local_s3_host="s3.internal.example.com",
        worker_local_s3_access_key="local-access",
        worker_local_s3_secret_key="local-secret",
        worker_local_s3_ssl=False,
        worker_local_s3_use_virtual_hosted_style=True,
        worker_local_s3_region="",
        worker_local_s3_modelscope_prefix=DEFAULT_LOCAL_S3_URI,
        worker_local_s3_modelscope_fallback=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _cipher():
    return ModelPreheatCredentialCipher(
        current_key=generate_model_preheat_credential_key(),
        current_key_version="v1",
        old_keys=None,
    )


@asynccontextmanager
async def _session(tmp_path, name="bootstrap.db"):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path}/{name}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    # expire_on_commit=False：避免提交后访问对象属性触发隐式异步刷新
    # （standalone 场景中同步属性访问会 MissingGreenlet）。
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
    await engine.dispose()


def _manual_profile(name="manual-default", **overrides):
    base = dict(
        name=name,
        endpoint="https://manual.example.com",
        bucket="manual",
        access_key_encrypted={"ciphertext": "a"},
        secret_key_encrypted={"ciphertext": "b"},
        encryption_key_version="v1",
    )
    base.update(overrides)
    return ModelPreheatS3Profile(**base)


def test_parse_local_s3_target_maps_config():
    target = parse_local_s3_target(_config())
    assert target is not None
    assert target["endpoint"] == "http://s3.internal.example.com"
    assert target["bucket"] == "bd-wind"
    assert target["prefix"] == "model-storage"
    assert target["tls_enabled"] is False
    assert target["source_fallback_enabled"] is True
    assert target["access_key"] == "local-access"
    assert target["secret_key"] == "local-secret"


def test_parse_local_s3_target_fallback_false_and_ssl():
    target = parse_local_s3_target(
        _config(
            worker_local_s3_ssl=True,
            worker_local_s3_modelscope_fallback=False,
            worker_local_s3_modelscope_prefix="s3://custom-bucket/cache/prefix",
        )
    )
    assert target["endpoint"] == "https://s3.internal.example.com"
    assert target["tls_enabled"] is True
    assert target["bucket"] == "custom-bucket"
    assert target["prefix"] == "cache/prefix"
    assert target["source_fallback_enabled"] is False


def test_parse_local_s3_target_not_configured():
    assert parse_local_s3_target(_config(worker_local_s3_access_key="")) is None
    assert parse_local_s3_target(_config(worker_local_s3_host="")) is None


def test_bootstrap_creates_system_profile_and_takes_default_when_none(tmp_path):
    async def run():
        async with _session(tmp_path) as session:
            profile = await bootstrap_worker_local_s3_profile(
                _config(), session, _cipher()
            )
            assert profile is not None
            assert profile.name == LOCAL_S3_PROFILE_NAME
            assert profile.provisioning_source == (
                ModelPreheatS3ProvisioningSourceEnum.WORKER_LOCAL_S3
            )
            assert profile.provisioning_key == "worker_local_s3"
            assert profile.system_managed is True
            assert profile.default_slot == DEFAULT_SLOT_GLOBAL
            assert profile.bucket == "bd-wind"
            assert profile.prefix == "model-storage"
            assert profile.tls_enabled is False
            assert profile.tls_verify is True
            assert profile.use_virtual_hosted_style is True
            assert profile.source_fallback_enabled is True
            assert profile.config_version == 1
            assert (
                profile.connectivity_state
                == ModelPreheatS3ConnectivityStateEnum.PENDING
            )

    asyncio.run(run())


def test_bootstrap_is_idempotent_on_restart(tmp_path):
    async def run():
        async with _session(tmp_path) as session:
            cipher = _cipher()
            first = await bootstrap_worker_local_s3_profile(_config(), session, cipher)
            second = await bootstrap_worker_local_s3_profile(_config(), session, cipher)
            assert first.id == second.id
            rows = list((await session.exec(select(ModelPreheatS3Profile))).all())
            assert len(rows) == 1
            # 配置未变化不递增 config_version。
            assert second.config_version == 1

    asyncio.run(run())


def test_bootstrap_does_not_reactivate_explicit_maintenance_profile(tmp_path):
    async def run():
        async with _session(tmp_path) as session:
            cipher = _cipher()
            first = await bootstrap_worker_local_s3_profile(_config(), session, cipher)
            first.lifecycle_state = ModelPreheatS3ProfileLifecycleStateEnum.MAINTENANCE
            first.active_storage_key = None
            first.default_slot = None
            session.add(first)
            await session.commit()

            restarted = await bootstrap_worker_local_s3_profile(
                _config(), session, cipher
            )

            assert restarted.id == first.id
            assert (
                restarted.lifecycle_state
                == ModelPreheatS3ProfileLifecycleStateEnum.MAINTENANCE
            )
            assert restarted.active_storage_key is None
            assert restarted.default_slot is None

    asyncio.run(run())


def test_bootstrap_same_active_manual_storage_creates_system_maintenance(tmp_path):
    async def run():
        async with _session(tmp_path) as session:
            target = parse_local_s3_target(_config())
            manual = _manual_profile(
                endpoint=target["endpoint"],
                bucket=target["bucket"],
                active_storage_key=model_preheat_s3_storage_key(
                    target["endpoint"], target["bucket"]
                ),
                default_slot=DEFAULT_SLOT_GLOBAL,
            )
            session.add(manual)
            await session.commit()

            system = await bootstrap_worker_local_s3_profile(
                _config(), session, _cipher()
            )

            assert system.system_managed is True
            assert (
                system.lifecycle_state
                == ModelPreheatS3ProfileLifecycleStateEnum.MAINTENANCE
            )
            assert system.active_storage_key is None
            assert system.default_slot is None
            refreshed_manual = await session.get(ModelPreheatS3Profile, manual.id)
            assert refreshed_manual.default_slot == DEFAULT_SLOT_GLOBAL

    asyncio.run(run())


def test_bootstrap_config_change_increments_version_and_resets_connectivity(tmp_path):
    async def run():
        async with _session(tmp_path) as session:
            cipher = _cipher()
            first = await bootstrap_worker_local_s3_profile(_config(), session, cipher)
            first.connectivity_state = ModelPreheatS3ConnectivityStateEnum.AVAILABLE
            first.last_connectivity_check_id = 7
            await session.commit()

            await bootstrap_worker_local_s3_profile(
                _config(worker_local_s3_access_key="rotated-access"),
                session,
                _cipher(),
            )
            refreshed = await session.get(ModelPreheatS3Profile, first.id)
            assert refreshed.config_version == 2
            assert (
                refreshed.connectivity_state
                == ModelPreheatS3ConnectivityStateEnum.PENDING
            )
            assert refreshed.last_connectivity_check_id is None

    asyncio.run(run())


def test_bootstrap_restart_preserves_ui_managed_runtime_s3_options(tmp_path):
    async def run():
        async with _session(tmp_path) as session:
            cipher = _cipher()
            profile = await bootstrap_worker_local_s3_profile(
                _config(), session, cipher
            )
            profile.tls_enabled = False
            profile.tls_verify = False
            profile.use_virtual_hosted_style = False
            profile.source_fallback_enabled = False
            await session.commit()

            refreshed = await bootstrap_worker_local_s3_profile(
                _config(
                    worker_local_s3_host="s3-new.internal.example.com",
                    worker_local_s3_ssl=True,
                    worker_local_s3_use_virtual_hosted_style=True,
                    worker_local_s3_modelscope_fallback=True,
                ),
                session,
                cipher,
            )

            assert refreshed.endpoint == "https://s3-new.internal.example.com"
            assert refreshed.tls_enabled is False
            assert refreshed.tls_verify is False
            assert refreshed.use_virtual_hosted_style is False
            assert refreshed.source_fallback_enabled is False
            assert refreshed.config_version == 2

    asyncio.run(run())


def test_bootstrap_fallback_change_does_not_invalidate_connectivity(tmp_path):
    async def run():
        async with _session(tmp_path) as session:
            cipher = _cipher()
            profile = await bootstrap_worker_local_s3_profile(
                _config(), session, cipher
            )
            profile.connectivity_state = ModelPreheatS3ConnectivityStateEnum.AVAILABLE
            profile.last_connectivity_check_id = 7
            await session.commit()

            refreshed = await bootstrap_worker_local_s3_profile(
                _config(worker_local_s3_modelscope_fallback=False),
                session,
                cipher,
            )

            assert refreshed.source_fallback_enabled is True
            assert refreshed.config_version == 1
            assert (
                refreshed.connectivity_state
                == ModelPreheatS3ConnectivityStateEnum.AVAILABLE
            )
            assert refreshed.last_connectivity_check_id == 7

    asyncio.run(run())


def test_bootstrap_does_not_steal_ui_manual_default(tmp_path):
    async def run():
        async with _session(tmp_path) as session:
            manual = _manual_profile(default_slot=DEFAULT_SLOT_GLOBAL)
            session.add(manual)
            await session.commit()
            manual_id = manual.id

            profile = await bootstrap_worker_local_s3_profile(
                _config(), session, _cipher()
            )
            profile_id = profile.id
            assert profile.default_slot is None
            refreshed_manual = await session.get(ModelPreheatS3Profile, manual_id)
            assert refreshed_manual.default_slot == DEFAULT_SLOT_GLOBAL

            # 重启幂等：仍不抢回。
            await bootstrap_worker_local_s3_profile(_config(), session, _cipher())
            refreshed_manual = await session.get(ModelPreheatS3Profile, manual_id)
            refreshed_profile = await session.get(ModelPreheatS3Profile, profile_id)
            assert refreshed_manual.default_slot == DEFAULT_SLOT_GLOBAL
            assert refreshed_profile.default_slot is None

    asyncio.run(run())


def test_default_slot_unique_constraint_rejects_second_default(tmp_path):
    async def run():
        async with _session(tmp_path) as session:
            first = await bootstrap_worker_local_s3_profile(
                _config(), session, _cipher()
            )
            assert first.default_slot == DEFAULT_SLOT_GLOBAL

            other = _manual_profile(
                name="other-default",
                endpoint="https://other.example.com",
                default_slot=DEFAULT_SLOT_GLOBAL,
            )
            session.add(other)
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()

    asyncio.run(run())


def test_no_default_after_manual_release_lets_bootstrap_take_global(tmp_path):
    async def run():
        async with _session(tmp_path) as session:
            profile = await bootstrap_worker_local_s3_profile(
                _config(), session, _cipher()
            )
            profile_id = profile.id
            profile.default_slot = None
            manual = _manual_profile(default_slot=DEFAULT_SLOT_GLOBAL)
            session.add(manual)
            await session.commit()

            # 用户把手工默认清空，此时重启引导应重新占 global（系统当前无默认）。
            manual.default_slot = None
            await session.commit()
            await bootstrap_worker_local_s3_profile(_config(), session, _cipher())
            refreshed = await session.get(ModelPreheatS3Profile, profile_id)
            assert refreshed.default_slot == DEFAULT_SLOT_GLOBAL

    asyncio.run(run())


def test_bootstrap_unchanged_config_still_occupies_global_when_no_default(
    tmp_path,
):
    """配置未变但系统当前无默认 Profile 时，重启仍需占 global（修复测试掩盖）。

    旧实现仅在“连接/凭据变化”分支末尾占位，配置未变时直接 ``return``，
    导致系统 Profile 丢失默认后无法通过未变化的重启重新占回 global。
    """

    async def run():
        async with _session(tmp_path) as session:
            cipher = _cipher()
            profile = await bootstrap_worker_local_s3_profile(
                _config(), session, cipher
            )
            profile_id = profile.id
            assert profile.default_slot == DEFAULT_SLOT_GLOBAL
            # 已有系统 Profile 且配置未变，但默认槽位被手工 Profile 抢占（先清空系统槽位）。
            profile.default_slot = None
            manual = _manual_profile(default_slot=DEFAULT_SLOT_GLOBAL)
            session.add(manual)
            await session.commit()
            # 用户把手工默认清空，系统当前无默认；配置未变的重启应重新占回 global。
            manual.default_slot = None
            await session.commit()

            second = await bootstrap_worker_local_s3_profile(_config(), session, cipher)
            assert second.id == profile_id
            assert second.config_version == 1
            assert second.default_slot == DEFAULT_SLOT_GLOBAL

    asyncio.run(run())


def test_bootstrap_does_not_fake_success_on_real_name_conflict(tmp_path):
    """名称被手工 Profile 占用且无系统 Profile 时，应把 IntegrityError 向上抛出，
    不能当作 default_slot 冲突伪装成功。"""

    async def run():
        async with _session(tmp_path) as session:
            # 手工 Profile 占用了系统 Profile 的稳定名称（provisioning_key 为 NULL）。
            manual = _manual_profile(
                name=LOCAL_S3_PROFILE_NAME,
                provisioning_key=None,
                provisioning_source=ModelPreheatS3ProvisioningSourceEnum.MANUAL,
                system_managed=False,
            )
            session.add(manual)
            await session.commit()

            with pytest.raises(IntegrityError):
                await bootstrap_worker_local_s3_profile(_config(), session, _cipher())

            system_rows = list(
                (
                    await session.exec(
                        select(ModelPreheatS3Profile).where(
                            ModelPreheatS3Profile.provisioning_key == "worker_local_s3"
                        )
                    )
                ).all()
            )
            assert system_rows == []

    asyncio.run(run())


def test_bootstrap_concurrent_first_create_adopts_winner(tmp_path, monkeypatch):
    """多 Server 首次并发：若另一 Server 已抢先创建系统 Profile，本实例应复用获胜者行，
    而不是把任意 IntegrityError 当 default_slot 冲突、或伪装成功/新建第二行。"""

    class _EmptyResult:
        def first(self):
            return None

        def all(self):
            return []

    async def run():
        async with _session(tmp_path) as session:
            # 预先创建“获胜者”系统 Profile（模拟另一 Server 已提交）。
            winner = _manual_profile(
                name=LOCAL_S3_PROFILE_NAME,
                provisioning_key="worker_local_s3",
                provisioning_source=(
                    ModelPreheatS3ProvisioningSourceEnum.WORKER_LOCAL_S3
                ),
                system_managed=True,
                default_slot=DEFAULT_SLOT_GLOBAL,
            )
            session.add(winner)
            await session.commit()
            winner_id = winner.id

            # 模拟 stale read：第一次 existing 查询返回空，使本实例进入创建路径，
            # 其 INSERT 因 name/provisioning_key 唯一约束冲突而失败，随后复用获胜者。
            original_exec = session.exec
            state = {"consumed": False}

            async def stale_exec(stmt, *args, **kwargs):
                result = await original_exec(stmt, *args, **kwargs)
                if not state["consumed"]:
                    state["consumed"] = True
                    return _EmptyResult()
                return result

            monkeypatch.setattr(session, "exec", stale_exec)
            profile = await bootstrap_worker_local_s3_profile(
                _config(), session, _cipher()
            )

            assert profile.id == winner_id
            assert profile.default_slot == DEFAULT_SLOT_GLOBAL
            system_rows = list(
                (
                    await session.exec(
                        select(ModelPreheatS3Profile).where(
                            ModelPreheatS3Profile.provisioning_key == "worker_local_s3"
                        )
                    )
                ).all()
            )
            assert len(system_rows) == 1

    asyncio.run(run())


def test_bootstrap_first_create_losing_global_race_still_succeeds_non_default(
    tmp_path, monkeypatch
):
    """bootstrap 首次创建时与手工 Profile 并发抢 global 失败，
    必须回退为「非默认创建成功」，只有 default_slot 之外的真实完整性错误才抛出。"""

    class _EmptyResult:
        def first(self):
            return None

        def all(self):
            return []

    async def run():
        async with _session(tmp_path) as session:
            # 模拟 stale read：引导开始时系统尚无默认 Profile（want_default=True），
            # 但 INSERT 时手工 Profile 抢先占用了 global。
            original_exec = session.exec
            state = {"consumed": False}

            async def stale_exec(stmt, *args, **kwargs):
                result = await original_exec(stmt, *args, **kwargs)
                if not state["consumed"]:
                    state["consumed"] = True
                    return _EmptyResult()
                return result

            monkeypatch.setattr(session, "exec", stale_exec)
            # 手工 Profile 抢占 global 槽位。
            manual = _manual_profile(default_slot=DEFAULT_SLOT_GLOBAL)
            session.add(manual)
            await session.commit()

            profile = await bootstrap_worker_local_s3_profile(
                _config(), session, _cipher()
            )
            # 创建成功但回退为非默认；手工 Profile 仍持有 global。
            assert profile is not None
            assert profile.default_slot is None
            assert profile.system_managed is True
            refreshed_manual = await session.get(ModelPreheatS3Profile, manual.id)
            assert refreshed_manual.default_slot == DEFAULT_SLOT_GLOBAL
            # 重启幂等：此时系统已有默认，不再尝试占位。
            second = await bootstrap_worker_local_s3_profile(
                _config(), session, _cipher()
            )
            assert second.id == profile.id
            assert second.default_slot is None

    asyncio.run(run())


def test_bootstrap_double_stale_adopts_winner_on_second_integrity_error(
    tmp_path, monkeypatch
):
    """多 Server 双 stale：首次 INSERT 先撞 default_slot（对手已占 global），
    回退 default_slot=None 的第二次 INSERT 又撞 provisioning_key/name
    （对手已创建系统 Profile）。第二次 IntegrityError 必须 rollback 后按
    provisioning_key 查询，存在则复用获胜行并继续；只有不存在才上抛真实冲突。"""

    class _EmptyResult:
        def first(self):
            return None

        def all(self):
            return []

    async def run():
        async with _session(tmp_path) as session:
            # 对手 Server 已提交系统 Profile 且占 global（本实例两次 INSERT 都冲突）。
            winner = _manual_profile(
                name=LOCAL_S3_PROFILE_NAME,
                provisioning_key="worker_local_s3",
                provisioning_source=(
                    ModelPreheatS3ProvisioningSourceEnum.WORKER_LOCAL_S3
                ),
                system_managed=True,
                default_slot=DEFAULT_SLOT_GLOBAL,
            )
            session.add(winner)
            await session.commit()
            winner_id = winner.id

            # 双 stale read：本实例的 existing 查询返回空，进入创建路径，
            # 首次 INSERT 撞 default_slot、第二次撞 provisioning_key。
            original_exec = session.exec
            state = {"stale_left": 2}

            async def stale_exec(stmt, *args, **kwargs):
                result = await original_exec(stmt, *args, **kwargs)
                if state["stale_left"] > 0:
                    # 前两次 exec 为 existing（系统 Profile）与 any_default（默认 Profile）
                    # 查询，双 stale 使其返回空：本实例据此 want_default=True 且走创建路径。
                    state["stale_left"] -= 1
                    return _EmptyResult()
                return result

            monkeypatch.setattr(session, "exec", stale_exec)
            profile = await bootstrap_worker_local_s3_profile(
                _config(), session, _cipher()
            )
            # 复用对手获胜行（而非抛出 provisioning_key/name 冲突）。
            assert profile is not None
            assert profile.id == winner_id
            # 对手已持 global，本实例不重复占位。
            assert profile.default_slot == DEFAULT_SLOT_GLOBAL
            system_rows = list(
                (
                    await session.exec(
                        select(ModelPreheatS3Profile).where(
                            ModelPreheatS3Profile.provisioning_key == "worker_local_s3"
                        )
                    )
                ).all()
            )
            assert len(system_rows) == 1

    asyncio.run(run())


def test_bootstrap_first_create_real_integrity_error_still_raises(
    tmp_path, monkeypatch
):
    """default_slot 之外的真实完整性错误（如手工 Profile 占用名称）不得伪装成功。"""

    class _EmptyResult:
        def first(self):
            return None

        def all(self):
            return []

    async def run():
        async with _session(tmp_path) as session:
            manual = _manual_profile(
                name=LOCAL_S3_PROFILE_NAME,
                provisioning_key=None,
            )
            session.add(manual)
            await session.commit()

            original_exec = session.exec
            state = {"consumed": False}

            async def stale_exec(stmt, *args, **kwargs):
                result = await original_exec(stmt, *args, **kwargs)
                if not state["consumed"]:
                    state["consumed"] = True
                    return _EmptyResult()
                return result

            monkeypatch.setattr(session, "exec", stale_exec)
            with pytest.raises(IntegrityError):
                await bootstrap_worker_local_s3_profile(_config(), session, _cipher())

    asyncio.run(run())
