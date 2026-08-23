"""``worker-local-s3-*`` 启动引导：幂等创建/更新系统 S3 Profile。

任务 2 步骤 4（设计文档 §5.2）：

- 完整 ``worker-local-s3-*`` 配置按 ``provisioning_key=worker_local_s3`` 幂等创建或更新
  系统 Profile；默认 URI 为 ``s3://bd-wind/model-storage``；
- ``worker_local_s3_modelscope_fallback`` 仅作为首次创建时
  ``source_fallback_enabled`` 的默认值；
- 仅当系统中当前没有默认 Profile 时才把系统 Profile 占为 ``global``；
  UI 选择手工默认后，重启不得抢回默认状态；
- 连接或凭据变化时递增 ``config_version`` 并重置连通性为 ``pending``；
- 默认槽位在同一事务内转移，依靠普通唯一约束保证跨数据库最多一个默认 Profile；
- 系统 Profile 不允许直接编辑/删除（见路由），移除启动参数不自动删除已落库 Profile。
"""

import logging
from typing import Optional
from urllib.parse import urlparse

from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
)
from gpustack.schemas.model_preheat_s3_profiles import (
    DEFAULT_SLOT_GLOBAL,
    PROVISIONING_KEY_WORKER_LOCAL_S3,
    ModelPreheatS3ConnectivityStateEnum,
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
    ModelPreheatS3ProvisioningSourceEnum,
    model_preheat_s3_storage_key,
    normalize_model_preheat_s3_prefix,
)
from gpustack.server.model_storage_credential_key import (
    ModelStorageCredentialKeyError,
)

logger = logging.getLogger(__name__)

# 系统引导 Profile 的稳定名称与默认 URI（设计文档 §5.2）。
LOCAL_S3_PROFILE_NAME = "Local S3"
DEFAULT_LOCAL_S3_URI = "s3://bd-wind/model-storage"

# 引导时由启动参数管理的连接字段。TLS、寻址和模型源回退可由 UI 在系统
# Profile 上调整，重启时必须保留其数据库值。
_BOOTSTRAP_CONNECTION_FIELDS = (
    "endpoint",
    "bucket",
    "prefix",
    "region",
)


def _is_default_slot_constraint_error(exc: IntegrityError) -> bool:
    """识别 INSERT 冲突是否**仅**由 ``default_slot`` 唯一约束（并发抢占 global）引起。

    解析各驱动的错误文本：

    - SQLite 形如 ``UNIQUE constraint failed: model_preheat_s3_profiles.default_slot``；
    - PostgreSQL 形如 ``duplicate key value violates unique constraint
      "uix_model_preheat_s3_profiles_default_slot"``；
    - MySQL 1062 的 ``Duplicate entry 'global' for key ...`` 无法从文本区分列，
      返回 False 交由调用方按“系统 Profile 是否已存在”保守处理
      （MySQL 下本路径仅首次并发创建才可能触发，复用/上抛均安全）。
    """
    texts = [str(exc)]
    orig = getattr(exc, "orig", None)
    while orig is not None:
        texts.append(str(orig))
        orig = getattr(orig, "orig", None)
    combined = "\n".join(texts)
    if "uix_model_preheat_s3_profiles_default_slot" in combined:
        return True
    return (
        "UNIQUE constraint failed: model_preheat_s3_profiles.default_slot" in combined
    )


def parse_local_s3_target(config) -> Optional[dict]:
    """把 ``worker-local-s3-*`` 启动参数解析为系统 Profile 的目标字段。

    返回 ``None`` 表示未完整配置（不触发引导）。返回的 dict 只包含
    endpoint/bucket/prefix/tls_enabled/tls_verify/region/use_virtual_hosted_style/
    source_fallback_enabled 以及 access_key/secret_key 明文（供调用方加密）。
    """
    host = (getattr(config, "worker_local_s3_host", "") or "").strip()
    access_key = getattr(config, "worker_local_s3_access_key", "") or ""
    secret_key = getattr(config, "worker_local_s3_secret_key", "") or ""
    prefix_uri = (
        getattr(config, "worker_local_s3_modelscope_prefix", "") or ""
    ).strip()
    if not host or not access_key or not secret_key:
        return None
    if not prefix_uri.startswith("s3://"):
        # 未提供合法 s3:// URI 时，回退默认 URI 的 bucket/prefix。
        prefix_uri = DEFAULT_LOCAL_S3_URI

    endpoint = _build_endpoint(host, getattr(config, "worker_local_s3_ssl", False))
    parsed = urlparse(prefix_uri)
    bucket = parsed.netloc
    prefix = normalize_model_preheat_s3_prefix(parsed.path.strip("/"))
    if not endpoint or not bucket:
        return None
    return {
        "endpoint": endpoint,
        "bucket": bucket,
        "prefix": prefix,
        "tls_enabled": _endpoint_is_https(endpoint),
        "tls_verify": True,
        "region": (getattr(config, "worker_local_s3_region", "") or None) or None,
        "use_virtual_hosted_style": bool(
            getattr(config, "worker_local_s3_use_virtual_hosted_style", True)
        ),
        "source_fallback_enabled": bool(
            getattr(config, "worker_local_s3_modelscope_fallback", True)
        ),
        "access_key": access_key,
        "secret_key": secret_key,
    }


def _build_endpoint(host: str, ssl: bool) -> str:
    if "://" in host:
        parsed = urlparse(host)
        if parsed.scheme in {"http", "https"}:
            return host
        # 非 http(s) 显式 scheme：按 ssl 归一到 https/http。
        scheme = "https" if ssl else "http"
        return f"{scheme}://{parsed.netloc or host}"
    netloc = host.rstrip("/")
    scheme = "https" if ssl else "http"
    return f"{scheme}://{netloc}"


def _endpoint_is_https(endpoint: str) -> bool:
    return urlparse(endpoint).scheme == "https"


def _field_equal(field: str, stored, target_value) -> bool:
    """比较连接字段，忽略 None 与空字符串的等价差异。

    例如 ``region`` 目标为 ``None``，但数据库将非 NULL String 存为 ``""``；
    两者语义等价，不应触发 config_version 递增。
    """
    if field == "region":
        return (stored or None) == (target_value or None)
    return stored == target_value


async def _maybe_occupy_default_slot(
    session: AsyncSession,
    profile: ModelPreheatS3Profile,
    want_default: bool,
) -> None:
    """重启不抢回：仅当系统当前无默认且本 Profile 未持有默认时占 global。

    单独提交，冲突（并发抢占）时回退非默认，不影响已提交的连接/凭据更新。
    """
    if (
        not want_default
        or profile.lifecycle_state != ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE
        or profile.default_slot == DEFAULT_SLOT_GLOBAL
    ):
        return
    try:
        profile.default_slot = DEFAULT_SLOT_GLOBAL
        session.add(profile)
        await session.commit()
    except IntegrityError:
        await session.rollback()
        await session.refresh(profile)


async def _find_system_profile(
    session: AsyncSession,
) -> Optional[ModelPreheatS3Profile]:
    profile_id = (
        await session.exec(
            select(ModelPreheatS3Profile.id).where(
                ModelPreheatS3Profile.provisioning_key
                == PROVISIONING_KEY_WORKER_LOCAL_S3
            )
        )
    ).first()
    if profile_id is None:
        return None
    return await session.get(ModelPreheatS3Profile, profile_id)


async def _adopt_system_winner_or_raise(
    session: AsyncSession,
    exc: IntegrityError,
    want_default: bool,
) -> ModelPreheatS3Profile:
    """首次创建 INSERT 失败（非 default_slot 冲突，或回退后再次失败）后的统一处理。

    rollback 后按 ``provisioning_key`` 查询系统 Profile：

    - 存在（多 Server 并发下对手已抢先创建）：复用获胜行，并在需要时补占
      ``global`` 槽位，视为成功；
    - 不存在（如手工 Profile 占用名称等**真实**完整性错误）：向上抛出原
      异常，绝不伪装成功。
    """
    await session.rollback()
    concurrent = await _find_system_profile(session)
    if concurrent is None:
        raise exc
    profile = concurrent
    await _maybe_occupy_default_slot(session, profile, want_default)
    return profile


async def bootstrap_worker_local_s3_profile(
    config,
    session: AsyncSession,
    cipher: ModelPreheatCredentialCipher,
    *,
    create_connectivity_check: Optional[object] = None,
) -> Optional[ModelPreheatS3Profile]:
    """按 ``provisioning_key`` 幂等引导系统 Profile 并返回其持久对象。

    ``create_connectivity_check`` 是可选的异步工厂（例如
    :func:`create_or_reuse_connectivity_check`）；连接/凭据变化且新增或更新了 Profile
    时调用它登记一次待检测。未配置 Local S3 时返回 ``None``。
    """
    target = parse_local_s3_target(config)
    if target is None:
        return None

    if not cipher.current_key:
        # 设计文档 §5.2 规则 6：配置了 Local S3 但无法安全加密时启动失败。
        raise ModelStorageCredentialKeyError("credential_encryption_unavailable")

    access_key_encrypted = cipher.encrypt(target["access_key"])
    secret_key_encrypted = cipher.encrypt(target["secret_key"])
    target_storage_key = model_preheat_s3_storage_key(
        target["endpoint"], target["bucket"]
    )

    async def _stored_credentials(profile):
        try:
            access = cipher.decrypt(profile.access_key_encrypted)
            secret = cipher.decrypt(profile.secret_key_encrypted)
        except Exception:
            # 无法解密（旧 key 版本缺失等）时按“凭据已变化”处理，重加密覆盖。
            return (None, None)
        return (access, secret)

    existing = (
        await session.exec(
            select(
                ModelPreheatS3Profile.id,
                ModelPreheatS3Profile.endpoint,
                ModelPreheatS3Profile.bucket,
                ModelPreheatS3Profile.prefix,
                ModelPreheatS3Profile.tls_enabled,
                ModelPreheatS3Profile.tls_verify,
                ModelPreheatS3Profile.region,
                ModelPreheatS3Profile.use_virtual_hosted_style,
                ModelPreheatS3Profile.access_key_encrypted,
                ModelPreheatS3Profile.secret_key_encrypted,
                ModelPreheatS3Profile.source_fallback_enabled,
                ModelPreheatS3Profile.default_slot,
                ModelPreheatS3Profile.lifecycle_state,
            ).where(
                ModelPreheatS3Profile.provisioning_key
                == PROVISIONING_KEY_WORKER_LOCAL_S3
            )
        )
    ).first()

    any_default = (
        await session.exec(
            select(ModelPreheatS3Profile).where(
                ModelPreheatS3Profile.default_slot == DEFAULT_SLOT_GLOBAL,
                ModelPreheatS3Profile.lifecycle_state
                == ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE,
            )
        )
    ).first() is not None
    # 仅当系统当前没有默认 Profile 时，引导 Profile 才占 global；否则不抢回。
    active_manual_collision = (
        await session.exec(
            select(ModelPreheatS3Profile.id).where(
                ModelPreheatS3Profile.active_storage_key == target_storage_key,
                or_(
                    ModelPreheatS3Profile.provisioning_key.is_(None),
                    ModelPreheatS3Profile.provisioning_key
                    != PROVISIONING_KEY_WORKER_LOCAL_S3,
                ),
            )
        )
    ).first()
    want_default = not any_default and active_manual_collision is None

    if existing is None:
        profile = ModelPreheatS3Profile(
            name=LOCAL_S3_PROFILE_NAME,
            description="System-managed Local S3 model store (worker-local-s3)",
            endpoint=target["endpoint"],
            bucket=target["bucket"],
            prefix=target["prefix"],
            tls_enabled=target["tls_enabled"],
            tls_verify=target["tls_verify"],
            region=target["region"],
            use_virtual_hosted_style=target["use_virtual_hosted_style"],
            access_key_encrypted=access_key_encrypted,
            secret_key_encrypted=secret_key_encrypted,
            encryption_key_version=cipher.current_key_version,
            provisioning_source=ModelPreheatS3ProvisioningSourceEnum.WORKER_LOCAL_S3,
            provisioning_key=PROVISIONING_KEY_WORKER_LOCAL_S3,
            system_managed=True,
            lifecycle_state=(
                ModelPreheatS3ProfileLifecycleStateEnum.MAINTENANCE
                if active_manual_collision is not None
                else ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE
            ),
            active_storage_key=(
                None if active_manual_collision is not None else target_storage_key
            ),
            default_slot=DEFAULT_SLOT_GLOBAL if want_default else None,
            source_fallback_enabled=target["source_fallback_enabled"],
            connectivity_state=ModelPreheatS3ConnectivityStateEnum.PENDING,
        )
        session.add(profile)
        try:
            await session.commit()
        except IntegrityError as exc:
            if _is_default_slot_constraint_error(exc):
                # 仅 default_slot 冲突（与手工 Profile 并发抢 global）：回退为
                # “非默认创建成功”，不抛出。但多 Server 双 stale 下，第二次
                # INSERT（default_slot=None）可能再撞 provisioning_key/name
                # （对手已创建系统 Profile）：必须按 provisioning_key 复用获胜行，
                # 只有系统 Profile 确实不存在时才上抛真实冲突。
                await session.rollback()
                profile.default_slot = None
                session.add(profile)
                try:
                    await session.commit()
                except IntegrityError as second_exc:
                    profile = await _adopt_system_winner_or_raise(
                        session, second_exc, want_default
                    )
                await session.refresh(profile)
            else:
                profile = await _adopt_system_winner_or_raise(
                    session, exc, want_default
                )
        else:
            await session.refresh(profile)
        if create_connectivity_check is not None:
            await create_connectivity_check(session, profile)
            await session.commit()
        return profile

    # 已存在：仅启动参数管理的连接字段和凭据可被引导更新。TLS、寻址及模型
    # 源回退由 UI 管理，必须保留数据库值；因此 fallback 单独变化不会使配置
    # 版本递增或让连通性失效。
    connection_changed = any(
        not _field_equal(field, getattr(existing, field), target[field])
        for field in _BOOTSTRAP_CONNECTION_FIELDS
    )
    stored_access, stored_secret = await _stored_credentials(existing)
    # 凭据比较基于解密后的明文，避免 AES-GCM 每次 nonce 不同导致的“假变化”。
    credential_changed = (
        stored_access != target["access_key"] or stored_secret != target["secret_key"]
    )
    existing_id = existing.id
    if not connection_changed and not credential_changed:
        existing_profile = await session.get(ModelPreheatS3Profile, existing_id)
        if (
            active_manual_collision is not None
            and existing_profile.lifecycle_state
            != ModelPreheatS3ProfileLifecycleStateEnum.MAINTENANCE
        ):
            existing_profile.lifecycle_state = (
                ModelPreheatS3ProfileLifecycleStateEnum.MAINTENANCE
            )
            existing_profile.active_storage_key = None
            existing_profile.default_slot = None
            session.add(existing_profile)
            await session.commit()
            await session.refresh(existing_profile)
        # 配置未变但系统当前无默认 Profile 时，仍需占 global（重启不抢回 UI 手工默认）。
        await _maybe_occupy_default_slot(session, existing_profile, want_default)
        return existing_profile

    existing_profile = await session.get(ModelPreheatS3Profile, existing_id)

    # 维护由管理员显式控制；启动参数刷新凭据和连接绝不把它自动恢复为 active。
    desired_lifecycle = existing_profile.lifecycle_state
    if active_manual_collision is not None:
        desired_lifecycle = ModelPreheatS3ProfileLifecycleStateEnum.MAINTENANCE

    existing_profile.endpoint = target["endpoint"]
    existing_profile.bucket = target["bucket"]
    existing_profile.prefix = target["prefix"]
    existing_profile.region = target["region"]
    existing_profile.access_key_encrypted = access_key_encrypted
    existing_profile.secret_key_encrypted = secret_key_encrypted
    existing_profile.encryption_key_version = cipher.current_key_version
    existing_profile.lifecycle_state = desired_lifecycle
    existing_profile.active_storage_key = (
        target_storage_key
        if desired_lifecycle == ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE
        else None
    )
    if desired_lifecycle == ModelPreheatS3ProfileLifecycleStateEnum.MAINTENANCE:
        existing_profile.default_slot = None
    existing_profile.config_version += 1
    existing_profile.connectivity_state = ModelPreheatS3ConnectivityStateEnum.PENDING
    existing_profile.last_connectivity_check_id = None
    existing_profile.last_connectivity_checked_at = None
    session.add(existing_profile)
    await session.commit()
    await session.refresh(existing_profile)

    # 重启不抢回：只在系统当前没有默认 Profile 且本 Profile 尚未持有默认时占位。
    # 默认槽位单独提交，失败不影响已提交的连接/凭据更新。
    await _maybe_occupy_default_slot(session, existing_profile, want_default)

    if create_connectivity_check is not None:
        await create_connectivity_check(session, existing_profile)
        await session.commit()
    return existing_profile
