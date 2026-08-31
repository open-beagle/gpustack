import base64
import binascii
import hashlib
import hmac
import secrets
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import Depends, Header, Request
from sqlalchemy import delete, update
from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import UnauthorizedException
from gpustack.schemas.model_preheats import (
    ModelPreheatWorkerIdentity,
    ModelPreheatWorkerPendingCredential,
)
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_session


WORKER_CREDENTIAL_TTL = timedelta(hours=24)
WORKER_CREDENTIAL_RECOVERY_TTL = WORKER_CREDENTIAL_TTL
WORKER_UPGRADE_PROOF_TTL = timedelta(hours=24)
WORKER_CREDENTIAL_RENEW_WINDOW = timedelta(hours=6)
WORKER_CREDENTIAL_PREFIX = "mpw"
WORKER_CREDENTIAL_HEADER = "X-GPUStack-Worker-Credential"


@dataclass(frozen=True)
class ModelPreheatWorkerPrincipal:
    worker_id: int
    worker_uuid: str
    credential_id: int
    token_version: int


async def issue_model_preheat_worker_credential(
    session, worker_id, worker_uuid, *, reset_pending=False
):
    worker = await session.get(Worker, worker_id)
    if worker is None or worker.worker_uuid != worker_uuid:
        raise ValueError("worker_registration_invalid")
    # 凭据签发与待确认候选的写入必须是同一事务。比较 token_version 和当前
    # token 状态，避免慢请求在另一台 Server 已确认候选后用旧 ORM 状态覆盖它。
    for _ in range(3):
        now = _utcnow()
        identity = (
            await session.exec(
                select(ModelPreheatWorkerIdentity).where(
                    ModelPreheatWorkerIdentity.worker_id == worker_id
                )
            )
        ).first()
        if identity is None:
            identity = ModelPreheatWorkerIdentity(
                worker_id=worker_id,
                worker_uuid=worker_uuid,
                bootstrap_required=False,
                expires_at=now + WORKER_CREDENTIAL_TTL,
            )
            session.add(identity)
            try:
                await session.flush()
            except IntegrityError:
                # 并发首签发由另一请求创建了同一 Worker 的身份，重新读取即可。
                await session.rollback()
                continue
            identity_id = identity.id
            pending_token_version = identity.token_version
        else:
            identity_id = identity.id
            token_hash_condition = (
                ModelPreheatWorkerIdentity.token_hash.is_(None)
                if identity.token_hash is None
                else ModelPreheatWorkerIdentity.token_hash == identity.token_hash
            )
            revoked_at_condition = (
                ModelPreheatWorkerIdentity.revoked_at.is_(None)
                if identity.revoked_at is None
                else ModelPreheatWorkerIdentity.revoked_at == identity.revoked_at
            )
            recovery_token_hash = identity.registration_recovery_token_hash
            recovery_issued_at = identity.registration_recovery_issued_at
            upgrade_proof_hash = identity.upgrade_proof_hash
            upgrade_proof_window_started_at = identity.upgrade_proof_window_started_at
            if reset_pending:
                recovery_token_hash = None
                recovery_issued_at = None
                upgrade_proof_hash = None
                upgrade_proof_window_started_at = None
            elif identity.bootstrap_required:
                upgrade_proof_hash = None
                upgrade_proof_window_started_at = None
            elif (
                recovery_token_hash is None
                or recovery_issued_at is None
                or recovery_issued_at + WORKER_CREDENTIAL_RECOVERY_TTL <= now
            ):
                if _identity_token_is_active(identity, now):
                    recovery_token_hash = identity.token_hash
                    recovery_issued_at = now
                else:
                    recovery_token_hash = None
                    recovery_issued_at = None
            result = await session.exec(
                update(ModelPreheatWorkerIdentity)
                .where(
                    ModelPreheatWorkerIdentity.id == identity_id,
                    ModelPreheatWorkerIdentity.token_version == identity.token_version,
                    ModelPreheatWorkerIdentity.bootstrap_required
                    == identity.bootstrap_required,
                    token_hash_condition,
                    revoked_at_condition,
                )
                .values(
                    worker_uuid=worker_uuid,
                    token_hash=None,
                    registration_recovery_token_hash=recovery_token_hash,
                    registration_recovery_issued_at=recovery_issued_at,
                    upgrade_proof_hash=upgrade_proof_hash,
                    upgrade_proof_window_started_at=upgrade_proof_window_started_at,
                    token_version=identity.token_version + 1,
                    bootstrap_required=False,
                    expires_at=now + WORKER_CREDENTIAL_TTL,
                    revoked_at=None,
                )
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                await session.rollback()
                continue
            pending_token_version = identity.token_version + 1
        if reset_pending:
            await session.exec(
                delete(ModelPreheatWorkerPendingCredential).where(
                    ModelPreheatWorkerPendingCredential.identity_id == identity_id
                )
            )
        await session.exec(
            update(ModelPreheatWorkerIdentity)
            .where(
                ModelPreheatWorkerIdentity.worker_uuid == worker_uuid,
                ModelPreheatWorkerIdentity.worker_id != worker_id,
                ModelPreheatWorkerIdentity.revoked_at.is_(None),
            )
            .values(revoked_at=now)
            .execution_options(synchronize_session=False)
        )
        secret = secrets.token_urlsafe(32)
        # 第三段携带签发代次。旧客户端仍按 ``mpw_<identity_id>_<secret>``
        # 解析身份 ID；新客户端据此丢弃乱序的旧注册响应。
        token = (
            f"{WORKER_CREDENTIAL_PREFIX}_{identity_id}_{pending_token_version}_{secret}"
        )
        session.add(
            ModelPreheatWorkerPendingCredential(
                identity_id=identity_id,
                token_hash=_hash_token(token),
                identity_token_version=pending_token_version,
                expires_at=now + WORKER_CREDENTIAL_TTL,
            )
        )
        await session.commit()
        return token
    raise RuntimeError("worker_credential_rotation_conflict")


async def get_model_preheat_worker_identity(
    request: Request,
    session=Depends(get_session),
    credential: Annotated[Optional[str], Header(alias=WORKER_CREDENTIAL_HEADER)] = None,
):
    if credential is None:
        raise _unauthorized()
    credential_id = _credential_id(credential)
    if credential_id is None:
        raise _unauthorized()
    now = _utcnow()
    identity = await _validated_confirmed_identity(session, credential, None, now)
    if identity is None:
        identity = await _confirm_pending_credential(session, credential, now)
    if identity is None:
        raise _unauthorized()
    principal = _principal(identity)
    request.state.model_preheat_worker = principal
    return principal


async def validate_model_preheat_worker_credential(
    session, credential, worker_uuid, *, require_current=True
):
    now = _utcnow()
    identity = await _validated_confirmed_identity(
        session, credential, worker_uuid, now, require_current=require_current
    )
    if identity is not None:
        return identity
    return await _validated_pending_identity(
        session, credential, worker_uuid, now, require_current=require_current
    )


async def validate_model_preheat_worker_registration_credential(
    session, credential, worker_uuid
):
    """仅用于 Worker 注册；恢复凭据不得进入任何任务或执行接口。"""
    identity = await validate_model_preheat_worker_credential(
        session, credential, worker_uuid
    )
    if identity is not None:
        return identity
    credential_id = _credential_id(credential)
    if credential_id is None:
        return None
    identity = await session.get(ModelPreheatWorkerIdentity, credential_id)
    if (
        identity is None
        or identity.worker_uuid != worker_uuid
        or identity.bootstrap_required
        or identity.revoked_at is not None
        or identity.registration_recovery_token_hash is None
        or identity.registration_recovery_issued_at is None
        or identity.registration_recovery_issued_at + WORKER_CREDENTIAL_RECOVERY_TTL
        <= _utcnow()
        or not hmac.compare_digest(
            identity.registration_recovery_token_hash, _hash_token(credential)
        )
    ):
        return None
    current = (
        await session.exec(
            select(Worker)
            .where(Worker.worker_uuid == worker_uuid)
            .order_by(Worker.id.desc())
        )
    ).first()
    if current is None or current.id != identity.worker_id:
        return None
    return identity


async def bind_new_model_preheat_worker_registration_proof(
    session, worker: Worker, upgrade_proof: str
) -> bool:
    """只为数据库中刚创建的新 Worker 绑定本地生成的恢复 proof。"""
    now = _utcnow()
    if not is_model_preheat_worker_registration_proof(upgrade_proof):
        return False
    identity = (
        await session.exec(
            select(ModelPreheatWorkerIdentity).where(
                ModelPreheatWorkerIdentity.worker_id == worker.id
            )
        )
    ).first()
    if identity is not None:
        return False
    identity = ModelPreheatWorkerIdentity(
        worker_id=worker.id,
        worker_uuid=worker.worker_uuid,
        bootstrap_required=False,
        upgrade_proof_hash=_hash_token(upgrade_proof),
        upgrade_proof_window_started_at=now,
        expires_at=now + WORKER_CREDENTIAL_TTL,
    )
    session.add(identity)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return False
    return True


async def validate_new_model_preheat_worker_registration_proof(
    session, worker: Worker, upgrade_proof: str
) -> bool:
    """仅允许首次创建时绑定的同一 proof 在窗口内补发候选凭据。"""
    now = _utcnow()
    if not is_model_preheat_worker_registration_proof(upgrade_proof):
        return False
    identity = (
        await session.exec(
            select(ModelPreheatWorkerIdentity).where(
                ModelPreheatWorkerIdentity.worker_id == worker.id
            )
        )
    ).first()
    if (
        identity is None
        or identity.bootstrap_required
        or identity.token_hash is not None
        or identity.registration_recovery_token_hash is not None
        or identity.revoked_at is not None
        or identity.upgrade_proof_window_started_at is None
        or identity.upgrade_proof_window_started_at <= now - WORKER_UPGRADE_PROOF_TTL
        or not hmac.compare_digest(
            identity.upgrade_proof_hash or "", _hash_token(upgrade_proof)
        )
    ):
        return False
    return True


async def _validated_confirmed_identity(
    session, credential, worker_uuid, now, *, require_current=True
):
    credential_id = _credential_id(credential)
    if credential_id is None:
        return None
    identity = await session.get(ModelPreheatWorkerIdentity, credential_id)
    if (
        not _identity_token_is_active(identity, now)
        or (worker_uuid is not None and identity.worker_uuid != worker_uuid)
        or not hmac.compare_digest(identity.token_hash, _hash_token(credential))
    ):
        return None
    if require_current and not await _identity_is_current_worker(session, identity):
        return None
    identity_id = identity.id
    renew_required = identity.expires_at <= now + WORKER_CREDENTIAL_RENEW_WINDOW
    if renew_required:
        await _renew_model_preheat_worker_credential(session, identity, now)
        return await session.get(ModelPreheatWorkerIdentity, identity_id)
    return identity


async def _renew_model_preheat_worker_credential(session, identity, now):
    if (
        identity.expires_at is None
        or identity.expires_at > now + WORKER_CREDENTIAL_RENEW_WINDOW
    ):
        return False
    result = await session.exec(
        update(ModelPreheatWorkerIdentity)
        .where(
            ModelPreheatWorkerIdentity.id == identity.id,
            ModelPreheatWorkerIdentity.token_version == identity.token_version,
            ModelPreheatWorkerIdentity.token_hash == identity.token_hash,
            ModelPreheatWorkerIdentity.expires_at == identity.expires_at,
            ModelPreheatWorkerIdentity.expires_at > now,
        )
        .values(expires_at=now + WORKER_CREDENTIAL_TTL)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount == 1:
        await session.commit()
        return True
    await session.rollback()
    return False


async def _validated_pending_identity(
    session, credential, worker_uuid, now, *, require_current=True
):
    credential_id = _credential_id(credential)
    if credential_id is None:
        return None
    pending = await _pending_credential(session, credential_id, credential, now)
    if pending is None:
        return None
    identity = await session.get(ModelPreheatWorkerIdentity, pending.identity_id)
    if (
        identity is None
        or identity.bootstrap_required
        or identity.revoked_at is not None
        or (worker_uuid is not None and identity.worker_uuid != worker_uuid)
    ):
        return None
    if require_current and not await _identity_is_current_worker(session, identity):
        return None
    return identity


async def _confirm_pending_credential(session, credential, now):
    credential_id = _credential_id(credential)
    if credential_id is None:
        return None
    pending = await _pending_credential(session, credential_id, credential, now)
    if pending is None:
        return None
    identity = await session.get(ModelPreheatWorkerIdentity, pending.identity_id)
    if identity is None or not await _identity_is_current_worker(session, identity):
        return None
    identity_id = identity.id
    result = await session.exec(
        update(ModelPreheatWorkerIdentity)
        .where(
            ModelPreheatWorkerIdentity.id == identity_id,
            ModelPreheatWorkerIdentity.token_hash.is_(None),
            ModelPreheatWorkerIdentity.token_version == pending.identity_token_version,
            ModelPreheatWorkerIdentity.bootstrap_required.is_(False),
            ModelPreheatWorkerIdentity.revoked_at.is_(None),
        )
        .values(
            token_hash=pending.token_hash,
            expires_at=now + WORKER_CREDENTIAL_TTL,
            registration_recovery_token_hash=None,
            registration_recovery_issued_at=None,
            upgrade_proof_hash=None,
            upgrade_proof_window_started_at=None,
        )
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        await session.rollback()
        return None
    await session.exec(
        delete(ModelPreheatWorkerPendingCredential).where(
            ModelPreheatWorkerPendingCredential.identity_id == identity_id
        )
    )
    await session.commit()
    return await session.get(ModelPreheatWorkerIdentity, identity_id)


async def _pending_credential(session, credential_id, credential, now):
    pending = (
        await session.exec(
            select(ModelPreheatWorkerPendingCredential).where(
                ModelPreheatWorkerPendingCredential.identity_id == credential_id,
                ModelPreheatWorkerPendingCredential.expires_at > now,
            )
        )
    ).all()
    expected_hash = _hash_token(credential)
    for candidate in pending:
        if hmac.compare_digest(candidate.token_hash, expected_hash):
            return candidate
    return None


async def _identity_is_current_worker(session, identity):
    current = (
        await session.exec(
            select(Worker)
            .where(Worker.worker_uuid == identity.worker_uuid)
            .order_by(Worker.id.desc())
        )
    ).first()
    return current is not None and current.id == identity.worker_id


def _identity_token_is_active(identity, now):
    return (
        identity is not None
        and not identity.bootstrap_required
        and identity.token_hash is not None
        and identity.revoked_at is None
        and identity.expires_at is not None
        and identity.expires_at > now
    )


def is_model_preheat_worker_registration_proof(proof: str) -> bool:
    if (
        not isinstance(proof, str)
        or len(proof) < 43
        or not all(character.isalnum() or character in "-_" for character in proof)
    ):
        return False
    try:
        decoded = base64.urlsafe_b64decode(proof + "=" * (-len(proof) % 4))
    except (ValueError, binascii.Error):
        return False
    return len(decoded) >= 32


def _principal(identity):
    return ModelPreheatWorkerPrincipal(
        worker_id=identity.worker_id,
        worker_uuid=identity.worker_uuid,
        credential_id=identity.id,
        token_version=identity.token_version,
    )


async def worker_uuid_has_credential(session, worker_uuid):
    identity_exists = (
        await session.exec(
            select(ModelPreheatWorkerIdentity.id).where(
                ModelPreheatWorkerIdentity.worker_uuid == worker_uuid,
            )
        )
    ).first() is not None
    if identity_exists:
        return True
    worker_exists = (
        await session.exec(
            select(Worker.id).where(Worker.worker_uuid == worker_uuid).limit(1)
        )
    ).first()
    return worker_exists is not None


async def issue_embedded_worker_credential_file(
    session: AsyncSession,
    worker_uuid: str,
    credential_path: str,
) -> bool:
    """为升级后的 embedded Worker 签发一次性引导凭据文件（任务 2 步骤 4）。

    仅在以下全部条件成立时写入，保证幂等且不放宽远程 Worker 身份隔离：

    - 数据库中确实存在匹配 ``worker_uuid`` 的既有 Worker（重复 UUID 时取最大
      ``worker.id``，即最新注册记录）；
    - 其预热身份仍为 ``bootstrap_required=True``（从未拿到可用凭据）；
    - 本地没有凭据文件（避免覆盖已轮换的新凭据）。

    写盘可恢复：先在会话内准备身份并 flush（尚未提交），原子写入
    ``credential_path``（POSIX ``0600``）成功后再提交；若写盘失败则回滚身份变更，
    使下次启动可重试，而不是留下“DB 已有凭据但无文件”的不可恢复状态。
    写盘成功但数据库 commit 失败时，必须删除**仅本次创建**的凭据文件并回滚
    身份变更，使重启可重新签发；调用前已存在的凭据文件绝不被删除
    （入口已按存在性短路，清理仅在文件确实由本次写盘产生时执行）。
    独立远程 Worker 不经过该路径：Server 不会为其写入本地凭据文件，
    且共享 token 不能接管既有 UUID（见 :func:`_authorize_worker_registration`）。
    返回是否实际写入了凭据。
    """
    import logging

    logger = logging.getLogger(__name__)
    preexisting = bool(credential_path) and os.path.exists(credential_path)
    if preexisting:
        return False
    # 重复 UUID 时取最新（最大 id）的 Worker，避免绑定到被替换的旧记录。
    worker = (
        await session.exec(
            select(Worker)
            .where(Worker.worker_uuid == worker_uuid)
            .order_by(Worker.id.desc())
        )
    ).first()
    if worker is None:
        return False
    identity = (
        await session.exec(
            select(ModelPreheatWorkerIdentity).where(
                ModelPreheatWorkerIdentity.worker_id == worker.id
            )
        )
    ).first()
    if identity is None or not identity.bootstrap_required:
        return False
    now = _utcnow()
    secret = secrets.token_urlsafe(32)
    if identity is None:
        identity = ModelPreheatWorkerIdentity(
            worker_id=worker.id,
            worker_uuid=worker_uuid,
            bootstrap_required=False,
            expires_at=now + WORKER_CREDENTIAL_TTL,
        )
        session.add(identity)
        await session.flush()
    else:
        identity.token_version += 1
        identity.worker_uuid = worker_uuid
        identity.expires_at = now + WORKER_CREDENTIAL_TTL
        identity.revoked_at = None
        identity.bootstrap_required = False
        session.add(identity)
        await session.flush()
    token = f"{WORKER_CREDENTIAL_PREFIX}_{identity.id}_{secret}"
    identity.token_hash = _hash_token(token)
    session.add(identity)
    # 先写盘再提交：写盘失败时回滚身份变更，保证下次可重试（可恢复）。
    try:
        _write_credential_file(credential_path, token)
    except OSError:
        await session.rollback()
        logger.error(
            "Failed to write embedded worker credential file; rolling back "
            "identity change so the next startup can retry"
        )
        return False
    try:
        await session.commit()
    except Exception:
        # 写盘成功但数据库 commit 失败：凭据未持久化，必须删除仅本次创建的
        # 无效凭据文件（preexisting 为 True 时入口已短路，这里必为本次新建），
        # 并回滚身份变更，使重启可重签；绝不删除调用前既有文件。
        try:
            if os.path.exists(credential_path):
                os.unlink(credential_path)
        except OSError:
            logger.error(
                "Failed to remove invalid embedded worker credential file "
                "after commit failure"
            )
        await session.rollback()
        logger.error(
            "Failed to commit embedded worker credential; rolled back identity "
            "and removed the invalid credential file so the next startup can retry"
        )
        return False
    return True


def _write_credential_file(credential_path: str, credential: str) -> None:
    os.makedirs(os.path.dirname(credential_path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix="." + os.path.basename(credential_path) + ".",
        dir=os.path.dirname(credential_path) or ".",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(credential)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, credential_path)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _credential_id(token):
    if not isinstance(token, str):
        return None
    parts = token.split("_", 2)
    if len(parts) != 3 or parts[0] != WORKER_CREDENTIAL_PREFIX:
        return None
    try:
        value = int(parts[1])
    except ValueError:
        return None
    return value if value > 0 else None


def _hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _unauthorized():
    return UnauthorizedException(message="Invalid worker authentication credentials")


def _utcnow():
    return datetime.now(timezone.utc)
