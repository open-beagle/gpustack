import hashlib
import hmac
import secrets
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import Depends, Header, Request
from sqlalchemy import update
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import UnauthorizedException
from gpustack.schemas.model_preheats import ModelPreheatWorkerIdentity
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_session


WORKER_CREDENTIAL_TTL = timedelta(hours=24)
WORKER_CREDENTIAL_RENEW_WINDOW = timedelta(hours=6)
WORKER_CREDENTIAL_PREFIX = "mpw"
WORKER_CREDENTIAL_HEADER = "X-GPUStack-Worker-Credential"


@dataclass(frozen=True)
class ModelPreheatWorkerPrincipal:
    worker_id: int
    worker_uuid: str
    credential_id: int
    token_version: int


async def issue_model_preheat_worker_credential(session, worker_id, worker_uuid):
    worker = await session.get(Worker, worker_id)
    if worker is None or worker.worker_uuid != worker_uuid:
        raise ValueError("worker_registration_invalid")
    now = _utcnow()
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
    identity = (
        await session.exec(
            select(ModelPreheatWorkerIdentity).where(
                ModelPreheatWorkerIdentity.worker_id == worker_id
            )
        )
    ).first()
    secret = secrets.token_urlsafe(32)
    if identity is None:
        identity = ModelPreheatWorkerIdentity(
            worker_id=worker_id,
            worker_uuid=worker_uuid,
            bootstrap_required=False,
            expires_at=now + WORKER_CREDENTIAL_TTL,
        )
        session.add(identity)
        await session.flush()
    else:
        # 新凭据尚未完成正常鉴权前，保留 Worker 本地仍可能持有的恢复凭据。
        # 连续丢失轮换响应时不能用 Worker 从未收到过的中间凭据替换它。
        if identity.registration_recovery_token_hash is None:
            if (
                identity.token_hash is not None
                and not identity.bootstrap_required
                and identity.revoked_at is None
                and identity.expires_at is not None
                and identity.expires_at > now
            ):
                identity.registration_recovery_token_hash = identity.token_hash
                identity.registration_recovery_issued_at = now
            else:
                identity.registration_recovery_issued_at = None
        identity.token_version += 1
        identity.worker_uuid = worker_uuid
        identity.expires_at = now + WORKER_CREDENTIAL_TTL
        identity.revoked_at = None
        identity.bootstrap_required = False
    token = f"{WORKER_CREDENTIAL_PREFIX}_{identity.id}_{secret}"
    identity.token_hash = _hash_token(token)
    session.add(identity)
    await session.commit()
    return token


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
    identity = await session.get(ModelPreheatWorkerIdentity, credential_id)
    now = _utcnow()
    if (
        identity is None
        or identity.bootstrap_required
        or identity.token_hash is None
        or identity.revoked_at is not None
        or identity.expires_at is None
        or identity.expires_at <= now
        or not hmac.compare_digest(identity.token_hash, _hash_token(credential))
    ):
        raise _unauthorized()
    current = (
        await session.exec(
            select(Worker)
            .where(Worker.worker_uuid == identity.worker_uuid)
            .order_by(Worker.id.desc())
        )
    ).first()
    if current is None or current.id != identity.worker_id:
        raise _unauthorized()
    principal = ModelPreheatWorkerPrincipal(
        worker_id=identity.worker_id,
        worker_uuid=identity.worker_uuid,
        credential_id=identity.id,
        token_version=identity.token_version,
    )
    renewed = identity.expires_at <= now + WORKER_CREDENTIAL_RENEW_WINDOW
    if renewed:
        identity.expires_at = now + WORKER_CREDENTIAL_TTL
    recovery_confirmed = identity.registration_recovery_token_hash is not None
    if recovery_confirmed:
        identity.registration_recovery_token_hash = None
        identity.registration_recovery_issued_at = None
    session.add(identity)
    if renewed or recovery_confirmed:
        await session.commit()
    request.state.model_preheat_worker = principal
    return principal


async def validate_model_preheat_worker_credential(
    session, credential, worker_uuid, *, require_current=True
):
    credential_id = _credential_id(credential)
    if credential_id is None:
        return None
    identity = await session.get(ModelPreheatWorkerIdentity, credential_id)
    now = _utcnow()
    if (
        identity is None
        or identity.worker_uuid != worker_uuid
        or identity.bootstrap_required
        or identity.token_hash is None
        or identity.revoked_at is not None
        or identity.expires_at is None
        or identity.expires_at <= now
        or not hmac.compare_digest(identity.token_hash, _hash_token(credential))
    ):
        return None
    if require_current:
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


async def validate_model_preheat_worker_registration_credential(
    session, credential, worker_uuid
):
    """仅用于 Worker 注册；恢复凭据不得进入任何任务或执行接口。"""
    identity = await validate_model_preheat_worker_credential(
        session, credential, worker_uuid
    )
    if identity is not None:
        if identity.registration_recovery_token_hash is not None:
            identity.registration_recovery_token_hash = None
            identity.registration_recovery_issued_at = None
            session.add(identity)
            await session.commit()
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
