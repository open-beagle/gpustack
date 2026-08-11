import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import Depends, Header, Request
from sqlalchemy import update
from sqlmodel import select

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
    if identity.expires_at <= now + WORKER_CREDENTIAL_RENEW_WINDOW:
        identity.expires_at = now + WORKER_CREDENTIAL_TTL
        session.add(identity)
        await session.commit()
    principal = ModelPreheatWorkerPrincipal(
        worker_id=identity.worker_id,
        worker_uuid=identity.worker_uuid,
        credential_id=identity.id,
        token_version=identity.token_version,
    )
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
