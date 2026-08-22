import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheats import ModelPreheatIdempotencyRecord


IDEMPOTENCY_TTL = timedelta(hours=24)


def canonical_request_hash(payload: dict) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def get_idempotency_record(
    session: AsyncSession, user_id: int, operation: str, idempotency_key: Optional[str]
) -> Optional[ModelPreheatIdempotencyRecord]:
    if not idempotency_key:
        return None
    statement = select(ModelPreheatIdempotencyRecord).where(
        ModelPreheatIdempotencyRecord.user_id == user_id,
        ModelPreheatIdempotencyRecord.operation == operation,
        ModelPreheatIdempotencyRecord.idempotency_key == idempotency_key,
    )
    record = (await session.exec(statement)).first()
    if record is None:
        return None
    now = datetime.now(timezone.utc)
    if record.expires_at <= now:
        await session.delete(record)
        await session.flush()
        return None
    return record


def new_idempotency_record(
    user_id: int,
    operation: str,
    idempotency_key: Optional[str],
    request_hash: str,
    resource_id: int,
    response_status: int = 200,
    resource_type: str = "model_preheat_task",
) -> Optional[ModelPreheatIdempotencyRecord]:
    if not idempotency_key:
        return None
    return ModelPreheatIdempotencyRecord(
        user_id=user_id,
        operation=operation,
        idempotency_key=idempotency_key,
        request_hash=request_hash,
        resource_type=resource_type,
        resource_id=resource_id,
        response_status=response_status,
        expires_at=datetime.now(timezone.utc) + IDEMPOTENCY_TTL,
    )
