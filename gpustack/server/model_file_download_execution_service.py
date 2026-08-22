"""普通 ModelFile 与私有下载执行配置的原子创建服务。"""

import json
from typing import Optional

from sqlmodel import select

from gpustack.api.exceptions import ConflictException, ServiceUnavailableException
from gpustack.model_preheat_credentials import (
    CredentialEncryptionUnavailable,
    ModelPreheatCredentialCipher,
)
from gpustack.schemas.model_file_download_executions import (
    ModelFileDownloadExecution,
    ModelFileDownloadExecutionProfilePin,
)
from gpustack.schemas.model_files import ModelFile
from gpustack.schemas.model_preheat_s3_profiles import (
    DEFAULT_SLOT_GLOBAL,
    ModelPreheatS3Profile,
)
from gpustack.schemas.models import SourceEnum, get_mmproj_filename
from gpustack.schemas.workers import MODEL_STORAGE_PROTOCOL_VERSION, Worker
from gpustack.server.bus import EventType
from gpustack.worker.model_preheat.identity import encode_path, normalize_source
from gpustack.worker.model_preheat.manifest import compute_request_digest


async def create_model_file_with_download_execution(session, model_file, config):
    """在同一事务中创建 ModelFile 及其唯一私有下载执行记录。"""
    worker = await _current_protocol_worker(session, model_file.worker_id)
    request_identity = _request_identity(model_file)
    profile = await _default_profile(session)
    snapshot = None
    key_version = None
    if profile is not None:
        cipher = _cipher(config)
        try:
            snapshot = cipher.encrypt(
                json.dumps(
                    _profile_snapshot(profile),
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        except CredentialEncryptionUnavailable as exc:
            raise ServiceUnavailableException(
                message="credential_encryption_unavailable"
            ) from exc
        key_version = cipher.current_key_version

    session.add(model_file)
    await session.flush()
    execution = ModelFileDownloadExecution(
        model_file_id=model_file.id,
        request_identity=request_identity,
        request_digest=_request_digest(request_identity),
        target_worker_id=worker.id,
        target_worker_uuid=worker.worker_uuid,
        default_profile_id=profile.id if profile is not None else None,
        default_profile_config_version=(
            profile.config_version if profile is not None else None
        ),
        credential_snapshot_encrypted=snapshot,
        encryption_key_version=key_version,
    )
    session.add(execution)
    await session.flush()
    if profile is not None:
        session.add(
            ModelFileDownloadExecutionProfilePin(
                execution_id=execution.id,
                profile_id=profile.id,
            )
        )
    await session.commit()
    await session.refresh(model_file)
    await ModelFile._publish_event(EventType.CREATED, model_file)
    return model_file


async def _current_protocol_worker(session, worker_id: Optional[int]) -> Worker:
    if worker_id is None:
        raise ConflictException(message="model_file_worker_required")
    worker = await session.get(Worker, worker_id)
    if worker is None:
        raise ConflictException(message="model_file_worker_not_found")
    latest = (
        await session.exec(
            select(Worker)
            .where(Worker.worker_uuid == worker.worker_uuid)
            .order_by(Worker.id.desc())
        )
    ).first()
    if latest is None or latest.id != worker.id:
        raise ConflictException(message="model_file_worker_stale_registration")
    if worker.model_storage_protocol_version != MODEL_STORAGE_PROTOCOL_VERSION:
        raise ConflictException(message="model_storage_protocol_mismatch")
    return worker


async def _default_profile(session) -> Optional[ModelPreheatS3Profile]:
    return (
        await session.exec(
            select(ModelPreheatS3Profile).where(
                ModelPreheatS3Profile.default_slot == DEFAULT_SLOT_GLOBAL
            )
        )
    ).first()


def _request_identity(model_file: ModelFile) -> dict:
    source_value = (
        model_file.source.value
        if hasattr(model_file.source, "value")
        else str(model_file.source)
    )
    if source_value not in {
        SourceEnum.HUGGING_FACE.value,
        SourceEnum.MODEL_SCOPE.value,
    }:
        # 普通 S3 模型库只服务 Hub 来源，其他来源仍需执行记录以保持协议门禁。
        return {
            "source": source_value,
            "model_id": model_file.model_source_index,
            "requested_revision": None,
            "include_patterns": [],
            "exclude_patterns": [],
        }
    source = normalize_source(source_value)
    if source == "huggingface":
        model_id = model_file.huggingface_repo_id
        patterns = [model_file.huggingface_filename]
    elif source == "modelscope":
        model_id = model_file.model_scope_model_id
        patterns = [model_file.model_scope_file_path]
    if not model_id:
        raise ConflictException(message="model_file_missing_model_id")
    extra = get_mmproj_filename(model_file)
    if extra:
        patterns.append(extra)
    return {
        "source": source,
        "model_id": model_id,
        "requested_revision": model_file.requested_revision,
        "include_patterns": sorted(pattern for pattern in patterns if pattern),
        "exclude_patterns": [],
    }


def _request_digest(identity: dict) -> str:
    source = identity["source"]
    if source not in {"huggingface", "modelscope"}:
        # 非 Hub 来源不参与 S3 查询，但仍需要稳定的请求摘要。
        import hashlib

        return hashlib.sha256(
            json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
    return compute_request_digest(
        source=source,
        model_id=encode_path(identity["model_id"]),
        requested_revision=(
            encode_path(identity["requested_revision"])
            if identity.get("requested_revision")
            else None
        ),
        include_patterns=[encode_path(item) for item in identity["include_patterns"]],
        exclude_patterns=[encode_path(item) for item in identity["exclude_patterns"]],
    )


def _profile_snapshot(profile: ModelPreheatS3Profile) -> dict:
    return {
        "id": profile.id,
        "config_version": profile.config_version,
        "endpoint": profile.endpoint,
        "bucket": profile.bucket,
        "prefix": profile.prefix,
        "tls_enabled": profile.tls_enabled,
        "tls_verify": profile.tls_verify,
        "region": profile.region,
        "use_virtual_hosted_style": profile.use_virtual_hosted_style,
        "source_fallback_enabled": profile.source_fallback_enabled,
        "access_key_encrypted": profile.access_key_encrypted,
        "secret_key_encrypted": profile.secret_key_encrypted,
    }


def _cipher(config) -> ModelPreheatCredentialCipher:
    return ModelPreheatCredentialCipher(
        current_key=getattr(config, "model_preheat_credential_key", None),
        current_key_version=getattr(
            config, "model_preheat_credential_key_version", None
        ),
        old_keys=getattr(config, "model_preheat_credential_old_keys", None),
    )
