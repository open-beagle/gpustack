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
    ModelPreheatS3ProfileLifecycleStateEnum,
)
from gpustack.schemas.models import SourceEnum, get_mmproj_filename
from gpustack.schemas.workers import (
    MODEL_STORAGE_PROTOCOL_VERSION,
    Worker,
    WorkerStateEnum,
)
from gpustack.server.bus import EventType
from gpustack.server.model_preheat_s3_profile_lifecycle import (
    ModelPreheatS3ProfileNotActive,
    lock_active_profile_for_new_work,
)
from gpustack.worker.model_preheat.identity import (
    encode_path,
    normalize_source,
    ollama_model_filename,
)
from gpustack.worker.model_preheat.manifest import compute_request_digest


async def create_model_file_with_download_execution(session, model_file, config):
    """在同一事务中创建 ModelFile 及其唯一私有下载执行记录。"""
    worker = await _current_protocol_worker(session, model_file.worker_id)
    model_file.worker_uuid_snapshot = worker.worker_uuid
    model_file.worker_name_snapshot = worker.name
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
        try:
            profile = await lock_active_profile_for_new_work(
                session,
                profile.id,
                profile.config_version,
                require_default=True,
            )
        except ModelPreheatS3ProfileNotActive:
            # 默认 Profile 在创建期间进入维护或被替换，本次普通下载明确
            # 降级为无 S3 固定配置。
            profile = None
            snapshot = None
            key_version = None

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
    if worker.state != WorkerStateEnum.READY:
        raise ConflictException(message="model_file_worker_not_ready")
    if worker.model_storage_protocol_version != MODEL_STORAGE_PROTOCOL_VERSION:
        raise ConflictException(message="model_storage_protocol_mismatch")
    return worker


async def _default_profile(session) -> Optional[ModelPreheatS3Profile]:
    return (
        await session.exec(
            select(ModelPreheatS3Profile).where(
                ModelPreheatS3Profile.default_slot == DEFAULT_SLOT_GLOBAL,
                ModelPreheatS3Profile.lifecycle_state
                == ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE,
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
        SourceEnum.OLLAMA_LIBRARY.value,
    }:
        # 统一模型存储只服务受支持来源；其他来源仍创建执行记录保持协议门禁。
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
    elif source == SourceEnum.OLLAMA_LIBRARY.value:
        model_id = model_file.ollama_library_model_name
        filename = ollama_model_filename(model_id) if model_id else None
        patterns = [filename, f"{filename}/**"] if filename else []
    if not model_id:
        raise ConflictException(message="model_file_missing_model_id")
    extra = (
        get_mmproj_filename(model_file)
        if source in {"huggingface", "modelscope"}
        else None
    )
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
    return compute_request_digest(
        source=identity["source"],
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
