"""Worker 私有的普通模型下载执行配置领取端点。"""

import asyncio
import fnmatch
import json
import re
from datetime import datetime, timezone
from glob import has_magic
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import and_, update
from sqlmodel import select

from gpustack.api.exceptions import (
    HTTPException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
    ModelPreheatCredentialError,
)
from gpustack.schemas.model_file_download_executions import (
    ModelFileDownloadExecution,
    ModelFileDownloadExecutionClaimed,
    ModelFileDownloadExecutionComplete,
    ModelFileDownloadExecutionFail,
    ModelFileDownloadExecutionProfile,
    ModelFileDownloadExecutionStateEnum,
    ModelFileTransferSourceEnum,
)
from gpustack.schemas.model_files import ModelFile, ModelFilePublic
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatInventoryManifestStateEnum,
)
from gpustack.schemas.workers import (
    MODEL_STORAGE_PROTOCOL_VERSION,
    Worker,
    WorkerStateEnum,
)
from gpustack.schemas.model_storage_sync import (
    ModelStorageSyncTask,
    ModelStorageSyncTaskStateEnum,
)
from gpustack.server.bus import EventType
from gpustack.server.deps import SessionDep
from gpustack.server.model_preheat_revision import (
    modelscope_upstream_revision,
    resolve_model_preheat_revision,
)
from gpustack.server.model_storage_scan_spec import compute_scan_spec
from gpustack.server.model_preheat_worker_identity import (
    ModelPreheatWorkerPrincipal,
    get_model_preheat_worker_identity,
)
from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentityError,
    encode_path,
    local_snapshot_revision,
)


router = APIRouter()
WorkerIdentityDep = Annotated[
    ModelPreheatWorkerPrincipal, Depends(get_model_preheat_worker_identity)
]
_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")


@router.post(
    "/{model_file_id}/download-executions/claim",
    response_model=ModelFileDownloadExecutionClaimed,
)
async def claim_model_file_download_execution(
    request: Request,
    response: Response,
    session: SessionDep,
    model_file_id: int,
    identity: WorkerIdentityDep,
):
    execution = (
        await session.exec(
            select(ModelFileDownloadExecution).where(
                ModelFileDownloadExecution.model_file_id == model_file_id
            )
        )
    ).first()
    if execution is None:
        raise NotFoundException(message="model_file_download_execution_not_found")
    model_file = await session.get(ModelFile, model_file_id)
    if model_file is None:
        raise NotFoundException(message="model_file_not_found")
    pinned_model_file_id = model_file.id
    await _authorize_execution(session, execution, model_file, identity)

    if execution.claimed_by_worker_uuid not in {None, identity.worker_uuid}:
        raise HTTPException(403, "worker_not_authorized", "worker_not_authorized")
    if execution.state in {
        ModelFileDownloadExecutionStateEnum.ERROR,
        ModelFileDownloadExecutionStateEnum.CANCELED,
    }:
        raise HTTPException(409, "execution_not_claimable", "execution_not_claimable")

    if execution.state == ModelFileDownloadExecutionStateEnum.PENDING:
        execution = await _claim_pending_execution(
            request, session, execution, model_file, identity
        )
    elif execution.resolved_revision is None:
        raise HTTPException(
            409, "execution_revision_not_pinned", "execution_revision_not_pinned"
        )

    resolved_revision = execution.resolved_revision
    artifact_id = execution.artifact_id
    manifest_path = execution.manifest_path
    artifact_total_size = execution.artifact_total_size
    request_identity = execution.request_identity or {}
    source = request_identity.get("source")
    model_id = request_identity.get("model_id")
    requested_revision = request_identity.get("requested_revision")
    profile, fallback_enabled = _execution_profile(request, execution)
    response.headers["Cache-Control"] = "no-store"
    return ModelFileDownloadExecutionClaimed(
        execution_id=execution.id,
        model_file_id=pinned_model_file_id,
        request_identity=request_identity,
        request_digest=execution.request_digest,
        source=source,
        model_id=model_id or "",
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        include_patterns=list(request_identity.get("include_patterns") or []),
        exclude_patterns=list(request_identity.get("exclude_patterns") or []),
        artifact_id=artifact_id,
        manifest_path=manifest_path,
        artifact_total_size=artifact_total_size,
        source_fallback_enabled=fallback_enabled,
        profile=profile,
    )


async def _claim_pending_execution(request, session, execution, model_file, identity):
    execution_id = execution.id
    request_identity = execution.request_identity or {}
    source = request_identity.get("source")
    model_id = request_identity.get("model_id")
    requested_revision = request_identity.get("requested_revision")
    resolved_revision = execution.resolved_revision
    artifact_id = execution.artifact_id
    manifest_path = execution.manifest_path
    artifact_total_size = execution.artifact_total_size

    if resolved_revision is None:
        if source in {"huggingface", "modelscope"}:
            resolved_revision = await _resolve_revision(
                request, source, model_id, requested_revision
            )
            expected_include_patterns = (
                await _resolve_artifact_selection(
                    request,
                    source,
                    model_id,
                    resolved_revision,
                    requested_revision,
                    request_identity.get("include_patterns") or [],
                    immutable_revision=_is_immutable_revision(requested_revision),
                )
                if execution.default_profile_id is not None
                else None
            )
            artifact = await _exact_artifact(
                session,
                execution,
                source,
                model_id,
                resolved_revision,
                expected_include_patterns,
                request_identity.get("exclude_patterns") or [],
            )
            artifact_id = artifact.artifact_id if artifact is not None else None
            manifest_path = artifact.manifest_path if artifact is not None else None
            artifact_total_size = artifact.total_size if artifact is not None else None
        elif source == "ollama_library":
            resolved_revision = local_snapshot_revision(
                source_index=model_file.source_index or model_file.model_source_index,
                source=source,
                resolved_paths=list(model_file.resolved_paths),
            )
            artifact = await _exact_artifact(
                session,
                execution,
                source,
                model_id,
                resolved_revision,
                sorted(
                    encode_path(pattern)
                    for pattern in request_identity.get("include_patterns") or []
                ),
                request_identity.get("exclude_patterns") or [],
            )
            artifact_id = artifact.artifact_id if artifact is not None else None
            manifest_path = artifact.manifest_path if artifact is not None else None
            artifact_total_size = artifact.total_size if artifact is not None else None
        else:
            resolved_revision = requested_revision or "not_applicable"

    claimed_at = datetime.now(timezone.utc)
    result = await session.exec(
        update(ModelFileDownloadExecution)
        .where(
            and_(
                ModelFileDownloadExecution.id == execution.id,
                ModelFileDownloadExecution.state
                == ModelFileDownloadExecutionStateEnum.PENDING,
            )
        )
        .values(
            resolved_revision=resolved_revision,
            artifact_id=artifact_id,
            manifest_path=manifest_path,
            artifact_total_size=artifact_total_size,
            claimed_by_worker_uuid=identity.worker_uuid,
            claimed_at=claimed_at,
            state=ModelFileDownloadExecutionStateEnum.RUNNING,
        )
    )
    if result.rowcount == 1:
        if execution.default_profile_id is not None:
            await session.exec(
                update(ModelPreheatS3Profile)
                .where(
                    ModelPreheatS3Profile.id == execution.default_profile_id,
                    ModelPreheatS3Profile.ever_used_at.is_(None),
                )
                .values(ever_used_at=claimed_at)
            )
        if source in {"huggingface", "modelscope", "ollama_library"}:
            model_file.requested_revision = requested_revision
            model_file.resolved_revision = resolved_revision
            session.add(model_file)
        await session.commit()
    else:
        await session.rollback()

    pinned = (
        await session.exec(
            select(ModelFileDownloadExecution)
            .where(ModelFileDownloadExecution.id == execution_id)
            .execution_options(populate_existing=True)
        )
    ).one_or_none()
    if pinned is None:
        raise NotFoundException(message="model_file_download_execution_not_found")
    if pinned.claimed_by_worker_uuid != identity.worker_uuid:
        raise HTTPException(403, "worker_not_authorized", "worker_not_authorized")
    if pinned.state not in {
        ModelFileDownloadExecutionStateEnum.RUNNING,
        ModelFileDownloadExecutionStateEnum.READY,
    }:
        raise HTTPException(409, "execution_not_claimable", "execution_not_claimable")
    return pinned


async def _authorize_execution(session, execution, model_file, identity) -> Worker:
    if (
        execution.target_worker_id != identity.worker_id
        or execution.target_worker_uuid != identity.worker_uuid
        or model_file.worker_id != identity.worker_id
    ):
        raise HTTPException(403, "worker_not_authorized", "worker_not_authorized")
    worker = await session.get(Worker, identity.worker_id)
    if worker is None or worker.worker_uuid != identity.worker_uuid:
        raise HTTPException(403, "worker_not_current", "worker_not_current")
    latest = (
        await session.exec(
            select(Worker)
            .where(Worker.worker_uuid == identity.worker_uuid)
            .order_by(Worker.id.desc())
        )
    ).first()
    if latest is None or latest.id != worker.id:
        raise HTTPException(403, "worker_not_current", "worker_not_current")
    if worker.state != WorkerStateEnum.READY:
        raise HTTPException(
            409,
            "model_file_worker_not_ready",
            "model_file_worker_not_ready",
        )
    if worker.model_storage_protocol_version != MODEL_STORAGE_PROTOCOL_VERSION:
        raise HTTPException(
            409, "model_storage_protocol_mismatch", "model_storage_protocol_mismatch"
        )
    return worker


async def _resolve_revision(request, source, model_id, requested_revision) -> str:
    if not isinstance(model_id, str) or not model_id:
        raise HTTPException(409, "invalid_model_identity", "invalid_model_identity")
    if _is_immutable_revision(requested_revision):
        return requested_revision.lower()
    resolver = getattr(
        request.app.state,
        "model_file_download_revision_resolver",
        resolve_model_preheat_revision,
    )
    try:
        return await asyncio.to_thread(
            resolver,
            source,
            model_id,
            requested_revision,
            token=getattr(request.app.state.server_config, "huggingface_token", None),
        )
    except Exception:
        raise ServiceUnavailableException(
            message="revision_resolution_unavailable"
        ) from None


async def _resolve_artifact_selection(
    request,
    source,
    model_id,
    resolved_revision,
    requested_revision,
    requested_patterns,
    *,
    immutable_revision=False,
):
    if not requested_patterns:
        return []
    if immutable_revision:
        if any(has_magic(pattern) for pattern in requested_patterns):
            # 固定 revision 禁止访问 Hub；glob 完整集合无法由局部库存离线证明。
            return None
        return _artifact_patterns_for_paths(requested_patterns)

    lister = getattr(
        request.app.state,
        "model_file_download_file_listing_resolver",
        _list_revision_files,
    )
    try:
        files = await asyncio.to_thread(
            lister,
            source,
            model_id,
            (
                modelscope_upstream_revision(resolved_revision, requested_revision)
                if source == "modelscope"
                else resolved_revision
            ),
            token=getattr(request.app.state.server_config, "huggingface_token", None),
        )
    except Exception:
        raise ServiceUnavailableException(
            message="file_selection_resolution_unavailable"
        ) from None

    valid_files = []
    try:
        for path in files:
            if isinstance(path, str):
                encode_path(path)
                valid_files.append(path)
    except ModelPreheatIdentityError:
        return None
    selected = sorted(
        {
            path
            for path in valid_files
            if any(fnmatch.fnmatch(path, pattern) for pattern in requested_patterns)
        }
    )
    if not selected:
        return None
    return _artifact_patterns_for_paths(selected)


def _is_immutable_revision(revision) -> bool:
    return isinstance(revision, str) and _COMMIT_SHA.fullmatch(revision) is not None


def _artifact_patterns_for_paths(paths):
    try:
        for path in paths:
            encode_path(path)
        _, concrete_patterns = compute_scan_spec(
            [f"/repository/{path}" for path in paths],
            repository_complete=False,
        )
        return sorted(encode_path(pattern) for pattern in concrete_patterns)
    except (ModelPreheatIdentityError, ValueError):
        # 任务 3 无法表达跨父目录或重名源路径时，不存在可精确复用的 Artifact。
        return None


def _list_revision_files(source, model_id, resolved_revision, *, token=None):
    if source == "huggingface":
        from huggingface_hub import HfApi

        return HfApi(token=token).list_repo_files(
            repo_id=model_id, revision=resolved_revision
        )

    from modelscope_hub.api import HubApi

    rows = HubApi().list_repo_files(
        model_id,
        "model",
        revision=resolved_revision,
        recursive=True,
    )
    return [row.path if hasattr(row, "path") else str(row) for row in rows]


async def _exact_artifact(
    session,
    execution,
    source,
    model_id,
    resolved_revision,
    expected_include_patterns,
    exclude_patterns,
):
    if execution.default_profile_id is None or expected_include_patterns is None:
        return None
    rows = (
        await session.exec(
            select(ModelPreheatArtifact).where(
                and_(
                    ModelPreheatArtifact.profile_id == execution.default_profile_id,
                    ModelPreheatArtifact.profile_config_version
                    == execution.default_profile_config_version,
                    ModelPreheatArtifact.source == source,
                    ModelPreheatArtifact.model_id == encode_path(model_id),
                    ModelPreheatArtifact.resolved_revision == resolved_revision,
                    ModelPreheatArtifact.manifest_state
                    == ModelPreheatInventoryManifestStateEnum.VALID,
                )
            )
        )
    ).all()
    expected_include = sorted(expected_include_patterns)
    expected_exclude = sorted(encode_path(item) for item in exclude_patterns)
    matched = [
        row
        for row in rows
        if list(row.include_patterns) == expected_include
        and list(row.exclude_patterns) == expected_exclude
    ]
    return matched[0] if len(matched) == 1 else None


def _execution_profile(request, execution):
    if execution.default_profile_id is None:
        return None, True
    cipher = _cipher(request)
    try:
        outer = execution.credential_snapshot_encrypted
        plaintext = cipher.decrypt(outer)
        snapshot = json.loads(plaintext)
        profile = ModelFileDownloadExecutionProfile(
            id=snapshot["id"],
            config_version=snapshot["config_version"],
            endpoint=snapshot["endpoint"],
            bucket=snapshot["bucket"],
            prefix=snapshot.get("prefix") or "",
            tls_enabled=snapshot.get("tls_enabled", True),
            tls_verify=snapshot.get("tls_verify", True),
            region=snapshot.get("region") or "",
            use_virtual_hosted_style=snapshot.get("use_virtual_hosted_style", True),
            access_key=cipher.decrypt(snapshot["access_key_encrypted"]),
            secret_key=cipher.decrypt(snapshot["secret_key_encrypted"]),
        )
        return profile, bool(snapshot.get("source_fallback_enabled", True))
    except (ModelPreheatCredentialError, KeyError, TypeError, ValueError):
        raise ServiceUnavailableException(
            message="execution_credentials_unavailable"
        ) from None


def _cipher(request):
    config = request.app.state.server_config
    return ModelPreheatCredentialCipher(
        current_key=getattr(config, "model_preheat_credential_key", None),
        current_key_version=getattr(
            config, "model_preheat_credential_key_version", None
        ),
        old_keys=getattr(config, "model_preheat_credential_old_keys", None),
    )


@router.post("/{model_file_id}/download-executions/complete")
async def complete_model_file_download_execution(
    session: SessionDep,
    model_file_id: int,
    complete: ModelFileDownloadExecutionComplete,
    identity: WorkerIdentityDep,
):
    execution, model_file = await _execution_and_model_file(session, model_file_id)
    await _authorize_execution(session, execution, model_file, identity)
    if execution.state == ModelFileDownloadExecutionStateEnum.READY:
        s3_replay = (
            complete.transfer_source == ModelFileTransferSourceEnum.S3
            and execution.transfer_source
            in {
                ModelFileTransferSourceEnum.S3,
                ModelFileTransferSourceEnum.PEER_VIA_S3,
            }
            and execution.transfer_profile_id == complete.transfer_profile_id
            and complete.source_worker_id is None
        )
        exact_replay = (
            execution.transfer_source == complete.transfer_source
            and execution.transfer_profile_id == complete.transfer_profile_id
            and execution.source_worker_id == complete.source_worker_id
        )
        if s3_replay or exact_replay:
            return {"state": execution.state}
        raise HTTPException(
            409, "execution_already_completed", "execution_already_completed"
        )
    if execution.state != ModelFileDownloadExecutionStateEnum.RUNNING:
        raise HTTPException(409, "execution_not_running", "execution_not_running")
    transfer_source, source_worker_id = await _normalized_transfer_result(
        session, execution, complete
    )
    if complete.transfer_profile_id not in {None, execution.default_profile_id}:
        raise HTTPException(422, "invalid_transfer_profile", "invalid_transfer_profile")
    if complete.source_worker_id not in {None, identity.worker_id}:
        raise HTTPException(422, "invalid_source_worker", "invalid_source_worker")
    if complete.transfer_source.value == "s3":
        if (
            execution.artifact_id is None
            or complete.transfer_profile_id != execution.default_profile_id
        ):
            raise HTTPException(
                422, "invalid_transfer_source", "invalid_transfer_source"
            )
    elif complete.transfer_profile_id is not None:
        raise HTTPException(422, "invalid_transfer_source", "invalid_transfer_source")
    execution.state = ModelFileDownloadExecutionStateEnum.READY
    execution.transfer_source = transfer_source
    execution.transfer_profile_id = complete.transfer_profile_id
    execution.source_worker_id = source_worker_id
    execution.error_code = None
    execution.state_message = None
    execution.finished_at = datetime.now(timezone.utc)
    session.add(execution)
    response_state = execution.state
    profile_name = None
    if complete.transfer_profile_id is not None:
        profile = await session.get(ModelPreheatS3Profile, complete.transfer_profile_id)
        profile_name = profile.name if profile is not None else None
    source_worker_name = None
    if source_worker_id is not None:
        source_worker = await session.get(Worker, source_worker_id)
        source_worker_name = source_worker.name if source_worker is not None else None
    model_file_event = ModelFilePublic.model_validate(
        model_file,
        update={
            "transfer_source": transfer_source,
            "transfer_profile_id": complete.transfer_profile_id,
            "transfer_profile_name": profile_name,
            "source_worker_id": source_worker_id,
            "source_worker_name": source_worker_name,
        },
    )
    await session.commit()
    await ModelFile._publish_event(EventType.UPDATED, model_file_event)
    return {"state": response_state}


@router.post("/{model_file_id}/download-executions/fail")
async def fail_model_file_download_execution(
    session: SessionDep,
    model_file_id: int,
    failure: ModelFileDownloadExecutionFail,
    identity: WorkerIdentityDep,
):
    execution, model_file = await _execution_and_model_file(session, model_file_id)
    await _authorize_execution(session, execution, model_file, identity)
    if execution.state == ModelFileDownloadExecutionStateEnum.READY:
        raise HTTPException(
            409, "execution_already_completed", "execution_already_completed"
        )
    if execution.state == ModelFileDownloadExecutionStateEnum.ERROR:
        return {"state": execution.state}
    if execution.state != ModelFileDownloadExecutionStateEnum.RUNNING:
        raise HTTPException(409, "execution_not_running", "execution_not_running")
    allowed = {
        "model_artifact_not_found",
        "revision_resolution_unavailable",
        "s3_manifest_invalid",
        "s3_manifest_missing",
        "s3_authentication_failed",
        "network_timeout",
        "worker_execution_failed",
    }
    execution.state = ModelFileDownloadExecutionStateEnum.ERROR
    execution.error_code = (
        failure.error_code
        if failure.error_code in allowed
        else "worker_execution_failed"
    )
    execution.state_message = execution.error_code
    execution.finished_at = datetime.now(timezone.utc)
    session.add(execution)
    response_state = execution.state
    await session.commit()
    return {"state": response_state}


async def _execution_and_model_file(session, model_file_id):
    execution = (
        await session.exec(
            select(ModelFileDownloadExecution).where(
                ModelFileDownloadExecution.model_file_id == model_file_id
            )
        )
    ).first()
    model_file = await session.get(ModelFile, model_file_id)
    if execution is None or model_file is None:
        raise NotFoundException(message="model_file_download_execution_not_found")
    return execution, model_file


async def _normalized_transfer_result(session, execution, complete):
    if complete.transfer_source != ModelFileTransferSourceEnum.S3:
        return complete.transfer_source, complete.source_worker_id
    provenance = (
        await session.exec(
            select(ModelStorageSyncTask)
            .where(
                ModelStorageSyncTask.profile_id == execution.default_profile_id,
                ModelStorageSyncTask.artifact_id == execution.artifact_id,
                ModelStorageSyncTask.state == ModelStorageSyncTaskStateEnum.READY,
            )
            .order_by(
                ModelStorageSyncTask.finished_at.desc(), ModelStorageSyncTask.id.desc()
            )
        )
    ).first()
    if provenance is None:
        return ModelFileTransferSourceEnum.S3, None
    return ModelFileTransferSourceEnum.PEER_VIA_S3, provenance.worker_id
