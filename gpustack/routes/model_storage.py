"""统一模型存储 API（任务 3）。

步骤 3/4（设计文档 §10.4）：

- 模型同步任务：只接受 ``model_file_id + profile_id``，Server 从 ModelFile
  推导请求身份、本地路径与目标对象 Key，浏览器不得提交任意对象 Key；
  创建时固定 request identity、Profile ID/config version 与加密凭据快照，
  库存精确命中时直接绑定 ``artifact_id``，否则保持 NULL 供 Worker CAS 绑定；
  支持 Idempotency-Key 重放与活动任务去重。
- 任务列表/详情/取消删除；详情分别返回模型 ``source``、本次
  ``transfer_source``、S3 Profile 与来源 Worker，字段不混用，不含凭据。
- Artifact 列表与手工刷新（从合法 Manifest 重建当前 config version 库存）。
- ``GET /model-storage/capabilities`` 与仅管理员可调用的
  ``POST /model-storage/connection-tests``（未保存表单，Server 侧短生命周期
  检查，分阶段报告，异常路径也清理临时对象）。

凭据快照保存在任务私有加密字段，只进入受 Worker 身份约束的执行 payload，
不进入 Public schema、SSE 或日志。
"""

import json
import ssl
from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, update
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    HTTPException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.model_preheat_credentials import (
    CredentialEncryptionUnavailable,
    ModelPreheatCredentialCipher,
    ModelPreheatCredentialError,
)
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatArtifactPublic,
    ModelPreheatInventoryManifestStateEnum,
)
from gpustack.schemas.model_file_download_executions import (
    ModelFileTransferSourceEnum,
)
from gpustack.schemas.model_storage_sync import (
    ModelStorageConnectionTestPublic,
    ModelStorageConnectionTestRequest,
    ModelStorageSyncCapabilitiesPublic,
    ModelStorageSyncExecutionPayload,
    ModelStorageSyncExecutionProfile,
    ModelStorageSyncTaskComplete,
    ModelStorageSyncTask,
    ModelStorageSyncTaskCreate,
    ModelStorageSyncTaskFail,
    ModelStorageSyncTaskDetail,
    ModelStorageSyncTaskProfilePublic,
    ModelStorageSyncTaskPublic,
    ModelStorageSyncTaskStateEnum,
    ModelStorageSyncTasksPublic,
)
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.deps import CurrentAdminUserDep, EngineDep, ListParamsDep, SessionDep
from gpustack.server.model_preheat_idempotency import (
    canonical_request_hash,
    get_idempotency_record,
    new_idempotency_record,
)
from gpustack.server.model_storage_connection_test import (
    run_model_storage_connection_test,
    validate_endpoint_url,
)
from gpustack.server.model_preheat_worker_identity import (
    ModelPreheatWorkerPrincipal,
    get_model_preheat_worker_identity,
)
from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    ModelPreheatIdentityError,
)

router = APIRouter()

worker_router = APIRouter()

WorkerIdentityDep = Annotated[
    ModelPreheatWorkerPrincipal, Depends(get_model_preheat_worker_identity)
]

SYNC_CREATE_OPERATION = "model_storage_sync_task.create"
# 活动任务：未达终态，同 (model_file_id, profile_id) 只允许一个。
_ACTIVE_STATES = (
    ModelStorageSyncTaskStateEnum.PENDING,
    ModelStorageSyncTaskStateEnum.SCANNING,
    ModelStorageSyncTaskStateEnum.PUBLISHING,
)
_TERMINAL_STATES = (
    ModelStorageSyncTaskStateEnum.READY,
    ModelStorageSyncTaskStateEnum.ERROR,
    ModelStorageSyncTaskStateEnum.CANCELED,
)


@router.get("/model-storage/capabilities", response_model=ModelStorageSyncCapabilitiesPublic)
async def get_model_storage_capabilities(request: Request):
    """只返回布尔能力，不返回密钥或敏感配置。"""
    cipher = _cipher_from_request(request)
    return ModelStorageSyncCapabilitiesPublic(
        credential_encryption_available=bool(cipher.current_key)
    )


@router.post(
    "/model-storage/connection-tests",
    response_model=ModelStorageConnectionTestPublic,
)
async def create_model_storage_connection_test(
    request: Request,
    current_user: CurrentAdminUserDep,
    body: ModelStorageConnectionTestRequest,
):
    """仅管理员可调用的保存前连接测试。

    直接使用未保存表单从 Server 检查连接/Bucket/写/读/删；不创建 Profile 或
    持久任务，不记录请求体；异常路径也清理临时对象。加密能力不可用时返回稳定
    错误码。
    """
    # Endpoint/TLS 与 Profile CRUD 共用校验。
    try:
        validate_endpoint_url(body.endpoint)
    except ValueError:
        raise HTTPException(422, "invalid_endpoint_scheme", "invalid_endpoint_scheme")

    result = run_model_storage_connection_test(
        endpoint=body.endpoint,
        bucket=body.bucket,
        prefix=body.prefix,
        access_key=body.access_key,
        secret_key=body.secret_key,
        tls_enabled=body.tls_enabled,
        tls_verify=body.tls_verify,
        region=body.region,
        use_virtual_hosted_style=body.use_virtual_hosted_style,
        client_factory=_minio_client_factory,
    )
    return result


@router.post(
    "/model-storage-sync-tasks",
    response_model=ModelStorageSyncTaskPublic,
)
async def create_model_storage_sync_task(
    request: Request,
    session: SessionDep,
    current_user: CurrentAdminUserDep,
    task_in: ModelStorageSyncTaskCreate,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    model_file = await ModelFile.one_by_id(session, task_in.model_file_id)
    if model_file is None:
        raise NotFoundException(message="model_file_not_found")

    identity, resolved_revision = _derive_task_identity(model_file)
    profile = await ModelPreheatS3Profile.one_by_id(session, task_in.profile_id)
    if profile is None:
        raise NotFoundException(message="s3_profile_not_found")

    cipher = _cipher_from_request(request)
    try:
        _ensure_current_key_configured(cipher)
        credential_snapshot = cipher.encrypt(
            json.dumps(
                {
                    "endpoint": profile.endpoint,
                    "bucket": profile.bucket,
                    "prefix": profile.prefix,
                    "tls_enabled": profile.tls_enabled,
                    "tls_verify": profile.tls_verify,
                    "region": profile.region,
                    "use_virtual_hosted_style": profile.use_virtual_hosted_style,
                    "access_key_encrypted": profile.access_key_encrypted,
                    "secret_key_encrypted": profile.secret_key_encrypted,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except CredentialEncryptionUnavailable as exc:
        raise ServiceUnavailableException(
            message=f"credential_encryption_unavailable: {exc}"
        )

    request_hash = canonical_request_hash(
        {
            "model_file_id": model_file.id,
            "profile_id": profile.id,
            "profile_config_version": profile.config_version,
            "request_digest": identity.request_digest,
        }
    )
    # Idempotency-Key 重放。
    record = await get_idempotency_record(
        session, current_user.id, SYNC_CREATE_OPERATION, idempotency_key
    )
    if record is not None:
        if record.request_hash != request_hash:
            raise HTTPException(409, "idempotency_key_reused", "idempotency_key_reused")
        replay = await ModelStorageSyncTask.one_by_id(session, record.resource_id)
        if replay is None:
            raise HTTPException(409, "idempotency_resource_not_found", "idempotency_resource_not_found")
        return _to_public(replay)

    # 活动任务去重：同 (model_file_id, profile_id) 已存在未终态任务。
    active = (
        await session.exec(
            select(ModelStorageSyncTask)
            .where(
                and_(
                    ModelStorageSyncTask.model_file_id == model_file.id,
                    ModelStorageSyncTask.profile_id == profile.id,
                    ModelStorageSyncTask.state.in_(list(_ACTIVE_STATES)),
                )
            )
        )
    ).first()
    if active is not None:
        return _to_public(active)

    # 库存精确命中时直接绑定 artifact_id；否则保持 NULL 供 Worker CAS 绑定。
    artifact_id = await _exact_artifact_match(session, profile, identity)
    worker = await Worker.one_by_id(session, model_file.worker_id)
    if worker is None:
        raise ConflictException(message="model_file_worker_not_ready")

    task = ModelStorageSyncTask(
        model_file_id=model_file.id,
        worker_id=model_file.worker_id,
        worker_uuid=worker.worker_uuid,
        profile_id=profile.id,
        profile_config_version=profile.config_version,
        # model_id 保存原始值（与 ModelPreheatTask 一致）：Worker 重建
        # ModelPreheatIdentity 时会统一 encode_path，这里不得预编码，
        # 否则特殊字符会被二次编码且与库存 model_id 不一致。
        request_identity={
            "source": identity.source,
            "model_id": identity.model_id,
            "requested_revision": identity.requested_revision,
            "include_patterns": list(identity.file_patterns),
            "exclude_patterns": list(identity.exclude_patterns),
        },
        request_digest=identity.request_digest,
        source=identity.source,
        model_id=identity.model_id,
        resolved_revision=resolved_revision,
        credential_snapshot_encrypted=credential_snapshot,
        encryption_key_version=cipher.current_key_version,
        artifact_id=artifact_id,
        state=ModelStorageSyncTaskStateEnum.PENDING,
        created_by_user_id=current_user.id,
    )
    try:
        task = await ModelStorageSyncTask.create(session, task)
    except ModelPreheatCredentialError as exc:
        raise ServiceUnavailableException(
            message=f"credential_encryption_unavailable: {exc}"
        )

    if idempotency_key:
        record = new_idempotency_record(
            current_user.id,
            SYNC_CREATE_OPERATION,
            idempotency_key,
            request_hash,
            task.id,
            response_status=200,
        )
        if record is not None:
            session.add(record)
            await session.commit()
            await session.refresh(task)
    return _to_public(task)


@router.get("/model-storage-sync-tasks", response_model=ModelStorageSyncTasksPublic)
async def list_model_storage_sync_tasks(
    engine: EngineDep,
    session: SessionDep,
    params: ListParamsDep,
    model_file_id: Optional[int] = None,
    profile_id: Optional[int] = None,
    state: Optional[ModelStorageSyncTaskStateEnum] = None,
):
    fields = {}
    if model_file_id is not None:
        fields["model_file_id"] = model_file_id
    if profile_id is not None:
        fields["profile_id"] = profile_id
    if state is not None:
        fields["state"] = state
    if params.watch:
        return StreamingResponse(
            ModelStorageSyncTask.streaming(engine, fields=fields),
            media_type="text/event-stream",
        )
    return await ModelStorageSyncTask.paginated_by_query(
        session=session,
        fields=fields,
        page=params.page,
        per_page=params.perPage,
    )


@router.get(
    "/model-storage-sync-tasks/{id}",
    response_model=ModelStorageSyncTaskDetail,
)
async def get_model_storage_sync_task(session: SessionDep, id: int):
    task = await ModelStorageSyncTask.one_by_id(session, id)
    if task is None:
        raise NotFoundException(message="model_storage_sync_task_not_found")
    return await _to_detail(session, task)


@router.delete("/model-storage-sync-tasks/{id}")
async def cancel_model_storage_sync_task(session: SessionDep, id: int):
    """取消/删除：活动任务置为 canceled；终态任务直接删除。"""
    task = await ModelStorageSyncTask.one_by_id(session, id)
    if task is None:
        raise NotFoundException(message="model_storage_sync_task_not_found")
    if task.state in _ACTIVE_STATES:
        task.state = ModelStorageSyncTaskStateEnum.CANCELED
        await task.update(session)
        return Response(status_code=200)
    await task.delete(session)
    return Response(status_code=204)


@router.get(
    "/model-storage-profiles/{profile_id}/artifacts",
    response_model=list[ModelPreheatArtifactPublic],
)
async def list_profile_artifacts(
    session: SessionDep,
    profile_id: int,
    manifest_state: Optional[ModelPreheatInventoryManifestStateEnum] = None,
):
    """Artifact 库存列表：必须精确匹配 profile + 当前 config version。"""
    profile = await ModelPreheatS3Profile.one_by_id(session, profile_id)
    if profile is None:
        raise NotFoundException(message="s3_profile_not_found")
    statement = select(ModelPreheatArtifact).where(
        and_(
            ModelPreheatArtifact.profile_id == profile.id,
            ModelPreheatArtifact.profile_config_version == profile.config_version,
        )
    )
    if manifest_state is not None:
        statement = statement.where(
            ModelPreheatArtifact.manifest_state == manifest_state
        )
    rows = (await session.exec(statement)).all()
    return [ModelPreheatArtifactPublic.model_validate(row) for row in rows]


@router.post(
    "/model-storage-profiles/{profile_id}/artifacts/refresh",
    response_model=dict,
)
async def refresh_profile_artifacts(request: Request, session: SessionDep, profile_id: int):
    """手工刷新：从合法 Manifest 重建当前 config version 库存；Profile 位置
    变化时旧版本库存标记 stale，新任务不得命中。

    复用 inventory 服务的 refresh job 能力；不可用时返回稳定错误码。
    """
    profile = await ModelPreheatS3Profile.one_by_id(session, profile_id)
    if profile is None:
        raise NotFoundException(message="s3_profile_not_found")
    service = getattr(request.app.state, "model_preheat_s3_inventory", None)
    if service is None:
        raise ServiceUnavailableException(message="inventory_service_unavailable")
    try:
        job = await service.create_refresh_job(
            session, profile.id, profile.config_version
        )
    except Exception:
        raise ServiceUnavailableException(message="inventory_service_unavailable")
    return {"job_id": getattr(job, "id", None)}


def _derive_task_identity(model_file: ModelFile):
    """从 ModelFile 推导运行时请求身份（source/model_id/requested_revision）。

    只支持 ModelScope / Hugging Face 来源；其他来源或字段缺失时拒绝。
    返回 ``(ModelPreheatIdentity, resolved_revision)``。
    """
    source = (
        model_file.source.value
        if hasattr(model_file.source, "value")
        else str(model_file.source)
    )
    if source == SourceEnum.MODEL_SCOPE.value:
        if not model_file.model_scope_model_id:
            raise ConflictException(message="model_file_missing_model_id")
        model_id = model_file.model_scope_model_id
    elif source == SourceEnum.HUGGING_FACE.value:
        if not model_file.huggingface_repo_id:
            raise ConflictException(message="model_file_missing_model_id")
        model_id = model_file.huggingface_repo_id
    else:
        raise ConflictException(message="model_sync_source_unsupported")
    if (
        model_file.state != ModelFileStateEnum.READY
        or model_file.worker_id is None
        or not model_file.resolved_paths
    ):
        raise ConflictException(message="model_file_not_ready")
    resolved_revision = model_file.resolved_revision or model_file.requested_revision
    if not resolved_revision:
        raise ConflictException(message="model_file_missing_resolved_revision")
    try:
        identity = ModelPreheatIdentity(
            source=source,
            model_id=model_id,
            revision=resolved_revision,
            requested_revision=model_file.requested_revision,
            file_patterns=(),
            exclude_patterns=(),
        )
    except ModelPreheatIdentityError as exc:
        raise ConflictException(message="invalid_model_identity") from exc
    return identity, resolved_revision


async def _exact_artifact_match(session, profile, identity) -> Optional[str]:
    """库存精确命中（唯一）时返回 artifact_id；否则 None。"""
    rows = (
        await session.exec(
            select(ModelPreheatArtifact.artifact_id).where(
                and_(
                    ModelPreheatArtifact.profile_id == profile.id,
                    ModelPreheatArtifact.profile_config_version == profile.config_version,
                    ModelPreheatArtifact.source == identity.source,
                    ModelPreheatArtifact.model_id == identity.model_path,
                    ModelPreheatArtifact.manifest_state
                    == ModelPreheatInventoryManifestStateEnum.VALID,
                )
            )
        )
    ).all()
    if len(rows) == 1:
        return rows[0]
    return None


def _to_public(task: ModelStorageSyncTask) -> ModelStorageSyncTaskPublic:
    return ModelStorageSyncTaskPublic(
        id=task.id,
        model_file_id=task.model_file_id,
        worker_id=task.worker_id,
        worker_uuid=task.worker_uuid,
        profile_id=task.profile_id,
        profile_config_version=task.profile_config_version,
        request_digest=task.request_digest,
        source=task.source,
        model_id=task.model_id,
        resolved_revision=task.resolved_revision,
        artifact_id=task.artifact_id,
        state=task.state,
        state_message=task.state_message,
        error_code=task.error_code,
        file_count=task.file_count,
        total_size=task.total_size,
        transfer_source=task.transfer_source,
        transfer_profile_id=task.transfer_profile_id,
        source_worker_id=task.source_worker_id,
        created_at=task.created_at,
        updated_at=task.updated_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
    )


async def _to_detail(session, task: ModelStorageSyncTask) -> ModelStorageSyncTaskDetail:
    profile = await ModelPreheatS3Profile.one_by_id(session, task.profile_id)
    profile_public = (
        ModelStorageSyncTaskProfilePublic(
            id=profile.id,
            name=profile.name,
            config_version=profile.config_version,
            system_managed=profile.system_managed,
        )
        if profile is not None
        else None
    )
    source_worker_name = None
    if task.source_worker_id is not None:
        source_worker = await Worker.one_by_id(session, task.source_worker_id)
        if source_worker is not None:
            source_worker_name = source_worker.name
    return ModelStorageSyncTaskDetail(
        id=task.id,
        model_file_id=task.model_file_id,
        worker_id=task.worker_id,
        worker_uuid=task.worker_uuid,
        source=task.source,
        model_id=task.model_id,
        resolved_revision=task.resolved_revision,
        request_digest=task.request_digest,
        profile_config_version=task.profile_config_version,
        profile=profile_public,
        transfer_source=task.transfer_source,
        transfer_profile_id=task.transfer_profile_id,
        source_worker_id=task.source_worker_id,
        source_worker_name=source_worker_name,
        artifact_id=task.artifact_id,
        state=task.state,
        state_message=task.state_message,
        error_code=task.error_code,
        file_count=task.file_count,
        total_size=task.total_size,
        created_at=task.created_at,
        updated_at=task.updated_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
    )


def _cipher_from_request(request: Request) -> ModelPreheatCredentialCipher:
    config = request.app.state.server_config
    return ModelPreheatCredentialCipher(
        current_key=getattr(config, "model_preheat_credential_key", None),
        current_key_version=getattr(
            config, "model_preheat_credential_key_version", None
        ),
        old_keys=getattr(config, "model_preheat_credential_old_keys", None),
    )


def _ensure_current_key_configured(cipher: ModelPreheatCredentialCipher):
    if not cipher.current_key:
        raise CredentialEncryptionUnavailable("credential_encryption_unavailable")


def _minio_client_factory(
    *,
    endpoint: str,
    access_key: str,
    secret_key: str,
    secure: bool,
    tls_verify: bool,
    region: Optional[str],
    use_virtual_hosted_style: bool,
    verified_host: Optional[str] = None,
):
    """构建 minio client；连接测试把已验证地址交给同一次连接使用。

    通过解析后的 netloc + secure 标志连接，TLS Server Name 仍用原始主机名
    （由 minio/urllib3 按 endpoint host 处理）；``verified_host`` 用于受控
    解析，避免校验与连接分别解析。
    """
    from minio import Minio

    import urllib3

    parsed = validate_endpoint_url(endpoint)
    netloc = parsed.netloc
    http_client = None
    if secure and not tls_verify:
        http_client = urllib3.PoolManager(
            cert_reqs=ssl.CERT_NONE,
            assert_hostname=False,
        )
    client = Minio(
        netloc,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
        region=region,
        http_client=http_client,
    )
    if use_virtual_hosted_style:
        client.enable_virtual_style_endpoint()
    else:
        client.disable_virtual_style_endpoint()
    return client


# ---------------------------------------------------------------------------
# 任务 3：Worker 侧同步执行端点（受 Worker 身份约束，凭据只进入执行 payload）
# ---------------------------------------------------------------------------


@worker_router.get(
    "/{task_id}/execution-payload",
    response_model=ModelStorageSyncExecutionPayload,
)
async def get_model_storage_sync_execution_payload(
    request: Request,
    session: SessionDep,
    task_id: int,
    identity: WorkerIdentityDep,
    response: Response,
):
    """Worker 领取后拉取一次性执行配置（含解密后的 Profile 凭据）。

    只允许任务所属 Worker（worker_uuid 匹配且为当前注册）读取；凭据不进
    Public schema、SSE 或日志，响应头禁止缓存。
    """
    task = await ModelStorageSyncTask.one_by_id(session, task_id)
    if task is None:
        raise NotFoundException(message="model_storage_sync_task_not_found")
    if task.worker_uuid != identity.worker_uuid:
        raise HTTPException(403, "worker_not_authorized", "worker_not_authorized")
    current = (
        await session.exec(
            select(Worker)
            .where(Worker.worker_uuid == identity.worker_uuid)
            .order_by(Worker.id.desc())
        )
    ).first()
    if current is None or current.id != task.worker_id:
        raise HTTPException(403, "worker_not_current", "worker_not_current")

    cipher = _cipher_from_request(request)
    try:
        profile = _decrypt_execution_profile(cipher, task)
    except (ModelPreheatCredentialError, KeyError, TypeError, ValueError):
        raise ServiceUnavailableException(message="execution_credentials_unavailable")
    response.headers["Cache-Control"] = "no-store"
    model_file = await ModelFile.one_by_id(session, task.model_file_id)
    return ModelStorageSyncExecutionPayload(
        task_id=task.id,
        state=task.state,
        source=task.source,
        model_id=task.model_id,
        resolved_revision=task.resolved_revision,
        request_identity=task.request_identity,
        request_digest=task.request_digest,
        source_paths=list(model_file.resolved_paths) if model_file else [],
        profile=profile,
    )


@worker_router.post("/{task_id}/complete")
async def complete_model_storage_sync_task(
    session: SessionDep,
    task_id: int,
    complete: ModelStorageSyncTaskComplete,
    identity: WorkerIdentityDep,
):
    """Worker 完成：CAS 将 ``artifact_id`` 从 NULL 绑定为计算结果；已绑定其他值
    时失败，不覆盖；并回写文件数/容量、来源字段与终态时间。"""
    task = await _authorized_sync_task(session, task_id, identity)
    if task.state in _TERMINAL_STATES:
        return Response(status_code=200)
    now = datetime.now(timezone.utc)
    values = {
        "state": ModelStorageSyncTaskStateEnum.READY,
        "file_count": complete.file_count,
        "total_size": complete.total_size,
        "transfer_source": ModelFileTransferSourceEnum.S3,
        "transfer_profile_id": task.profile_id,
        "source_worker_id": task.worker_id,
        "state_message": None,
        "error_code": None,
        "finished_at": now,
    }
    # artifact_id CAS：仅从 NULL 绑定，已绑定其他值则失败不覆盖。
    result = await session.exec(
        update(ModelStorageSyncTask)
        .where(
            ModelStorageSyncTask.id == task.id,
            ModelStorageSyncTask.worker_uuid == identity.worker_uuid,
            ModelStorageSyncTask.artifact_id.is_(None),
            # 仅活动状态可完成：读取后任务若被取消，CAS 不再写 ready。
            ModelStorageSyncTask.state.in_(list(_ACTIVE_STATES)),
        )
        .values(artifact_id=complete.artifact_id, **values)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount == 0:
        # artifact_id 已被绑定（其他值）或已终态：视为幂等成功但不覆盖。
        return Response(status_code=200)
    await session.commit()
    return Response(status_code=200)


@worker_router.post("/{task_id}/fail")
async def fail_model_storage_sync_task(
    session: SessionDep,
    task_id: int,
    failure: ModelStorageSyncTaskFail,
    identity: WorkerIdentityDep,
):
    task = await _authorized_sync_task(session, task_id, identity)
    if task.state in _TERMINAL_STATES:
        return Response(status_code=200)
    now = datetime.now(timezone.utc)
    result = await session.exec(
        update(ModelStorageSyncTask)
        .where(
            ModelStorageSyncTask.id == task.id,
            ModelStorageSyncTask.worker_uuid == identity.worker_uuid,
            ModelStorageSyncTask.state.in_(
                [
                    ModelStorageSyncTaskStateEnum.PENDING,
                    ModelStorageSyncTaskStateEnum.SCANNING,
                    ModelStorageSyncTaskStateEnum.PUBLISHING,
                ]
            ),
        )
        .values(
            state=ModelStorageSyncTaskStateEnum.ERROR,
            error_code=failure.error_code,
            finished_at=now,
        )
        .execution_options(synchronize_session=False)
    )
    if result.rowcount == 0:
        return Response(status_code=200)
    await session.commit()
    return Response(status_code=200)


async def _authorized_sync_task(session, task_id, identity) -> ModelStorageSyncTask:
    task = await ModelStorageSyncTask.one_by_id(session, task_id)
    if task is None:
        raise NotFoundException(message="model_storage_sync_task_not_found")
    if task.worker_uuid != identity.worker_uuid:
        raise HTTPException(403, "worker_not_authorized", "worker_not_authorized")
    current = (
        await session.exec(
            select(Worker)
            .where(Worker.worker_uuid == identity.worker_uuid)
            .order_by(Worker.id.desc())
        )
    ).first()
    if current is None or current.id != task.worker_id:
        raise HTTPException(403, "worker_not_current", "worker_not_current")
    return task


def _decrypt_execution_profile(
    cipher: ModelPreheatCredentialCipher, task: ModelStorageSyncTask
) -> ModelStorageSyncExecutionProfile:
    snapshot = task.credential_snapshot_encrypted
    if isinstance(snapshot, str):
        plaintext = cipher.decrypt(snapshot)
    else:
        plaintext = cipher.decrypt(json.dumps(snapshot))
    payload = json.loads(plaintext)
    return ModelStorageSyncExecutionProfile(
        endpoint=payload["endpoint"],
        bucket=payload["bucket"],
        prefix=payload.get("prefix", ""),
        tls_enabled=payload.get("tls_enabled", True),
        tls_verify=payload.get("tls_verify", True),
        region=payload.get("region") or "",
        use_virtual_hosted_style=payload.get("use_virtual_hosted_style", True),
        access_key=cipher.decrypt(payload["access_key_encrypted"]),
        secret_key=cipher.decrypt(payload["secret_key_encrypted"]),
    )
