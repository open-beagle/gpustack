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
import hmac
import secrets
from datetime import datetime, timezone
from typing import Annotated, Callable, Optional

from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, update
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
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
    ModelStorageSyncTaskDedupeSlot,
)
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.deps import (
    CurrentAdminUserDep,
    EngineDep,
    ListParamsDep,
    SessionDep,
)
from gpustack.server.model_preheat_idempotency import (
    canonical_request_hash,
    get_idempotency_record,
    new_idempotency_record,
)
from gpustack.server.bus import EventType
from gpustack.server.model_storage_connection_test import (
    VerifiedEndpoint,
    build_pinned_http_client,
    run_model_storage_connection_test,
    validate_endpoint_url,
)
from gpustack.server.model_storage_scan_spec import compute_scan_spec
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
# Idempotency-Key 记录的 resource_type：统一为同步任务资源类型。
SYNC_TASK_RESOURCE_TYPE = "model_storage_sync_task"
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


def _issue_lease_token() -> str:
    """签发一次性执行 lease token（随机、不可猜测；明文不进库）。"""
    return secrets.token_urlsafe(32)


def _lease_token_matches(
    cipher: ModelPreheatCredentialCipher,
    lease_token_encrypted,
    lease_token: str,
) -> bool:
    """校验 complete/fail 携带的 lease token 与任务签发的 lease 一致。

    任务 lease 以 AES-GCM 加密快照存储（不发明文）；快照缺失、解密失败或
    不一致都稳定返回 False（拒绝），不泄露差异细节。
    """
    if lease_token_encrypted is None:
        return False
    try:
        expected = cipher.decrypt(lease_token_encrypted)
    except Exception:
        return False
    return hmac.compare_digest(str(expected), str(lease_token or ""))


@router.get(
    "/model-storage/capabilities", response_model=ModelStorageSyncCapabilitiesPublic
)
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
    持久任务，不记录请求体；异常路径也清理临时对象。表单凭据只透传给受控
    连接工厂；Server 加密能力（凭据快照密钥）缺失或密钥不可用时，在进入
    网络探测之前即以稳定错误码 ``credential_encryption_unavailable`` 失败，
    不误归类为连接失败。
    """
    # Endpoint/TLS 与 Profile CRUD 共用校验。
    try:
        validate_endpoint_url(body.endpoint)
    except ValueError:
        raise HTTPException(422, "invalid_endpoint_scheme", "invalid_endpoint_scheme")

    # 加密能力门禁：连接测试表单中的凭据必须可被同一套加密能力保管
    # （Profile 保存路径同样要求）。不仅要求密钥存在，还要求当前密钥实际
    # 可用（可完成一次加密）；缺失或不可用都稳定失败，不误归类为连接失败。
    cipher = _cipher_from_request(request)
    try:
        _ensure_credential_encryption_available(cipher)
    except CredentialEncryptionUnavailable:
        raise ServiceUnavailableException(message="credential_encryption_unavailable")

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
    mf_id = task_in.model_file_id
    pf_id = task_in.profile_id

    # 稳定请求 hash：**仅**用稳定请求体字段（model_file_id + profile_id）计算，
    # 不读取/验证 ModelFile、Profile、revision/path/READY/密钥（第二轮定向
    # 复审：Idempotency-Key 查询必须先于任何读取/验证，且不混入可变派生状态）。
    request_hash = canonical_request_hash(
        {
            "model_file_id": mf_id,
            "profile_id": pf_id,
        }
    )

    # Idempotency-Key 重放（在任何读取/验证之前查询）：已存在 Key 严格校验
    # hash 后按 resource_id 返回原任务；hash 不一致（同 Key 指向不同请求体）
    # 或原任务已删除 → 稳定 409（不复用、不新建）。
    if idempotency_key is not None:
        record = await get_idempotency_record(
            session, current_user.id, SYNC_CREATE_OPERATION, idempotency_key
        )
        if record is not None:
            if record.request_hash != request_hash:
                raise HTTPException(
                    409, "idempotency_key_reused", "idempotency_key_reused"
                )
            replay = await ModelStorageSyncTask.one_by_id(session, record.resource_id)
            if replay is not None:
                return _to_public(replay)
            raise HTTPException(409, "idempotency_key_reused", "idempotency_key_reused")

    # 现在才读取/验证 ModelFile、Profile、revision/path/READY、Worker 与密钥。
    model_file = await ModelFile.one_by_id(session, mf_id)
    if model_file is None:
        raise NotFoundException(message="model_file_not_found")

    identity, resolved_revision, scan_spec = _derive_task_identity(model_file)
    profile = await ModelPreheatS3Profile.one_by_id(session, pf_id)
    if profile is None:
        raise NotFoundException(message="s3_profile_not_found")
    # IntegrityError 回滚前缓存 model_file/profile 标量：回滚后 ORM 实例属性
    # 访问会触发过期重载（MissingGreenlet），恢复路径只允许使用这里缓存的
    # 纯 Python 标量（id/config_version/源路径/扫描规约）。
    mf_id, mf_worker_id = model_file.id, model_file.worker_id
    pf_id, pf_config_version = profile.id, profile.config_version
    source_paths = list(model_file.resolved_paths)

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
                    # 物理执行快照与 canonical request identity 分离；绝对路径
                    # 仅存在于 AES-GCM 私有快照，绝不进入 request identity/digest。
                    "source_paths": source_paths,
                    "scan_spec": scan_spec,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except CredentialEncryptionUnavailable as exc:
        raise ServiceUnavailableException(
            message=f"credential_encryption_unavailable: {exc}"
        )

    # 活动任务去重：同 (model_file_id, profile_id) 已持有去重槽位（并发安全
    # 的最终保证在下方去重槽位的数据库级唯一约束）。
    active = await _active_sync_task(session, _dedupe_key(mf_id, pf_id))
    if active is not None:
        # 先构造 Public（commit 会过期 ORM 实例，之后不得再访问 active 属性）。
        active_id = active.id
        active_public = _to_public(active)
        # 活动任务命中新 Idempotency-Key：把该 Key 绑定到既有活动任务
        # （冲突语义稳定：后续同一 Key 重放稳定返回同一任务，而不是 409
        # 或新建）。无 Key 或 Key 已绑定时不重复写入。
        if idempotency_key is not None:
            bound_task_id = await _bind_idempotency_key_to_existing_task(
                session,
                current_user.id,
                idempotency_key,
                request_hash,
                active_id,
            )
            if bound_task_id != active_id:
                bound_task = await ModelStorageSyncTask.one_by_id(
                    session, bound_task_id
                )
                if bound_task is None:
                    raise HTTPException(
                        409, "idempotency_key_reused", "idempotency_key_reused"
                    )
                return _to_public(bound_task)
        return active_public

    # 库存精确命中时直接绑定 artifact_id；否则保持 NULL 供 Worker CAS 绑定。
    artifact_id = await _exact_artifact_match(session, profile, identity)
    worker = await _latest_ready_worker_for_model_file(session, model_file)

    task = ModelStorageSyncTask(
        model_file_id=mf_id,
        worker_id=mf_worker_id,
        worker_uuid=worker.worker_uuid,
        profile_id=pf_id,
        profile_config_version=pf_config_version,
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
        # 一次性执行 lease：加密快照入库（不发明文），明文只进入执行 payload。
        lease_token_encrypted=cipher.encrypt(_issue_lease_token()),
        artifact_id=artifact_id,
        state=ModelStorageSyncTaskStateEnum.PENDING,
        created_by_user_id=current_user.id,
    )
    # 任务、去重槽位与 Idempotency-Key 记录必须在**同一个事务**中原子提交：
    # 并发竞争时 dedupe_key 唯一约束拒绝后到者（数据库级保证，SQLite/
    # PostgreSQL/MySQL 通用），后到者整体回滚后按等价结果返回先创建者已提交
    # 的任务，不产生重复任务或遗留任务。
    # 回滚会使本实例的所有 ORM 属性过期：flush 成功后立即缓存 task 标量，
    # IntegrityError 路径只允许使用这些纯 Python 值。
    try:
        session.add(task)
        await session.flush()
        session.add(
            ModelStorageSyncTaskDedupeSlot(
                dedupe_key=_dedupe_key(mf_id, pf_id),
                task_id=task.id,
            )
        )
        if idempotency_key:
            session.add(
                new_idempotency_record(
                    current_user.id,
                    SYNC_CREATE_OPERATION,
                    idempotency_key,
                    request_hash,
                    task.id,
                    response_status=200,
                    resource_type=SYNC_TASK_RESOURCE_TYPE,
                )
            )
        await session.commit()
    except IntegrityError:
        await session.rollback()
        # 唯一冲突：去重槽位或 Idempotency-Key 被并发请求占用。本事务整体回滚，
        # 本次 own_task **未持久化**：恢复路径**只能**查询持久化状态（活动
        # 槽位 / 持久化幂等记录），绝不返回已回滚的 own_task（第二轮定向
        # 复审）。
        # 1) 活动槽位：并发者已提交活动任务，返回既有任务并把 Key 绑定到它
        #    （后续该 Key 重放稳定返回同一任务，不在 409/新建之间摇摆）。
        active = await _active_sync_task(session, _dedupe_key(mf_id, pf_id))
        if active is not None:
            # 先构造 Public（绑定提交会过期 ORM 实例，之后不得再访问 active）。
            active_id = active.id
            active_public = _to_public(active)
            if idempotency_key is not None:
                bound_task_id = await _bind_idempotency_key_to_existing_task(
                    session,
                    current_user.id,
                    idempotency_key,
                    request_hash,
                    active_id,
                )
                if bound_task_id != active_id:
                    bound_task = await ModelStorageSyncTask.one_by_id(
                        session, bound_task_id
                    )
                    if bound_task is None:
                        raise HTTPException(
                            409,
                            "idempotency_key_reused",
                            "idempotency_key_reused",
                        )
                    return _to_public(bound_task)
            return active_public
        # 2) 持久化幂等记录：并发者已以同 Key 提交（任务可能刚进入终态并
        #    释放槽位）。按 resource_id 返回原任务。
        if idempotency_key is not None:
            persisted = await get_idempotency_record(
                session, current_user.id, SYNC_CREATE_OPERATION, idempotency_key
            )
            if persisted is not None and persisted.request_hash == request_hash:
                original = await ModelStorageSyncTask.one_by_id(
                    session, persisted.resource_id
                )
                if original is not None:
                    return _to_public(original)
        # 无持久化等价结果：稳定冲突（绝不返回已回滚 own_task）。
        raise HTTPException(409, "idempotency_key_reused", "idempotency_key_reused")
    await session.refresh(task)
    await ModelStorageSyncTask._publish_event(EventType.CREATED, task)
    return _to_public(task)


async def _bind_idempotency_key_to_existing_task(
    session,
    user_id: int,
    idempotency_key: str,
    request_hash: str,
    task_id: int,
) -> int:
    """活动任务命中新 Idempotency-Key：把该 Key 绑定到既有任务并独立提交。

    竞争事务已整体回滚，这里在一个新的小事务中持久化 Key → 既有任务的
    绑定：绑定成功即“冲突语义稳定”——同一 Key 之后的任何重放都返回同一
    既有任务。若 Key 已被另一并发请求先行绑定（唯一约束冲突），视为等价
    结果。返回 Key 最终绑定的持久化任务 ID；若并发者以不同请求占用同一
    Key，则稳定冲突，调用方不得返回当前活动任务。
    """
    existing = await get_idempotency_record(
        session, user_id, SYNC_CREATE_OPERATION, idempotency_key
    )
    if existing is not None:
        if existing.request_hash != request_hash:
            raise HTTPException(409, "idempotency_key_reused", "idempotency_key_reused")
        return existing.resource_id
    try:
        session.add(
            new_idempotency_record(
                user_id,
                SYNC_CREATE_OPERATION,
                idempotency_key,
                request_hash,
                task_id,
                response_status=200,
                resource_type=SYNC_TASK_RESOURCE_TYPE,
            )
        )
        await session.commit()
        return task_id
    except IntegrityError:
        await session.rollback()
        existing = await get_idempotency_record(
            session, user_id, SYNC_CREATE_OPERATION, idempotency_key
        )
        if existing is None or existing.request_hash != request_hash:
            raise HTTPException(409, "idempotency_key_reused", "idempotency_key_reused")
        return existing.resource_id


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


async def _latest_ready_worker_for_model_file(session, model_file: ModelFile) -> Worker:
    """创建同步任务时校验 ModelFile 绑定的 Worker 可用性。

    ``model_file.worker_id`` 必须仍是同 ``worker_uuid`` 的**最新**注册记录
    且 ``state=READY``：Worker 离线（非 READY）或已被重新注册（旧注册记录
    不再是最新）时执行端会永久拒绝执行（``worker_not_current``），因此在
    创建时就返回稳定错误，避免产生不可执行的任务。
    """
    worker = await Worker.one_by_id(session, model_file.worker_id)
    if worker is None:
        raise ConflictException(message="model_file_worker_not_ready")
    if worker.state != WorkerStateEnum.READY:
        raise ConflictException(message="model_file_worker_not_ready")
    latest = (
        await session.exec(
            select(Worker)
            .where(Worker.worker_uuid == worker.worker_uuid)
            .order_by(Worker.id.desc())
        )
    ).first()
    if latest is None or latest.id != worker.id:
        raise ConflictException(message="model_file_worker_stale_registration")
    return worker


def _dedupe_key(model_file_id: int, profile_id: int) -> str:
    """活动同步任务去重槽位键：同 (model_file_id, profile_id) 全局唯一。"""
    return f"msync:{model_file_id}:{profile_id}"


async def _active_sync_task(session, dedupe_key: str) -> Optional[ModelStorageSyncTask]:
    """按去重槽位取活动任务；槽位缺失/悬挂时按 (model_file_id, profile_id)
    活动状态兜底查询（历史数据或异常悬挂的兜底路径）。"""
    slot = (
        await session.exec(
            select(ModelStorageSyncTaskDedupeSlot).where(
                ModelStorageSyncTaskDedupeSlot.dedupe_key == dedupe_key
            )
        )
    ).first()
    if slot is not None and slot.task_id is not None:
        task = await ModelStorageSyncTask.one_by_id(session, slot.task_id)
        if task is not None and task.state in _ACTIVE_STATES:
            return task
    model_file_id, profile_id = (int(part) for part in dedupe_key.split(":")[1:])
    return (
        await session.exec(
            select(ModelStorageSyncTask)
            .where(
                and_(
                    ModelStorageSyncTask.model_file_id == model_file_id,
                    ModelStorageSyncTask.profile_id == profile_id,
                    ModelStorageSyncTask.state.in_(list(_ACTIVE_STATES)),
                )
            )
            .order_by(ModelStorageSyncTask.id)
        )
    ).first()


async def _release_sync_task_slot(session, task: ModelStorageSyncTask) -> None:
    """任务进入终态时释放去重槽位。

    释放即删除槽位行（而不是把 ``task_id`` 置 NULL）：``dedupe_key`` 全局
    唯一，保留行会阻塞后续同键创建。只 flush 不 commit：由调用方与状态变更
    在**同一事务**中提交，保证“终态 + 释放槽位”原子生效。
    """
    slot = (
        await session.exec(
            select(ModelStorageSyncTaskDedupeSlot).where(
                ModelStorageSyncTaskDedupeSlot.task_id == task.id
            )
        )
    ).first()
    if slot is not None:
        await session.delete(slot)
        await session.flush()


@router.delete("/model-storage-sync-tasks/{id}")
async def cancel_model_storage_sync_task(session: SessionDep, id: int):
    """取消/删除：活动任务置为 canceled；终态任务直接删除。"""
    task = await ModelStorageSyncTask.one_by_id(session, id)
    if task is None:
        raise NotFoundException(message="model_storage_sync_task_not_found")
    if task.state in _ACTIVE_STATES:
        task.state = ModelStorageSyncTaskStateEnum.CANCELED
        # 终态与槽位释放在同一事务提交：失败时整体回滚，槽位不会悬挂。
        await _release_sync_task_slot(session, task)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            raise HTTPException(409, "sync_task_conflict", "sync_task_conflict")
        # 提交后刷新再广播 UPDATED（不重复 commit）。
        await session.refresh(task)
        await ModelStorageSyncTask._publish_event(EventType.UPDATED, task)
        return Response(status_code=200)
    # 删除终态任务前先断开槽位外键引用（兼容未开启 FK 的 SQLite 部署），
    # 随任务删除在同一事务提交。
    await _release_sync_task_slot(session, task)
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
async def refresh_profile_artifacts(
    request: Request, session: SessionDep, profile_id: int
):
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
    """从 ModelFile 推导运行时请求身份（source/model_id/实际文件选择）。

    只支持 ModelScope / Hugging Face 来源；其他来源或字段缺失时拒绝。
    ``file_patterns`` 描述模型仓库内的逻辑文件选择，物理绝对路径仅保存在
    ``scan_spec.root`` 供 Worker 扫描，不进入 request digest、Artifact 身份
    或 Manifest 相对路径。

    ``resolved_revision`` 只读取下载阶段写入的可信字段；缺失时直接拒绝，
    不回退到 ``requested_revision``，也不在这里猜测分支或标签名称。
    返回 ``(ModelPreheatIdentity, resolved_revision)``。identity 附带
    ``scan_spec``（冻结扫描规约）与 ``raw_patterns``（未编码规约，与库存
    存储形态一致）。
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
    # resolved_revision 必须只接受**真实** resolved_revision：缺失直接稳定拒绝。
    # 完全删除 ``requested_revision`` 回退与 moving-alias denylist（不猜分支名）：
    # dev/release 等请求分支缺失真实 resolved_revision 时一律拒绝，不得用
    # requested_revision 或别名冒充不可变 revision（任务身份与库存精确匹配
    # 依赖不可变 revision）。
    resolved_revision = model_file.resolved_revision
    if not resolved_revision:
        raise ConflictException(message="model_file_missing_resolved_revision")
    try:
        repository_complete = (
            model_file.huggingface_filename is None
            and model_file.model_scope_file_path is None
        )
        scan_root, raw_patterns = compute_scan_spec(
            list(model_file.resolved_paths),
            repository_complete=repository_complete,
        )
    except ValueError as exc:
        raise ConflictException(message="model_file_not_ready") from exc
    try:
        identity = ModelPreheatIdentity(
            source=source,
            model_id=model_id,
            revision=resolved_revision,
            requested_revision=model_file.requested_revision,
            file_patterns=tuple(raw_patterns),
            exclude_patterns=(),
        )
    except ModelPreheatIdentityError as exc:
        raise ConflictException(message="invalid_model_identity") from exc
    # 附带冻结扫描规约（供 request_identity 持久化与执行 payload 使用）。
    # patterns 统一使用规范化（编码+排序）形态：与库存
    # （``manifest.identity.file_patterns``）存储形态一致，预绑定精确匹配
    # 与 Worker 重建 Manifest 使用同一形态。
    # ModelPreheatIdentity 是 frozen dataclass：不向其实例挂属性，
    # 扫描规约通过独立 dict 返回。
    scan_spec = {
        "root": scan_root,
        "include_patterns": sorted(identity.file_patterns),
        "exclude_patterns": [],
    }
    return identity, resolved_revision, scan_spec


async def _exact_artifact_match(session, profile, identity) -> Optional[str]:
    """库存精确命中（唯一）时返回 artifact_id；否则 None。

    预绑定必须**精确**匹配请求身份的核心：source、model_id（编码后）、
    resolved revision（原始值）与实际文件选择（include/exclude patterns）。
    只按粗粒度 identity（source+model_id）匹配会把同一模型不同 revision 或
    不同文件选择的旧 Artifact 预绑定到本任务，导致发布无关文件。

    ``resolved_revision`` 库存与任务均存**原始**值（非编码），故与
    ``identity.revision``（原始）比较。JSON patterns 列的相等在三库语义不一致
    （SQLite 比 JSON 文本、PG 比 jsonb、MySQL 比文本），因此按三库兼容原则
    在 Python 侧做规范化列表比较。
    """
    rows = (
        await session.exec(
            select(ModelPreheatArtifact).where(
                and_(
                    ModelPreheatArtifact.profile_id == profile.id,
                    ModelPreheatArtifact.profile_config_version
                    == profile.config_version,
                    ModelPreheatArtifact.source == identity.source,
                    ModelPreheatArtifact.model_id == identity.model_path,
                    ModelPreheatArtifact.resolved_revision == identity.revision,
                    ModelPreheatArtifact.manifest_state
                    == ModelPreheatInventoryManifestStateEnum.VALID,
                )
            )
        )
    ).all()
    matched = [
        row.artifact_id
        for row in rows
        if list(row.include_patterns) == sorted(identity.file_patterns)
        and list(row.exclude_patterns) == sorted(identity.exclude_patterns)
    ]
    if len(matched) == 1:
        return matched[0]
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


def _ensure_credential_encryption_available(cipher: ModelPreheatCredentialCipher):
    """连接测试加密门禁：当前密钥存在且实际可用。

    用一次最小加密探针验证密钥可完成 AES-GCM 加密；密钥缺失、格式非法或
    加密失败都归为稳定的 ``CredentialEncryptionUnavailable``，由调用方统一
    映射为 ``credential_encryption_unavailable``（503），不误归类为连接失败。
    """
    if not cipher.current_key:
        raise CredentialEncryptionUnavailable("credential_encryption_unavailable")
    try:
        cipher.encrypt("")
    except CredentialEncryptionUnavailable:
        raise
    except Exception:
        raise CredentialEncryptionUnavailable(
            "credential_encryption_unavailable"
        ) from None


def _minio_client_factory(
    *,
    endpoint: str,
    access_key: str,
    secret_key: str,
    secure: bool,
    tls_verify: bool,
    region: Optional[str],
    use_virtual_hosted_style: bool,
    verified: Optional["VerifiedEndpoint"] = None,
    verified_host: Optional[str] = None,
    resolver: Optional[Callable[[str], Optional[str]]] = None,
):
    """构建连接测试专用的 minio client：DNS 固定 + 禁止重定向 + 短超时。

    - 请求 URL 固定为 ``{scheme}://{原始host:port}``（path style），Host 头、
      TLS SNI 与证书主机名校验全部基于**原始主机名**（证书语义）；
    - ``resolver``（来自受控 DNS 解析）被注入 urllib3 连接池：每条 TCP 连接
      一律连向已验证 IP，未验证的 host（包括 virtual 风格派生的
      ``{bucket}.{host}`` 或重定向目标）无法建立连接，防 DNS rebinding；
    - 不启用 virtual-style：连接测试统一 path style，保证 SNI/证书校验目标
      唯一且与已验证 IP 一一对应（入参 ``use_virtual_hosted_style`` 仅作为
      调用方语义参考，实际连接不派生新的请求 host）。
    """
    from minio import Minio

    parsed = validate_endpoint_url(endpoint)
    if verified is None:
        if not verified_host:
            raise ValueError("missing_verified_host")
        # 兼容旧调用：只有 IP 时退回默认解析（不再推荐，测试注入路径除外）。
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        verified = VerifiedEndpoint(parsed.scheme, host, port, verified_host)
        if resolver is None:
            resolver = lambda host: verified.verified_ip  # noqa: E731
    netloc = verified.netloc()
    http_client = build_pinned_http_client(
        verified=verified,
        resolver=resolver,
        tls_verify=secure and tls_verify,
    )
    client = Minio(
        netloc,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
        region=region,
        http_client=http_client,
    )
    # 连接测试统一 path style：请求 host 始终等于已验证的原始 host，
    # 避免派生 ``{bucket}.{host}`` 造成 SNI/证书与 DNS 固定语义分叉。
    client.disable_virtual_style_endpoint()
    return client


# ---------------------------------------------------------------------------
# 任务 3：Worker 侧同步执行端点（受 Worker 身份约束，凭据只进入执行 payload）
# ---------------------------------------------------------------------------


@worker_router.get(
    "",
    response_model=ModelStorageSyncTasksPublic,
)
async def list_or_watch_model_storage_sync_tasks(
    engine: EngineDep,
    session: SessionDep,
    params: ListParamsDep,
    identity: WorkerIdentityDep,
    worker_id: Optional[int] = None,
):
    """Worker 私有任务根 list/watch 端点（与项目 SSE watch 协议一致）。

    以认证 Worker principal 为唯一过滤依据：只返回本 Worker（最新注册
    ``worker_id`` 与 ``worker_uuid``）的任务；客户端传入的 ``worker_id``
    仅允许等于 principal 的 ``worker_id``，否则 403，不能用于越权查看
    其他 Worker 的任务。``watch=true`` 时复用
    :meth:`ModelStorageSyncTask.streaming` 事件流（心跳、CREATED/UPDATED/
    DELETED，事件 data 为 Public schema，不含凭据）。
    list 分支把 ORM 结果显式转换为 Public schema（不依赖默认 ORM 序列化），
    保证敏感字段（credential_snapshot_encrypted、encryption_key_version、
    lease_token_encrypted）不泄露。
    """
    if worker_id is not None and worker_id != identity.worker_id:
        raise HTTPException(403, "worker_not_authorized", "worker_not_authorized")
    fields = {
        "worker_id": identity.worker_id,
        "worker_uuid": identity.worker_uuid,
    }
    if params.watch:
        return StreamingResponse(
            ModelStorageSyncTask.streaming(engine, fields=fields),
            media_type="text/event-stream",
        )
    page = await ModelStorageSyncTask.paginated_by_query(
        session=session,
        fields=fields,
        page=params.page,
        per_page=params.perPage,
    )
    # 显式 Public 转换：逐条转为 Public schema（ORM → Public），
    # 敏感字段不进入响应。
    return ModelStorageSyncTasksPublic(
        items=[_to_public(item) for item in page.items],
        pagination=page.pagination,
    )


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

    canonical ``request_identity`` 只包含逻辑请求字段；物理 ``source_paths``
    与 ``scan_spec`` 来自任务创建时冻结的 AES-GCM 私有执行快照，绝不重读
    当前 ModelFile。``lease_token`` 明文只出现在本响应，complete/fail 必须回传。
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
        snapshot = _decrypt_execution_snapshot(cipher, task)
        profile = _execution_profile_from_snapshot(cipher, snapshot)
    except (ModelPreheatCredentialError, KeyError, TypeError, ValueError):
        raise ServiceUnavailableException(message="execution_credentials_unavailable")
    if task.lease_token_encrypted is None:
        # lease 快照缺失（密钥不可用/历史数据）：无法签发可校验的一次性
        # lease，稳定失败而不是发放无保护凭据。
        raise ServiceUnavailableException(message="execution_credentials_unavailable")
    try:
        lease_token = cipher.decrypt(task.lease_token_encrypted)
    except (ModelPreheatCredentialError, KeyError, TypeError, ValueError):
        raise ServiceUnavailableException(message="execution_credentials_unavailable")
    response.headers["Cache-Control"] = "no-store"
    # 物理执行语义来自加密私有快照；canonical request_identity 不含绝对路径。
    scan_spec = snapshot.get("scan_spec") or {}
    source_paths = list(snapshot.get("source_paths") or [])
    return ModelStorageSyncExecutionPayload(
        task_id=task.id,
        state=task.state,
        source=task.source,
        model_id=task.model_id,
        resolved_revision=task.resolved_revision,
        request_identity=task.request_identity,
        request_digest=task.request_digest,
        source_paths=source_paths,
        scan_spec={
            "root": scan_spec.get("root") or "",
            "include_patterns": list(scan_spec.get("include_patterns") or []),
            "exclude_patterns": list(scan_spec.get("exclude_patterns") or []),
        },
        lease_token=lease_token,
        profile=profile,
    )


@worker_router.post("/{task_id}/complete")
async def complete_model_storage_sync_task(
    request: Request,
    session: SessionDep,
    task_id: int,
    complete: ModelStorageSyncTaskComplete,
    identity: WorkerIdentityDep,
):
    """Worker 完成：校验 + CAS 推进终态并写入统一 Artifact 库存。

    契约校验（任务 3 子阶段 C）：

    - 身份隔离/lease：只允许任务所属且**当前注册**的 Worker（
      :func:`_authorized_sync_task`：worker_uuid 匹配且为最新注册），并且
      必须携带执行 payload 签发的**一次性 lease token**（数据库只存加密
      快照；错 lease/无 lease 一律稳定拒绝，防串任务与凭据泄露后的滥用）。
    - ``request_digest`` 必须等于任务创建时固定的值：拒绝过期/重放/串任务
      的完成请求。
    - ``artifact_id`` 一致性：未绑定时绑定本请求值，已预绑定时只允许确认
      同一 artifact（CAS，不覆盖为其他值）。
    - ``file_count`` / ``total_size`` 非负（Pydantic 校验）。

    终态语义（CAS 失败不再一律 200）：

    - 任务已 ready 且本次 complete 与完成时固定的值**完全一致**（同一
      lease、同一 request_digest、artifact_id、manifest_digest/path）：
      等价重放，幂等成功（200，不重复写库存）；
    - 任务已 ready 但 artifact_id 或 manifest 与固定值不同（不同 artifact /
      过期执行）：稳定冲突（409）；
    - 任务已 error/canceled 或 CAS 因状态/绑定变化未生效：稳定冲突（409）。

    成功后在同一事务：置 ready + 释放去重槽位 + upsert 统一 Artifact 库存
    （manifest_state=valid），再广播 UPDATED 事件；失败/取消保持槽位事务语义。
    库存写入使用**任务创建时固定的** ``task.profile_config_version``（而非
    当前 Profile 版本）：任务语义在创建时冻结，后续 Profile 配置变化不得
    改变本次发布结果的库存归属版本。
    """
    task = await _authorized_sync_task(session, task_id, identity)
    # lease 校验：错 lease/无 lease/lease 不可解析一律稳定拒绝（409）。
    if not _verify_sync_task_lease(request, task, complete.lease_token):
        raise ConflictException(message="lease_token_invalid")
    if task.state in _TERMINAL_STATES:
        if task.state == ModelStorageSyncTaskStateEnum.READY and _is_equivalent_replay(
            task, complete
        ):
            # 同一已完成执行的等价重放：幂等成功，不重复写库存/广播。
            return Response(status_code=200)
        # 不同 artifact / 过期执行 / 已被取消或失败：稳定冲突，不覆盖。
        raise ConflictException(message="sync_task_already_terminal")
    # request_digest 一致性：拒绝过期/重放/串任务的完成请求。
    if complete.request_digest != task.request_digest:
        raise ConflictException(message="request_digest_mismatch")
    # artifact_id 一致性：已预绑定且与本请求不同时不得覆盖（稳定冲突）。
    if task.artifact_id is not None and task.artifact_id != complete.artifact_id:
        raise ConflictException(message="artifact_id_mismatch")
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
        # 完成时固定的发布结果：等价重放幂等判定与审计使用。
        "manifest_digest": complete.manifest_digest,
        "manifest_path": complete.manifest_path,
        "finished_at": now,
    }
    # artifact_id CAS：仅当“未绑定”或“绑定为同一值”时推进，防止重复/过期
    # 完成覆盖。仅活动状态可完成：读取后任务若被取消，CAS 不再写 ready。
    result = await session.exec(
        update(ModelStorageSyncTask)
        .where(
            ModelStorageSyncTask.id == task.id,
            ModelStorageSyncTask.worker_uuid == identity.worker_uuid,
            or_(
                ModelStorageSyncTask.artifact_id.is_(None),
                ModelStorageSyncTask.artifact_id == complete.artifact_id,
            ),
            # 仅活动状态可完成：读取后任务若被取消，CAS 不再写 ready。
            ModelStorageSyncTask.state.in_(list(_ACTIVE_STATES)),
        )
        .values(artifact_id=complete.artifact_id, **values)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount == 0:
        # CAS 未生效：任务在读取后被取消/失败，或 artifact 绑定被并发改变。
        # 不覆盖、不折叠为成功：稳定冲突（Worker 重试将看到终态并走对应分支）。
        raise ConflictException(message="sync_task_conflict")
    # 成功后在同一事务写入/更新统一 Artifact 库存并发 UPDATED 事件。
    profile = await ModelPreheatS3Profile.one_by_id(session, task.profile_id)
    if profile is not None:
        await _upsert_sync_task_artifact(session, task, profile, complete, now)
    # 任务即将进入终态 ready：在同一事务内释放去重槽位，保证
    # “ready + 库存 + 槽位释放”原子生效，新任务可以立即创建。
    await _release_sync_task_slot(session, task)
    await session.commit()
    await task.refresh(session)
    await ModelStorageSyncTask._publish_event(EventType.UPDATED, task)
    return Response(status_code=200)


def _is_equivalent_replay(
    task: ModelStorageSyncTask, complete: ModelStorageSyncTaskComplete
) -> bool:
    """同一已完成执行的等价重放判定。

    任务已 ready 时，只有当 complete 携带的发布结果与完成时固定的值
    （artifact_id、request_digest、manifest_digest、manifest_path）全部一致
    才视为等价重放（lease 已在校验阶段确认一致）；任一项不同即“不同
    artifact / 过期执行”，必须稳定冲突。
    """
    if complete.artifact_id != task.artifact_id:
        return False
    if complete.request_digest != task.request_digest:
        return False
    if complete.manifest_digest != (task.manifest_digest or ""):
        return False
    if complete.manifest_path != (task.manifest_path or ""):
        return False
    return True


def _verify_sync_task_lease(
    request: Request, task: ModelStorageSyncTask, lease_token: str
) -> bool:
    """校验 complete/fail 携带的 lease token（解密任务加密快照做恒定时间比较）。

    任务 lease 快照缺失（历史数据/密钥不可用）或解密失败都稳定返回 False：
    拒绝优于误放行。
    """
    cipher = _cipher_from_request(request)
    if not cipher.current_key:
        return False
    return _lease_token_matches(cipher, task.lease_token_encrypted, lease_token)


@worker_router.post("/{task_id}/fail")
async def fail_model_storage_sync_task(
    request: Request,
    session: SessionDep,
    task_id: int,
    failure: ModelStorageSyncTaskFail,
    identity: WorkerIdentityDep,
):
    """Worker 失败：回写稳定错误码（需携带执行 lease token）。

    - 身份隔离/lease 与 complete 一致：无 lease/错 lease 稳定 409；
    - 任务已终态（ready/error/canceled）：不覆盖，稳定 409（重试的失败回写
      不得折叠为成功）；
    - CAS 未生效（读取后状态已变）：稳定 409。
    """
    task = await _authorized_sync_task(session, task_id, identity)
    # lease 校验：错 lease/无 lease 一律稳定拒绝（409）。
    if not _verify_sync_task_lease(request, task, failure.lease_token):
        raise ConflictException(message="lease_token_invalid")
    if task.state in _TERMINAL_STATES:
        # 已终态：失败回写不得覆盖（也不折叠为 200，避免 Worker 误判成功）。
        raise ConflictException(message="sync_task_already_terminal")
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
        # CAS 未生效：任务在读取后被取消或完成，不覆盖、不折叠为成功。
        raise ConflictException(message="sync_task_conflict")
    # 任务即将进入终态 error：同一事务释放去重槽位。
    await _release_sync_task_slot(session, task)
    await session.commit()
    await task.refresh(session)
    await ModelStorageSyncTask._publish_event(EventType.UPDATED, task)
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


async def _upsert_sync_task_artifact(
    session,
    task: ModelStorageSyncTask,
    profile: ModelPreheatS3Profile,
    complete: ModelStorageSyncTaskComplete,
    now,
) -> None:
    """同步任务成功后写入/更新统一 Artifact 库存（同一事务，只 flush）。

    库存字段与任务 1 发布器保持一致：``model_id``/``resolved_revision`` 存
    **原始**值（与任务及库存刷新路径一致），``include_patterns``/
    ``exclude_patterns`` 存规范化（编码、排序）列表。Manifest 对象 Key 必须
    由 Worker 按任务冻结的 Profile 快照生成并回传。``manifest_state=valid``，``last_verified_at``
    为完成时刻。只 flush 不 commit，随 complete 事务原子提交/回滚。

    库存归属版本必须是**任务创建时固定的** ``task.profile_config_version``
    （而非当前 ``profile.config_version``）：任务语义在创建时冻结，Profile
    后续配置变化不得把本次发布结果的库存挪到新版本（否则旧版本库存丢失、
    新版本被未扫描的发布污染）。Profile 行缺失时仍以任务固定版本写入库存
    （profile_id 外键保证 Profile 存在；缺失视为异常数据，库存归属版本不变）。
    """
    rid = task.request_identity or {}
    include_patterns = sorted(rid.get("include_patterns", []) or [])
    exclude_patterns = sorted(rid.get("exclude_patterns", []) or [])
    # 任务创建时固定的配置版本（不随 Profile 后续变化漂移）。
    pinned_config_version = task.profile_config_version
    existing = (
        await session.exec(
            select(ModelPreheatArtifact).where(
                and_(
                    ModelPreheatArtifact.profile_id == profile.id,
                    ModelPreheatArtifact.profile_config_version
                    == pinned_config_version,
                    ModelPreheatArtifact.artifact_id == complete.artifact_id,
                )
            )
        )
    ).first()
    if existing is None:
        existing = ModelPreheatArtifact(
            profile_id=profile.id,
            profile_config_version=pinned_config_version,
            artifact_id=complete.artifact_id,
        )
        session.add(existing)
    existing.source = task.source
    existing.model_id = task.model_id
    existing.resolved_revision = task.resolved_revision
    existing.include_patterns = include_patterns
    existing.exclude_patterns = exclude_patterns
    existing.manifest_path = complete.manifest_path
    existing.manifest_digest = complete.manifest_digest
    existing.file_count = complete.file_count
    existing.total_size = complete.total_size
    existing.manifest_state = ModelPreheatInventoryManifestStateEnum.VALID
    existing.last_verified_at = now
    await session.flush()


def _decrypt_execution_snapshot(
    cipher: ModelPreheatCredentialCipher, task: ModelStorageSyncTask
) -> dict:
    snapshot = task.credential_snapshot_encrypted
    if isinstance(snapshot, str):
        plaintext = cipher.decrypt(snapshot)
    else:
        plaintext = cipher.decrypt(json.dumps(snapshot))
    payload = json.loads(plaintext)
    if not isinstance(payload, dict):
        raise ValueError("invalid_execution_snapshot")
    return payload


def _execution_profile_from_snapshot(
    cipher: ModelPreheatCredentialCipher, payload: dict
) -> ModelStorageSyncExecutionProfile:
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
