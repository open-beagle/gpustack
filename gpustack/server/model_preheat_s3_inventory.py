import asyncio
import hashlib
import json
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Callable, Protocol
import uuid
from contextlib import asynccontextmanager

from sqlalchemy import and_, delete, exists, func, insert, literal, or_, update
from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.model_preheat_credentials import ModelPreheatCredentialCipher
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import (
    ModelPreheatCachedModel,
    ModelPreheatDesiredStateEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatInventoryGeneration,
    ModelPreheatInventoryGenerationStateEnum,
    ModelPreheatInventoryJob,
    ModelPreheatInventoryJobStateEnum,
    ModelPreheatInventoryManifestStateEnum,
    ModelPreheatInventoryScanSnapshot,
    ModelPreheatInventorySelectionLock,
    ModelPreheatPublicationMarker,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
from gpustack.worker.model_preheat.manifest import MAX_MANIFEST_BYTES
from gpustack.worker.model_preheat.s3_client import (
    MAX_READY_BYTES,
    ModelPreheatS3Client,
    ModelPreheatS3ManifestError,
)


MAX_INVENTORY_OBJECT_PATH = 4096
MAX_INVENTORY_OBJECTS = 100_000
MAX_GC_GENERATION_OBJECTS = 10_000
INVENTORY_JOB_LEASE = timedelta(minutes=5)
SELECTION_LOCK_LEASE = timedelta(minutes=5)
PUBLICATION_MARKER_RECENT = timedelta(hours=24)


class InventoryS3Error(RuntimeError):
    pass


@dataclass(frozen=True)
class InventoryRecord:
    cache_key: str
    source: str
    model_id: str
    resolved_revision: str
    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...]
    generation_id: str
    ready_path: str
    manifest_path: str
    manifest_digest: str
    file_count: int
    total_size: int
    manifest_state: str
    created_by_task_id: int | None = None


@dataclass(frozen=True)
class ScannedGeneration:
    generation_path: str
    ready_path: str
    ready_fingerprint: str | None
    referenced: bool
    ready_generation_path: str | None = None
    selection_key: str | None = None
    cache_key: str | None = None


@dataclass(frozen=True)
class InventoryScan:
    records: tuple[InventoryRecord, ...]
    generations: tuple[ScannedGeneration, ...]


@dataclass(frozen=True)
class InventoryGCResult:
    deleted: int = 0
    skipped: int = 0
    failed: int = 0


class InventoryStore(Protocol):
    def scan(self, profile) -> InventoryScan: ...

    def read_ready_reference(self, profile, ready_path: str): ...

    def iter_generation_objects(self, profile, generation_path: str): ...

    def delete_object(self, profile, object_path: str): ...


class ModelPreheatS3Inventory:
    def __init__(
        self,
        engine,
        *,
        store_factory: Callable[[object], InventoryStore] | None = None,
        config=None,
        poll_interval: float = 2,
        apply_batch_size: int = 128,
        max_gc_objects: int = MAX_GC_GENERATION_OBJECTS,
    ):
        self._engine = engine
        self._config = config
        self._store_factory = store_factory or (
            lambda profile: MinioInventoryStore(profile, config)
        )
        self._poll_interval = poll_interval
        self._apply_batch_size = max(1, apply_batch_size)
        self._max_gc_objects = max(1, max_gc_objects)

    @asynccontextmanager
    async def selection_guard(self, profile_id, selection_key, owner_token, operation):
        acquired = await self.acquire_selection_lock(
            profile_id, selection_key, owner_token, operation
        )
        try:
            yield owner_token if acquired else None
        finally:
            if acquired:
                await self.release_selection_lock(
                    profile_id, selection_key, owner_token
                )

    async def acquire_selection_lock(
        self, profile_id, selection_key, owner_token, operation
    ) -> bool:
        _validate_safe_identifier(selection_key)
        _validate_safe_identifier(owner_token)
        now = _utcnow()
        expires = now + SELECTION_LOCK_LEASE
        async with AsyncSession(self._engine) as session:
            existing = (
                await session.exec(
                    select(ModelPreheatInventorySelectionLock).where(
                        ModelPreheatInventorySelectionLock.profile_id == profile_id,
                        ModelPreheatInventorySelectionLock.selection_key
                        == selection_key,
                    )
                )
            ).first()
            if existing is None:
                session.add(
                    ModelPreheatInventorySelectionLock(
                        profile_id=profile_id,
                        selection_key=selection_key,
                        owner_token=owner_token,
                        operation=operation,
                        lease_expires_at=expires,
                    )
                )
                try:
                    await session.commit()
                    return True
                except IntegrityError:
                    await session.rollback()
            result = await session.exec(
                update(ModelPreheatInventorySelectionLock)
                .where(
                    ModelPreheatInventorySelectionLock.profile_id == profile_id,
                    ModelPreheatInventorySelectionLock.selection_key == selection_key,
                    or_(
                        ModelPreheatInventorySelectionLock.owner_token == owner_token,
                        ModelPreheatInventorySelectionLock.lease_expires_at <= now,
                    ),
                )
                .values(
                    owner_token=owner_token,
                    operation=operation,
                    lease_expires_at=expires,
                )
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            return result.rowcount == 1

    async def renew_selection_lock(self, profile_id, selection_key, owner_token):
        async with AsyncSession(self._engine) as session:
            result = await session.exec(
                update(ModelPreheatInventorySelectionLock)
                .where(
                    ModelPreheatInventorySelectionLock.profile_id == profile_id,
                    ModelPreheatInventorySelectionLock.selection_key == selection_key,
                    ModelPreheatInventorySelectionLock.owner_token == owner_token,
                )
                .values(lease_expires_at=_utcnow() + SELECTION_LOCK_LEASE)
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            return result.rowcount == 1

    async def release_selection_lock(self, profile_id, selection_key, owner_token):
        async with AsyncSession(self._engine) as session:
            await session.exec(
                delete(ModelPreheatInventorySelectionLock).where(
                    ModelPreheatInventorySelectionLock.profile_id == profile_id,
                    ModelPreheatInventorySelectionLock.selection_key == selection_key,
                    ModelPreheatInventorySelectionLock.owner_token == owner_token,
                )
            )
            await session.commit()

    async def ensure_publication_marker(self, task: ModelPreheatTask) -> bool:
        if task.id is None or not _is_preheat_generation_id(task.generation_id):
            return False
        snapshot = {
            "task_id": task.id,
            "attempt": task.attempt,
            "profile_id": task.s3_profile_id,
            "profile_version": task.s3_profile_config_version,
            "selection_key": task.cache_key,
            "generation_id": task.generation_id,
        }
        owner_token = uuid.uuid4().hex
        async with self.selection_guard(
            snapshot["profile_id"],
            snapshot["selection_key"],
            owner_token,
            "publication",
        ) as lock_owner:
            if lock_owner is None:
                return False
            async with AsyncSession(self._engine) as session:
                current = await session.get(ModelPreheatTask, snapshot["task_id"])
                if not _task_matches_publication_marker(current, snapshot):
                    return False
                values = {
                    "task_id": current.id,
                    "parent_attempt": current.attempt,
                    "profile_config_version": current.s3_profile_config_version,
                    "terminated_at": None,
                }
                marker = (
                    await session.exec(
                        select(ModelPreheatPublicationMarker).where(
                            ModelPreheatPublicationMarker.profile_id
                            == current.s3_profile_id,
                            ModelPreheatPublicationMarker.selection_key
                            == current.cache_key,
                            ModelPreheatPublicationMarker.generation_id
                            == current.generation_id,
                        )
                    )
                ).first()
                if marker is not None:
                    updated = await session.exec(
                        update(ModelPreheatPublicationMarker)
                        .where(
                            ModelPreheatPublicationMarker.id == marker.id,
                            ModelPreheatPublicationMarker.task_id == marker.task_id,
                            ModelPreheatPublicationMarker.parent_attempt
                            == marker.parent_attempt,
                            ModelPreheatPublicationMarker.profile_config_version
                            == marker.profile_config_version,
                            ModelPreheatPublicationMarker.terminated_at
                            == marker.terminated_at,
                        )
                        .values(**values)
                        .execution_options(synchronize_session=False)
                    )
                    if updated.rowcount == 0:
                        await session.rollback()
                        return False
                else:
                    try:
                        async with session.begin_nested():
                            session.add(
                                ModelPreheatPublicationMarker(
                                    profile_id=current.s3_profile_id,
                                    selection_key=current.cache_key,
                                    generation_id=current.generation_id,
                                    **values,
                                )
                            )
                            await session.flush()
                    except IntegrityError:
                        marker = (
                            await session.exec(
                                select(ModelPreheatPublicationMarker).where(
                                    ModelPreheatPublicationMarker.profile_id
                                    == current.s3_profile_id,
                                    ModelPreheatPublicationMarker.selection_key
                                    == current.cache_key,
                                    ModelPreheatPublicationMarker.generation_id
                                    == current.generation_id,
                                )
                            )
                        ).first()
                        if marker is None:
                            await session.rollback()
                            return False
                        retry = await session.exec(
                            update(ModelPreheatPublicationMarker)
                            .where(
                                ModelPreheatPublicationMarker.id == marker.id,
                                ModelPreheatPublicationMarker.task_id == marker.task_id,
                                ModelPreheatPublicationMarker.parent_attempt
                                == marker.parent_attempt,
                                ModelPreheatPublicationMarker.profile_config_version
                                == marker.profile_config_version,
                                ModelPreheatPublicationMarker.terminated_at
                                == marker.terminated_at,
                            )
                            .values(**values)
                            .execution_options(synchronize_session=False)
                        )
                        if retry.rowcount == 0:
                            await session.rollback()
                            return False
                await session.commit()
                return True

    async def release_verified_publication_marker(self, session, task):
        await session.exec(
            delete(ModelPreheatPublicationMarker).where(
                ModelPreheatPublicationMarker.profile_id == task.s3_profile_id,
                ModelPreheatPublicationMarker.selection_key == task.cache_key,
                ModelPreheatPublicationMarker.generation_id == task.generation_id,
            )
        )

    async def terminate_publication_marker(self, session, task):
        result = await session.exec(
            update(ModelPreheatPublicationMarker)
            .where(
                ModelPreheatPublicationMarker.task_id == task.id,
                ModelPreheatPublicationMarker.parent_attempt == task.attempt,
                ModelPreheatPublicationMarker.generation_id == task.generation_id,
                ModelPreheatPublicationMarker.terminated_at.is_(None),
            )
            .values(terminated_at=_utcnow())
            .execution_options(synchronize_session=False)
        )
        return result.rowcount > 0

    async def start(self):
        while True:
            await self.run_pending_jobs()
            await asyncio.sleep(self._poll_interval)

    async def create_refresh_job(self, session, profile_id: int, config_version: int):
        return await self._create_job(session, profile_id, config_version, "refresh")

    async def create_gc_job(self, session, profile_id: int, config_version: int):
        return await self._create_job(session, profile_id, config_version, "gc")

    async def _create_job(self, session, profile_id, config_version, kind):
        active_key = f"{kind}:{profile_id}:{config_version}"
        existing = (
            await session.exec(
                select(ModelPreheatInventoryJob).where(
                    ModelPreheatInventoryJob.active_key == active_key
                )
            )
        ).first()
        if existing is not None:
            return existing
        job = ModelPreheatInventoryJob(
            profile_id=profile_id,
            profile_config_version=config_version,
            kind=kind,
            active_key=active_key,
        )
        session.add(job)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            existing = (
                await session.exec(
                    select(ModelPreheatInventoryJob).where(
                        ModelPreheatInventoryJob.active_key == active_key
                    )
                )
            ).first()
            if existing is None:
                raise
            return existing
        await session.refresh(job)
        return job

    async def run_pending_jobs(self):
        async with AsyncSession(self._engine) as session:
            now = _utcnow()
            await session.exec(
                update(ModelPreheatInventoryJob)
                .where(
                    ModelPreheatInventoryJob.state
                    == ModelPreheatInventoryJobStateEnum.RUNNING,
                    ModelPreheatInventoryJob.lease_expires_at < now,
                )
                .values(
                    state=ModelPreheatInventoryJobStateEnum.PENDING,
                    lease_expires_at=None,
                    claim_token=None,
                )
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            ids = (
                await session.exec(
                    select(ModelPreheatInventoryJob.id).where(
                        ModelPreheatInventoryJob.state
                        == ModelPreheatInventoryJobStateEnum.PENDING
                    )
                )
            ).all()
        for job_id in ids:
            await self.run_job(job_id)

    async def run_job(self, job_id: int):
        now = _utcnow()
        claim_token = uuid.uuid4().hex
        async with AsyncSession(self._engine) as session:
            claimed = await session.exec(
                update(ModelPreheatInventoryJob)
                .where(
                    ModelPreheatInventoryJob.id == job_id,
                    ModelPreheatInventoryJob.state
                    == ModelPreheatInventoryJobStateEnum.PENDING,
                )
                .values(
                    state=ModelPreheatInventoryJobStateEnum.RUNNING,
                    started_at=now,
                    scan_started_at=now,
                    lease_expires_at=now + INVENTORY_JOB_LEASE,
                    claim_token=claim_token,
                )
                .execution_options(synchronize_session=False)
            )
            if claimed.rowcount != 1:
                await session.rollback()
                return
            job = await session.get(ModelPreheatInventoryJob, job_id)
            profile = await session.get(ModelPreheatS3Profile, job.profile_id)
            profile_changed = (
                profile is not None
                and profile.config_version != job.profile_config_version
            )
            if profile is None:
                profile = SimpleNamespace(
                    id=job.profile_id, config_version=job.profile_config_version
                )
            if job.kind == "refresh":
                await session.exec(
                    delete(ModelPreheatInventoryScanSnapshot).where(
                        ModelPreheatInventoryScanSnapshot.job_id == job.id
                    )
                )
                await session.exec(
                    insert(ModelPreheatInventoryScanSnapshot).from_select(
                        ["job_id", "cached_model_id", "revision"],
                        select(
                            literal(job.id),
                            ModelPreheatCachedModel.id,
                            ModelPreheatCachedModel.revision,
                        ).where(ModelPreheatCachedModel.profile_id == job.profile_id),
                    )
                )
            await session.commit()
            await session.refresh(job)
            if isinstance(profile, ModelPreheatS3Profile):
                await session.refresh(profile)

        if profile_changed:
            async with AsyncSession(self._engine) as session:
                await session.exec(
                    update(ModelPreheatInventoryJob)
                    .where(
                        ModelPreheatInventoryJob.id == job_id,
                        ModelPreheatInventoryJob.claim_token == claim_token,
                        ModelPreheatInventoryJob.lease_expires_at > now,
                        _profile_version_is_current(
                            job.profile_id, job.profile_config_version
                        ),
                        ModelPreheatInventoryJob.state
                        == ModelPreheatInventoryJobStateEnum.RUNNING,
                    )
                    .values(
                        state=ModelPreheatInventoryJobStateEnum.ERROR,
                        active_key=None,
                        lease_expires_at=None,
                        claim_token=None,
                        error_code="inventory_profile_changed",
                        error_message="inventory_profile_changed",
                        finished_at=_utcnow(),
                    )
                )
                await session.commit()
            return

        if job.kind == "gc":
            retention_seconds = getattr(
                self._config, "model_preheat_s3_orphan_retention_seconds", 86400
            )
            retention_seconds = min(max(int(retention_seconds), 3600), 30 * 86400)
            try:
                result = await self.run_gc(
                    job.profile_id,
                    retention=timedelta(seconds=retention_seconds),
                    job_id=job_id,
                    claim_token=claim_token,
                )
            except Exception:
                result = InventoryGCResult(failed=1)
            async with AsyncSession(self._engine) as session:
                await session.exec(
                    update(ModelPreheatInventoryJob)
                    .where(
                        ModelPreheatInventoryJob.id == job_id,
                        ModelPreheatInventoryJob.claim_token == claim_token,
                        ModelPreheatInventoryJob.state
                        == ModelPreheatInventoryJobStateEnum.RUNNING,
                    )
                    .values(
                        state=(
                            ModelPreheatInventoryJobStateEnum.ERROR
                            if result.failed
                            else ModelPreheatInventoryJobStateEnum.READY
                        ),
                        active_key=None,
                        lease_expires_at=None,
                        claim_token=None,
                        deleted_count=result.deleted,
                        skipped_count=result.skipped,
                        failed_count=result.failed,
                        error_code=("inventory_gc_failed" if result.failed else None),
                        error_message=(
                            "inventory_gc_failed" if result.failed else None
                        ),
                        finished_at=_utcnow(),
                    )
                )
                await session.commit()
            return

        try:
            scan = await asyncio.to_thread(self._store_factory(profile).scan, profile)
            await self._apply_scan(job_id, scan, claim_token)
        except Exception:
            async with AsyncSession(self._engine) as session:
                await session.exec(
                    update(ModelPreheatInventoryJob)
                    .where(
                        ModelPreheatInventoryJob.id == job_id,
                        ModelPreheatInventoryJob.claim_token == claim_token,
                        ModelPreheatInventoryJob.state
                        == ModelPreheatInventoryJobStateEnum.RUNNING,
                    )
                    .values(
                        state=ModelPreheatInventoryJobStateEnum.ERROR,
                        active_key=None,
                        lease_expires_at=None,
                        claim_token=None,
                        error_code="inventory_scan_failed",
                        error_message="inventory_scan_failed",
                        finished_at=_utcnow(),
                    )
                )
                await session.commit()

    async def _apply_scan(self, job_id: int, scan: InventoryScan, claim_token: str):
        now = _utcnow()
        lock_keys = sorted(
            {record.cache_key for record in scan.records}
            | {
                scanned.selection_key
                or scanned.cache_key
                or "invalid-" + _path_key(scanned.ready_path)
                for scanned in scan.generations
            }
        )
        acquired = []
        try:
            try:
                scan_profile_id = await self._claimed_profile_id(job_id, claim_token)
            except InventoryS3Error:
                return
            lock_keys = sorted(
                set(lock_keys) | await self._selection_keys_for_profile(scan_profile_id)
            )
            next_acquisition_renewal = _utcnow() + SELECTION_LOCK_LEASE / 3
            for selection_key in lock_keys:
                if not await self.acquire_selection_lock(
                    scan_profile_id,
                    selection_key,
                    claim_token,
                    "refresh",
                ):
                    raise InventoryS3Error("inventory_selection_busy")
                acquired.append((scan_profile_id, selection_key))
                if _utcnow() >= next_acquisition_renewal:
                    if not await self._renew_apply_selection_locks(
                        scan_profile_id,
                        [key for _, key in acquired],
                        claim_token,
                    ):
                        raise InventoryS3Error("inventory_selection_lost")
                    next_acquisition_renewal = _utcnow() + SELECTION_LOCK_LEASE / 3

            if not await self._renew_apply_selection_locks(
                scan_profile_id, lock_keys, claim_token
            ):
                raise InventoryS3Error("inventory_selection_lost")
            next_lock_renewal = _utcnow() + SELECTION_LOCK_LEASE / 3

            async with AsyncSession(self._engine) as session:
                job = await session.get(ModelPreheatInventoryJob, job_id)
                if (
                    job is None
                    or job.state != ModelPreheatInventoryJobStateEnum.RUNNING
                    or job.claim_token != claim_token
                ):
                    return
                cached = (
                    await session.exec(
                        select(ModelPreheatCachedModel).where(
                            ModelPreheatCachedModel.profile_id == job.profile_id
                        )
                    )
                ).all()
                cached_by_key = {item.cache_key: item for item in cached}
                snapshot_revisions = dict(
                    (
                        await session.exec(
                            select(
                                ModelPreheatInventoryScanSnapshot.cached_model_id,
                                ModelPreheatInventoryScanSnapshot.revision,
                            ).where(ModelPreheatInventoryScanSnapshot.job_id == job.id)
                        )
                    ).all()
                )
                generations = (
                    await session.exec(
                        select(ModelPreheatInventoryGeneration).where(
                            ModelPreheatInventoryGeneration.profile_id == job.profile_id
                        )
                    )
                ).all()
                generations_by_key = {item.generation_key: item for item in generations}
                seen = set()
                valid_count = 0
                invalid_count = 0
                processed = 0
                for record in scan.records:
                    _validate_record(record)
                    if record.cache_key in seen:
                        raise InventoryS3Error("inventory_duplicate_cache_key")
                    seen.add(record.cache_key)
                    state = ModelPreheatInventoryManifestStateEnum(
                        record.manifest_state
                    )
                    valid_count += state == ModelPreheatInventoryManifestStateEnum.VALID
                    invalid_count += (
                        state == ModelPreheatInventoryManifestStateEnum.INVALID
                    )
                    item = cached_by_key.get(record.cache_key)
                    values = _record_values(record, now, job.profile_config_version)
                    if item is None:
                        item = ModelPreheatCachedModel(
                            profile_id=job.profile_id, **values
                        )
                        session.add(item)
                    else:
                        snapshot_revision = snapshot_revisions.get(item.id)
                        if snapshot_revision is not None:
                            await session.exec(
                                update(ModelPreheatCachedModel)
                                .where(
                                    ModelPreheatCachedModel.id == item.id,
                                    ModelPreheatCachedModel.revision
                                    == snapshot_revision,
                                )
                                .values(
                                    **values,
                                    revision=ModelPreheatCachedModel.revision + 1,
                                )
                                .execution_options(synchronize_session=False)
                            )
                    processed += 1
                    if processed % self._apply_batch_size == 0:
                        if not await self._renew_claim(job_id, claim_token):
                            await session.rollback()
                            return
                        if _utcnow() >= next_lock_renewal:
                            if not await self._renew_apply_selection_locks(
                                scan_profile_id, lock_keys, claim_token
                            ):
                                await session.rollback()
                                return
                            next_lock_renewal = _utcnow() + SELECTION_LOCK_LEASE / 3

                missing_predicates = [
                    ModelPreheatCachedModel.profile_id == job.profile_id,
                    exists(
                        select(ModelPreheatInventoryScanSnapshot.id).where(
                            ModelPreheatInventoryScanSnapshot.job_id == job.id,
                            ModelPreheatInventoryScanSnapshot.cached_model_id
                            == ModelPreheatCachedModel.id,
                            ModelPreheatInventoryScanSnapshot.revision
                            == ModelPreheatCachedModel.revision,
                        )
                    ),
                ]
                if seen:
                    missing_predicates.append(
                        ModelPreheatCachedModel.cache_key.not_in(seen)
                    )
                await session.exec(
                    update(ModelPreheatCachedModel)
                    .where(*missing_predicates)
                    .values(
                        manifest_state=ModelPreheatInventoryManifestStateEnum.MISSING,
                        last_verified_at=now,
                        revision=ModelPreheatCachedModel.revision + 1,
                    )
                    .execution_options(synchronize_session=False)
                )

                orphan_count = 0
                for scanned in scan.generations:
                    _validate_generation(scanned)
                    state = (
                        ModelPreheatInventoryGenerationStateEnum.REFERENCED
                        if scanned.referenced
                        else ModelPreheatInventoryGenerationStateEnum.ORPHAN
                    )
                    orphan_count += not scanned.referenced
                    generation_key = _path_key(scanned.generation_path)
                    item = generations_by_key.get(generation_key)
                    previous_state = item.state if item is not None else None
                    if item is None:
                        item = ModelPreheatInventoryGeneration(
                            profile_id=job.profile_id,
                            generation_key=generation_key,
                            selection_key=scanned.selection_key
                            or scanned.cache_key
                            or "invalid-" + _path_key(scanned.ready_path),
                            cache_key=scanned.cache_key,
                            generation_path=scanned.generation_path,
                            ready_path=scanned.ready_path,
                            first_seen_at=now,
                            last_seen_at=now,
                            state=state,
                        )
                    item.ready_path = scanned.ready_path
                    item.ready_fingerprint = scanned.ready_fingerprint
                    item.ready_generation_path = scanned.ready_generation_path
                    item.last_seen_at = now
                    item.state = state
                    if state == ModelPreheatInventoryGenerationStateEnum.REFERENCED:
                        item.orphaned_at = None
                    elif previous_state not in {
                        ModelPreheatInventoryGenerationStateEnum.ORPHAN,
                        ModelPreheatInventoryGenerationStateEnum.ERROR,
                    }:
                        item.orphaned_at = now
                    item.error_code = None
                    session.add(item)
                    processed += 1
                    if processed % self._apply_batch_size == 0:
                        if not await self._renew_claim(job_id, claim_token):
                            await session.rollback()
                            return
                        if _utcnow() >= next_lock_renewal:
                            if not await self._renew_apply_selection_locks(
                                scan_profile_id, lock_keys, claim_token
                            ):
                                await session.rollback()
                                return
                            next_lock_renewal = _utcnow() + SELECTION_LOCK_LEASE / 3

                completed = await session.exec(
                    update(ModelPreheatInventoryJob)
                    .where(
                        ModelPreheatInventoryJob.id == job_id,
                        ModelPreheatInventoryJob.state
                        == ModelPreheatInventoryJobStateEnum.RUNNING,
                        ModelPreheatInventoryJob.claim_token == claim_token,
                        ModelPreheatInventoryJob.lease_expires_at > now,
                        _profile_version_is_current(
                            job.profile_id, job.profile_config_version
                        ),
                    )
                    .values(
                        state=ModelPreheatInventoryJobStateEnum.READY,
                        active_key=None,
                        lease_expires_at=None,
                        claim_token=None,
                        scanned_count=len(scan.records),
                        valid_count=valid_count,
                        invalid_count=invalid_count,
                        orphan_count=orphan_count,
                        finished_at=now,
                    )
                    .execution_options(synchronize_session=False)
                )
                if completed.rowcount != 1:
                    await session.rollback()
                    return
                if not await self._validate_apply_selection_locks(
                    scan_profile_id, lock_keys, claim_token
                ):
                    await session.rollback()
                    return
                await session.commit()
        finally:
            for profile_id, selection_key in reversed(acquired):
                await self.release_selection_lock(
                    profile_id, selection_key, claim_token
                )

    async def _renew_apply_selection_locks(
        self, profile_id, selection_keys, owner_token
    ):
        if not selection_keys:
            return True
        now = _utcnow()
        for keys in _chunks(selection_keys, 500):
            async with AsyncSession(self._engine) as session:
                renewed = await session.exec(
                    update(ModelPreheatInventorySelectionLock)
                    .where(
                        ModelPreheatInventorySelectionLock.profile_id == profile_id,
                        ModelPreheatInventorySelectionLock.selection_key.in_(keys),
                        ModelPreheatInventorySelectionLock.owner_token == owner_token,
                        ModelPreheatInventorySelectionLock.lease_expires_at > now,
                    )
                    .values(lease_expires_at=now + SELECTION_LOCK_LEASE)
                    .execution_options(synchronize_session=False)
                )
                await session.commit()
                if renewed.rowcount != len(keys):
                    return False
        return True

    async def _validate_apply_selection_locks(
        self, profile_id, selection_keys, owner_token
    ):
        if not selection_keys:
            return True
        now = _utcnow()
        for keys in _chunks(selection_keys, 500):
            async with AsyncSession(self._engine) as session:
                count = (
                    await session.exec(
                        select(func.count(ModelPreheatInventorySelectionLock.id)).where(
                            ModelPreheatInventorySelectionLock.profile_id == profile_id,
                            ModelPreheatInventorySelectionLock.selection_key.in_(keys),
                            ModelPreheatInventorySelectionLock.owner_token
                            == owner_token,
                            ModelPreheatInventorySelectionLock.lease_expires_at > now,
                        )
                    )
                ).one()
                if count != len(keys):
                    return False
        return True

    async def _claimed_profile_id(self, job_id, claim_token):
        async with AsyncSession(self._engine) as session:
            job = (
                await session.exec(
                    select(ModelPreheatInventoryJob).where(
                        ModelPreheatInventoryJob.id == job_id,
                        ModelPreheatInventoryJob.state
                        == ModelPreheatInventoryJobStateEnum.RUNNING,
                        ModelPreheatInventoryJob.claim_token == claim_token,
                        ModelPreheatInventoryJob.lease_expires_at > _utcnow(),
                    )
                )
            ).first()
            if job is None:
                raise InventoryS3Error("inventory_claim_lost")
            return job.profile_id

    async def _selection_keys_for_profile(self, profile_id):
        async with AsyncSession(self._engine) as session:
            cached = (
                await session.exec(
                    select(ModelPreheatCachedModel.cache_key).where(
                        ModelPreheatCachedModel.profile_id == profile_id
                    )
                )
            ).all()
            generations = (
                await session.exec(
                    select(ModelPreheatInventoryGeneration.selection_key).where(
                        ModelPreheatInventoryGeneration.profile_id == profile_id
                    )
                )
            ).all()
            return set(cached) | set(generations)

    async def _renew_claim(self, job_id, claim_token):
        now = _utcnow()
        async with AsyncSession(self._engine) as session:
            renewed = await session.exec(
                update(ModelPreheatInventoryJob)
                .where(
                    ModelPreheatInventoryJob.id == job_id,
                    ModelPreheatInventoryJob.state
                    == ModelPreheatInventoryJobStateEnum.RUNNING,
                    ModelPreheatInventoryJob.claim_token == claim_token,
                    ModelPreheatInventoryJob.lease_expires_at > now,
                )
                .values(lease_expires_at=now + INVENTORY_JOB_LEASE)
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            return renewed.rowcount == 1

    async def _publication_marker_protects_generation(
        self, profile_id, candidate, lock_owner
    ):
        generation_id = candidate.generation_path.rsplit("/", 1)[-1]
        now = _utcnow()
        async with AsyncSession(self._engine) as session:
            lock = (
                await session.exec(
                    select(ModelPreheatInventorySelectionLock.id).where(
                        ModelPreheatInventorySelectionLock.profile_id == profile_id,
                        ModelPreheatInventorySelectionLock.selection_key
                        == candidate.selection_key,
                        ModelPreheatInventorySelectionLock.owner_token == lock_owner,
                        ModelPreheatInventorySelectionLock.lease_expires_at > now,
                    )
                )
            ).first()
            if lock is None:
                raise InventoryS3Error("inventory_selection_lost")
            marker = (
                await session.exec(
                    select(ModelPreheatPublicationMarker).where(
                        ModelPreheatPublicationMarker.profile_id == profile_id,
                        ModelPreheatPublicationMarker.selection_key
                        == candidate.selection_key,
                        ModelPreheatPublicationMarker.generation_id == generation_id,
                    )
                )
            ).first()
            if marker is None:
                return False
            task = (
                await session.get(ModelPreheatTask, marker.task_id)
                if marker.task_id is not None
                else None
            )
            parent_matches = (
                task is not None
                and task.attempt == marker.parent_attempt
                and task.s3_profile_id == marker.profile_id
                and task.s3_profile_config_version == marker.profile_config_version
                and task.cache_key == marker.selection_key
                and task.generation_id == marker.generation_id
            )
            terminal_states = {
                ModelPreheatExecutionStateEnum.READY,
                ModelPreheatExecutionStateEnum.PARTIAL,
                ModelPreheatExecutionStateEnum.ERROR,
                ModelPreheatExecutionStateEnum.CANCELED,
            }
            parent_live = parent_matches and task.execution_state not in terminal_states
            parent_terminal = parent_matches and task.execution_state in terminal_states
            active_lease = False
            if marker.task_id is not None:
                active_lease = (
                    await session.exec(
                        select(ModelPreheatWorkerTask.id).where(
                            ModelPreheatWorkerTask.task_id == marker.task_id,
                            ModelPreheatWorkerTask.parent_attempt
                            == marker.parent_attempt,
                            ModelPreheatWorkerTask.role
                            == ModelPreheatWorkerTaskRoleEnum.SEED,
                            ModelPreheatWorkerTask.state
                            == ModelPreheatWorkerTaskStateEnum.RUNNING,
                            ModelPreheatWorkerTask.lease_expires_at.is_not(None),
                            ModelPreheatWorkerTask.lease_expires_at > now,
                        )
                    )
                ).first() is not None
            if parent_live or active_lease:
                return True
            if parent_terminal and marker.terminated_at is None:
                marked = await session.exec(
                    update(ModelPreheatPublicationMarker)
                    .where(
                        ModelPreheatPublicationMarker.id == marker.id,
                        ModelPreheatPublicationMarker.updated_at == marker.updated_at,
                        ModelPreheatPublicationMarker.terminated_at.is_(None),
                    )
                    .values(terminated_at=now)
                    .execution_options(synchronize_session=False)
                )
                await session.commit()
                return True
            grace_anchor = marker.terminated_at or marker.updated_at
            if grace_anchor >= now - PUBLICATION_MARKER_RECENT:
                return True
            removed = await session.exec(
                delete(ModelPreheatPublicationMarker).where(
                    ModelPreheatPublicationMarker.id == marker.id,
                    ModelPreheatPublicationMarker.updated_at == marker.updated_at,
                    ModelPreheatPublicationMarker.terminated_at == marker.terminated_at,
                )
            )
            await session.commit()
            return removed.rowcount == 0

    async def run_gc(
        self,
        profile_id: int,
        *,
        retention: timedelta,
        job_id: int | None = None,
        claim_token: str | None = None,
    ):
        now = _utcnow()
        cutoff = now - retention
        async with AsyncSession(self._engine) as session:
            active_lock = exists(
                select(ModelPreheatInventorySelectionLock.id).where(
                    ModelPreheatInventorySelectionLock.profile_id
                    == ModelPreheatInventoryGeneration.profile_id,
                    ModelPreheatInventorySelectionLock.selection_key
                    == ModelPreheatInventoryGeneration.selection_key,
                    ModelPreheatInventorySelectionLock.lease_expires_at > now,
                )
            )
            await session.exec(
                update(ModelPreheatInventoryGeneration)
                .where(
                    ModelPreheatInventoryGeneration.profile_id == profile_id,
                    ModelPreheatInventoryGeneration.state
                    == ModelPreheatInventoryGenerationStateEnum.DELETING,
                    ~active_lock,
                )
                .values(
                    state=ModelPreheatInventoryGenerationStateEnum.ERROR,
                    error_code="inventory_gc_interrupted",
                )
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            profile = await session.get(ModelPreheatS3Profile, profile_id)
            if profile is None:
                profile = SimpleNamespace(id=profile_id)
            candidates = (
                await session.exec(
                    select(ModelPreheatInventoryGeneration).where(
                        ModelPreheatInventoryGeneration.profile_id == profile_id,
                        ModelPreheatInventoryGeneration.state.in_(
                            (
                                ModelPreheatInventoryGenerationStateEnum.ORPHAN,
                                ModelPreheatInventoryGenerationStateEnum.ERROR,
                            )
                        ),
                        ModelPreheatInventoryGeneration.orphaned_at.is_not(None),
                        ModelPreheatInventoryGeneration.orphaned_at < cutoff,
                    )
                )
            ).all()
        store = self._store_factory(profile)
        deleted = skipped = failed = 0
        for candidate in candidates:
            if job_id is not None and not await self._renew_claim(job_id, claim_token):
                raise InventoryS3Error("inventory_claim_lost")
            owner_token = claim_token or uuid.uuid4().hex
            acquired = await self.acquire_selection_lock(
                profile_id,
                candidate.selection_key,
                owner_token,
                "gc",
            )
            if not acquired:
                skipped += 1
                continue
            ownership = {
                "profile_id": profile_id,
                "lock_owner": owner_token,
                "job_id": job_id,
                "claim_token": claim_token,
            }
            try:
                if await self._publication_marker_protects_generation(
                    profile_id, candidate, owner_token
                ):
                    skipped += 1
                    continue
                if not await self._claim_generation_for_gc(candidate, **ownership):
                    skipped += 1
                    continue
                reference = await asyncio.to_thread(
                    store.read_ready_reference, profile, candidate.ready_path
                )
                if reference is None:
                    reference = (None, None)
                if reference[0] == candidate.generation_path or reference != (
                    candidate.ready_generation_path,
                    candidate.ready_fingerprint,
                ):
                    skipped += 1
                    await self._finish_generation_gc(
                        candidate,
                        ModelPreheatInventoryGenerationStateEnum.ORPHAN,
                        None,
                        **ownership,
                    )
                    continue
                objects = await self._spool_generation_objects(
                    store,
                    profile,
                    candidate.generation_path,
                    profile_id,
                    candidate.selection_key,
                    owner_token,
                    job_id,
                    claim_token,
                )
                try:
                    for raw_path in objects:
                        if job_id is not None and not await self._renew_claim(
                            job_id, claim_token
                        ):
                            raise InventoryS3Error("inventory_claim_lost")
                        if not await self.renew_selection_lock(
                            profile_id, candidate.selection_key, owner_token
                        ):
                            raise InventoryS3Error("inventory_selection_lost")
                        if await self._publication_marker_protects_generation(
                            profile_id, candidate, owner_token
                        ):
                            raise _PublishingGenerationProtected
                        current_reference = await asyncio.to_thread(
                            store.read_ready_reference,
                            profile,
                            candidate.ready_path,
                        )
                        if current_reference is None:
                            current_reference = (None, None)
                        if current_reference != (
                            candidate.ready_generation_path,
                            candidate.ready_fingerprint,
                        ):
                            raise _ReadyReferenceChanged
                        object_path = raw_path.decode("utf-8").rstrip("\n")
                        _validate_object_path(object_path)
                        await asyncio.to_thread(
                            store.delete_object, profile, object_path
                        )
                finally:
                    objects.close()
                if await self._finish_generation_gc(
                    candidate,
                    ModelPreheatInventoryGenerationStateEnum.DELETED,
                    None,
                    deleted_at=now,
                    **ownership,
                ):
                    deleted += 1
                else:
                    failed += 1
            except _ReadyReferenceChanged:
                skipped += 1
                await self._finish_generation_gc(
                    candidate,
                    ModelPreheatInventoryGenerationStateEnum.ORPHAN,
                    None,
                    **ownership,
                )
                continue
            except _PublishingGenerationProtected:
                skipped += 1
                await self._finish_generation_gc(
                    candidate,
                    ModelPreheatInventoryGenerationStateEnum.ORPHAN,
                    None,
                    **ownership,
                )
                continue
            except Exception:
                failed += 1
                await self._finish_generation_gc(
                    candidate,
                    ModelPreheatInventoryGenerationStateEnum.ERROR,
                    "inventory_gc_failed",
                    **ownership,
                )
                continue
            finally:
                await self.release_selection_lock(
                    profile_id, candidate.selection_key, owner_token
                )
        return InventoryGCResult(deleted=deleted, skipped=skipped, failed=failed)

    async def _spool_generation_objects(
        self,
        store,
        profile,
        generation_path,
        profile_id,
        selection_key,
        lock_owner,
        job_id,
        claim_token,
    ):
        spool = tempfile.TemporaryFile(mode="w+b")
        try:
            count = 0
            iterator = iter(store.iter_generation_objects(profile, generation_path))
            while True:
                has_object, object_path = await asyncio.to_thread(
                    _next_or_none, iterator
                )
                if not has_object:
                    break
                if job_id is not None and not await self._renew_claim(
                    job_id, claim_token
                ):
                    raise InventoryS3Error("inventory_claim_lost")
                if not await self.renew_selection_lock(
                    profile_id, selection_key, lock_owner
                ):
                    raise InventoryS3Error("inventory_selection_lost")
                _validate_object_path(object_path)
                count += 1
                if count > self._max_gc_objects:
                    raise InventoryS3Error("inventory_gc_object_limit")
                spool.write(object_path.encode("utf-8") + b"\n")
            spool.seek(0)
            return spool
        except Exception:
            spool.close()
            raise

    async def _claim_generation_for_gc(
        self,
        candidate,
        *,
        profile_id,
        lock_owner,
        job_id=None,
        claim_token=None,
    ):
        async with AsyncSession(self._engine) as session:
            ownership = _gc_ownership_predicates(
                profile_id,
                candidate.selection_key,
                lock_owner,
                job_id,
                claim_token,
            )
            result = await session.exec(
                update(ModelPreheatInventoryGeneration)
                .where(
                    ModelPreheatInventoryGeneration.id == candidate.id,
                    ModelPreheatInventoryGeneration.state.in_(
                        (
                            ModelPreheatInventoryGenerationStateEnum.ORPHAN,
                            ModelPreheatInventoryGenerationStateEnum.ERROR,
                        )
                    ),
                    ModelPreheatInventoryGeneration.ready_fingerprint
                    == candidate.ready_fingerprint,
                    ModelPreheatInventoryGeneration.ready_generation_path
                    == candidate.ready_generation_path,
                    ModelPreheatInventoryGeneration.orphaned_at
                    == candidate.orphaned_at,
                    *ownership,
                )
                .values(
                    state=ModelPreheatInventoryGenerationStateEnum.DELETING,
                    error_code=None,
                )
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            return result.rowcount == 1

    async def _finish_generation_gc(
        self,
        candidate,
        state,
        error_code,
        deleted_at=None,
        *,
        profile_id,
        lock_owner,
        job_id=None,
        claim_token=None,
    ):
        async with AsyncSession(self._engine) as session:
            ownership = _gc_ownership_predicates(
                profile_id,
                candidate.selection_key,
                lock_owner,
                job_id,
                claim_token,
            )
            result = await session.exec(
                update(ModelPreheatInventoryGeneration)
                .where(
                    ModelPreheatInventoryGeneration.id == candidate.id,
                    ModelPreheatInventoryGeneration.state
                    == ModelPreheatInventoryGenerationStateEnum.DELETING,
                    ModelPreheatInventoryGeneration.ready_fingerprint
                    == candidate.ready_fingerprint,
                    ModelPreheatInventoryGeneration.ready_generation_path
                    == candidate.ready_generation_path,
                    ModelPreheatInventoryGeneration.orphaned_at
                    == candidate.orphaned_at,
                    *ownership,
                )
                .values(
                    state=state,
                    error_code=error_code,
                    deleted_at_s3=deleted_at,
                )
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            return result.rowcount == 1


class MinioInventoryStore:
    def __init__(self, profile, config):
        if config is None:
            raise InventoryS3Error("inventory_client_unavailable")
        cipher = ModelPreheatCredentialCipher(
            current_key=getattr(config, "model_preheat_credential_key", None),
            current_key_version=getattr(
                config, "model_preheat_credential_key_version", None
            ),
            old_keys=getattr(config, "model_preheat_credential_old_keys", None),
        )
        access_key = cipher.decrypt(profile.access_key_encrypted)
        secret_key = cipher.decrypt(profile.secret_key_encrypted)
        self.client = ModelPreheatS3Client.from_minio(
            profile.endpoint,
            access_key,
            secret_key,
            secure=profile.tls_enabled,
            tls_verify=profile.tls_verify,
            region=profile.region,
            use_virtual_hosted_style=profile.use_virtual_hosted_style,
        )

    def scan(self, profile) -> InventoryScan:
        root = self.client._join_object_name(
            self.client._encoded_prefix(profile.prefix), "model-cache", "v1"
        )
        objects = list(
            self.client.iter_objects(
                profile.bucket, root, max_objects=MAX_INVENTORY_OBJECTS
            )
        )
        for path in objects:
            _validate_object_path(path)
        ready_paths = [path for path in objects if path.endswith("/ready.json")]
        records = []
        ready_references = {}
        ready_cache_keys = {}
        for ready_path in ready_paths:
            record, reference = self._inspect_ready(profile, ready_path)
            records.append(record)
            ready_references[ready_path] = reference
            ready_cache_keys[ready_path] = record.cache_key

        generations = {}
        for object_path in objects:
            marker = "/generations/"
            if marker not in object_path:
                continue
            before, after = object_path.split(marker, 1)
            generation_id = after.split("/", 1)[0]
            generation_path = f"{before}{marker}{generation_id}"
            ready_path = f"{before}/ready.json"
            reference = ready_references.get(ready_path, (None, None))
            generations[generation_path] = ScannedGeneration(
                generation_path=generation_path,
                ready_path=ready_path,
                ready_fingerprint=reference[1],
                referenced=reference[0] == generation_path,
                ready_generation_path=reference[0],
                selection_key=ready_cache_keys.get(ready_path)
                or "invalid-" + _path_key(ready_path),
                cache_key=ready_cache_keys.get(ready_path),
            )
        return InventoryScan(tuple(records), tuple(generations.values()))

    def _inspect_ready(self, profile, ready_path):
        try:
            payload = self.client._read_object_bytes(
                profile.bucket, ready_path, max_bytes=MAX_READY_BYTES
            )
            if payload is None:
                raise ModelPreheatS3ManifestError("s3_manifest_invalid")
            fingerprint = hashlib.sha256(payload).hexdigest()
            ready = json.loads(payload.decode("utf-8"))
            manifest_path = ready.get("manifest_object")
            if not isinstance(manifest_path, str):
                raise ModelPreheatS3ManifestError("s3_manifest_invalid")
            _validate_object_path(manifest_path)
            manifest_payload = self.client._read_object_bytes(
                profile.bucket, manifest_path, max_bytes=MAX_MANIFEST_BYTES
            )
            if manifest_payload is None:
                raise ModelPreheatS3ManifestError("s3_manifest_invalid")
            manifest = self.client._manifest_from_payload(
                json.loads(manifest_payload.decode("utf-8"))
            )
            strict = self.client.read_ready_manifest(
                profile.bucket,
                profile.prefix,
                manifest.identity,
                cache_key=manifest.cache_key,
                selection_digest=manifest.selection_digest,
            )
            if (
                strict is None
                or self.client.ready_object(profile.prefix, strict) != ready_path
            ):
                raise ModelPreheatS3ManifestError("s3_manifest_invalid")
            return _manifest_record(strict, ready_path, manifest_path), (
                self.client.generation_prefix(profile.prefix, strict),
                fingerprint,
            )
        except (
            InventoryS3Error,
            ModelPreheatS3ManifestError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            manifest = locals().get("manifest")
            manifest_path = locals().get("manifest_path")
            if manifest is not None and isinstance(manifest_path, str):
                return replace(
                    _manifest_record(manifest, ready_path, manifest_path),
                    manifest_state="invalid",
                ), (
                    self.client.generation_prefix(profile.prefix, manifest),
                    hashlib.sha256(locals().get("payload") or b"").hexdigest(),
                )
            cache_key = _safe_identifier_from_ready(locals().get("ready"), ready_path)
            generation_id = _safe_ready_value(locals().get("ready"), "generation_id")
            manifest_path = _safe_ready_path(locals().get("ready"), "manifest_object")
            fingerprint = hashlib.sha256(locals().get("payload") or b"").hexdigest()
            record = InventoryRecord(
                cache_key=cache_key,
                source="huggingface",
                model_id="invalid",
                resolved_revision="invalid",
                include_patterns=(),
                exclude_patterns=(),
                generation_id=generation_id or "invalid",
                ready_path=ready_path,
                manifest_path=manifest_path or ready_path,
                manifest_digest="0" * 64,
                file_count=0,
                total_size=0,
                manifest_state="invalid",
            )
            generation_path = None
            if manifest_path and "/.gpustack-manifest.json" in manifest_path:
                generation_path = manifest_path.rsplit("/", 1)[0]
            return record, (generation_path, fingerprint)

    def read_ready_reference(self, profile, ready_path):
        payload = self.client._read_object_bytes(
            profile.bucket, ready_path, max_bytes=MAX_READY_BYTES
        )
        if payload is None:
            return None, None
        _, reference = self._inspect_ready(profile, ready_path)
        return reference

    def iter_generation_objects(self, profile, generation_path):
        yield from self.client.iter_objects(
            profile.bucket,
            generation_path + "/",
            max_objects=MAX_GC_GENERATION_OBJECTS + 1,
        )

    def delete_object(self, profile, object_path):
        self.client.remove_object(profile.bucket, object_path)


async def upsert_verified_publication(
    session,
    task: ModelPreheatTask,
    ready,
    *,
    expected_attempt: int,
    expected_profile_version: int,
    lock_owner: str | None = None,
):
    if task.id is None:
        return False
    current = await session.get(ModelPreheatTask, task.id, populate_existing=True)
    if (
        current is None
        or current.attempt != expected_attempt
        or current.s3_profile_config_version != expected_profile_version
        or current.cache_key != task.cache_key
        or current.selection_digest != task.selection_digest
    ):
        return False
    profile = await session.get(ModelPreheatS3Profile, current.s3_profile_id)
    if profile is None or profile.config_version != expected_profile_version:
        return False
    if (
        ready.cache_key != current.cache_key
        or ready.selection_digest != current.selection_digest
        or ready.profile_config_version != expected_profile_version
        or not _is_sha256(ready.manifest_digest)
        or not _is_preheat_generation_id(ready.generation_id)
        or not isinstance(ready.file_count, int)
        or ready.file_count < 1
        or ready.file_count > 1024
        or not isinstance(ready.total_size, int)
        or ready.total_size < 0
        or ready.total_size > 1 << 50
    ):
        return False
    identity = ModelPreheatIdentity(
        source=current.source,
        model_id=current.model_id,
        revision=current.resolved_revision,
        file_patterns=current.include_patterns,
    )
    client = ModelPreheatS3Client(None)
    selection_prefix = client._selection_prefix(
        profile.prefix, identity, current.selection_digest
    )
    generation_id = ready.generation_id
    generation_prefix = client._join_object_name(
        selection_prefix, "generations", generation_id
    )
    expected_ready_path = client._join_object_name(selection_prefix, "ready.json")
    expected_manifest_path = client._join_object_name(
        generation_prefix, ".gpustack-manifest.json"
    )
    if (
        ready.ready_path != expected_ready_path
        or ready.manifest_path != expected_manifest_path
    ):
        return False
    if lock_owner is not None:
        lock = (
            await session.exec(
                select(ModelPreheatInventorySelectionLock).where(
                    ModelPreheatInventorySelectionLock.profile_id
                    == current.s3_profile_id,
                    ModelPreheatInventorySelectionLock.selection_key
                    == current.cache_key,
                    ModelPreheatInventorySelectionLock.owner_token == lock_owner,
                    ModelPreheatInventorySelectionLock.lease_expires_at > _utcnow(),
                )
            )
        ).first()
        if lock is None:
            return False
    values = {
        "profile_config_version": expected_profile_version,
        "source": current.source,
        "model_id": current.model_id,
        "resolved_revision": current.resolved_revision,
        "include_patterns": list(current.include_patterns),
        "exclude_patterns": list(current.exclude_patterns),
        "generation_id": generation_id,
        "ready_path": expected_ready_path,
        "manifest_path": expected_manifest_path,
        "manifest_digest": ready.manifest_digest,
        "file_count": ready.file_count,
        "total_size": ready.total_size,
        "manifest_state": ModelPreheatInventoryManifestStateEnum.VALID,
        "last_verified_at": _utcnow(),
        "created_by_task_id": current.id,
        "source_parent_attempt": expected_attempt,
    }
    update_result = await session.exec(
        update(ModelPreheatCachedModel)
        .where(
            ModelPreheatCachedModel.profile_id == current.s3_profile_id,
            ModelPreheatCachedModel.cache_key == current.cache_key,
            _publication_not_newer(
                current.id, expected_attempt, expected_profile_version
            ),
        )
        .values(**values, revision=ModelPreheatCachedModel.revision + 1)
        .execution_options(synchronize_session=False)
    )
    if update_result.rowcount == 1:
        return True
    existing = (
        await session.exec(
            select(ModelPreheatCachedModel).where(
                ModelPreheatCachedModel.profile_id == current.s3_profile_id,
                ModelPreheatCachedModel.cache_key == current.cache_key,
            )
        )
    ).first()
    if existing is not None:
        return False
    try:
        async with session.begin_nested():
            session.add(
                ModelPreheatCachedModel(
                    profile_id=current.s3_profile_id,
                    cache_key=current.cache_key,
                    revision=1,
                    **values,
                )
            )
            await session.flush()
        return True
    except IntegrityError:
        retry = await session.exec(
            update(ModelPreheatCachedModel)
            .where(
                ModelPreheatCachedModel.profile_id == current.s3_profile_id,
                ModelPreheatCachedModel.cache_key == current.cache_key,
                _publication_not_newer(
                    current.id, expected_attempt, expected_profile_version
                ),
            )
            .values(**values, revision=ModelPreheatCachedModel.revision + 1)
            .execution_options(synchronize_session=False)
        )
        return retry.rowcount == 1


def _publication_not_newer(task_id, attempt, profile_version):
    same_version = and_(
        ModelPreheatCachedModel.profile_config_version == profile_version,
        or_(
            ModelPreheatCachedModel.created_by_task_id.is_(None),
            ModelPreheatCachedModel.created_by_task_id < task_id,
            and_(
                ModelPreheatCachedModel.created_by_task_id == task_id,
                ModelPreheatCachedModel.source_parent_attempt <= attempt,
            ),
        ),
    )
    return or_(
        ModelPreheatCachedModel.profile_config_version < profile_version,
        same_version,
    )


def _gc_ownership_predicates(
    profile_id, selection_key, lock_owner, job_id, claim_token
):
    predicates = [
        exists(
            select(ModelPreheatInventorySelectionLock.id).where(
                ModelPreheatInventorySelectionLock.profile_id == profile_id,
                ModelPreheatInventorySelectionLock.selection_key == selection_key,
                ModelPreheatInventorySelectionLock.owner_token == lock_owner,
                ModelPreheatInventorySelectionLock.lease_expires_at > _utcnow(),
            )
        )
    ]
    if job_id is not None:
        predicates.append(
            exists(
                select(ModelPreheatInventoryJob.id).where(
                    ModelPreheatInventoryJob.id == job_id,
                    ModelPreheatInventoryJob.state
                    == ModelPreheatInventoryJobStateEnum.RUNNING,
                    ModelPreheatInventoryJob.claim_token == claim_token,
                    ModelPreheatInventoryJob.lease_expires_at > _utcnow(),
                )
            )
        )
    return predicates


def _profile_version_is_current(profile_id, profile_version):
    any_profile = exists(
        select(ModelPreheatS3Profile.id).where(ModelPreheatS3Profile.id == profile_id)
    )
    matching_profile = exists(
        select(ModelPreheatS3Profile.id).where(
            ModelPreheatS3Profile.id == profile_id,
            ModelPreheatS3Profile.config_version == profile_version,
        )
    )
    return or_(~any_profile, matching_profile)


def _manifest_record(manifest, ready_path, manifest_path):
    return InventoryRecord(
        cache_key=manifest.cache_key,
        source=manifest.identity.source,
        model_id=manifest.identity.model_id,
        resolved_revision=manifest.identity.revision,
        include_patterns=tuple(manifest.identity.file_patterns),
        exclude_patterns=tuple(manifest.exclude_patterns),
        generation_id=manifest.generation_id,
        ready_path=ready_path,
        manifest_path=manifest_path,
        manifest_digest=manifest.digest,
        file_count=len(manifest.files),
        total_size=manifest.total_size,
        manifest_state="valid",
    )


def _record_values(record, now, profile_config_version):
    values = {
        "profile_config_version": profile_config_version,
        "cache_key": record.cache_key,
        "source": record.source,
        "model_id": record.model_id,
        "resolved_revision": record.resolved_revision,
        "include_patterns": list(record.include_patterns),
        "exclude_patterns": list(record.exclude_patterns),
        "generation_id": record.generation_id,
        "ready_path": record.ready_path,
        "manifest_path": record.manifest_path,
        "manifest_digest": record.manifest_digest,
        "file_count": record.file_count,
        "total_size": record.total_size,
        "manifest_state": ModelPreheatInventoryManifestStateEnum(record.manifest_state),
        "last_verified_at": now,
    }
    if record.created_by_task_id is not None:
        values["created_by_task_id"] = record.created_by_task_id
    return values


def _validate_record(record):
    if len(record.cache_key) > 256 or not record.cache_key:
        raise InventoryS3Error("inventory_record_invalid")
    _validate_object_path(record.ready_path)
    _validate_object_path(record.manifest_path)
    if (
        record.source not in {"huggingface", "modelscope"}
        or not _is_sha256(record.manifest_digest)
        or record.file_count < 0
        or record.file_count > 1024
        or record.total_size < 0
        or record.total_size > 1 << 50
        or len(record.include_patterns) > 128
        or len(record.exclude_patterns) > 128
        or any(len(pattern) > 1024 for pattern in record.include_patterns)
        or any(len(pattern) > 1024 for pattern in record.exclude_patterns)
    ):
        raise InventoryS3Error("inventory_record_invalid")


def _validate_generation(generation):
    _validate_object_path(generation.generation_path)
    _validate_object_path(generation.ready_path)
    if (
        generation.ready_fingerprint is not None
        and len(generation.ready_fingerprint) != 64
    ):
        raise InventoryS3Error("inventory_generation_invalid")
    if generation.ready_generation_path is not None:
        _validate_object_path(generation.ready_generation_path)


def _validate_object_path(path):
    if (
        not isinstance(path, str)
        or not path
        or len(path) > MAX_INVENTORY_OBJECT_PATH
        or path.startswith("/")
        or "\\" in path
        or any(part in ("", ".", "..") for part in path.split("/"))
        or any(ord(char) < 32 or ord(char) == 127 for char in path)
    ):
        raise InventoryS3Error("inventory_path_invalid")


def _safe_identifier_from_ready(ready, ready_path):
    value = _safe_ready_value(ready, "cache_key")
    return value or "invalid-" + hashlib.sha256(ready_path.encode()).hexdigest()


def _safe_ready_value(ready, field):
    value = ready.get(field) if isinstance(ready, dict) else None
    if (
        isinstance(value, str)
        and 0 < len(value) <= 256
        and "/" not in value
        and "\\" not in value
        and not any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        return value
    return None


def _safe_ready_path(ready, field):
    value = ready.get(field) if isinstance(ready, dict) else None
    try:
        _validate_object_path(value)
    except Exception:
        return None
    return value


def _utcnow():
    return datetime.now(timezone.utc)


def _path_key(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def _next_or_none(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


def _chunks(values, size):
    for index in range(0, len(values), size):
        yield values[index : index + size]


class _ReadyReferenceChanged(Exception):
    pass


class _PublishingGenerationProtected(Exception):
    pass


def _task_matches_publication_marker(task, snapshot):
    return (
        task is not None
        and task.attempt == snapshot["attempt"]
        and task.s3_profile_id == snapshot["profile_id"]
        and task.s3_profile_config_version == snapshot["profile_version"]
        and task.cache_key == snapshot["selection_key"]
        and task.generation_id == snapshot["generation_id"]
        and task.desired_state == ModelPreheatDesiredStateEnum.RUNNING
        and task.execution_state
        not in {
            ModelPreheatExecutionStateEnum.READY,
            ModelPreheatExecutionStateEnum.PARTIAL,
            ModelPreheatExecutionStateEnum.ERROR,
            ModelPreheatExecutionStateEnum.CANCELED,
        }
    )


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_safe_identifier(value):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or "/" in value
        or "\\" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise InventoryS3Error("inventory_identifier_invalid")


def _is_preheat_generation_id(value):
    if not isinstance(value, str) or not value.startswith("preheat-"):
        return False
    try:
        parsed = uuid.UUID(value.removeprefix("preheat-"))
    except ValueError:
        return False
    return str(parsed) == value.removeprefix("preheat-")
