import asyncio
import hashlib
import itertools
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func, inspect, or_, update
from sqlmodel import delete, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialError,
    ModelPreheatCredentialCipher,
)
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatInventoryJobStateEnum,
    ModelPreheatInventoryManifestStateEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
    ModelPreheatDistributionPolicyArtifact,
)
from gpustack.schemas.model_storage_sync import ModelStorageSyncTask
from gpustack.worker.model_preheat.identity import decode_path
from gpustack.worker.model_preheat.s3_client import (
    ModelPreheatS3Client,
    ModelPreheatS3ManifestConflict,
    ModelPreheatS3ManifestError,
)


MAX_OBJECTS_PER_REFRESH = 100_000
REFRESH_LEASE_SECONDS = 300


class InventoryRefreshError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ArtifactRefreshJob:
    id: int
    profile_id: int
    profile_config_version: int
    kind: str
    state: ModelPreheatInventoryJobStateEnum
    scanned_count: int
    valid_count: int
    invalid_count: int
    orphan_count: int
    deleted_count: int
    skipped_count: int
    failed_count: int
    error_code: str | None
    started_at: datetime
    finished_at: datetime
    created_at: datetime
    updated_at: datetime


class ModelPreheatS3Inventory:
    """统一 Artifact 库存刷新服务。

    只扫描合法 ``<artifact_id>/manifest.json``。无 Manifest 的半成品、旧
    ready.json/generation 目录及非法 Manifest 均不进入数据库库存；首版不做
    后台 GC，也不删除 S3 对象。
    """

    _job_ids = itertools.count(1)

    def __init__(self, engine, *, config=None, poll_interval=30, **kwargs):
        self._engine = engine
        self._config = config
        self._poll_interval = poll_interval

    async def start(self):
        next_refresh_at: dict[tuple[int, int], datetime] = {}
        while True:
            async with AsyncSession(self._engine, expire_on_commit=False) as session:
                await self._refresh_due_profiles(session, next_refresh_at)
            await asyncio.sleep(self._poll_interval)

    async def _refresh_due_profiles(self, session, next_refresh_at, now=None):
        profiles = (
            await session.exec(
                select(ModelPreheatS3Profile).where(
                    ModelPreheatS3Profile.inventory_refresh_interval_seconds.is_not(
                        None
                    )
                )
            )
        ).all()
        now = now or _utcnow()
        active_keys = set()
        for profile in profiles:
            interval = profile.inventory_refresh_interval_seconds
            if interval is None:
                continue
            key = (profile.id, profile.config_version)
            active_keys.add(key)
            due_at = next_refresh_at.setdefault(
                key, now + timedelta(seconds=interval)
            )
            if now < due_at:
                continue
            next_refresh_at[key] = now + timedelta(seconds=interval)
            try:
                await self.create_refresh_job(session, profile.id, profile.config_version)
            except Exception:
                # 单个 Profile 的凭据、网络或租约失败不影响其他 Profile。
                pass
        for key in set(next_refresh_at) - active_keys:
            del next_refresh_at[key]

    async def create_refresh_job(self, session, profile_id: int, config_version: int):
        started = _utcnow()
        counts = await self.refresh_profile(session, profile_id, config_version)
        finished = _utcnow()
        return ArtifactRefreshJob(
            id=next(self._job_ids),
            profile_id=profile_id,
            profile_config_version=config_version,
            kind="refresh",
            state=ModelPreheatInventoryJobStateEnum.READY,
            scanned_count=counts["scanned"],
            valid_count=counts["valid"],
            invalid_count=counts["invalid"],
            orphan_count=0,
            deleted_count=counts["deleted"],
            skipped_count=0,
            failed_count=0,
            error_code=None,
            started_at=started,
            finished_at=finished,
            created_at=started,
            updated_at=finished,
        )

    async def create_gc_job(self, session, profile_id: int, config_version: int):
        raise ValueError("artifact_gc_not_supported")

    async def refresh_profile(self, session, profile_id: int, config_version: int):
        attempt_at = _utcnow()
        claim_token = uuid.uuid4().hex
        claimed = await self._claim_refresh(
            session, profile_id, config_version, attempt_at, claim_token
        )
        if not claimed:
            raise InventoryRefreshError("inventory_refresh_in_progress")
        profile = await session.get(
            ModelPreheatS3Profile, profile_id, populate_existing=True
        )
        if profile is None or profile.config_version != config_version:
            raise InventoryRefreshError("stale_profile_config")
        try:
            snapshot = _profile_snapshot(self._cipher(), profile)
            records, scanned, invalid, invalid_paths = await asyncio.to_thread(
                _scan_profile, snapshot
            )
        except Exception as exc:
            error_code = _inventory_error_code(exc)
            await self._finish_failed_refresh(
                session, profile_id, config_version, attempt_at, error_code, claim_token
            )
            raise InventoryRefreshError(error_code) from None

        now = _utcnow()
        return await self._commit_refresh(
            session,
            profile,
            config_version,
            attempt_at,
            now,
            records,
            scanned,
            invalid,
            invalid_paths,
            claim_token,
        )

    async def _claim_refresh(self, session, profile_id, config_version, now, claim_token):
        claimed = await session.exec(
            update(ModelPreheatS3Profile)
            .where(
                ModelPreheatS3Profile.id == profile_id,
                ModelPreheatS3Profile.config_version == config_version,
                or_(
                    ModelPreheatS3Profile.inventory_refresh_owner.is_(None),
                    ModelPreheatS3Profile.inventory_refresh_lease_expires_at.is_(None),
                    ModelPreheatS3Profile.inventory_refresh_lease_expires_at <= now,
                    ModelPreheatS3Profile.inventory_refresh_config_version
                    != config_version,
                ),
            )
            .values(
                inventory_refresh_owner=claim_token,
                inventory_refresh_config_version=config_version,
                inventory_refresh_lease_expires_at=now
                + timedelta(seconds=REFRESH_LEASE_SECONDS),
            )
        )
        await session.commit()
        return claimed.rowcount == 1

    async def _finish_failed_refresh(
        self, session, profile_id, config_version, attempt_at, error_code, claim_token
    ):
        now = _utcnow()
        await session.exec(
            update(ModelPreheatS3Profile)
            .where(*self._lease_conditions(profile_id, config_version, claim_token, now))
            .values(
                inventory_last_attempt_at=attempt_at,
                inventory_last_error_code=error_code,
                inventory_refresh_owner=None,
                inventory_refresh_config_version=None,
                inventory_refresh_lease_expires_at=None,
            )
        )
        await session.commit()

    def _lease_conditions(self, profile_id, config_version, claim_token, now):
        return (
            ModelPreheatS3Profile.id == profile_id,
            ModelPreheatS3Profile.config_version == config_version,
            ModelPreheatS3Profile.inventory_refresh_owner == claim_token,
            ModelPreheatS3Profile.inventory_refresh_config_version == config_version,
            ModelPreheatS3Profile.inventory_refresh_lease_expires_at > now,
        )

    async def _commit_refresh(
        self,
        session,
        profile,
        config_version,
        attempt_at,
        now,
        records,
        scanned,
        invalid,
        invalid_paths,
        claim_token,
    ):
        profile_id = profile.id
        await session.exec(
            update(ModelPreheatArtifact)
            .where(
                ModelPreheatArtifact.profile_id == profile_id,
                ModelPreheatArtifact.profile_config_version != config_version,
            )
            .values(manifest_state=ModelPreheatInventoryManifestStateEnum.STALE)
            .execution_options(synchronize_session=False)
        )
        seen = {record["artifact_id"] for record in records}
        current = (
            await session.exec(
                select(ModelPreheatArtifact).where(
                    ModelPreheatArtifact.profile_id == profile_id,
                    ModelPreheatArtifact.profile_config_version == config_version,
                )
            )
        ).all()
        missing_rows = [row for row in current if row.artifact_id not in seen]
        referenced_ids = await _referenced_artifact_ids(
            session, profile_id, config_version, missing_rows
        )
        deleted_ids = []
        for artifact in missing_rows:
            if artifact.manifest_path in invalid_paths:
                artifact.manifest_state = ModelPreheatInventoryManifestStateEnum.INVALID
                session.add(artifact)
            elif artifact.created_by_task_id is not None or artifact.id in referenced_ids:
                artifact.manifest_state = ModelPreheatInventoryManifestStateEnum.STALE
                session.add(artifact)
            else:
                deleted_ids.append(artifact.id)
        if deleted_ids:
            await session.exec(
                delete(ModelPreheatArtifact).where(ModelPreheatArtifact.id.in_(deleted_ids))
            )
        by_artifact = {row.artifact_id: row for row in current}
        for values in records:
            artifact = by_artifact.get(values["artifact_id"])
            if artifact is None:
                artifact = ModelPreheatArtifact(
                    profile_id=profile_id,
                    profile_config_version=config_version,
                    **values,
                    last_verified_at=now,
                )
            else:
                for field, value in values.items():
                    setattr(artifact, field, value)
                artifact.last_verified_at = now
            session.add(artifact)
        final = await session.exec(
            update(ModelPreheatS3Profile)
            .where(*self._lease_conditions(profile_id, config_version, claim_token, now))
            .values(
                inventory_last_attempt_at=attempt_at,
                inventory_last_success_at=now,
                inventory_last_scan_count=len(
                    {(record["source"], record["model_id"]) for record in records}
                ),
                inventory_last_error_code=None,
                ever_used_at=(
                    func.coalesce(ModelPreheatS3Profile.ever_used_at, now)
                    if records
                    else ModelPreheatS3Profile.ever_used_at
                ),
                inventory_refresh_owner=None,
                inventory_refresh_config_version=None,
                inventory_refresh_lease_expires_at=None,
            )
        )
        if final.rowcount != 1:
            await session.rollback()
            raise InventoryRefreshError("stale_profile_config")
        await session.commit()
        return {
            "scanned": scanned,
            "valid": len(records),
            "invalid": invalid,
            "deleted": len(deleted_ids),
        }

    def _cipher(self):
        config = self._config
        return ModelPreheatCredentialCipher(
            getattr(config, "model_preheat_credential_key", None),
            getattr(config, "model_preheat_credential_key_version", None),
            getattr(config, "model_preheat_credential_old_keys", None),
        )


def _profile_snapshot(cipher, profile):
    return {
        "endpoint": profile.endpoint,
        "bucket": profile.bucket,
        "prefix": profile.prefix,
        "tls_enabled": profile.tls_enabled,
        "tls_verify": profile.tls_verify,
        "region": profile.region,
        "use_virtual_hosted_style": profile.use_virtual_hosted_style,
        "access_key": cipher.decrypt(profile.access_key_encrypted),
        "secret_key": cipher.decrypt(profile.secret_key_encrypted),
    }


def _scan_profile(profile):
    client = ModelPreheatS3Client.from_minio(
        endpoint=profile["endpoint"],
        access_key=profile["access_key"],
        secret_key=profile["secret_key"],
        secure=profile["tls_enabled"],
        tls_verify=profile["tls_verify"],
        region=profile["region"] or None,
        use_virtual_hosted_style=profile["use_virtual_hosted_style"],
    )
    prefix = profile["prefix"].strip("/")
    list_prefix = f"{prefix}/" if prefix else ""
    listed = client._client.list_objects(
        profile["bucket"], prefix=list_prefix, recursive=True
    )
    records = []
    scanned = 0
    invalid = 0
    invalid_paths = set()
    for item in listed:
        path = getattr(item, "object_name", None)
        scanned += 1
        if scanned > MAX_OBJECTS_PER_REFRESH:
            raise ValueError("inventory_scan_limit_exceeded")
        if not isinstance(path, str) or not _path_within_prefix(path, list_prefix):
            continue
        if not path.endswith("/manifest.json"):
            continue
        try:
            manifest = client.read_artifact_manifest_path(profile["bucket"], path)
            if (
                manifest is None
                or client.artifact_manifest_object(prefix, manifest) != path
            ):
                invalid += 1
                invalid_paths.add(path)
                continue
            payload = manifest.to_artifact_json_bytes()
            records.append(
                {
                    "artifact_id": manifest.artifact_id,
                    "source": manifest.identity.source,
                    "model_id": manifest.identity.model_path,
                    "resolved_revision": decode_path(manifest.identity.revision_path),
                    "include_patterns": list(manifest.identity.file_patterns),
                    "exclude_patterns": list(manifest.exclude_patterns),
                    "manifest_path": path,
                    "manifest_digest": hashlib.sha256(payload).hexdigest(),
                    "file_count": len(manifest.files),
                    "total_size": manifest.total_size,
                    "manifest_state": ModelPreheatInventoryManifestStateEnum.VALID,
                }
            )
        except (
            ModelPreheatS3ManifestError,
            ModelPreheatS3ManifestConflict,
            ValueError,
            json.JSONDecodeError,
        ):
            invalid += 1
            invalid_paths.add(path)
    unique = {record["artifact_id"]: record for record in records}
    return list(unique.values()), scanned, invalid, invalid_paths


def _utcnow():
    return datetime.now(timezone.utc)


def _path_within_prefix(path: str, list_prefix: str) -> bool:
    return not list_prefix or path.startswith(list_prefix)


async def _referenced_artifact_ids(session, profile_id, config_version, artifacts):
    if not artifacts:
        return set()

    artifact_ids = [artifact.id for artifact in artifacts]
    artifact_by_name = {artifact.artifact_id: artifact.id for artifact in artifacts}
    connection = await session.connection()
    table_names = await connection.run_sync(
        lambda sync_connection: set(inspect(sync_connection).get_table_names())
    )
    referenced = set()
    if ModelPreheatDistributionPolicy.__tablename__ in table_names:
        referenced.update(
            await session.exec(
                select(ModelPreheatDistributionPolicy.source_artifact_id).where(
                    ModelPreheatDistributionPolicy.profile_id == profile_id,
                    ModelPreheatDistributionPolicy.profile_config_version
                    == config_version,
                    ModelPreheatDistributionPolicy.source_artifact_id.in_(artifact_ids),
                )
            )
        )
    if ModelPreheatDistributionPolicyArtifact.__tablename__ in table_names:
        referenced.update(
            await session.exec(
                select(ModelPreheatDistributionPolicyArtifact.artifact_id).where(
                    ModelPreheatDistributionPolicyArtifact.artifact_id.in_(artifact_ids)
                )
            )
        )
    if (
        ModelPreheatWorkerTask.__tablename__ in table_names
        and ModelPreheatDistributionPolicy.__tablename__ in table_names
    ):
        from gpustack.server.model_preheat_worker_reconciler import (
            MAX_DISTRIBUTION_ATTEMPTS,
            RETRYABLE_DISTRIBUTION_ERRORS,
        )

        task_artifact_ids = (
            await session.exec(
                select(ModelPreheatWorkerTask.distribution_artifact_id)
                .join(
                    ModelPreheatDistributionPolicy,
                    ModelPreheatDistributionPolicy.id
                    == ModelPreheatWorkerTask.distribution_policy_id,
                )
                .where(
                    ModelPreheatDistributionPolicy.profile_id == profile_id,
                    ModelPreheatDistributionPolicy.profile_config_version
                    == config_version,
                    ModelPreheatWorkerTask.distribution_artifact_id.in_(
                        artifact_by_name
                    ),
                    or_(
                        ModelPreheatWorkerTask.state.in_(
                            [
                                ModelPreheatWorkerTaskStateEnum.PENDING,
                                ModelPreheatWorkerTaskStateEnum.RUNNING,
                                ModelPreheatWorkerTaskStateEnum.PAUSED,
                            ]
                        ),
                        and_(
                            ModelPreheatWorkerTask.state
                            == ModelPreheatWorkerTaskStateEnum.ERROR,
                            ModelPreheatWorkerTask.error_code.in_(
                                RETRYABLE_DISTRIBUTION_ERRORS
                            ),
                            ModelPreheatWorkerTask.attempt
                            < MAX_DISTRIBUTION_ATTEMPTS,
                            ModelPreheatWorkerTask.finished_at.is_not(None),
                        ),
                    ),
                )
            )
        ).all()
        referenced.update(
            artifact_by_name[artifact_id] for artifact_id in task_artifact_ids
        )
    if ModelPreheatTask.__tablename__ in table_names:
        task_artifact_ids = (
            await session.exec(
                select(ModelPreheatTask.artifact_id).where(
                    ModelPreheatTask.s3_profile_id == profile_id,
                    ModelPreheatTask.s3_profile_config_version == config_version,
                    ModelPreheatTask.artifact_id.in_(artifact_by_name),
                )
            )
        ).all()
        referenced.update(artifact_by_name[artifact_id] for artifact_id in task_artifact_ids)
    if ModelStorageSyncTask.__tablename__ in table_names:
        task_artifact_ids = (
            await session.exec(
                select(ModelStorageSyncTask.artifact_id).where(
                    ModelStorageSyncTask.profile_id == profile_id,
                    ModelStorageSyncTask.profile_config_version == config_version,
                    ModelStorageSyncTask.artifact_id.in_(artifact_by_name),
                )
            )
        ).all()
        referenced.update(artifact_by_name[artifact_id] for artifact_id in task_artifact_ids)
    return referenced


def _inventory_error_code(exc: Exception) -> str:
    if isinstance(exc, ModelPreheatCredentialError):
        return "inventory_credential_unavailable"
    if isinstance(exc, ValueError) and str(exc) == "inventory_scan_limit_exceeded":
        return "inventory_scan_limit_exceeded"
    return "inventory_scan_failed"
