import asyncio
import hashlib
import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import update
from sqlmodel import delete, select

from gpustack.model_preheat_credentials import ModelPreheatCredentialCipher
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatInventoryJobStateEnum,
    ModelPreheatInventoryManifestStateEnum,
)
from gpustack.worker.model_preheat.identity import decode_path
from gpustack.worker.model_preheat.s3_client import (
    ModelPreheatS3Client,
    ModelPreheatS3ManifestError,
)


MAX_ARTIFACTS_PER_REFRESH = 100_000


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
        self._locks: dict[tuple[int, int], asyncio.Lock] = {}

    async def start(self):
        while True:
            await asyncio.sleep(self._poll_interval)

    async def create_refresh_job(self, session, profile_id: int, config_version: int):
        key = (profile_id, config_version)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
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
        profile = await session.get(ModelPreheatS3Profile, profile_id)
        if profile is None:
            raise ValueError("model_storage_profile_not_found")
        if profile.config_version != config_version:
            raise ValueError("stale_profile_config")
        snapshot = _profile_snapshot(self._cipher(), profile)
        records, scanned, invalid = await asyncio.to_thread(_scan_profile, snapshot)

        now = _utcnow()
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
        deleted_count = len([row for row in current if row.artifact_id not in seen])
        if seen:
            await session.exec(
                delete(ModelPreheatArtifact).where(
                    ModelPreheatArtifact.profile_id == profile_id,
                    ModelPreheatArtifact.profile_config_version == config_version,
                    ModelPreheatArtifact.artifact_id.not_in(seen),
                )
            )
        else:
            await session.exec(
                delete(ModelPreheatArtifact).where(
                    ModelPreheatArtifact.profile_id == profile_id,
                    ModelPreheatArtifact.profile_config_version == config_version,
                )
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
        if records:
            await session.exec(
                update(ModelPreheatS3Profile)
                .where(
                    ModelPreheatS3Profile.id == profile_id,
                    ModelPreheatS3Profile.ever_used_at.is_(None),
                )
                .values(ever_used_at=now)
            )
        await session.commit()
        return {
            "scanned": scanned,
            "valid": len(records),
            "invalid": invalid,
            "deleted": deleted_count,
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
    listed = client._client.list_objects(
        profile["bucket"], prefix=prefix, recursive=True
    )
    records = []
    scanned = 0
    invalid = 0
    for item in listed:
        path = getattr(item, "object_name", None)
        if not isinstance(path, str) or not path.endswith("/manifest.json"):
            continue
        scanned += 1
        if scanned > MAX_ARTIFACTS_PER_REFRESH:
            raise ValueError("inventory_scan_limit_exceeded")
        try:
            manifest = client.read_artifact_manifest_path(profile["bucket"], path)
            if (
                manifest is None
                or client.artifact_manifest_object(prefix, manifest) != path
            ):
                invalid += 1
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
        except (ModelPreheatS3ManifestError, ValueError, json.JSONDecodeError):
            invalid += 1
    unique = {record["artifact_id"]: record for record in records}
    return list(unique.values()), scanned, invalid


def _utcnow():
    return datetime.now(timezone.utc)
