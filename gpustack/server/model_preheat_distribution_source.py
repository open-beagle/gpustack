from dataclasses import dataclass
from types import SimpleNamespace

from sqlmodel import select

from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatExecutionStateEnum,
    ModelPreheatInventoryManifestStateEnum,
    ModelPreheatTask,
)
from gpustack.schemas.model_storage_sync import (
    ModelStorageSyncTask,
    ModelStorageSyncTaskStateEnum,
)


class DistributionSourceUnavailable(Exception):
    pass


@dataclass
class DistributionSource:
    artifact: ModelPreheatArtifact
    profile: ModelPreheatS3Profile
    payload: dict
    encrypted_profile: dict
    attempt: int
    preheat_task: ModelPreheatTask | None = None


async def resolve_distribution_source(session, policy) -> DistributionSource:
    profile = await session.get(ModelPreheatS3Profile, policy.profile_id)
    if (
        profile is None
        or profile.lifecycle_state != ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE
        or profile.config_version != policy.profile_config_version
    ):
        raise DistributionSourceUnavailable("distribution_profile_not_active")

    referenced_artifact = (
        await session.get(ModelPreheatArtifact, policy.source_artifact_id)
        if policy.source_artifact_id is not None
        else None
    )
    if policy.source_artifact_id is not None and referenced_artifact is None:
        raise DistributionSourceUnavailable("distribution_artifact_not_ready")

    sync_task = await _ready_sync_task(session, policy)
    preheat_task = await _ready_preheat_task(session, policy)
    if sync_task is not None:
        artifact = await _artifact_for_identity(
            session,
            sync_task.profile_id,
            sync_task.profile_config_version,
            sync_task.artifact_id,
        )
        encrypted_profile = sync_task.credential_snapshot_encrypted
        attempt = 1
        request_identity = sync_task.request_identity
        trusted_preheat_task = None
    elif preheat_task is not None:
        artifact = await _artifact_for_identity(
            session,
            preheat_task.s3_profile_id,
            preheat_task.s3_profile_config_version,
            preheat_task.artifact_id,
        )
        if artifact is None and preheat_task.s3_manifest_path:
            artifact = _legacy_preheat_artifact(preheat_task)
        encrypted_profile = preheat_task.s3_profile_snapshot_encrypted
        attempt = preheat_task.attempt
        request_identity = preheat_task.request_identity
        trusted_preheat_task = preheat_task
    else:
        artifact = referenced_artifact
        encrypted_profile = None
        attempt = 1
        request_identity = policy.request_identity
        trusted_preheat_task = None

    if referenced_artifact is not None and not _same_artifact(
        referenced_artifact, artifact
    ):
        raise DistributionSourceUnavailable("distribution_source_identity_mismatch")
    if sync_task is not None and preheat_task is not None:
        preheat_artifact = await _artifact_for_identity(
            session,
            preheat_task.s3_profile_id,
            preheat_task.s3_profile_config_version,
            preheat_task.artifact_id,
        )
        if preheat_artifact is None and preheat_task.s3_manifest_path:
            preheat_artifact = _legacy_preheat_artifact(preheat_task)
        if (
            not _same_artifact(artifact, preheat_artifact)
            or sync_task.request_digest != preheat_task.request_digest
        ):
            raise DistributionSourceUnavailable("distribution_source_identity_mismatch")
    if (
        artifact is None
        or artifact.profile_id != policy.profile_id
        or artifact.profile_config_version != policy.profile_config_version
        or artifact.manifest_state != ModelPreheatInventoryManifestStateEnum.VALID
    ):
        raise DistributionSourceUnavailable("distribution_artifact_not_ready")
    if policy.request_digest != (
        sync_task.request_digest
        if sync_task is not None
        else (
            preheat_task.request_digest
            if preheat_task is not None
            else policy.request_digest
        )
    ):
        raise DistributionSourceUnavailable("distribution_source_identity_mismatch")
    if encrypted_profile is None:
        encrypted_profile = {
            "endpoint": profile.endpoint,
            "bucket": profile.bucket,
            "prefix": profile.prefix,
            "tls_enabled": profile.tls_enabled,
            "tls_verify": profile.tls_verify,
            "region": profile.region,
            "use_virtual_hosted_style": profile.use_virtual_hosted_style,
            "access_key_encrypted": profile.access_key_encrypted,
            "secret_key_encrypted": profile.secret_key_encrypted,
        }
    payload = {
        "id": policy.id,
        "source": artifact.source,
        "model_id": artifact.model_id,
        "requested_revision": request_identity.get("requested_revision"),
        "resolved_revision": artifact.resolved_revision,
        "include_patterns": list(artifact.include_patterns),
        "exclude_patterns": list(artifact.exclude_patterns),
        "request_digest": policy.request_digest,
        "artifact_id": artifact.artifact_id,
        "s3_manifest_path": artifact.manifest_path,
    }
    return DistributionSource(
        artifact=artifact,
        profile=profile,
        payload=payload,
        encrypted_profile=encrypted_profile,
        attempt=attempt,
        preheat_task=trusted_preheat_task,
    )


async def _ready_sync_task(session, policy):
    if policy.source_sync_task_id is None:
        return None
    task = await session.get(ModelStorageSyncTask, policy.source_sync_task_id)
    if (
        task is None
        or task.state != ModelStorageSyncTaskStateEnum.READY
        or not task.artifact_id
        or task.profile_id != policy.profile_id
        or task.profile_config_version != policy.profile_config_version
    ):
        raise DistributionSourceUnavailable("distribution_sync_task_not_ready")
    return task


async def _ready_preheat_task(session, policy):
    if policy.created_by_task_id is None:
        return None
    task = await session.get(ModelPreheatTask, policy.created_by_task_id)
    if (
        task is None
        or task.execution_state != ModelPreheatExecutionStateEnum.READY
        or not task.artifact_id
        or task.s3_profile_id != policy.profile_id
        or task.s3_profile_config_version != policy.profile_config_version
    ):
        raise DistributionSourceUnavailable("distribution_preheat_task_not_ready")
    return task


def _legacy_preheat_artifact(task):
    return SimpleNamespace(
        id=None,
        profile_id=task.s3_profile_id,
        profile_config_version=task.s3_profile_config_version,
        artifact_id=task.artifact_id,
        source=task.source,
        model_id=task.model_id,
        resolved_revision=task.resolved_revision,
        include_patterns=list(task.include_patterns),
        exclude_patterns=list(task.exclude_patterns),
        manifest_path=task.s3_manifest_path,
        manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
    )


def _same_artifact(left, right):
    if left is None or right is None:
        return False
    return (
        all(
            getattr(left, field) == getattr(right, field)
            for field in (
                "profile_id",
                "profile_config_version",
                "artifact_id",
                "source",
                "model_id",
                "resolved_revision",
            )
        )
        and list(left.include_patterns) == list(right.include_patterns)
        and list(left.exclude_patterns) == list(right.exclude_patterns)
    )


async def _artifact_for_identity(session, profile_id, version, artifact_id):
    return (
        await session.exec(
            select(ModelPreheatArtifact).where(
                ModelPreheatArtifact.profile_id == profile_id,
                ModelPreheatArtifact.profile_config_version == version,
                ModelPreheatArtifact.artifact_id == artifact_id,
            )
        )
    ).first()
