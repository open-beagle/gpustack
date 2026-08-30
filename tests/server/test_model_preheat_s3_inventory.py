import asyncio
import hashlib
import re
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatInventoryManifestStateEnum,
    ModelPreheatTargetScopeEnum,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
    ModelPreheatDistributionPolicyArtifact,
    ModelPreheatDistributionPolicyCreate,
    ModelPreheatDistributionPolicyRun,
    ModelPreheatDistributionPolicyRunTask,
    ModelPreheatDistributionSelectionModeEnum,
)
from gpustack.schemas.model_storage_sync import (
    ModelStorageSyncExecutionPayload,
    ModelStorageSyncExecutionProfile,
    ModelStorageSyncTaskStateEnum,
)
from gpustack.schemas.users import User
from gpustack.model_preheat_credentials import ModelPreheatCredentialError
from gpustack.routes.model_preheat_distribution_policies import (
    create_distribution_policy,
)
from gpustack.server import model_preheat_s3_inventory as inventory_module
from gpustack.server.model_preheat_s3_inventory import (
    InventoryRefreshError,
    ModelPreheatS3Inventory,
)
from gpustack.worker.model_preheat.executor import (
    SeedExecutionRequest,
    TargetExecutionRequest,
    execute_seed_preheat,
    execute_target_preheat,
)
from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    decode_path,
    ollama_model_filename,
)
from gpustack.worker.model_preheat.manifest import build_model_preheat_manifest
from gpustack.worker.model_preheat.s3_client import ModelPreheatS3Client
from gpustack.worker.model_preheat.s3_client import ModelPreheatS3ManifestError
from gpustack.worker.model_storage_sync_manager import ModelStorageSyncManager
from tests.worker.model_preheat.test_seed_executor import InMemoryMinio


def _identity():
    return ModelPreheatIdentity(
        source="huggingface",
        model_id="org/model",
        revision="resolved-commit",
        file_patterns=("config.json",),
    )


def test_scan_only_accepts_canonical_artifact_manifests(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("model", encoding="utf-8")
    manifest = build_model_preheat_manifest(source, _identity())
    canonical_path = f"storage/{manifest.artifact_prefix('')}/manifest.json"

    class FakeClient:
        def __init__(self):
            self._client = self

        def list_objects(self, bucket, prefix, recursive):
            assert (bucket, prefix, recursive) == ("models", "storage/", True)
            return [
                SimpleNamespace(object_name=canonical_path),
                SimpleNamespace(object_name="storage/legacy/ready.json"),
                SimpleNamespace(object_name="storage/partial/files/config.json"),
                SimpleNamespace(object_name="storage/invalid/manifest.json"),
                SimpleNamespace(object_name="storage-other/foreign/manifest.json"),
            ]

        def read_artifact_manifest_path(self, bucket, path):
            if path == canonical_path:
                return manifest
            raise ModelPreheatS3ManifestError("invalid_manifest")

        def artifact_manifest_object(self, prefix, value):
            return f"{prefix}/{value.artifact_prefix('')}/manifest.json"

    fake = FakeClient()
    monkeypatch.setattr(
        inventory_module.ModelPreheatS3Client,
        "from_minio",
        lambda **kwargs: fake,
    )

    records, scanned, invalid, invalid_paths = inventory_module._scan_profile(
        {
            "endpoint": "https://s3.example.com",
            "bucket": "models",
            "prefix": "storage",
            "tls_enabled": True,
            "tls_verify": True,
            "region": "",
            "use_virtual_hosted_style": True,
            "access_key": "access",
            "secret_key": "secret",
        }
    )

    assert scanned == 5
    assert invalid == 1
    assert invalid_paths == {"storage/invalid/manifest.json"}
    assert [record["artifact_id"] for record in records] == [manifest.artifact_id]
    assert records[0]["resolved_revision"] == "resolved-commit"


def test_scan_stops_after_a_hard_object_enumeration_limit(monkeypatch):
    class FakeClient:
        def __init__(self):
            self._client = self

        def list_objects(self, bucket, prefix, recursive):
            return [
                SimpleNamespace(object_name=f"storage/file-{index}")
                for index in range(3)
            ]

    monkeypatch.setattr(
        inventory_module.ModelPreheatS3Client,
        "from_minio",
        lambda **kwargs: FakeClient(),
    )
    monkeypatch.setattr(inventory_module, "MAX_OBJECTS_PER_REFRESH", 2)

    try:
        inventory_module._scan_profile(
            {
                "endpoint": "https://s3.example.com",
                "bucket": "models",
                "prefix": "storage",
                "tls_enabled": True,
                "tls_verify": True,
                "region": "",
                "use_virtual_hosted_style": True,
                "access_key": "access",
                "secret_key": "secret",
            }
        )
    except ValueError as exc:
        assert str(exc) == "inventory_scan_limit_exceeded"
    else:
        raise AssertionError("枚举超过上限时必须中止扫描")


def test_refresh_rebuilds_current_version_and_marks_old_version_stale(
    monkeypatch, tmp_path
):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'inventory.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                ],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = ModelPreheatS3Profile(
                name="profile",
                endpoint="https://s3.example.com",
                bucket="models",
                prefix="storage",
                access_key_encrypted={"ciphertext": "access"},
                secret_key_encrypted={"ciphertext": "secret"},
                encryption_key_version="v1",
                config_version=2,
            )
            session.add(profile)
            await session.flush()
            session.add_all(
                [
                    _artifact(profile.id, 1, "a" * 64),
                    _artifact(profile.id, 2, "b" * 64),
                ]
            )
            profile_id = profile.id
            await session.commit()

        first_record = _artifact_values("c" * 64)
        first_record.pop("last_verified_at")
        second_record = _artifact_values("d" * 64)
        second_record.pop("last_verified_at")
        second_record["resolved_revision"] = "another-commit"
        monkeypatch.setattr(
            inventory_module,
            "_scan_profile",
            lambda snapshot: ([first_record, second_record], 2, 0, set()),
        )
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")
        async with AsyncSession(engine) as session:
            counts = await service.refresh_profile(session, profile_id, 2)
        async with AsyncSession(engine) as session:
            rows = (
                await session.exec(
                    select(ModelPreheatArtifact).order_by(
                        ModelPreheatArtifact.profile_config_version,
                        ModelPreheatArtifact.artifact_id,
                    )
                )
            ).all()
            refreshed_profile = await session.get(ModelPreheatS3Profile, profile_id)
            ever_used_at = refreshed_profile.ever_used_at
            inventory_last_scan_count = refreshed_profile.inventory_last_scan_count
        await engine.dispose()
        return counts, rows, ever_used_at, inventory_last_scan_count

    counts, rows, ever_used_at, inventory_last_scan_count = asyncio.run(run())

    assert counts == {"scanned": 2, "valid": 2, "invalid": 0, "deleted": 1}
    assert [(row.profile_config_version, row.artifact_id) for row in rows] == [
        (1, "a" * 64),
        (2, "c" * 64),
        (2, "d" * 64),
    ]
    assert rows[0].manifest_state == ModelPreheatInventoryManifestStateEnum.STALE
    assert rows[1].manifest_state == ModelPreheatInventoryManifestStateEnum.VALID
    assert ever_used_at is not None
    assert inventory_last_scan_count == 1


def test_refresh_discovers_three_sources_from_shared_s3_in_independent_database(
    monkeypatch, tmp_path
):
    async def create_profile(engine, name):
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                    ModelPreheatDistributionPolicy.__table__,
                    ModelPreheatDistributionPolicyArtifact.__table__,
                    ModelPreheatDistributionPolicyRun.__table__,
                    ModelPreheatDistributionPolicyRunTask.__table__,
                    ModelPreheatWorkerTask.__table__,
                ],
            )
        async with AsyncSession(engine) as session:
            profile = ModelPreheatS3Profile(
                name=name,
                endpoint="https://s3.example.com",
                bucket="models",
                prefix="shared",
                access_key_encrypted={"ciphertext": "access"},
                secret_key_encrypted={"ciphertext": "secret"},
                encryption_key_version="v1",
                config_version=1,
            )
            session.add(profile)
            await session.flush()
            profile_id = profile.id
            await session.commit()
            return profile_id

    shared_minio = InMemoryMinio()
    shared_s3 = ModelPreheatS3Client(shared_minio)

    monkeypatch.setattr(
        inventory_module.ModelPreheatS3Client,
        "from_minio",
        lambda **kwargs: shared_s3,
    )

    profile = ModelStorageSyncExecutionProfile(
        endpoint="https://s3.example.com",
        bucket="models",
        prefix="shared",
        access_key="stack-a-access",
        secret_key="stack-a-secret",
    )
    sync_root = tmp_path / "stack-a-modelscope"
    sync_root.mkdir()
    (sync_root / "model.bin").write_bytes(b"modelscope")
    sync_identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="modelscope/model",
        revision="revision-1",
        file_patterns=("model.bin",),
    )
    sync_result = ModelStorageSyncManager(
        worker_id=1, clientset=SimpleNamespace(), cfg=SimpleNamespace()
    )._publish(
        ModelStorageSyncExecutionPayload(
            task_id=1,
            state=ModelStorageSyncTaskStateEnum.PUBLISHING,
            source="modelscope",
            model_id="modelscope/model",
            resolved_revision="revision-1",
            request_identity={
                "source": "modelscope",
                "model_id": "modelscope/model",
                "requested_revision": None,
                "include_patterns": ["model.bin"],
                "exclude_patterns": [],
            },
            request_digest=sync_identity.request_digest,
            source_paths=[str(sync_root / "model.bin")],
            scan_spec={"root": str(sync_root), "include_patterns": ["model.bin"]},
            lease_token="stack-a-sync-lease",
            profile=profile,
        ),
        threading.Event(),
    )

    def seed(source, model_id, revision, task_id, content):
        identity = ModelPreheatIdentity(
            source=source,
            model_id=model_id,
            revision=revision,
            file_patterns=() if source == "ollama_library" else ("model.bin",),
        )
        filename = (
            ollama_model_filename(model_id)
            if source == "ollama_library"
            else "model.bin"
        )

        def download(_identity, staging, **_kwargs):
            (staging / filename).write_bytes(content)

        result = execute_seed_preheat(
            SeedExecutionRequest(
                cache_dir=tmp_path / f"stack-a-{source}-cache",
                target_dir=tmp_path / f"stack-a-{source}-target",
                task_id=task_id,
                attempt=1,
                request_digest=identity.request_digest,
                identity=identity,
                exclude_patterns=(),
                bucket="models",
                prefix="shared",
                source_fallback_enabled=True,
                install_local=False,
            ),
            shared_s3,
            download_to_staging=download,
        )
        assert result["state"] == "ready"
        return result

    huggingface_result = seed(
        "huggingface", "huggingface/model", "revision-1", 2, b"huggingface"
    )
    ollama_result = seed(
        "ollama_library",
        "ollama/model:latest",
        "ollama-pending",
        3,
        b"ollama",
    )
    assert re.fullmatch(
        r"local-snapshot-[0-9a-f]{64}", ollama_result["resolved_revision"]
    )

    async def run():
        stack_a = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'a.db'}")
        stack_b = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'b.db'}")
        stack_a_profile_id = await create_profile(stack_a, "stack-a")
        profile_id = await create_profile(stack_b, "stack-b")

        stack_a_service = ModelPreheatS3Inventory(stack_a)
        stack_b_service = ModelPreheatS3Inventory(stack_b)
        for service in (stack_a_service, stack_b_service):
            service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")
        async with AsyncSession(stack_a) as session:
            stack_a_counts = await stack_a_service.refresh_profile(
                session, stack_a_profile_id, 1
            )
        async with AsyncSession(stack_b, expire_on_commit=False) as session:
            counts = await stack_b_service.refresh_profile(session, profile_id, 1)
        async with AsyncSession(stack_b, expire_on_commit=False) as session:
            rows = (await session.exec(select(ModelPreheatArtifact))).all()
            profile = await session.get(ModelPreheatS3Profile, profile_id)
            huggingface_artifact = next(
                row for row in rows if row.source == "huggingface"
            )
            ollama_artifact = next(
                row for row in rows if row.source == "ollama_library"
            )
            stack_b_ollama = {
                "artifact_id": ollama_artifact.artifact_id,
                "source": ollama_artifact.source,
                "model_id": ollama_artifact.model_id,
                "resolved_revision": ollama_artifact.resolved_revision,
                "include_patterns": list(ollama_artifact.include_patterns),
                "exclude_patterns": list(ollama_artifact.exclude_patterns),
                "manifest_path": ollama_artifact.manifest_path,
                "file_count": ollama_artifact.file_count,
                "total_size": ollama_artifact.total_size,
                "bucket": profile.bucket,
                "prefix": profile.prefix,
            }
            policy = await create_distribution_policy(
                session,
                User(id=1, username="admin", is_admin=True, hashed_password=""),
                ModelPreheatDistributionPolicyCreate(
                    name="shared-huggingface",
                    profile_id=profile_id,
                    artifact_id=huggingface_artifact.artifact_id,
                    target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                    worker_selector={"worker_uuids": ["stack-b-worker"]},
                    gpu_selector={},
                ),
            )
        await stack_a.dispose()
        await stack_b.dispose()
        return stack_a_counts, counts, rows, profile, policy, stack_b_ollama

    stack_a_counts, counts, rows, profile, policy, stack_b_ollama = asyncio.run(run())

    assert sync_result["artifact_id"] in {row.artifact_id for row in rows}
    assert stack_a_counts["valid"] == 3
    assert counts["valid"] == 3
    assert {row.source for row in rows} == {
        "modelscope",
        "huggingface",
        "ollama_library",
    }
    assert all(row.created_by_task_id is None for row in rows)
    assert profile.inventory_last_success_at is not None
    assert profile.inventory_last_scan_count == 3
    assert policy.source_artifact == huggingface_result["artifact_id"]
    assert policy.request_identity["source"] == "huggingface"

    assert stack_b_ollama["source"] == "ollama_library"
    assert re.fullmatch(
        r"local-snapshot-[0-9a-f]{64}", stack_b_ollama["resolved_revision"]
    )
    assert stack_b_ollama["include_patterns"] == []
    assert stack_b_ollama["exclude_patterns"] == []
    assert stack_b_ollama["file_count"] == 1
    assert stack_b_ollama["total_size"] == len(b"ollama")
    stack_b_ollama_manifest = shared_s3.read_artifact_manifest_path(
        stack_b_ollama["bucket"], stack_b_ollama["manifest_path"]
    )
    assert stack_b_ollama_manifest.identity.source == stack_b_ollama["source"]
    assert (
        decode_path(stack_b_ollama_manifest.identity.revision_path)
        == stack_b_ollama["resolved_revision"]
    )
    assert [decode_path(file.path) for file in stack_b_ollama_manifest.files] == [
        ollama_model_filename(decode_path(stack_b_ollama["model_id"]))
    ]
    assert stack_b_ollama_manifest.files[0].size == len(b"ollama")
    assert (
        stack_b_ollama_manifest.files[0].sha256 == hashlib.sha256(b"ollama").hexdigest()
    )
    stack_b_ollama_identity = ModelPreheatIdentity(
        source=stack_b_ollama["source"],
        model_id=decode_path(stack_b_ollama["model_id"]),
        revision=stack_b_ollama["resolved_revision"],
        file_patterns=tuple(
            decode_path(pattern) for pattern in stack_b_ollama["include_patterns"]
        ),
        exclude_patterns=tuple(
            decode_path(pattern) for pattern in stack_b_ollama["exclude_patterns"]
        ),
    )
    target_request = TargetExecutionRequest(
        cache_dir=tmp_path / "stack-b-cache",
        target_dir=tmp_path / "stack-b-ollama",
        task_id=4,
        attempt=1,
        request_digest=stack_b_ollama_identity.request_digest,
        identity=stack_b_ollama_identity,
        exclude_patterns=stack_b_ollama["exclude_patterns"],
        bucket=stack_b_ollama["bucket"],
        prefix=stack_b_ollama["prefix"],
        artifact_id=stack_b_ollama["artifact_id"],
        manifest_path=stack_b_ollama["manifest_path"],
    )
    file_downloads_before = len(
        [path for path in shared_minio.downloads if not path.endswith("/manifest.json")]
    )
    first_distribution = execute_target_preheat(target_request, shared_s3)
    file_downloads_after_first = len(
        [path for path in shared_minio.downloads if not path.endswith("/manifest.json")]
    )
    second_distribution = execute_target_preheat(target_request, shared_s3)
    assert first_distribution["state"] == "ready"
    assert first_distribution["downloaded"] == 1
    assert first_distribution["skipped"] == 0
    assert second_distribution["state"] == "ready"
    assert second_distribution["downloaded"] == 0
    assert second_distribution["skipped"] == 1
    assert file_downloads_after_first == file_downloads_before + 1
    assert (
        len(
            [
                path
                for path in shared_minio.downloads
                if not path.endswith("/manifest.json")
            ]
        )
        == file_downloads_after_first
    )
    assert (
        target_request.target_dir
        / ollama_model_filename(decode_path(stack_b_ollama_identity.model_path))
    ).read_bytes() == b"ollama"

    manifest_payloads = [
        stored.data
        for (bucket, path), stored in shared_minio.objects.items()
        if bucket == "models" and path.endswith("/manifest.json")
    ]
    assert len(manifest_payloads) == 3
    assert all(b"stack-a-access" not in payload for payload in manifest_payloads)
    assert all(b"stack-a-secret" not in payload for payload in manifest_payloads)


def test_refresh_keeps_existing_artifact_when_manifest_is_invalid(
    monkeypatch, tmp_path
):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'invalid.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                ],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = _profile()
            session.add(profile)
            await session.flush()
            artifact = _artifact(profile.id, 1, "d" * 64)
            session.add(artifact)
            await session.commit()
            profile_id = profile.id
            manifest_path = artifact.manifest_path
        monkeypatch.setattr(
            inventory_module,
            "_scan_profile",
            lambda snapshot: ([], 1, 1, {manifest_path}),
        )
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")
        async with AsyncSession(engine) as session:
            counts = await service.refresh_profile(session, profile_id, 1)
        async with AsyncSession(engine) as session:
            row = (await session.exec(select(ModelPreheatArtifact))).one()
        await engine.dispose()
        return counts, row

    counts, row = asyncio.run(run())

    assert counts["deleted"] == 0
    assert row.manifest_state == ModelPreheatInventoryManifestStateEnum.INVALID


def test_refresh_keeps_all_current_artifact_frozen_by_active_worker_task(
    monkeypatch, tmp_path
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'all-current-reference.db'}"
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                    ModelPreheatDistributionPolicy.__table__,
                    ModelPreheatDistributionPolicyArtifact.__table__,
                    ModelPreheatWorkerTask.__table__,
                ],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = _profile()
            session.add(profile)
            await session.flush()
            artifacts = [
                _artifact(profile.id, profile.config_version, str(index) * 64)
                for index in range(4, 8)
            ]
            session.add_all(artifacts)
            policy = ModelPreheatDistributionPolicy(
                name="all-current",
                selection_mode=ModelPreheatDistributionSelectionModeEnum.ALL_CURRENT,
                profile_id=profile.id,
                profile_config_version=profile.config_version,
                request_identity={"selection_mode": "all_current"},
                request_digest="8" * 64,
                target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                worker_selector={"worker_uuids": ["worker-a"]},
                gpu_selector={},
                selector_digest="9" * 64,
            )
            session.add(policy)
            await session.flush()
            task_cases = [
                (artifacts[3], ModelPreheatWorkerTaskStateEnum.PENDING, None, 0),
                (
                    artifacts[2],
                    ModelPreheatWorkerTaskStateEnum.ERROR,
                    "network_timeout",
                    2,
                ),
                (
                    artifacts[1],
                    ModelPreheatWorkerTaskStateEnum.ERROR,
                    "local_cache_conflict",
                    2,
                ),
                (
                    artifacts[0],
                    ModelPreheatWorkerTaskStateEnum.ERROR,
                    "network_timeout",
                    5,
                ),
            ]
            session.add_all(
                [
                    ModelPreheatWorkerTask(
                        distribution_policy_id=policy.id,
                        distribution_artifact_id=artifact.artifact_id,
                        distribution_request_digest="6" * 64,
                        operation_key=f"operation-{index}",
                        worker_uuid=f"worker-{index}",
                        role=ModelPreheatWorkerTaskRoleEnum.DISTRIBUTE,
                        state=state,
                        error_code=error_code,
                        attempt=attempt,
                        finished_at=(
                            datetime.now(timezone.utc)
                            if state == ModelPreheatWorkerTaskStateEnum.ERROR
                            else None
                        ),
                    )
                    for index, (artifact, state, error_code, attempt) in enumerate(
                        task_cases
                    )
                ]
            )
            profile_id = profile.id
            await session.commit()

        monkeypatch.setattr(
            inventory_module,
            "_scan_profile",
            lambda snapshot: ([], 0, 0, set()),
        )
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")
        async with AsyncSession(engine) as session:
            counts = await service.refresh_profile(session, profile_id, 1)
        async with AsyncSession(engine) as session:
            rows = (
                await session.exec(
                    select(ModelPreheatArtifact).order_by(
                        ModelPreheatArtifact.artifact_id
                    )
                )
            ).all()
        await engine.dispose()
        return counts, rows

    counts, rows = asyncio.run(run())

    assert counts["deleted"] == 2
    assert [row.artifact_id for row in rows] == ["6" * 64, "7" * 64]
    assert all(
        row.manifest_state == ModelPreheatInventoryManifestStateEnum.STALE
        for row in rows
    )


def test_refresh_records_credential_failure_without_leaking_ciphertext(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'credential.db'}"
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                ],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = _profile()
            session.add(profile)
            await session.flush()
            profile_id = profile.id
            await session.commit()
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(
            decrypt=lambda value: (_ for _ in ()).throw(
                ModelPreheatCredentialError("credential_decryption_failed")
            )
        )
        async with AsyncSession(engine) as session:
            with pytest.raises(InventoryRefreshError) as exc_info:
                await service.refresh_profile(session, profile_id, 1)
        async with AsyncSession(engine) as session:
            profile = await session.get(ModelPreheatS3Profile, profile_id)
        await engine.dispose()
        return exc_info.value.code, profile

    error_code, profile = asyncio.run(run())

    assert error_code == "inventory_credential_unavailable"
    assert profile.inventory_last_attempt_at is not None
    assert profile.inventory_last_error_code == "inventory_credential_unavailable"
    assert profile.inventory_refresh_owner is None


def test_refresh_failure_preserves_previous_valid_inventory(monkeypatch, tmp_path):
    previous_success = datetime(2026, 8, 24, tzinfo=timezone.utc)

    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'failure.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                ],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = _profile()
            profile.inventory_last_success_at = previous_success
            session.add(profile)
            await session.flush()
            artifact = _artifact(profile.id, profile.config_version, "e" * 64)
            session.add(artifact)
            await session.commit()
            profile_id = profile.id
            artifact_id = artifact.id

        def fail_scan(_snapshot):
            raise OSError("network unavailable")

        monkeypatch.setattr(inventory_module, "_scan_profile", fail_scan)
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")
        async with AsyncSession(engine) as session:
            with pytest.raises(InventoryRefreshError) as exc_info:
                await service.refresh_profile(session, profile_id, 1)
        async with AsyncSession(engine) as session:
            saved_artifact = await session.get(ModelPreheatArtifact, artifact_id)
            saved_profile = await session.get(ModelPreheatS3Profile, profile_id)
        await engine.dispose()
        return exc_info.value.code, saved_artifact, saved_profile

    error_code, artifact, profile = asyncio.run(run())

    assert error_code == "inventory_scan_failed"
    assert artifact is not None
    assert artifact.manifest_state == ModelPreheatInventoryManifestStateEnum.VALID
    assert profile.inventory_last_success_at == previous_success
    assert profile.inventory_last_attempt_at is not None
    assert profile.inventory_last_error_code == "inventory_scan_failed"
    assert profile.inventory_refresh_owner is None


def test_refresh_preserves_artifacts_referenced_by_tasks_and_policy(
    monkeypatch, tmp_path
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'references.db'}"
        )
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                ],
            )
            for statement in (
                "CREATE TABLE model_preheat_tasks (id INTEGER PRIMARY KEY, s3_profile_id INTEGER, s3_profile_config_version INTEGER, artifact_id VARCHAR(64))",
                "CREATE TABLE model_storage_sync_tasks (id INTEGER PRIMARY KEY, profile_id INTEGER, profile_config_version INTEGER, artifact_id VARCHAR(64))",
                "CREATE TABLE model_preheat_distribution_policies (id INTEGER PRIMARY KEY, profile_id INTEGER, profile_config_version INTEGER, source_artifact_id INTEGER)",
            ):
                await connection.execute(text(statement))
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = _profile()
            session.add(profile)
            await session.flush()
            artifacts = [_artifact(profile.id, 1, char * 64) for char in "abc"]
            session.add_all(artifacts)
            await session.flush()
            await session.exec(
                text(
                    "INSERT INTO model_preheat_tasks VALUES (1, :profile_id, 1, :artifact_id)"
                ).bindparams(
                    profile_id=profile.id, artifact_id=artifacts[0].artifact_id
                )
            )
            await session.exec(
                text(
                    "INSERT INTO model_storage_sync_tasks VALUES (1, :profile_id, 1, :artifact_id)"
                ).bindparams(
                    profile_id=profile.id, artifact_id=artifacts[1].artifact_id
                )
            )
            await session.exec(
                text(
                    "INSERT INTO model_preheat_distribution_policies VALUES (1, :profile_id, 1, :artifact_id)"
                ).bindparams(profile_id=profile.id, artifact_id=artifacts[2].id)
            )
            await session.commit()
            profile_id = profile.id
        monkeypatch.setattr(
            inventory_module, "_scan_profile", lambda snapshot: ([], 0, 0, set())
        )
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")
        async with AsyncSession(engine) as session:
            await service.refresh_profile(session, profile_id, 1)
        async with AsyncSession(engine) as session:
            rows = (await session.exec(select(ModelPreheatArtifact))).all()
        await engine.dispose()
        return rows

    rows = asyncio.run(run())

    assert len(rows) == 3
    assert {row.manifest_state for row in rows} == {
        ModelPreheatInventoryManifestStateEnum.STALE
    }


def test_refresh_rolls_back_when_profile_config_changes_during_scan(
    monkeypatch, tmp_path
):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'lease.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                ],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = _profile()
            session.add(profile)
            await session.flush()
            session.add(_artifact(profile.id, 1, "a" * 64))
            await session.commit()
            profile_id = profile.id

        started = threading.Event()
        release = threading.Event()

        def scan(snapshot):
            started.set()
            assert release.wait(timeout=2)
            values = _artifact_values("b" * 64)
            values.pop("last_verified_at")
            return ([values], 1, 0, set())

        monkeypatch.setattr(inventory_module, "_scan_profile", scan)
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")
        async with AsyncSession(engine) as old_session:
            refresh = asyncio.create_task(
                service.refresh_profile(old_session, profile_id, 1)
            )
            await asyncio.to_thread(started.wait)
            async with AsyncSession(engine, expire_on_commit=False) as new_session:
                profile = await new_session.get(ModelPreheatS3Profile, profile_id)
                profile.config_version = 2
                new_session.add(_artifact(profile_id, 2, "c" * 64))
                await new_session.commit()
            release.set()
            with pytest.raises(InventoryRefreshError) as exc_info:
                await refresh
        async with AsyncSession(engine) as session:
            rows = (await session.exec(select(ModelPreheatArtifact))).all()
        await engine.dispose()
        return exc_info.value.code, rows

    error_code, rows = asyncio.run(run())

    assert error_code == "stale_profile_config"
    assert [(row.profile_config_version, row.manifest_state) for row in rows] == [
        (1, ModelPreheatInventoryManifestStateEnum.VALID),
        (2, ModelPreheatInventoryManifestStateEnum.VALID),
    ]


def test_expired_claim_cannot_commit_over_new_claim_from_same_service(
    monkeypatch, tmp_path
):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'claim.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                ],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = _profile()
            session.add(profile)
            await session.flush()
            session.add(_artifact(profile.id, 1, "a" * 64))
            await session.commit()
            profile_id = profile.id

        first_started = threading.Event()
        second_started = threading.Event()
        release_first = threading.Event()
        release_second = threading.Event()
        scan_count = 0

        def scan(snapshot):
            nonlocal scan_count
            scan_count += 1
            values = _artifact_values(("b" if scan_count == 1 else "c") * 64)
            values.pop("last_verified_at")
            if scan_count == 1:
                first_started.set()
                assert release_first.wait(timeout=2)
            else:
                second_started.set()
                assert release_second.wait(timeout=2)
            return ([values], 1, 0, set())

        monkeypatch.setattr(inventory_module, "_scan_profile", scan)
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")

        async with AsyncSession(engine) as first_session:
            first_refresh = asyncio.create_task(
                service.refresh_profile(first_session, profile_id, 1)
            )
            await asyncio.to_thread(first_started.wait)
            async with AsyncSession(engine) as session:
                await session.exec(
                    update(ModelPreheatS3Profile)
                    .where(ModelPreheatS3Profile.id == profile_id)
                    .values(
                        inventory_refresh_lease_expires_at=datetime.now(timezone.utc)
                        - timedelta(seconds=1)
                    )
                )
                await session.commit()

            async def refresh_second():
                async with AsyncSession(engine) as second_session:
                    return await service.refresh_profile(second_session, profile_id, 1)

            second_refresh = asyncio.create_task(refresh_second())
            await asyncio.to_thread(second_started.wait)
            release_first.set()
            first_error = None
            try:
                await first_refresh
            except InventoryRefreshError as exc:
                first_error = exc.code
            release_second.set()
            second_counts = await second_refresh

        async with AsyncSession(engine) as session:
            rows = (await session.exec(select(ModelPreheatArtifact))).all()
            profile = await session.get(ModelPreheatS3Profile, profile_id)
        await engine.dispose()
        return first_error, second_counts, rows, profile

    first_error, second_counts, rows, profile = asyncio.run(run())

    assert first_error == "stale_profile_config"
    assert second_counts["valid"] == 1
    assert [row.artifact_id for row in rows] == ["c" * 64]
    assert profile.inventory_refresh_owner is None


def test_periodic_refresh_continues_after_one_profile_fails(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'loop.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[ModelPreheatS3Profile.__table__],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            first = _profile()
            first.name = "first"
            first.inventory_refresh_interval_seconds = 60
            second = _profile()
            second.name = "second"
            second.bucket = "other-models"
            second.inventory_refresh_interval_seconds = 60
            session.add_all([first, second])
            await session.commit()

        service = ModelPreheatS3Inventory(engine)
        calls = []

        async def refresh(session, profile_id, config_version):
            calls.append(profile_id)
            if len(calls) == 1:
                raise InventoryRefreshError("inventory_scan_failed")

        service.create_refresh_job = refresh
        next_refresh_at = {}
        now = datetime(2026, 8, 24, tzinfo=timezone.utc)
        async with AsyncSession(engine) as session:
            await service._refresh_due_profiles(session, next_refresh_at, now)
            await service._refresh_due_profiles(
                session, next_refresh_at, now + timedelta(seconds=61)
            )
        await engine.dispose()
        return calls

    calls = asyncio.run(run())

    assert len(calls) == 2


def test_gc_is_explicitly_unsupported():
    service = ModelPreheatS3Inventory(None)

    try:
        asyncio.run(service.create_gc_job(None, 1, 1))
    except ValueError as exc:
        assert str(exc) == "artifact_gc_not_supported"
    else:
        raise AssertionError("首版不得执行 Artifact GC")


def _artifact(profile_id, version, artifact_id):
    return ModelPreheatArtifact(
        profile_id=profile_id,
        profile_config_version=version,
        **_artifact_values(artifact_id),
    )


def _profile():
    return ModelPreheatS3Profile(
        name="profile",
        endpoint="https://s3.example.com",
        bucket="models",
        prefix="storage",
        access_key_encrypted={"ciphertext": "access"},
        secret_key_encrypted={"ciphertext": "secret"},
        encryption_key_version="v1",
        config_version=1,
    )


def _artifact_values(artifact_id):
    return {
        "artifact_id": artifact_id,
        "source": "huggingface",
        "model_id": "org/model",
        "resolved_revision": "resolved-commit",
        "include_patterns": ["config.json"],
        "exclude_patterns": [],
        "manifest_path": f"storage/huggingface/org/model/{artifact_id}/manifest.json",
        "manifest_digest": "d" * 64,
        "file_count": 1,
        "total_size": 5,
        "manifest_state": ModelPreheatInventoryManifestStateEnum.VALID,
        "last_verified_at": datetime.now(timezone.utc),
    }
